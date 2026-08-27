"""
Pipeline Azure ML : entraînement + enregistrement modèle + déploiement endpoint
Usage: python pipelines/training_pipeline.py [--deploy]
"""
import argparse
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import (
    Environment, ManagedOnlineEndpoint, ManagedOnlineDeployment, Model
)
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import yaml, os

parser = argparse.ArgumentParser()
parser.add_argument("--deploy", action="store_true", help="Déployer l'endpoint après entraînement")
parser.add_argument("--experiment", type=str, default="ctr-multimodal")
args = parser.parse_args()

# ==================== CONNEXION WORKSPACE ====================
with open("config.yml") as f:
    cfg = yaml.safe_load(f)

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=cfg["subscription_id"],
    resource_group_name=cfg["resource_group"],
    workspace_name=cfg["workspace_name"],
)
print(f"Connecté au workspace: {cfg['workspace_name']}")

# ==================== ENVIRONNEMENT ====================
env = Environment(
    name="ctr-env",
    conda_file="environments/conda.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
)

# ==================== JOB D'ENTRAÎNEMENT ====================
job = command(
    code="./src",
    command=(
        "python train.py "
        "--data_path ${{inputs.data_path}} "
        "--epochs ${{inputs.epochs}} "
        "--lr ${{inputs.lr}} "
        "--batch_size ${{inputs.batch_size}} "
        "--dropout ${{inputs.dropout}} "
        "--emb_dim ${{inputs.emb_dim}} "
        "--output_dir ./outputs"
    ),
    inputs={
        "data_path": Input(type=AssetTypes.URI_FOLDER, path=cfg["data_asset"]),
        "epochs": 15,
        "lr": 0.002,
        "batch_size": 4096,
        "dropout": 0.3,
        "emb_dim": 64,
    },
    environment=env,
    compute=cfg["compute_cluster"],
    experiment_name=args.experiment,
    display_name="ctr-training-run",
)

print("Soumission du job d'entraînement...")
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job soumis: {returned_job.name}")
print(f"Suivi: {returned_job.studio_url}")

# Attendre la fin du job
ml_client.jobs.stream(returned_job.name)

# ==================== ENREGISTREMENT DU MODÈLE ====================
print("Enregistrement du modèle dans le registre Azure ML...")
model = ml_client.models.create_or_update(
    Model(
        name="ctr-multimodal",
        path=f"azureml://jobs/{returned_job.name}/outputs/artifacts/outputs",
        type=AssetTypes.CUSTOM_MODEL,
        description="CTR model CLIP+Attention+DNN — MicroLens-1M",
        tags={"auc": "0.7752", "framework": "pytorch", "dataset": "MicroLens-1M"},
    )
)
print(f"Modèle enregistré: {model.name} v{model.version}")

# ==================== DÉPLOIEMENT ENDPOINT (optionnel) ====================
if args.deploy:
    endpoint_name = "ctr-endpoint"

    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Endpoint CTR multimodal MicroLens",
        auth_mode="key",
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint créé: {endpoint_name}")

    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint_name,
        model=model.id,
        environment=env,
        code_configuration={"code": "./src", "scoring_script": "score.py"},
        instance_type="Standard_DS3_v2",
        instance_count=1,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # 100% du trafic vers ce déploiement
    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Déploiement terminé. Endpoint: {endpoint_name}")
