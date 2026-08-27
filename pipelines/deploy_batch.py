"""
Déploiement Batch Endpoint Azure ML
Usage: python pipelines/deploy_batch.py [--model-version 1]
"""
import argparse
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import (
    BatchEndpoint, BatchDeployment, BatchRetrySettings, Environment, CodeConfiguration, Model
)
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--model-version", type=str, default="1")
args = parser.parse_args()

with open("config.yml") as f:
    cfg = yaml.safe_load(f)

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=cfg["subscription_id"],
    resource_group_name=cfg["resource_group"],
    workspace_name=cfg["workspace_name"],
)
print(f"Connecté au workspace: {cfg['workspace_name']}")

compute_name = cfg.get("compute_cluster", "gpu-cluster").strip() or "gpu-cluster"
endpoint_name = "ctr-batch-endpoint"

# ==================== ENDPOINT (idempotent) ====================
try:
    endpoint = ml_client.batch_endpoints.get(endpoint_name)
    print(f"Endpoint existant récupéré: {endpoint_name}")
except ResourceNotFoundError:
    endpoint = BatchEndpoint(
        name=endpoint_name,
        description="Batch endpoint CTR multimodal MicroLens",
    )
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint créé: {endpoint_name}")

# ==================== MODÈLE ====================
model = ml_client.models.get("ctr-multimodal", version=args.model_version)
print(f"Modèle récupéré: {model.name} v{model.version}")

# ==================== ENVIRONNEMENT ====================
env = Environment(
    name="ctr-env",
    conda_file="environments/conda.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
)

# ==================== DEPLOYMENT ====================
deployment = BatchDeployment(
    name="ctr-batch-deployment",
    endpoint_name=endpoint_name,
    model=model.id,
    environment=env,
    code_configuration=CodeConfiguration(code="./src", scoring_script="batch_score.py"),
    compute=compute_name,
    instance_count=1,
    max_concurrency_per_instance=2,
    mini_batch_size=10,
    output_action=BatchDeploymentOutputAction.APPEND_ROW,
    output_file_name="predictions.csv",
    retry_settings=BatchRetrySettings(max_retries=2, timeout=300),
)
ml_client.batch_deployments.begin_create_or_update(deployment).result()
print(f"Déploiement créé: ctr-batch-deployment")

# Définir comme déploiement par défaut
endpoint = ml_client.batch_endpoints.get(endpoint_name)
endpoint.defaults = {"deployment_name": "ctr-batch-deployment"}
ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
print(f"Déploiement par défaut défini.")
print(f"\nBatch endpoint prêt: {endpoint_name}")
