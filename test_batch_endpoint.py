"""
Test du batch endpoint: soumet batch_test_sample.parquet et affiche les prédictions
Usage: python test_batch_endpoint.py
"""
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import yaml, time

with open("config.yml") as f:
    cfg = yaml.safe_load(f)

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=cfg["subscription_id"],
    resource_group_name=cfg["resource_group"],
    workspace_name=cfg["workspace_name"],
)

endpoint_name = "ctr-batch-endpoint"

print("Soumission du job batch...")
job = ml_client.batch_endpoints.invoke(
    endpoint_name=endpoint_name,
    input=Input(type=AssetTypes.URI_FILE, path="./data/batch_test_sample.parquet"),
)
print(f"Job soumis: {job.name}")

# Attendre la fin
ml_client.jobs.stream(job.name)

# Télécharger et afficher les résultats
import pandas as pd, os, tempfile

output_dir = tempfile.mkdtemp()
ml_client.jobs.download(job.name, download_path=output_dir, output_name="score")

pred_file = os.path.join(output_dir, "score", "predictions.csv")
if os.path.exists(pred_file):
    preds = pd.read_csv(pred_file)
    print(f"\nPrédictions ({len(preds)} lignes):")
    print(preds.head(10).to_string(index=False))
else:
    print(f"Fichier de prédictions non trouvé dans {output_dir}")
