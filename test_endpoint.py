from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import yaml, json, urllib.request

with open("config.yml") as f:
    cfg = yaml.safe_load(f)

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=cfg["subscription_id"],
    resource_group_name=cfg["resource_group"],
    workspace_name=cfg["workspace_name"],
)

endpoint = ml_client.online_endpoints.get("ctr-endpoint")
keys = ml_client.online_endpoints.get_keys("ctr-endpoint")

# Requête de test
payload = {
    "user_id": [1, 2],
    "item_id": [10, 20],
    "item_emb": [[0.0] * 128, [0.0] * 128],
    "seq_embs": [[[0.0] * 128] * 50, [[0.0] * 128] * 50],
    "seq_len": [5, 12]
}

req = urllib.request.Request(
    endpoint.scoring_uri,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {keys.primary_key}"}
)
response = urllib.request.urlopen(req)
print(json.loads(response.read()))
