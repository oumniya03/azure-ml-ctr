import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class CTRModel(nn.Module):
    def __init__(self, num_items, emb_dim=64, item_feat_dim=128,
                 hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        self.user_hash_size = 100_000
        self.user_emb = nn.Embedding(self.user_hash_size, emb_dim)
        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
        self.item_proj = nn.Linear(item_feat_dim, emb_dim)
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(emb_dim)
        input_dim = emb_dim * 3
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, user_id, item_id, item_emb, seq_embs, seq_len):
        u = self.user_emb(user_id % self.user_hash_size)
        i_id = self.item_emb(item_id)
        i_feat = self.item_proj(item_emb)
        seq_proj = self.item_proj(seq_embs)
        attn_out, _ = self.attention(i_feat.unsqueeze(1), seq_proj, seq_proj)
        attn_out = self.attn_norm(attn_out.squeeze(1) + i_feat)
        x = torch.cat([u, i_id, attn_out], dim=-1)
        return self.dnn(x).squeeze(-1)


def init():
    global model, device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")

    meta_path = None
    for root, _, files in os.walk(model_dir):
        if "model_metadata.json" in files:
            meta_path = os.path.join(root, "model_metadata.json")
            break
    if meta_path is None:
        raise FileNotFoundError(f"model_metadata.json introuvable dans {model_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    pt_path = None
    for root, _, files in os.walk(model_dir):
        if "best_model.pt" in files:
            pt_path = os.path.join(root, "best_model.pt")
            break
    if pt_path is None:
        raise FileNotFoundError(f"best_model.pt introuvable dans {model_dir}")

    state_dict = torch.load(pt_path, map_location=device)
    # Déduire num_items depuis le checkpoint pour éviter mismatch
    num_items = state_dict["item_emb.weight"].shape[0] - 1
    model = CTRModel(num_items=num_items, emb_dim=meta["emb_dim"], dropout=meta["dropout"])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Modèle CTR chargé.")


def run(mini_batch):
    """
    mini_batch: liste de chemins vers des fichiers parquet
    Chaque fichier doit contenir les colonnes:
        user_id, item_id, item_emb (list[128]), seq_embs (list[50x128]), seq_len
    Retourne un DataFrame avec colonnes [user_id, item_id, prediction]
    """
    results = []

    for file_path in mini_batch:
        df = pd.read_parquet(file_path)

        user_id = torch.tensor(df["user_id"].values, dtype=torch.long).to(device)
        item_id = torch.tensor(df["item_id"].values, dtype=torch.long).to(device)
        item_emb = torch.tensor(np.array(df["item_emb"].tolist(), dtype=np.float32)).to(device)
        seq_embs_list = [np.array(s, dtype=np.float32).reshape(50, 128) for s in df["seq_embs"]]
        seq_embs = torch.tensor(np.stack(seq_embs_list)).to(device)
        seq_len = torch.tensor(df["seq_len"].values, dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(user_id, item_id, item_emb, seq_embs, seq_len)
            probs = torch.sigmoid(logits).cpu().numpy()

        batch_result = pd.DataFrame({
            "user_id": df["user_id"].values,
            "item_id": df["item_id"].values,
            "prediction": probs,
        })
        results.append(batch_result)

    return pd.concat(results, ignore_index=True)
