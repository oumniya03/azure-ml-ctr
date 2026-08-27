import os
import json
import torch
import torch.nn as nn
import numpy as np


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

    with open(os.path.join(model_dir, "model_metadata.json")) as f:
        meta = json.load(f)

    model = CTRModel(
        num_items=meta["num_items"],
        emb_dim=meta["emb_dim"],
        dropout=meta["dropout"]
    )
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device))
    model.to(device)
    model.eval()
    print("Modèle CTR chargé.")


def run(raw_data):
    """
    Entrée attendue (JSON):
    {
        "user_id": [1, 2],
        "item_id": [10, 20],
        "item_emb": [[...128 floats...], [...]],
        "seq_embs": [[[...128 floats...] x 50], [...]],
        "seq_len": [5, 12]
    }
    """
    data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data

    user_id = torch.tensor(data["user_id"], dtype=torch.long).to(device)
    item_id = torch.tensor(data["item_id"], dtype=torch.long).to(device)
    item_emb = torch.tensor(data["item_emb"], dtype=torch.float32).to(device)
    seq_embs = torch.tensor(data["seq_embs"], dtype=torch.float32).to(device)
    seq_len = torch.tensor(data["seq_len"], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(user_id, item_id, item_emb, seq_embs, seq_len)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()

    return {"predictions": probs}
