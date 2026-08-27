import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import mlflow
import mlflow.pytorch
import warnings
warnings.filterwarnings('ignore')

# ==================== ARGS ====================
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data")
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--emb_dim", type=int, default=64)
parser.add_argument("--weight_decay", type=float, default=5e-6)
parser.add_argument("--patience", type=int, default=4)
parser.add_argument("--output_dir", type=str, default="./outputs")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.output_dir, exist_ok=True)

# ==================== DATASET ====================
class CTRDataset(Dataset):
    def __init__(self, df, item_emb_map, item_seq_map, max_seq_len=50):
        self.df = df.reset_index(drop=True)
        self.item_emb_map = item_emb_map
        self.item_seq_map = item_seq_map
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_id = int(row['user_id'])
        item_id = int(row['item_id'])
        label = float(row['label'])

        item_emb = self.item_emb_map.get(item_id, np.zeros(128, dtype=np.float32))
        item_emb = item_emb.astype(np.float32)

        seq = self.item_seq_map.get(user_id, [])
        seq_embs = [self.item_emb_map.get(i, np.zeros(128, dtype=np.float32)).astype(np.float32) for i in seq[-self.max_seq_len:]]
        seq_len = len(seq_embs)

        # Padding
        while len(seq_embs) < self.max_seq_len:
            seq_embs.append(np.zeros(128, dtype=np.float32))

        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'item_id': torch.tensor(item_id, dtype=torch.long),
            'item_emb': torch.tensor(item_emb, dtype=torch.float32),
            'seq_embs': torch.tensor(np.array(seq_embs), dtype=torch.float32),
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float32),
        }

# ==================== MODEL ====================
class CTRModel(nn.Module):
    def __init__(self, num_items, emb_dim=64, item_feat_dim=128,
                 hidden_dims=[512, 256, 128], dropout=0.3, attention_dim=256):
        super().__init__()
        # Hashing trick: 100K buckets au lieu de 1M -> 4x moins de RAM
        self.user_hash_size = 100_000
        self.user_emb = nn.Embedding(self.user_hash_size, emb_dim)
        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)

        self.item_proj = nn.Linear(item_feat_dim, emb_dim)

        # Multi-head attention sur l'historique
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(emb_dim)

        # DNN
        input_dim = emb_dim * 3  # user + item_id_emb + item_feat_emb + attn_out
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, user_id, item_id, item_emb, seq_embs, seq_len):
        u = self.user_emb(user_id % self.user_hash_size)  # hashing trick
        i_id = self.item_emb(item_id)
        i_feat = self.item_proj(item_emb)

        # Attention sur séquence historique
        seq_proj = self.item_proj(seq_embs)
        attn_out, _ = self.attention(i_feat.unsqueeze(1), seq_proj, seq_proj)
        attn_out = self.attn_norm(attn_out.squeeze(1) + i_feat)

        x = torch.cat([u, i_id, attn_out], dim=-1)
        return self.dnn(x).squeeze(-1)

# ==================== TRAINING ====================
def train():
    print(f"Device: {DEVICE}")

    print("Chargement train/valid...")
    train_df = pd.read_parquet(os.path.join(args.data_path, "train.parquet"),
                               columns=['user_id', 'item_id', 'label'])
    valid_df = pd.read_parquet(os.path.join(args.data_path, "valid.parquet"),
                               columns=['user_id', 'item_id', 'label'])

    # Chargement séquentiel pour éviter les pics mémoire
    print("Chargement item_info...")
    item_info = pd.read_parquet(os.path.join(args.data_path, "item_info_new_fusion.parquet"),
                                columns=['item_id', 'item_emb_d128'])
    item_emb_map = dict(zip(
        item_info['item_id'],
        item_info['item_emb_d128'].apply(lambda x: np.array(x, dtype=np.float16))
    ))
    del item_info
    import gc; gc.collect()
    print(f"item_emb_map: {len(item_emb_map)} items")

    # item_seq = 6M lignes (plusieurs par user) -> charger par batch, garder derniere seq par user
    print("Chargement item_seq par batches...")
    import gc, pyarrow.parquet as pq
    item_seq_map = {}
    pf = pq.ParquetFile(os.path.join(args.data_path, "item_seq.parquet"))
    for batch in pf.iter_batches(batch_size=500_000, columns=['user_id', 'item_seq']):
        batch_df = batch.to_pandas()
        for uid, seq in zip(batch_df['user_id'], batch_df['item_seq']):
            item_seq_map[int(uid)] = [int(x) for x in seq if x != 0]
        del batch_df
        gc.collect()
    print(f"item_seq_map: {len(item_seq_map)} users")

    num_items = max(item_emb_map.keys())

    train_ds = CTRDataset(train_df, item_emb_map, item_seq_map)
    valid_ds = CTRDataset(valid_df, item_emb_map, item_seq_map)
    del train_df, valid_df
    gc.collect()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = CTRModel(num_items=num_items, emb_dim=args.emb_dim, dropout=args.dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    mlflow.log_params({
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "dropout": args.dropout, "emb_dim": args.emb_dim, "weight_decay": args.weight_decay,
        "hidden_dims": "[512,256,128]", "attention_heads": 4,
    })

    best_auc, patience_counter = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(batch['user_id'], batch['item_id'],
                           batch['item_emb'], batch['seq_embs'], batch['seq_len'])
            loss = criterion(logits, batch['label'])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits = model(batch['user_id'], batch['item_id'],
                               batch['item_emb'], batch['seq_embs'], batch['seq_len'])
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())

        auc = roc_auc_score(all_labels, all_preds)
        scheduler.step(1 - auc)

        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | AUC: {auc:.4f}")
        mlflow.log_metrics({"train_loss": avg_loss, "val_auc": auc}, step=epoch)

        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            mlflow.log_metric("best_auc", best_auc)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Sauvegarde finale + enregistrement MLflow
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt")))
    mlflow.pytorch.log_model(model, "ctr_model")

    # Sauvegarde des métadonnées pour le score.py
    metadata = {"num_items": int(num_items), "emb_dim": args.emb_dim, "dropout": args.dropout}
    import json
    with open(os.path.join(args.output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f)
    mlflow.log_artifact(os.path.join(args.output_dir, "model_metadata.json"))

    print(f"Entraînement terminé. Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    with mlflow.start_run():
        train()
