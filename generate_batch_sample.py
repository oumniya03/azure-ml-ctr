"""
Génère data/batch_test_sample.parquet depuis valid.parquet + item_info
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = "./data"
N = 50  # nombre de lignes

print("Chargement valid.parquet...")
valid_df = pd.read_parquet(os.path.join(DATA_DIR, "valid.parquet"),
                           columns=["user_id", "item_id", "label"]).head(N)

print("Chargement item_info...")
item_info = pd.read_parquet(os.path.join(DATA_DIR, "item_info_new_fusion.parquet"),
                            columns=["item_id", "item_emb_d128"])
item_emb_map = dict(zip(item_info["item_id"],
                        item_info["item_emb_d128"].apply(lambda x: np.array(x, dtype=np.float32))))
del item_info

MAX_SEQ_LEN = 50

rows = []
for _, row in valid_df.iterrows():
    item_id = int(row["item_id"])
    item_emb = item_emb_map.get(item_id, np.zeros(128, dtype=np.float32)).astype(np.float32)
    seq_embs = np.zeros((MAX_SEQ_LEN, 128), dtype=np.float32)
    rows.append({
        "user_id": int(row["user_id"]),
        "item_id": item_id,
        "item_emb": item_emb.tolist(),
        "seq_embs": seq_embs.tolist(),
        "seq_len": 0,
        "label": float(row["label"]),
    })

sample_df = pd.DataFrame(rows)
out_path = os.path.join(DATA_DIR, "batch_test_sample.parquet")
sample_df.to_parquet(out_path, index=False)
print(f"Sauvegardé: {out_path} ({len(sample_df)} lignes)")
