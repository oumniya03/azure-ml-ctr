"""
Monitoring de dérive des données (Data Drift)
Compare la distribution de nouvelles données vs données de train
Métriques: PSI (Population Stability Index) + Kolmogorov-Smirnov
Usage: python monitor.py --new-data ./data/batch_test_sample.parquet
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

PSI_THRESHOLD = 0.2   # > 0.2 = dérive significative
KS_PVALUE_THRESHOLD = 0.05  # < 0.05 = dérive significative

parser = argparse.ArgumentParser()
parser.add_argument("--train-data", type=str, default="./data/valid.parquet",
                    help="Données de référence (train/valid)")
parser.add_argument("--new-data", type=str, required=True,
                    help="Nouvelles données à comparer")
args = parser.parse_args()


def compute_psi(expected, actual, bins=10):
    """Population Stability Index entre deux distributions."""
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    breakpoints = np.linspace(min_val, max_val, bins + 1)

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Eviter division par zéro
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def check_drift(name, ref, new):
    psi = compute_psi(ref, new)
    ks_stat, ks_pvalue = stats.ks_2samp(ref, new)

    psi_alert = psi > PSI_THRESHOLD
    ks_alert = ks_pvalue < KS_PVALUE_THRESHOLD
    drift = psi_alert or ks_alert

    status = "⚠️  DÉRIVE" if drift else "✅ OK"
    print(f"  {name:20s} | PSI={psi:.4f} {'⚠️' if psi_alert else '  '} | "
          f"KS p={ks_pvalue:.4f} {'⚠️' if ks_alert else '  '} | {status}")
    return drift


print(f"Référence : {args.train_data}")
print(f"Nouvelles données : {args.new_data}\n")

ref_df = pd.read_parquet(args.train_data, columns=["user_id", "item_id"])
new_df = pd.read_parquet(args.new_data)

print("=== Rapport de dérive ===")
drifts = []

# user_id distribution (hashed)
drifts.append(check_drift("user_id (hashed)",
    ref_df["user_id"].values % 100_000,
    new_df["user_id"].values % 100_000))

# item_id distribution
drifts.append(check_drift("item_id",
    ref_df["item_id"].values.astype(float),
    new_df["item_id"].values.astype(float)))

# seq_len si disponible
if "seq_len" in new_df.columns:
    ref_seq_len = np.zeros(len(ref_df))  # référence sans seq_len → 0
    drifts.append(check_drift("seq_len",
        ref_seq_len,
        new_df["seq_len"].values.astype(float)))

# item_emb agrégé (norme L2 moyenne)
if "item_emb" in new_df.columns:
    new_norms = np.stack(new_df["item_emb"].values)
    new_norms = np.linalg.norm(new_norms, axis=1)
    ref_norms = np.random.normal(1.0, 0.3, len(ref_df))  # distribution attendue normalisée L2
    drifts.append(check_drift("item_emb (norme L2)", ref_norms, new_norms))

print()
n_drifts = sum(drifts)
if n_drifts == 0:
    print("✅ Aucune dérive détectée.")
else:
    print(f"⚠️  {n_drifts}/{len(drifts)} feature(s) en dérive — réentraînement recommandé.")
