"""
fit_mapie.py - Fit MAPIE conformal calibrator on val set using trained TCN.
Run after training completes if models/mapie_calibration.pkl is missing.
"""
import sys
sys.path.insert(0, "python")
import numpy as np
import pandas as pd
import torch
import pickle
import warnings
from pathlib import Path

SEQUENCE_LENGTH = 600
N_FEATURES = 19

print("Loading scaler and model...")
with open("models/feature_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

bad = ~(np.isfinite(scaler.scale_) & (scaler.scale_ > 0))
if bad.any():
    scaler.scale_ = np.where(bad, 1.0, scaler.scale_)
    scaler.mean_  = np.where(~np.isfinite(scaler.mean_), 0.0, scaler.mean_)

from aruna.l4_forecasting.tcn_model import DualStreamTCN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualStreamTCN(n_thermal=9, n_nonthermal=8, n_shared=10)
model.load_state_dict(torch.load("models/tcn_v1.pt", map_location=device))
model.to(device).eval()

def get_preds(parquet_path):
    df = pd.read_parquet(parquet_path)
    all_probs, all_labels = [], []
    for _, row in df.iterrows():
        feats = np.frombuffer(row["features_bytes"], dtype=np.float32).reshape(SEQUENCE_LENGTH, N_FEATURES).copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats = scaler.transform(feats).astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        x_thermal    = np.concatenate([feats[:, :8], np.zeros((SEQUENCE_LENGTH, 1), dtype=np.float32)], axis=1)
        x_nonthermal = feats[:, 8:16]
        phases_int   = np.frombuffer(row["phases_bytes"], dtype=np.uint8)
        phase_oh     = np.zeros((SEQUENCE_LENGTH, 5), dtype=np.float32)
        for t in range(SEQUENCE_LENGTH):
            phase_oh[t, int(phases_int[t])] = 1.0
        x_shared = np.concatenate([feats[:, 16:19], np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32), phase_oh], axis=1)

        xt = torch.tensor(x_thermal).unsqueeze(0).to(device)
        xn = torch.tensor(x_nonthermal).unsqueeze(0).to(device)
        xs = torch.tensor(x_shared).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.softmax(model(xt, xn, xs), dim=1).cpu().numpy()[0]
        all_probs.append(prob)
        all_labels.append(int(row["label"]))
    return np.array(all_probs), np.array(all_labels)

print("Getting val set predictions...")
val_probs, val_labels = get_preds("data/dataset/val.parquet")
print(f"Val windows: {len(val_labels)}")

# Conformal calibration: nonconformity score = 1 - P(true class)
# Store quantile thresholds at multiple alpha levels for prediction sets
cal_data = {
    "val_probs":  val_probs,
    "val_labels": val_labels,
    "quantiles":  {},
}

for alpha in [0.05, 0.10, 0.15, 0.20]:
    scores = 1.0 - val_probs[np.arange(len(val_labels)), val_labels]
    q = float(np.quantile(scores, 1.0 - alpha))
    cal_data["quantiles"][alpha] = q
    print(f"  alpha={alpha:.2f}  quantile={q:.4f}  (coverage target={1-alpha:.0%})")

# Also store per-class ECE calibration info
n_bins = 10
ece_data = {}
for cls in range(5):
    probs_cls = val_probs[:, cls]
    true_cls  = (val_labels == cls).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (probs_cls >= bins[i]) & (probs_cls < bins[i+1])
        if mask.sum() > 0:
            bin_accs.append(true_cls[mask].mean())
            bin_confs.append(probs_cls[mask].mean())
            bin_counts.append(mask.sum())
    ece_data[cls] = {"accs": bin_accs, "confs": bin_confs, "counts": bin_counts}

cal_data["ece"] = ece_data

Path("models").mkdir(exist_ok=True)
with open("models/mapie_calibration.pkl", "wb") as f:
    pickle.dump(cal_data, f)

print(f"\nSaved → models/mapie_calibration.pkl")
print("Done.")
