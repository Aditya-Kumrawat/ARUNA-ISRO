import sys
sys.path.insert(0, "python")
import numpy as np
import pandas as pd
import torch
import pickle
import warnings

SEQUENCE_LENGTH = 600

with open("models/feature_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

bad = ~(np.isfinite(scaler.scale_) & (scaler.scale_ > 0))
if bad.any():
    print(f"Patching {bad.sum()} bad scaler columns: {np.where(bad)[0]}")
    scaler.scale_ = np.where(bad, 1.0, scaler.scale_)
    scaler.mean_ = np.where(~np.isfinite(scaler.mean_), 0.0, scaler.mean_)

df = pd.read_parquet("data/dataset/test.parquet")
print(f"Test windows: {len(df)}")

from aruna.l4_forecasting.tcn_model import DualStreamTCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DualStreamTCN(n_thermal=9, n_nonthermal=8, n_shared=10)
model.load_state_dict(torch.load("models/tcn_v1.pt", map_location=device))
model.to(device).eval()

all_preds = []
all_labels = []

for i, (_, row) in enumerate(df.iterrows()):
    if i % 500 == 0:
        print(f"  [{i}/{len(df)}]")

    feats = np.frombuffer(row["features_bytes"], dtype=np.float32).reshape(SEQUENCE_LENGTH, 19).copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = scaler.transform(feats).astype(np.float32)

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # thermal: 8 base + 1 tnci zero → (600, 9)
    x_thermal = np.concatenate(
        [feats[:, :8], np.zeros((SEQUENCE_LENGTH, 1), dtype=np.float32)], axis=1
    )

    # nonthermal: cols 8:16 → (600, 8)
    x_nonthermal = feats[:, 8:16]

    # shared: cols 16:19 + 2 suit zeros + 5 phase one-hot → (600, 10)
    phases_int = np.frombuffer(row["phases_bytes"], dtype=np.uint8)
    phase_oh = np.zeros((SEQUENCE_LENGTH, 5), dtype=np.float32)
    for t in range(SEQUENCE_LENGTH):
        phase_oh[t, int(phases_int[t])] = 1.0
    x_shared = np.concatenate(
        [feats[:, 16:19], np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32), phase_oh], axis=1
    )

    xt = torch.tensor(x_thermal).unsqueeze(0).to(device)
    xn = torch.tensor(x_nonthermal).unsqueeze(0).to(device)
    xs = torch.tensor(x_shared).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.softmax(model(xt, xn, xs), dim=1).cpu().numpy()[0]

    all_preds.append(pred)
    all_labels.append(int(row["label"]))

preds = np.array(all_preds)
labels = np.array(all_labels)

pred_mx = (preds[:, 3] + preds[:, 4]) >= 0.5
true_mx = labels >= 3

tp = int((pred_mx & true_mx).sum())
fn = int((~pred_mx & true_mx).sum())
fp = int((pred_mx & ~true_mx).sum())
tn = int((~pred_mx & ~true_mx).sum())

tss = tp / (tp + fn + 1e-9) - fp / (fp + tn + 1e-9)
pod = tp / (tp + fn + 1e-9)
far = fp / (fp + tn + 1e-9)

mx_probs = preds[:, 3] + preds[:, 4]
print(f"\nM/X probability distribution:")
print(f"  max:  {mx_probs.max():.6f}")
print(f"  mean: {mx_probs.mean():.6f}")
print(f"  p95:  {np.percentile(mx_probs, 95):.6f}")
print(f"  p99:  {np.percentile(mx_probs, 99):.6f}")

print(f"\nTSS vs threshold:")
for thresh in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
    pm = mx_probs >= thresh
    tp_ = int((pm & true_mx).sum())
    fn_ = int((~pm & true_mx).sum())
    fp_ = int((pm & ~true_mx).sum())
    tn_ = int((~pm & ~true_mx).sum())
    t = tp_ / (tp_ + fn_ + 1e-9) - fp_ / (fp_ + tn_ + 1e-9)
    print(f"  thresh={thresh:.2f}  TSS={t:.4f}  TP={tp_}  FN={fn_}  FP={fp_}")

print(f"\n=== Test Set Results (thresh=0.50) ===")
print(f"TSS:              {tss:.4f}  (target > 0.60)")
print(f"M/X recall (POD): {pod:.4f}")
print(f"False alarm rate: {far:.4f}")
print(f"TP={tp}  FN={fn}  FP={fp}  TN={tn}")
print(f"M/X windows in test: {true_mx.sum()}")
