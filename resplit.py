"""
resplit.py - Re-split existing dataset with a val set that contains M/X flares.

New split:
  Train: 2023-09-02 to 2024-04-30  (Sep–Apr, no May superstorm)
  Val:   2024-05-01 to 2024-06-30  (May 2024 superstorm = X-class flares!)
  Test:  2024-07-01 onwards
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DATASET_DIR = Path("data/dataset")
SCALER_OUT  = Path("models/feature_scaler.pkl")
SEQUENCE_LENGTH = 600
N_FEATURES = 19

TRAIN_END = pd.Timestamp("2024-04-30 23:59:59", tz="UTC")
VAL_START = pd.Timestamp("2024-05-01", tz="UTC")
VAL_END   = pd.Timestamp("2024-06-30 23:59:59", tz="UTC")
TEST_START = pd.Timestamp("2024-07-01", tz="UTC")

print("Loading all splits...")
dfs = []
for split in ("train", "val", "test"):
    p = DATASET_DIR / f"{split}.parquet"
    if p.exists():
        dfs.append(pd.read_parquet(p))
all_df = pd.concat(dfs, ignore_index=True)
print(f"Total windows: {len(all_df)}")

def assign_split(window_end_str):
    ts = pd.Timestamp(window_end_str).tz_convert("UTC") if pd.Timestamp(window_end_str).tzinfo else pd.Timestamp(window_end_str, tz="UTC")
    if ts <= TRAIN_END:
        return "train"
    if VAL_START <= ts <= VAL_END:
        return "val"
    if ts >= TEST_START:
        return "test"
    return "other"

all_df["split"] = all_df["window_end"].apply(assign_split)

counts = all_df.groupby("split").size()
print(f"\nNew split counts: {dict(counts)}")

CLASS_NAMES = {0:"QUIET",1:"B",2:"C",3:"M",4:"X"}
for split_name in ("train", "val", "test"):
    sub = all_df[all_df["split"] == split_name]
    if sub.empty:
        print(f"\n{split_name.upper()}: EMPTY!")
        continue
    print(f"\n{split_name.upper()} ({len(sub)} windows):")
    for lbl, name in CLASS_NAMES.items():
        count = (sub["label"] == lbl).sum()
        print(f"  {name}: {count:>6}  ({count/len(sub)*100:5.1f}%)")

# Save new splits
for split_name in ("train", "val", "test"):
    sub = all_df[all_df["split"] == split_name].drop(columns=["split"])
    out = DATASET_DIR / f"{split_name}.parquet"
    sub.to_parquet(out, index=False)
    print(f"Saved {len(sub)} {split_name} windows → {out}")

# Refit scaler on new train only
print("\nRefitting scaler on new train set...")
train_df = all_df[all_df["split"] == "train"]
scaler = StandardScaler()
for raw in train_df["features_bytes"].to_numpy():
    feats = np.frombuffer(raw, dtype=np.float32).reshape(SEQUENCE_LENGTH, N_FEATURES)
    scaler.partial_fit(feats)

with open(SCALER_OUT, "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler saved → {SCALER_OUT}")
print("\nDone. Now run: python python/scripts/train_tcn.py")
