"""
resplit_by_event.py — Fast event-level train/val/test split (70/15/15).

Loads only label + window_end columns for split logic, then filters
the full parquets by index to avoid loading all byte arrays twice.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR   = Path("data/dataset")
CLASS_NAMES = {0: "QUIET", 1: "B", 2: "C", 3: "M", 4: "X"}
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
EVENT_GAP_MINUTES = 30

# ── 1. Load only lightweight columns ─────────────────────────────────────────

print("Loading label + window_end columns from all splits ...")
light_dfs = []
full_paths = {}

for split in ("train", "val", "test"):
    p = DATA_DIR / f"{split}.parquet"
    if not p.exists():
        print(f"  MISSING: {p}")
        continue
    df = pd.read_parquet(p, columns=["label", "window_end"])
    df["_source"] = split
    df["_source_idx"] = np.arange(len(df))
    light_dfs.append(df)
    full_paths[split] = p
    print(f"  {split}: {len(df)} rows")

meta = pd.concat(light_dfs, ignore_index=True)
meta["window_end_ts"] = pd.to_datetime(meta["window_end"], utc=True)
meta = meta.sort_values("window_end_ts").reset_index(drop=True)
print(f"Total: {len(meta)} rows")

# ── 2. Detect flare events ────────────────────────────────────────────────────

flare_mask = meta["label"] > 0
flare_meta = meta[flare_mask].copy()

dt       = flare_meta["window_end_ts"].diff()
new_event = (dt > pd.Timedelta(minutes=EVENT_GAP_MINUTES)) | dt.isna()
flare_meta["event_id"] = new_event.cumsum().astype(int)

events = (
    flare_meta.groupby("event_id")
    .agg(start=("window_end_ts","min"),
         end=("window_end_ts","max"),
         n_windows=("label","count"),
         peak_class=("label","max"))
    .reset_index()
    .sort_values("start")
    .reset_index(drop=True)
)

print(f"\nFlare events detected: {len(events)}")
for cls in range(1, 5):
    print(f"  {CLASS_NAMES[cls]}: {(events['peak_class']==cls).sum()} events")

# ── 3. Split events temporally ───────────────────────────────────────────────

n = len(events)
n_train = int(np.floor(n * TRAIN_FRAC))
n_val   = int(np.floor(n * VAL_FRAC))

train_ids = set(events.iloc[:n_train]["event_id"])
val_ids   = set(events.iloc[n_train:n_train+n_val]["event_id"])
test_ids  = set(events.iloc[n_train+n_val:]["event_id"])

print(f"\nEvent split: train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

# ── 4. Assign split to every row ─────────────────────────────────────────────

# Flare windows: follow their event
eid_map  = flare_meta.set_index(flare_meta.index)["event_id"]
def flare_split(eid):
    return "train" if eid in train_ids else ("val" if eid in val_ids else "test")

flare_meta["split"] = flare_meta["event_id"].map(flare_split)
meta.loc[flare_mask, "split"] = flare_meta["split"].values

# QUIET windows: assign by temporal boundary
train_end_ts = events.iloc[n_train - 1]["end"]   if n_train > 0          else meta["window_end_ts"].min()
val_end_ts   = events.iloc[n_train + n_val - 1]["end"] if n_val > 0 else train_end_ts

quiet_mask = ~flare_mask
meta.loc[quiet_mask & (meta["window_end_ts"] <= train_end_ts), "split"] = "train"
meta.loc[quiet_mask & (meta["window_end_ts"] >  train_end_ts) &
         (meta["window_end_ts"] <= val_end_ts), "split"] = "val"
meta.loc[quiet_mask & (meta["window_end_ts"] >  val_end_ts), "split"] = "test"

# ── 5. Print distributions ────────────────────────────────────────────────────

print("\nFinal distributions:")
for split_name in ("train", "val", "test"):
    sub = meta[meta["split"] == split_name]
    print(f"\n{split_name.upper()} ({len(sub)} windows):")
    for lbl, name in CLASS_NAMES.items():
        n_cls = (sub["label"] == lbl).sum()
        print(f"  {name}: {n_cls:>6}")

# ── 6. Save: reload full parquets, filter by assigned indices, write ──────────

import time as _time
print("\nSaving new splits (reloading full data) ...", flush=True)
print("  NOTE: each source file is reloaded in full; train.parquet (~1.2 GB)",
      "takes 1-2 min to load with no output. This is normal — do not interrupt.",
      flush=True)

# Build a lookup: (source_split, source_idx) -> new_split
meta["_key"] = meta["_source"] + "_" + meta["_source_idx"].astype(str)
key_to_split = dict(zip(meta["_key"], meta["split"]))

for src_split, src_path in full_paths.items():
    _t0 = _time.monotonic()
    print(f"  [{src_split}] loading {src_path.name} ...", flush=True)
    full_df = pd.read_parquet(src_path)
    print(f"  [{src_split}] loaded {len(full_df)} rows in {_time.monotonic()-_t0:.0f}s; "
          f"assigning splits ...", flush=True)
    full_df["_key"] = src_split + "_" + np.arange(len(full_df)).astype(str)
    full_df["split"] = full_df["_key"].map(key_to_split)

    for new_split in ("train", "val", "test"):
        subset = full_df[full_df["split"] == new_split].drop(columns=["_key", "split"])
        out_path = DATA_DIR / f"_tmp_{new_split}_{src_split}.parquet"
        subset.to_parquet(out_path, index=False)
    print(f"  [{src_split}] written to temp files "
          f"({_time.monotonic()-_t0:.0f}s total)", flush=True)
    del full_df  # free ~1.2 GB before loading the next source

print("  Merging temp files ...", flush=True)
for new_split in ("train", "val", "test"):
    parts = []
    for src_split in full_paths:
        tmp = DATA_DIR / f"_tmp_{new_split}_{src_split}.parquet"
        if tmp.exists():
            parts.append(pd.read_parquet(tmp))
            tmp.unlink()
    merged = pd.concat(parts, ignore_index=True)
    out = DATA_DIR / f"{new_split}.parquet"
    merged.to_parquet(out, index=False)
    print(f"  Saved {len(merged)} rows -> {out}")

print("\nDone. Next: python python/scripts/rebalance_dataset.py")
