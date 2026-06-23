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
import pyarrow as pa
import pyarrow.parquet as pq

print("\nSaving new splits (streaming — bounded memory) ...", flush=True)

# Per-source split assignment as a numpy array indexed by original row order.
# meta carries _source and _source_idx, so sort by _source_idx to recover order.
split_by_source = {}
for src in full_paths:
    sub = meta[meta["_source"] == src].sort_values("_source_idx")
    split_by_source[src] = sub["split"].to_numpy()

# One output writer per split, opened lazily on first batch. Write to _out_*
# temp files first, then atomically replace the source files at the end so we
# never clobber an input while still reading from it.
out_final = {s: DATA_DIR / f"{s}.parquet"      for s in ("train", "val", "test")}
out_tmp   = {s: DATA_DIR / f"_out_{s}.parquet"  for s in ("train", "val", "test")}
writers   = {s: None for s in ("train", "val", "test")}
counts    = {s: 0    for s in ("train", "val", "test")}

BATCH = 2000  # rows per batch → ~90 MB peak per batch, safe on free Colab

for src_split, src_path in full_paths.items():
    _t0 = _time.monotonic()
    pf  = pq.ParquetFile(src_path)
    arr = split_by_source[src_split]
    total = pf.metadata.num_rows
    print(f"  [{src_split}] streaming {total} rows from {src_path.name} ...", flush=True)

    row_ptr = 0
    for batch in pf.iter_batches(batch_size=BATCH):
        table = pa.Table.from_batches([batch])
        n = table.num_rows
        batch_splits = arr[row_ptr:row_ptr + n]
        row_ptr += n

        for s in ("train", "val", "test"):
            mask = (batch_splits == s)
            if not mask.any():
                continue
            sub_table = table.filter(pa.array(mask))
            if writers[s] is None:
                writers[s] = pq.ParquetWriter(str(out_tmp[s]), table.schema)
            writers[s].write_table(sub_table)
            counts[s] += sub_table.num_rows

    print(f"  [{src_split}] done in {_time.monotonic()-_t0:.0f}s", flush=True)

for w in writers.values():
    if w is not None:
        w.close()

# Atomically replace inputs with the regrouped outputs
for s in ("train", "val", "test"):
    if out_tmp[s].exists():
        if out_final[s].exists():
            out_final[s].unlink()
        out_tmp[s].rename(out_final[s])
    print(f"  Saved {counts[s]} rows -> {out_final[s]}", flush=True)

print("\nDone. Next: python python/scripts/rebalance_dataset.py")
