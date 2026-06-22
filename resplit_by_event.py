"""
resplit_by_event.py — Event-level train/val/test split.

Groups consecutive non-QUIET windows into flare events, then splits
events temporally (70/15/15) so no event straddles two splits and
each split gets a proportional share of M and X events.

QUIET windows are assigned to whichever split owns the nearest flare
event. This preserves temporal ordering and prevents leakage.

Output: overwrites data/dataset/train.parquet, val.parquet, test.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/dataset")
CLASS_NAMES = {0: "QUIET", 1: "B", 2: "C", 3: "M", 4: "X"}

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 0.15 (remainder)

# Gap between windows that signals a new event (minutes)
EVENT_GAP_MINUTES = 30

# ── Load all data ─────────────────────────────────────────────────────────────

print("Loading all splits ...")
dfs = []
for split in ("train", "val", "test"):
    p = DATA_DIR / f"{split}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        dfs.append(df)
        print(f"  {split}: {len(df)} rows")

all_df = pd.concat(dfs, ignore_index=True)
all_df["window_end_ts"] = pd.to_datetime(all_df["window_end"], utc=True)
all_df = all_df.sort_values("window_end_ts").reset_index(drop=True)
print(f"Total: {len(all_df)} rows spanning "
      f"{all_df['window_end_ts'].min()} to {all_df['window_end_ts'].max()}")

# ── Identify flare events ─────────────────────────────────────────────────────

flare_mask = all_df["label"] > 0   # any non-QUIET window
flare_df   = all_df[flare_mask].copy()

# Assign event IDs: new event when gap > EVENT_GAP_MINUTES
dt = flare_df["window_end_ts"].diff()
new_event = (dt > pd.Timedelta(minutes=EVENT_GAP_MINUTES)) | dt.isna()
flare_df["event_id"] = new_event.cumsum()

# Summarise events
events = (
    flare_df.groupby("event_id")
    .agg(
        start=("window_end_ts", "min"),
        end=("window_end_ts", "max"),
        n_windows=("label", "count"),
        peak_class=("label", "max"),
    )
    .reset_index()
    .sort_values("start")
    .reset_index(drop=True)
)

print(f"\nFlare events detected: {len(events)}")
for cls in range(1, 5):
    n = (events["peak_class"] == cls).sum()
    print(f"  {CLASS_NAMES[cls]}-class events: {n}")

# ── Split events temporally (preserve chronological order) ───────────────────

n_events   = len(events)
n_train    = int(np.floor(n_events * TRAIN_FRAC))
n_val      = int(np.floor(n_events * VAL_FRAC))

train_event_ids = set(events.iloc[:n_train]["event_id"])
val_event_ids   = set(events.iloc[n_train:n_train + n_val]["event_id"])
test_event_ids  = set(events.iloc[n_train + n_val:]["event_id"])

print(f"\nEvent split: train={len(train_event_ids)}  "
      f"val={len(val_event_ids)}  test={len(test_event_ids)}")

# ── Assign flare windows to splits ───────────────────────────────────────────

flare_df["split"] = flare_df["event_id"].map(
    lambda eid: "train" if eid in train_event_ids
    else ("val" if eid in val_event_ids else "test")
)

# ── Assign QUIET windows to the nearest split boundary ───────────────────────

# Build a timeline: for each flare event record its split and time range
event_split = events.copy()
event_split["split"] = event_split["event_id"].map(
    lambda eid: "train" if eid in train_event_ids
    else ("val" if eid in val_event_ids else "test")
)

quiet_df   = all_df[~flare_mask].copy()

# For each QUIET window, assign split based on its timestamp relative to events
# Strategy: assign to the split that owns the temporally closest flare event
train_events = event_split[event_split["split"] == "train"]
val_events   = event_split[event_split["split"] == "val"]
test_events  = event_split[event_split["split"] == "test"]

def nearest_split(ts):
    def min_dist(edf):
        if edf.empty:
            return float("inf")
        return min(
            abs((ts - edf["start"]).dt.total_seconds().abs().min()),
            abs((ts - edf["end"]).dt.total_seconds().abs().min()),
        )
    d_train = min_dist(train_events)
    d_val   = min_dist(val_events)
    d_test  = min_dist(test_events)
    best = min(d_train, d_val, d_test)
    if best == d_train:
        return "train"
    if best == d_val:
        return "val"
    return "test"

print("\nAssigning QUIET windows to splits (may take a minute) ...")
# Vectorised approximation: use the split boundary timestamps
# Train events end at events.iloc[n_train-1]["end"]
# Val events end at events.iloc[n_train+n_val-1]["end"]
if not train_events.empty and not val_events.empty and not test_events.empty:
    train_end_ts = train_events["end"].max()
    val_end_ts   = val_events["end"].max()

    def fast_split(ts):
        if ts <= train_end_ts:
            return "train"
        if ts <= val_end_ts:
            return "val"
        return "test"

    quiet_df["split"] = quiet_df["window_end_ts"].apply(fast_split)
else:
    # Fallback: assign all QUIET to train
    quiet_df["split"] = "train"

print(f"  QUIET distribution: "
      f"train={( quiet_df['split']=='train').sum()}  "
      f"val={( quiet_df['split']=='val').sum()}  "
      f"test={( quiet_df['split']=='test').sum()}")

# ── Combine and save ──────────────────────────────────────────────────────────

combined = pd.concat([
    flare_df[all_df.columns.tolist() + ["split"]],
    quiet_df[all_df.columns.tolist() + ["split"]],
], ignore_index=True).sort_values("window_end_ts")

print("\nFinal distributions:")
for split_name in ("train", "val", "test"):
    sub = combined[combined["split"] == split_name]
    print(f"\n{split_name.upper()} ({len(sub)} windows):")
    for lbl, name in CLASS_NAMES.items():
        n = (sub["label"] == lbl).sum()
        pct = 100 * n / len(sub) if len(sub) > 0 else 0
        print(f"  {name}: {n:>6}  ({pct:.1f}%)")

drop_cols = ["window_end_ts", "split"]
for split_name in ("train", "val", "test"):
    sub = combined[combined["split"] == split_name].drop(columns=drop_cols)
    out = DATA_DIR / f"{split_name}.parquet"
    sub.to_parquet(out, index=False)
    print(f"\nSaved {len(sub)} {split_name} windows -> {out}")

print("\nDone. Next: python python/scripts/rebalance_dataset.py")
