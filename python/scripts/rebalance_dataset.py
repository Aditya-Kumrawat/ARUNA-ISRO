"""
ARUNA dataset rebalancing and augmentation.

train.parquet is NEVER modified. Output goes to train_balanced_augmented.parquet.
val.parquet and test.parquet are NEVER touched.

Workflow:
  1. Inspect train.parquet — label mapping, class distribution, data representation
  2. Downsample QUIET to ~3000 windows
  3. Keep ALL real flare samples intact
  4. Augment rare flare classes with physically plausible perturbations
     (augmentation type selected based on detected data representation)
  5. Shuffle and save
  6. Print full before/after summary
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────────

SEED = 42
rng = np.random.default_rng(SEED)

DATA_DIR = Path("data/dataset")
INPUT_PATH  = DATA_DIR / "train.parquet"
OUTPUT_PATH = DATA_DIR / "train_balanced_augmented.parquet"

CLASS_NAMES = {0: "QUIET", 1: "B", 2: "C", 3: "M", 4: "X"}
FLARE_CLASSES = {1, 2, 3, 4}  # all non-QUIET

# ── Step 1: inspect ──────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1 -- Inspecting train.parquet")
print("=" * 60)

df = pd.read_parquet(INPUT_PATH)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"dtypes:\n{df.dtypes}\n")

label_col = "label"
assert label_col in df.columns, "Expected 'label' column not found"

orig_dist = df[label_col].value_counts().sort_index()
print("Class distribution (original):")
for cls_id, cnt in orig_dist.items():
    name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
    print(f"  {cls_id} ({name:<6}): {cnt:>6}")

# Detect data representation from bytes columns
bytes_cols = [c for c in df.columns if "bytes" in c.lower()]
print(f"\nBytes columns detected: {bytes_cols}")

repr_type = "unknown"
n_timesteps = None
n_features_per_step = None

if bytes_cols:
    sample_row = df.iloc[0]
    for bcol in bytes_cols:
        arr = np.frombuffer(sample_row[bcol], dtype=np.float32)
        print(f"  {bcol}: {len(arr)} float32 values")

    # Primary feature column
    feat_col = bytes_cols[0]
    feat_arr  = np.frombuffer(sample_row[feat_col], dtype=np.float32)
    feat_len  = len(feat_arr)

    phases_col = next((c for c in bytes_cols if "phase" in c.lower()), None)
    if phases_col:
        ph_arr = np.frombuffer(sample_row[phases_col], dtype=np.float32)
        ph_len = len(ph_arr)
        # If phases length divides features length evenly, assume temporal
        if feat_len % ph_len == 0:
            n_timesteps = ph_len
            n_features_per_step = feat_len // ph_len
            repr_type = "temporal_sequence"
            print(f"\nRepresentation: TEMPORAL SEQUENCE")
            print(f"  Timesteps     : {n_timesteps}")
            print(f"  Features/step : {n_features_per_step}")
        else:
            repr_type = "feature_vector"
            print(f"\nRepresentation: AGGREGATED FEATURE VECTOR (len={feat_len})")
    else:
        repr_type = "feature_vector"
        print(f"\nRepresentation: FEATURE VECTOR (len={feat_len})")
else:
    feat_col   = None
    phases_col = None
    print("No bytes columns found; skipping augmentation of sequence data.")

# ── Step 2 & 3: downsample QUIET, keep all flares ────────────────────────────

print("\n" + "=" * 60)
print("STEP 2-3 -- Downsample QUIET; keep all flare samples")
print("=" * 60)

quiet_df = df[df[label_col] == 0]
flare_df = df[df[label_col] != 0]

# Target: QUIET roughly 3x the total real flare count (keeps imbalance reduced
# without discarding too many real observations).
real_flare_total = len(flare_df)
quiet_target = max(3000, real_flare_total * 2)  # at least 3000, at most 2x flares
quiet_target = min(quiet_target, len(quiet_df))  # never exceed what we have

quiet_sampled = quiet_df.sample(n=quiet_target, random_state=SEED)
print(f"QUIET: {len(quiet_df)} -> {quiet_target} (downsampled)")
print(f"Flare samples kept: {len(flare_df)} (all real, no removals)")

# ── augmentation helpers ─────────────────────────────────────────────────────

def bytes_to_arr(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).copy()

def arr_to_bytes(a: np.ndarray) -> bytes:
    return a.astype(np.float32).tobytes()

def augment_temporal(feats: np.ndarray, phases: np.ndarray | None,
                     T: int, F: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Augment a (T, F) time series with one physically plausible perturbation."""
    mat = feats.reshape(T, F)
    choice = rng.integers(0, 5)

    if choice == 0:
        # Gaussian detector noise: sigma = 0.5% of each channel's std
        sigma = np.std(mat, axis=0, keepdims=True) * 0.005 + 1e-9
        mat = mat + rng.normal(0, sigma, size=mat.shape)

    elif choice == 1:
        # Amplitude scaling +/-5-10%
        scale = rng.uniform(0.90, 1.10)
        mat = mat * scale

    elif choice == 2:
        # Temporal shift: roll along time axis by +/-1..3 steps
        shift = int(rng.integers(-3, 4))
        mat = np.roll(mat, shift, axis=0)
        if phases is not None:
            phases = np.roll(phases, shift)

    elif choice == 3:
        # Mild temporal stretch then crop/pad back to T
        factor  = rng.uniform(0.95, 1.05)
        new_T   = max(1, int(round(T * factor)))
        src_idx = np.arange(T, dtype=float)           # original positions
        dst_idx = np.linspace(0, T - 1, new_T)        # resampled positions
        stretched = np.stack(
            [np.interp(dst_idx, src_idx, mat[:, f]) for f in range(F)],
            axis=1,
        )  # shape (new_T, F)
        if new_T >= T:
            mat = stretched[:T, :]   # crop
        else:
            # pad by repeating last row
            pad = np.tile(stretched[-1:, :], (T - new_T, 1))
            mat = np.concatenate([stretched, pad], axis=0)

    elif choice == 4:
        # Tiny baseline drift per channel (linear ramp, up to +/-1% of range)
        rng_per_ch = np.ptp(mat, axis=0)
        rng_per_ch = np.where(rng_per_ch > 0, rng_per_ch, 1.0)
        start = rng.uniform(-0.01, 0.01, size=(F,)) * rng_per_ch
        end   = rng.uniform(-0.01, 0.01, size=(F,)) * rng_per_ch
        ramp  = np.linspace(start, end, T)   # (T, F)
        mat   = mat + ramp

    return mat.flatten(), phases


def augment_vector(feats: np.ndarray) -> np.ndarray:
    """Augment a flat feature vector with small amplitude or noise perturbation."""
    choice = rng.integers(0, 3)
    if choice == 0:
        sigma = np.abs(feats) * 0.005 + 1e-9
        feats = feats + rng.normal(0, sigma)
    elif choice == 1:
        feats = feats * rng.uniform(0.90, 1.10)
    elif choice == 2:
        rng_val = np.ptp(feats) if np.ptp(feats) > 0 else 1.0
        drift = rng.uniform(-0.01, 0.01) * rng_val
        feats = feats + drift
    return feats


def augment_rows(src_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sample n rows from src_df and augment each one."""
    if n <= 0 or len(src_df) == 0:
        return pd.DataFrame(columns=df.columns)

    indices = rng.integers(0, len(src_df), size=n)
    rows = []
    for idx in indices:
        row = src_df.iloc[idx].copy()

        if feat_col and repr_type == "temporal_sequence":
            feats  = bytes_to_arr(row[feat_col])
            phases = bytes_to_arr(row[phases_col]) if phases_col else None
            feats, phases = augment_temporal(feats, phases, n_timesteps, n_features_per_step)
            row[feat_col] = arr_to_bytes(feats)
            if phases_col and phases is not None:
                row[phases_col] = arr_to_bytes(phases)

        elif feat_col:
            feats = bytes_to_arr(row[feat_col])
            feats = augment_vector(feats)
            row[feat_col] = arr_to_bytes(feats)

        rows.append(row)

    return pd.DataFrame(rows)

# ── Step 4 & 5: compute augmentation targets from actual distribution ─────────

print("\n" + "=" * 60)
print("STEP 4-5 -- Computing augmentation targets from actual data")
print("=" * 60)

actual_counts = {}
for cls_id in FLARE_CLASSES:
    actual_counts[cls_id] = int((flare_df[label_col] == cls_id).sum())

# Strategy: bring rare classes up so that the rarest is at least 2x real count,
# but no class exceeds the natural scale relative to others.
# C and M have hundreds of real samples -- no augmentation needed (TSS is driven
# by B and X recall, which are nearly absent).
# B: boost to ~4x if present, or note absence.
# X: boost to ~2x.
# C, M: leave unchanged unless very sparse (< 100).
AUG_TARGETS = {}
for cls_id in FLARE_CLASSES:
    real = actual_counts[cls_id]
    if real == 0:
        AUG_TARGETS[cls_id] = 0
        continue
    name = CLASS_NAMES[cls_id]
    if name == "B":
        AUG_TARGETS[cls_id] = max(0, 4 * real - real)   # 3x extra (total = 4x)
    elif name == "X":
        AUG_TARGETS[cls_id] = max(0, 2 * real - real)   # 1x extra (total = 2x)
    elif name in ("C", "M") and real < 100:
        AUG_TARGETS[cls_id] = max(0, 200 - real)
    else:
        AUG_TARGETS[cls_id] = 0   # sufficient real samples

print(f"{'Class':<8} {'Real':>8} {'AugTarget':>10} {'Total':>8}")
print("-" * 38)
for cls_id in sorted(FLARE_CLASSES):
    name  = CLASS_NAMES.get(cls_id, str(cls_id))
    real  = actual_counts[cls_id]
    n_aug = AUG_TARGETS[cls_id]
    print(f"{name:<8} {real:>8} {n_aug:>10} {real + n_aug:>8}")

# ── Step 5 cont: run augmentation ────────────────────────────────────────────

augmented_frames = []
aug_counts = {}

for cls_id in FLARE_CLASSES:
    n_aug = AUG_TARGETS[cls_id]
    aug_counts[cls_id] = n_aug
    if n_aug > 0:
        cls_src = flare_df[flare_df[label_col] == cls_id]
        aug_df  = augment_rows(cls_src, n_aug)
        augmented_frames.append(aug_df)
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        print(f"Augmented {name}: +{n_aug} samples generated")

# ── Step 6: combine, shuffle, save ───────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6 -- Combining, shuffling, saving")
print("=" * 60)

all_parts = [quiet_sampled, flare_df] + augmented_frames
balanced_df = pd.concat(all_parts, ignore_index=True)
balanced_df = balanced_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

balanced_df.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved: {OUTPUT_PATH}")
print(f"Shape: {balanced_df.shape}")

# ── Step 7: print summary ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7 -- Summary")
print("=" * 60)

final_dist = balanced_df[label_col].value_counts().sort_index()
total_aug  = sum(aug_counts.values())

print(f"\n{'Class':<8} {'Original':>10} {'Balanced':>10} {'Augmented':>10}")
print("-" * 42)
all_ids = sorted(set(orig_dist.index) | set(final_dist.index))
for cls_id in all_ids:
    name   = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
    orig   = int(orig_dist.get(cls_id, 0))
    final  = int(final_dist.get(cls_id, 0))
    n_aug  = aug_counts.get(cls_id, 0)
    print(f"{name:<8} {orig:>10} {final:>10} {n_aug:>10}")

print(f"\nAugmented samples created : {total_aug}")
print(f"Original training size    : {len(df)}")
print(f"Final training size       : {len(balanced_df)}")
ratio_quiet = final_dist.get(0, 0) / max(1, sum(final_dist.get(c, 0) for c in FLARE_CLASSES))
print(f"QUIET:flare ratio (after) : {ratio_quiet:.2f}:1")
print(f"\nOutput: {OUTPUT_PATH.resolve()}")
print("\nNOTE: val.parquet and test.parquet are untouched.")
print("NOTE: For training, also consider:")
print("  - Focal loss or class-weighted CrossEntropy")
print("  - WeightedRandomSampler (no need to modify parquet)")
print("  - Per-class threshold optimization after training")
