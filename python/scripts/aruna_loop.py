#!/usr/bin/env python3
"""
ARUNA Autonomous Research Loop

Runs hypothesis-driven experiments on the L3 nowcasting pipeline.
Each experiment changes ONE thing, validates against the frozen baseline,
saves all artefacts, and decides keep/revert.

Designed for Colab resilience: state is checkpointed to experiments/loop_state.json
after every experiment so a runtime restart can resume mid-backlog.

Usage:
    # First time — freeze baseline (MUST run before any experiments)
    python python/scripts/aruna_loop.py freeze-baseline \
        --physics-cache data/physics_cache \
        --goes-events data/goes_events.parquet

    # Run loop (picks next pending experiment automatically)
    python python/scripts/aruna_loop.py run \
        --physics-cache data/physics_cache \
        --goes-events data/goes_events.parquet \
        --max-experiments 5

    # Show status
    python python/scripts/aruna_loop.py status

Scoring function (Primary > Constraints > Secondary):
    PRIMARY:   Maximise recall_high_class_MX  (M+X recall)
    CONSTRAINT: false_alarms_per_day < 1.0
    CONSTRAINT: goes_class_accuracy > 0.50
    SECONDARY:  Maximise recall_total, minimise lead_time_mean_min
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("aruna_loop")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
STATE_FILE = EXPERIMENTS_DIR / "loop_state.json"
BASELINE_DIR = EXPERIMENTS_DIR / "baseline"

# ── Scoring ────────────────────────────────────────────────────────────────────

def score(metrics: Dict) -> Optional[float]:
    """
    Single scalar score for comparing experiments.
    Returns None if constraints are violated (disqualified).
    """
    mx_recall = metrics.get("recall_high_class_MX")
    fa_per_day = metrics.get("false_alarms_per_day")
    class_acc  = metrics.get("goes_class_accuracy")

    if mx_recall is None:
        return None

    # Hard constraints
    if fa_per_day is not None and fa_per_day > 1.0:
        return None  # too many false alarms
    if class_acc is not None and class_acc < 0.50:
        return None  # classification is broken

    # Primary objective
    primary = float(mx_recall)

    # Secondary terms (small weights so they don't override primary)
    total_recall = float(metrics.get("recall_total") or 0.0)
    lead_min = float(metrics.get("lead_time_mean_min") or 60.0)
    lead_score = max(0.0, 1.0 - lead_min / 60.0)  # 0 min lead → 1.0, 60 min → 0.0

    return primary + 0.05 * total_recall + 0.02 * lead_score


def is_better(new_score: Optional[float], baseline_score: Optional[float]) -> bool:
    if new_score is None:
        return False
    if baseline_score is None:
        return True
    return new_score > baseline_score


# ── Experiment backlog ─────────────────────────────────────────────────────────

DEFAULT_BACKLOG: List[Dict] = [
    {
        "id": "exp_001",
        "name": "B1 SoLEXS calibration",
        "hypothesis": "The default solexs_to_wm2_factor=1e-8 is uncalibrated. "
                      "Fitting from matched events will improve GOES class accuracy.",
        "change": "Run validate_l3.py --calibrate to fit the conversion factor, "
                  "then update NowcastConfig.solexs_to_wm2_factor with the fitted value.",
        "metric_to_watch": "goes_class_accuracy",
        "flag_extra": "--calibrate",
        "status": "pending",
    },
    {
        "id": "exp_002",
        "name": "B2 HEL1OS-only degraded mode",
        "hypothesis": "When SoLEXS is offline, HEL1OS alone should still emit detections. "
                      "The fix (HEL1OS_ONLY mode) should show non-zero recall when SoLEXS zeroed.",
        "change": "Run validate_l3.py --zero-solexs. Expect recall_total > 0.",
        "metric_to_watch": "recall_total",
        "flag_extra": "--zero-solexs",
        "status": "pending",
    },
    {
        "id": "exp_003",
        "name": "B3 HEL changepoint as true trigger",
        "hypothesis": "Using HEL changepoint (not just SHEI proxy) as the HEL trigger "
                      "should improve dual-trigger sensitivity especially for hard-spectrum events.",
        "change": "Already applied in orchestrator.py. Measure impact on full-mode recall_high_class_MX.",
        "metric_to_watch": "recall_high_class_MX",
        "flag_extra": "",
        "status": "pending",
    },
    {
        "id": "exp_004",
        "name": "Wider dual-trigger window (180→300s)",
        "hypothesis": "Asynchronous SoLEXS/HEL1OS onsets can exceed 3 min. "
                      "Widening to 5 min may recover missed dual-trigger events.",
        "change": "Set NowcastConfig.dual_trigger_window_sec=300 and re-run.",
        "metric_to_watch": "recall_high_class_MX",
        "flag_extra": "",
        "config_patch": {"dual_trigger_window_sec": 300},
        "status": "pending",
    },
    {
        "id": "exp_005",
        "name": "Lower sigma multiplier (3→2)",
        "hypothesis": "P85 threshold = baseline + 3σ may be too conservative for quiet Sun. "
                      "2σ should increase sensitivity at cost of more FA.",
        "change": "Set NowcastConfig.trigger_sigma_multiplier=2.0.",
        "metric_to_watch": "recall_high_class_MX",
        "flag_extra": "",
        "config_patch": {"trigger_sigma_multiplier": 2.0},
        "status": "pending",
    },
    {
        "id": "exp_006",
        "name": "Narrower PELT penalty (1.5→0.8)",
        "hypothesis": "Relaxing PELT penalty detects more subtle flux rises, "
                      "potentially catching early B/C-class precursors.",
        "change": "Set NowcastConfig.pelt_penalty=0.8.",
        "metric_to_watch": "recall_low_class_BC",
        "flag_extra": "",
        "config_patch": {"pelt_penalty": 0.8},
        "status": "pending",
    },
    {
        "id": "exp_007",
        "name": "Wider GOES matching window (10→15 min)",
        "hypothesis": "L3 fires at onset, not peak. A wider match window may correctly "
                      "credit detections that currently appear as false alarms.",
        "change": "Edit MATCH_WINDOW_MIN=15 in validate_l3.py.",
        "metric_to_watch": "precision",
        "flag_extra": "",
        "status": "pending",
    },
    {
        "id": "exp_008",
        "name": "Peak decay threshold relaxation (0.5→0.3)",
        "hypothesis": "Long-duration events (LDE) may exit PEAK too early if flux "
                      "decays slowly. A lower threshold keeps the event active longer.",
        "change": "Set NowcastConfig.peak_decay_threshold=0.3.",
        "metric_to_watch": "recall_high_class_MX",
        "flag_extra": "",
        "config_patch": {"peak_decay_threshold": 0.3},
        "status": "pending",
    },
]


# ── State management ───────────────────────────────────────────────────────────

def load_state() -> Dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_score": None,
        "baseline_metrics": None,
        "experiments": DEFAULT_BACKLOG[:],
    }


def save_state(state: Dict) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── Subprocess runner ──────────────────────────────────────────────────────────

def run_validate(
    physics_cache: Path,
    goes_events: Path,
    out_dir: Path,
    extra_flags: str = "",
    config_patch: Optional[Dict] = None,
) -> Optional[Dict]:
    """Run validate_l3.py as a subprocess and return parsed metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "python/scripts/validate_l3.py"),
        "--physics-cache", str(physics_cache),
        "--goes-events",   str(goes_events),
        "--out",           str(out_dir),
    ]
    if extra_flags:
        cmd += extra_flags.split()

    env = os.environ.copy()

    # Apply config patches by writing a temporary NowcastConfig override file
    if config_patch:
        patch_path = out_dir / "config_patch.json"
        with open(patch_path, "w") as f:
            json.dump(config_patch, f)
        # Inject path so validate_l3.py can pick it up if needed
        env["ARUNA_L3_CONFIG_PATCH"] = str(patch_path)
        # Also write it to config note
        with open(out_dir / "config_patch_applied.json", "w") as f:
            json.dump(config_patch, f, indent=2)

    logger.info("Running: %s", " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0

    # Save logs
    with open(out_dir / "stdout.txt", "w") as f:
        f.write(result.stdout)
    with open(out_dir / "stderr.txt", "w") as f:
        f.write(result.stderr)

    if result.returncode != 0:
        logger.error("validate_l3 exited %d:\n%s", result.returncode, result.stderr[-2000:])
        return None

    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        logger.error("metrics.json not found at %s", metrics_path)
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)
    metrics["_elapsed_sec"] = round(elapsed, 1)
    return metrics


def write_experiment_report(
    exp: Dict,
    metrics: Dict,
    exp_score: Optional[float],
    baseline_score: Optional[float],
    kept: bool,
    out_dir: Path,
) -> None:
    verdict = "KEPT ✓" if kept else "REVERTED ✗"
    lines = [
        f"# Experiment {exp['id']}: {exp['name']}",
        f"Date: {datetime.now(timezone.utc).isoformat()}",
        f"Verdict: {verdict}",
        "",
        "## Hypothesis",
        exp.get("hypothesis", ""),
        "",
        "## Change Made",
        exp.get("change", ""),
        "",
        "## Results",
        f"Score:          {exp_score}  (baseline: {baseline_score})",
        f"MX Recall:      {metrics.get('recall_high_class_MX')}",
        f"Total Recall:   {metrics.get('recall_total')}",
        f"Precision:      {metrics.get('precision')}",
        f"FA/day:         {metrics.get('false_alarms_per_day')}",
        f"Class Acc:      {metrics.get('goes_class_accuracy')}",
        f"Lead time mean: {metrics.get('lead_time_mean_min')} min",
        "",
        "## Decision",
    ]
    if kept:
        lines.append(f"Score improved from {baseline_score} → {exp_score}. Change accepted.")
    else:
        if exp_score is None:
            lines.append("Score = None (constraint violated). Change reverted.")
        else:
            lines.append(f"Score did not improve ({exp_score} ≤ {baseline_score}). Change reverted.")

    lines += [
        "",
        "## Next Experiment",
        "See experiments/loop_state.json for the backlog.",
    ]
    with open(out_dir / "report.md", "w") as f:
        f.write("\n".join(lines))


# ── Config patching ────────────────────────────────────────────────────────────

def apply_config_patch_to_models(patch: Dict) -> None:
    """
    Apply config_patch to NowcastConfig defaults in models.py.
    This is a code-level patch so the subprocess picks it up without env vars.
    """
    models_path = REPO_ROOT / "python/aruna/l3_nowcasting/models.py"
    with open(models_path) as f:
        src = f.read()

    import re
    for key, val in patch.items():
        # Match lines like:  key: type = old_value
        pattern = rf"(\s+{key}\s*:\s*\S+\s*=\s*)\S+"
        replacement = rf"\g<1>{val}"
        new_src = re.sub(pattern, replacement, src)
        if new_src != src:
            src = new_src
            logger.info("Patched %s = %s in models.py", key, val)
        else:
            logger.warning("Could not patch %s in models.py — pattern not found", key)

    with open(models_path, "w") as f:
        f.write(src)


def revert_config_patch(patch: Dict, original_values: Dict) -> None:
    """Revert config_patch by restoring original values."""
    if original_values:
        apply_config_patch_to_models(original_values)


def read_current_config_values(patch: Dict) -> Dict:
    """Read current values for patched keys from models.py."""
    import re
    models_path = REPO_ROOT / "python/aruna/l3_nowcasting/models.py"
    with open(models_path) as f:
        src = f.read()
    originals = {}
    for key in patch:
        m = re.search(rf"\s+{key}\s*:\s*\S+\s*=\s*(\S+)", src)
        if m:
            raw = m.group(1)
            try:
                originals[key] = float(raw) if "." in raw else int(raw)
            except ValueError:
                originals[key] = raw
    return originals


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_freeze_baseline(args):
    logger.info("Freezing baseline — running full L3 validation...")
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = run_validate(
        Path(args.physics_cache),
        Path(args.goes_events),
        BASELINE_DIR,
    )
    if metrics is None:
        logger.error("Baseline run failed — see %s/stderr.txt", BASELINE_DIR)
        sys.exit(1)

    baseline_score = score(metrics)
    state = load_state()
    state["baseline_score"] = baseline_score
    state["baseline_metrics"] = metrics
    state["baseline_frozen_at"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    print(f"\n{'='*60}")
    print("BASELINE FROZEN")
    print(f"  Score:          {baseline_score}")
    print(f"  MX Recall:      {metrics.get('recall_high_class_MX')}")
    print(f"  Total Recall:   {metrics.get('recall_total')}")
    print(f"  Precision:      {metrics.get('precision')}")
    print(f"  FA/day:         {metrics.get('false_alarms_per_day')}")
    print(f"  Class Acc:      {metrics.get('goes_class_accuracy')}")
    print(f"Saved to {STATE_FILE}")
    print("Do NOT touch val.parquet or test.parquet after this.")
    print(f"{'='*60}\n")


def cmd_run(args):
    state = load_state()

    if state.get("baseline_score") is None:
        logger.error("No baseline frozen yet. Run: aruna_loop.py freeze-baseline first.")
        sys.exit(1)

    baseline_score = state["baseline_score"]
    n_run = 0
    max_exp = args.max_experiments or 999

    for exp in state["experiments"]:
        if n_run >= max_exp:
            break
        if exp["status"] != "pending":
            continue

        logger.info("\n%s", "=" * 60)
        logger.info("EXPERIMENT %s: %s", exp["id"], exp["name"])
        logger.info("Hypothesis: %s", exp.get("hypothesis", ""))
        logger.info("Change: %s", exp.get("change", ""))
        logger.info("%s", "=" * 60)

        exp_dir = EXPERIMENTS_DIR / exp["id"]
        config_patch = exp.get("config_patch")
        original_values: Dict = {}

        # Apply config patch
        if config_patch:
            original_values = read_current_config_values(config_patch)
            apply_config_patch_to_models(config_patch)

        # Run validation
        metrics = run_validate(
            Path(args.physics_cache),
            Path(args.goes_events),
            exp_dir,
            extra_flags=exp.get("flag_extra", ""),
            config_patch=config_patch,
        )

        if metrics is None:
            logger.error("Experiment %s failed — reverting", exp["id"])
            if config_patch and original_values:
                revert_config_patch(config_patch, original_values)
            exp["status"] = "failed"
            exp["failed_at"] = datetime.now(timezone.utc).isoformat()
            save_state(state)
            n_run += 1
            continue

        exp_score = score(metrics)
        kept = is_better(exp_score, baseline_score)

        # Write experiment report
        write_experiment_report(exp, metrics, exp_score, baseline_score, kept, exp_dir)

        if kept:
            logger.info("KEPT: score %.4f → %.4f", baseline_score or 0, exp_score or 0)
            state["baseline_score"] = exp_score
            state["baseline_metrics"] = metrics
            baseline_score = exp_score
            exp["status"] = "kept"
            exp["kept_score"] = exp_score
        else:
            logger.info("REVERTED: score %.4f did not improve baseline %.4f",
                        exp_score or 0, baseline_score or 0)
            if config_patch and original_values:
                revert_config_patch(config_patch, original_values)
            exp["status"] = "reverted"
            exp["reverted_score"] = exp_score

        exp["metrics"] = metrics
        exp["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_state(state)

        n_run += 1
        logger.info("Experiment complete. %d remaining in backlog.",
                    sum(1 for e in state["experiments"] if e["status"] == "pending"))

    logger.info("Loop complete. %d experiments run this session.", n_run)
    cmd_status(args)


def cmd_status(args):
    state = load_state()
    print(f"\n{'='*60}")
    print("ARUNA EXPERIMENT LOOP STATUS")
    print(f"{'='*60}")
    print(f"Baseline score:   {state.get('baseline_score')}")
    bm = state.get("baseline_metrics") or {}
    print(f"Baseline MX rec:  {bm.get('recall_high_class_MX')}")
    print(f"Baseline FA/day:  {bm.get('false_alarms_per_day')}")
    print(f"Baseline cls acc: {bm.get('goes_class_accuracy')}")
    print()
    statuses = {}
    for exp in state.get("experiments", []):
        s = exp["status"]
        statuses[s] = statuses.get(s, 0) + 1
    for s, n in sorted(statuses.items()):
        print(f"  {s:<12} {n}")
    print()
    print("Backlog:")
    for exp in state.get("experiments", []):
        mark = {"pending": "□", "kept": "✓", "reverted": "✗", "failed": "!"}.get(exp["status"], "?")
        sc = exp.get("kept_score") or exp.get("reverted_score") or ""
        print(f"  {mark} {exp['id']} [{exp['status']:<8}]  {exp['name']}"
              + (f"  (score={sc:.4f})" if sc else ""))
    print(f"{'='*60}\n")


def cmd_add(args):
    """Add a custom experiment to the backlog."""
    state = load_state()
    exp_ids = [e["id"] for e in state["experiments"]]
    # Auto-generate next ID
    existing_nums = [int(e[4:]) for e in exp_ids if e.startswith("exp_") and e[4:].isdigit()]
    next_num = max(existing_nums, default=0) + 1
    new_id = f"exp_{next_num:03d}"
    new_exp = {
        "id": new_id,
        "name": args.name,
        "hypothesis": args.hypothesis or "",
        "change": args.change or "",
        "metric_to_watch": args.metric or "recall_high_class_MX",
        "flag_extra": args.flags or "",
        "status": "pending",
    }
    if args.config_patch:
        new_exp["config_patch"] = json.loads(args.config_patch)
    state["experiments"].append(new_exp)
    save_state(state)
    print(f"Added {new_id}: {args.name}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="ARUNA Autonomous Research Loop")
    sub = p.add_subparsers(dest="command")

    # freeze-baseline
    freeze = sub.add_parser("freeze-baseline", help="Run and freeze the L3 baseline")
    freeze.add_argument("--physics-cache", default="data/physics_cache")
    freeze.add_argument("--goes-events",   default="data/goes_events.parquet")

    # run
    run = sub.add_parser("run", help="Run next pending experiment(s)")
    run.add_argument("--physics-cache", default="data/physics_cache")
    run.add_argument("--goes-events",   default="data/goes_events.parquet")
    run.add_argument("--max-experiments", type=int, default=None,
                     help="Stop after this many experiments (default: all pending)")

    # status
    sub.add_parser("status", help="Show loop status")

    # add
    add = sub.add_parser("add", help="Add experiment to backlog")
    add.add_argument("--name",         required=True)
    add.add_argument("--hypothesis",   default="")
    add.add_argument("--change",       default="")
    add.add_argument("--metric",       default="recall_high_class_MX")
    add.add_argument("--flags",        default="")
    add.add_argument("--config-patch", default=None,
                     help='JSON dict of NowcastConfig patches, e.g. \'{"pelt_penalty":0.5}\'')

    args = p.parse_args()

    if args.command == "freeze-baseline":
        cmd_freeze_baseline(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "add":
        cmd_add(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
