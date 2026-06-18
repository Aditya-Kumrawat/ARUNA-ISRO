# ARUNA — Jupyter Notebooks

Internal scientific workspace for physics inspection, model validation, and one-off analysis.

**Not for operators.** Use the React dashboard (`web/`) for mission operations.

## Convention

| Prefix | Purpose |
|--------|---------|
| `physics_` | Spectral fitting, T/EM diagnostics, instrument calibration checks |
| `model_` | TCN / Moirai training runs, confusion matrices, calibration plots |
| `validation_` | V&V comparisons against published Sarwade et al. values |
| `event_` | Single-event deep-dive (e.g. `event_x63_20240222.ipynb`) |
| `catalog_` | Master Flare Catalog analysis, statistics |

## Setup

```bash
conda activate aruna
jupyter lab notebooks/
```
