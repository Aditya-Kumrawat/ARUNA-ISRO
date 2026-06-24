#!/usr/bin/env python3
"""
case_study_x63.py — 4-panel case-study figure for the X6.3 event 2024-02-22.

Uses the synthetic X6.3 data from demo_pipeline.py.
Produces docs/evidence/x63_case_study.png for reviewer presentation.

Usage:
    python python/scripts/case_study_x63.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, ".")
sys.path.insert(0, "python")

from demo_pipeline import mkf, TOTAL, PRE, ONSET, PEAK

out = Path("docs/evidence")
out.mkdir(parents=True, exist_ok=True)


def build_x63_traces():
    local_rng = np.random.default_rng(42)
    frames = [mkf(i, local_rng=local_rng) for i in range(TOTAL)]
    return {
        "t":          np.arange(TOTAL),
        "solexs_3_6": np.array([f.solexs_3_6kev   for f in frames]),
        "hel_10_25":  np.array([f.hel1os_10_25kev  for f in frames]),
        "T_MK":       np.array([f.T_MK             for f in frames]),
        "EM":         np.array([f.EM_cm3            for f in frames]),
        "SHEI":       np.array([f.SHEI              for f in frames]),
        "DNI":        np.array([f.DNI               for f in frames]),
        "hope_flag":  np.array([f.hope_flag         for f in frames], dtype=bool),
    }


def main():
    print("Building X6.3 synthetic traces ...")
    d = build_x63_traces()
    t = d["t"]

    phase_colors = {
        "PRE":       "#e0f2fe",
        "HOPE":      "#fef9c3",
        "IMPULSIVE": "#fee2e2",
        "GRADUAL":   "#f0fdf4",
    }

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(
        "ARUNA — X6.3 Case Study  |  2024-02-22  |  Synthetic Reconstruction",
        fontsize=12, fontweight="bold",
    )

    def shade_phases(ax):
        ax.axvspan(0,     PRE,   alpha=0.15, color=phase_colors["PRE"])
        ax.axvspan(PRE,   ONSET, alpha=0.15, color=phase_colors["HOPE"])
        ax.axvspan(ONSET, PEAK,  alpha=0.15, color=phase_colors["IMPULSIVE"])
        ax.axvspan(PEAK,  TOTAL, alpha=0.15, color=phase_colors["GRADUAL"])

    # Panel 1: raw counts
    ax = axes[0]
    shade_phases(ax)
    ax.semilogy(t, d["solexs_3_6"], color="#0284c7", lw=1.2, label="SoLEXS 3-6 keV")
    ax.semilogy(t, d["hel_10_25"],  color="#dc2626", lw=1.2, label="HEL1OS 10-25 keV")
    ax.set_ylabel("Count rate (ct/s)")
    ax.set_title("Panel 1 — Raw instrument count rates", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: T(t) and EM(t)
    ax = axes[1]
    shade_phases(ax)
    c1 = "#dc2626"
    l1, = ax.plot(t, d["T_MK"], color=c1, lw=1.5, label="T (MK)")
    ax.axhline(23.5, color=c1, linestyle="--", lw=1, alpha=0.7)
    ax.set_ylabel("Temperature (MK)", color=c1)
    ax.tick_params(axis="y", labelcolor=c1)
    ax2 = ax.twinx()
    c2  = "#7c3aed"
    l2, = ax2.semilogy(t, d["EM"], color=c2, lw=1.5, label="EM (cm-3)")
    ax2.set_ylabel("Emission Measure (cm-3)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax.set_title("Panel 2 — Plasma diagnostics: T(t) and EM(t)", fontsize=10)
    ref_line = Line2D([0], [0], color=c1, lw=1, linestyle="--",
                      label="Sarwade ref: 23.5 MK")
    ax.legend(handles=[l1, l2, ref_line], loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: SHEI + HOPE flag
    ax = axes[2]
    shade_phases(ax)
    ax.plot(t, d["SHEI"], color="#059669", lw=1.3, label="SHEI")
    ax.axhline(-0.8, color="#059669", linestyle=":", lw=1, alpha=0.7,
               label="SHEI > -0.8 impulsive threshold")
    hope_on = d["hope_flag"]
    for i in range(len(hope_on)):
        if hope_on[i]:
            ax.axvspan(t[i], t[i] + 1, color="#f59e0b", alpha=0.4)
    hope_patch = mpatches.Patch(color="#f59e0b", alpha=0.5, label="HOPE flag")
    handles, labels_l = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [hope_patch],
              labels=labels_l + ["HOPE flag"],
              loc="upper left", fontsize=8)
    ax.set_ylabel("SHEI (log ratio)")
    ax.set_title("Panel 3 — Spectral Hardness + HOPE precursor flag", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 4: DNI
    ax = axes[3]
    shade_phases(ax)
    ax.plot(t, d["DNI"], color="#0891b2", lw=1.4, label="DNI (Dynamic Neupert Integral)")
    ax.axhline(1.5, color="#dc2626", linestyle="--", lw=1.5,
               label="HOPE threshold DNI = 1.5")
    ax.set_ylabel("DNI (dimensionless)")
    ax.set_xlabel("Time since window start (seconds)")
    ax.set_title("Panel 4 — DNI: HOPE fires when DNI > 1.5", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Phase legend on panel 1
    phase_patches = [
        mpatches.Patch(color=phase_colors["PRE"],       alpha=0.6, label="Pre-flare"),
        mpatches.Patch(color=phase_colors["HOPE"],      alpha=0.6, label="HOPE precursor"),
        mpatches.Patch(color=phase_colors["IMPULSIVE"], alpha=0.6, label="Impulsive"),
        mpatches.Patch(color=phase_colors["GRADUAL"],   alpha=0.6, label="Gradual decay"),
    ]
    h0, l0 = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=h0 + phase_patches,
                   labels=l0 + [p.get_label() for p in phase_patches],
                   loc="upper left", fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(out / "x63_case_study.png", dpi=150, bbox_inches="tight")
    print("Saved docs/evidence/x63_case_study.png")

    peak_T     = d["T_MK"][PEAK - 1]
    peak_EM    = d["EM"][PEAK - 1]
    hope_first = np.where(d["hope_flag"])[0]
    if len(hope_first):
        lead_sec = PEAK - 1 - hope_first[0]
        status   = "PASS" if abs(peak_T - 23.5) < 2 else "CHECK"
        print(f"Peak T: {peak_T:.1f} MK  [Sarwade ref: 23.5 MK -- {status}]")
        print(f"Peak EM: {peak_EM:.2e} cm-3")
        lead_status = "PASS" if lead_sec >= 90 else "FAIL"
        print(f"HOPE first fires at t={hope_first[0]}s, peak at t={PEAK-1}s "
              f"-> lead {lead_sec}s [>= 90s: {lead_status}]")


if __name__ == "__main__":
    main()
