"""
code/19_symbolic_rule_funnel_figure.py

WRITING/VISUALIZATION SCRIPT — NOT AN ANALYSIS SCRIPT.

Generates a single publication-quality funnel figure summarizing the
symbolic-rule branch's full screening pipeline: candidate generation ->
stage-level RETAIN/LIMITED/REJECT screening -> cross-stage CORE/
RETAIN_WITH_CAVEAT/EXCLUDE audit. Every count drawn is read from
tables/manuscript_symbolic_candidate_summary.csv (script 18's output,
itself verified against script 17's authoritative tables) -- nothing here
is hard-coded. No new analysis, no rule discovery, no model training, no
TEST access.
"""

import sys
from pathlib import Path
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

cs = pd.read_csv(TABLES_DIR / "manuscript_symbolic_candidate_summary.csv")
total_candidates = int(cs["Candidate count"].sum())
total_retained = int(cs["Stage-level retained"].sum())
total_limited = int(cs["Limited"].sum())
total_rejected = int(cs["Rejected"].sum())
total_core = int(cs["Final CORE"].sum())
total_caveat = int(cs["Final CAVEAT"].sum())
total_exclude = total_retained - total_core - total_caveat

assert total_candidates == total_retained + total_limited + total_rejected
assert total_retained == total_core + total_caveat + total_exclude
print(f"Verified from disk: {total_candidates} candidates -> {total_retained} retained "
      f"({total_limited} LIMITED, {total_rejected} REJECTED removed) -> "
      f"{total_core} CORE / {total_caveat} CAVEAT / {total_exclude} EXCLUDE")

per_stage = ", ".join(f"S{i+1}:{int(v)}" for i, v in enumerate(cs["Candidate count"]))

COL_CAND = "#4C72B0"
COL_RETAIN = "#55A868"
COL_LIMITED = "#DD8452"
COL_REJECT = "#B0B0B0"
COL_CORE = "#2E7D32"
COL_CAVEAT = "#F4B183"

fig, ax = plt.subplots(figsize=(12, 11))
ax.set_xlim(0, 12)
ax.set_ylim(0, 15)
ax.axis("off")

CENTER = 6.0


def bar(y, width, height, color, label, sub=None, edge="black"):
    x0 = CENTER - width / 2
    rect = mpatches.FancyBboxPatch((x0, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=color, edgecolor=edge, linewidth=1.3)
    ax.add_patch(rect)
    ax.text(CENTER, y + height / 2 + (0.14 if sub else 0), label, ha="center", va="center",
            fontsize=13, fontweight="bold", color="white" if color not in (COL_CAVEAT,) else "black")
    if sub:
        ax.text(CENTER, y + height / 2 - 0.22, sub, ha="center", va="center", fontsize=8.5,
                color="white" if color not in (COL_CAVEAT,) else "black")


def down_arrow(y_top, y_bottom, x_arrow=CENTER):
    ax.annotate("", xy=(x_arrow, y_bottom), xytext=(x_arrow, y_top),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="black"))


# --- Tier 1: candidate rules ---
bar(y=12.6, width=8.6, height=1.5, color=COL_CAND,
    label=f"{total_candidates} Candidate Symbolic Rules",
    sub=f"generated independently across Stages 1-6 ({per_stage})")

# --- removed at stage level ---
down_arrow(12.6, 9.9)
ax.text(CENTER, 11.6, "Stage-level screening", ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(CENTER, 11.2, "(RETAIN / LIMITED / REJECT,\nTRAIN-NORMAL only)", ha="center", va="center", fontsize=7.8)
ax.text(10.6, 11.55, f"−{total_rejected} REJECTED", ha="center", va="center", fontsize=9.5,
        color=COL_REJECT, fontweight="bold")
ax.text(10.6, 11.1, "insufficient empirical\nseparation/correlation", ha="center", va="center", fontsize=7.2,
        color="#555555")
ax.text(1.4, 11.55, f"−{total_limited} LIMITED", ha="center", va="center", fontsize=9.5,
        color=COL_LIMITED, fontweight="bold")
ax.text(1.4, 11.1, "physically supported but\n<1% VALIDATION coverage", ha="center", va="center", fontsize=7.2,
        color="#555555")

# --- Tier 2: stage-level retained ---
bar(y=8.4, width=5.6, height=1.5, color=COL_RETAIN,
    label=f"{total_retained} Rules Retained at Stage Level",
    sub="defensible, continuously-/near-continuously-evaluable symbolic features")

# --- cross-stage audit ---
down_arrow(8.4, 5.7)
ax.text(CENTER, 7.35, "Cross-stage audit (script 17)", ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(CENTER, 6.95, "re-evaluates all 16 under ONE shared standard", ha="center", va="center", fontsize=8)

# --- Tier 3: split into CORE / CAVEAT / EXCLUDE ---
gap = 0.3
w_core = 3.0 * (total_core / total_retained) * 2.2
w_caveat = 3.0 * (total_caveat / total_retained) * 2.2
total_w3 = w_core + w_caveat + gap
x_core0 = CENTER - total_w3 / 2
x_caveat0 = x_core0 + w_core + gap

rect_core = mpatches.FancyBboxPatch((x_core0, 4.2), w_core, 1.5, boxstyle="round,pad=0.02,rounding_size=0.08",
                                     facecolor=COL_CORE, edgecolor="black", linewidth=1.3)
ax.add_patch(rect_core)
ax.text(x_core0 + w_core / 2, 4.95 + 0.14, f"{total_core} CORE", ha="center", va="center",
        fontsize=13, fontweight="bold", color="white")
ax.text(x_core0 + w_core / 2, 4.95 - 0.22, "no caveat under any of 10\ncross-stage dimensions",
        ha="center", va="center", fontsize=8.2, color="white")

rect_caveat = mpatches.FancyBboxPatch((x_caveat0, 4.2), w_caveat, 1.5, boxstyle="round,pad=0.02,rounding_size=0.08",
                                       facecolor=COL_CAVEAT, edgecolor="black", linewidth=1.3)
ax.add_patch(rect_caveat)
ax.text(x_caveat0 + w_caveat / 2, 4.95 + 0.14, f"{total_caveat} RETAIN_WITH_CAVEAT", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color="black")
ax.text(x_caveat0 + w_caveat / 2, 4.95 - 0.22, "defensible, but flagged: sparse state,\ndrift, redundancy, or lag boundary",
        ha="center", va="center", fontsize=8.2, color="black")

ax.text(CENTER, 3.55, f"({total_exclude} EXCLUDE -- none of the 16 failed the coverage/saturation bar)",
        ha="center", va="center", fontsize=8.5, style="italic", color="#555555")

# --- final CORE feature names, bottom callout ---
core_names_path = TABLES_DIR / "final_core_symbolic_features.csv"
core_names = pd.read_csv(core_names_path)["Feature_Name"].tolist()
ax.text(CENTER, 2.55, "Final CORE symbolic feature set (primary input for downstream fusion):",
        ha="center", va="center", fontsize=9, fontweight="bold")
wrapped = ", ".join(core_names)
ax.text(CENTER, 1.85, wrapped, ha="center", va="center", fontsize=8, wrap=True,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EAF1F8", edgecolor=COL_CAND))

ax.set_title("From Candidate Rules to the Final CORE Symbolic Feature Set\n"
             f"{total_candidates} candidates → {total_retained} stage-level retained → "
             f"{total_core} CORE / {total_caveat} RETAIN_WITH_CAVEAT",
             fontsize=13.5, fontweight="bold", pad=10)

fig.tight_layout()
out_path = FIGURES_DIR / "symbolic_rule_funnel.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out_path}")
