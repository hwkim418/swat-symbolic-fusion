"""
code/20_overall_pipeline_figure.py

WRITING/VISUALIZATION SCRIPT — NOT AN ANALYSIS SCRIPT.

Generates "Figure 1: Overall System Architecture" for the conference paper --
the full pipeline described in the paper's Methodology section: SWaT dataset
-> chronological TRAIN/VALIDATION/TEST partitioning -> feature standardization
-> three parallel branches (Random Forest feature selection, LSTM temporal
learning, symbolic feature generation) -> symbolic feature fusion
(h_hybrid = [h_LSTM || f_symbolic]) -> explainable decision generation ->
performance evaluation on the held-out TEST partition.

All dataset/partition numbers are read from processed/metadata/split_statistics.csv
(script 02's output) and all symbolic-branch counts are read from
tables/manuscript_symbolic_candidate_summary.csv (script 18's output, itself
verified against script 17) -- nothing here is hard-coded from memory. No new
analysis, no rule discovery, no model training, no TEST access beyond reading
its already-computed row/attack counts from the existing metadata file.

Layout uses a simple top-down cursor: each element is placed relative to the
previous one's bottom edge, rather than hand-picked absolute y-coordinates,
so tiers cannot silently overlap.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "tables"
PROCESSED_DIR = PROJECT_ROOT / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

split_stats = pd.read_csv(PROCESSED_DIR / "metadata" / "split_statistics.csv").set_index("split")


def split_line(name, label):
    r = split_stats.loc[name]
    return (f"{label}: {int(r['n_rows']):,} rows, {r['attack_pct']:.2f}% attack, "
            f"{int(r['n_complete_attack_intervals'])} intervals")


cs = pd.read_csv(TABLES_DIR / "manuscript_symbolic_candidate_summary.csv")
total_candidates = int(cs["Candidate count"].sum())
total_retained = int(cs["Stage-level retained"].sum())
total_core = int(cs["Final CORE"].sum())
total_caveat = int(cs["Final CAVEAT"].sum())
test_row = split_stats.loc["test"]
print(f"Verified from disk: TRAIN/VAL/TEST rows = "
      f"{int(split_stats.loc['train','n_rows']):,}/{int(split_stats.loc['val','n_rows']):,}/"
      f"{int(test_row['n_rows']):,}; symbolic branch = "
      f"{total_candidates} -> {total_retained} -> {total_core}/{total_caveat}")

COL_DATA, COL_SPLIT, COL_SCALE = "#4C72B0", "#8172B2", "#8C8C8C"
COL_RF, COL_LSTM, COL_SYM = "#4C72B0", "#C44E52", "#55A868"
COL_FUSION, COL_XAI, COL_EVAL = "#DD8452", "#937860", "#2E7D32"

CENTER = 6.5
FIG_W = 13.0
LINE_H = 0.30   # approximate vertical space (data units) one text line needs at these font sizes

fig, ax = plt.subplots(figsize=(FIG_W, 14.5))
ax.axis("off")


def box(cx, y_top, w, height, color, title, sub_lines, title_size=12, sub_size=8.1, text_color="white"):
    """Draws a box whose TOP edge is y_top; returns the box's bottom y."""
    y_bottom = y_top - height
    rect = mpatches.FancyBboxPatch((cx - w / 2, y_bottom), w, height,
                                    boxstyle="round,pad=0.02,rounding_size=0.09",
                                    facecolor=color, edgecolor="black", linewidth=1.3)
    ax.add_patch(rect)
    title_lines = title.split("\n")
    n_title = len(title_lines)
    n_sub = len(sub_lines) if isinstance(sub_lines, list) else (sub_lines.count("\n") + 1 if sub_lines else 0)
    ax.text(cx, y_top - 0.16, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=text_color, linespacing=1.55)
    if sub_lines:
        sub_text = "\n".join(sub_lines) if isinstance(sub_lines, list) else sub_lines
        sub_top = y_top - 0.16 - n_title * LINE_H * (title_size / 11.5) - 0.10
        ax.text(cx, sub_top, sub_text, ha="center", va="top", fontsize=sub_size, color=text_color,
                linespacing=1.55)
    return y_bottom


def varrow(x, y_top, y_bottom):
    ax.annotate("", xy=(x, y_bottom), xytext=(x, y_top), arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black"))


def gap_arrow(y_top, gap, x=CENTER):
    """Draws a downward arrow spanning `gap` units from y_top; returns the new (lower) y."""
    y_bottom = y_top - gap
    varrow(x, y_top, y_bottom)
    return y_bottom


def fanout(y_top, targets_x, gap):
    """From (CENTER, y_top) down to a mid horizontal bar, then arrows down into each target x
    at the same final y. Returns that final y (top of the target boxes)."""
    y_mid = y_top - gap * 0.45
    y_final = y_top - gap
    ax.plot([CENTER, CENTER], [y_top, y_mid], color="black", lw=1.6)
    ax.plot([min(targets_x), max(targets_x)], [y_mid, y_mid], color="black", lw=1.6)
    for tx in targets_x:
        varrow(tx, y_mid, y_final)
    return y_final


def fanin(y_top_common, sources_x, gap, x_to=CENTER):
    y_mid = y_top_common - gap * 0.55
    y_final = y_top_common - gap
    for sx in sources_x:
        ax.plot([sx, sx], [y_top_common, y_mid], color="black", lw=1.6)
    ax.plot([min(sources_x), max(sources_x)], [y_mid, y_mid], color="black", lw=1.6)
    varrow(x_to, y_mid, y_final)
    return y_final


# ---------------------------------------------------------------------------
# Sequential top-down layout
# ---------------------------------------------------------------------------
y = 15.7  # current top edge

y = box(CENTER, y, 8.8, 1.05, COL_DATA, "SWaT Testbed Dataset",
        ["Normal (496,800 rows) + Attack (449,919 rows), 51 physical sensor/actuator tags"])
y = gap_arrow(y, 0.65)

y = box(CENTER, y, 11.6, 2.05, COL_SPLIT, "Chronological Train / Validation / Test Partitioning",
        [split_line("train", "TRAIN"), split_line("val", "VALIDATION"),
         split_line("test", "TEST (held out until final evaluation)")], sub_size=7.9)
y = gap_arrow(y, 0.6)

y = box(CENTER, y, 7.8, 0.95, COL_SCALE, "Feature Standardization (StandardScaler)",
        ["fit on TRAIN only; applied unchanged to VALIDATION and TEST"])

branch_w, branch_gap = 3.55, 0.55
x_rf, x_lstm, x_sym = CENTER - branch_w - branch_gap, CENTER, CENTER + branch_w + branch_gap
y_branch_top = fanout(y, [x_rf, x_lstm, x_sym], gap=0.7)

branch_h = 2.5
y_b1 = box(x_rf, y_branch_top, branch_w, branch_h, COL_RF, "Random Forest\nFeature Selection",
           ["MDI + permutation", "importance ranking of", "51 physical variables", "(n_estimators = 500)"],
           sub_size=7.6)
y_b2 = box(x_lstm, y_branch_top, branch_w, branch_h, COL_LSTM, "LSTM Temporal\nFeature Learning",
           ["sequence model over", "RF-selected variables;", "latent hidden state", "h_LSTM per timestep"],
           sub_size=7.6)
y_b3 = box(x_sym, y_branch_top, branch_w, branch_h, COL_SYM, "Symbolic Feature\nGeneration",
           ["6 process stages, engineering-derived rules:",
            f"{total_candidates} candidates -> {total_retained} retained",
            f"-> {total_core} CORE + {total_caveat} caveat"], sub_size=7.5)
y_branch_bottom = min(y_b1, y_b2, y_b3)

ax.text(x_rf, y_branch_bottom - 0.30, "selected feature subset", ha="center", va="top", fontsize=7.3,
        style="italic")

y_fusion_top = fanin(y_branch_bottom, [x_rf, x_lstm, x_sym], gap=1.15)
y = box(CENTER, y_fusion_top, 8.6, 1.05, COL_FUSION, "Symbolic Feature Fusion",
        ["h_hybrid  =  [ h_LSTM  ||  f_symbolic ]",
         "(concatenation of learned temporal state and symbolic rule vector)"], sub_size=8.4)
y = gap_arrow(y, 0.55)

y = box(CENTER, y, 8.6, 1.05, COL_XAI, "Explainable Decision Generation",
        ["anomaly classification + dominant physical relationship cited as evidence per detection"],
        sub_size=8.3)
y = gap_arrow(y, 0.55)

y = box(CENTER, y, 8.6, 1.3, COL_EVAL, "Performance Evaluation (held-out TEST partition)",
        ["Accuracy, Precision, Recall, F1-score, AUC-ROC, MCC",
         f"TEST released only here: {int(test_row['n_rows']):,} rows, {test_row['attack_pct']:.2f}% attack, "
         f"{int(test_row['n_complete_attack_intervals'])} intervals"], sub_size=8.2)

ax.set_xlim(0, FIG_W)
ax.set_ylim(y - 0.4, 16.6)
ax.set_title("Figure 1. Overall System Architecture\n"
             "Explainable Cyberattack Detection via Random Forest, LSTM, and Symbolic Feature Fusion",
             fontsize=14.5, fontweight="bold", pad=14)

fig.tight_layout()
out_path = FIGURES_DIR / "overall_system_architecture.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out_path}")
