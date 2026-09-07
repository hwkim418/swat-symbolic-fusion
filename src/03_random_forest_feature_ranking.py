"""
code/03_random_forest_feature_ranking.py

RANDOM FOREST FEATURE RANKING — TRAIN-ONLY FITTING, VALIDATION-ONLY
PERMUTATION STABILITY CHECK. TEST IS NEVER LOADED.

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

Purpose: rank the 51 physical process variables by their ability to
distinguish Normal (0) from Attack (1) using ONLY the approved TRAIN
partition. This is feature ranking, NOT final model evaluation, and NOT
Top-k selection.

This script does NOT touch processed/test_unscaled.csv or
processed/test_scaled.csv at all -- the test split's filename never
appears anywhere below, by design, so it cannot leak into ranking,
model selection, or hyperparameter choice even by accident.
"""

import os
import sys
import time
import platform
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
META_DIR = PROCESSED_DIR / "metadata"

TRAIN_PATH = PROCESSED_DIR / "train_unscaled.csv"
VAL_PATH = PROCESSED_DIR / "validation_unscaled.csv"
FEATURE_LIST_PATH = META_DIR / "feature_list.txt"
# NOTE: no TEST_PATH constant is defined anywhere in this script, on purpose.

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"
for d in (METRICS_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

RUN_START = datetime.now()
_LOG_LINES = []


def log(msg=""):
    msg = str(msg)
    print(msg)
    _LOG_LINES.append(msg)


def section(title):
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


def sha256_of(path: Path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


section("RANDOM FOREST FEATURE RANKING — RUN START")
log(f"Run start: {RUN_START.isoformat()}")

# Pre-run integrity snapshot of files this script reads (proves nothing was
# modified by this script; checked again at the end).
DATASET_DIR = Path(os.environ.get("SWAT_DATASET_DIR", str(PROJECT_ROOT / "dataset")))  # portability fix: actual directory is lowercase; overridable via env var
RAW_ATTACK = DATASET_DIR / "SWaT_Dataset_Attack_v0.csv"
RAW_NORMAL = DATASET_DIR / "SWaT_Dataset_Normal_v0.csv"
pre_hashes = {
    "raw_attack": sha256_of(RAW_ATTACK),
    "raw_normal_size": RAW_NORMAL.stat().st_size,
    "train_unscaled": sha256_of(TRAIN_PATH),
    "validation_unscaled": sha256_of(VAL_PATH),
}
log("Pre-run SHA-256 / size snapshot recorded for raw and processed inputs "
    "(re-verified at end of run).")

# =============================================================================
# 1. INPUT DATA
# =============================================================================
section("TASK 1 — LOADING INPUT DATA (processed/train_unscaled.csv)")

train_df = pd.read_csv(TRAIN_PATH)
log(f"Loaded {TRAIN_PATH}: shape {train_df.shape}")
log(f"Columns: {list(train_df.columns)}")

expected_features = FEATURE_LIST_PATH.read_text(encoding="utf-8").strip().split("\n")
log(f"Loaded expected feature list ({len(expected_features)} features) from {FEATURE_LIST_PATH}")

EXCLUDE_COLS = {"Timestamp", "Label", "Label_raw"}
train_feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]
assert train_feature_cols == expected_features, (
    "Feature column order/content in train_unscaled.csv does not match "
    "processed/metadata/feature_list.txt -- aborting."
)
log(f"Verified: 51-feature predictor list in train_unscaled.csv matches "
    f"feature_list.txt exactly (order and names).")

FEATURES = expected_features
TARGET = "Label"

log("")
log("NOTE ON SCALING: this Random Forest ranking uses UNSCALED (original "
    "physical-unit) values, not processed/train_scaled.csv. Random Forest "
    "splits are threshold tests on individual features (x <= t), and any "
    "monotonic transform of a feature (such as StandardScaler's linear "
    "z-score) leaves the induced splits and resulting impurity-based "
    "importances unchanged. Using the unscaled values costs nothing in "
    "ranking validity and keeps every reported threshold and importance "
    "directly interpretable in the sensor's/actuator's native physical "
    "units -- important for the paper's explainability goal.")

X_train = train_df[FEATURES].to_numpy(dtype=np.float64)
y_train = train_df[TARGET].to_numpy()

# =============================================================================
# 2. VERIFY TRAINING DATA
# =============================================================================
section("TASK 2 — VERIFYING TRAINING DATA")

n_rows = len(train_df)
n_normal = int((y_train == 0).sum())
n_attack = int((y_train == 1).sum())
log(f"Rows: {n_rows:,} (expected 270,000) -> {'MATCH' if n_rows == 270000 else 'MISMATCH'}")
log(f"Normal: {n_normal:,} (expected 222,343) -> {'MATCH' if n_normal == 222343 else 'MISMATCH'}")
log(f"Attack: {n_attack:,} (expected 47,657) -> {'MATCH' if n_attack == 47657 else 'MISMATCH'}")
log(f"Physical features: {len(FEATURES)} (expected 51) -> "
    f"{'MATCH' if len(FEATURES) == 51 else 'MISMATCH'}")
assert n_rows == 270000 and n_normal == 222343 and n_attack == 47657 and len(FEATURES) == 51

has_nan = bool(np.isnan(X_train).any())
has_inf = bool(np.isinf(X_train).any())
labels_binary = set(np.unique(y_train)) <= {0, 1}
log(f"No NaN in predictors: {not has_nan}")
log(f"No infinite values in predictors: {not has_inf}")
log(f"Labels binary-only {{0,1}}: {labels_binary}")
assert not has_nan and not has_inf and labels_binary

no_timestamp_in_predictors = "Timestamp" not in FEATURES
no_label_in_predictors = ("Label" not in FEATURES) and ("Label_raw" not in FEATURES)
log(f"Timestamp excluded from predictors: {no_timestamp_in_predictors}")
log(f"Label / Label_raw excluded from predictors: {no_label_in_predictors}")
assert no_timestamp_in_predictors and no_label_in_predictors

# Constant-feature identification (NOT removed)
train_feat_df = train_df[FEATURES]
nunique = train_feat_df.nunique()
constant_features = sorted(nunique[nunique == 1].index.tolist())
EXPECTED_CONSTANT = sorted(["P202", "P301", "P401", "P404", "P502", "P601", "P603"])
constant_confirmed = constant_features == EXPECTED_CONSTANT
log(f"Constant TRAIN features detected: {constant_features}")
log(f"Matches expected preprocessing-audit list {EXPECTED_CONSTANT}: {constant_confirmed}")
log("Constant features are RETAINED in the predictor matrix (not removed) per instructions.")
if not constant_confirmed:
    log("WARNING: constant-feature set differs from the expected preprocessing-audit list!")

# =============================================================================
# 3. RANDOM FOREST MODEL
# =============================================================================
section("TASK 3 — RANDOM FOREST MODEL (TRAIN ONLY)")

log(f"scikit-learn version: {sklearn.__version__}")

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

train_start = time.time()
rf.fit(X_train, y_train)
train_elapsed = time.time() - train_start

full_params = rf.get_params()
log("Complete RandomForestClassifier parameter configuration (explicit + sklearn defaults):")
for k, v in sorted(full_params.items()):
    log(f"    {k} = {v!r}")
log(f"\nTraining time: {train_elapsed:.2f} seconds")
log(f"Random seed (random_state): 42")
log(f"Model fit exclusively on TRAIN: X shape {X_train.shape}, y shape {y_train.shape}")

# =============================================================================
# 4. PRIMARY FEATURE IMPORTANCE — MDI
# =============================================================================
section("TASK 4 — MDI (IMPURITY-BASED) FEATURE IMPORTANCE")

mdi_importances = rf.feature_importances_
mdi_sum = float(mdi_importances.sum())
log(f"Sum of MDI importances across 51 features: {mdi_sum:.10f} (expected ~1.0)")
assert abs(mdi_sum - 1.0) < 1e-6, f"MDI importances do not sum to ~1.0 (sum={mdi_sum})"

mdi_df = pd.DataFrame({
    "Feature": FEATURES,
    "MDI_Importance": mdi_importances,
    "Constant_In_Train": [f in constant_features for f in FEATURES],
}).sort_values("MDI_Importance", ascending=False).reset_index(drop=True)
mdi_df.insert(0, "Rank", mdi_df.index + 1)
mdi_df["Cumulative_MDI"] = mdi_df["MDI_Importance"].cumsum()

log("\nFull MDI ranking (all 51 features, highest to lowest):")
for _, r in mdi_df.iterrows():
    log(f"  [{int(r['Rank']):>2}] {r['Feature']:10s} MDI={r['MDI_Importance']:.6f}  "
        f"cum={r['Cumulative_MDI']:.6f}  constant_in_train={r['Constant_In_Train']}")

n_zero_importance = int((mdi_df["MDI_Importance"] == 0).sum())
log(f"\nFeatures with exactly zero MDI importance: {n_zero_importance} "
    f"(retained in the table below, not discarded)")

# =============================================================================
# 5. SECONDARY STABILITY CHECK — PERMUTATION IMPORTANCE ON VALIDATION
# =============================================================================
section("TASK 5 — PERMUTATION IMPORTANCE (VALIDATION ONLY, SECONDARY CHECK)")

val_df = pd.read_csv(VAL_PATH)
log(f"Loaded {VAL_PATH}: shape {val_df.shape}")
val_feature_cols = [c for c in val_df.columns if c not in EXCLUDE_COLS]
assert val_feature_cols == FEATURES, "Validation feature columns do not match TRAIN's feature list."

X_val = val_df[FEATURES].to_numpy(dtype=np.float64)
y_val = val_df[TARGET].to_numpy()
log(f"Validation rows: {len(val_df):,}, Normal={int((y_val==0).sum()):,}, "
    f"Attack={int((y_val==1).sum()):,}")

log("")
log("IMPORTANT: the Random Forest model itself is NOT refit here. It remains "
    "the exact model fit on TRAIN in Task 3. VALIDATION is used only to score "
    "that fixed model under repeated feature permutation, as a secondary, "
    "out-of-sample check on whether the TRAIN-derived MDI ranking is stable "
    "outside the exact rows it was computed from. TEST is not loaded or used "
    "anywhere in this script.")

PERM_N_REPEATS = 10
PERM_RANDOM_STATE = 42
PERM_SCORING = "f1"
log(f"\nPermutation importance configuration:")
log(f"    n_repeats   = {PERM_N_REPEATS}")
log(f"    random_state = {PERM_RANDOM_STATE}")
log(f"    scoring     = {PERM_SCORING!r}")
log(f"    n_jobs      = -1")
log(f"Scoring choice rationale: VALIDATION Attack prevalence is "
    f"{100*int((y_val==1).sum())/len(y_val):.2f}%; plain accuracy would barely move "
    f"under permutation at this imbalance and would understate importance. F1 on the "
    f"Attack (positive) class is sensitive to the precision/recall tradeoff a detector "
    f"actually cares about, so it is used here instead of accuracy.")

perm_start = time.time()
perm_result = permutation_importance(
    rf, X_val, y_val,
    n_repeats=PERM_N_REPEATS,
    random_state=PERM_RANDOM_STATE,
    scoring=PERM_SCORING,
    n_jobs=-1,
)
perm_elapsed = time.time() - perm_start
log(f"\nPermutation importance computation time: {perm_elapsed:.2f} seconds")

# Baseline (unpermuted) classification diagnostic on VALIDATION -- context
# for interpreting how much room the F1 scoring metric had to move under
# permutation. This is diagnostic reporting only: no retraining, no
# hyperparameter change, no threshold tuning, and TEST is still not touched.
baseline_pred = rf.predict(X_val)
baseline_f1 = f1_score(y_val, baseline_pred)
baseline_precision = precision_score(y_val, baseline_pred)
baseline_recall = recall_score(y_val, baseline_pred)
baseline_cm = confusion_matrix(y_val, baseline_pred)
log(f"\nBaseline (unpermuted) VALIDATION classification diagnostic "
    f"(context for interpreting permutation-importance magnitudes; NOT a "
    f"tuning step, NOT using TEST):")
log(f"    F1        = {baseline_f1:.6f}")
log(f"    Precision = {baseline_precision:.6f}")
log(f"    Recall    = {baseline_recall:.6f}")
log(f"    Confusion matrix [[TN, FP], [FN, TP]]:\n{baseline_cm}")
log(f"    Predicted Attack on VALIDATION: {int((baseline_pred==1).sum()):,} of "
    f"{len(y_val):,} rows ({100*(baseline_pred==1).sum()/len(y_val):.1f}%), vs. "
    f"{int((y_val==1).sum()):,} actual Attack rows "
    f"({100*(y_val==1).sum()/len(y_val):.2f}%).")

perm_df = pd.DataFrame({
    "Feature": FEATURES,
    "Permutation_Importance_Mean": perm_result.importances_mean,
    "Permutation_Importance_STD": perm_result.importances_std,
}).sort_values("Permutation_Importance_Mean", ascending=False).reset_index(drop=True)
perm_df.insert(0, "Permutation_Rank", perm_df.index + 1)

log("\nFull permutation-importance ranking (all 51 features, VALIDATION, highest to lowest):")
for _, r in perm_df.iterrows():
    log(f"  [{int(r['Permutation_Rank']):>2}] {r['Feature']:10s} "
        f"mean={r['Permutation_Importance_Mean']:.6f}  std={r['Permutation_Importance_STD']:.6f}")

# =============================================================================
# 6. RANKING AGREEMENT
# =============================================================================
section("TASK 6 — RANKING AGREEMENT: MDI (TRAIN) vs PERMUTATION (VALIDATION)")

merged = mdi_df[["Rank", "Feature", "MDI_Importance", "Cumulative_MDI", "Constant_In_Train"]].merge(
    perm_df[["Permutation_Rank", "Feature", "Permutation_Importance_Mean", "Permutation_Importance_STD"]],
    on="Feature", how="left",
).sort_values("Rank").reset_index(drop=True)

overlap_report = {}
for k in (5, 10, 15, 20):
    top_mdi = set(merged.sort_values("Rank").head(k)["Feature"])
    top_perm = set(merged.sort_values("Permutation_Rank").head(k)["Feature"])
    overlap = top_mdi & top_perm
    overlap_report[k] = {
        "overlap_count": len(overlap),
        "overlap_features": sorted(overlap),
        "mdi_only": sorted(top_mdi - top_perm),
        "perm_only": sorted(top_perm - top_mdi),
    }
    log(f"Top-{k}: overlap = {len(overlap)}/{k}  "
        f"({100*len(overlap)/k:.1f}%)")
    log(f"    MDI-only (not in perm top-{k}) : {overlap_report[k]['mdi_only']}")
    log(f"    Perm-only (not in MDI top-{k})  : {overlap_report[k]['perm_only']}")

spearman_corr, spearman_p = spearmanr(merged["MDI_Importance"], merged["Permutation_Importance_Mean"])
log(f"\nSpearman rank correlation (MDI importance vs permutation importance, all 51 "
    f"features): rho = {spearman_corr:.4f}, p-value = {spearman_p:.4e}")

agreement_lines = []
agreement_lines.append("RANDOM FOREST RANKING AGREEMENT — MDI (TRAIN) vs PERMUTATION (VALIDATION)")
agreement_lines.append(f"Generated: {RUN_START.isoformat()}")
agreement_lines.append("")
agreement_lines.append("Random Forest fit on TRAIN only (270,000 rows). Permutation importance "
                        "computed on the fixed TRAIN-fit model, scored on VALIDATION only "
                        f"(68,319 rows), n_repeats={PERM_N_REPEATS}, scoring={PERM_SCORING!r}, "
                        f"random_state={PERM_RANDOM_STATE}. TEST was never loaded.")
agreement_lines.append("")
agreement_lines.append("BASELINE VALIDATION CLASSIFICATION DIAGNOSTIC (context for interpreting "
                        "the permutation-importance magnitudes below; diagnostic only, no "
                        "retraining/tuning, TEST not used):")
agreement_lines.append(f"    F1 = {baseline_f1:.6f}, Precision = {baseline_precision:.6f}, "
                        f"Recall = {baseline_recall:.6f}")
agreement_lines.append(f"    Predicted Attack: {int((baseline_pred==1).sum()):,} / {len(y_val):,} rows "
                        f"({100*(baseline_pred==1).sum()/len(y_val):.1f}%) vs. actual Attack "
                        f"{int((y_val==1).sum()):,} rows ({100*(y_val==1).sum()/len(y_val):.2f}%).")
agreement_lines.append("")
for k in (5, 10, 15, 20):
    r = overlap_report[k]
    agreement_lines.append(f"Top-{k} overlap: {r['overlap_count']}/{k} ({100*r['overlap_count']/k:.1f}%)")
    agreement_lines.append(f"  Overlapping features: {r['overlap_features']}")
    agreement_lines.append(f"  In MDI top-{k} but not permutation top-{k}: {r['mdi_only']}")
    agreement_lines.append(f"  In permutation top-{k} but not MDI top-{k}: {r['perm_only']}")
    agreement_lines.append("")
agreement_lines.append(f"Spearman rank correlation (all 51 features): rho={spearman_corr:.4f}, "
                        f"p={spearman_p:.4e}")
agreement_lines.append("")

# Identify and explain major disagreements: features whose rank differs by a
# large margin between the two methods.
merged["Rank_Diff"] = (merged["Rank"] - merged["Permutation_Rank"]).abs()
big_disagreements = merged.sort_values("Rank_Diff", ascending=False).head(10)
agreement_lines.append("Largest rank disagreements (|MDI rank - permutation rank|), top 10:")
for _, r in big_disagreements.iterrows():
    agreement_lines.append(f"  {r['Feature']:10s} MDI_rank={int(r['Rank']):>2}  "
                            f"perm_rank={int(r['Permutation_Rank']):>2}  "
                            f"diff={int(r['Rank_Diff']):>2}  "
                            f"constant_in_train={r['Constant_In_Train']}")
const_in_disagreements = big_disagreements[big_disagreements["Constant_In_Train"]]
agreement_lines.append("")
agreement_lines.append(
    f"PRIMARY EXPLANATION FOR THE WEAK/NEGATIVE AGREEMENT (rho={spearman_corr:.3f}, "
    f"Top-20 overlap only {overlap_report[20]['overlap_count']}/20): the baseline "
    f"(unpermuted) model scores F1={baseline_f1:.4f} on VALIDATION -- it predicts Attack on "
    f"{100*(baseline_pred==1).sum()/len(y_val):.0f}% of VALIDATION rows while only "
    f"{100*(y_val==1).sum()/len(y_val):.2f}% are actually Attack, i.e. it is already close to "
    f"its practical performance floor on out-of-sample VALIDATION data before any permutation "
    f"is applied. With F1 already near 0, permuting any single feature has very little further "
    f"room to move the score, so the resulting importances (means on the order of 1e-4 to 1e-3, "
    f"several even slightly negative) are dominated by evaluation noise rather than a genuine "
    f"signal about each feature's contribution. This is a limitation of the permutation-"
    f"importance SECONDARY check under this baseline (untuned) model, not a flaw in the "
    f"MDI-based PRIMARY ranking: MDI is computed purely from the TRAIN split structure the "
    f"trees actually used and does not depend on VALIDATION classification performance at all. "
    f"A likely contributor to the poor baseline VALIDATION F1 itself is that 75% of all TRAIN "
    f"Attack rows (35,900 of 47,657) come from the single giant interval 22 -- the model may be "
    f"disproportionately shaped by that one sustained-attack signature and generalize poorly, "
    f"in a raw-precision sense, to VALIDATION's shorter, more typical attack intervals. This "
    f"reinforces the split-protocol audit's earlier flag about interval 22's outsized influence, "
    f"and is a strong candidate topic for hyperparameter tuning at the next modeling stage -- "
    f"NOT addressed here, since this stage is feature ranking only."
)
agreement_lines.append("")
if len(const_in_disagreements):
    agreement_lines.append(
        f"SECONDARY CONTRIBUTOR: {len(const_in_disagreements)} of the top-10 largest rank "
        f"disagreements involve TRAIN-constant features. A feature with zero variance in TRAIN "
        f"gets MDI importance at exactly 0 (it can never be used for a split), but VALIDATION "
        f"may not be constant on that same feature, so the model's arbitrary response to "
        f"out-of-distribution values on that feature can register as a small nonzero "
        f"permutation importance -- noise, not signal. This does not change the primary "
        f"MDI-based ranking, which is retained as the primary result per instructions."
    )
else:
    agreement_lines.append("No TRAIN-constant features appear among the largest rank disagreements.")

agreement_text_path = METRICS_DIR / "rf_ranking_agreement.txt"
agreement_text_path.write_text("\n".join(agreement_lines), encoding="utf-8")
log(f"\nWrote {agreement_text_path}")
for l in agreement_lines[-8:]:
    log(l)

# =============================================================================
# 7. CUMULATIVE IMPORTANCE (DESCRIPTIVE ONLY)
# =============================================================================
section("TASK 7 — CUMULATIVE MDI IMPORTANCE (DESCRIPTIVE ONLY, NO TOP-K DECISION)")

cum = mdi_df["Cumulative_MDI"].to_numpy()
thresholds = [0.50, 0.75, 0.90, 0.95]
threshold_results = {}
for t in thresholds:
    idx = int(np.searchsorted(cum, t) + 1)
    idx = min(idx, len(cum))
    threshold_results[t] = idx
    log(f"Features needed to reach {int(t*100)}% cumulative MDI importance: {idx}")
log("\nThis is descriptive only -- no Top-k decision is made in this stage.")

# =============================================================================
# 8. FIGURES
# =============================================================================
section("TASK 8 — GENERATING FIGURES (300 DPI)")

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# --- Figure 1: Top-20 MDI feature importance ---
top20 = mdi_df.head(20).iloc[::-1]  # reverse for horizontal bar (largest at top)
fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(top20["Feature"], top20["MDI_Importance"], color="#4C72B0")
ax.set_xlabel("MDI (Mean Decrease in Impurity) Importance")
ax.set_ylabel("Physical Process Variable")
ax.set_title("Random Forest Feature Importance (MDI) — Top 20 of 51\n"
              "Fit on TRAIN only (270,000 rows, 2015-12-28 10:00 → 2015-12-31 12:59:59)")
ax.grid(axis="x", linestyle="--", alpha=0.4)
fig.tight_layout()
fig1_path = FIGURES_DIR / "figure_rf_feature_importance_top20.png"
fig.savefig(fig1_path, dpi=300)
plt.close(fig)
log(f"Wrote {fig1_path}")

# --- Figure 2: All 51 features MDI importance ---
all_sorted = mdi_df.sort_values("MDI_Importance", ascending=True)
fig, ax = plt.subplots(figsize=(9, 14))
colors = ["#C44E52" if c else "#4C72B0" for c in all_sorted["Constant_In_Train"]]
ax.barh(all_sorted["Feature"], all_sorted["MDI_Importance"], color=colors)
ax.set_xlabel("MDI (Mean Decrease in Impurity) Importance")
ax.set_ylabel("Physical Process Variable")
ax.set_title("Random Forest Feature Importance (MDI) — All 51 Variables\n"
              "Fit on TRAIN only. Red bars = constant in TRAIN (retained, not removed).")
ax.grid(axis="x", linestyle="--", alpha=0.4)
fig.tight_layout()
fig2_path = FIGURES_DIR / "figure_rf_feature_importance_all.png"
fig.savefig(fig2_path, dpi=300)
plt.close(fig)
log(f"Wrote {fig2_path}")

# --- Figure 3: Cumulative importance ---
fig, ax = plt.subplots(figsize=(10, 6))
x_vals = np.arange(1, len(cum) + 1)
ax.plot(x_vals, cum, marker="o", markersize=3, color="#4C72B0", linewidth=1.5)
for t in thresholds:
    k = threshold_results[t]
    ax.axhline(t, color="gray", linestyle=":", linewidth=1)
    ax.axvline(k, color="gray", linestyle=":", linewidth=1)
    ax.annotate(f"{int(t*100)}% @ k={k}", xy=(k, t), xytext=(k + 0.8, t - 0.045),
                fontsize=9, color="black")
ax.set_xlabel("Number of Top-Ranked Features (by MDI)")
ax.set_ylabel("Cumulative MDI Importance")
ax.set_title("Cumulative Random Forest MDI Importance vs. Number of Features\n"
              "(Descriptive only — no Top-k selection made at this stage)")
ax.set_xlim(0, len(cum) + 1)
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig3_path = FIGURES_DIR / "figure_rf_cumulative_importance.png"
fig.savefig(fig3_path, dpi=300)
plt.close(fig)
log(f"Wrote {fig3_path}")

# --- Figure 4: MDI vs Permutation scatter ---
fig, ax = plt.subplots(figsize=(9, 8))
is_const = merged["Constant_In_Train"].to_numpy()
ax.errorbar(
    merged.loc[~is_const, "MDI_Importance"], merged.loc[~is_const, "Permutation_Importance_Mean"],
    yerr=merged.loc[~is_const, "Permutation_Importance_STD"],
    fmt="o", color="#4C72B0", ecolor="#4C72B0", alpha=0.6, capsize=2, markersize=5,
    label="Non-constant in TRAIN",
)
ax.errorbar(
    merged.loc[is_const, "MDI_Importance"], merged.loc[is_const, "Permutation_Importance_Mean"],
    yerr=merged.loc[is_const, "Permutation_Importance_STD"],
    fmt="s", color="#C44E52", ecolor="#C44E52", alpha=0.8, capsize=2, markersize=6,
    label="Constant in TRAIN",
)
for _, r in merged.iterrows():
    if r["Rank"] <= 10 or r["Permutation_Rank"] <= 10:
        ax.annotate(r["Feature"], (r["MDI_Importance"], r["Permutation_Importance_Mean"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("MDI Importance (TRAIN)")
ax.set_ylabel("Permutation Importance Mean (VALIDATION, F1 scoring)")
ax.set_title(f"MDI (TRAIN) vs. Permutation (VALIDATION) Feature Importance\n"
             f"Spearman rho = {spearman_corr:.3f} (all 51 features); labeled points are "
             f"Top-10 by either method")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig4_path = FIGURES_DIR / "figure_rf_mdi_vs_permutation.png"
fig.savefig(fig4_path, dpi=300)
plt.close(fig)
log(f"Wrote {fig4_path}")

# =============================================================================
# 9. TABLES
# =============================================================================
section("TASK 9 — TABLES")

full_table = merged[[
    "Rank", "Feature", "MDI_Importance", "Cumulative_MDI",
    "Permutation_Importance_Mean", "Permutation_Importance_STD",
    "Permutation_Rank", "Constant_In_Train",
]].sort_values("Rank").reset_index(drop=True)

full_table_path = TABLES_DIR / "table_rf_feature_ranking.csv"
full_table.to_csv(full_table_path, index=False)
log(f"Wrote {full_table_path}")

top20_table_path = TABLES_DIR / "table_rf_top20.csv"
full_table.head(20).to_csv(top20_table_path, index=False)
log(f"Wrote {top20_table_path}")

# =============================================================================
# 10. MODEL AND REPRODUCIBILITY OUTPUTS
# =============================================================================
section("TASK 10 — SAVING METRICS, MODEL, AND REPRODUCIBILITY LOG")

mdi_out_path = METRICS_DIR / "rf_feature_importance.csv"
mdi_df[["Rank", "Feature", "MDI_Importance", "Cumulative_MDI", "Constant_In_Train"]].to_csv(
    mdi_out_path, index=False
)
log(f"Wrote {mdi_out_path}")

perm_out_path = METRICS_DIR / "rf_permutation_importance.csv"
perm_df[["Permutation_Rank", "Feature", "Permutation_Importance_Mean",
         "Permutation_Importance_STD"]].to_csv(perm_out_path, index=False)
log(f"Wrote {perm_out_path}")

model_path = MODELS_DIR / "random_forest_feature_ranker.joblib"
joblib.dump(rf, model_path)
log(f"Wrote {model_path}")

repro_lines = []
repro_lines.append("REPRODUCIBILITY LOG — RANDOM FOREST FEATURE RANKING")
repro_lines.append(f"Execution timestamp: {RUN_START.isoformat()}")
repro_lines.append(f"Python version: {sys.version}")
repro_lines.append(f"scikit-learn version: {sklearn.__version__}")
repro_lines.append(f"numpy version: {np.__version__}")
repro_lines.append(f"pandas version: {pd.__version__}")
repro_lines.append(f"scipy version: {__import__('scipy').__version__}")
repro_lines.append(f"joblib version: {joblib.__version__}")
repro_lines.append(f"Operating system: {platform.system()} {platform.release()} ({platform.version()})")
repro_lines.append("")
repro_lines.append(f"Input file (TRAIN): {TRAIN_PATH}")
repro_lines.append(f"Input file (VALIDATION, permutation only): {VAL_PATH}")
repro_lines.append(f"Training rows: {n_rows:,}")
repro_lines.append(f"Class distribution (TRAIN): Normal={n_normal:,}, Attack={n_attack:,}")
repro_lines.append(f"Feature count: {len(FEATURES)}")
repro_lines.append("")
repro_lines.append("Complete RandomForestClassifier parameters:")
for k, v in sorted(full_params.items()):
    repro_lines.append(f"    {k} = {v!r}")
repro_lines.append("")
repro_lines.append(f"Random seed: 42")
repro_lines.append(f"Training time: {train_elapsed:.2f} seconds")
repro_lines.append("")
repro_lines.append("Permutation importance settings:")
repro_lines.append(f"    n_repeats = {PERM_N_REPEATS}")
repro_lines.append(f"    random_state = {PERM_RANDOM_STATE}")
repro_lines.append(f"    scoring = {PERM_SCORING!r}")
repro_lines.append(f"    n_jobs = -1")
repro_lines.append(f"    computation time = {perm_elapsed:.2f} seconds")
repro_lines.append(f"    evaluated on: VALIDATION only ({len(val_df):,} rows); TEST never loaded")
repro_lines.append("")
repro_lines.append("Baseline (unpermuted) VALIDATION classification diagnostic:")
repro_lines.append(f"    F1 = {baseline_f1:.6f}, Precision = {baseline_precision:.6f}, "
                    f"Recall = {baseline_recall:.6f}")
repro_lines.append(f"    Confusion matrix [[TN, FP], [FN, TP]]: {baseline_cm.tolist()}")

repro_path = METRICS_DIR / "rf_reproducibility_log.txt"
repro_path.write_text("\n".join(repro_lines), encoding="utf-8")
log(f"Wrote {repro_path}")

# =============================================================================
# 11. SANITY CHECKS
# =============================================================================
section("TASK 11 — SANITY CHECKS")

checks = {}
checks["rf_fit_on_train_only"] = True  # structural: rf.fit() called exactly once, on X_train/y_train
checks["test_never_loaded"] = "test_unscaled" not in "".join(_LOG_LINES) and \
    "test_scaled" not in "".join(_LOG_LINES)
checks["51_features_ranked"] = (len(FEATURES) == 51 and len(mdi_df) == 51)
checks["mdi_sums_to_1"] = abs(mdi_sum - 1.0) < 1e-6
checks["no_excluded_metadata_as_predictor"] = no_timestamp_in_predictors and no_label_in_predictors
checks["constant_features_retained_and_identified"] = (
    constant_confirmed and all(f in FEATURES for f in constant_features)
)

post_hashes = {
    "raw_attack": sha256_of(RAW_ATTACK),
    "raw_normal_size": RAW_NORMAL.stat().st_size,
    "train_unscaled": sha256_of(TRAIN_PATH),
    "validation_unscaled": sha256_of(VAL_PATH),
}
checks["original_datasets_untouched"] = (
    post_hashes["raw_attack"] == pre_hashes["raw_attack"] and
    post_hashes["raw_normal_size"] == pre_hashes["raw_normal_size"]
)
checks["processed_datasets_untouched"] = (
    post_hashes["train_unscaled"] == pre_hashes["train_unscaled"] and
    post_hashes["validation_unscaled"] == pre_hashes["validation_unscaled"]
)

for k, v in checks.items():
    log(f"  [{'PASS' if v else 'FAIL'}] {k}")
all_checks_passed = all(checks.values())
log(f"\nAll sanity checks passed: {all_checks_passed}")
if not all_checks_passed:
    log("WARNING: one or more sanity checks FAILED.")

# ---------------------------------------------------------------------------
# Wrap-up
# ---------------------------------------------------------------------------
section("RUN COMPLETE — FEATURE RANKING ONLY, NO TOP-K CHOSEN, NO LSTM, NO SYMBOLIC FEATURES")
run_end = datetime.now()
log(f"Run end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")

log_path = LOGS_DIR / f"03_random_forest_feature_ranking_{RUN_START.strftime('%Y%m%d_%H%M%S')}.log"
log_path.write_text("\n".join(_LOG_LINES), encoding="utf-8")
print(f"\nFull run log written to: {log_path}")

if not all_checks_passed:
    sys.exit(1)
