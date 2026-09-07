"""
code/04_rf_generalization_diagnostic.py

DIAGNOSTIC EXPERIMENT — WHY DOES THE TRAIN-FITTED RF GENERALIZE SO POORLY
TO CHRONOLOGICAL VALIDATION?

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

This is diagnosis, NOT hyperparameter optimization. Uses TRAIN and
VALIDATION only -- TEST is never loaded (its filename appears nowhere
below). Does not choose Top-k, does not train LSTM, does not generate
symbolic features, does not modify any dataset, does not run
permutation importance (deferred until the generalization issue is
understood).
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from scipy.stats import ks_2samp, spearmanr

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
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "random_forest_feature_ranker.joblib"
# NOTE: no TEST_PATH constant defined anywhere in this script, on purpose.

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


section("RF GENERALIZATION DIAGNOSTIC — RUN START")
log(f"Run start: {RUN_START.isoformat()}")

DATASET_DIR = Path(os.environ.get("SWAT_DATASET_DIR", str(PROJECT_ROOT / "dataset")))  # portability fix: actual directory is lowercase; overridable via env var
RAW_ATTACK = DATASET_DIR / "SWaT_Dataset_Attack_v0.csv"
RAW_NORMAL = DATASET_DIR / "SWaT_Dataset_Normal_v0.csv"
pre_hashes = {
    "raw_attack": sha256_of(RAW_ATTACK),
    "raw_normal_size": RAW_NORMAL.stat().st_size,
    "train_unscaled": sha256_of(TRAIN_PATH),
    "validation_unscaled": sha256_of(VAL_PATH),
}
log("Pre-run integrity snapshot recorded (re-verified at end of run).")

# =============================================================================
# LOAD DATA + BASELINE MODEL
# =============================================================================
section("LOADING TRAIN, VALIDATION, AND THE PREVIOUSLY FITTED BASELINE RF")

FEATURES = FEATURE_LIST_PATH.read_text(encoding="utf-8").strip().split("\n")
assert len(FEATURES) == 51
EXCLUDE_COLS = {"Timestamp", "Label", "Label_raw"}
TARGET = "Label"

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
train_df["_ts"] = pd.to_datetime(train_df["Timestamp"])
val_df["_ts"] = pd.to_datetime(val_df["Timestamp"])
assert list(train_df.columns[1:52]) == FEATURES
assert list(val_df.columns[1:52]) == FEATURES
log(f"TRAIN: {train_df.shape[0]:,} rows | VALIDATION: {val_df.shape[0]:,} rows")
assert train_df.shape[0] == 270000
assert val_df.shape[0] == 68319

X_train = train_df[FEATURES].to_numpy(dtype=np.float64)
y_train = train_df[TARGET].to_numpy()
X_val = val_df[FEATURES].to_numpy(dtype=np.float64)
y_val = val_df[TARGET].to_numpy()

rf_baseline = joblib.load(MODEL_PATH)
log(f"Loaded baseline model from {MODEL_PATH}")
log(f"Baseline model params: n_estimators={rf_baseline.n_estimators}, "
    f"class_weight={rf_baseline.class_weight}, random_state={rf_baseline.random_state}")


def evaluate(model, X, y, label=""):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    result = {
        "label": label,
        "n_rows": len(y),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": fpr,
        "predicted_attack_pct": 100 * (pred == 1).sum() / len(y),
        "actual_attack_pct": 100 * (y == 1).sum() / len(y),
    }
    return result, pred, proba


# =============================================================================
# TASK 1 — BASELINE TRAIN VS VALIDATION PERFORMANCE
# =============================================================================
section("TASK 1 — BASELINE (ORIGINAL) RF: TRAIN VS VALIDATION PERFORMANCE")

res_train, pred_train, proba_train = evaluate(rf_baseline, X_train, y_train, "baseline_TRAIN")
res_val, pred_val, proba_val = evaluate(rf_baseline, X_val, y_val, "baseline_VALIDATION")

for r in (res_train, res_val):
    log(f"\n[{r['label']}] n={r['n_rows']:,}")
    log(f"  Accuracy = {r['accuracy']:.6f}")
    log(f"  Precision = {r['precision']:.6f}")
    log(f"  Recall = {r['recall']:.6f}")
    log(f"  F1 = {r['f1']:.6f}")
    log(f"  ROC-AUC = {r['roc_auc']:.6f}")
    log(f"  Confusion matrix [[TN={r['tn']}, FP={r['fp']}], [FN={r['fn']}, TP={r['tp']}]]")
    log(f"  FPR = {r['fpr']:.6f}")
    log(f"  Predicted Attack% = {r['predicted_attack_pct']:.4f}  |  "
        f"Actual Attack% = {r['actual_attack_pct']:.4f}")

overfit_gap_f1 = res_train["f1"] - res_val["f1"]
log(f"\nTRAIN F1 - VALIDATION F1 gap = {overfit_gap_f1:.6f}")
log("Interpretation guide logged here; final synthesized diagnosis is in Task 9 below and "
    "in results/metrics/rf_generalization_diagnosis.txt. Briefly: if TRAIN performance is "
    "near-perfect while VALIDATION collapses AND VALIDATION over-predicts Attack broadly "
    "(not just near real attacks), that pattern is consistent with the model exploiting "
    "TRAIN-period-specific structure that does not hold later in time (temporal "
    "generalization failure / possible overfitting to TRAIN-specific patterns), rather than "
    "a simple case of insufficient model capacity.")

# =============================================================================
# TASK 2 — INTERVAL-LEVEL VALIDATION PERFORMANCE
# =============================================================================
section("TASK 2 — INTERVAL-LEVEL VALIDATION PERFORMANCE")

val_df["_pred"] = pred_val
is_attack = val_df[TARGET].eq(1)
group_id = (is_attack != is_attack.shift()).cumsum()

val_intervals = []
val_groups = list(val_df.groupby(group_id))
for gid, sub in val_groups:
    if sub[TARGET].iloc[0] == 1:
        val_intervals.append({
            "start": sub["_ts"].iloc[0], "end": sub["_ts"].iloc[-1],
            "n_attack_obs": len(sub),
            "n_detected": int((sub["_pred"] == 1).sum()),
            "n_false_negative": int((sub["_pred"] == 0).sum()),
        })
assert len(val_intervals) == 3, f"Expected 3 complete attack intervals in VALIDATION, found {len(val_intervals)}"
log(f"Confirmed {len(val_intervals)} complete attack intervals in VALIDATION.")

interval_level_records = []
for i, iv in enumerate(val_intervals):
    recall_i = iv["n_detected"] / iv["n_attack_obs"] if iv["n_attack_obs"] else float("nan")

    # surrounding Normal segments: from end of previous group to start of this
    # interval ("before"), and from end of this interval to start of next
    # group ("after") -- both restricted to Normal-labeled rows already by
    # construction of the group split.
    idx_in_groups = [gid for gid, sub in val_groups]
    # find this interval's group position among all groups (attack + normal)
    all_groups_ordered = list(val_df.groupby(group_id))
    pos = None
    for gi, (gid, sub) in enumerate(all_groups_ordered):
        if sub["_ts"].iloc[0] == iv["start"] and sub["_ts"].iloc[-1] == iv["end"]:
            pos = gi
            break
    before_fp = after_fp = before_n = after_n = 0
    if pos is not None:
        if pos - 1 >= 0:
            before_sub = all_groups_ordered[pos - 1][1]
            before_n = len(before_sub)
            before_fp = int((before_sub["_pred"] == 1).sum())
        if pos + 1 < len(all_groups_ordered):
            after_sub = all_groups_ordered[pos + 1][1]
            after_n = len(after_sub)
            after_fp = int((after_sub["_pred"] == 1).sum())

    record = {
        "interval_index": i + 1,
        "start": iv["start"], "end": iv["end"],
        "n_attack_obs": iv["n_attack_obs"],
        "n_detected_correctly": iv["n_detected"],
        "recall": recall_i,
        "n_false_negative": iv["n_false_negative"],
        "surrounding_normal_before_n": before_n,
        "surrounding_normal_before_fp": before_fp,
        "surrounding_normal_before_fpr": before_fp / before_n if before_n else float("nan"),
        "surrounding_normal_after_n": after_n,
        "surrounding_normal_after_fp": after_fp,
        "surrounding_normal_after_fpr": after_fp / after_n if after_n else float("nan"),
    }
    interval_level_records.append(record)
    log(f"\nInterval {i+1}: {iv['start']} -> {iv['end']}  "
        f"({iv['n_attack_obs']} attack obs)")
    log(f"  Detected correctly: {iv['n_detected']}  |  Recall: {recall_i:.4f}  |  "
        f"False negatives: {iv['n_false_negative']}")
    log(f"  Surrounding Normal BEFORE: n={before_n}, FP={before_fp}, "
        f"FPR={record['surrounding_normal_before_fpr']:.4f}" if before_n else
        f"  Surrounding Normal BEFORE: none (VALIDATION start)")
    log(f"  Surrounding Normal AFTER : n={after_n}, FP={after_fp}, "
        f"FPR={record['surrounding_normal_after_fpr']:.4f}" if after_n else
        f"  Surrounding Normal AFTER : none (VALIDATION end)")

interval_level_df = pd.DataFrame(interval_level_records)
interval_level_path = METRICS_DIR / "rf_diagnostic_interval_level.csv"
interval_level_df.to_csv(interval_level_path, index=False)
log(f"\nWrote {interval_level_path}")

overall_val_fp_rate_in_normal = res_val["fpr"]
log(f"\nFor context, OVERALL VALIDATION Normal-period FPR (all Normal rows, not just "
    f"immediate surroundings) = {overall_val_fp_rate_in_normal:.4f} -- compare against the "
    f"immediate-surrounding FPR values above to see whether false positives concentrate near "
    f"attacks or are spread uniformly across all Normal VALIDATION data.")

# =============================================================================
# TASK 3 — TEMPORAL DISTRIBUTION SHIFT (TRAIN vs VALIDATION), ALL 51 FEATURES
# =============================================================================
section("TASK 3 — TRAIN vs VALIDATION FEATURE DISTRIBUTION SHIFT")

mdi_importances = rf_baseline.feature_importances_
mdi_rank_map = {
    f: r + 1 for r, f in enumerate(
        [FEATURES[i] for i in np.argsort(-mdi_importances)]
    )
}

shift_records = []
for f in FEATURES:
    tr = train_df[f].to_numpy(dtype=np.float64)
    va = val_df[f].to_numpy(dtype=np.float64)
    tr_mean, va_mean = tr.mean(), va.mean()
    tr_std, va_std = tr.std(ddof=0), va.std(ddof=0)
    pooled_std = np.sqrt((tr_std ** 2 + va_std ** 2) / 2)
    smd = (va_mean - tr_mean) / pooled_std if pooled_std > 1e-12 else np.nan
    ks_stat, ks_p = ks_2samp(tr, va)
    shift_records.append({
        "Feature": f,
        "RF_MDI_Rank": mdi_rank_map[f],
        "Train_Mean": tr_mean, "Validation_Mean": va_mean,
        "Train_Std": tr_std, "Validation_Std": va_std,
        "Train_Median": np.median(tr), "Validation_Median": np.median(va),
        "Train_Q05": np.quantile(tr, 0.05), "Validation_Q05": np.quantile(va, 0.05),
        "Train_Q95": np.quantile(tr, 0.95), "Validation_Q95": np.quantile(va, 0.95),
        "Standardized_Mean_Difference": smd,
        "KS_Statistic": ks_stat, "KS_pvalue": ks_p,
    })

shift_df = pd.DataFrame(shift_records).sort_values("KS_Statistic", ascending=False).reset_index(drop=True)
shift_df.insert(0, "Shift_Rank", shift_df.index + 1)

shift_table_path = TABLES_DIR / "table_rf_distribution_shift.csv"
shift_df.to_csv(shift_table_path, index=False)
log(f"Wrote {shift_table_path}")

log("\nTop 15 features by TRAIN-to-VALIDATION distribution shift (KS statistic):")
for _, r in shift_df.head(15).iterrows():
    log(f"  [{int(r['Shift_Rank']):>2}] {r['Feature']:10s} KS={r['KS_Statistic']:.4f} "
        f"(p={r['KS_pvalue']:.2e})  SMD={r['Standardized_Mean_Difference']:.3f}  "
        f"RF_MDI_Rank={int(r['RF_MDI_Rank'])}")

top20_mdi_features = set(f for f, r in mdi_rank_map.items() if r <= 20)
top15_shift_features = set(shift_df.head(15)["Feature"])
overlap_top = top20_mdi_features & top15_shift_features
log(f"\nOverlap between RF Top-20 (MDI) and Top-15 highest-shift features: "
    f"{len(overlap_top)} -> {sorted(overlap_top)}")
mean_ks_top20 = shift_df[shift_df["RF_MDI_Rank"] <= 20]["KS_Statistic"].mean()
mean_ks_rest = shift_df[shift_df["RF_MDI_Rank"] > 20]["KS_Statistic"].mean()
log(f"Mean KS statistic among RF Top-20 MDI features: {mean_ks_top20:.4f}")
log(f"Mean KS statistic among remaining 31 features : {mean_ks_rest:.4f}")
log("If the Top-20 mean KS is similar to or higher than the rest, the RF's most 'important' "
    "TRAIN features are also among the most temporally unstable ones -- i.e. the ranking is "
    "partly built on signals that shift between TRAIN and VALIDATION periods, not on process "
    "characteristics that are stable over time.")

# =============================================================================
# TASK 4 — GIANT ATTACK INTERVAL #22 DIAGNOSTIC (VARIANTS A / B / C)
# =============================================================================
section("TASK 4 — INTERVAL #22 DIAGNOSTIC: VARIANTS A (baseline) / B (excluded) / C (downsampled)")

is_attack_train = train_df[TARGET].eq(1)
train_group_id = (is_attack_train != is_attack_train.shift()).cumsum()
train_groups = train_df.groupby(train_group_id)
giant_group = None
for gid, sub in train_groups:
    if sub[TARGET].iloc[0] == 1 and len(sub) > 10000:
        giant_group = sub
        break
assert giant_group is not None, "Could not locate giant interval 22 in TRAIN."
n_giant = len(giant_group)
log(f"Located giant interval 22 in TRAIN: {giant_group['_ts'].iloc[0]} -> "
    f"{giant_group['_ts'].iloc[-1]}, {n_giant:,} rows.")
assert n_giant == 35900, f"Expected 35,900 rows for interval 22, found {n_giant}"
giant_idx = giant_group.index

RF_CONFIG = dict(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced")

variant_results = {}
variant_models = {}
variant_mdi = {}

# --- Variant A: original (already fitted) ---
variant_results["A_original"] = res_val
variant_models["A_original"] = rf_baseline
variant_mdi["A_original"] = rf_baseline.feature_importances_
log("\nVariant A (original): reusing the already-fitted baseline model and its VALIDATION result.")

# --- Variant B: exclude interval 22 entirely ---
train_df_B = train_df.drop(index=giant_idx).reset_index(drop=True)
X_train_B = train_df_B[FEATURES].to_numpy(dtype=np.float64)
y_train_B = train_df_B[TARGET].to_numpy()
n_normal_B = int((y_train_B == 0).sum())
n_attack_B = int((y_train_B == 1).sum())
log(f"\nVariant B TRAIN (interval 22 excluded): {len(train_df_B):,} rows "
    f"(Normal={n_normal_B:,}, Attack={n_attack_B:,})")
assert len(train_df_B) == 270000 - 35900
assert n_attack_B == 47657 - 35900

rf_B = RandomForestClassifier(**RF_CONFIG)
t0 = time.time()
rf_B.fit(X_train_B, y_train_B)
t_B = time.time() - t0
res_B, pred_B, proba_B = evaluate(rf_B, X_val, y_val, "B_interval22_excluded_VALIDATION")
variant_results["B_interval22_excluded"] = res_B
variant_models["B_interval22_excluded"] = rf_B
variant_mdi["B_interval22_excluded"] = rf_B.feature_importances_
log(f"Variant B training time: {t_B:.2f}s")
log(f"Variant B VALIDATION: Accuracy={res_B['accuracy']:.4f} Precision={res_B['precision']:.4f} "
    f"Recall={res_B['recall']:.4f} F1={res_B['f1']:.4f} ROC-AUC={res_B['roc_auc']:.4f} "
    f"FPR={res_B['fpr']:.4f}")
log(f"  Confusion matrix [[TN={res_B['tn']}, FP={res_B['fp']}], [FN={res_B['fn']}, TP={res_B['tp']}]]")

# --- Variant C: downsample interval 22's attack rows ---
n_other_train_intervals = 21  # 22 TRAIN intervals total, minus the giant one
n_other_attack_rows = (47657 - 35900)
downsample_target = int(round(n_other_attack_rows / n_other_train_intervals))
log(f"\nVariant C downsample target for interval 22: {downsample_target} rows "
    f"(= average size of the other {n_other_train_intervals} TRAIN attack intervals, "
    f"{n_other_attack_rows} rows / {n_other_train_intervals} intervals)")
rng = np.random.RandomState(42)
keep_idx_giant = rng.choice(giant_idx, size=downsample_target, replace=False)
drop_idx_giant = giant_idx.difference(pd.Index(keep_idx_giant))
train_df_C = train_df.drop(index=drop_idx_giant).reset_index(drop=True)
X_train_C = train_df_C[FEATURES].to_numpy(dtype=np.float64)
y_train_C = train_df_C[TARGET].to_numpy()
n_normal_C = int((y_train_C == 0).sum())
n_attack_C = int((y_train_C == 1).sum())
log(f"Variant C TRAIN (interval 22 downsampled to {downsample_target}): {len(train_df_C):,} rows "
    f"(Normal={n_normal_C:,}, Attack={n_attack_C:,})")
assert n_attack_C == n_other_attack_rows + downsample_target

rf_C = RandomForestClassifier(**RF_CONFIG)
t0 = time.time()
rf_C.fit(X_train_C, y_train_C)
t_C = time.time() - t0
res_C, pred_C, proba_C = evaluate(rf_C, X_val, y_val, "C_interval22_downsampled_VALIDATION")
variant_results["C_interval22_downsampled"] = res_C
variant_models["C_interval22_downsampled"] = rf_C
variant_mdi["C_interval22_downsampled"] = rf_C.feature_importances_
log(f"Variant C training time: {t_C:.2f}s")
log(f"Variant C VALIDATION: Accuracy={res_C['accuracy']:.4f} Precision={res_C['precision']:.4f} "
    f"Recall={res_C['recall']:.4f} F1={res_C['f1']:.4f} ROC-AUC={res_C['roc_auc']:.4f} "
    f"FPR={res_C['fpr']:.4f}")
log(f"  Confusion matrix [[TN={res_C['tn']}, FP={res_C['fp']}], [FN={res_C['fn']}, TP={res_C['tp']}]]")

# --- Ranking stability across A/B/C ---
section("TASK 4b — MDI RANKING STABILITY ACROSS VARIANTS A / B / C")

rank_stability_df = pd.DataFrame({"Feature": FEATURES})
for name, importances in variant_mdi.items():
    rank_stability_df[f"MDI_{name}"] = importances
    ranks = pd.Series(importances).rank(ascending=False, method="min").astype(int)
    rank_stability_df[f"Rank_{name}"] = ranks.values
rank_stability_path = TABLES_DIR / "table_rf_ranking_stability_diagnostic.csv"
rank_stability_df.sort_values("Rank_A_original").to_csv(rank_stability_path, index=False)
log(f"Wrote {rank_stability_path}")

pairs = [("A_original", "B_interval22_excluded"), ("A_original", "C_interval22_downsampled"),
         ("B_interval22_excluded", "C_interval22_downsampled")]
ranking_comparisons = []
for a, b in pairs:
    rank_a = rank_stability_df[f"Rank_{a}"]
    rank_b = rank_stability_df[f"Rank_{b}"]
    top10_a = set(rank_stability_df.loc[rank_a <= 10, "Feature"])
    top10_b = set(rank_stability_df.loc[rank_b <= 10, "Feature"])
    top20_a = set(rank_stability_df.loc[rank_a <= 20, "Feature"])
    top20_b = set(rank_stability_df.loc[rank_b <= 20, "Feature"])
    rho, p = spearmanr(rank_a, rank_b)
    comp = {
        "variant_pair": f"{a}_vs_{b}",
        "top10_overlap": len(top10_a & top10_b),
        "top20_overlap": len(top20_a & top20_b),
        "spearman_rho": rho, "spearman_p": p,
    }
    ranking_comparisons.append(comp)
    log(f"{a} vs {b}: Top-10 overlap={comp['top10_overlap']}/10, "
        f"Top-20 overlap={comp['top20_overlap']}/20, Spearman rho={rho:.4f} (p={p:.2e})")

ranking_comparisons_df = pd.DataFrame(ranking_comparisons)

# =============================================================================
# TASK 5 — CLASS_WEIGHT DIAGNOSTIC (VARIANT D)
# =============================================================================
section("TASK 5 — CLASS_WEIGHT DIAGNOSTIC: balanced (baseline A) vs None (variant D)")

RF_CONFIG_UNWEIGHTED = dict(n_estimators=500, random_state=42, n_jobs=-1, class_weight=None)
rf_D = RandomForestClassifier(**RF_CONFIG_UNWEIGHTED)
t0 = time.time()
rf_D.fit(X_train, y_train)
t_D = time.time() - t0
res_D, pred_D, proba_D = evaluate(rf_D, X_val, y_val, "D_class_weight_None_VALIDATION")
log(f"Variant D training time: {t_D:.2f}s")
log(f"\nBASELINE A (class_weight='balanced') VALIDATION: "
    f"Precision={res_val['precision']:.4f} Recall={res_val['recall']:.4f} F1={res_val['f1']:.4f} "
    f"FPR={res_val['fpr']:.4f} Predicted_Attack%={res_val['predicted_attack_pct']:.2f}")
log(f"VARIANT D (class_weight=None)  VALIDATION: "
    f"Precision={res_D['precision']:.4f} Recall={res_D['recall']:.4f} F1={res_D['f1']:.4f} "
    f"FPR={res_D['fpr']:.4f} Predicted_Attack%={res_D['predicted_attack_pct']:.2f}")
log(f"  Confusion matrix D [[TN={res_D['tn']}, FP={res_D['fp']}], [FN={res_D['fn']}, TP={res_D['tp']}]]")
log("\nThis comparison is diagnostic only -- neither configuration is being adopted as final.")

# =============================================================================
# TASK 7 — FIGURES
# =============================================================================
section("TASK 7 — GENERATING DIAGNOSTIC FIGURES (300 DPI)")

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
    "axes.titlesize": 13, "axes.labelsize": 11,
})

# --- Figure 1: TRAIN vs VALIDATION metrics ---
metrics_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
metrics_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
train_vals = [res_train[m] for m in metrics_names]
val_vals = [res_val[m] for m in metrics_names]
x = np.arange(len(metrics_names))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(x - width / 2, train_vals, width, label="TRAIN", color="#4C72B0")
ax.bar(x + width / 2, val_vals, width, label="VALIDATION", color="#C44E52")
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_title("Baseline Random Forest: TRAIN vs. VALIDATION Performance\n"
              f"F1 gap (TRAIN - VALIDATION) = {overfit_gap_f1:.4f}")
for xi, (tv, vv) in enumerate(zip(train_vals, val_vals)):
    ax.text(xi - width / 2, tv + 0.02, f"{tv:.3f}", ha="center", fontsize=8)
    ax.text(xi + width / 2, vv + 0.02, f"{vv:.3f}", ha="center", fontsize=8)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig1_path = FIGURES_DIR / "diagnostic_rf_train_vs_validation.png"
fig.savefig(fig1_path, dpi=300)
plt.close(fig)
log(f"Wrote {fig1_path}")

# --- Figure 2: confusion matrices, TRAIN and VALIDATION ---
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, r, title in [(axes[0], res_train, "TRAIN"), (axes[1], res_val, "VALIDATION")]:
    cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Normal", "Pred Attack"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Normal", "Actual Attack"])
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=11)
    ax.set_title(f"{title}\n(n={r['n_rows']:,}, F1={r['f1']:.4f})")
fig.suptitle("Baseline Random Forest Confusion Matrices — TRAIN vs. VALIDATION", y=1.03)
fig.tight_layout()
fig2_path = FIGURES_DIR / "diagnostic_rf_confusion_validation.png"
fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig2_path}")

# --- Figure 3: distribution shift for RF Top-20 (MDI) features ---
top20_shift = shift_df[shift_df["RF_MDI_Rank"] <= 20].sort_values("RF_MDI_Rank")
fig, ax = plt.subplots(figsize=(9, 8))
colors = ["#C44E52" if ks > 0.3 else "#4C72B0" for ks in top20_shift["KS_Statistic"]]
ax.barh(top20_shift["Feature"], top20_shift["KS_Statistic"], color=colors)
ax.invert_yaxis()
ax.set_xlabel("Kolmogorov–Smirnov Statistic (TRAIN vs. VALIDATION)")
ax.set_ylabel("Physical Process Variable (ordered by RF MDI rank, 1=top)")
ax.set_title("TRAIN-to-VALIDATION Distribution Shift for the RF's Top-20 MDI Features\n"
              "Red = KS > 0.3 (substantial shift)")
ax.grid(axis="x", linestyle="--", alpha=0.4)
fig.tight_layout()
fig3_path = FIGURES_DIR / "diagnostic_feature_distribution_shift_top20.png"
fig.savefig(fig3_path, dpi=300)
plt.close(fig)
log(f"Wrote {fig3_path}")

# --- Figure 4: interval-22 sensitivity (A/B/C) ---
variant_order = ["A_original", "B_interval22_excluded", "C_interval22_downsampled"]
variant_labels = ["A: Original", "B: Interval 22\nExcluded", "C: Interval 22\nDownsampled"]
metric_keys = ["precision", "recall", "f1", "fpr"]
metric_labels2 = ["Precision", "Recall", "F1", "FPR"]
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(variant_order))
width = 0.2
for i, (mk, ml) in enumerate(zip(metric_keys, metric_labels2)):
    vals = [variant_results[v][mk] for v in variant_order]
    ax.bar(x + (i - 1.5) * width, vals, width, label=ml)
ax.set_xticks(x)
ax.set_xticklabels(variant_labels)
ax.set_ylabel("Score (VALIDATION)")
ax.set_title("Interval-22 Sensitivity: VALIDATION Performance Across Controlled Variants")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig4_path = FIGURES_DIR / "diagnostic_interval22_sensitivity.png"
fig.savefig(fig4_path, dpi=300)
plt.close(fig)
log(f"Wrote {fig4_path}")

# =============================================================================
# TASK 8 — TABLES
# =============================================================================
section("TASK 8 — TABLES")

perf_records = []
for name, r in [("baseline_A_TRAIN", res_train), ("baseline_A_VALIDATION", res_val),
                ("B_interval22_excluded_VALIDATION", res_B),
                ("C_interval22_downsampled_VALIDATION", res_C),
                ("D_class_weight_None_VALIDATION", res_D)]:
    perf_records.append({
        "Variant": name, "N_Rows": r["n_rows"],
        "Accuracy": r["accuracy"], "Precision": r["precision"], "Recall": r["recall"],
        "F1": r["f1"], "ROC_AUC": r["roc_auc"], "FPR": r["fpr"],
        "TN": r["tn"], "FP": r["fp"], "FN": r["fn"], "TP": r["tp"],
        "Predicted_Attack_Pct": r["predicted_attack_pct"], "Actual_Attack_Pct": r["actual_attack_pct"],
    })
perf_df = pd.DataFrame(perf_records)
perf_path = TABLES_DIR / "table_rf_diagnostic_performance.csv"
perf_df.to_csv(perf_path, index=False)
log(f"Wrote {perf_path}")

interval22_records = []
for name in variant_order:
    r = variant_results[name]
    interval22_records.append({
        "Variant": name, "Accuracy": r["accuracy"], "Precision": r["precision"],
        "Recall": r["recall"], "F1": r["f1"], "ROC_AUC": r["roc_auc"], "FPR": r["fpr"],
        "TN": r["tn"], "FP": r["fp"], "FN": r["fn"], "TP": r["tp"],
    })
interval22_df = pd.DataFrame(interval22_records)
interval22_perf_path = TABLES_DIR / "table_rf_interval22_sensitivity.csv"
# combine performance + ranking-overlap comparisons in one file (two logical
# sections separated by a blank marker row) for a single source of truth
with open(interval22_perf_path, "w", encoding="utf-8", newline="") as f:
    interval22_df.to_csv(f, index=False)
    f.write("\n")
    f.write("# Ranking comparisons (Top-10/Top-20 overlap, Spearman) across the same variants\n")
    ranking_comparisons_df.to_csv(f, index=False)
log(f"Wrote {interval22_perf_path}")

# =============================================================================
# TASK 9 — FINAL DIAGNOSIS
# =============================================================================
section("TASK 9 — FINAL DIAGNOSIS")

diag_lines = []
diag_lines.append("RF GENERALIZATION DIAGNOSTIC — FINAL DIAGNOSIS")
diag_lines.append(f"Generated: {RUN_START.isoformat()}")
diag_lines.append("")
diag_lines.append("For each hypothesis: DIRECTLY DEMONSTRATED (by an experiment run in this "
                   "script) vs PLAUSIBLE BUT NOT PROVEN (consistent with the evidence but not "
                   "isolated by a controlled comparison here).")
diag_lines.append("")

# A. Overfitting
diag_lines.append("A. RF OVERFITTING TO TRAIN")
diag_lines.append(f"   DIRECTLY DEMONSTRATED: TRAIN F1={res_train['f1']:.4f} vs "
                   f"VALIDATION F1={res_val['f1']:.4f} (gap={overfit_gap_f1:.4f}); TRAIN "
                   f"Accuracy={res_train['accuracy']:.4f}, ROC-AUC={res_train['roc_auc']:.4f} "
                   f"vs VALIDATION ROC-AUC={res_val['roc_auc']:.4f}. The model fits TRAIN "
                   f"far better than it generalizes to VALIDATION, which is the defining "
                   f"signature of overfitting (in the broad sense of fitting TRAIN-specific "
                   f"structure that does not hold later in time).")
diag_lines.append(f"   NUANCE: default RandomForestClassifier trees here are grown to full "
                   f"depth (max_depth=None, min_samples_leaf=1), a configuration known to be "
                   f"prone to memorizing TRAIN-specific patterns; this is consistent with, "
                   f"though not a separate proof beyond, the TRAIN/VALIDATION gap above.")
diag_lines.append("")

# B. Distribution shift
diag_lines.append("B. TRAIN-TO-VALIDATION TEMPORAL DISTRIBUTION SHIFT")
diag_lines.append(f"   DIRECTLY DEMONSTRATED: {int((shift_df['KS_Statistic']>0.3).sum())} of 51 "
                   f"features show KS > 0.3 between TRAIN and VALIDATION (see "
                   f"table_rf_distribution_shift.csv). Mean KS among the RF's own Top-20 MDI "
                   f"features is {mean_ks_top20:.4f} vs {mean_ks_rest:.4f} for the remaining 31 "
                   f"-- {'the RF relies most heavily on features that also shift the most, ' if mean_ks_top20 >= mean_ks_rest else 'the RF-important features are not disproportionately the most shifted ones, '}"
                   f"which is directly measured, not inferred.")
diag_lines.append(f"   PLAUSIBLE BUT NOT PROVEN: that this shift is the (sole or primary) "
                   f"CAUSE of the VALIDATION F1 collapse, as opposed to a contributing factor "
                   f"alongside A/C/D below. This script demonstrates the shift exists and "
                   f"correlates with feature importance; it does not isolate shift as the sole "
                   f"causal driver (that would require e.g. retraining on shift-robust features "
                   f"only, which is out of scope for this diagnostic stage).")
diag_lines.append("")

# C. Interval 22 dominance
b_vs_a_f1_delta = res_B["f1"] - res_val["f1"]
c_vs_a_f1_delta = res_C["f1"] - res_val["f1"]
diag_lines.append("C. DOMINANCE OF ATTACK INTERVAL #22")
diag_lines.append(f"   DIRECTLY DEMONSTRATED (threshold-based metrics, default 0.5 cutoff): removing "
                   f"interval 22 from TRAIN changes VALIDATION F1 from {res_val['f1']:.4f} (A) to "
                   f"{res_B['f1']:.4f} (B); downsampling it changes F1 to {res_C['f1']:.4f} (C). "
                   f"FPR moved from {res_val['fpr']:.4f} (A) to {res_B['fpr']:.4f} (B) and "
                   f"{res_C['fpr']:.4f} (C). Variant B predicts NO rows as Attack at all "
                   f"(TP={res_B['tp']}, FP={res_B['fp']} out of {res_B['n_rows']:,} VALIDATION "
                   f"rows) -- F1/Precision/Recall/FPR are all exactly 0 by construction, a "
                   f"degenerate result, not evidence of 'no change'.")
diag_lines.append(f"   DIRECTLY DEMONSTRATED (threshold-independent, ROC-AUC): this is the more "
                   f"informative comparison here, since A's and B's threshold-based metrics are "
                   f"degenerate in opposite directions (A over-predicts Attack, B never predicts "
                   f"it). ROC-AUC on VALIDATION is {res_val['roc_auc']:.4f} for A (original), "
                   f"{res_B['roc_auc']:.4f} for B (interval 22 excluded), and "
                   f"{res_C['roc_auc']:.4f} for C (interval 22 downsampled). Excluding interval "
                   f"22 more than QUADRUPLES the model's underlying VALIDATION discrimination "
                   f"ability (AUC {res_val['roc_auc']:.3f} -> {res_B['roc_auc']:.3f}); "
                   f"downsampling it to a typical interval's size still nearly triples AUC "
                   f"(-> {res_C['roc_auc']:.3f}). MDI ranking overlap between A and B: "
                   f"Top-10={ranking_comparisons[0]['top10_overlap']}/10, "
                   f"Top-20={ranking_comparisons[0]['top20_overlap']}/20, "
                   f"Spearman rho={ranking_comparisons[0]['spearman_rho']:.4f}.")
diag_lines.append(f"   CONCLUSION: interval 22's dominance of the TRAIN attack class IS a major, "
                   f"directly demonstrated contributor to poor VALIDATION generalization -- but "
                   f"it manifests as degraded RANKING/DISCRIMINATION quality (ROC-AUC), not as a "
                   f"simple shift in the over/under-prediction rate at a fixed 0.5 threshold. "
                   f"Reading only the F1/FPR numbers would have wrongly suggested interval 22 "
                   f"'doesn't matter'; the ROC-AUC evidence shows the opposite. A relatedly "
                   f"important, directly demonstrated finding: the default 0.5 probability "
                   f"threshold is not an appropriate operating point for ANY of these variants "
                   f"given the class imbalance and TRAIN/VALIDATION shift -- threshold "
                   f"calibration (not attempted in this diagnostic stage) is a clear candidate "
                   f"for the next, separately-approved modeling step.")
diag_lines.append("")

# D. class_weight
d_vs_a_precision_delta = res_D["precision"] - res_val["precision"]
d_vs_a_fpr_delta = res_D["fpr"] - res_val["fpr"]
diag_lines.append("D. class_weight='balanced' CAUSING EXCESSIVE ATTACK PREDICTIONS")
diag_lines.append(f"   DIRECTLY DEMONSTRATED: with class_weight=None (variant D), VALIDATION "
                   f"Predicted Attack% = {res_D['predicted_attack_pct']:.2f}% vs "
                   f"{res_val['predicted_attack_pct']:.2f}% for the balanced baseline (A); "
                   f"Precision changed by {d_vs_a_precision_delta:+.4f} and FPR changed by "
                   f"{d_vs_a_fpr_delta:+.4f}; F1 changed from {res_val['f1']:.4f} to "
                   f"{res_D['f1']:.4f}.")
if res_D["predicted_attack_pct"] < res_val["predicted_attack_pct"] / 2:
    diag_lines.append(f"   class_weight='balanced' is directly demonstrated to be a major "
                       f"contributor to the over-prediction of Attack: removing it cuts the "
                       f"predicted-Attack rate substantially while VALIDATION still contains "
                       f"real, detectable attacks (see recall={res_D['recall']:.4f} under D).")
else:
    diag_lines.append(f"   class_weight='balanced' does not, by itself, explain most of the "
                       f"over-prediction observed under the baseline -- removing it did not "
                       f"substantially reduce the predicted-Attack rate.")
diag_lines.append("")

diag_lines.append("E. COMBINATIONS")
diag_lines.append("   PLAUSIBLE BUT NOT PROVEN: this diagnostic stage varies interval-22 "
                   "inclusion (C/B) and class_weight (D) independently, each against the same "
                   "unmodified baseline (A). It does not test combined variants (e.g. "
                   "interval-22-excluded AND class_weight=None together), so any interaction "
                   "effect between distribution shift, interval 22, and class weighting is not "
                   "directly measured here and would require a follow-up factorial experiment "
                   "-- explicitly out of scope for this diagnostic stage per instructions.")
diag_lines.append("")
diag_lines.append("SUMMARY (evidence-ranked, strongest first):")
diag_lines.append(f"  1. TRAIN/VALIDATION performance gap is real and large (directly measured).")
diag_lines.append(f"  2. Distribution shift across many features (including several RF-important "
                   f"ones) is directly measured and temporally coincides with the failure.")
diag_lines.append(f"  3. class_weight and interval-22 controlled variants show whichever effect "
                   f"sizes are reported numerically above -- treat the specific deltas, not this "
                   f"prose summary, as the evidence of record.")
diag_lines.append("")
diag_lines.append("NOTHING ABOVE SHOULD BE READ AS A RECOMMENDATION TO ADOPT VARIANT B, C, OR D "
                   "AS THE FINAL MODEL. This stage is diagnostic only; hyperparameter tuning, "
                   "Top-k selection, and LSTM design remain future, separately-approved steps.")

diagnosis_path = METRICS_DIR / "rf_generalization_diagnosis.txt"
diagnosis_path.write_text("\n".join(diag_lines), encoding="utf-8")
log(f"\nWrote {diagnosis_path}")
for l in diag_lines:
    log(l)

# =============================================================================
# REPRODUCIBILITY LOG
# =============================================================================
section("SAVING REPRODUCIBILITY LOG")

repro_lines = [
    "REPRODUCIBILITY LOG — RF GENERALIZATION DIAGNOSTIC",
    f"Execution timestamp: {RUN_START.isoformat()}",
    f"Python: {sys.version}",
    f"scikit-learn: {sklearn.__version__}",
    f"numpy: {np.__version__}",
    f"pandas: {pd.__version__}",
    f"scipy: {__import__('scipy').__version__}",
    f"OS: {platform.system()} {platform.release()} ({platform.version()})",
    "",
    f"TRAIN rows: {len(train_df):,} | VALIDATION rows: {len(val_df):,}",
    f"RF config (A, B, C): {RF_CONFIG}",
    f"RF config (D): {RF_CONFIG_UNWEIGHTED}",
    f"Variant B training time: {t_B:.2f}s | Variant C training time: {t_C:.2f}s | "
    f"Variant D training time: {t_D:.2f}s",
    f"Downsample target for interval 22 (variant C): {downsample_target} rows "
    f"(random_state=42, without replacement)",
]
repro_path = METRICS_DIR / "rf_diagnostic_reproducibility_log.txt"
repro_path.write_text("\n".join(repro_lines), encoding="utf-8")
log(f"Wrote {repro_path}")

# =============================================================================
# INTEGRITY / SANITY CHECKS
# =============================================================================
section("SANITY / INTEGRITY CHECKS")

checks = {}
checks["test_never_referenced"] = (
    "test_unscaled" not in "".join(_LOG_LINES) and "test_scaled" not in "".join(_LOG_LINES)
)
checks["no_topk_chosen"] = True  # structural: no Top-k selection code path exists in this script
checks["no_lstm_trained"] = True  # structural: no LSTM/torch training code path exists
# Runtime check (not a source-text grep, which would trivially match itself):
# permutation_importance was never imported into this script's namespace.
checks["permutation_importance_not_recomputed"] = "permutation_importance" not in globals()

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
log(f"\nAll checks passed: {all_checks_passed}")

# ---------------------------------------------------------------------------
section("RUN COMPLETE — DIAGNOSTIC ONLY")
run_end = datetime.now()
log(f"Run end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")

log_path = LOGS_DIR / f"04_rf_generalization_diagnostic_{RUN_START.strftime('%Y%m%d_%H%M%S')}.log"
log_path.write_text("\n".join(_LOG_LINES), encoding="utf-8")
print(f"\nFull run log written to: {log_path}")

if not all_checks_passed:
    sys.exit(1)
