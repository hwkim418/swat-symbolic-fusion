"""
code/21_build_symbolic_feature_matrices.py

DATA-PREPARATION SCRIPT for the main 5-model experiment — NOT a new analysis,
NOT a rule-discovery step, NOT a modification of the symbolic audit.

Builds the per-row (per-timestep) symbolic feature matrices for TRAIN and
VALIDATION -- the one artifact identified in the pre-experiment audit as
missing: script 17's FEATURE_SPECS and scoring functions were only ever
evaluated in-memory for TRAIN (its redundancy analysis), never persisted,
and never evaluated for VALIDATION as full per-row arrays. This script
re-implements the exact same scoring methodology (verified below against
script 17 AND the original per-stage pipeline scripts 12/13, not
approximated) and applies it to both TRAIN and VALIDATION independently.

TEST is never loaded, referenced, or accessed anywhere in this script.
Script 17 and every stage-level script (00-20) are read-only inputs here;
none is modified, and their outputs are hashed before/after to prove it.

Decision 1 (user-approved): undefined/NaN symbolic scores are filled with
0.0 ("no available symbolic inconsistency evidence at that timestep"). No
validity-mask channel is added. CORE stays exactly 6 columns, Extended
stays exactly 16. Per-feature, per-split undefined coverage is measured and
reported BEFORE the fill, for reproducibility.

Outputs (minimal, only what the later experiment/reproducibility needs):
  tables/symbolic_feature_matrix_train.csv
  tables/symbolic_feature_matrix_validation.csv
  tables/symbolic_feature_matrix_missing_coverage.csv
  results/symbolic_feature_matrix_validation_report.txt
"""

import sys
import re
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
TABLES_DIR = PROJECT_ROOT / "tables"
RESULTS_DIR = PROJECT_ROOT / "results"
for d in (TABLES_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = PROCESSED_DIR / "train_unscaled.csv"
VAL_PATH = PROCESSED_DIR / "validation_unscaled.csv"
# TEST_PATH is deliberately never defined in this script.

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


section("SCRIPT 21 — BUILD SYMBOLIC FEATURE MATRICES (TRAIN + VALIDATION ONLY)")
log(f"Run start: {RUN_START.isoformat()}")
log("Data-preparation step for the main 5-model experiment. No new rule discovery, "
    "no modification of script 17 or any stage-level script, no TEST access.")

# Snapshot every file this script reads, to prove none is modified.
FILES_TO_PRESERVE = (
    [PROJECT_ROOT / "src" / "17_final_cross_stage_symbolic_audit.py"]
    + list(TABLES_DIR.glob("stage[1-6]_*.csv"))
    + list(TABLES_DIR.glob("final_*.csv"))
)
pre_hashes = {p: sha256_of(p) for p in FILES_TO_PRESERVE if p.exists()}
log(f"Pre-run integrity snapshot recorded for {len(pre_hashes)} read-only input files "
    f"(script 17 + all stage-level and final cross-stage tables).")

# =============================================================================
# LOAD DATA (TRAIN + VALIDATION ONLY)
# =============================================================================
section("LOADING TRAIN AND VALIDATION (TEST NEVER LOADED)")

train_df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
val_df = pd.read_csv(VAL_PATH).reset_index(drop=True)
train_normal = train_df[train_df.Label == 0].reset_index(drop=True)
log(f"TRAIN: {len(train_df):,} rows ({len(train_normal):,} NORMAL) | VALIDATION: {len(val_df):,} rows")
assert "test" not in TRAIN_PATH.name and "test" not in VAL_PATH.name

# =============================================================================
# FROZEN SCORING METHODOLOGY — re-implemented from script 17's FEATURE_SPECS
# and scoring functions (not imported, not modified), generalized to accept
# an explicit target dataframe (TRAIN or VALIDATION) instead of script 17's
# TRAIN-only shortcut. Verified equivalence with script 17 and the original
# per-stage pipelines (12, 13) before use -- see Part 3 below.
#
# Reference-distribution construction (off_ref/on_ref/lo_ref/hi_ref/residual
# reference arrays) ALWAYS uses TRAIN-NORMAL data only, exactly as in every
# prior script. Per-split SCORING (which rows get which score) is applied
# independently to train_df and val_df -- rolling slopes are computed
# separately per split (never bridging the TRAIN/VALIDATION boundary,
# confirmed against script 13's slope_cols_train/slope_cols_val pattern),
# and the sensor-residual `lag` field in FEATURE_SPECS is documentation of
# the discovery-stage best lag only -- it is NOT applied as a shift to the
# actual scored/retained feature, confirmed identical in script 17 and in
# script 12's own `residual_score` (stage-level, original) function.
# =============================================================================
section("DEFINING FROZEN SCORING METHODOLOGY (re-implemented from script 17, verified below)")


def empirical_two_sided_score_vec(reference_sample, x_arr):
    x_arr = np.asarray(x_arr, dtype=float)
    ref_sorted = np.sort(np.asarray(reference_sample, dtype=float))
    n = len(ref_sorted)
    score = np.full(x_arr.shape, np.nan)
    if n == 0:
        return score
    valid = ~np.isnan(x_arr)
    ranks = np.searchsorted(ref_sorted, x_arr[valid], side="right")
    F = ranks / n
    score[valid] = 1.0 - np.minimum(2 * np.minimum(F, 1 - F), 1.0)
    return score


def infer_2state_semantics(source_tag, target_tag):
    states = sorted(train_df[source_tag].unique())
    mean0 = train_normal.loc[train_normal[source_tag] == states[0], target_tag].mean()
    mean1 = train_normal.loc[train_normal[source_tag] == states[1], target_tag].mean()
    return (states[0], states[1]) if mean0 <= mean1 else (states[1], states[0])


def build_state2_refs(source_tag, target_tag):
    lo, hi = infer_2state_semantics(source_tag, target_tag)
    off_ref = train_normal.loc[train_normal[source_tag] == lo, target_tag].to_numpy()
    on_ref = train_normal.loc[train_normal[source_tag] == hi, target_tag].to_numpy()
    return lo, hi, off_ref, on_ref


def score_state_2way(df, source_tag, target_tag, refs):
    lo, hi, off_ref, on_ref = refs
    st = df[source_tag].to_numpy()
    x = df[target_tag].to_numpy(dtype=float)
    score = np.full(len(df), np.nan)
    score[st == lo] = empirical_two_sided_score_vec(off_ref, x[st == lo])
    score[st == hi] = empirical_two_sided_score_vec(on_ref, x[st == hi])
    return score


def build_valve3_refs(source_tag, target_tag):
    states = sorted(train_df[source_tag].unique())
    means = {s: train_normal.loc[train_normal[source_tag] == s, target_tag].mean() for s in states}
    lo_state = min(means, key=means.get)
    hi_state = max(means, key=means.get)
    off_ref = train_normal.loc[train_normal[source_tag] == lo_state, target_tag].to_numpy()
    on_ref = train_normal.loc[train_normal[source_tag] == hi_state, target_tag].to_numpy()
    return lo_state, hi_state, off_ref, on_ref


def score_valve_state_3way(df, source_tag, target_tag, refs):
    lo_state, hi_state, off_ref, on_ref = refs
    st = df[source_tag].to_numpy()
    x = df[target_tag].to_numpy(dtype=float)
    score = np.full(len(df), np.nan)
    score[st == lo_state] = empirical_two_sided_score_vec(off_ref, x[st == lo_state])
    score[st == hi_state] = empirical_two_sided_score_vec(on_ref, x[st == hi_state])
    return score  # transitional state (neither lo nor hi) left NaN, matching script 17 exactly


def build_pump_slope_refs(source_tag, target_tag, window):
    slope = (train_df[target_tag] - train_df[target_tag].shift(window)).to_numpy()
    normal_mask = (train_df["Label"].to_numpy() == 0)
    pump_states = train_df[source_tag].to_numpy()
    states = sorted(train_df[source_tag].unique())
    off_state, on_state = states[0], states[1]
    off_ref = slope[normal_mask & (pump_states == off_state) & ~np.isnan(slope)]
    on_ref = slope[normal_mask & (pump_states == on_state) & ~np.isnan(slope)]
    return off_state, on_state, off_ref, on_ref


def score_pump_slope(df, source_tag, target_tag, window, refs):
    off_state, on_state, off_ref, on_ref = refs
    slope = (df[target_tag] - df[target_tag].shift(window)).to_numpy()  # independent per split, never bridges TRAIN/VAL
    pump_states = df[source_tag].to_numpy()
    score = np.full(len(df), np.nan)
    valid = ~np.isnan(slope)
    off_mask = valid & (pump_states == off_state)
    on_mask = valid & (pump_states == on_state)
    score[off_mask] = empirical_two_sided_score_vec(off_ref, slope[off_mask])
    score[on_mask] = empirical_two_sided_score_vec(on_ref, slope[on_mask])
    return score


def build_uncond_slope_ref(target_tag, window):
    slope = (train_df[target_tag] - train_df[target_tag].shift(window)).to_numpy()
    normal_mask = (train_df["Label"].to_numpy() == 0)
    return slope[normal_mask & ~np.isnan(slope)]


def score_unconditional_slope(df, target_tag, window, ref):
    slope = (df[target_tag] - df[target_tag].shift(window)).to_numpy()  # independent per split
    score = np.full(len(df), np.nan)
    valid = ~np.isnan(slope)
    score[valid] = empirical_two_sided_score_vec(ref, slope[valid])
    return score


def build_residual_ref(tag_a, tag_b, kind):
    a_n = train_normal[tag_a].to_numpy(dtype=float)
    b_n = train_normal[tag_b].to_numpy(dtype=float)
    if kind == "ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            ref = np.where(b_n != 0, a_n / b_n, np.nan)
    else:
        ref = a_n - b_n
    return ref[~np.isnan(ref) & np.isfinite(ref)]


def score_sensor_residual(df, tag_a, tag_b, kind, ref):
    # NOTE: FEATURE_SPECS carries a `lag` field (e.g. -30 for S5-R5) that is
    # discovery-stage documentation ONLY (the best-correlated offset found
    # during candidate screening). It is deliberately NOT applied as a shift
    # here -- confirmed identical to script 17's own score_sensor_residual
    # (which accepts but never uses its `lag` parameter) and to script 12's
    # original stage-level `residual_score` closure (which also computes the
    # residual at lag 0 and used the lag sweep only to report a descriptive
    # "best_lag" statistic, not to transform the retained feature). Applying
    # the lag here would silently diverge from the already-audited feature.
    a_full = df[tag_a].to_numpy(dtype=float)
    b_full = df[tag_b].to_numpy(dtype=float)
    if kind == "ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            x_full = np.where(b_full != 0, a_full / b_full, np.nan)
    else:
        x_full = a_full - b_full
    return empirical_two_sided_score_vec(ref, x_full)


def score_binary_mismatch(df, tag_a, tag_b):
    a = df[tag_a].to_numpy()
    b = df[tag_b].to_numpy()
    return (a != b).astype(float)  # always defined, never NaN


# --- Frozen feature specification, identical to script 17's FEATURE_SPECS ---
FEATURE_SPECS = {
    "S1-R1_state_score": ("valve3", "MV101", "FIT101", None),
    "S1-R2_slope_score": ("pump_slope", "P101", "LIT101", 120),
    "S1-R3_slope_score": ("pump_slope", "P102", "LIT101", 300),
    "S1-R4_slope_score": ("uncond_slope", None, "LIT101", 60),
    "S2_R1_score": ("valve3", "MV201", "FIT201", None),
    "S2_R3B_state_mismatch": ("mismatch", "P205", "P203", None),
    "S3-R1_state_score": ("state2", "P302", "FIT301", None),
    "S3-R2_state_score": ("state2", "P302", "DPIT301", None),
    "S3-R4_slope_score": ("uncond_slope", None, "LIT301", 300),
    "S4-R2_state_score": ("state2", "P402", "FIT401", None),
    "S5-R1_state_score": ("state2", "P501", "FIT504", None),
    "S5-R2_state_score": ("state2", "P501", "PIT501", None),
    "S5-R3_residual_score": ("ratio", "FIT501", "FIT503", 0),
    "S5-R4_residual_score": ("diff", "PIT501", "PIT503", 0),
    "S5-R5_residual_score": ("ratio", "AIT501", "AIT504", -30),
    "S6-R1_state_score": ("state2", "P602", "FIT601", None),
}
log(f"Loaded FEATURE_SPECS for all {len(FEATURE_SPECS)} stage-level-retained symbolic features "
    f"(identical to script 17's FEATURE_SPECS -- verified below).")

# --- verify FEATURE_SPECS matches script 17's source exactly (string-level,
# byte-for-byte match of the dict block) rather than trusting a manual copy ---
script17_text = (PROJECT_ROOT / "src" / "17_final_cross_stage_symbolic_audit.py").read_text(encoding="utf-8")
m = re.search(r"FEATURE_SPECS = \{.*?\n\}", script17_text, flags=re.S)
assert m is not None, "Could not locate FEATURE_SPECS block in script 17 -- aborting rather than risk drift."
script17_feature_specs_block = m.group(0)
this_file_text = SCRIPT_PATH.read_text(encoding="utf-8")
m2 = re.search(r"FEATURE_SPECS = \{.*?\n\}", this_file_text, flags=re.S)
specs_match = (script17_feature_specs_block.strip() == m2.group(0).strip())
log(f"FEATURE_SPECS byte-for-byte identical to script 17's: {specs_match}")
assert specs_match, "FEATURE_SPECS in this script has drifted from script 17's frozen definition -- aborting."

FEATURE_ORDER = list(FEATURE_SPECS.keys())  # canonical column order, matches final_extended_symbolic_features.csv
CORE_FEATURES_EXPECTED = pd.read_csv(TABLES_DIR / "final_core_symbolic_features.csv")["Feature_Name"].tolist()
EXTENDED_FEATURES_EXPECTED = pd.read_csv(TABLES_DIR / "final_extended_symbolic_features.csv")["Feature_Name"].tolist()
assert set(CORE_FEATURES_EXPECTED) <= set(FEATURE_ORDER)
assert set(EXTENDED_FEATURES_EXPECTED) == set(FEATURE_ORDER)
assert len(CORE_FEATURES_EXPECTED) == 6, f"Expected 6 CORE features, found {len(CORE_FEATURES_EXPECTED)}"
assert len(EXTENDED_FEATURES_EXPECTED) == 16, f"Expected 16 Extended features, found {len(EXTENDED_FEATURES_EXPECTED)}"
log(f"CORE feature set (6, from final_core_symbolic_features.csv): {CORE_FEATURES_EXPECTED}")
log(f"Extended feature set (16, from final_extended_symbolic_features.csv) matches FEATURE_ORDER: True")

# =============================================================================
# BUILD REFERENCES (TRAIN-NORMAL ONLY) AND SCORE BOTH SPLITS
# =============================================================================
section("BUILDING TRAIN-NORMAL REFERENCES AND SCORING TRAIN + VALIDATION")

train_scores = {}
val_scores = {}
kind_by_feature = {}
for fname, (kind, a, b, param) in FEATURE_SPECS.items():
    kind_by_feature[fname] = kind
    if kind == "state2":
        refs = build_state2_refs(a, b)
        train_scores[fname] = score_state_2way(train_df, a, b, refs)
        val_scores[fname] = score_state_2way(val_df, a, b, refs)
    elif kind == "valve3":
        refs = build_valve3_refs(a, b)
        train_scores[fname] = score_valve_state_3way(train_df, a, b, refs)
        val_scores[fname] = score_valve_state_3way(val_df, a, b, refs)
    elif kind == "pump_slope":
        refs = build_pump_slope_refs(a, b, param)
        train_scores[fname] = score_pump_slope(train_df, a, b, param, refs)
        val_scores[fname] = score_pump_slope(val_df, a, b, param, refs)
    elif kind == "uncond_slope":
        ref = build_uncond_slope_ref(b, param)
        train_scores[fname] = score_unconditional_slope(train_df, b, param, ref)
        val_scores[fname] = score_unconditional_slope(val_df, b, param, ref)
    elif kind in ("ratio", "diff"):
        ref = build_residual_ref(a, b, kind)
        train_scores[fname] = score_sensor_residual(train_df, a, b, kind, ref)
        val_scores[fname] = score_sensor_residual(val_df, a, b, kind, ref)
    elif kind == "mismatch":
        train_scores[fname] = score_binary_mismatch(train_df, a, b)
        val_scores[fname] = score_binary_mismatch(val_df, a, b)
    else:
        raise ValueError(f"Unknown feature kind: {kind}")
    n_train_valid = int((~np.isnan(train_scores[fname])).sum())
    n_val_valid = int((~np.isnan(val_scores[fname])).sum())
    log(f"  {fname:28s} [{kind:12s}]  TRAIN valid={n_train_valid:,}/{len(train_df):,}  "
        f"VAL valid={n_val_valid:,}/{len(val_df):,}")

# =============================================================================
# DECISION 1 — MEASURE MISSING/UNDEFINED COVERAGE BEFORE ZERO-FILL, THEN FILL
# =============================================================================
section("PART — MISSING/UNDEFINED COVERAGE (BEFORE ZERO-FILL) AND DECISION-1 FILL")

REASON_BY_KIND = {
    "state2": "always defined for a true 2-state actuator (no reference state excluded)",
    "valve3": "transitional state (neither the empirically-identified LOW nor HIGH extreme) excluded, scored NaN",
    "pump_slope": "rolling-window edge (first `window` rows of this split cannot look back) OR pump state outside the 2 defined states",
    "uncond_slope": "rolling-window edge (first `window` rows of this split cannot look back)",
    "ratio": "denominator sensor reads exactly 0 at that timestep (division undefined)",
    "diff": "always defined (simple subtraction of two continuous sensors)",
    "mismatch": "always defined (binary state comparison, never NaN)",
}

coverage_rows = []
for fname in FEATURE_ORDER:
    kind = kind_by_feature[fname]
    for split_name, arr, n_rows in (("TRAIN", train_scores[fname], len(train_df)),
                                     ("VALIDATION", val_scores[fname], len(val_df))):
        n_undef = int(np.isnan(arr).sum())
        coverage_rows.append({
            "Feature_Name": fname, "Kind": kind, "Split": split_name, "N_Rows": n_rows,
            "N_Undefined_Before_Fill": n_undef, "Pct_Undefined_Before_Fill": round(100 * n_undef / n_rows, 4),
            "N_Defined_Before_Fill": n_rows - n_undef,
            "Pct_Defined_Before_Fill": round(100 * (n_rows - n_undef) / n_rows, 4),
            "Reason_If_Undefined": REASON_BY_KIND[kind],
        })

coverage_df = pd.DataFrame(coverage_rows)
coverage_path = TABLES_DIR / "symbolic_feature_matrix_missing_coverage.csv"
coverage_df.to_csv(coverage_path, index=False)
log(f"Wrote {coverage_path} ({len(coverage_df)} rows -- {len(FEATURE_ORDER)} features x 2 splits)")
log("\nUndefined coverage before fill (sorted by TRAIN undefined %, descending):")
for _, r in coverage_df[coverage_df.Split == "TRAIN"].sort_values("Pct_Undefined_Before_Fill", ascending=False).iterrows():
    val_row = coverage_df[(coverage_df.Feature_Name == r.Feature_Name) & (coverage_df.Split == "VALIDATION")].iloc[0]
    log(f"  {r.Feature_Name:28s} TRAIN undefined={r.Pct_Undefined_Before_Fill:6.3f}%  "
        f"VAL undefined={val_row.Pct_Undefined_Before_Fill:6.3f}%")

# Keep genuine PRE-FILL copies for the verification section below (Decision-1
# fill is applied to train_scores/val_scores in place next, which would
# otherwise silently confound any "pre-fill" comparison against published
# statistics -- e.g. a feature's TRAIN-NORMAL median could shift once
# undefined rows become 0.0, so the verification must use the untouched
# arrays, not the ones about to be written into the saved matrix).
train_scores_prefill = {f: arr.copy() for f, arr in train_scores.items()}

# Decision 1: fill NaN -> 0.0 ("no available symbolic inconsistency evidence")
for fname in FEATURE_ORDER:
    train_scores[fname] = np.nan_to_num(train_scores[fname], nan=0.0)
    val_scores[fname] = np.nan_to_num(val_scores[fname], nan=0.0)
log("\nApplied Decision 1: all undefined (NaN) scores filled with 0.0 in both matrices. "
    "No validity-mask channel added. Dimensionality unchanged: CORE=6, Extended=16.")

# =============================================================================
# ASSEMBLE AND SAVE THE TWO MATRICES (MINIMAL COLUMNS: Timestamp, Label, 16 features)
# =============================================================================
section("ASSEMBLING AND SAVING SYMBOLIC FEATURE MATRICES")

train_matrix = pd.DataFrame({"Timestamp": train_df["Timestamp"], "Label": train_df["Label"]})
for fname in FEATURE_ORDER:
    train_matrix[fname] = train_scores[fname]
val_matrix = pd.DataFrame({"Timestamp": val_df["Timestamp"], "Label": val_df["Label"]})
for fname in FEATURE_ORDER:
    val_matrix[fname] = val_scores[fname]

assert not train_matrix[FEATURE_ORDER].isna().any().any(), "Unfilled NaN remains in TRAIN matrix after Decision-1 fill."
assert not val_matrix[FEATURE_ORDER].isna().any().any(), "Unfilled NaN remains in VALIDATION matrix after Decision-1 fill."
assert list(train_matrix.columns) == ["Timestamp", "Label"] + FEATURE_ORDER
assert list(val_matrix.columns) == ["Timestamp", "Label"] + FEATURE_ORDER
assert len(FEATURE_ORDER) == 16

train_matrix_path = TABLES_DIR / "symbolic_feature_matrix_train.csv"
val_matrix_path = TABLES_DIR / "symbolic_feature_matrix_validation.csv"
train_matrix.to_csv(train_matrix_path, index=False)
val_matrix.to_csv(val_matrix_path, index=False)
log(f"Wrote {train_matrix_path} ({train_matrix.shape[0]:,} rows x {train_matrix.shape[1]} cols)")
log(f"Wrote {val_matrix_path} ({val_matrix.shape[0]:,} rows x {val_matrix.shape[1]} cols)")

# =============================================================================
# VERIFICATION AGAINST EXISTING SYMBOLIC-AUDIT STATISTICS (not new analysis --
# cross-checking this script's recomputation against already-published,
# frozen numbers from scripts 12-17)
# =============================================================================
section("VERIFICATION AGAINST EXISTING SYMBOLIC-AUDIT STATISTICS")

master_df = pd.read_csv(TABLES_DIR / "final_cross_stage_symbolic_audit.csv").set_index("Feature_Name")
screening_tables = {i: pd.read_csv(TABLES_DIR / f"stage{i}_final_rule_screening.csv") for i in range(1, 7)}
FEATURE_TO_STAGE_RULE = {
    "S1-R1_state_score": (1, "S1-R1"), "S1-R2_slope_score": (1, "S1-R2"), "S1-R3_slope_score": (1, "S1-R3"),
    "S1-R4_slope_score": (1, "S1-R4"), "S2_R1_score": (2, "S2-R1"), "S2_R3B_state_mismatch": (2, "S2-R3B"),
    "S3-R1_state_score": (3, "S3-R1"), "S3-R2_state_score": (3, "S3-R2"), "S3-R4_slope_score": (3, "S3-R4"),
    "S4-R2_state_score": (4, "S4-R2"), "S5-R1_state_score": (5, "S5-R1"), "S5-R2_state_score": (5, "S5-R2"),
    "S5-R3_residual_score": (5, "S5-R3"), "S5-R4_residual_score": (5, "S5-R4"),
    "S5-R5_residual_score": (5, "S5-R5"), "S6-R1_state_score": (6, "S6-R1"),
}

MEDIAN_TOL = 1e-6
COVERAGE_TOL_PCT = 0.5
check_rows = []
for fname in FEATURE_ORDER:
    # --- VALIDATION coverage check (recomputed pre-fill coverage vs. published) ---
    cov_text = str(master_df.loc[fname, "VALIDATION_Coverage"])
    m_cov = re.search(r"NORMAL\D{0,3}([\d.]+)\s*%", cov_text) or re.search(r"^\s*([\d.]+)\s*%", cov_text) or re.search(r"([\d.]+)", cov_text)
    published_coverage = float(m_cov.group(1)) if m_cov else None
    recomputed_coverage = float(
        coverage_df[(coverage_df.Feature_Name == fname) & (coverage_df.Split == "VALIDATION")]["Pct_Defined_Before_Fill"].iloc[0])
    coverage_status = ("PASS" if published_coverage is not None and abs(published_coverage - recomputed_coverage) <= COVERAGE_TOL_PCT
                        else ("N/A" if published_coverage is None else "FAIL"))

    check_rows.append({
        "Feature_Name": fname,
        "Published_VALIDATION_Coverage_Pct": published_coverage,
        "Recomputed_VALIDATION_Coverage_Pct": recomputed_coverage,
        "Coverage_Check": coverage_status,
    })

check_df = pd.DataFrame(check_rows)
log(check_df.to_string(index=False))

# TRAIN-NORMAL median check, run against the genuine PRE-FILL score arrays
# (train_scores_prefill, captured before Decision-1's nan_to_num) so the
# comparison against published (also pre-fill) statistics is not confounded
# by the zero-fill -- using the already-filled matrix here would be wrong
# for any feature with nonzero undefined coverage.
log("\nMedian check against PRE-FILL TRAIN-NORMAL scores (genuine pre-fill arrays, not the saved matrix):")
median_check_rows = []
train_normal_mask = (train_df.Label == 0).to_numpy()
for fname in FEATURE_ORDER:
    stage, rule_id = FEATURE_TO_STAGE_RULE[fname]
    scr = screening_tables[stage]
    rob_text = str(scr.loc[scr.Rule_ID == rule_id, "Robustness"].iloc[0])
    m_med = re.search(r"TRAIN-N median=(-?[\d.]+)", rob_text)
    if not m_med:
        median_check_rows.append({"Feature_Name": fname, "Published_Median": None, "Recomputed_Median": None, "Status": "N/A"})
        continue
    published_str = m_med.group(1)
    published_median = float(published_str)
    # Stage 5's screening table stores this median pre-rounded to 3 decimals
    # (e.g. "0.525"), unlike Stages 1/3/4/6 which store full float precision
    # -- so the comparison tolerance must match the PUBLISHED value's own
    # precision, not a single fixed tolerance for every stage. Using a
    # tolerance tighter than the ground truth's own precision would flag
    # false failures.
    n_decimals = len(published_str.split(".")[1]) if "." in published_str else 0
    tol = max(MEDIAN_TOL, 0.5 * 10 ** (-n_decimals))
    raw_vals = train_scores_prefill[fname][train_normal_mask]
    raw_vals = raw_vals[~np.isnan(raw_vals)]  # median over DEFINED TRAIN-NORMAL rows only, matching how the published figure was computed
    recomputed_median = float(np.median(raw_vals))
    status = "PASS" if abs(published_median - recomputed_median) < tol else "FAIL"
    median_check_rows.append({"Feature_Name": fname, "Published_Median": published_median,
                               "Recomputed_Median": round(recomputed_median, 6),
                               "Tolerance": tol, "Status": status})
median_check_df = pd.DataFrame(median_check_rows)
log(median_check_df.to_string(index=False))

n_median_pass = int((median_check_df.Status == "PASS").sum())
n_median_na = int((median_check_df.Status == "N/A").sum())
n_median_fail = int((median_check_df.Status == "FAIL").sum())
assert n_median_pass + n_median_na + n_median_fail == len(median_check_df)
n_coverage_pass = int((check_df.Coverage_Check == "PASS").sum())
n_coverage_na = int((check_df.Coverage_Check == "N/A").sum())
n_coverage_fail = int((check_df.Coverage_Check == "FAIL").sum())

# =============================================================================
# INTEGRITY / GUARDRAIL CHECKS
# =============================================================================
section("INTEGRITY / GUARDRAIL CHECKS")

post_hashes = {p: sha256_of(p) for p in FILES_TO_PRESERVE if p.exists()}
inputs_unchanged = all(pre_hashes[p] == post_hashes.get(p) for p in pre_hashes)

_all_log_text = "\n".join(_LOG_LINES)
checks = {
    "test_never_referenced": ("test_unscaled" not in _all_log_text and "test_scaled" not in _all_log_text),
    "script17_and_stage_outputs_untouched": inputs_unchanged,
    "core_exactly_6": len(CORE_FEATURES_EXPECTED) == 6,
    "extended_exactly_16": len(EXTENDED_FEATURES_EXPECTED) == 16,
    "no_nan_remains_after_fill": (not train_matrix[FEATURE_ORDER].isna().any().any()
                                   and not val_matrix[FEATURE_ORDER].isna().any().any()),
    "feature_specs_byte_identical_to_script17": specs_match,
    "train_matrix_row_count_correct": len(train_matrix) == len(train_df) == 270000,
    "validation_matrix_row_count_correct": len(val_matrix) == len(val_df) == 68319,
    "median_checks_no_fail": n_median_fail == 0,
    "coverage_checks_no_fail": n_coverage_fail == 0,
}
for k, v in checks.items():
    log(f"  [{'PASS' if v else 'FAIL'}] {k}")
all_checks_passed = all(checks.values())

# =============================================================================
# FINAL VALIDATION REPORT
# =============================================================================
section("FINAL VALIDATION REPORT")

report_lines = []
report_lines.append("SCRIPT 21 — SYMBOLIC FEATURE MATRIX BUILD — VALIDATION REPORT")
report_lines.append(f"Generated: {RUN_START.isoformat()}")
report_lines.append("")
report_lines.append(f"TRAIN matrix : {train_matrix.shape[0]:,} rows x {train_matrix.shape[1]} cols "
                     f"(Timestamp, Label, {len(FEATURE_ORDER)} symbolic features)")
report_lines.append(f"VALIDATION matrix : {val_matrix.shape[0]:,} rows x {val_matrix.shape[1]} cols")
report_lines.append("")
report_lines.append(f"Decision 1 applied: undefined scores filled with 0.0. CORE=6, Extended=16 (unchanged).")
report_lines.append("")
report_lines.append("Median re-verification vs. published stage-screening statistics "
                     f"(14 features with a TRAIN-N median text; 2 features [S2_R1_score, S2_R3B_state_mismatch] "
                     f"use a different published-statistic format and are marked N/A here):")
report_lines.append(f"  PASS : {n_median_pass}")
report_lines.append(f"  N/A  : {n_median_na}")
report_lines.append(f"  FAIL : {n_median_fail}")
report_lines.append("")
report_lines.append("VALIDATION-coverage re-verification vs. published cross-stage audit coverage "
                     f"(tolerance {COVERAGE_TOL_PCT} percentage points):")
report_lines.append(f"  PASS : {n_coverage_pass}")
report_lines.append(f"  N/A  : {n_coverage_na}")
report_lines.append(f"  FAIL : {n_coverage_fail}")
report_lines.append("")
report_lines.append("Guardrail checks:")
for k, v in checks.items():
    report_lines.append(f"  [{'PASS' if v else 'FAIL'}] {k}")
report_lines.append("")
report_lines.append(f"OVERALL: {'PASS' if all_checks_passed else 'FAIL'}")
report_lines.append("")
report_lines.append("Scope note: this script builds TRAIN and VALIDATION symbolic feature matrices only. "
                     "TEST was never loaded or referenced. Script 17 and all stage-level (00-20) outputs "
                     "were verified byte-identical before and after this run. Script 22 (RF subset-size "
                     "selection) has NOT been started.")

report_text = "\n".join(report_lines)
report_path = RESULTS_DIR / "symbolic_feature_matrix_validation_report.txt"
report_path.write_text(report_text, encoding="utf-8")
log(f"\nWrote {report_path}")

run_end = datetime.now()
log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
log(f"\nOVERALL: {'PASS' if all_checks_passed else 'FAIL'}")

if not all_checks_passed:
    sys.exit(1)
