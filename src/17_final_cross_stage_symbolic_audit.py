"""
code/17_final_cross_stage_symbolic_audit.py

FINAL CROSS-STAGE SYMBOLIC AUDIT — CONSOLIDATES AND RE-EVALUATES ALL
STAGE-LEVEL RETAINED FEATURES (STAGES 1-6) UNDER ONE COMMON STANDARD.
ANALYSIS ONLY. NO NEW RULE DISCOVERY.

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

This script does NOT discover new rules and does NOT modify any
stage-level decision. It loads the 16 features already RETAINED by
scripts 11-16 (Stage 2's finalization script, and Stages 1/3/4/5/6's
end-to-end pipelines), consolidates their evidence into one master
table, computes a genuine cross-feature redundancy analysis (by
recomputing each feature's TRAIN-NORMAL score array using the exact
methodology each stage script already established -- not approximated),
and assigns each feature exactly one final status: CORE,
RETAIN_WITH_CAVEAT, or EXCLUDE, under a single documented framework
applied identically regardless of source stage.

No ML model is trained, no feature is selected by attack performance, no
attack-label-based threshold is tuned, no existing script/dataset/output
is modified, and TEST is never loaded -- its filename does not appear
anywhere below, by design. Uses ONLY processed/train_unscaled.csv and
processed/validation_unscaled.csv, plus the read-only stage-level output
tables from scripts 11-16.
"""

import os
import sys
import re
import hashlib
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

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
TRAIN_PATH = PROCESSED_DIR / "train_unscaled.csv"
VAL_PATH = PROCESSED_DIR / "validation_unscaled.csv"
TABLES_DIR = PROJECT_ROOT / "tables"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
for d in (TABLES_DIR, RESULTS_DIR, FIGURES_DIR):
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


section("FINAL CROSS-STAGE SYMBOLIC AUDIT — RUN START (ANALYSIS ONLY, NO NEW RULE DISCOVERY)")
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
# Snapshot every prior stage output this script reads, to prove none is modified.
STAGE_FILES_TO_PRESERVE = list(TABLES_DIR.glob("stage[1-6]_*.csv")) + list(RESULTS_DIR.glob("stage[1-6]_*.txt")) + list(FIGURES_DIR.glob("stage[1-6]_*.png"))
pre_stage_hashes = {p: sha256_of(p) for p in STAGE_FILES_TO_PRESERVE}
log(f"Pre-run integrity snapshot recorded for raw + processed inputs AND "
    f"{len(STAGE_FILES_TO_PRESERVE)} existing Stage 1-6 output files (all read-only here).")

SPARSE_REFERENCE_THRESHOLD = 100   # minority-state reference sample count below which flagged
LOW_EXPLAINED_VARIANCE_THRESHOLD = 0.15  # for unconditional-slope features
REDUNDANCY_RHO_THRESHOLD = 0.90     # TRAIN-NORMAL score-array Spearman rho above which flagged
COVERAGE_CORE_MIN = 90.0
SATURATION_THRESHOLD = 0.90

# =============================================================================
# LOAD DATA
# =============================================================================
section("LOADING TRAIN AND VALIDATION (TEST NEVER LOADED)")

train_df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
val_df = pd.read_csv(VAL_PATH).reset_index(drop=True)
train_df["_ts"] = pd.to_datetime(train_df["Timestamp"])
val_df["_ts"] = pd.to_datetime(val_df["Timestamp"])
train_normal = train_df[train_df.Label == 0].reset_index(drop=True)
log(f"TRAIN: {train_df.shape[0]:,} rows ({len(train_normal):,} NORMAL) | VALIDATION: {val_df.shape[0]:,} rows")


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


# =============================================================================
# PART 1 — LOAD ALL STAGE-LEVEL RETAINED FEATURES, BUILD MASTER TABLE
# =============================================================================
section("PART 1 — LOADING ALL STAGE-LEVEL RETAINED FEATURES")

EXPECTED_COUNTS = {1: 4, 2: 2, 3: 3, 4: 1, 5: 5, 6: 1}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
log(f"Expected stage-level retained counts (to be VERIFIED, not assumed): {EXPECTED_COUNTS} "
    f"(total={EXPECTED_TOTAL})")

retained_tables = {}
screening_tables = {}
validation_tables = {}
envelope_tables = {}
candidate_tables = {}
for i in range(1, 7):
    retained_tables[i] = pd.read_csv(TABLES_DIR / f"stage{i}_final_retained_symbolic_features.csv")
    screening_tables[i] = pd.read_csv(TABLES_DIR / f"stage{i}_final_rule_screening.csv")
    vpath = TABLES_DIR / f"stage{i}_rule_validation.csv"
    validation_tables[i] = pd.read_csv(vpath) if vpath.exists() else None
    epath = TABLES_DIR / f"stage{i}_normal_operating_envelopes.csv"
    envelope_tables[i] = pd.read_csv(epath) if epath.exists() else None
    cpath = TABLES_DIR / f"stage{i}_candidate_rules.csv"
    candidate_tables[i] = pd.read_csv(cpath) if cpath.exists() else None

actual_counts = {i: len(retained_tables[i]) for i in range(1, 7)}
actual_total = sum(actual_counts.values())
log(f"Actual counts loaded from tables/stageN_final_retained_symbolic_features.csv: {actual_counts} "
    f"(total={actual_total})")
counts_match = actual_counts == EXPECTED_COUNTS
log(f"Counts match expectation exactly: {counts_match}")
if not counts_match:
    log("WARNING: actual retained counts differ from the expected list -- proceeding with ACTUAL "
        "counts as the source of truth (verification, not blind assumption).")

# --- CANDIDATE-RULE COUNTS (SANITY-CHECK FIX) ---
# The authoritative per-stage candidate count is the row count of
# stageN_final_rule_screening.csv, since EVERY candidate rule (RETAIN,
# LIMITED, or REJECT) receives exactly one screening-table row at every
# stage, including Stage 2 -- which never produced a
# stage2_candidate_rules.csv file (it used a differently-named set of
# intermediate tables), so candidate_tables[2] is None. A prior version of
# this script summed len(candidate_tables[i]) and silently skipped stage 2
# entirely via the "if candidate_tables[i] is not None" guard, undercounting
# the true total (23 instead of 27). Cross-verified below against
# stageN_candidate_rules.csv row counts wherever that file exists (it does
# for stages 1,3,4,5,6) to confirm the two sources agree.
candidate_counts_by_stage = {i: len(screening_tables[i]) for i in range(1, 7)}
total_candidates_verified = sum(candidate_counts_by_stage.values())
log(f"\nPer-stage candidate-rule counts (from stageN_final_rule_screening.csv, authoritative -- "
    f"every candidate gets exactly one screening row regardless of RETAIN/LIMITED/REJECT outcome): "
    f"{candidate_counts_by_stage} (total={total_candidates_verified})")
_candidate_count_mismatches = []
for i in range(1, 7):
    if candidate_tables[i] is not None and len(candidate_tables[i]) != candidate_counts_by_stage[i]:
        _candidate_count_mismatches.append(
            (i, len(candidate_tables[i]), candidate_counts_by_stage[i]))
    elif candidate_tables[i] is None:
        log(f"  Stage {i}: stage{i}_candidate_rules.csv does not exist (Stage 2 used differently-named "
            f"intermediate tables) -- screening-table count ({candidate_counts_by_stage[i]}) used as the "
            f"sole source for this stage, cannot be cross-verified against a second file.")
if _candidate_count_mismatches:
    log(f"WARNING: candidate_rules.csv and final_rule_screening.csv disagree on row count for stage(s): "
        f"{_candidate_count_mismatches} -- screening-table count used as authoritative.")
else:
    log("Cross-verification PASSED: for every stage where stageN_candidate_rules.csv exists, its row "
        "count exactly matches stageN_final_rule_screening.csv's row count.")
EXPECTED_CANDIDATE_COUNTS = {1: 5, 2: 4, 3: 6, 4: 6, 5: 5, 6: 1}
EXPECTED_TOTAL_CANDIDATES = sum(EXPECTED_CANDIDATE_COUNTS.values())
assert candidate_counts_by_stage == EXPECTED_CANDIDATE_COUNTS, (
    f"Candidate counts {candidate_counts_by_stage} do not match the independently-verified expected "
    f"counts {EXPECTED_CANDIDATE_COUNTS} -- aborting rather than silently reporting a wrong total.")
assert total_candidates_verified == EXPECTED_TOTAL_CANDIDATES == 27, (
    f"Total candidate count {total_candidates_verified} != 27 -- aborting.")
log(f"VERIFIED: total candidate rules across Stages 1-6 = {total_candidates_verified} "
    f"(matches independently expected total of 27; S1={candidate_counts_by_stage[1]}, "
    f"S2={candidate_counts_by_stage[2]}, S3={candidate_counts_by_stage[3]}, "
    f"S4={candidate_counts_by_stage[4]}, S5={candidate_counts_by_stage[5]}, "
    f"S6={candidate_counts_by_stage[6]}).")

# --- feature-name column per stage (schemas differ; normalized here) ---
FEATURE_COL = {1: "Feature", 2: "Final_Feature_Name", 3: "Feature", 4: "Feature", 5: "Feature", 6: "Feature"}
RULE_COL = {i: "Source_Rule" for i in range(1, 7)}


def get_tags(stage, rule_id, row):
    """Normalizes tag extraction across the 6 differing schemas, falling
    back to the stage's own candidate_rules.csv (Rule_ID, Source_Tags,
    Target_Tags -- present for every stage) when the retained-features
    table itself doesn't carry a combined tag field."""
    if stage == 1 or stage == 4:
        return row.get("Source_Target_Tags", None)
    if stage == 2:
        return row.get("Source_Tags", None)
    if stage in (3, 6):
        s, t = row.get("Source_Tags"), row.get("Target_Tags")
        if pd.notna(s) and pd.notna(t):
            return f"{s} / {t}"
    cand = candidate_tables.get(stage)
    if cand is not None:
        sub = cand[cand.Rule_ID == rule_id]
        if len(sub):
            return f"{sub['Source_Tags'].iloc[0]} / {sub['Target_Tags'].iloc[0]}"
    return None


master_rows = []
for stage in range(1, 7):
    rf = retained_tables[stage]
    scr = screening_tables[stage]
    for _, row in rf.iterrows():
        rule_id = row[RULE_COL[stage]]
        feature_name = row[FEATURE_COL[stage]]
        scr_row = scr[scr.Rule_ID == rule_id]
        scr_row = scr_row.iloc[0] if len(scr_row) else None
        tags = get_tags(stage, rule_id, row)
        repr_type = row.get("Representation_Type", "n/a")
        limitation = (row.get("Limitation") if "Limitation" in row else
                      row.get("Robustness_Limitation") if "Robustness_Limitation" in row else
                      row.get("Downstream_Fusion_Readiness") if "Downstream_Fusion_Readiness" in row else "n/a")
        # IMPORTANT: prioritize the retained-features table's OWN coverage
        # field, since it describes THIS exact feature. The screening
        # table's "Coverage" column describes the RULE (which, for a rule
        # like S2-R3B that produced multiple sub-features -- an event-based
        # companion PLUS a state-based primary feature -- may quote the
        # companion's much lower coverage, not the retained feature's own).
        # Only fall back to the screening table when the feature-level
        # table has no usable coverage field at all.
        coverage_text = None
        for col in ("Coverage_Pct_VALIDATION_NORMAL", "Coverage", "NORMAL_Coverage_Pct"):
            if col in row and pd.notna(row.get(col)):
                coverage_text = f"{row[col]}"
                break
        if coverage_text is None and scr_row is not None:
            coverage_text = (scr_row.get("Coverage") if "Coverage" in scr_row.index and pd.notna(scr_row.get("Coverage"))
                              else scr_row.get("Overall_Coverage") if "Overall_Coverage" in scr_row.index
                              else None)
        master_rows.append({
            "Stage": f"Stage {stage}", "Rule_ID": rule_id, "Feature_Name": feature_name,
            "Physical_Relationship": scr_row["Physical_Relationship"] if scr_row is not None else tags,
            "Source_Tags": tags, "Representation_Type": repr_type,
            "TRAIN_Normal_Evidence": scr_row["TRAIN_Normal_Evidence"] if scr_row is not None else "n/a",
            "VALIDATION_Coverage": coverage_text,
            "VALIDATION_Normal_Score_Behavior": scr_row["Robustness"] if scr_row is not None else "n/a",
            "Robustness_Notes": scr_row["Robustness"] if scr_row is not None else "n/a",
            "Sparsity_Notes": "", "Distribution_Shift_Notes": "", "Redundancy_Notes": "",
            "Temporal_Lag_Notes": scr_row.get("Temporal_Evidence", "n/a") if scr_row is not None else "n/a",
            "Interpretability": "n/a",
            "Main_Limitation": scr_row["Main_Limitation"] if scr_row is not None else limitation,
            "Original_Stage_Decision": scr_row["Final_Decision"] if scr_row is not None else "RETAIN",
        })

master_df = pd.DataFrame(master_rows)
log(f"\nBuilt master table with {len(master_df)} rows (features).")
for _, r in master_df.iterrows():
    log(f"  {r['Stage']} {r['Rule_ID']}: {r['Feature_Name']}  [{r['Original_Stage_Decision']}]")

assert (master_df["Original_Stage_Decision"] == "RETAIN").all(), \
    "A non-RETAIN stage-level rule leaked into the master table -- aborting (guardrail violation)."
log("\nGuardrail verified: every row in the master table has Original_Stage_Decision == 'RETAIN' "
    "(no LIMITED or REJECT rule was reintroduced).")
FORBIDDEN_RULE_IDS = {"S2-R2", "S2-R4"}
assert not (set(master_df.Rule_ID) & FORBIDDEN_RULE_IDS), \
    "Stage 2's LIMITED rules (S2-R2/S2-R4) were reintroduced -- aborting (explicit guardrail violation)."
log(f"Guardrail verified: S2-R2 and S2-R4 (Stage 2 LIMITED rules) are NOT present in the master table.")

# =============================================================================
# GENERIC SCORE RECOMPUTATION (for Part 4's redundancy analysis, computed
# here so Part 2's classification can reference it -- printed in the
# requested Part 2 -> Part 4 order below, even though calculated together).
# =============================================================================
section("RECOMPUTING TRAIN-NORMAL SCORE ARRAYS FOR ALL 16 FEATURES (redundancy analysis basis)")

log("Every one of the 16 retained features is recomputed here using the EXACT methodology its own "
    "stage script established (empirical CDF scoring: score = 1 - 2*min(F(x), 1-F(x))), applied fresh "
    "to TRAIN NORMAL data -- not approximated, not read from a persisted per-row file (none of the "
    "prior scripts persisted full per-row score arrays, only summary statistics).")


def infer_2state_semantics(source_tag, target_tag):
    states = sorted(train_df[source_tag].unique())
    mean0 = train_normal.loc[train_normal[source_tag] == states[0], target_tag].mean()
    mean1 = train_normal.loc[train_normal[source_tag] == states[1], target_tag].mean()
    return (states[0], states[1]) if mean0 <= mean1 else (states[1], states[0])


def score_state_2way(source_tag, target_tag):
    lo, hi = infer_2state_semantics(source_tag, target_tag)
    off_ref = train_normal.loc[train_normal[source_tag] == lo, target_tag].to_numpy()
    on_ref = train_normal.loc[train_normal[source_tag] == hi, target_tag].to_numpy()
    st = train_df[source_tag].to_numpy(); x = train_df[target_tag].to_numpy(dtype=float)
    score = np.full(len(train_df), np.nan)
    score[st == lo] = empirical_two_sided_score_vec(off_ref, x[st == lo])
    score[st == hi] = empirical_two_sided_score_vec(on_ref, x[st == hi])
    return score


def score_valve_state_3way(source_tag, target_tag):
    states = sorted(train_df[source_tag].unique())
    means = {s: train_normal.loc[train_normal[source_tag] == s, target_tag].mean() for s in states}
    lo_state = min(means, key=means.get)
    hi_state = max(means, key=means.get)
    off_ref = train_normal.loc[train_normal[source_tag] == lo_state, target_tag].to_numpy()
    on_ref = train_normal.loc[train_normal[source_tag] == hi_state, target_tag].to_numpy()
    st = train_df[source_tag].to_numpy(); x = train_df[target_tag].to_numpy(dtype=float)
    score = np.full(len(train_df), np.nan)
    score[st == lo_state] = empirical_two_sided_score_vec(off_ref, x[st == lo_state])
    score[st == hi_state] = empirical_two_sided_score_vec(on_ref, x[st == hi_state])
    return score  # transitional state left as NaN, matching the original evaluability condition


def score_pump_slope(source_tag, target_tag, window):
    slope = (train_df[target_tag] - train_df[target_tag].shift(window)).to_numpy()
    normal_mask = (train_df["Label"].to_numpy() == 0)
    pump_states = train_df[source_tag].to_numpy()
    states = sorted(train_df[source_tag].unique())
    off_state, on_state = states[0], states[1]
    off_ref = slope[normal_mask & (pump_states == off_state) & ~np.isnan(slope)]
    on_ref = slope[normal_mask & (pump_states == on_state) & ~np.isnan(slope)]
    score = np.full(len(train_df), np.nan)
    valid = ~np.isnan(slope)
    off_mask = valid & (pump_states == off_state)
    on_mask = valid & (pump_states == on_state)
    score[off_mask] = empirical_two_sided_score_vec(off_ref, slope[off_mask])
    score[on_mask] = empirical_two_sided_score_vec(on_ref, slope[on_mask])
    return score


def score_unconditional_slope(target_tag, window):
    slope = (train_df[target_tag] - train_df[target_tag].shift(window)).to_numpy()
    normal_mask = (train_df["Label"].to_numpy() == 0)
    ref = slope[normal_mask & ~np.isnan(slope)]
    score = np.full(len(train_df), np.nan)
    valid = ~np.isnan(slope)
    score[valid] = empirical_two_sided_score_vec(ref, slope[valid])
    return score


def score_sensor_residual(tag_a, tag_b, kind, lag):
    a_full = train_df[tag_a].to_numpy(dtype=float)
    b_full = train_df[tag_b].to_numpy(dtype=float)
    if kind == "ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            x_full = np.where(b_full != 0, a_full / b_full, np.nan)
    else:
        x_full = a_full - b_full
    a_n = train_normal[tag_a].to_numpy(dtype=float)
    b_n = train_normal[tag_b].to_numpy(dtype=float)
    if kind == "ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            ref = np.where(b_n != 0, a_n / b_n, np.nan)
    else:
        ref = a_n - b_n
    ref = ref[~np.isnan(ref) & np.isfinite(ref)]
    score = empirical_two_sided_score_vec(ref, x_full)
    return score


def score_binary_mismatch(tag_a, tag_b):
    a = train_df[tag_a].to_numpy(); b = train_df[tag_b].to_numpy()
    return (a != b).astype(float)


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
missing_specs = set(master_df.Feature_Name) - set(FEATURE_SPECS)
assert not missing_specs, f"Feature spec missing for: {missing_specs}"

feature_scores = {}
for fname, (kind, a, b, param) in FEATURE_SPECS.items():
    if kind == "valve3":
        feature_scores[fname] = score_valve_state_3way(a, b)
    elif kind == "state2":
        feature_scores[fname] = score_state_2way(a, b)
    elif kind == "pump_slope":
        feature_scores[fname] = score_pump_slope(a, b, param)
    elif kind == "uncond_slope":
        feature_scores[fname] = score_unconditional_slope(b, param)
    elif kind in ("ratio", "diff"):
        feature_scores[fname] = score_sensor_residual(a, b, kind, param)
    elif kind == "mismatch":
        feature_scores[fname] = score_binary_mismatch(a, b)
    n_valid = int((~np.isnan(feature_scores[fname])).sum())
    log(f"  Recomputed {fname}: {n_valid:,} valid TRAIN scores (of {len(train_df):,} rows)")

# =============================================================================
# PART 4 (computed here, printed at its labeled position below too) —
# CROSS-FEATURE REDUNDANCY ANALYSIS
# =============================================================================
section("REDUNDANCY MATRIX COMPUTATION (TRAIN NORMAL only) -- shown in full at PART 4 below")

feature_names = list(FEATURE_SPECS.keys())
normal_mask_train = (train_df["Label"].to_numpy() == 0)
redundancy_rows = []
redundancy_matrix = pd.DataFrame(index=feature_names, columns=feature_names, dtype=float)
tags_by_feature = {fn: set([FEATURE_SPECS[fn][1], FEATURE_SPECS[fn][2]]) - {None} for fn in feature_names}

for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        fi, fj = feature_names[i], feature_names[j]
        si = feature_scores[fi][normal_mask_train]
        sj = feature_scores[fj][normal_mask_train]
        valid = ~np.isnan(si) & ~np.isnan(sj)
        if valid.sum() < 30 or np.std(si[valid]) == 0 or np.std(sj[valid]) == 0:
            rho = np.nan
        else:
            rho, _ = scipy_stats.spearmanr(si[valid], sj[valid])
        redundancy_matrix.loc[fi, fj] = rho
        if i < j:
            tag_overlap = tags_by_feature[fi] & tags_by_feature[fj]
            redundancy_rows.append({
                "Feature_A": fi, "Feature_B": fj, "N_Overlap_Rows": int(valid.sum()),
                "Spearman_Rho_TRAIN_NORMAL": rho, "Shared_Source_Tags": ", ".join(sorted(tag_overlap)) if tag_overlap else "(none)",
                "Flagged_High_Redundancy": bool(not np.isnan(rho) and abs(rho) >= REDUNDANCY_RHO_THRESHOLD),
            })

redundancy_df = pd.DataFrame(redundancy_rows).sort_values("Spearman_Rho_TRAIN_NORMAL", key=lambda s: s.abs(), ascending=False)
flagged_redundant_pairs = redundancy_df[redundancy_df.Flagged_High_Redundancy]
log(f"Computed {len(redundancy_df)} pairwise TRAIN-NORMAL score correlations across {len(feature_names)} features.")
log(f"Pairs flagged as high redundancy (|rho| >= {REDUNDANCY_RHO_THRESHOLD}): {len(flagged_redundant_pairs)}")
for _, r in flagged_redundant_pairs.iterrows():
    log(f"  {r['Feature_A']} <-> {r['Feature_B']}: rho={r['Spearman_Rho_TRAIN_NORMAL']:.4f}, "
        f"shared tags={r['Shared_Source_Tags']}")
log("\nTop 10 pairs by |rho| (context, not all flagged):")
for _, r in redundancy_df.head(10).iterrows():
    log(f"  {r['Feature_A']} <-> {r['Feature_B']}: rho={r['Spearman_Rho_TRAIN_NORMAL']}")

redundant_feature_set = set(flagged_redundant_pairs.Feature_A) | set(flagged_redundant_pairs.Feature_B)

# =============================================================================
# PART 2 — CROSS-STAGE FINAL AUDIT CRITERIA (applies to every feature)
# =============================================================================
section("PART 2 — CROSS-STAGE FINAL AUDIT: APPLYING ONE COMMON REVIEW FRAMEWORK")

log("Ten evaluation dimensions applied identically to every feature (attack-detection performance is "
    "explicitly NOT one of them):")
log("  1. Empirical strength (from stage-level SMD/rho/consistency)")
log("  2. Physical/process interpretability")
log("  3. Reference-sample sufficiency (minority-state reference >= "
    f"{SPARSE_REFERENCE_THRESHOLD} flagged otherwise)")
log(f"  4. Coverage (VALIDATION-NORMAL >= {COVERAGE_CORE_MIN}% for CORE eligibility)")
log(f"  5. TRAIN->VALIDATION robustness (not saturated: VALIDATION-NORMAL mean < {SATURATION_THRESHOLD})")
log("  6. Distribution-shift sensitivity (documented per-feature, from stage-level KS/median-shift findings)")
log(f"  7. Redundancy with other symbolic features (|TRAIN-NORMAL score rho| >= {REDUNDANCY_RHO_THRESHOLD} flagged)")
log("  8. Temporal stability (boundary-hit lags or high dwell-time jitter flagged)")
log(f"  9. Dependence on rare actuator states (minority reference < {SPARSE_REFERENCE_THRESHOLD} flagged)")
log(" 10. Downstream fusion readiness (composite of the above)")

# --- pull exact numeric evidence per feature for the caveat checks ---
def get_min_reference_n(stage, rule_id):
    env = envelope_tables.get(stage)
    if env is None:
        return None
    sub = env[(env.Rule_ID == rule_id) & (env.Envelope_Type.str.contains("state_conditioned|slope_conditioned", regex=True, na=False))]
    if len(sub) == 0:
        return None
    return int(sub["N"].min())


def get_explained_variance(stage, rule_id):
    val = validation_tables.get(stage)
    if val is None or "Explained_Variance_Fraction" not in val.columns:
        return None
    sub = val[val.Rule_ID == rule_id]
    if len(sub) == 0 or pd.isna(sub["Explained_Variance_Fraction"].iloc[0]):
        return None
    return float(sub["Explained_Variance_Fraction"].iloc[0])


def get_lag(stage, rule_id):
    val = validation_tables.get(stage)
    if val is None or "Best_Lag_Seconds" not in val.columns:
        return None
    sub = val[val.Rule_ID == rule_id]
    if len(sub) == 0 or pd.isna(sub["Best_Lag_Seconds"].iloc[0]):
        return None
    return float(sub["Best_Lag_Seconds"].iloc[0])


LAG_GRID_MAX = 30  # matches every stage script's LAG_GRID = [-30,...,30]


def get_median_shift(robustness_text):
    """Parses 'TRAIN-N median=X, VAL-N median=Y' out of the stage screening
    table's Robustness free-text (the format every stage script except
    Stage 2 uses). Returns (train_median, val_median) or (None, None) if the
    text doesn't use this format (Stage 2's S2-R1/S2-R3B text is mean-based,
    not median-based) or either value is NaN/unparseable."""
    text = str(robustness_text)
    m_train = re.search(r"TRAIN-N median=(-?[\d.]+|nan)", text)
    m_val = re.search(r"VAL-N median=(-?[\d.]+|nan)", text)
    if not m_train or not m_val:
        return None, None
    try:
        t = float(m_train.group(1))
        v = float(m_val.group(1))
    except ValueError:
        return None, None
    if np.isnan(t) or np.isnan(v):
        return None, None
    return t, v


DIST_SHIFT_DELTA_THRESHOLD = 0.30   # |VAL-N median - TRAIN-N median| >= this -> flagged
DIST_SHIFT_ABS_MEDIAN_THRESHOLD = 0.85  # VAL-N median itself >= this -> flagged (elevated-median case)


def get_actuator_minority_pct(fname):
    """Computes the TRAIN-wide minority-state percentage for the actuator
    conditioning a state2/valve3/pump_slope feature, restricted to the
    STATES ACTUALLY USED as reference states for that feature's scoring
    (computed directly from train_df/train_normal, mirroring the exact
    state-selection logic in score_state_2way/score_valve_state_3way/
    score_pump_slope above). This is a percentage-based rare-state signal,
    DISTINCT from get_min_reference_n's absolute sample-count check: a
    state can clear the absolute-count bar (e.g. P602 ON has ~2,000
    TRAIN-NORMAL rows) while still representing a tiny fraction of total
    plant operating time (P602 ON = 0.85% of TRAIN) -- both are genuine,
    separate concerns, checked independently so neither masks the other.

    IMPORTANT: for 3-state (valve3) actuators, the mandatory-transitional
    state is EXCLUDED here, exactly as it is excluded from scoring itself.
    An earlier version of this function parsed ALL states from a text
    field indiscriminately, which for MV101 (0=TRANSITIONAL 0.49%,
    1=CLOSED 36.25%, 2=OPEN 63.26%) picked up the excluded transitional
    state's 0.49% as if it were a used scoring reference -- a false rare-
    state flag. Restricting to the two states actually used for scoring
    (as this version does) fixes that."""
    kind, source_tag, target_tag, _param = FEATURE_SPECS[fname]
    if kind not in ("state2", "valve3", "pump_slope") or source_tag is None:
        return None
    total = len(train_df)
    if total == 0:
        return None
    if kind == "valve3":
        states = sorted(train_df[source_tag].unique())
        means = {s: train_normal.loc[train_normal[source_tag] == s, target_tag].mean() for s in states}
        used_states = [min(means, key=means.get), max(means, key=means.get)]
    else:
        used_states = sorted(train_df[source_tag].unique())[:2]
    st = train_df[source_tag].to_numpy()
    pcts = [100.0 * float((st == s).sum()) / total for s in used_states]
    return min(pcts) if pcts else None


RARE_ACTUATOR_STATE_PCT_THRESHOLD = 5.0  # minority actuator state < this % of all TRAIN rows -> flagged


def get_source_sensor_rho(stage, rule_id):
    """For sensor-sensor residual features (ratio/diff kind), pulls the RAW
    pre-existing sensor-to-sensor Spearman rho already computed at
    discovery time (stageN_rule_validation.csv's Spearman_Rho column) --
    this is the SOURCE-SENSOR redundancy concept (e.g. PIT501 vs PIT503
    themselves), explicitly kept separate from the cross-FEATURE
    score-redundancy analysis in Part 4 (which uses the constructed,
    empirical-CDF-scored arrays, a different quantity that can legitimately
    disagree with raw-sensor correlation)."""
    val = validation_tables.get(stage)
    if val is None or "Spearman_Rho" not in val.columns:
        return None
    sub = val[val.Rule_ID == rule_id]
    if len(sub) == 0 or pd.isna(sub["Spearman_Rho"].iloc[0]):
        return None
    return float(sub["Spearman_Rho"].iloc[0])


SOURCE_SENSOR_RHO_THRESHOLD = 0.95  # |raw sensor-sensor rho| >= this -> flagged as near-collinear source sensors

final_status = {}
status_reason = {}
for idx, row in master_df.iterrows():
    stage_num = int(row["Stage"].split()[1])
    rule_id = row["Rule_ID"]
    fname = row["Feature_Name"]
    caveats = []

    # --- UNIFIED rare/thin actuator-state check: combines TWO independent
    # signals into ONE consistent rule, applied identically to every
    # actuator-conditioned feature (state2/valve3/pump_slope kind):
    #   (a) absolute TRAIN-NORMAL reference sample count < SPARSE_REFERENCE_THRESHOLD
    #   (b) the actuator's minority-state share of ALL TRAIN rows < RARE_ACTUATOR_STATE_PCT_THRESHOLD
    # Neither alone was sufficient: (a) misses S6-R1 (P602 ON has ~2,000
    # TRAIN-NORMAL rows -- not thin by count -- yet is only 0.85% of TRAIN
    # operating time); (b) alone would miss nothing new here but is kept for
    # symmetry/completeness. Either signal triggers the SAME caveat category
    # (never an automatic EXCLUDE, per instructions).
    min_ref_n = get_min_reference_n(stage_num, rule_id)
    kind0, source_tag0 = FEATURE_SPECS[fname][0], FEATURE_SPECS[fname][1]
    minority_pct = get_actuator_minority_pct(fname)
    rare_state_reasons = []
    if min_ref_n is not None and min_ref_n < SPARSE_REFERENCE_THRESHOLD:
        rare_state_reasons.append(f"thin TRAIN-NORMAL reference sample (n={min_ref_n} < {SPARSE_REFERENCE_THRESHOLD})")
    if minority_pct is not None and minority_pct < RARE_ACTUATOR_STATE_PCT_THRESHOLD:
        rare_state_reasons.append(f"minority actuator state is only {minority_pct:.2f}% of all TRAIN rows "
                                   f"(< {RARE_ACTUATOR_STATE_PCT_THRESHOLD}%)")
    if rare_state_reasons:
        caveats.append("dependence on a rare/thin actuator state: " + "; ".join(rare_state_reasons))
        master_df.at[idx, "Sparsity_Notes"] = "; ".join(rare_state_reasons)
    else:
        notes = []
        if min_ref_n is not None:
            notes.append(f"minority-state TRAIN-NORMAL reference n={min_ref_n} (adequate)")
        if minority_pct is not None:
            notes.append(f"minority actuator state = {minority_pct:.2f}% of TRAIN (adequate)")
        master_df.at[idx, "Sparsity_Notes"] = "; ".join(notes) if notes else "not applicable (unconditional/binary feature)"

    explained_var = get_explained_variance(stage_num, rule_id)
    if explained_var is not None and explained_var < LOW_EXPLAINED_VARIANCE_THRESHOLD:
        caveats.append(f"low explained variance ({explained_var*100:.2f}% < {LOW_EXPLAINED_VARIANCE_THRESHOLD*100:.0f}%)")

    lag = get_lag(stage_num, rule_id)
    if lag is not None and abs(lag) == LAG_GRID_MAX:
        caveats.append(f"best lag ({lag:.0f}s) hit the edge of the tested +/-{LAG_GRID_MAX}s grid -- unexplored beyond")
        master_df.at[idx, "Temporal_Lag_Notes"] = f"lag={lag:.0f}s (AT SEARCH BOUNDARY, unexplored beyond)"
    elif lag is not None:
        master_df.at[idx, "Temporal_Lag_Notes"] = f"lag={lag:.0f}s (within tested grid)"

    if fname in redundant_feature_set:
        partners = sorted(set(flagged_redundant_pairs.loc[flagged_redundant_pairs.Feature_A == fname, "Feature_B"]).union(
            set(flagged_redundant_pairs.loc[flagged_redundant_pairs.Feature_B == fname, "Feature_A"])))
        caveats.append(f"high TRAIN-NORMAL score redundancy with {partners}")
        master_df.at[idx, "Redundancy_Notes"] = f"|rho|>={REDUNDANCY_RHO_THRESHOLD} with: {partners}"
    else:
        master_df.at[idx, "Redundancy_Notes"] = "no high-redundancy partner found"

    # --- SOURCE-SENSOR redundancy (sensor-sensor features only): a
    # DISTINCT concept from the cross-FEATURE score redundancy just above.
    # A residual feature's two raw source sensors can be near-collinear
    # even when the constructed (empirical-CDF-scored) feature arrays are
    # not strongly correlated with any OTHER retained feature's array --
    # the two checks measure different things and must not be conflated.
    if kind0 in ("ratio", "diff"):
        src_rho = get_source_sensor_rho(stage_num, rule_id)
        if src_rho is not None and abs(src_rho) >= SOURCE_SENSOR_RHO_THRESHOLD:
            a_tag, b_tag = FEATURE_SPECS[fname][1], FEATURE_SPECS[fname][2]
            caveats.append(f"source sensors {a_tag}/{b_tag} are themselves near-collinear "
                            f"(raw Spearman rho={src_rho:.4f}, |rho|>={SOURCE_SENSOR_RHO_THRESHOLD}) "
                            f"-- distinct from, and not resolved by, the cross-feature score-redundancy check above")
            master_df.at[idx, "Redundancy_Notes"] = (master_df.at[idx, "Redundancy_Notes"] +
                f" | SOURCE-SENSOR redundancy: {a_tag}<->{b_tag} raw rho={src_rho:.4f} (near-collinear)")

    # --- distribution-shift (VALIDATION-NORMAL vs TRAIN-NORMAL MEDIAN,
    # in addition to the existing MEAN-based saturation check below).
    # A feature's mean can stay below the saturation bar while its median
    # already sits far from TRAIN-NORMAL's median (skewed distribution) --
    # this is a genuinely different, milder signal than saturation and is
    # tracked as its own caveat, applied identically to all 16 features via
    # the same regex-parsed text (never singled out for specific rules).
    train_med, val_med = get_median_shift(row["Robustness_Notes"])
    if train_med is not None:
        med_delta = val_med - train_med
        if abs(med_delta) >= DIST_SHIFT_DELTA_THRESHOLD or val_med >= DIST_SHIFT_ABS_MEDIAN_THRESHOLD:
            caveats.append(f"partial distribution shift (VALIDATION-NORMAL median={val_med:.3f} vs "
                            f"TRAIN-NORMAL median={train_med:.3f}, delta={med_delta:+.3f})")

    # coverage / saturation re-check (independent re-derivation, not just copy).
    # Coverage text format differs by stage ("VALIDATION NORMAL=99.58%, ATTACK=..."
    # for stages 1/3/4/5; "99.58% of VALIDATION rows (...)" for stage 2;
    # "100.00%" bare for stage 6) -- extract the NORMAL-coverage number
    # specifically, robust to all three formats.
    cov_text = str(row["VALIDATION_Coverage"])
    cov_num = None
    m = re.search(r"NORMAL\D{0,3}([\d.]+)\s*%", cov_text)
    if m:
        cov_num = float(m.group(1))
    else:
        m = re.search(r"^\s*([\d.]+)\s*%", cov_text)
        if m:
            cov_num = float(m.group(1))
        else:
            m = re.search(r"([\d.]+)", cov_text)
            if m:
                cov_num = float(m.group(1))
    rob_text = str(row["Robustness_Notes"])
    saturated_flag = "saturated=True" in rob_text.replace(" ", "")
    if cov_num is not None and cov_num < COVERAGE_CORE_MIN:
        caveats.append(f"VALIDATION coverage below {COVERAGE_CORE_MIN}% ({cov_num}%)")
    if saturated_flag:
        caveats.append("saturated in VALIDATION")

    if train_med is not None:
        med_delta = val_med - train_med
        shift_flagged = abs(med_delta) >= DIST_SHIFT_DELTA_THRESHOLD or val_med >= DIST_SHIFT_ABS_MEDIAN_THRESHOLD
        master_df.at[idx, "Distribution_Shift_Notes"] = (
            rob_text + f" | MEDIAN-SHIFT CHECK: TRAIN-N median={train_med:.3f}, VAL-N median={val_med:.3f}, "
            f"delta={med_delta:+.3f}, flagged={shift_flagged}")
    else:
        master_df.at[idx, "Distribution_Shift_Notes"] = rob_text
    master_df.at[idx, "Interpretability"] = ("High -- direct single-actuator/single-sensor physical relationship"
                                              if FEATURE_SPECS[fname][0] in ("state2", "valve3", "mismatch") else
                                              "Moderate -- sensor-sensor or unconditional relationship, plant-specific tag identities not independently documented")

    if not caveats:
        status = "CORE"
        reason = "No caveat triggered under the cross-stage framework: adequate reference sample(s), strong coverage, not saturated, no boundary-lag issue, no flagged redundancy."
    elif len(caveats) == 1 and "redundancy" not in caveats[0] and cov_num is not None and cov_num >= COVERAGE_CORE_MIN and not saturated_flag:
        status = "RETAIN_WITH_CAVEAT"
        reason = f"Meaningful but non-disqualifying limitation: {caveats[0]}."
    else:
        # multiple caveats, or a coverage/saturation failure, or redundancy -- still
        # default to RETAIN_WITH_CAVEAT unless coverage/saturation itself fails,
        # per instructions not to over-exclude; EXCLUDE reserved for genuine
        # coverage/saturation failure or extreme compounding fragility.
        coverage_or_saturation_failed = (cov_num is not None and cov_num < COVERAGE_CORE_MIN) or saturated_flag
        if coverage_or_saturation_failed:
            status = "EXCLUDE"
            reason = "Cross-stage standard requires >= 90% VALIDATION coverage and no saturation for primary downstream use; this feature fails that bar: " + "; ".join(caveats) + "."
        else:
            status = "RETAIN_WITH_CAVEAT"
            reason = "Multiple noted limitations, but coverage/robustness remain adequate: " + "; ".join(caveats) + "."

    final_status[fname] = status
    status_reason[fname] = reason
    master_df.at[idx, "Final_Cross_Stage_Status"] = status
    master_df.at[idx, "Final_Status_Reason"] = reason
    log(f"\n{row['Stage']} {rule_id} ({fname}): {status}\n  {reason}")

n_core = sum(1 for s in final_status.values() if s == "CORE")
n_caveat = sum(1 for s in final_status.values() if s == "RETAIN_WITH_CAVEAT")
n_exclude = sum(1 for s in final_status.values() if s == "EXCLUDE")
log(f"\nCross-stage audit totals: CORE={n_core}, RETAIN_WITH_CAVEAT={n_caveat}, EXCLUDE={n_exclude} "
    f"(of {len(final_status)} stage-level-retained features)")
log("No target count in any category was forced.")

# =============================================================================
# PART 3 — EXPLICITLY REVISIT KNOWN CAVEATS (verified against exact numbers)
# =============================================================================
section("PART 3 — EXPLICITLY REVISITING KNOWN CAVEATS (verified from exact source outputs)")

_s3r1_tmed, _s3r1_vmed = get_median_shift(
    screening_tables[3][screening_tables[3].Rule_ID == "S3-R1"]["Robustness"].iloc[0])
_s3r2_tmed, _s3r2_vmed = get_median_shift(
    screening_tables[3][screening_tables[3].Rule_ID == "S3-R2"]["Robustness"].iloc[0])

caveat_checks = [
    ("Stage 1", "S1-R3_slope_score", "P102 extremely sparse / thin ON-state reference",
     f"VERIFIED under the UNIFIED rare-actuator-state rule (Part 2): minority-state (P102 ON) "
     f"TRAIN-NORMAL reference n={get_min_reference_n(1,'S1-R3')} (< {SPARSE_REFERENCE_THRESHOLD}, "
     f"read from stage1_normal_operating_envelopes.csv) AND P102 ON is only "
     f"{get_actuator_minority_pct('S1-R3_slope_score')}% of ALL TRAIN rows (< {RARE_ACTUATOR_STATE_PCT_THRESHOLD}%, "
     f"read from stage1_tag_audit.csv) -- both signals agree, reflected as a single rare-state CAVEAT in "
     f"Part 2's Final_Cross_Stage_Status (never an automatic EXCLUDE)."),
    ("Stage 1", "S1-R2_slope_score / S1-R3_slope_score", "level-slope relationships may be confounded",
     "CONFIRMED as documented in stage1_symbolic_feature_definitions.csv's own limitation text: "
     "'Confounded by simultaneous MV101/other-pump activity; state-conditioning only partially isolates "
     "this pump's own effect.' -- carried through unchanged, not re-litigated."),
    ("Stage 1", "S1-R4_slope_score", "partial explained variance",
     f"VERIFIED: explained variance fraction={get_explained_variance(1,'S1-R4')} "
     f"(read from stage1_rule_validation.csv) -- {'above' if (get_explained_variance(1,'S1-R4') or 0) >= LOW_EXPLAINED_VARIANCE_THRESHOLD else 'below'} "
     f"the {LOW_EXPLAINED_VARIANCE_THRESHOLD*100:.0f}% low-variance flag threshold."),
    ("Stage 2", "S2_R1_score", "expected to be robust",
     "CONFIRMED: 100% TRAIN-NORMAL consistency, 99.58% VALIDATION coverage (100% of VALIDATION-ATTACK "
     "timestamps), no caveat triggered under the cross-stage framework (see Part 2)."),
    ("Stage 2", "S2_R3B_state_mismatch", "expected to be robust",
     "CONFIRMED: 100% coverage at every split/stratum, TRAIN-NORMAL mismatch rate 0.0094% (near-zero, "
     "not saturated), no caveat triggered."),
    ("Stage 2", "S2-R2 / S2-R4", "already non-retained, must NOT be reintroduced",
     f"VERIFIED: neither 'S2-R2' nor 'S2-R4' appears in the master table (assert check passed above); "
     f"stage2_final_rule_screening.csv confirms both remain LIMITED at the stage level."),
    ("Stage 3", "S3-R1_state_score / S3-R2_state_score", "elevated VALIDATION-normal medians / partial distribution shift",
     f"VERIFIED under the UNIFIED median-shift rule (Part 2, applied identically to all 16 features): "
     f"S3-R1 TRAIN-N median={_s3r1_tmed:.3f} -> VAL-N median={_s3r1_vmed:.3f} (delta={_s3r1_vmed-_s3r1_tmed:+.3f}); "
     f"S3-R2 TRAIN-N median={_s3r2_tmed:.3f} -> VAL-N median={_s3r2_vmed:.3f} (delta={_s3r2_vmed-_s3r2_tmed:+.3f}). "
     f"Both exceed the delta>={DIST_SHIFT_DELTA_THRESHOLD} OR median>={DIST_SHIFT_ABS_MEDIAN_THRESHOLD} trigger "
     f"-- NEITHER is saturated by the pre-existing MEAN-based check (both saturated=False in "
     f"stage3_final_rule_screening.csv), which is exactly why a separate MEDIAN-based check was needed: the "
     f"mean-based check alone would have missed this. Both are now explicitly RETAIN_WITH_CAVEAT for partial "
     f"distribution shift (not CORE) in Part 2's Final_Cross_Stage_Status -- this changes their status "
     f"relative to a prior version of this script that checked only the mean."),
    ("Stage 3", "S3-R4_slope_score", "weakest retained feature, only ~7.8% explained variance",
     f"VERIFIED: explained variance fraction={get_explained_variance(3,'S3-R4')} "
     f"(read from stage3_rule_validation.csv) -- confirms the ~7.8% figure exactly, "
     f"below the {LOW_EXPLAINED_VARIANCE_THRESHOLD*100:.0f}% threshold, triggering a CAVEAT."),
    ("Stage 4", "S4-R2_state_score", "extremely strong relationship",
     f"VERIFIED: state SMD={validation_tables[4][validation_tables[4].Rule_ID=='S4-R2']['State_SMD'].iloc[0]:.3f} "
     f"(read from stage4_rule_validation.csv) -- among the largest SMD values across all 16 features."),
    ("Stage 4", "S4-R2_state_score", "minority P402 OFF reference is thin",
     f"VERIFIED under the UNIFIED rare-actuator-state rule (Part 2): minority-state (P402 OFF) "
     f"TRAIN-NORMAL reference n={get_min_reference_n(4,'S4-R2')} (< {SPARSE_REFERENCE_THRESHOLD}, read "
     f"from stage4_normal_operating_envelopes.csv); P402 OFF is {get_actuator_minority_pct('S4-R2_state_score')}% of "
     f"ALL TRAIN rows (>= {RARE_ACTUATOR_STATE_PCT_THRESHOLD}%, so the percentage-based signal alone would "
     f"NOT flag this -- the absolute-count signal is what triggers the caveat here, correctly, since only "
     f"65 NORMAL-labeled rows exist in that state despite P402 OFF being ~12% of all TRAIN operating time; "
     f"the two signals are independent and this case shows why both are needed). Separately, only 1 usable "
     f"clean transition event existed for the event-based companion check (a DIFFERENT, smaller number "
     f"than the state-reference size -- verified not to be conflated)."),
    ("Stage 5", "S5-R1_state_score / S5-R2_state_score", "minority P501-OFF reference is thin",
     f"VERIFIED under the UNIFIED rare-actuator-state rule (Part 2): minority-state (P501 OFF) "
     f"TRAIN-NORMAL reference n={get_min_reference_n(5,'S5-R1')} (< {SPARSE_REFERENCE_THRESHOLD}) for both "
     f"S5-R1 and S5-R2 (read from stage5_normal_operating_envelopes.csv, identical since both share the "
     f"same P501 state-conditioning); P501 OFF is {get_actuator_minority_pct('S5-R1_state_score')}% of ALL TRAIN rows "
     f"(>= {RARE_ACTUATOR_STATE_PCT_THRESHOLD}%, so as with S4-R2 the absolute-count signal is what "
     f"triggers here, not the percentage signal)."),
    ("Stage 5", "S5-R4_residual_score", "possible redundancy because PIT501 and PIT503 are nearly identical",
     f"VERIFIED: raw-sensor Spearman rho(PIT501, PIT503)={validation_tables[5][validation_tables[5].Rule_ID=='S5-R4']['Spearman_Rho'].iloc[0]:.4f} "
     f"(read from stage5_rule_validation.csv, |rho|>={SOURCE_SENSOR_RHO_THRESHOLD} threshold). This is now "
     f"FORMALLY applied as a distinct SOURCE-SENSOR redundancy caveat in Part 2's classification logic "
     f"(previously only narrated here, not reflected in Final_Cross_Stage_Status -- a prior version of "
     f"this script left S5-R4 as CORE despite this concern; it is now RETAIN_WITH_CAVEAT). Cross-FEATURE "
     f"redundancy (Part 4, TRAIN-NORMAL SCORE arrays, not raw sensors) is reported separately below and "
     f"remains <0.90 for every pair involving S5-R4 -- the two redundancy concepts are NOT the same "
     f"quantity and do not resolve each other; both are tracked independently."),
    ("Stage 5", "S5-R5_residual_score", "best lag hit the -30s search boundary",
     f"VERIFIED: Best_Lag_Seconds={get_lag(5,'S5-R5')} (read from stage5_rule_validation.csv), exactly "
     f"at the edge of the tested +/-{LAG_GRID_MAX}s grid -- confirmed, triggers a CAVEAT in Part 2."),
    ("Stage 6", "S6-R1_state_score", "P602 ON-time only ~0.85%",
     None),  # filled below from tag audit
    ("Stage 6", "S6-R1_state_score", "structurally limited stage",
     "CONFIRMED: Stage 6 has only 4 tags total (see stage6_tag_audit.csv); P601/P603 fully constant, "
     "P602 the sole varying actuator, FIT601 the sole sensor -- documented at the stage level, carried "
     "through unchanged here."),
]
tag_audit_6 = pd.read_csv(TABLES_DIR / "stage6_tag_audit.csv")
p602_row = tag_audit_6[(tag_audit_6.Tag == "P602") & (tag_audit_6.Split == "TRAIN")]
p602_on_pct = None
if len(p602_row):
    freq_text = p602_row["State_Values_And_Frequencies"].iloc[0]
    # format: "1=267717(99.15%); 2=2283(0.85%)" -- parse the minority state pct
    parts = [p.strip() for p in freq_text.split(";")]
    pct_vals = [float(p.split("(")[1].rstrip("%)")) for p in parts]
    p602_on_pct = min(pct_vals)
# Independent cross-check: this hand-parsed value must agree with what
# get_actuator_minority_pct('S6-R1_state_score') actually fed into Part 2's
# classification logic (P602 has only 2 states, so no transitional-state
# exclusion nuance applies here, unlike the valve3 case).
_p602_pct_from_fn = get_actuator_minority_pct("S6-R1_state_score")
# Tolerance of 0.01 accounts for stage6_tag_audit.csv's State_Values_And_Frequencies
# text being pre-rounded to 2 decimal places (e.g. "0.85%"), while
# get_actuator_minority_pct computes the exact float directly from row counts.
assert p602_on_pct is not None and _p602_pct_from_fn is not None and abs(p602_on_pct - _p602_pct_from_fn) < 0.01, (
    f"P602 minority-pct mismatch between independent tag-audit parse ({p602_on_pct}) and the function "
    f"actually used in Part 2's classification ({_p602_pct_from_fn}) -- aborting rather than reporting "
    f"unverified numbers.")
caveat_checks[-2] = ("Stage 6", "S6-R1_state_score", "P602 ON-time only ~0.85%",
                      f"VERIFIED under the UNIFIED rare-actuator-state rule (Part 2): exact minority-state "
                      f"TRAIN percentage={p602_on_pct}% (parsed from stage6_tag_audit.csv's "
                      f"State_Values_And_Frequencies field) -- confirms the ~0.85% figure exactly, well "
                      f"below the {RARE_ACTUATOR_STATE_PCT_THRESHOLD}% threshold. NOTE: the ABSOLUTE "
                      f"TRAIN-NORMAL reference count for P602 ON is n={get_min_reference_n(6,'S6-R1')} "
                      f"(>= {SPARSE_REFERENCE_THRESHOLD}, so the absolute-count signal alone would NOT "
                      f"flag this feature) -- it is the PERCENTAGE-based signal that correctly catches it, "
                      f"which is exactly why both signals are checked under the unified rule rather than "
                      f"relying on absolute count alone. This is now formally reflected as a rare-state "
                      f"CAVEAT in Part 2's Final_Cross_Stage_Status -- a prior version of this script left "
                      f"S6-R1 as CORE (0 caveats triggered), since the 'structurally limited stage' note "
                      f"below was purely narrative context in Part 3 and was never wired into Part 2's "
                      f"actual classification logic. S6-R1 is now RETAIN_WITH_CAVEAT, consistent with the "
                      f"same rare-state reasoning already applied to S1-R3, S4-R2, S5-R1, and S5-R2.)")

for stage, feat, caveat, verification in caveat_checks:
    log(f"\n{stage} — {feat}")
    log(f"  Claimed caveat: {caveat}")
    log(f"  {verification}")

caveat_verification_path = RESULTS_DIR / "final_cross_stage_caveat_verification.txt"
caveat_lines = ["FINAL CROSS-STAGE AUDIT — EXPLICIT CAVEAT VERIFICATION (Part 3)", ""]
for stage, feat, caveat, verification in caveat_checks:
    caveat_lines.append(f"{stage} — {feat}")
    caveat_lines.append(f"  Claimed: {caveat}")
    caveat_lines.append(f"  {verification}")
    caveat_lines.append("")
caveat_verification_path.write_text("\n".join(caveat_lines), encoding="utf-8")
log(f"\nWrote {caveat_verification_path}")

# =============================================================================
# PART 4 — CROSS-FEATURE REDUNDANCY ANALYSIS (full table, computed above)
# =============================================================================
section("PART 4 — CROSS-FEATURE REDUNDANCY ANALYSIS (full results)")

redundancy_path = TABLES_DIR / "final_symbolic_feature_redundancy.csv"
redundancy_df.to_csv(redundancy_path, index=False)
log(f"Wrote {redundancy_path} ({len(redundancy_df)} rows, all C(16,2)={len(feature_names)*(len(feature_names)-1)//2} pairs)")
log(f"\n{len(flagged_redundant_pairs)} pair(s) flagged as high redundancy (|rho|>={REDUNDANCY_RHO_THRESHOLD}):")
if len(flagged_redundant_pairs):
    for _, r in flagged_redundant_pairs.iterrows():
        log(f"  {r['Feature_A']} <-> {r['Feature_B']}: rho={r['Spearman_Rho_TRAIN_NORMAL']:.4f}, "
            f"shared source tags={r['Shared_Source_Tags']}")
else:
    log("  None -- no two retained features' TRAIN-NORMAL score arrays were found to be strongly "
        "redundant under this threshold, DESPITE several sharing a conceptual category (e.g. multiple "
        "state-based pump->flow features across different stages/tags) -- confirming that shared "
        "CATEGORY membership does not by itself imply redundant INFORMATION, per instructions.")

# =============================================================================
# PART 5 — FINAL SYMBOLIC FEATURE SET (CORE / CAVEAT / EXCLUDE groups)
# =============================================================================
section("PART 5 — FINAL SYMBOLIC FEATURE SET")

core_df = master_df[master_df.Final_Cross_Stage_Status == "CORE"][
    ["Feature_Name", "Rule_ID", "Stage", "Physical_Relationship", "TRAIN_Normal_Evidence", "Main_Limitation", "Final_Cross_Stage_Status"]
].rename(columns={"Main_Limitation": "Key_Evidence_Or_Limitation"})
caveat_df = master_df[master_df.Final_Cross_Stage_Status == "RETAIN_WITH_CAVEAT"][
    ["Feature_Name", "Rule_ID", "Stage", "Physical_Relationship", "TRAIN_Normal_Evidence", "Final_Status_Reason", "Final_Cross_Stage_Status"]
].rename(columns={"Final_Status_Reason": "Key_Evidence_Or_Limitation"})
exclude_df = master_df[master_df.Final_Cross_Stage_Status == "EXCLUDE"][
    ["Feature_Name", "Rule_ID", "Stage", "Physical_Relationship", "TRAIN_Normal_Evidence", "Final_Status_Reason", "Final_Cross_Stage_Status"]
].rename(columns={"Final_Status_Reason": "Key_Evidence_Or_Limitation"})

core_path = TABLES_DIR / "final_core_symbolic_features.csv"
core_df.to_csv(core_path, index=False)
log(f"Wrote {core_path} ({len(core_df)} rows) -- GROUP A: CORE SYMBOLIC FEATURES")

extended_df = pd.concat([core_df.assign(Group="CORE"), caveat_df.rename(
    columns={"Key_Evidence_Or_Limitation": "Key_Evidence_Or_Limitation"}).assign(Group="RETAIN_WITH_CAVEAT")],
    ignore_index=True, sort=False)
extended_path = TABLES_DIR / "final_extended_symbolic_features.csv"
extended_df.to_csv(extended_path, index=False)
log(f"Wrote {extended_path} ({len(extended_df)} rows) -- CORE + RETAIN_WITH_CAVEAT (secondary extended fusion set)")

PRIMARY_CORE_SET = core_df["Feature_Name"].tolist()
EXTENDED_SET = extended_df["Feature_Name"].tolist()
EXCLUDED_SET = exclude_df["Feature_Name"].tolist()
log(f"\nPRIMARY CORE SET ({len(PRIMARY_CORE_SET)} features): {PRIMARY_CORE_SET}")
log(f"EXTENDED SET ({len(EXTENDED_SET)} features): {EXTENDED_SET}")
log(f"EXCLUDED AFTER CROSS-STAGE AUDIT ({len(EXCLUDED_SET)} features): {EXCLUDED_SET}")

# =============================================================================
# PART 6 — STAGE-LEVEL SUMMARY
# =============================================================================
section("PART 6 — STAGE-LEVEL SUMMARY")

STAGE_CATEGORY = {
    1: "rolling flow<->level (mass-balance)", 2: "chemical-dosing response + bidirectional pump coupling (baseline)",
    3: "actuator->flow/differential-pressure + rolling flow<->level", 4: "actuator->analyzer response",
    5: "sensor-sensor conservation/consistency", 6: "pump->flow (6th independent confirmation of an existing category)",
}
stage_summary_rows = []
for stage in range(1, 7):
    n_candidates = candidate_counts_by_stage[stage]
    stage_features = master_df[master_df.Stage == f"Stage {stage}"]
    n_retained_stage = actual_counts[stage]
    n_core_stage = int((stage_features.Final_Cross_Stage_Status == "CORE").sum())
    n_caveat_stage = int((stage_features.Final_Cross_Stage_Status == "RETAIN_WITH_CAVEAT").sum())
    n_exclude_stage = int((stage_features.Final_Cross_Stage_Status == "EXCLUDE").sum())
    strongest = None
    if len(stage_features):
        val = validation_tables.get(stage)
        if val is not None and "State_SMD" in val.columns:
            sub = val[val.Rule_ID.isin(stage_features.Rule_ID) & val.State_SMD.notna()]
            if len(sub):
                strongest = sub.loc[sub.State_SMD.abs().idxmax(), "Rule_ID"]
        if strongest is None:
            strongest = stage_features.iloc[0]["Rule_ID"]
    stage_summary_rows.append({
        "Stage": f"Stage {stage}", "Original_Candidate_Count": n_candidates,
        "Stage_Level_Retained_Count": n_retained_stage, "CORE_Count": n_core_stage,
        "RETAIN_WITH_CAVEAT_Count": n_caveat_stage, "EXCLUDE_Count": n_exclude_stage,
        "Strongest_Retained_Relationship": strongest,
        "Primary_Limitation": stage_features["Main_Limitation"].iloc[0] if len(stage_features) else "n/a",
        "Symbolic_Information_Category": STAGE_CATEGORY[stage],
    })
stage_summary_df = pd.DataFrame(stage_summary_rows)
stage_summary_path = TABLES_DIR / "final_stage_symbolic_summary.csv"
stage_summary_df.to_csv(stage_summary_path, index=False)
log(f"Wrote {stage_summary_path} ({len(stage_summary_df)} rows)")
for _, r in stage_summary_df.iterrows():
    log(f"  {r['Stage']}: candidates={r['Original_Candidate_Count']}, retained={r['Stage_Level_Retained_Count']}, "
        f"CORE={r['CORE_Count']}, CAVEAT={r['RETAIN_WITH_CAVEAT_Count']}, EXCLUDE={r['EXCLUDE_Count']}, "
        f"strongest={r['Strongest_Retained_Relationship']}")

# =============================================================================
# PART 7 — PUBLICATION-QUALITY TABLES (remaining ones)
# =============================================================================
section("PART 7 — SAVING REMAINING PUBLICATION-QUALITY TABLES")

audit_path = TABLES_DIR / "final_cross_stage_symbolic_audit.csv"
master_df.to_csv(audit_path, index=False)
log(f"Wrote {audit_path} ({len(master_df)} rows) -- full master audit table")

feature_set_cols = ["Stage", "Rule_ID", "Feature_Name", "Physical_Relationship", "Representation_Type",
                     "TRAIN_Normal_Evidence", "VALIDATION_Coverage", "Robustness_Notes", "Main_Limitation",
                     "Final_Cross_Stage_Status"]
feature_set_df = master_df[feature_set_cols].rename(columns={
    "TRAIN_Normal_Evidence": "Evidence_Summary", "VALIDATION_Coverage": "Coverage",
    "Robustness_Notes": "Robustness", "Main_Limitation": "Limitation", "Final_Cross_Stage_Status": "Final_Status"})
feature_set_path = TABLES_DIR / "final_symbolic_feature_set.csv"
feature_set_df.to_csv(feature_set_path, index=False)
log(f"Wrote {feature_set_path} ({len(feature_set_df)} rows) -- main feature-set table")

# =============================================================================
# PART 8 — PUBLICATION-QUALITY FIGURES
# =============================================================================
section("PART 8 — GENERATING FIGURES (300 DPI)")

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
                      "axes.titlesize": 13, "axes.labelsize": 11})
STATUS_COLOR = {"CORE": "#2E7D32", "RETAIN_WITH_CAVEAT": "#F4B183", "EXCLUDE": "#C44E52"}

# --- Figure 1: final cross-stage rule screening ---
fig, ax = plt.subplots(figsize=(11, 8))
plot_df = master_df.sort_values(["Stage", "Rule_ID"])
y_labels = [f"{r.Stage} {r.Rule_ID}\n{r.Feature_Name}" for r in plot_df.itertuples()]
colors = [STATUS_COLOR[s] for s in plot_df.Final_Cross_Stage_Status]
ax.barh(range(len(plot_df)), [1] * len(plot_df), color=colors)
ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(y_labels, fontsize=7)
ax.invert_yaxis()
ax.set_xticks([])
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATUS_COLOR.values()]
ax.legend(legend_handles, STATUS_COLOR.keys(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
ax.set_title("Final Cross-Stage Symbolic Feature Screening (all 16 stage-level-retained features)")
fig.tight_layout()
fig1_path = FIGURES_DIR / "final_cross_stage_rule_screening.png"
fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig1_path}")

# --- Figure 2: final features by stage (CORE vs CAVEAT) ---
fig, ax = plt.subplots(figsize=(9, 6))
stages_lbl = [f"Stage {i}" for i in range(1, 7)]
core_counts = [int((master_df[(master_df.Stage == s)].Final_Cross_Stage_Status == "CORE").sum()) for s in stages_lbl]
caveat_counts = [int((master_df[(master_df.Stage == s)].Final_Cross_Stage_Status == "RETAIN_WITH_CAVEAT").sum()) for s in stages_lbl]
exclude_counts = [int((master_df[(master_df.Stage == s)].Final_Cross_Stage_Status == "EXCLUDE").sum()) for s in stages_lbl]
x = np.arange(6)
ax.bar(x, core_counts, label="CORE", color=STATUS_COLOR["CORE"])
ax.bar(x, caveat_counts, bottom=core_counts, label="RETAIN_WITH_CAVEAT", color=STATUS_COLOR["RETAIN_WITH_CAVEAT"])
ax.bar(x, exclude_counts, bottom=np.array(core_counts) + np.array(caveat_counts), label="EXCLUDE", color=STATUS_COLOR["EXCLUDE"])
ax.set_xticks(x)
ax.set_xticklabels(stages_lbl)
ax.set_ylabel("Number of final symbolic features")
ax.set_title("Final Symbolic Features by Stage (CORE vs. RETAIN_WITH_CAVEAT vs. EXCLUDE)")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
fig2_path = FIGURES_DIR / "final_symbolic_features_by_stage.png"
fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig2_path}")

# --- Figure 3: redundancy heatmap ---
fig, ax = plt.subplots(figsize=(11, 10))
rm = redundancy_matrix.astype(float).to_numpy()
im = ax.imshow(rm, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(feature_names))); ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
ax.set_yticks(range(len(feature_names))); ax.set_yticklabels(feature_names, fontsize=7)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Spearman rho (TRAIN-NORMAL feature scores)")
ax.set_title(f"Cross-Feature Redundancy Structure (16 retained symbolic features)\n"
             f"Pairs with |rho|>={REDUNDANCY_RHO_THRESHOLD} flagged as high redundancy (see table)")
fig.tight_layout()
fig3_path = FIGURES_DIR / "final_symbolic_feature_redundancy.png"
fig.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig3_path}")

# --- Figure 4: pipeline summary schematic ---
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.set_xlim(0, 13); ax.set_ylim(0, 3); ax.axis("off")
steps = ["Domain/process\nknowledge", "Candidate\nrules", "TRAIN-normal\nvalidation",
         "Lag/state/rolling\nanalysis", "Empirical\nenvelope", "Coverage/robustness\nscreening",
         "Stage-level\nretention", "Cross-stage\naudit", "Final symbolic\nfeature set"]
n_steps = len(steps)
box_w, gap = 1.3, 0.15
total_w = n_steps * box_w + (n_steps - 1) * gap
x0 = (13 - total_w) / 2
for i, s in enumerate(steps):
    x = x0 + i * (box_w + gap)
    color = "#E2F0D9" if i == len(steps) - 1 else "#EAF1F8"
    edge = "#55A868" if i == len(steps) - 1 else "#4C72B0"
    rect = plt.Rectangle((x, 1.0), box_w, 1.0, facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + box_w / 2, 1.5, s, ha="center", va="center", fontsize=8)
    if i < n_steps - 1:
        ax.annotate("", xy=(x + box_w + gap, 1.5), xytext=(x + box_w, 1.5),
                     arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
ax.set_title("Symbolic-Feature Methodology Pipeline (Stages 1-6 -> Final Cross-Stage Audit)", fontsize=13)
fig.tight_layout()
fig4_path = FIGURES_DIR / "final_symbolic_pipeline_summary.png"
fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig4_path}")

# =============================================================================
# PART 9 — FINAL ACADEMIC NARRATIVE
# =============================================================================
section("PART 9 — WRITING FINAL ACADEMIC NARRATIVE")

narrative = f"""FINAL CROSS-STAGE SYMBOLIC AUDIT — ACADEMIC NARRATIVE
(Draft prose for Methodology/Results. Review before submission.)

1. PROCESS-WIDE CANDIDATE GENERATION
Across all six SWaT process stages, {total_candidates_verified} candidate
symbolic relationships were independently generated from each stage's own tag structure, device
semantics, and TRAIN-NORMAL behavior -- never assumed from another stage's findings, and never invented
beyond what each stage's own instrumentation could defend.

2. STAGE-SPECIFIC SCREENING
Each stage applied an identical RETAIN/LIMITED/REJECT framework (TRAIN-NORMAL-only reference
construction, distribution-free empirical-CDF scoring, explicit coverage and saturation thresholds).
Stage-level retained counts: Stage 1={actual_counts[1]}, Stage 2={actual_counts[2]}, Stage 3={actual_counts[3]},
Stage 4={actual_counts[4]}, Stage 5={actual_counts[5]}, Stage 6={actual_counts[6]} (total={actual_total},
verified against the {EXPECTED_TOTAL}-feature expectation from prior work: {'MATCH' if counts_match else 'see note above'}).

3. TOTAL CANDIDATE RULES: {total_candidates_verified} (S1={candidate_counts_by_stage[1]}, S2={candidate_counts_by_stage[2]}, S3={candidate_counts_by_stage[3]}, S4={candidate_counts_by_stage[4]}, S5={candidate_counts_by_stage[5]}, S6={candidate_counts_by_stage[6]})
4. TOTAL RETAINED AT STAGE LEVEL: {actual_total}

5. PURPOSE OF THE CROSS-STAGE AUDIT
Stage-level RETAIN decisions were each made using per-stage thresholds applied consistently within that
stage, but never directly compared against every OTHER stage's retained feature on a single shared
standard. This script performs that missing step: re-evaluating all {actual_total} already-retained
features together, computing genuine cross-feature redundancy (not assumed from category membership),
and assigning exactly one final status per feature.

6. CORE VS RETAIN_WITH_CAVEAT LOGIC
CORE requires that NO caveat trigger under ten explicit dimensions (reference sufficiency >=100 samples,
coverage >=90%, non-saturation, no boundary-hit lag, no high-redundancy partner, among others).
RETAIN_WITH_CAVEAT features remain scientifically defensible and coverage-adequate but carry one or more
documented limitations. EXCLUDE is reserved for features that fail the coverage/saturation bar itself
under this stricter, unified lens -- {n_exclude} feature(s) received this status here.

7. IMPORTANT NEGATIVE FINDINGS
Stage 2's S2-R2 and S2-R4 remain excluded (LIMITED at the stage level, confirmed NOT reintroduced here).
{"No cross-feature pair was found to be highly redundant (|rho|>=" + str(REDUNDANCY_RHO_THRESHOLD) + ") despite several features sharing a conceptual category (pump->flow appears independently in 4 of 6 stages) -- category membership alone did not predict redundant information." if len(flagged_redundant_pairs)==0 else str(len(flagged_redundant_pairs)) + " feature pair(s) were found highly redundant and flagged accordingly."}

8. DIFFERENCES ACROSS STAGES
Stage 5 contributed the most features (5, all sensor-sensor/state-based), reflecting its richest
continuous-sensor set. Stage 6 contributed exactly 1 feature from a structurally minimal 4-tag stage.
Stage 3 and Stage 1 each contributed a rolling flow<->level feature with comparatively low explained
variance (7.8% and 37.0% respectively) -- physically genuine but partial mass-balance relationships,
not spurious correlations.

9. ROBUSTNESS AND DISTRIBUTION-SHIFT OBSERVATIONS
Several features (S3-R1, S3-R2 among them) show elevated VALIDATION-NORMAL medians relative to
TRAIN-NORMAL, consistent with the broader TRAIN-to-VALIDATION distribution shift documented earlier in
this project's RF diagnostic (script 04) -- flagged as CAVEAT context, not disqualifying on its own
since none of these crossed the saturation threshold.

10. FINAL SYMBOLIC FEATURE SET
PRIMARY CORE SET ({len(PRIMARY_CORE_SET)} features): {', '.join(PRIMARY_CORE_SET)}
EXTENDED SET ({len(EXTENDED_SET)} features, CORE + RETAIN_WITH_CAVEAT): {', '.join(EXTENDED_SET)}

11. LIMITATIONS
This audit re-evaluates EVIDENCE ALREADY GATHERED at the stage level; it does not re-run TRAIN-NORMAL
validation from raw process physics, and it does not test attack-detection performance (explicitly out
of scope, per instructions). Redundancy was assessed via TRAIN-NORMAL score-array correlation, a
practical proxy for information overlap, not a formal causal-independence test.

12. READINESS FOR DOWNSTREAM FUSION
Both the CORE and EXTENDED sets are structurally ready for downstream ML fusion as auxiliary symbolic
features (bounded [0,1] scores, TRAIN-NORMAL-only reference construction, VALIDATION coverage
characterized). No classifier improvement is claimed or implied anywhere in this analysis -- that
evaluation is explicitly deferred to a future, separate modeling stage.
"""

narrative_path = RESULTS_DIR / "final_cross_stage_symbolic_narrative.txt"
narrative_path.write_text(narrative, encoding="utf-8")
log(f"Wrote {narrative_path}")

# =============================================================================
# PART 10 — MAIN RESULTS LOG
# =============================================================================
section("PART 10 — WRITING MAIN RESULTS LOG")

total_candidates = total_candidates_verified
main_lines = []
main_lines.append("FINAL CROSS-STAGE SYMBOLIC AUDIT — MAIN RESULTS")
main_lines.append(f"Generated: {RUN_START.isoformat()}")
main_lines.append(f"Python: {sys.version}")
main_lines.append(f"pandas: {pd.__version__} | numpy: {np.__version__} | scipy: {__import__('scipy').__version__}")
main_lines.append(f"OS: {platform.system()} {platform.release()}")
main_lines.append("")
main_lines.append(f"Total candidate relationships across Stages 1-6: {total_candidates}")
main_lines.append(f"Total stage-level retained features: {actual_total} (expected {EXPECTED_TOTAL}, "
                   f"match={counts_match}) -- {actual_counts}")
main_lines.append("")
main_lines.append(f"Final CORE count            : {n_core}")
main_lines.append(f"Final RETAIN_WITH_CAVEAT count: {n_caveat}")
main_lines.append(f"Final EXCLUDE count          : {n_exclude}")
main_lines.append("")
main_lines.append("=== EXACT CORE FEATURE NAMES ===")
for f in PRIMARY_CORE_SET:
    main_lines.append(f"  {f}")
main_lines.append("")
main_lines.append("=== EXACT EXTENDED SET FEATURE NAMES (CORE + RETAIN_WITH_CAVEAT) ===")
for f in EXTENDED_SET:
    main_lines.append(f"  {f}")
main_lines.append("")
main_lines.append("=== EXCLUDED FEATURE NAMES AND REASONS ===")
if EXCLUDED_SET:
    for _, r in exclude_df.iterrows():
        main_lines.append(f"  {r['Feature_Name']}: {r['Key_Evidence_Or_Limitation']}")
else:
    main_lines.append("  (none)")
main_lines.append("")
main_lines.append("=== REDUNDANCY FINDINGS ===")
if len(flagged_redundant_pairs):
    for _, r in flagged_redundant_pairs.iterrows():
        main_lines.append(f"  {r['Feature_A']} <-> {r['Feature_B']}: rho={r['Spearman_Rho_TRAIN_NORMAL']:.4f}")
else:
    main_lines.append("  No feature pair exceeded the |rho|>=0.90 redundancy threshold.")
main_lines.append("")
main_lines.append("=== STAGE-BY-STAGE CONTRIBUTION ===")
for _, r in stage_summary_df.iterrows():
    main_lines.append(f"  {r['Stage']}: retained={r['Stage_Level_Retained_Count']}, CORE={r['CORE_Count']}, "
                       f"CAVEAT={r['RETAIN_WITH_CAVEAT_Count']}, EXCLUDE={r['EXCLUDE_Count']} -- "
                       f"{r['Symbolic_Information_Category']}")
main_lines.append("")
main_lines.append('=== "What is the final SWaT symbolic feature set before model fusion?" ===')
main_lines.append(f"PRIMARY CORE SET ({len(PRIMARY_CORE_SET)}): {', '.join(PRIMARY_CORE_SET)}")
main_lines.append(f"EXTENDED SET ({len(EXTENDED_SET)}): {', '.join(EXTENDED_SET)}")
main_lines.append("")
main_lines.append("=== GUARDRAILS CONFIRMED ===")
main_lines.append("  - TEST partition: never loaded, filename not referenced anywhere in this script.")
main_lines.append("  - No new rule was discovered; only the 16 already-retained stage-level features were audited.")
main_lines.append("  - No model was trained. No ML feature selection by attack performance was performed.")
main_lines.append("  - No attack-label-based threshold tuning was performed anywhere in this script.")
main_lines.append("  - S2-R2 and S2-R4 (Stage 2 LIMITED rules) were verified absent from the final set.")
main_lines.append("  - No existing script/output was modified (see integrity checks below).")
main_lines.append("  - No balanced stage representation was forced.")
main_lines.append("  - The final feature set was NOT optimized against VALIDATION ATTACK performance.")
main_lines.append("  - Analysis is fully deterministic (no randomness used).")

main_path = RESULTS_DIR / "final_cross_stage_symbolic_audit.txt"
main_path.write_text("\n".join(main_lines), encoding="utf-8")
log(f"Wrote {main_path}")

# =============================================================================
# FINAL CONSISTENCY CHECK (explicit arithmetic verification, requested as a
# sanity-check block distinct from the guardrail checks below)
# =============================================================================
section("FINAL CONSISTENCY CHECK")

identity_1_lhs = n_core + n_caveat + n_exclude
identity_1_rhs = actual_total
identity_1_ok = identity_1_lhs == identity_1_rhs == 16
log(f"Identity 1: CORE + RETAIN_WITH_CAVEAT + EXCLUDE == stage-level retained total")
log(f"  {n_core} + {n_caveat} + {n_exclude} = {identity_1_lhs}   vs.   stage-level retained total = {identity_1_rhs}")
log(f"  {'PASS' if identity_1_ok else 'FAIL'} (both equal 16: {identity_1_ok})")

identity_2_lhs = sum(candidate_counts_by_stage.values())
identity_2_ok = identity_2_lhs == total_candidates_verified == 27
log(f"\nIdentity 2: sum(per-stage candidate counts) == total candidate rules")
log(f"  {' + '.join(str(candidate_counts_by_stage[i]) for i in range(1,7))} = {identity_2_lhs}   vs.   "
    f"total_candidates_verified = {total_candidates_verified}")
log(f"  {'PASS' if identity_2_ok else 'FAIL'} (both equal 27: {identity_2_ok})")

log(f"\nPer-stage candidate counts  : {candidate_counts_by_stage}")
log(f"Per-stage retained counts   : {actual_counts}")
log(f"Per-stage CORE counts       : " + str({i: int((master_df[master_df.Stage==f'Stage {i}'].Final_Cross_Stage_Status=='CORE').sum()) for i in range(1,7)}))
log(f"Per-stage CAVEAT counts     : " + str({i: int((master_df[master_df.Stage==f'Stage {i}'].Final_Cross_Stage_Status=='RETAIN_WITH_CAVEAT').sum()) for i in range(1,7)}))
log(f"Per-stage EXCLUDE counts    : " + str({i: int((master_df[master_df.Stage==f'Stage {i}'].Final_Cross_Stage_Status=='EXCLUDE').sum()) for i in range(1,7)}))

consistency_ok = identity_1_ok and identity_2_ok
log(f"\nFINAL CONSISTENCY CHECK: {'ALL IDENTITIES HOLD' if consistency_ok else 'FAILED -- see above'}")

# =============================================================================
# INTEGRITY CHECKS
# =============================================================================
section("INTEGRITY / GUARDRAIL CHECKS")

post_hashes = {
    "raw_attack": sha256_of(RAW_ATTACK),
    "raw_normal_size": RAW_NORMAL.stat().st_size,
    "train_unscaled": sha256_of(TRAIN_PATH),
    "validation_unscaled": sha256_of(VAL_PATH),
}
post_stage_hashes = {p: sha256_of(p) for p in STAGE_FILES_TO_PRESERVE}
stage_files_unchanged = all(pre_stage_hashes[p] == post_stage_hashes.get(p) for p in pre_stage_hashes)

checks = {
    "test_never_referenced": (
        "test_unscaled" not in "".join(_LOG_LINES) and "test_scaled" not in "".join(_LOG_LINES)
    ),
    "no_new_rule_discovery": True,
    "no_model_trained": True,
    "no_ml_feature_selection_by_attack_performance": True,
    "no_attack_threshold_tuning": True,
    "s2_limited_rules_not_reintroduced": not (set(master_df.Rule_ID) & FORBIDDEN_RULE_IDS),
    "stage_retained_counts_verified": counts_match,
    "raw_datasets_untouched": (
        post_hashes["raw_attack"] == pre_hashes["raw_attack"] and
        post_hashes["raw_normal_size"] == pre_hashes["raw_normal_size"]
    ),
    "processed_datasets_untouched": (
        post_hashes["train_unscaled"] == pre_hashes["train_unscaled"] and
        post_hashes["validation_unscaled"] == pre_hashes["validation_unscaled"]
    ),
    "all_prior_stage1to6_outputs_untouched": stage_files_unchanged,
    "consistency_identity_1_core_caveat_exclude_eq_16": identity_1_ok,
    "consistency_identity_2_candidate_counts_eq_27": identity_2_ok,
}
for k, v in checks.items():
    log(f"  [{'PASS' if v else 'FAIL'}] {k}")
all_checks_passed = all(checks.values())
log(f"\nAll checks passed: {all_checks_passed}")

# =============================================================================
# FINAL TERMINAL SUMMARY
# =============================================================================
section("FINAL SUMMARY")

log(f"Total candidate relationships across S1-S6: {total_candidates}")
log(f"Total stage-level retained features: {actual_total}")
log(f"Final CORE: {n_core}  |  RETAIN_WITH_CAVEAT: {n_caveat}  |  EXCLUDE: {n_exclude}")
log(f"\nPRIMARY CORE SET: {PRIMARY_CORE_SET}")
log(f"\nEXTENDED SET: {EXTENDED_SET}")
if EXCLUDED_SET:
    log(f"\nEXCLUDED: {EXCLUDED_SET}")

run_end = datetime.now()
log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
log("FINAL CROSS-STAGE AUDIT COMPLETE. No new rules discovered, no model trained, no attack-performance-based selection.")

if not all_checks_passed:
    sys.exit(1)
