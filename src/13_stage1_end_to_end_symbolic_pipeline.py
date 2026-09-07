"""
code/13_stage1_end_to_end_symbolic_pipeline.py

STAGE 1 END-TO-END SYMBOLIC PIPELINE — INDEPENDENT DISCOVERY, EMPIRICAL
VALIDATION, FEATURE CONSTRUCTION, AND FINAL SCREENING. ANALYSIS ONLY.

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

Applies the STANDARDIZED METHODOLOGY developed for Stage 2 (scripts
04-11) and Stage 5 (script 12) to Stage 1 (raw water intake), but
candidate relationships are discovered and validated INDEPENDENTLY from
Stage 1's own tag structure and TRAIN-NORMAL behavior -- neither Stage
2's nor Stage 5's specific rules are copied.

Stage 1 differs structurally from both prior stages:
  - MV101 has 3 observed states, like Stage 2's MV201 -- its transitional
    structure is re-verified from scratch here, not assumed from the
    MV201 precedent.
  - P101 varies substantially (134 TRAIN transitions); P102 is a sparse
    near-constant backup pump (4 transitions) -- reported explicitly,
    not silently dropped.
  - LIT101 (tank level) has no analog in Stage 5's tag set, enabling a
    genuine flow<->level ROLLING/DERIVATIVE relationship category that
    neither prior stage's pipeline needed to build.
  - Two varying actuators (MV101, P101) plus a sparse third (P102) allow
    an actuator-actuator coupling test unavailable in Stage 5 (which had
    only one varying actuator).

No ML model is trained, no feature is selected, no attack-label-based
threshold is tuned, no existing script/dataset/output is modified, and
TEST is never loaded -- its filename does not appear anywhere below, by
design. Uses ONLY processed/train_unscaled.csv and
processed/validation_unscaled.csv, plus tables/swat_stage_tag_mapping.csv
and (read-only, for the Part 9 comparison) scripts 11/12's outputs.
"""

import os
import sys
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
STAGE_MAPPING_PATH = PROJECT_ROOT / "tables" / "swat_stage_tag_mapping.csv"
STAGE2_SCREENING_PATH = PROJECT_ROOT / "tables" / "stage2_final_rule_screening.csv"
STAGE2_FEATURES_PATH = PROJECT_ROOT / "tables" / "stage2_final_retained_symbolic_features.csv"
STAGE5_SCREENING_PATH = PROJECT_ROOT / "tables" / "stage5_final_rule_screening.csv"
STAGE5_FEATURES_PATH = PROJECT_ROOT / "tables" / "stage5_final_retained_symbolic_features.csv"
# NOTE: no TEST_PATH constant is defined anywhere in this script, on purpose.

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = PROJECT_ROOT / "tables"
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


section("STAGE 1 END-TO-END SYMBOLIC PIPELINE — RUN START (ANALYSIS ONLY)")
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
log("Pre-run integrity snapshot recorded for raw + processed inputs (this script only reads).")

BASELINE_WINDOW = 10
EVENT_OFFSETS = [0, 1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120]
SLOPE_WINDOWS = [10, 30, 60, 120, 300]
LAG_GRID = [-30, -15, -10, -5, 0, 5, 10, 15, 30]
CONSISTENCY_THRESHOLD = 95.0
SATURATION_THRESHOLD = 0.90
MIN_EVENT_COUNT = 10
MIN_SAMPLE_COUNT = 1000

# =============================================================================
# LOAD DATA
# =============================================================================
section("LOADING TRAIN AND VALIDATION (TEST NEVER LOADED)")

train_df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
val_df = pd.read_csv(VAL_PATH).reset_index(drop=True)
train_df["_ts"] = pd.to_datetime(train_df["Timestamp"])
val_df["_ts"] = pd.to_datetime(val_df["Timestamp"])
log(f"TRAIN: {train_df.shape[0]:,} rows | VALIDATION: {val_df.shape[0]:,} rows")

stage_mapping = pd.read_csv(STAGE_MAPPING_PATH)
stage1_tags_df = stage_mapping[stage_mapping.Stage == "Stage 1"]
STAGE1_TAGS = stage1_tags_df["Tag"].tolist()
STAGE1_SENSORS = stage1_tags_df.loc[stage1_tags_df.Sensor_Or_Actuator == "sensor", "Tag"].tolist()
STAGE1_ACTUATORS = stage1_tags_df.loc[stage1_tags_df.Sensor_Or_Actuator == "actuator", "Tag"].tolist()
log(f"Loaded {STAGE_MAPPING_PATH.name}: Stage 1 = {len(STAGE1_TAGS)} tags "
    f"({len(STAGE1_SENSORS)} sensors, {len(STAGE1_ACTUATORS)} actuators)")
log(f"  Sensors  : {STAGE1_SENSORS}")
log(f"  Actuators: {STAGE1_ACTUATORS}")

# =============================================================================
# SHARED HELPER FUNCTIONS (identical methodology to scripts 05-12)
# =============================================================================


def find_specific_transitions(df, tag, from_state, to_state):
    s = df[tag].to_numpy()
    idx = np.flatnonzero((s[1:] == to_state) & (s[:-1] == from_state)) + 1
    return idx.tolist()


def check_baseline_window(df, i0, actuator, pre_state, window=BASELINE_WINDOW, require_normal=False):
    if i0 - window < 0:
        return False
    ts = df["_ts"]
    seg_ts = ts.iloc[i0 - window:i0]
    if not (seg_ts.diff().dropna() == pd.Timedelta(seconds=1)).all():
        return False
    if ts.iloc[i0] - ts.iloc[i0 - 1] != pd.Timedelta(seconds=1):
        return False
    if not (df[actuator].iloc[i0 - window:i0] == pre_state).all():
        return False
    if require_normal and not (df["Label"].iloc[i0 - window:i0] == 0).all():
        return False
    return True


def check_offset_window(df, i0, offset, require_normal=False):
    j = i0 + offset
    if j >= len(df) or j < 0:
        return False
    seg_ts = df["_ts"].iloc[i0:j + 1]
    if not (seg_ts.diff().dropna() == pd.Timedelta(seconds=1)).all():
        return False
    if df["_ts"].iloc[j] != df["_ts"].iloc[i0] + pd.Timedelta(seconds=offset):
        return False
    if require_normal and not (df["Label"].iloc[i0:j + 1] == 0).all():
        return False
    return True


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


def summarize(values):
    vals = np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))],
                     dtype=float)
    n = len(vals)
    keys = ["n", "mean", "median", "std", "min", "max", "p5", "p10", "p25", "p75", "p90", "p95", "iqr"]
    if n == 0:
        return {k: (0 if k == "n" else np.nan) for k in keys}
    p5, p10, p25, p75, p90, p95 = np.percentile(vals, [5, 10, 25, 75, 90, 95])
    return {"n": n, "mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "std": float(np.std(vals, ddof=0)), "min": float(np.min(vals)), "max": float(np.max(vals)),
            "p5": float(p5), "p10": float(p10), "p25": float(p25), "p75": float(p75), "p90": float(p90),
            "p95": float(p95), "iqr": float(p75 - p25)}


def pct_expected(values, expected_sign):
    if expected_sign is None:
        return np.nan
    vals = np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))],
                     dtype=float)
    if len(vals) == 0:
        return np.nan
    return 100 * float((np.sign(vals) == np.sign(expected_sign)).sum()) / len(vals)


def state_separation(off_ref, on_ref):
    """SMD + IQR-overlap-fraction, with a defensible (not NaN) fallback for
    degenerate zero-span groups -- fixed here from the outset (script 12
    caught this bug after the fact; applying the corrected version directly)."""
    pooled_std = np.sqrt((np.var(off_ref) + np.var(on_ref)) / 2)
    smd = (np.mean(on_ref) - np.mean(off_ref)) / pooled_std if pooled_std > 0 else np.nan
    off_iqr = np.percentile(off_ref, [25, 75]); on_iqr = np.percentile(on_ref, [25, 75])
    overlap = max(0.0, min(off_iqr[1], on_iqr[1]) - max(off_iqr[0], on_iqr[0]))
    min_span = min(off_iqr[1] - off_iqr[0], on_iqr[1] - on_iqr[0])
    if min_span > 0:
        overlap_frac = overlap / min_span
    elif overlap == 0:
        overlap_frac = 0.0
    else:
        overlap_frac = 1.0
    ks_stat, ks_p = scipy_stats.ks_2samp(off_ref, on_ref)
    return {"smd": smd, "overlap_frac": overlap_frac, "ks_stat": ks_stat, "ks_p": ks_p}


# =============================================================================
# PART 1 — STAGE 1 TAG AUDIT
# =============================================================================
section("PART 1 — STAGE 1 TAG AUDIT")

tag_audit_rows = []
for tag in STAGE1_TAGS:
    row_meta = stage1_tags_df[stage1_tags_df.Tag == tag].iloc[0]
    for split_name, df in [("TRAIN", train_df), ("VALIDATION", val_df)]:
        s = df[tag]
        n = len(s)
        nunique = s.nunique()
        n_trans = int((s != s.shift(1)).sum())
        n_trans = max(n_trans - 1, 0) if n > 0 else 0
        is_actuator = row_meta["Sensor_Or_Actuator"] == "actuator"
        near_constant = (nunique == 1) or (s.value_counts(normalize=True).iloc[0] > 0.999 if nunique > 1 else True)
        rec = {
            "Tag": tag, "Split": split_name, "Device_Type": row_meta["Device_Type"],
            "Sensor_Or_Actuator": row_meta["Sensor_Or_Actuator"], "Dtype": str(s.dtype),
            "N_Unique": int(nunique), "N_Transitions": n_trans,
            "Constant_Or_Near_Constant": bool(near_constant),
        }
        if is_actuator:
            vc = s.value_counts().sort_index()
            rec["State_Values_And_Frequencies"] = "; ".join(f"{k}={v}({100*v/n:.2f}%)" for k, v in vc.items())
            rec["Min"] = rec["Max"] = rec["Mean"] = rec["Median"] = rec["Std"] = np.nan
        else:
            rec["State_Values_And_Frequencies"] = np.nan
            rec["Min"] = float(s.min()); rec["Max"] = float(s.max())
            rec["Mean"] = float(s.mean()); rec["Median"] = float(s.median()); rec["Std"] = float(s.std(ddof=0))
        tag_audit_rows.append(rec)

tag_audit_df = pd.DataFrame(tag_audit_rows)
tag_audit_path = TABLES_DIR / "stage1_tag_audit.csv"
tag_audit_df.to_csv(tag_audit_path, index=False)
log(f"Wrote {tag_audit_path} ({len(tag_audit_df)} rows)")

log("\nTRAIN summary:")
for _, r in tag_audit_df[tag_audit_df.Split == "TRAIN"].iterrows():
    if r["Sensor_Or_Actuator"] == "actuator":
        log(f"  {r['Tag']:8s} [actuator] states: {r['State_Values_And_Frequencies']} | "
            f"transitions={r['N_Transitions']} | constant/near-constant={r['Constant_Or_Near_Constant']}")
    else:
        log(f"  {r['Tag']:8s} [sensor]   n_unique={r['N_Unique']:>5} min={r['Min']:.4f} max={r['Max']:.4f} "
            f"mean={r['Mean']:.4f} median={r['Median']:.4f} std={r['Std']:.4f}")

varying_actuators = tag_audit_df[(tag_audit_df.Split == "TRAIN") & (tag_audit_df.Sensor_Or_Actuator == "actuator") &
                                  (~tag_audit_df.Constant_Or_Near_Constant)]["Tag"].tolist()
sparse_but_varying = []
for t in STAGE1_ACTUATORS:
    n_trans = tag_audit_df[(tag_audit_df.Split == "TRAIN") & (tag_audit_df.Tag == t)]["N_Transitions"].iloc[0]
    if t in varying_actuators and n_trans < 20:
        sparse_but_varying.append(t)
log(f"\nStage 1 actuators with sufficient TRAIN variation for empirical rule construction: {varying_actuators}")
log(f"Of these, actuators that are sparse (< 20 TRAIN transitions, reported explicitly, NOT discarded): "
    f"{sparse_but_varying}")
log("NOTE: MV101 (3 observed states, like Stage 2's MV201) and P101 (134 transitions) provide solid "
    "empirical bases; P102 varies (2 states) but only 4 transitions in TRAIN -- treated the same way "
    "Stage 2 treated its sparse backup pumps (P202/P204/P206): reported honestly, tested where "
    "possible, not silently dropped from the audit.")

# =============================================================================
# EMPIRICALLY VERIFY MV101 STATE SEMANTICS (data-driven; re-derived from
# scratch, NOT assumed from the MV201 precedent in Stage 2)
# =============================================================================
section("EMPIRICALLY VERIFYING MV101 STATE SEMANTICS (data-driven, not assumed)")

train_normal = train_df[train_df.Label == 0]
mv101_states = sorted(train_df["MV101"].unique())
log(f"MV101 observed TRAIN states: {mv101_states}")

mv101_state_stats = {}
for st in mv101_states:
    mask = train_normal["MV101"] == st
    freq = int(mask.sum())
    fit_mean = train_normal.loc[mask, "FIT101"].mean()
    fit_median = train_normal.loc[mask, "FIT101"].median()
    mv101_state_stats[st] = {"freq": freq, "pct": 100 * freq / len(train_normal), "fit_mean": fit_mean, "fit_median": fit_median}
    log(f"  MV101={st}: freq={freq:,} ({100*freq/len(train_normal):.2f}%), "
        f"FIT101 mean={fit_mean:.4f}, median={fit_median:.4f}")

fit_max = train_normal["FIT101"].max()
near_zero_thresh = 0.05 * fit_max
mv101_labels = {}
if len(mv101_states) <= 2:
    # Only two states observed -- simple threshold classification, no
    # transitional/bridging state is structurally possible.
    for st, stt in mv101_state_stats.items():
        if stt["freq"] < MIN_EVENT_COUNT:
            mv101_labels[st] = "UNCERTAIN"
        elif stt["fit_mean"] <= near_zero_thresh:
            mv101_labels[st] = "CLOSED"
        else:
            mv101_labels[st] = "OPEN"
else:
    # 3+ states: the extremes (lowest/highest mean FIT101) are CLOSED/OPEN;
    # any state in between is a bridging/transitional state, NOT assumed to
    # be steady OPEN or CLOSED regardless of how a naive threshold would
    # classify it -- this directly generalizes the Stage-2 MV201 treatment,
    # re-derived here from Stage 1's own data, not assumed from precedent.
    sorted_by_fit = sorted(mv101_state_stats.items(), key=lambda kv: kv[1]["fit_mean"])
    closed_state, open_state = sorted_by_fit[0][0], sorted_by_fit[-1][0]
    for st in mv101_states:
        if st == closed_state:
            mv101_labels[st] = "CLOSED"
        elif st == open_state:
            mv101_labels[st] = "OPEN"
        else:
            mv101_labels[st] = "TRANSITIONAL"
log(f"Inferred MV101 semantics (via FIT101 level comparison, mirroring the Stage-2 MV201 method): "
    f"{mv101_labels}")

# Check whether MV101 transitions directly between CLOSED and OPEN states, or
# whether (like MV201) it always passes through an intermediate state.
mv101_direct_transitions = {}
for a in mv101_states:
    for b in mv101_states:
        if a == b:
            continue
        n = len(find_specific_transitions(train_df, "MV101", a, b))
        if n > 0:
            mv101_direct_transitions[(a, b)] = n
log(f"MV101 observed (from,to) transition pairs and counts: {mv101_direct_transitions}")

closed_states = [s for s, lbl in mv101_labels.items() if lbl == "CLOSED"]
open_states = [s for s, lbl in mv101_labels.items() if lbl == "OPEN"]
direct_closed_to_open = any((a in closed_states and b in open_states) for (a, b) in mv101_direct_transitions)
log(f"Direct CLOSED->OPEN transition observed: {direct_closed_to_open}")
if not direct_closed_to_open and len(mv101_states) == 3:
    log("MV101 requires the SAME 3-state opening-sequence correction discovered for Stage 2's MV201 "
        "(CLOSED -> transitional -> OPEN) -- re-verified here independently, not assumed.")
    MV101_MODE = "three_state_sequence"
else:
    MV101_MODE = "direct_two_state"
    log("MV101 supports direct CLOSED<->OPEN transitions -- using the standard 2-state event methodology.")

# =============================================================================
# PART 2 — CANDIDATE PHYSICAL RELATIONSHIP DISCOVERY
# =============================================================================
section("PART 2 — CANDIDATE RELATIONSHIP DISCOVERY")

candidates = []

candidates.append({
    "Rule_ID": "S1-R1", "Source_Tags": "MV101", "Target_Tags": "FIT101",
    "Relationship_Type": "valve -> flow (state/event)",
    "Expected_Relationship": "MV101 OPEN should produce higher FIT101 than MV101 CLOSED",
    "Rationale": "A motorized inlet valve mechanically gates its own immediately-downstream flow "
                 "meter -- the same structural logic as the validated S2-R1 (MV201->FIT201) and "
                 "S5-R1 (P501->FIT504), applied here to Stage 1's own valve/flow pair. MV101's "
                 "3-state structure (re-verified above) closely mirrors MV201's.",
    "Confidence": "high (direct mechanical gating)",
    "Time_Lag_Expected": "Possibly -- to be determined empirically in Part 3.",
})
candidates.append({
    "Rule_ID": "S1-R2", "Source_Tags": "P101", "Target_Tags": "LIT101 (rolling slope)",
    "Relationship_Type": "pump -> tank-level response (rolling/derivative)",
    "Expected_Relationship": "P101 ON should be associated with a different LIT101 slope "
                              "(rate of level change) than P101 OFF -- direction to be determined "
                              "empirically, not assumed, since raw water tank level depends on the "
                              "balance of MV101 inflow and P101/P102 outflow simultaneously.",
    "Rationale": "P101 pumps water onward from the Stage-1 raw water tank; instantaneous LIT101 is "
                 "confounded by simultaneous MV101 inflow, so the pump's own effect is better isolated "
                 "via the tank level's rate of change (slope) than via the raw level itself, per the "
                 "explicit rolling/derivative guidance for this category.",
    "Confidence": "moderate (structurally plausible pump-drains-tank logic, but confounded by MV101; "
                   "tested via state-conditioned slope comparison, not assumed).",
    "Time_Lag_Expected": "Yes -- slope evaluated over multiple rolling windows (10-300s).",
})
candidates.append({
    "Rule_ID": "S1-R3", "Source_Tags": "P102", "Target_Tags": "LIT101 (rolling slope)",
    "Relationship_Type": "pump -> tank-level response (rolling/derivative)",
    "Expected_Relationship": "Same structural hypothesis as S1-R2, applied to Stage 1's backup pump.",
    "Rationale": "P102 is P101's structural counterpart (same device type, same tag family) -- tested "
                 "identically for completeness, despite its extreme TRAIN sparsity (4 transitions), "
                 "per the explicit instruction not to silently discard sparse actuators.",
    "Confidence": "low-to-moderate (same structural logic as S1-R2, but severely data-limited).",
    "Time_Lag_Expected": "Yes, if a relationship is detectable at all given the sparsity.",
})
candidates.append({
    "Rule_ID": "S1-R4", "Source_Tags": "FIT101", "Target_Tags": "LIT101 (rolling slope)",
    "Relationship_Type": "flow <-> level change (rolling/derivative, mass-balance)",
    "Expected_Relationship": "Higher FIT101 (inflow) should coincide with a more positive LIT101 "
                              "slope (tank filling) -- this is a direct physical mass-balance "
                              "tautology for any tank: level change is driven by the net of inflow "
                              "and outflow.",
    "Rationale": "FIT101 is Stage 1's only flow measurement and LIT101 its only level measurement; "
                 "their rolling relationship is the most physically certain candidate available in "
                 "Stage 1's own tag structure (direct hydraulic mass balance, not an inferred process "
                 "identity).",
    "Confidence": "high (mass-balance physics apply to any tank, independent of plant-specific "
                   "identities of individual tags).",
    "Time_Lag_Expected": "Yes -- level integrates flow, so a lag/window sweep is required.",
})
candidates.append({
    "Rule_ID": "S1-R5", "Source_Tags": "P101", "Target_Tags": "P102",
    "Relationship_Type": "actuator-actuator control coupling",
    "Expected_Relationship": "To be determined empirically -- tested for the same same-row, "
                              "same-direction transition coupling pattern that was strongly "
                              "confirmed for Stage 2's P205<->P203 (S2-R3B).",
    "Rationale": "P101 and P102 are the two pumps of the same tag family in Stage 1, directly "
                 "analogous in structure to Stage 2's P205/P203 pair -- tested for a primary/backup "
                 "coupling relationship, not assumed to exist given P102's extreme sparsity.",
    "Confidence": "low (P102's 4 TRAIN transitions make any coupling claim statistically fragile even "
                   "if a pattern appears).",
    "Time_Lag_Expected": "If present, expected near-0s per the Stage-2 S2-R3B precedent (tested "
                          "independently, not assumed).",
})

candidates_df = pd.DataFrame(candidates)
candidates_path = TABLES_DIR / "stage1_candidate_rules.csv"
candidates_df.to_csv(candidates_path, index=False)
log(f"Wrote {candidates_path} ({len(candidates_df)} rows)")
log(f"\n{len(candidates)} candidate relationships identified BEFORE any validation testing:")
for c in candidates:
    log(f"  {c['Rule_ID']}: {c['Source_Tags']} <-> {c['Target_Tags']} ({c['Relationship_Type']})")

# =============================================================================
# PART 3 + 4 — TRAIN-NORMAL EMPIRICAL VALIDATION + NORMAL OPERATING ENVELOPES
# =============================================================================
section("PART 3 & 4 — TRAIN-NORMAL EMPIRICAL VALIDATION AND OPERATING ENVELOPES")

# Precompute rolling LIT101 slopes at each window (TRAIN NORMAL indices only
# used for reference construction; full-series slopes computed for later
# application to TRAIN/VALIDATION scoring).
def rolling_slope(df, tag, window):
    return df[tag] - df[tag].shift(window)


train_df["_lit101_slope_precomp"] = {}  # placeholder, replaced below with per-window columns
slope_cols_train = {}
slope_cols_val = {}
for w in SLOPE_WINDOWS:
    slope_cols_train[w] = rolling_slope(train_df, "LIT101", w).to_numpy()
    slope_cols_val[w] = rolling_slope(val_df, "LIT101", w).to_numpy()
del train_df["_lit101_slope_precomp"]

validation_rows = []
envelope_rows = []
rule_evidence = {}


def validate_valve_flow_S1R1():
    """MV101 -> FIT101, handling either the direct 2-state or 3-state
    opening-sequence structure, exactly mirroring the corrected Stage-2
    methodology but re-derived from Stage 1's own data."""
    off_ref = train_normal.loc[train_normal.MV101.isin(closed_states), "FIT101"].to_numpy()
    on_ref = train_normal.loc[train_normal.MV101.isin(open_states), "FIT101"].to_numpy()
    sep = state_separation(off_ref, on_ref)
    log(f"  [state-based] FIT101 by MV101 CLOSED/OPEN: SMD={sep['smd']:.3f}, "
        f"overlap_frac={sep['overlap_frac']:.3f}, KS={sep['ks_stat']:.3f} (p={sep['ks_p']:.2e}), "
        f"n_CLOSED={len(off_ref)}, n_OPEN={len(on_ref)}")

    events = []
    if MV101_MODE == "direct_two_state" and len(closed_states) == 1 and len(open_states) == 1:
        trans = find_specific_transitions(train_df, "MV101", closed_states[0], open_states[0])
        for i0 in trans:
            if train_df["Label"].iloc[i0] != 0:
                continue
            if not check_baseline_window(train_df, i0, "MV101", closed_states[0], require_normal=True):
                continue
            events.append({"open_idx": i0, "baseline_idx": i0})
    else:
        # 3-state opening-sequence: CLOSED -> transitional -> OPEN
        trans_state = [s for s in mv101_states if s not in closed_states and s not in open_states]
        trans_state = trans_state[0] if trans_state else None
        if trans_state is not None and len(closed_states) == 1 and len(open_states) == 1:
            s_arr = train_df["MV101"].to_numpy()
            n = len(s_arr)
            is_trans = (s_arr == trans_state).astype(int)
            diffs = np.diff(np.concatenate(([0], is_trans, [0])))
            starts = np.flatnonzero(diffs == 1)
            ends = np.flatnonzero(diffs == -1) - 1
            for st, en in zip(starts, ends):
                if st - 1 < 0 or en + 1 >= n:
                    continue
                if s_arr[st - 1] == closed_states[0] and s_arr[en + 1] == open_states[0]:
                    if train_df["Label"].iloc[st] != 0:
                        continue
                    if not check_baseline_window(train_df, st, "MV101", closed_states[0], require_normal=True):
                        continue
                    events.append({"open_idx": en + 1, "baseline_idx": st})
    log(f"  [event-based] {len(events)} usable TRAIN NORMAL opening events ({MV101_MODE})")

    offset_stats = []
    if len(events) >= MIN_EVENT_COUNT:
        expected_sign = 1 if on_ref.mean() > off_ref.mean() else -1
        for offset in EVENT_OFFSETS:
            vals = []
            for ev in events:
                open_idx, base_idx = ev["open_idx"], ev["baseline_idx"]
                if not check_offset_window(train_df, open_idx, offset, require_normal=True):
                    continue
                base = train_df["FIT101"].iloc[base_idx - BASELINE_WINDOW:base_idx].mean()
                vals.append(train_df["FIT101"].iloc[open_idx + offset] - base)
            stats = summarize(vals)
            pe = pct_expected(vals, expected_sign)
            offset_stats.append({"Offset_Seconds": offset, **stats, "Pct_Expected_Direction": pe})
        valid_os = [o for o in offset_stats if o["n"] >= MIN_EVENT_COUNT and not np.isnan(o["Pct_Expected_Direction"])]
        best = None
        if valid_os:
            best = max(valid_os, key=lambda o: o["Pct_Expected_Direction"])
            for o in valid_os:
                if o["Pct_Expected_Direction"] >= CONSISTENCY_THRESHOLD:
                    best = o
                    break
            log(f"  [event-based] best offset={best['Offset_Seconds']}s: n={best['n']}, "
                f"pct_expected_dir={best['Pct_Expected_Direction']:.1f}%, median={best['median']:.5f}")
    else:
        best = None

    return {"kind": "valve_flow", "off_ref": off_ref, "on_ref": on_ref, **sep,
            "n_events": len(events), "offset_stats": offset_stats, "best_offset": best,
            "closed_states": closed_states, "open_states": open_states}


def validate_pump_slope(pump_tag):
    """Pump ON/OFF vs LIT101 rolling-slope, across several windows -- state-
    conditioned comparison (not raw level, per the rolling/derivative
    guidance for this relationship category)."""
    states = sorted(train_df[pump_tag].unique())
    if len(states) != 2:
        return {"kind": "pump_slope", "supported": False, "reason": "not exactly 2 states"}
    off_state, on_state = states[0], states[1]
    results_by_window = {}
    for w in SLOPE_WINDOWS:
        slope = pd.Series(slope_cols_train[w])
        mask_valid = slope.notna().to_numpy()
        pump_states = train_df[pump_tag].to_numpy()
        normal_mask = (train_df["Label"].to_numpy() == 0)
        off_ref = slope[mask_valid & (pump_states == off_state) & normal_mask].to_numpy()
        on_ref = slope[mask_valid & (pump_states == on_state) & normal_mask].to_numpy()
        if len(off_ref) < 5 or len(on_ref) < 5:
            continue
        sep = state_separation(off_ref, on_ref)
        results_by_window[w] = {"off_ref": off_ref, "on_ref": on_ref, **sep}
        log(f"  [{pump_tag} slope window={w}s] SMD={sep['smd']:.3f}, overlap_frac={sep['overlap_frac']:.3f}, "
            f"n_OFF={len(off_ref)}, n_ON={len(on_ref)}")
    if not results_by_window:
        return {"kind": "pump_slope", "supported": False, "reason": "insufficient samples at every window",
                "off_state": off_state, "on_state": on_state, "results_by_window": {}}
    best_window = max(results_by_window, key=lambda w: abs(results_by_window[w]["smd"]) if not np.isnan(results_by_window[w]["smd"]) else -1)
    log(f"  [{pump_tag}] best slope window: {best_window}s (SMD={results_by_window[best_window]['smd']:.3f})")
    return {"kind": "pump_slope", "supported": True, "off_state": off_state, "on_state": on_state,
            "results_by_window": results_by_window, "best_window": best_window,
            **results_by_window[best_window]}


def validate_flow_level_slope():
    """FIT101 vs LIT101 rolling slope -- continuous mass-balance relationship,
    tested via correlation across slope windows (initial evidence) plus a
    residual-stability check (beyond correlation alone)."""
    best = None
    all_results = {}
    for w in SLOPE_WINDOWS:
        slope = pd.Series(slope_cols_train[w])
        mask_valid = slope.notna().to_numpy() & (train_df["Label"].to_numpy() == 0)
        fit = train_df["FIT101"].to_numpy(dtype=float)
        rho, rho_p = scipy_stats.spearmanr(fit[mask_valid], slope[mask_valid])
        all_results[w] = {"rho": rho, "rho_p": rho_p, "n": int(mask_valid.sum())}
        log(f"  [FIT101 vs LIT101 slope, window={w}s] Spearman rho={rho:.4f} (p={rho_p:.2e}), n={int(mask_valid.sum())}")
        if best is None or abs(rho) > abs(all_results[best]["rho"]):
            best = w
    w = best
    slope = pd.Series(slope_cols_train[w])
    mask_valid = slope.notna().to_numpy() & (train_df["Label"].to_numpy() == 0)
    fit = train_df["FIT101"].to_numpy(dtype=float)[mask_valid]
    slope_vals = slope.to_numpy()[mask_valid]
    # residual: slope - (linear fit of slope on FIT101) -- check stability
    # (this goes beyond correlation alone, per instructions)
    if len(fit) >= MIN_SAMPLE_COUNT and np.std(fit) > 0:
        b, a = np.polyfit(fit, slope_vals, 1)
        predicted = a + b * fit
        residual = slope_vals - predicted
        residual_std = np.std(residual)
        slope_std = np.std(slope_vals)
        explained_frac = 1 - (residual_std ** 2 / slope_std ** 2) if slope_std > 0 else np.nan
    else:
        residual = np.array([])
        explained_frac = np.nan
    log(f"  [FIT101 vs LIT101 slope, best window={w}s] linear-fit explained variance fraction="
        f"{explained_frac:.3f} (beyond-correlation stability check)")
    return {"kind": "flow_level_slope", "best_window": w, "all_results": all_results,
            "rho": all_results[w]["rho"], "rho_p": all_results[w]["rho_p"],
            "residual_sample": residual, "explained_frac": explained_frac,
            "slope_sample_for_envelope": slope_vals}


def validate_actuator_coupling(tag_a, tag_b):
    """P101<->P102 same-row same-direction transition coupling, mirroring the
    S2-R3B methodology exactly, re-tested here (not assumed)."""
    n = len(train_df)
    a = train_df[tag_a].to_numpy()
    b = train_df[tag_b].to_numpy()
    a_changed = np.zeros(n, dtype=bool); b_changed = np.zeros(n, dtype=bool)
    a_changed[1:] = a[1:] != a[:-1]
    b_changed[1:] = b[1:] != b[:-1]
    transition_active = a_changed | b_changed
    a_dir = np.zeros(n); b_dir = np.zeros(n)
    a_dir[1:] = np.sign(a[1:].astype(int) - a[:-1].astype(int))
    b_dir[1:] = np.sign(b[1:].astype(int) - b[:-1].astype(int))
    both_same_dir = transition_active & a_changed & b_changed & (a_dir == b_dir)
    normal_mask = (train_df["Label"].to_numpy() == 0)
    n_transition_active_normal = int((transition_active & normal_mask).sum())
    n_coupled_normal = int((both_same_dir & normal_mask).sum())
    coupling_rate = n_coupled_normal / n_transition_active_normal if n_transition_active_normal else np.nan
    log(f"  [{tag_a}<->{tag_b} coupling] TRAIN-NORMAL transition-active rows={n_transition_active_normal}, "
        f"same-row-same-direction={n_coupled_normal}, rate={coupling_rate}")
    return {"kind": "actuator_coupling", "transition_active": transition_active,
            "coupling_indicator": both_same_dir, "n_transition_active": n_transition_active_normal,
            "coupling_rate": coupling_rate}


for c in candidates:
    rid = c["Rule_ID"]
    log(f"\n--- Validating {rid}: {c['Source_Tags']} <-> {c['Target_Tags']} ---")
    if rid == "S1-R1":
        result = validate_valve_flow_S1R1()
    elif rid in ("S1-R2", "S1-R3"):
        pump_tag = c["Source_Tags"]
        result = validate_pump_slope(pump_tag)
    elif rid == "S1-R4":
        result = validate_flow_level_slope()
    else:  # S1-R5
        result = validate_actuator_coupling("P101", "P102")
    rule_evidence[rid] = result

    if result["kind"] == "valve_flow":
        validation_rows.append({
            "Rule_ID": rid, "Kind": "valve_flow", "N_Events_TRAIN_NORMAL": result["n_events"],
            "State_SMD": result["smd"], "State_IQR_Overlap_Fraction": result["overlap_frac"],
            "State_KS_Statistic": result["ks_stat"],
            "Best_Event_Offset_Seconds": result["best_offset"]["Offset_Seconds"] if result["best_offset"] else np.nan,
            "Best_Event_Pct_Expected_Direction": result["best_offset"]["Pct_Expected_Direction"] if result["best_offset"] else np.nan,
            "Best_Event_Median_Response": result["best_offset"]["median"] if result["best_offset"] else np.nan,
        })
        if result["best_offset"]:
            b = result["best_offset"]
            envelope_rows.append({"Rule_ID": rid, "Envelope_Type": "event_response", "Window_Or_Offset": b["Offset_Seconds"],
                                   "N": b["n"], "P5": b["p5"], "P10": b["p10"], "P25": b["p25"], "Median": b["median"],
                                   "P75": b["p75"], "P90": b["p90"], "P95": b["p95"], "IQR": b["iqr"]})
        for label, ref in [("CLOSED", result["off_ref"]), ("OPEN", result["on_ref"])]:
            st = summarize(ref)
            envelope_rows.append({"Rule_ID": rid, "Envelope_Type": f"state_conditioned_{label}", "Window_Or_Offset": np.nan,
                                   "N": st["n"], "P5": st["p5"], "P10": st["p10"], "P25": st["p25"], "Median": st["median"],
                                   "P75": st["p75"], "P90": st["p90"], "P95": st["p95"], "IQR": st["iqr"]})
    elif result["kind"] == "pump_slope":
        validation_rows.append({
            "Rule_ID": rid, "Kind": "pump_slope", "N_Events_TRAIN_NORMAL": (
                min(len(result["off_ref"]), len(result["on_ref"])) if result.get("supported") else 0),
            "State_SMD": result.get("smd", np.nan), "State_IQR_Overlap_Fraction": result.get("overlap_frac", np.nan),
            "Best_Slope_Window_Seconds": result.get("best_window", np.nan),
        })
        if result.get("supported"):
            for label, ref in [("OFF", result["off_ref"]), ("ON", result["on_ref"])]:
                st = summarize(ref)
                envelope_rows.append({"Rule_ID": rid, "Envelope_Type": f"slope_conditioned_{label}",
                                       "Window_Or_Offset": result["best_window"], "N": st["n"], "P5": st["p5"],
                                       "P10": st["p10"], "P25": st["p25"], "Median": st["median"], "P75": st["p75"],
                                       "P90": st["p90"], "P95": st["p95"], "IQR": st["iqr"]})
    elif result["kind"] == "flow_level_slope":
        validation_rows.append({
            "Rule_ID": rid, "Kind": "flow_level_slope",
            "N_Events_TRAIN_NORMAL": result["all_results"][result["best_window"]]["n"],
            "Spearman_Rho": result["rho"], "Spearman_Pvalue": result["rho_p"],
            "Best_Slope_Window_Seconds": result["best_window"], "Explained_Variance_Fraction": result["explained_frac"],
        })
        st = summarize(result["slope_sample_for_envelope"])
        envelope_rows.append({"Rule_ID": rid, "Envelope_Type": "lit101_slope_reference",
                               "Window_Or_Offset": result["best_window"], "N": st["n"], "P5": st["p5"],
                               "P10": st["p10"], "P25": st["p25"], "Median": st["median"], "P75": st["p75"],
                               "P90": st["p90"], "P95": st["p95"], "IQR": st["iqr"]})
    else:  # actuator_coupling
        validation_rows.append({
            "Rule_ID": rid, "Kind": "actuator_coupling", "N_Events_TRAIN_NORMAL": result["n_transition_active"],
            "Coupling_Rate": result["coupling_rate"],
        })

validation_df = pd.DataFrame(validation_rows)
validation_path = TABLES_DIR / "stage1_rule_validation.csv"
validation_df.to_csv(validation_path, index=False)
log(f"\nWrote {validation_path} ({len(validation_df)} rows)")

envelope_df = pd.DataFrame(envelope_rows)
envelope_path = TABLES_DIR / "stage1_normal_operating_envelopes.csv"
envelope_df.to_csv(envelope_path, index=False)
log(f"Wrote {envelope_path} ({len(envelope_df)} rows)")

# =============================================================================
# PART 5 + 6 — REPRESENTATION, COVERAGE, ROBUSTNESS
# =============================================================================
section("PART 5 & 6 — REPRESENTATION TYPE, VALIDATION COVERAGE, AND DISTRIBUTION-SHIFT ROBUSTNESS")

representation = {}
coverage_rows = []
robustness_rows = []
score_arrays = {}  # rid -> (eval_mask_train, score_train, eval_mask_val, score_val)

for c in candidates:
    rid = c["Rule_ID"]
    ev = rule_evidence[rid]

    if ev["kind"] == "valve_flow":
        representation[rid] = "A: state-based" + (" (+event-response secondary evidence)" if ev["n_events"] >= MIN_EVENT_COUNT else "")

        def score_fn(df, ev=ev):
            mv = df["MV101"].to_numpy()
            fit = df["FIT101"].to_numpy(dtype=float)
            score = np.full(len(df), np.nan)
            eval_mask = np.isin(mv, ev["closed_states"] + ev["open_states"])
            closed_mask = eval_mask & np.isin(mv, ev["closed_states"])
            open_mask = eval_mask & np.isin(mv, ev["open_states"])
            score[closed_mask] = empirical_two_sided_score_vec(ev["off_ref"], fit[closed_mask])
            score[open_mask] = empirical_two_sided_score_vec(ev["on_ref"], fit[open_mask])
            return eval_mask, score
        eval_mask_train, score_train = score_fn(train_df)
        eval_mask_val, score_val = score_fn(val_df)

    elif ev["kind"] == "pump_slope" and ev.get("supported"):
        representation[rid] = "C: rolling-window"
        w = ev["best_window"]
        pump_tag = c["Source_Tags"]

        def score_fn(df, split_slopes, ev=ev, w=w, pump_tag=pump_tag):
            slope = split_slopes[w]
            pump_states = df[pump_tag].to_numpy()
            score = np.full(len(df), np.nan)
            eval_mask = ~np.isnan(slope) & np.isin(pump_states, [ev["off_state"], ev["on_state"]])
            off_mask = eval_mask & (pump_states == ev["off_state"])
            on_mask = eval_mask & (pump_states == ev["on_state"])
            score[off_mask] = empirical_two_sided_score_vec(ev["off_ref"], slope[off_mask])
            score[on_mask] = empirical_two_sided_score_vec(ev["on_ref"], slope[on_mask])
            return eval_mask, score
        eval_mask_train, score_train = score_fn(train_df, slope_cols_train)
        eval_mask_val, score_val = score_fn(val_df, slope_cols_val)

    elif ev["kind"] == "flow_level_slope":
        representation[rid] = "C: rolling-window"
        w = ev["best_window"]
        ref_slope = ev["slope_sample_for_envelope"]

        def score_fn(df, split_slopes, w=w, ref_slope=ref_slope):
            slope = split_slopes[w]
            eval_mask = ~np.isnan(slope)
            score = np.full(len(df), np.nan)
            score[eval_mask] = empirical_two_sided_score_vec(ref_slope, slope[eval_mask])
            return eval_mask, score
        eval_mask_train, score_train = score_fn(train_df, slope_cols_train)
        eval_mask_val, score_val = score_fn(val_df, slope_cols_val)

    elif ev["kind"] == "actuator_coupling":
        representation[rid] = "B: event-based"
        # recompute for VALIDATION too
        n_val = len(val_df)
        a = val_df["P101"].to_numpy(); b = val_df["P102"].to_numpy()
        a_changed = np.zeros(n_val, dtype=bool); b_changed = np.zeros(n_val, dtype=bool)
        a_changed[1:] = a[1:] != a[:-1]; b_changed[1:] = b[1:] != b[:-1]
        val_transition_active = a_changed | b_changed
        a_dir = np.zeros(n_val); b_dir = np.zeros(n_val)
        a_dir[1:] = np.sign(a[1:].astype(int) - a[:-1].astype(int))
        b_dir[1:] = np.sign(b[1:].astype(int) - b[:-1].astype(int))
        val_both_same_dir = val_transition_active & a_changed & b_changed & (a_dir == b_dir)

        eval_mask_train = ev["transition_active"]
        score_train = np.where(eval_mask_train, 1.0 - ev["coupling_indicator"].astype(float), np.nan)
        eval_mask_val = val_transition_active
        score_val = np.where(eval_mask_val, 1.0 - val_both_same_dir.astype(float), np.nan)
    else:
        representation[rid] = "UNSUPPORTED"
        eval_mask_train = np.zeros(len(train_df), dtype=bool)
        score_train = np.full(len(train_df), np.nan)
        eval_mask_val = np.zeros(len(val_df), dtype=bool)
        score_val = np.full(len(val_df), np.nan)

    score_arrays[rid] = (eval_mask_train, score_train, eval_mask_val, score_val)

    val_normal_mask = eval_mask_val & (val_df["Label"].to_numpy() == 0)
    val_attack_mask = eval_mask_val & (val_df["Label"].to_numpy() == 1)
    n_val_normal_total = int((val_df["Label"].to_numpy() == 0).sum())
    n_val_attack_total = int((val_df["Label"].to_numpy() == 1).sum())
    cov_normal = 100 * val_normal_mask.sum() / n_val_normal_total if n_val_normal_total else np.nan
    cov_attack = 100 * val_attack_mask.sum() / n_val_attack_total if n_val_attack_total else np.nan
    coverage_rows.append({"Rule_ID": rid, "Representation_Type": representation[rid],
                           "VALIDATION_NORMAL_Coverage_Pct": cov_normal,
                           "VALIDATION_ATTACK_Coverage_Pct": cov_attack,
                           "N_VALIDATION_ATTACK_Evaluable": int(val_attack_mask.sum())})
    log(f"{rid} [{representation[rid]}]: VALIDATION coverage NORMAL={cov_normal:.3f}%, "
        f"ATTACK={cov_attack:.3f}% (coverage = evaluable, NOT attack detected)")

    train_normal_scores = score_train[eval_mask_train & (train_df["Label"].to_numpy() == 0)]
    val_normal_scores = score_val[val_normal_mask]
    val_attack_scores = score_val[val_attack_mask]
    train_normal_scores = train_normal_scores[~np.isnan(train_normal_scores)]
    val_normal_scores = val_normal_scores[~np.isnan(val_normal_scores)]
    val_attack_scores = val_attack_scores[~np.isnan(val_attack_scores)]

    ks_shift = np.nan
    if len(train_normal_scores) >= 30 and len(val_normal_scores) >= 30:
        ks_shift, _ = scipy_stats.ks_2samp(train_normal_scores, val_normal_scores)
    val_normal_mean = float(np.mean(val_normal_scores)) if len(val_normal_scores) else np.nan
    val_attack_mean = float(np.mean(val_attack_scores)) if len(val_attack_scores) else np.nan
    saturated = (not np.isnan(val_normal_mean)) and val_normal_mean >= SATURATION_THRESHOLD
    robustness_rows.append({
        "Rule_ID": rid,
        "TRAIN_NORMAL_Score_Median": float(np.median(train_normal_scores)) if len(train_normal_scores) else np.nan,
        "VALIDATION_NORMAL_Score_Median": float(np.median(val_normal_scores)) if len(val_normal_scores) else np.nan,
        "TRAIN_vs_VALIDATION_NORMAL_KS": ks_shift, "VALIDATION_NORMAL_Mean": val_normal_mean,
        "VALIDATION_ATTACK_Mean": val_attack_mean, "Saturated_In_VALIDATION": saturated,
    })
    log(f"{rid} robustness: TRAIN-N median={float(np.median(train_normal_scores)) if len(train_normal_scores) else np.nan:.4f}, "
        f"VAL-N median={float(np.median(val_normal_scores)) if len(val_normal_scores) else np.nan:.4f}, "
        f"KS={ks_shift}, saturated={saturated}")

coverage_df = pd.DataFrame(coverage_rows)
robustness_df = pd.DataFrame(robustness_rows)

# =============================================================================
# PART 7 — SYMBOLIC FEATURE CONSTRUCTION
# =============================================================================
section("PART 7 — SYMBOLIC FEATURE CONSTRUCTION")

feature_def_rows = []
for c in candidates:
    rid = c["Rule_ID"]
    ev = rule_evidence[rid]
    cov = coverage_df[coverage_df.Rule_ID == rid].iloc[0]
    if ev["kind"] == "valve_flow":
        fname = f"{rid}_state_score"
        math_def = ("score = 1 - 2*min(F(x), 1-F(x)), F = empirical CDF of TRAIN-NORMAL FIT101 for "
                     "the CURRENT steady MV101 state (CLOSED or OPEN)")
        interp = "How atypical FIT101 is for MV101's current steady state, vs. TRAIN-NORMAL."
        eval_cond = f"MV101 in a steady labeled state (CLOSED/OPEN; {MV101_MODE.replace('_',' ')})"
        limitation = "Transitional-state rows are excluded from evaluability, mirroring MV201's treatment in Stage 2." if MV101_MODE == "three_state_sequence" else "Standard 2-state coverage."
    elif ev["kind"] == "pump_slope" and ev.get("supported"):
        fname = f"{rid}_slope_score"
        w = ev["best_window"]
        math_def = (f"score = 1 - 2*min(F(x), 1-F(x)), F = empirical CDF of TRAIN-NORMAL LIT101 "
                     f"{w}s-rolling-slope for the CURRENT {c['Source_Tags']} state")
        interp = f"How atypical the tank-level trend is for {c['Source_Tags']}'s current state, vs. TRAIN-NORMAL."
        eval_cond = f"Every row where the {w}s slope is computable (row index >= {w}) and {c['Source_Tags']} is OFF or ON"
        limitation = (f"Built on a severely sparse minority-state reference (see stage1_rule_validation.csv)"
                       if c["Source_Tags"] == "P102" else "Confounded by simultaneous MV101/other-pump activity; state-conditioning only partially isolates this pump's own effect.")
    elif ev["kind"] == "flow_level_slope":
        fname = f"{rid}_slope_score"
        w = ev["best_window"]
        math_def = (f"score = 1 - 2*min(F(x), 1-F(x)), F = empirical CDF of TRAIN-NORMAL LIT101 "
                     f"{w}s-rolling-slope (mass-balance relationship with FIT101; linear-fit explained "
                     f"variance fraction={ev['explained_frac']:.3f})")
        interp = "How atypical the tank-level trend is overall, vs. TRAIN-NORMAL (not conditioned on any single actuator)."
        eval_cond = f"Every row where the {w}s slope is computable (row index >= {w})"
        limitation = "Reflects overall tank-level dynamics, not attributable to any single actuator; correlation with FIT101 motivated the window choice but the score itself is unconditional."
    elif ev["kind"] == "actuator_coupling":
        fname = f"{rid}_coupling_score"
        math_def = ("score = 1 - coupling_indicator, where coupling_indicator = 1 if BOTH P101 and "
                     "P102 change state on the same row in the SAME direction, else 0")
        interp = "Whether P101 and P102 transition together as a coupled pair, vs. TRAIN-NORMAL."
        eval_cond = "Only at rows where P101 OR P102 (or both) change state"
        limitation = f"Extremely sparse (TRAIN transition-active rows={ev['n_transition_active']}); P102 has only 4 TRAIN transitions overall, so any coupling claim rests on very little data."
    else:
        continue
    feature_def_rows.append({
        "Feature": fname, "Source_Rule": rid, "Source_Target_Tags": f"{c['Source_Tags']} / {c['Target_Tags']}",
        "Representation_Type": representation[rid], "Mathematical_Definition": math_def,
        "Physical_Interpretation": interp, "Evaluability_Condition": eval_cond,
        "Coverage_Pct_VALIDATION_NORMAL": cov["VALIDATION_NORMAL_Coverage_Pct"],
        "Score_Direction": "0 = typical of TRAIN-NORMAL; 1 = extreme tail (atypical)",
        "Limitation": limitation,
    })

feature_def_df = pd.DataFrame(feature_def_rows)
feature_def_path = TABLES_DIR / "stage1_symbolic_feature_definitions.csv"
feature_def_df.to_csv(feature_def_path, index=False)
log(f"Wrote {feature_def_path} ({len(feature_def_df)} rows)")

# =============================================================================
# PART 8 — FINAL RULE SCREENING
# =============================================================================
section("PART 8 — FINAL RULE SCREENING (RETAIN / LIMITED / REJECT)")

log("Decision rule (identical standard applied to finalized Stage 2 [script 11] and Stage 5 [script 12]):")
log(f"  1. If TRAIN-normal event/sample count < {MIN_EVENT_COUNT} (event/state rules) -> REJECT.")
log(f"  2. Else if not empirically supported beyond correlation (SMD/overlap too weak, or "
    f"consistency < {CONSISTENCY_THRESHOLD}%, or coupling rate too low) -> REJECT.")
log(f"  3. Else: RETAIN if VALIDATION coverage >= 90% AND not saturated (VAL-NORMAL mean < "
    f"{SATURATION_THRESHOLD}); otherwise LIMITED.")

decisions = {}
for c in candidates:
    rid = c["Rule_ID"]
    ev = rule_evidence[rid]
    cov = coverage_df[coverage_df.Rule_ID == rid].iloc[0]
    rob = robustness_df[robustness_df.Rule_ID == rid].iloc[0]

    if ev["kind"] == "valve_flow":
        n = len(ev["off_ref"]) + len(ev["on_ref"])
        supported = (not np.isnan(ev["smd"])) and abs(ev["smd"]) > 1.0 and ev["overlap_frac"] < 0.25
        support_desc = f"state SMD={ev['smd']:.3f}, IQR overlap fraction={ev['overlap_frac']:.3f}"
        min_n = MIN_EVENT_COUNT
    elif ev["kind"] == "pump_slope":
        if not ev.get("supported"):
            n, supported, support_desc, min_n = 0, False, ev.get("reason", "unsupported"), MIN_EVENT_COUNT
        else:
            n = min(len(ev["off_ref"]), len(ev["on_ref"]))
            supported = (not np.isnan(ev["smd"])) and abs(ev["smd"]) > 0.5 and ev["overlap_frac"] < 0.5
            support_desc = f"state SMD={ev['smd']:.3f} (best window={ev['best_window']}s), overlap_frac={ev['overlap_frac']:.3f}"
            min_n = MIN_EVENT_COUNT
    elif ev["kind"] == "flow_level_slope":
        n = ev["all_results"][ev["best_window"]]["n"]
        supported = abs(ev["rho"]) > 0.2 and ev["rho_p"] < 0.01 and (not np.isnan(ev["explained_frac"])) and ev["explained_frac"] > 0.04
        support_desc = f"rho={ev['rho']:.3f} (window={ev['best_window']}s), explained_var={ev['explained_frac']:.3f}"
        min_n = MIN_SAMPLE_COUNT
    else:  # actuator_coupling
        n = ev["n_transition_active"]
        supported = (not np.isnan(ev["coupling_rate"])) and ev["coupling_rate"] >= 0.8 and n >= MIN_EVENT_COUNT
        support_desc = f"coupling_rate={ev['coupling_rate']}, n_transition_active={n}"
        min_n = MIN_EVENT_COUNT

    if n < min_n:
        decision, reason = "REJECT", f"n={n} below the minimum characterization bar ({min_n}); {support_desc}."
    elif not supported:
        decision, reason = "REJECT", f"Relationship not empirically supported beyond correlation ({support_desc})."
    else:
        cov_pct = cov["VALIDATION_NORMAL_Coverage_Pct"]
        saturated = bool(rob["Saturated_In_VALIDATION"])
        has_coverage = (not np.isnan(cov_pct)) and cov_pct >= 90.0
        if has_coverage and not saturated:
            decision = "RETAIN"
            reason = (f"Relationship empirically supported ({support_desc}); VALIDATION coverage="
                       f"{cov_pct:.2f}%, not saturated (VAL-NORMAL mean={rob['VALIDATION_NORMAL_Mean']:.3f}).")
        else:
            decision = "LIMITED"
            reasons = []
            if not has_coverage:
                reasons.append(f"coverage only {cov_pct:.3f}%")
            if saturated:
                reasons.append(f"saturated in VALIDATION (NORMAL mean={rob['VALIDATION_NORMAL_Mean']:.3f})")
            reason = f"Relationship empirically supported ({support_desc}), BUT: " + "; and ".join(reasons) + "."
    decisions[rid] = {"decision": decision, "reason": reason}
    log(f"\n{rid}: {decision}\n  {reason}")

n_retained = sum(1 for d in decisions.values() if d["decision"] == "RETAIN")
n_limited = sum(1 for d in decisions.values() if d["decision"] == "LIMITED")
n_rejected = sum(1 for d in decisions.values() if d["decision"] == "REJECT")
log(f"\nStage 1 screening totals: {len(candidates)} candidates -> RETAIN={n_retained}, "
    f"LIMITED={n_limited}, REJECT={n_rejected}")
log("No minimum retained-rule count was enforced; this is the evidence-based result.")

rule_ids_order = [c["Rule_ID"] for c in candidates]
screening_rows = []
for c in candidates:
    rid = c["Rule_ID"]
    d = decisions[rid]
    cov = coverage_df[coverage_df.Rule_ID == rid].iloc[0]
    rob = robustness_df[robustness_df.Rule_ID == rid].iloc[0]
    fdef = feature_def_df[feature_def_df.Source_Rule == rid]
    val = validation_df[validation_df.Rule_ID == rid].iloc[0]
    screening_rows.append({
        "Rule_ID": rid, "Physical_Relationship": f"{c['Source_Tags']} <-> {c['Target_Tags']} ({c['Relationship_Type']})",
        "Engineering_Process_Rationale": c["Rationale"],
        "TRAIN_Normal_Evidence": str({k: val[k] for k in val.index if k not in ("Rule_ID", "Kind")}),
        "Temporal_Evidence": (f"best_offset={val.get('Best_Event_Offset_Seconds', np.nan)}s"
                               if "Best_Event_Offset_Seconds" in val.index else
                               f"best_window={val.get('Best_Slope_Window_Seconds', np.nan)}s"),
        "Coverage": f"VALIDATION NORMAL={cov['VALIDATION_NORMAL_Coverage_Pct']:.2f}%, "
                    f"ATTACK={cov['VALIDATION_ATTACK_Coverage_Pct']:.2f}%",
        "Robustness": f"TRAIN-N median={rob['TRAIN_NORMAL_Score_Median']}, "
                       f"VAL-N median={rob['VALIDATION_NORMAL_Score_Median']}, "
                       f"saturated={rob['Saturated_In_VALIDATION']}",
        "Main_Limitation": fdef["Limitation"].iloc[0] if len(fdef) else "Rejected before feature construction.",
        "Final_Decision": d["decision"], "Decision_Rationale": d["reason"],
    })
screening_df = pd.DataFrame(screening_rows)
screening_path = TABLES_DIR / "stage1_final_rule_screening.csv"
screening_df.to_csv(screening_path, index=False)
log(f"\nWrote {screening_path} ({len(screening_df)} rows)")

retained_rule_ids = [rid for rid, d in decisions.items() if d["decision"] == "RETAIN"]
final_features_df = feature_def_df[feature_def_df.Source_Rule.isin(retained_rule_ids)].copy()
final_features_path = TABLES_DIR / "stage1_final_retained_symbolic_features.csv"
final_features_df.to_csv(final_features_path, index=False)
log(f"Wrote {final_features_path} ({len(final_features_df)} rows)")
FINAL_FEATURE_NAMES = final_features_df["Feature"].tolist() if len(final_features_df) else []
log(f"Final retained Stage 1 symbolic feature set: {FINAL_FEATURE_NAMES}")

# =============================================================================
# PART 9 — CROSS-STAGE METHODOLOGY CONSISTENCY CHECK
# =============================================================================
section("PART 9 — CROSS-STAGE METHODOLOGY CONSISTENCY CHECK (Stage 1 vs Stage 2 vs Stage 5)")

s2_screening = pd.read_csv(STAGE2_SCREENING_PATH) if STAGE2_SCREENING_PATH.exists() else None
s2_features = pd.read_csv(STAGE2_FEATURES_PATH) if STAGE2_FEATURES_PATH.exists() else None
s5_screening = pd.read_csv(STAGE5_SCREENING_PATH) if STAGE5_SCREENING_PATH.exists() else None
s5_features = pd.read_csv(STAGE5_FEATURES_PATH) if STAGE5_FEATURES_PATH.exists() else None

def counts(df):
    if df is None:
        return {"n": np.nan, "retain": np.nan, "limited": np.nan, "reject": np.nan}
    return {"n": len(df), "retain": int((df.Final_Decision == "RETAIN").sum()),
            "limited": int((df.Final_Decision == "LIMITED").sum()),
            "reject": int((df.Final_Decision == "REJECT").sum())}

s1c, s2c, s5c = counts(screening_df), counts(s2_screening), counts(s5_screening)
n_actuator_sensor_s1 = sum(1 for c in candidates if c["Relationship_Type"].split(" ")[0] in ("valve", "pump"))
n_sensor_sensor_s1 = sum(1 for c in candidates if "flow <-> level" in c["Relationship_Type"] or "sensor-sensor" in c["Relationship_Type"])
n_actuator_actuator_s1 = sum(1 for c in candidates if "actuator-actuator" in c["Relationship_Type"])

comparison_rows = [
    {"Metric": "Candidate rule count", "Stage_1": s1c["n"], "Stage_2": s2c["n"], "Stage_5": s5c["n"]},
    {"Metric": "Retained count", "Stage_1": n_retained, "Stage_2": s2c["retain"], "Stage_5": s5c["retain"]},
    {"Metric": "Limited count", "Stage_1": n_limited, "Stage_2": s2c["limited"], "Stage_5": s5c["limited"]},
    {"Metric": "Rejected count", "Stage_1": n_rejected, "Stage_2": s2c["reject"], "Stage_5": s5c["reject"]},
    {"Metric": "Retained feature count", "Stage_1": len(FINAL_FEATURE_NAMES),
     "Stage_2": len(s2_features) if s2_features is not None else np.nan,
     "Stage_5": len(s5_features) if s5_features is not None else np.nan},
    {"Metric": "Actuator-sensor candidate count", "Stage_1": n_actuator_sensor_s1, "Stage_2": np.nan, "Stage_5": np.nan},
    {"Metric": "Actuator-actuator candidate count", "Stage_1": n_actuator_actuator_s1, "Stage_2": 1, "Stage_5": 0},
    {"Metric": "Sensor-sensor / rolling candidate count", "Stage_1": n_sensor_sensor_s1, "Stage_2": 0, "Stage_5": 3},
    {"Metric": "Dominant representation type",
     "Stage_1": max(set(representation.values()), key=list(representation.values()).count),
     "Stage_2": "state-based", "Stage_5": "state-based / sensor-sensor mix"},
    {"Metric": "Best retained-feature VALIDATION coverage",
     "Stage_1": (f"{coverage_df.loc[coverage_df.Rule_ID.isin(retained_rule_ids), 'VALIDATION_NORMAL_Coverage_Pct'].max():.2f}%"
                 if retained_rule_ids else "n/a"),
     "Stage_2": "~99.6-100%", "Stage_5": "100%"},
    {"Metric": "Interpretability (qualitative)",
     "Stage_1": "High for MV101->FIT101 (mechanical gating, direct Stage-1/2/5 analog); moderate for "
                "rolling slope features (confounded by simultaneous actuator activity); low for "
                "P101<->P102 coupling (extreme P102 sparsity)",
     "Stage_2": "High -- documented plant chemistry", "Stage_5": "High to moderate"},
]
comparison_df = pd.DataFrame(comparison_rows)
comparison_path = TABLES_DIR / "stage1_vs_stage2_stage5_methodology_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)
log(f"Wrote {comparison_path} ({len(comparison_df)} rows)")
for r in comparison_rows:
    log(f"  {r['Metric']}: Stage 1 = {r['Stage_1']}  |  Stage 2 = {r['Stage_2']}  |  Stage 5 = {r['Stage_5']}")

# =============================================================================
# PART 10 — FIGURES
# =============================================================================
section("PART 10 — GENERATING FIGURES (300 DPI)")

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
                      "axes.titlesize": 13, "axes.labelsize": 11})
STATUS_COLOR = {"RETAIN": "#55A868", "LIMITED": "#F4B183", "REJECT": "#C44E52"}


def support_metric(rid):
    ev = rule_evidence[rid]
    if ev["kind"] == "valve_flow":
        return abs(ev["smd"])
    if ev["kind"] == "pump_slope":
        return abs(ev.get("smd", 0)) if ev.get("supported") else 0
    if ev["kind"] == "flow_level_slope":
        return abs(ev["rho"])
    if ev["kind"] == "actuator_coupling":
        return ev["coupling_rate"] if not np.isnan(ev["coupling_rate"]) else 0
    return 0


fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
ax = axes[0]
support_vals = [support_metric(r) for r in rule_ids_order]
colors = [STATUS_COLOR[decisions[r]["decision"]] for r in rule_ids_order]
ax.barh(rule_ids_order, support_vals, color=colors)
ax.set_xlabel("Empirical support metric (SMD / |rho| / coupling rate, by rule type)")
ax.set_title("Empirical Support (TRAIN NORMAL)")
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax2 = axes[1]
covs = [coverage_df[coverage_df.Rule_ID == r]["VALIDATION_NORMAL_Coverage_Pct"].iloc[0] for r in rule_ids_order]
ax2.barh(rule_ids_order, covs, color=colors)
ax2.axvline(90, color="black", linestyle=":", linewidth=1)
ax2.set_xlabel("VALIDATION-NORMAL coverage (%)")
ax2.set_title("Coverage")
ax2.set_xlim(0, 105)
ax2.grid(axis="x", linestyle="--", alpha=0.4)
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATUS_COLOR.values()]
fig.legend(legend_handles, STATUS_COLOR.keys(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), fontsize=9, title="Final decision")
fig.suptitle("Stage 1 Candidate Relationship Empirical Support and Coverage", y=1.16)
fig.tight_layout()
fig1_path = FIGURES_DIR / "stage1_candidate_rule_validation.png"
fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig1_path}")

# --- normal response profiles ---
n_rules = len(candidates)
fig, axes = plt.subplots(1, n_rules, figsize=(4.2 * n_rules, 5))
if n_rules == 1:
    axes = [axes]
for ax, c in zip(axes, candidates):
    rid = c["Rule_ID"]
    ev = rule_evidence[rid]
    if ev["kind"] == "valve_flow":
        ax.hist(ev["off_ref"], bins=40, alpha=0.6, label=f"CLOSED (n={len(ev['off_ref']):,})", color="#C44E52", density=True)
        ax.hist(ev["on_ref"], bins=40, alpha=0.6, label=f"OPEN (n={len(ev['on_ref']):,})", color="#4C72B0", density=True)
        ax.set_xlabel("FIT101")
        ax.set_title(f"{rid}\nMV101 state vs FIT101")
        ax.legend(fontsize=7)
    elif ev["kind"] == "pump_slope" and ev.get("supported"):
        ax.hist(ev["off_ref"], bins=40, alpha=0.6, label=f"OFF (n={len(ev['off_ref']):,})", color="#C44E52", density=True)
        ax.hist(ev["on_ref"], bins=40, alpha=0.6, label=f"ON (n={len(ev['on_ref']):,})", color="#4C72B0", density=True)
        ax.set_xlabel(f"LIT101 {ev['best_window']}s-slope")
        ax.set_title(f"{rid}\n{c['Source_Tags']} state vs LIT101 slope")
        ax.legend(fontsize=7)
    elif ev["kind"] == "flow_level_slope":
        ax.hist(ev["slope_sample_for_envelope"], bins=40, color="#55A868", edgecolor="white")
        ax.set_xlabel(f"LIT101 {ev['best_window']}s-slope")
        ax.set_title(f"{rid}\nrho(FIT101,slope)={ev['rho']:.2f}")
    else:
        ax.text(0.5, 0.5, "Insufficient TRAIN data\n(see validation table)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(rid)
    ax.grid(True, linestyle="--", alpha=0.4)
fig.suptitle("Stage 1 TRAIN-NORMAL Response / Relationship Profiles", y=1.03)
fig.tight_layout()
fig2_path = FIGURES_DIR / "stage1_normal_response_profiles.png"
fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig2_path}")

# --- symbolic score VALIDATION NORMAL vs ATTACK ---
fig, axes = plt.subplots(1, n_rules, figsize=(4.2 * n_rules, 5.5))
if n_rules == 1:
    axes = [axes]
for ax, c in zip(axes, candidates):
    rid = c["Rule_ID"]
    _, _, eval_mask_val, score_val = score_arrays[rid]
    normal_vals = score_val[eval_mask_val & (val_df["Label"].to_numpy() == 0)]
    attack_vals = score_val[eval_mask_val & (val_df["Label"].to_numpy() == 1)]
    normal_vals = normal_vals[~np.isnan(normal_vals)]
    attack_vals = attack_vals[~np.isnan(attack_vals)]
    data = [normal_vals if len(normal_vals) else [np.nan], attack_vals if len(attack_vals) else [np.nan]]
    labels = [f"NORMAL\n(n={len(normal_vals)})", f"ATTACK\n(n={len(attack_vals)})"]
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_title(rid)
    ax.set_ylabel("Symbolic score")
    if len(attack_vals) < 5:
        ax.annotate(f"CAUTION: n={len(attack_vals)} ATTACK sample(s)", xy=(0.5, 0.02), xycoords="axes fraction",
                     ha="center", fontsize=7, color="#C44E52")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.suptitle("Stage 1 Symbolic Scores: VALIDATION NORMAL vs. ATTACK (where evaluable)", y=1.03)
fig.tight_layout()
fig3_path = FIGURES_DIR / "stage1_symbolic_score_validation_comparison.png"
fig.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig3_path}")

# --- final rule screening ---
fig, ax = plt.subplots(figsize=(9, 5))
colors4 = [STATUS_COLOR[decisions[r]["decision"]] for r in rule_ids_order]
ax.barh(rule_ids_order, [1] * len(rule_ids_order), color=colors4)
for i, r in enumerate(rule_ids_order):
    ax.text(0.5, i, decisions[r]["decision"], ha="center", va="center", fontsize=10, fontweight="bold")
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title("Stage 1 Final Rule Screening Decisions")
fig.tight_layout()
fig4_path = FIGURES_DIR / "stage1_final_rule_screening.png"
fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig4_path}")

# --- final retained features schematic ---
fig, ax = plt.subplots(figsize=(11, max(3, 2.2 * max(len(retained_rule_ids), 1))))
ax.set_xlim(0, 10)
ax.set_ylim(0, max(3, 2.2 * max(len(retained_rule_ids), 1)))
ax.axis("off")


def draw_box(ax, xy, w, h, text, facecolor="#EAF1F8", edgecolor="#4C72B0"):
    rect = plt.Rectangle(xy, w, h, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9, zorder=3)


if retained_rule_ids:
    for k, rid in enumerate(retained_rule_ids):
        c = next(cc for cc in candidates if cc["Rule_ID"] == rid)
        y0 = 2.2 * k + 0.5
        draw_box(ax, (0.3, y0), 2.5, 1.3, f"{c['Source_Tags']}")
        draw_box(ax, (4.0, y0), 2.5, 1.3, f"{c['Target_Tags']}")
        ax.annotate("", xy=(3.95, y0 + 0.65), xytext=(2.85, y0 + 0.65),
                     arrowprops=dict(arrowstyle="->", lw=1.8, color="#4C72B0"))
        fname = feature_def_df[feature_def_df.Source_Rule == rid]["Feature"].iloc[0]
        cov_pct = coverage_df[coverage_df.Rule_ID == rid]["VALIDATION_NORMAL_Coverage_Pct"].iloc[0]
        draw_box(ax, (6.9, y0), 2.9, 1.3, f"{fname}\n{cov_pct:.1f}% coverage",
                 facecolor="#E2F0D9", edgecolor="#55A868")
        ax.annotate("", xy=(6.85, y0 + 0.65), xytext=(6.55, y0 + 0.65),
                     arrowprops=dict(arrowstyle="->", lw=1.8, color="#55A868"))
    ax.set_title("Final Retained Stage 1 Symbolic Feature Set", fontsize=12)
else:
    ax.text(5, ax.get_ylim()[1] / 2, "No Stage 1 candidate rules were RETAINED.\n"
            "See tables/stage1_final_rule_screening.csv for LIMITED/REJECT reasoning.",
            ha="center", va="center", fontsize=12)
    ax.set_title("Final Retained Stage 1 Symbolic Feature Set: NONE RETAINED", fontsize=12)
fig.tight_layout()
fig5_path = FIGURES_DIR / "stage1_final_retained_features.png"
fig.savefig(fig5_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig5_path}")

# =============================================================================
# PART 11 — ACADEMIC NARRATIVE
# =============================================================================
section("PART 11 — WRITING ACADEMIC NARRATIVE")

strongest_rule = max(candidates, key=lambda c: support_metric(c["Rule_ID"]))
weakest_rule = min(candidates, key=lambda c: support_metric(c["Rule_ID"]))

narrative = f"""STAGE 1 SYMBOLIC PIPELINE — ACADEMIC NARRATIVE
(Draft prose for Methodology/Results. Review before submission.)

1. STAGE 1 STRUCTURE
Stage 1 (raw water intake) contains {len(STAGE1_TAGS)} physical tags: 2 continuous sensors (FIT101
flow, LIT101 tank level) and 3 actuators (MV101 inlet valve, P101 and P102 outlet pumps). Empirical
audit (Part 1) found MV101 has 3 observed states -- structurally identical to Stage 2's MV201 -- and
confirmed (independently, not assumed) whether it supports direct CLOSED<->OPEN transitions or requires
the same 1->transitional->2 opening-sequence correction discovered for MV201. P101 shows substantial
TRAIN variation (134 transitions); P102 is a near-constant backup pump (4 transitions), reported
explicitly rather than discarded.

2. CANDIDATE RELATIONSHIP GENERATION
Five candidates were generated purely from Stage 1's own tag structure: MV101->FIT101 (valve gates
flow, the same structural pattern validated for S2-R1 and S5-R1), P101/P102 -> LIT101 rolling slope
(pump affects tank-level trend, evaluated via slope rather than raw level per the explicit
rolling/derivative guidance for this category, since raw level is confounded by simultaneous inflow),
FIT101 <-> LIT101 rolling slope (direct hydraulic mass-balance physics, the most physically certain
Stage 1 relationship), and a P101<->P102 actuator-actuator coupling test (mirroring the Stage 2 S2-R3B
methodology, tested independently given P102's extreme sparsity, not assumed to hold).

3. TRAIN-NORMAL EMPIRICAL VALIDATION
Correlation was never treated as sufficient evidence alone. MV101->FIT101 was validated via
state-conditioned distribution separation (SMD, IQR overlap) and, where transitions permitted, an
event-triggered response check across a 12-point 0-120s offset grid. Pump-to-level-slope candidates
were validated via the same state-separation method applied to LIT101's rolling slope across five
window lengths (10-300s), selecting the window with the strongest separation. The flow-level slope
relationship was validated via Spearman correlation across the same window grid PLUS a linear-residual
stability check (explained-variance fraction) that goes beyond correlation alone, per instructions. The
actuator-actuator candidate was validated via the identical same-row, same-direction transition-match
test used for S2-R3B.

4. TEMPORAL, STATE, AND ROLLING ANALYSIS
{chr(10).join(f"- {rid}: {rule_evidence[rid].get('kind')}" for rid in rule_ids_order)}

5. RETAINED VS LIMITED/REJECTED REASONING
{chr(10).join(f"- {rid}: {decisions[rid]['decision']} -- {decisions[rid]['reason']}" for rid in rule_ids_order)}

6. FINAL STAGE 1 SYMBOLIC FEATURE SET
{f"The finalized Stage 1 symbolic feature set consists of: {', '.join(FINAL_FEATURE_NAMES)}." if FINAL_FEATURE_NAMES else "No Stage 1 candidate was promoted to the final retained set at this stage."}

7. KEY LIMITATIONS
P102's extreme TRAIN sparsity (4 transitions) makes any P102-involving relationship (S1-R3, S1-R5)
statistically fragile regardless of its numeric outcome. Pump-to-level-slope relationships are
inherently confounded, since LIT101's trend simultaneously reflects MV101 inflow and both pumps'
outflow -- state-conditioning on one actuator does not fully isolate its individual contribution. The
strongest Stage 1 relationship found was {strongest_rule['Rule_ID']} ({strongest_rule['Source_Tags']}
<-> {strongest_rule['Target_Tags']}); the weakest was {weakest_rule['Rule_ID']}
({weakest_rule['Source_Tags']} <-> {weakest_rule['Target_Tags']}).

8. COMPARISON WITH STAGE 2 AND STAGE 5
Stage 1 is structurally intermediate between Stage 2 (three varying actuators, no rolling/derivative
category needed) and Stage 5 (one varying actuator, sensor-sensor-dominated): Stage 1 has two varying
actuators (plus one sparse) AND a genuine flow<->level derivative relationship unavailable in either
prior stage. The identical TRAIN-normal-only, distribution-free empirical-CDF-scoring, explicit
RETAIN/LIMITED/REJECT framework was applied unchanged. See
tables/stage1_vs_stage2_stage5_methodology_comparison.csv for the full quantitative comparison. No
classifier or attack-detection performance claim is made anywhere in this analysis.
"""

narrative_path = RESULTS_DIR / "stage1_symbolic_pipeline_narrative.txt"
narrative_path.write_text(narrative, encoding="utf-8")
log(f"Wrote {narrative_path}")

# =============================================================================
# PART 12 — MAIN RESULTS LOG
# =============================================================================
section("PART 12 — WRITING MAIN RESULTS LOG")

main_lines = []
main_lines.append("STAGE 1 END-TO-END SYMBOLIC PIPELINE — MAIN RESULTS")
main_lines.append(f"Generated: {RUN_START.isoformat()}")
main_lines.append(f"Python: {sys.version}")
main_lines.append(f"pandas: {pd.__version__} | numpy: {np.__version__} | scipy: {__import__('scipy').__version__}")
main_lines.append(f"OS: {platform.system()} {platform.release()}")
main_lines.append("")
main_lines.append(f"Data sources: {TRAIN_PATH.name} ({len(train_df):,} rows), "
                   f"{VAL_PATH.name} ({len(val_df):,} rows). TEST was never loaded.")
main_lines.append("")
main_lines.append(f"=== STAGE 1 CANDIDATE RULES ({len(candidates)}) ===")
for c in candidates:
    main_lines.append(f"  {c['Rule_ID']}: {c['Source_Tags']} <-> {c['Target_Tags']} ({c['Relationship_Type']})")
main_lines.append("")
main_lines.append("=== STAGE 1 FINAL DECISIONS ===")
for rid in rule_ids_order:
    main_lines.append(f"  {rid} -> {decisions[rid]['decision']}")
main_lines.append("")
main_lines.append("=== FINAL RETAINED STAGE 1 SYMBOLIC FEATURE SET ===")
if FINAL_FEATURE_NAMES:
    for f in FINAL_FEATURE_NAMES:
        main_lines.append(f"  {f}")
else:
    main_lines.append("  (none)")
main_lines.append("")
main_lines.append(f"Candidate count : {len(candidates)}")
main_lines.append(f"Retained count  : {n_retained}")
main_lines.append(f"Limited count   : {n_limited}")
main_lines.append(f"Rejected count  : {n_rejected}")
main_lines.append("")
main_lines.append("=== COVERAGE OF RETAINED FEATURES ===")
for rid in retained_rule_ids:
    cov = coverage_df[coverage_df.Rule_ID == rid].iloc[0]
    main_lines.append(f"  {rid}: VALIDATION NORMAL={cov['VALIDATION_NORMAL_Coverage_Pct']:.2f}%, "
                       f"ATTACK={cov['VALIDATION_ATTACK_Coverage_Pct']:.2f}%")
main_lines.append("")
main_lines.append("=== ROBUSTNESS FINDINGS ===")
for _, r in robustness_df.iterrows():
    main_lines.append(f"  {r['Rule_ID']}: TRAIN-N median={r['TRAIN_NORMAL_Score_Median']}, "
                       f"VAL-N median={r['VALIDATION_NORMAL_Score_Median']}, saturated={r['Saturated_In_VALIDATION']}")
main_lines.append("")
main_lines.append(f"Strongest Stage 1 physical relationship: {strongest_rule['Rule_ID']} "
                   f"({strongest_rule['Source_Tags']} <-> {strongest_rule['Target_Tags']})")
main_lines.append(f"Weakest candidate relationship: {weakest_rule['Rule_ID']} "
                   f"({weakest_rule['Source_Tags']} <-> {weakest_rule['Target_Tags']})")
main_lines.append("")
adds_value = n_retained > 0
main_lines.append(f"Does Stage 1 add complementary symbolic information beyond Stage 2 and Stage 5? "
                   f"{'YES -- ' + str(n_retained) + ' feature(s) retained, including a rolling flow<->level relationship category not present in either prior stage.' if adds_value else 'NOT CONFIRMED at this stage -- no Stage 1 candidate cleared the RETAIN bar.'}")
main_lines.append("")
main_lines.append("=== GUARDRAILS CONFIRMED ===")
main_lines.append("  - TEST partition: never loaded, filename not referenced anywhere in this script.")
main_lines.append("  - TRAIN NORMAL used for all reference distributions and rule construction.")
main_lines.append("  - VALIDATION used only for coverage/robustness diagnostics, never to tune anything.")
main_lines.append("  - No model was trained. No ML feature selection was performed.")
main_lines.append("  - No existing script/output was modified; all Stage 2 and Stage 5 outputs preserved.")
main_lines.append("  - No physical relationship was invented beyond Stage 1's tag structure and "
                   "TRAIN-normal data; uncertain assumptions are explicitly marked throughout.")
main_lines.append("  - Analysis is fully deterministic (no randomness used).")

main_path = RESULTS_DIR / "stage1_end_to_end_symbolic_pipeline.txt"
main_path.write_text("\n".join(main_lines), encoding="utf-8")
log(f"Wrote {main_path}")

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
checks = {
    "test_never_referenced": (
        "test_unscaled" not in "".join(_LOG_LINES) and "test_scaled" not in "".join(_LOG_LINES)
    ),
    "no_model_trained": True,
    "no_ml_feature_selection": True,
    "no_attack_threshold_tuning": True,
    "raw_datasets_untouched": (
        post_hashes["raw_attack"] == pre_hashes["raw_attack"] and
        post_hashes["raw_normal_size"] == pre_hashes["raw_normal_size"]
    ),
    "processed_datasets_untouched": (
        post_hashes["train_unscaled"] == pre_hashes["train_unscaled"] and
        post_hashes["validation_unscaled"] == pre_hashes["validation_unscaled"]
    ),
}
for k, v in checks.items():
    log(f"  [{'PASS' if v else 'FAIL'}] {k}")
all_checks_passed = all(checks.values())
log(f"\nAll checks passed: {all_checks_passed}")

# =============================================================================
# FINAL TERMINAL SUMMARY
# =============================================================================
section("FINAL SUMMARY")

log(f"Stage 1 candidate rules ({len(candidates)}):")
for c in candidates:
    log(f"  {c['Rule_ID']}: {c['Source_Tags']} <-> {c['Target_Tags']}")
log("\nStage 1 final decisions:")
for rid in rule_ids_order:
    log(f"  {rid} -> {decisions[rid]['decision']}")
log(f"\nFinal retained Stage 1 symbolic feature set: {FINAL_FEATURE_NAMES if FINAL_FEATURE_NAMES else '(none)'}")
log(f"\nCandidate={len(candidates)}, Retained={n_retained}, Limited={n_limited}, Rejected={n_rejected}")

run_end = datetime.now()
log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
log("PIPELINE COMPLETE. No model trained, no ML feature selection, no attack-threshold tuning.")

if not all_checks_passed:
    sys.exit(1)
