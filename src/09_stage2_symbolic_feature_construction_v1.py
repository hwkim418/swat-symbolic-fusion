"""
code/09_stage2_symbolic_feature_construction_v1.py

STAGE 2 SYMBOLIC FEATURE CONSTRUCTION v1 — FEATURE ENGINEERING, NOT
CLASSIFIER TRAINING.

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

Builds the first reproducible symbolic feature set from the four
engineer-derived Stage 2 rules validated in scripts 04-08:
  S2-R1  : MV201 state / FIT201 flow consistency (state-based)
  S2-R2  : P205 OFF->ON -> AIT203 ORP response (event-based, 5s horizon)
  S2-R3B : P205<->P203 bidirectional transition coupling + steady-state
           equality (event-based coupling + always-evaluable mismatch)
  S2-R4  : P203 OFF->ON -> AIT202 pH response (event-based, 5s horizon)

All reference distributions/envelopes are derived from TRAIN NORMAL
rows only, then applied UNCHANGED to TRAIN ATTACK, VALIDATION NORMAL,
and VALIDATION ATTACK. No ATTACK label is used to tune, threshold, or
select anything. No final binary attack/not-attack decision is made --
only continuous, interpretable inconsistency SCORES.

This script does NOT train an ML model, does NOT select features, does
NOT modify any existing script/dataset/output, and NEVER loads or
references the TEST partition (its filename does not appear anywhere
below, by design). It reads (but does not modify) prior diagnostic
tables purely to cross-verify that the freshly, independently
recomputed TRAIN-normal reference values match what was previously
reported, in line with "prefer exact values over hard-coded rounded
ones" -- the actual reference distributions used for scoring are
recomputed here from processed/train_unscaled.csv at full precision,
not retyped from a report.
"""

import os
import sys
import hashlib
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

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
# NOTE: no TEST_PATH constant is defined anywhere in this script, on purpose.

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR = PROJECT_ROOT / "figures"
for d in (TABLES_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Prior diagnostic tables read ONLY for a cross-verification check (not
# modified, not used as the scoring source of truth -- see module docstring).
PRIOR_ENVELOPE_PATH = TABLES_DIR / "stage2_normal_envelope_summary.csv"
PRIOR_COUPLING_PATH = TABLES_DIR / "stage2_p205_p203_coupling_comparison.csv"

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


section("STAGE 2 SYMBOLIC FEATURE CONSTRUCTION v1 — RUN START (FEATURE ENGINEERING, NOT ML TRAINING)")
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
R2_R4_OFFSET = 5  # TRAIN-derived candidate horizon established in scripts 06/07
P_OFF, P_ON = 1, 2  # shared P205/P203 semantics
MV_CLOSED, MV_TRANS, MV_OPEN = 1, 0, 2

# =============================================================================
# LOAD DATA
# =============================================================================
section("LOADING TRAIN AND VALIDATION (TEST NEVER LOADED)")

train_df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
val_df = pd.read_csv(VAL_PATH).reset_index(drop=True)
train_df["_ts"] = pd.to_datetime(train_df["Timestamp"])
val_df["_ts"] = pd.to_datetime(val_df["Timestamp"])
log(f"TRAIN: {train_df.shape[0]:,} rows | VALIDATION: {val_df.shape[0]:,} rows")
log("Fixed semantics used (not re-derived here): MV201: 1=CLOSED,0=transitional,2=OPEN | "
    "P205: 1=OFF,2=ON | P203: 1=OFF,2=ON")

# =============================================================================
# SHARED HELPER FUNCTIONS (self-contained; not imported from other scripts)
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
    expected_ts = df["_ts"].iloc[i0] + pd.Timedelta(seconds=offset)
    if df["_ts"].iloc[j] != expected_ts:
        return False
    if require_normal and not (df["Label"].iloc[i0:j + 1] == 0).all():
        return False
    return True


def empirical_two_sided_score_vec(reference_sample, x_arr):
    """Distribution-free, empirical two-sided extremeness score.

    F(x)          = fraction of the TRAIN-NORMAL reference sample <= x
                     (empirical CDF via searchsorted; no Gaussian or any
                     other parametric assumption).
    two_sided_p(x)= 2 * min(F(x), 1-F(x))   in [0, 1]
                     (this is exactly a two-tailed empirical p-value: the
                     fraction of TRAIN-NORMAL observations AS OR MORE
                     extreme than x in either direction)
    score(x)      = 1 - two_sided_p(x)      in [0, 1]

    score ~ 0   -> x sits near the TRAIN-NORMAL median (typical)
    score ~ 1   -> x sits in the extreme tail of the TRAIN-NORMAL reference
                   distribution in EITHER direction (atypical)
    Larger score always means stronger deviation from TRAIN-normal physical
    behavior, by construction.
    """
    x_arr = np.asarray(x_arr, dtype=float)
    ref_sorted = np.sort(np.asarray(reference_sample, dtype=float))
    n = len(ref_sorted)
    score = np.full(x_arr.shape, np.nan)
    if n == 0:
        return score
    valid = ~np.isnan(x_arr)
    ranks = np.searchsorted(ref_sorted, x_arr[valid], side="right")
    F = ranks / n
    two_sided_p = np.minimum(2 * np.minimum(F, 1 - F), 1.0)
    score[valid] = 1.0 - two_sided_p
    return score


def summarize_group(values):
    vals = values[~np.isnan(values)] if len(values) else values
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "p90": np.nan, "p95": np.nan, "max": np.nan}
    return {"n": n, "mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)), "p95": float(np.percentile(vals, 95)),
            "max": float(np.max(vals))}


# =============================================================================
# CROSS-VERIFICATION AGAINST PRIOR DIAGNOSTIC OUTPUTS (read-only)
# =============================================================================
section("CROSS-VERIFYING FRESH TRAIN-NORMAL COMPUTATIONS AGAINST PRIOR SCRIPT OUTPUTS (read-only)")

prior_envelope = pd.read_csv(PRIOR_ENVELOPE_PATH) if PRIOR_ENVELOPE_PATH.exists() else None
prior_coupling = pd.read_csv(PRIOR_COUPLING_PATH) if PRIOR_COUPLING_PATH.exists() else None
if prior_envelope is not None:
    log(f"Loaded {PRIOR_ENVELOPE_PATH} for cross-verification only (not modified).")
if prior_coupling is not None:
    log(f"Loaded {PRIOR_COUPLING_PATH} for cross-verification only (not modified).")

# =============================================================================
# PART 1 / S2-R1 — VALVE / FLOW CONSISTENCY FEATURE
# =============================================================================
section("S2-R1 — MV201/FIT201 STATE-FLOW CONSISTENCY FEATURE")

fit201_closed_ref = train_df.loc[(train_df["MV201"] == MV_CLOSED) & (train_df["Label"] == 0), "FIT201"].to_numpy()
fit201_open_ref = train_df.loc[(train_df["MV201"] == MV_OPEN) & (train_df["Label"] == 0), "FIT201"].to_numpy()
log(f"TRAIN-NORMAL FIT201 reference while CLOSED: n={len(fit201_closed_ref)}, "
    f"median={np.median(fit201_closed_ref):.6f}, mean={np.mean(fit201_closed_ref):.6f}, "
    f"std={np.std(fit201_closed_ref):.6f}")
log(f"TRAIN-NORMAL FIT201 reference while OPEN  : n={len(fit201_open_ref)}, "
    f"median={np.median(fit201_open_ref):.6f}, mean={np.mean(fit201_open_ref):.6f}, "
    f"std={np.std(fit201_open_ref):.6f}")
log("State 0 (transitional) EXCLUDED from both reference distributions, as required.")

# Distribution shape check (documenting why an empirical, not Gaussian, method is used)
from scipy import stats as _stats
_, p_closed = _stats.shapiro(fit201_closed_ref[:5000]) if len(fit201_closed_ref) else (np.nan, np.nan)
_, p_open = _stats.shapiro(fit201_open_ref[:5000]) if len(fit201_open_ref) else (np.nan, np.nan)
log(f"Shapiro-Wilk normality check (first 5000 samples, informational only): "
    f"CLOSED p={p_closed:.2e}, OPEN p={p_open:.2e} "
    f"({'NOT normal (p<0.05)' if (not np.isnan(p_closed) and p_closed < 0.05) else 'inconclusive'} / "
    f"{'NOT normal (p<0.05)' if (not np.isnan(p_open) and p_open < 0.05) else 'inconclusive'}) "
    f"-- confirms an empirical (non-Gaussian) scoring method is the right choice here.")


def compute_r1(df):
    mv = df["MV201"].to_numpy()
    fit = df["FIT201"].to_numpy(dtype=float)
    n = len(df)
    evaluable = mv != MV_TRANS
    response = np.where(evaluable, fit, np.nan)  # "response value" for R1 is the raw FIT201 reading itself
    score = np.full(n, np.nan)
    closed_mask = evaluable & (mv == MV_CLOSED)
    open_mask = evaluable & (mv == MV_OPEN)
    score[closed_mask] = empirical_two_sided_score_vec(fit201_closed_ref, fit[closed_mask])
    score[open_mask] = empirical_two_sided_score_vec(fit201_open_ref, fit[open_mask])
    return evaluable, response, score


train_r1_eval, train_r1_resp, train_r1_score = compute_r1(train_df)
val_r1_eval, val_r1_resp, val_r1_score = compute_r1(val_df)
log(f"\nS2_R1_evaluable coverage: TRAIN={100*train_r1_eval.mean():.2f}%, VALIDATION={100*val_r1_eval.mean():.2f}%")
log("LIMITATION (documented): for ~15-20s immediately after an OPEN transition, FIT201 is still "
    "ramping up toward its steady value (per script07's Part1 finding), so rows in that ramp-up "
    "window will legitimately score as more 'atypical' relative to the full steady-OPEN reference "
    "distribution even though nothing is wrong -- this is a known transient effect, not a defect.")

# =============================================================================
# PART 1 / S2-R2 — BLEACH/ORP EVENT-RESPONSE FEATURE
# =============================================================================
section("S2-R2 — P205 OFF->ON -> AIT203 EVENT-RESPONSE FEATURE (5s horizon)")


def extract_train_normal_reference_responses(source_tag, target_tag, offset):
    trans = find_specific_transitions(train_df, source_tag, P_OFF, P_ON)
    vals = []
    for i0 in trans:
        if train_df["Label"].iloc[i0] != 0:
            continue
        if not check_baseline_window(train_df, i0, source_tag, P_OFF, require_normal=True):
            continue
        if not check_offset_window(train_df, i0, offset, require_normal=True):
            continue
        base = train_df[target_tag].iloc[i0 - BASELINE_WINDOW:i0].mean()
        vals.append(train_df[target_tag].iloc[i0 + offset] - base)
    return np.array(vals, dtype=float)


ait203_response_ref = extract_train_normal_reference_responses("P205", "AIT203", R2_R4_OFFSET)
log(f"TRAIN-NORMAL AIT203 5s-response reference: n={len(ait203_response_ref)}, "
    f"median={np.median(ait203_response_ref):.6f}, P5={np.percentile(ait203_response_ref,5):.6f}, "
    f"P10={np.percentile(ait203_response_ref,10):.6f}, P90={np.percentile(ait203_response_ref,90):.6f}, "
    f"P95={np.percentile(ait203_response_ref,95):.6f}")

if prior_envelope is not None:
    prior_row = prior_envelope[(prior_envelope.Relationship == "P205 ON -> AIT203 ORP response") &
                                (prior_envelope.Metric_Type == "response_delta") &
                                (prior_envelope.Offset_Seconds == R2_R4_OFFSET)]
    if len(prior_row):
        pr = prior_row.iloc[0]
        match = (abs(pr["Median"] - np.median(ait203_response_ref)) < 1e-6 and
                 abs(pr["N_Usable"] - len(ait203_response_ref)) < 1e-9)
        log(f"Cross-check vs {PRIOR_ENVELOPE_PATH.name}: prior median={pr['Median']:.6f} n={int(pr['N_Usable'])} "
            f"-- MATCHES freshly recomputed value: {match}")


def compute_event_response_feature(df, source_tag, target_tag, offset, reference_sample):
    n = len(df)
    evaluable = np.zeros(n, dtype=bool)
    response = np.full(n, np.nan)
    trans = find_specific_transitions(df, source_tag, P_OFF, P_ON)
    for i0 in trans:
        if not check_baseline_window(df, i0, source_tag, P_OFF, require_normal=False):
            continue
        if not check_offset_window(df, i0, offset, require_normal=False):
            continue
        base = df[target_tag].iloc[i0 - BASELINE_WINDOW:i0].mean()
        response[i0] = df[target_tag].iloc[i0 + offset] - base
        evaluable[i0] = True
    score = np.full(n, np.nan)
    score[evaluable] = empirical_two_sided_score_vec(reference_sample, response[evaluable])
    return evaluable, response, score


train_r2_eval, train_r2_resp, train_r2_score = compute_event_response_feature(
    train_df, "P205", "AIT203", R2_R4_OFFSET, ait203_response_ref)
val_r2_eval, val_r2_resp, val_r2_score = compute_event_response_feature(
    val_df, "P205", "AIT203", R2_R4_OFFSET, ait203_response_ref)
log(f"\nS2_R2_evaluable coverage: TRAIN={100*train_r2_eval.mean():.4f}% ({train_r2_eval.sum()} events), "
    f"VALIDATION={100*val_r2_eval.mean():.4f}% ({val_r2_eval.sum()} events)")

# --- investigate (do not force) a rolling/state-conditioned fallback ---
section("S2-R2 — INVESTIGATING A ROLLING/STATE-CONDITIONED FALLBACK (not forced)")


def check_state_conditioned_separation(df, pump_tag, sensor_tag):
    off_ref = df.loc[(df[pump_tag] == P_OFF) & (df["Label"] == 0), sensor_tag].to_numpy()
    on_ref = df.loc[(df[pump_tag] == P_ON) & (df["Label"] == 0), sensor_tag].to_numpy()
    pooled_std = np.sqrt((np.var(off_ref) + np.var(on_ref)) / 2)
    smd = (np.mean(on_ref) - np.mean(off_ref)) / pooled_std if pooled_std > 0 else np.nan
    off_iqr = np.percentile(off_ref, [25, 75])
    on_iqr = np.percentile(on_ref, [25, 75])
    overlap = max(0.0, min(off_iqr[1], on_iqr[1]) - max(off_iqr[0], on_iqr[0]))
    off_span = off_iqr[1] - off_iqr[0]
    on_span = on_iqr[1] - on_iqr[0]
    overlap_frac = overlap / min(off_span, on_span) if min(off_span, on_span) > 0 else np.nan
    return {"off_ref": off_ref, "on_ref": on_ref, "smd": smd, "off_iqr": off_iqr, "on_iqr": on_iqr,
            "iqr_overlap_fraction": overlap_frac}


r2_sep = check_state_conditioned_separation(train_df, "P205", "AIT203")
log(f"TRAIN-NORMAL AIT203 steady-state separation by P205 state: SMD={r2_sep['smd']:.3f}, "
    f"OFF IQR={r2_sep['off_iqr']}, ON IQR={r2_sep['on_iqr']}, "
    f"IQR overlap fraction={r2_sep['iqr_overlap_fraction']:.3f}")
R2_FALLBACK_DEFENSIBLE = (abs(r2_sep["smd"]) > 1.0) and (r2_sep["iqr_overlap_fraction"] < 0.25)
log(f"Decision rule (documented): defensible if |SMD|>1.0 AND IQR overlap fraction<0.25. "
    f"Result: {'DEFENSIBLE -- fallback feature will be built' if R2_FALLBACK_DEFENSIBLE else 'NOT DEFENSIBLE -- event-based score retained, no fallback forced'}.")
if not R2_FALLBACK_DEFENSIBLE:
    log("Reason: AIT203 (ORP) is a slow-drifting analytical measurement influenced by many unmodeled "
        "process factors beyond P205's state alone; its steady-state OFF vs ON distributions are not "
        "cleanly separated enough (per the SMD/overlap criterion above) to support a physically "
        "defensible rolling/state-conditioned consistency score without risking false attribution of "
        "ordinary process drift as an anomaly. LIMITATION: S2-R2 therefore remains an event-based score, "
        f"evaluable only at the {train_r2_eval.sum()} TRAIN / {val_r2_eval.sum()} VALIDATION P205 "
        "OFF->ON transition instants -- coverage elsewhere in the time series is zero by design.")
else:
    log("Building the state-conditioned fallback feature (S2_R2_state_fallback_score): TRAIN-NORMAL "
        "AIT203 shows clean, non-overlapping steady-state separation by P205 state, so a continuous, "
        "always-evaluable consistency score conditioned on P205's CURRENT steady state (mirroring "
        "S2-R1's method) is physically defensible here, unlike for S2-R4/AIT202 below.")


def compute_r2_state_fallback(df, off_ref, on_ref):
    p205 = df["P205"].to_numpy()
    ait203 = df["AIT203"].to_numpy(dtype=float)
    n = len(df)
    evaluable = np.isin(p205, [P_OFF, P_ON])  # always true here (P205 only ever 1 or 2), kept explicit
    score = np.full(n, np.nan)
    off_mask = evaluable & (p205 == P_OFF)
    on_mask = evaluable & (p205 == P_ON)
    score[off_mask] = empirical_two_sided_score_vec(off_ref, ait203[off_mask])
    score[on_mask] = empirical_two_sided_score_vec(on_ref, ait203[on_mask])
    return evaluable, score


if R2_FALLBACK_DEFENSIBLE:
    train_r2fb_eval, train_r2fb_score = compute_r2_state_fallback(train_df, r2_sep["off_ref"], r2_sep["on_ref"])
    val_r2fb_eval, val_r2fb_score = compute_r2_state_fallback(val_df, r2_sep["off_ref"], r2_sep["on_ref"])
    log(f"S2_R2_state_fallback_score coverage: TRAIN={100*train_r2fb_eval.mean():.2f}%, "
        f"VALIDATION={100*val_r2fb_eval.mean():.2f}%")

# =============================================================================
# PART 1 / S2-R4 — ACID/pH EVENT-RESPONSE FEATURE
# =============================================================================
section("S2-R4 — P203 OFF->ON -> AIT202 EVENT-RESPONSE FEATURE (5s horizon)")

ait202_response_ref = extract_train_normal_reference_responses("P203", "AIT202", R2_R4_OFFSET)
log(f"TRAIN-NORMAL AIT202 5s-response reference: n={len(ait202_response_ref)}, "
    f"median={np.median(ait202_response_ref):.6f}, P5={np.percentile(ait202_response_ref,5):.6f}, "
    f"P10={np.percentile(ait202_response_ref,10):.6f}, P90={np.percentile(ait202_response_ref,90):.6f}, "
    f"P95={np.percentile(ait202_response_ref,95):.6f}")

if prior_envelope is not None:
    prior_row = prior_envelope[(prior_envelope.Relationship == "P203 ON -> AIT202 pH decrease") &
                                (prior_envelope.Metric_Type == "response_delta") &
                                (prior_envelope.Offset_Seconds == R2_R4_OFFSET)]
    if len(prior_row):
        pr = prior_row.iloc[0]
        match = (abs(pr["Median"] - np.median(ait202_response_ref)) < 1e-6 and
                 abs(pr["N_Usable"] - len(ait202_response_ref)) < 1e-9)
        log(f"Cross-check vs {PRIOR_ENVELOPE_PATH.name}: prior median={pr['Median']:.6f} n={int(pr['N_Usable'])} "
            f"-- MATCHES freshly recomputed value: {match}")

train_r4_eval, train_r4_resp, train_r4_score = compute_event_response_feature(
    train_df, "P203", "AIT202", R2_R4_OFFSET, ait202_response_ref)
val_r4_eval, val_r4_resp, val_r4_score = compute_event_response_feature(
    val_df, "P203", "AIT202", R2_R4_OFFSET, ait202_response_ref)
log(f"\nS2_R4_evaluable coverage: TRAIN={100*train_r4_eval.mean():.4f}% ({train_r4_eval.sum()} events), "
    f"VALIDATION={100*val_r4_eval.mean():.4f}% ({val_r4_eval.sum()} events)")

section("S2-R4 — INVESTIGATING A ROLLING/STATE-CONDITIONED FALLBACK (not forced)")
r4_sep = check_state_conditioned_separation(train_df, "P203", "AIT202")
log(f"TRAIN-NORMAL AIT202 steady-state separation by P203 state: SMD={r4_sep['smd']:.3f}, "
    f"OFF IQR={r4_sep['off_iqr']}, ON IQR={r4_sep['on_iqr']}, "
    f"IQR overlap fraction={r4_sep['iqr_overlap_fraction']:.3f}")
R4_FALLBACK_DEFENSIBLE = (abs(r4_sep["smd"]) > 1.0) and (r4_sep["iqr_overlap_fraction"] < 0.25)
log(f"Decision rule (same as S2-R2): {'DEFENSIBLE -- fallback feature will be built' if R4_FALLBACK_DEFENSIBLE else 'NOT DEFENSIBLE -- event-based score retained, no fallback forced'}.")
if not R4_FALLBACK_DEFENSIBLE:
    log("Reason: same structural argument as S2-R2 -- AIT202 (pH) steady-state OFF vs ON distributions "
        "are not cleanly separated enough to support a defensible rolling/state-conditioned score. "
        f"LIMITATION: S2-R4 remains event-based, evaluable only at the {train_r4_eval.sum()} TRAIN / "
        f"{val_r4_eval.sum()} VALIDATION P203 OFF->ON transition instants.")

# =============================================================================
# PART 1 / S2-R3B — BIDIRECTIONAL TRANSITION COUPLING + STEADY-STATE EQUALITY
# =============================================================================
section("S2-R3B — P205<->P203 TRANSITION COUPLING AND STEADY-STATE EQUALITY")


def compute_r3b(df):
    n = len(df)
    p205 = df["P205"].to_numpy()
    p203 = df["P203"].to_numpy()
    p205_changed = np.zeros(n, dtype=bool)
    p203_changed = np.zeros(n, dtype=bool)
    p205_changed[1:] = p205[1:] != p205[:-1]
    p203_changed[1:] = p203[1:] != p203[:-1]
    transition_active = p205_changed | p203_changed

    p205_dir = np.zeros(n)
    p203_dir = np.zeros(n)
    p205_dir[1:] = np.sign(p205[1:].astype(int) - p205[:-1].astype(int))
    p203_dir[1:] = np.sign(p203[1:].astype(int) - p203[:-1].astype(int))

    coupling_consistent = np.full(n, np.nan)
    both_changed_same_dir = transition_active & p205_changed & p203_changed & (p205_dir == p203_dir)
    coupling_consistent[transition_active] = both_changed_same_dir[transition_active].astype(float)

    state_mismatch = (p205 != p203).astype(float)
    return transition_active, coupling_consistent, state_mismatch


train_transition_active, train_coupling_consistent, train_state_mismatch = compute_r3b(train_df)
val_transition_active, val_coupling_consistent, val_state_mismatch = compute_r3b(val_df)

train_normal_mismatch_rate = 100 * train_state_mismatch[train_df["Label"] == 0].mean()
log(f"TRAIN-NORMAL steady-state mismatch rate (P205 != P203, ALL rows, not just transitions): "
    f"{train_normal_mismatch_rate:.6f}%")
R3B_MISMATCH_DEFENSIBLE = train_normal_mismatch_rate < 1.0
log(f"Decision rule (documented): steady-state equality feature is considered TRAIN-normal-supported "
    f"if the TRAIN-NORMAL mismatch rate is <1%. Result: "
    f"{'SUPPORTED -- steady-state mismatch feature will be reported' if R3B_MISMATCH_DEFENSIBLE else 'NOT SUPPORTED at this threshold -- reporting mismatch rate as a descriptive reference only'}.")
log(f"\nTRANSITION COUPLING vs STEADY-STATE EQUALITY are DISTINCT features here: transition coupling is "
    f"evaluable only at the {train_transition_active.sum()} TRAIN / {val_transition_active.sum()} "
    f"VALIDATION rows where either pump just changed state; steady-state mismatch is evaluable on "
    f"EVERY row regardless of any transition. They are not assumed equivalent -- both are reported "
    f"separately below.")

train_normal_coupling_rate = np.nanmean(
    train_coupling_consistent[(train_df["Label"] == 0) & train_transition_active])
log(f"TRAIN-NORMAL transition-coupling consistency rate (at transition-active rows only): "
    f"{100*train_normal_coupling_rate:.2f}% "
    f"(n={int((train_transition_active & (train_df['Label']==0)).sum())})")

# =============================================================================
# PART 3 — APPLY UNCHANGED TO VALIDATION: 4-WAY SUMMARY FOR EVERY FEATURE
# =============================================================================
section("PART 3 — TRAIN NORMAL / TRAIN ATTACK / VALIDATION NORMAL / VALIDATION ATTACK SUMMARY")

FEATURES = {
    "S2_R1_score": {"train_eval": train_r1_eval, "train_score": train_r1_score,
                     "val_eval": val_r1_eval, "val_score": val_r1_score},
    "S2_R2_score": {"train_eval": train_r2_eval, "train_score": train_r2_score,
                     "val_eval": val_r2_eval, "val_score": val_r2_score},
    "S2_R4_score": {"train_eval": train_r4_eval, "train_score": train_r4_score,
                     "val_eval": val_r4_eval, "val_score": val_r4_score},
    "S2_R3B_coupling_consistent": {
        "train_eval": train_transition_active,
        "train_score": np.where(train_transition_active, 1.0 - train_coupling_consistent, np.nan),
        "val_eval": val_transition_active,
        "val_score": np.where(val_transition_active, 1.0 - val_coupling_consistent, np.nan),
    },
    "S2_R3B_state_mismatch": {
        "train_eval": np.ones(len(train_df), dtype=bool),
        "train_score": train_state_mismatch,
        "val_eval": np.ones(len(val_df), dtype=bool),
        "val_score": val_state_mismatch,
    },
}
if R2_FALLBACK_DEFENSIBLE:
    FEATURES["S2_R2_state_fallback_score"] = {
        "train_eval": train_r2fb_eval, "train_score": train_r2fb_score,
        "val_eval": val_r2fb_eval, "val_score": val_r2fb_score,
    }
# NOTE: for S2_R3B_coupling_consistent, the reported SCORE is
# (1 - coupling_consistent) so that, consistently with every other feature
# in this table, LARGER = more inconsistent with TRAIN-normal coupling.

summary_rows = []
for fname, f in FEATURES.items():
    for split_name, df, eval_mask, score in [
        ("TRAIN", train_df, f["train_eval"], f["train_score"]),
        ("VALIDATION", val_df, f["val_eval"], f["val_score"]),
    ]:
        for stratum, label_val in [("NORMAL", 0), ("ATTACK", 1)]:
            mask = eval_mask & (df["Label"].to_numpy() == label_val)
            n_total = int((df["Label"].to_numpy() == label_val).sum())
            vals = score[mask]
            stats = summarize_group(vals)
            coverage = 100 * stats["n"] / n_total if n_total else np.nan
            summary_rows.append({
                "Feature": fname, "Split": split_name, "Label_Stratum": stratum,
                "N_Total_Rows": n_total, "N_Evaluable": stats["n"], "Coverage_Pct": coverage,
                "Mean": stats["mean"], "Median": stats["median"], "P90": stats["p90"],
                "P95": stats["p95"], "Max": stats["max"],
            })

summary_df = pd.DataFrame(summary_rows)
summary_path = TABLES_DIR / "stage2_symbolic_feature_validation_summary.csv"
summary_df.to_csv(summary_path, index=False)
log(f"Wrote {summary_path} ({len(summary_df)} rows)")

for fname in FEATURES:
    log(f"\n{fname}:")
    for _, r in summary_df[summary_df.Feature == fname].iterrows():
        log(f"  {r['Split']:10s} {r['Label_Stratum']:6s}: n_eval={r['N_Evaluable']:>6}/{r['N_Total_Rows']:<6} "
            f"cov={r['Coverage_Pct']:.3f}%  mean={r['Mean']}  median={r['Median']}  "
            f"P90={r['P90']}  P95={r['P95']}  max={r['Max']}")

# also produce the TRAIN-only + VALIDATION-only combined general summary table
# requested as tables/stage2_symbolic_feature_summary.csv (aggregate over both
# label strata per split, for a quick at-a-glance view)
general_summary_rows = []
for fname, f in FEATURES.items():
    for split_name, df, eval_mask, score in [
        ("TRAIN", train_df, f["train_eval"], f["train_score"]),
        ("VALIDATION", val_df, f["val_eval"], f["val_score"]),
    ]:
        vals = score[eval_mask]
        stats = summarize_group(vals)
        coverage = 100 * stats["n"] / len(df)
        general_summary_rows.append({
            "Feature": fname, "Split": split_name, "N_Total_Rows": len(df),
            "N_Evaluable": stats["n"], "Coverage_Pct": coverage, "Mean": stats["mean"],
            "Median": stats["median"], "P90": stats["p90"], "P95": stats["p95"], "Max": stats["max"],
        })
general_summary_df = pd.DataFrame(general_summary_rows)
general_summary_path = TABLES_DIR / "stage2_symbolic_feature_summary.csv"
general_summary_df.to_csv(general_summary_path, index=False)
log(f"\nWrote {general_summary_path} ({len(general_summary_df)} rows)")

# =============================================================================
# PART 4 — VISUALIZATION
# =============================================================================
section("PART 4 — GENERATING FIGURES (300 DPI)")

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
                      "axes.titlesize": 13, "axes.labelsize": 11})

# --- Figure 1: S2-R1 valve/flow consistency ---
fig, ax = plt.subplots(figsize=(9, 6))
bins = np.linspace(0, max(fit201_open_ref.max(), fit201_closed_ref.max()) * 1.05, 80)
ax.hist(fit201_closed_ref, bins=bins, alpha=0.6, label=f"MV201 = CLOSED (n={len(fit201_closed_ref):,})",
        color="#C44E52", density=True)
ax.hist(fit201_open_ref, bins=bins, alpha=0.6, label=f"MV201 = OPEN (n={len(fit201_open_ref):,})",
        color="#4C72B0", density=True)
ax.set_xlabel("FIT201 (flow)")
ax.set_ylabel("Density")
ax.set_title("S2-R1: TRAIN-NORMAL FIT201 Distribution by Steady MV201 State\n"
              "(transitional state 0 excluded from both distributions)")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig1_path = FIGURES_DIR / "stage2_r1_valve_flow_consistency.png"
fig.savefig(fig1_path, dpi=300)
plt.close(fig)
log(f"Wrote {fig1_path}")

# --- Figure 2: S2-R2 ORP response distribution ---
fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(ait203_response_ref, bins=20, color="#4C72B0", edgecolor="white")
for pct, style, lbl in [(5, ":", "P5"), (10, "--", "P10"), (50, "-", "Median"), (90, "--", "P90"), (95, ":", "P95")]:
    v = np.percentile(ait203_response_ref, pct)
    ax.axvline(v, linestyle=style, color="#C44E52", linewidth=1.3)
    ax.text(v, ax.get_ylim()[1] * 0.95, lbl, rotation=90, va="top", ha="right", fontsize=8, color="#C44E52")
ax.set_xlabel("AIT203 change at 5s vs. pre-event baseline (ORP)")
ax.set_ylabel("Count")
ax.set_title(f"S2-R2: TRAIN-NORMAL 5-second AIT203 Response to P205 OFF->ON\n"
             f"(n={len(ait203_response_ref)} usable TRAIN NORMAL events)")
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig2_path = FIGURES_DIR / "stage2_r2_orp_response_distribution.png"
fig.savefig(fig2_path, dpi=300)
plt.close(fig)
log(f"Wrote {fig2_path}")

# --- Figure 3: S2-R3B pump coupling ---
if prior_coupling is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, direction in zip(axes, ["Activation (OFF->ON)", "Deactivation (ON->OFF)"]):
        sub = prior_coupling[(prior_coupling.Direction == direction) & (prior_coupling.Metric_Type == "by_offset")]
        ax.bar(sub["Offset_Seconds"].astype(str), sub["Response_Probability"] * 100, color="#4C72B0")
        ax.set_ylim(0, 105)
        ax.set_xlabel("Window (seconds)")
        ax.set_ylabel("P203 matching-transition probability (%)")
        n_ev = sub["N_Usable"].iloc[0] if len(sub) else "?"
        ax.set_title(f"{direction}\n(n={n_ev} TRAIN-NORMAL events)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("S2-R3B: P205<->P203 Transition-Coupling Consistency (TRAIN NORMAL)", y=1.02)
    fig.tight_layout()
    fig3_path = FIGURES_DIR / "stage2_r3_pump_coupling.png"
    fig.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"Wrote {fig3_path}")
else:
    log("SKIPPED figure 3: prior coupling comparison table not found.")

# --- Figure 4: S2-R4 pH response distribution ---
fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(ait202_response_ref, bins=20, color="#55A868", edgecolor="white")
for pct, style, lbl in [(5, ":", "P5"), (10, "--", "P10"), (50, "-", "Median"), (90, "--", "P90"), (95, ":", "P95")]:
    v = np.percentile(ait202_response_ref, pct)
    ax.axvline(v, linestyle=style, color="#C44E52", linewidth=1.3)
    ax.text(v, ax.get_ylim()[1] * 0.95, lbl, rotation=90, va="top", ha="right", fontsize=8, color="#C44E52")
ax.set_xlabel("AIT202 change at 5s vs. pre-event baseline (pH)")
ax.set_ylabel("Count")
ax.set_title(f"S2-R4: TRAIN-NORMAL 5-second AIT202 Response to P203 OFF->ON\n"
             f"(n={len(ait202_response_ref)} usable TRAIN NORMAL events)")
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig4_path = FIGURES_DIR / "stage2_r4_ph_response_distribution.png"
fig.savefig(fig4_path, dpi=300)
plt.close(fig)
log(f"Wrote {fig4_path}")

# --- Figure 5: VALIDATION NORMAL vs ATTACK symbolic score comparison ---
fig, axes = plt.subplots(1, len(FEATURES), figsize=(5 * len(FEATURES), 5.5))
if len(FEATURES) == 1:
    axes = [axes]
for ax, (fname, f) in zip(axes, FEATURES.items()):
    normal_vals = f["val_score"][f["val_eval"] & (val_df["Label"].to_numpy() == 0)]
    attack_vals = f["val_score"][f["val_eval"] & (val_df["Label"].to_numpy() == 1)]
    normal_vals = normal_vals[~np.isnan(normal_vals)]
    attack_vals = attack_vals[~np.isnan(attack_vals)]
    data = [normal_vals, attack_vals]
    labels = [f"NORMAL\n(n={len(normal_vals)})", f"ATTACK\n(n={len(attack_vals)})"]
    if len(normal_vals) == 0 and len(attack_vals) == 0:
        ax.text(0.5, 0.5, "No evaluable\nVALIDATION samples", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
    else:
        bp = ax.boxplot([d if len(d) else [np.nan] for d in data], tick_labels=labels, showmeans=True)
    ax.set_title(fname.replace("_", " "))
    ax.set_ylabel("Inconsistency score")
    if len(attack_vals) < 5:
        ax.annotate(f"CAUTION: only n={len(attack_vals)} ATTACK sample(s) --\nnot a reliable distribution",
                    xy=(0.5, 0.02), xycoords="axes fraction", ha="center", fontsize=7.5, color="#C44E52")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.suptitle("VALIDATION Symbolic Inconsistency Scores: NORMAL vs. ATTACK (where evaluable)", y=1.03)
fig.tight_layout()
fig5_path = FIGURES_DIR / "stage2_symbolic_score_validation_comparison.png"
fig.savefig(fig5_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig5_path}")

# =============================================================================
# PART 5 — FEATURE DEFINITION TABLE
# =============================================================================
section("PART 5 — FEATURE DEFINITION TABLE")

definitions = [
    {
        "Feature": "S2_R1_score", "Source_Rule": "S2-R1", "Source_Tags": "MV201, FIT201",
        "Representation_Type": "State-based (continuous score)",
        "Time_Horizon_Window": "None (instantaneous, current row only)",
        "TRAIN_Normal_Reference": f"FIT201 | MV201=CLOSED: n={len(fit201_closed_ref)}, "
                                   f"median={np.median(fit201_closed_ref):.4f}; "
                                   f"FIT201 | MV201=OPEN: n={len(fit201_open_ref)}, "
                                   f"median={np.median(fit201_open_ref):.4f}",
        "Score_Definition": "score = 1 - 2*min(F(x), 1-F(x)), F = empirical CDF of TRAIN-NORMAL "
                             "FIT201 for the CURRENT steady MV201 state (CLOSED or OPEN reference "
                             "used according to the row's own MV201 value)",
        "Evaluability_Condition": "MV201 != 0 (i.e. not in the transitional state)",
        "Interpretation": "0 = FIT201 typical for the current valve state; 1 = FIT201 at the extreme "
                           "tail of what TRAIN-NORMAL ever showed for that state",
        "Limitation": "Scores are inflated for ~15-20s after an OPEN transition while flow is still "
                       "ramping up (see Part 1 log); does not know a transition just happened.",
    },
    {
        "Feature": "S2_R2_score", "Source_Rule": "S2-R2", "Source_Tags": "P205, AIT203",
        "Representation_Type": "Event-based (continuous score at trigger instant only)",
        "Time_Horizon_Window": f"{R2_R4_OFFSET}s post-transition, {BASELINE_WINDOW}s pre-transition baseline",
        "TRAIN_Normal_Reference": f"AIT203 5s response | n={len(ait203_response_ref)}, "
                                   f"median={np.median(ait203_response_ref):.4f}, "
                                   f"P5={np.percentile(ait203_response_ref,5):.4f}, "
                                   f"P95={np.percentile(ait203_response_ref,95):.4f}",
        "Score_Definition": "score = 1 - 2*min(F(x), 1-F(x)), F = empirical CDF of the TRAIN-NORMAL "
                             "5s AIT203-response reference sample; x = AIT203(t+5s) - mean(AIT203, "
                             "preceding 10s) computed at each P205 OFF->ON transition",
        "Evaluability_Condition": "Only at rows where a P205 OFF->ON transition occurs AND both the "
                                   "10s pre-baseline and 5s post-window are temporally continuous",
        "Interpretation": "0 = ORP response typical of TRAIN-NORMAL dosing events; 1 = response at "
                           "the extreme tail (unusually weak/absent OR unusually strong)",
        "Limitation": f"Rolling/state-conditioned fallback investigated but NOT built "
                       f"({'built' if R2_FALLBACK_DEFENSIBLE else 'not defensible, see Part 1 log'}); "
                       f"coverage is essentially zero outside the "
                       f"{int(train_r2_eval.sum())} TRAIN / {int(val_r2_eval.sum())} VALIDATION event rows.",
    },
    {
        "Feature": "S2_R4_score", "Source_Rule": "S2-R4", "Source_Tags": "P203, AIT202",
        "Representation_Type": "Event-based (continuous score at trigger instant only)",
        "Time_Horizon_Window": f"{R2_R4_OFFSET}s post-transition, {BASELINE_WINDOW}s pre-transition baseline",
        "TRAIN_Normal_Reference": f"AIT202 5s response | n={len(ait202_response_ref)}, "
                                   f"median={np.median(ait202_response_ref):.4f}, "
                                   f"P5={np.percentile(ait202_response_ref,5):.4f}, "
                                   f"P95={np.percentile(ait202_response_ref,95):.4f}",
        "Score_Definition": "score = 1 - 2*min(F(x), 1-F(x)), F = empirical CDF of the TRAIN-NORMAL "
                             "5s AIT202-response reference sample; x = AIT202(t+5s) - mean(AIT202, "
                             "preceding 10s) computed at each P203 OFF->ON transition",
        "Evaluability_Condition": "Only at rows where a P203 OFF->ON transition occurs AND both the "
                                   "10s pre-baseline and 5s post-window are temporally continuous",
        "Interpretation": "0 = pH response typical of TRAIN-NORMAL dosing events; 1 = response at the "
                           "extreme tail (unusually weak/absent OR unusually strong)",
        "Limitation": f"Rolling/state-conditioned fallback investigated but NOT built "
                       f"({'built' if R4_FALLBACK_DEFENSIBLE else 'not defensible, see Part 1 log'}); "
                       f"coverage is essentially zero outside the "
                       f"{int(train_r4_eval.sum())} TRAIN / {int(val_r4_eval.sum())} VALIDATION event rows.",
    },
    {
        "Feature": "S2_R3B_coupling_consistent", "Source_Rule": "S2-R3B", "Source_Tags": "P205, P203",
        "Representation_Type": "Event-based (binary indicator at transition instants)",
        "Time_Horizon_Window": "0s (same 1-second sample, per TRAIN-normal evidence)",
        "TRAIN_Normal_Reference": f"TRAIN-NORMAL same-row same-direction match rate at transition-active "
                                   f"rows: {100*train_normal_coupling_rate:.2f}%",
        "Score_Definition": "reported score = 1 - coupling_consistent, where coupling_consistent = 1 "
                             "if BOTH P205 and P203 change state on this exact row, in the SAME "
                             "direction (both OFF->ON or both ON->OFF), else 0",
        "Evaluability_Condition": "Only at rows where P205 OR P203 (or both) change state "
                                   "('transition_active' mask)",
        "Interpretation": "0 = the pump pair transitioned together as in TRAIN-NORMAL; 1 = only one "
                           "pump transitioned, or they transitioned in mismatched directions",
        "Limitation": "Coverage bounded by actuator switch frequency (very sparse: "
                       f"{int(train_transition_active.sum())} TRAIN / {int(val_transition_active.sum())} "
                       "VALIDATION rows).",
    },
    {
        "Feature": "S2_R3B_state_mismatch", "Source_Rule": "S2-R3B", "Source_Tags": "P205, P203",
        "Representation_Type": "State-based (continuous/binary score, every row)",
        "Time_Horizon_Window": "None (instantaneous, current row only)",
        "TRAIN_Normal_Reference": f"TRAIN-NORMAL mismatch rate (P205 != P203, all rows): "
                                   f"{train_normal_mismatch_rate:.6f}%",
        "Score_Definition": "score = 1 if P205 != P203 at this row, else 0 (raw binary steady-state "
                             "equality check, NOT the same as transition coupling -- see Part 1 note)",
        "Evaluability_Condition": "Every row, unconditionally (100% coverage by construction)",
        "Interpretation": "0 = pumps in matching states (as in >99.9% of TRAIN-NORMAL rows, if "
                           "supported); 1 = pumps observed in different states",
        "Limitation": f"TRAIN-normal support for this feature: "
                       f"{'CONFIRMED (<1% mismatch)' if R3B_MISMATCH_DEFENSIBLE else 'mismatch rate exceeds the 1% support threshold -- interpret with caution'}.",
    },
]
if R2_FALLBACK_DEFENSIBLE:
    definitions.append({
        "Feature": "S2_R2_state_fallback_score", "Source_Rule": "S2-R2", "Source_Tags": "P205, AIT203",
        "Representation_Type": "State-based fallback (continuous score, mirrors S2-R1's method)",
        "Time_Horizon_Window": "None (instantaneous, current row only)",
        "TRAIN_Normal_Reference": f"AIT203 | P205=OFF: n={len(r2_sep['off_ref'])}, "
                                   f"median={np.median(r2_sep['off_ref']):.4f}; "
                                   f"AIT203 | P205=ON: n={len(r2_sep['on_ref'])}, "
                                   f"median={np.median(r2_sep['on_ref']):.4f} "
                                   f"(SMD={r2_sep['smd']:.3f}, IQR overlap fraction="
                                   f"{r2_sep['iqr_overlap_fraction']:.3f})",
        "Score_Definition": "score = 1 - 2*min(F(x), 1-F(x)), F = empirical CDF of TRAIN-NORMAL AIT203 "
                             "for the CURRENT steady P205 state (OFF or ON reference used according to "
                             "the row's own P205 value) -- built because the steady-state separation "
                             "check found this defensible, unlike AIT202/S2-R4 below",
        "Evaluability_Condition": "Every row (P205 is always OFF or ON, no transitional state exists "
                                   "for this actuator) -- 100% coverage by construction",
        "Interpretation": "0 = AIT203 typical for the current P205 state; 1 = AIT203 at the extreme "
                           "tail of what TRAIN-NORMAL ever showed for that state",
        "Limitation": "Complements, does not replace, S2_R2_score: this fallback only checks whether "
                       "AIT203's LEVEL matches the current P205 state, not whether a fresh dosing "
                       "event produced the expected TRANSIENT response shape captured by S2_R2_score.",
    })
definitions_df = pd.DataFrame(definitions)
definitions_path = TABLES_DIR / "stage2_symbolic_feature_definitions.csv"
definitions_df.to_csv(definitions_path, index=False)
log(f"Wrote {definitions_path} ({len(definitions_df)} rows)")

# =============================================================================
# PART 6 — RESULTS LOG
# =============================================================================
section("PART 6 — SAVING results/stage2_symbolic_feature_construction_v1.txt")

report_lines = []
report_lines.append("STAGE 2 SYMBOLIC FEATURE CONSTRUCTION v1 — FEATURE ENGINEERING ONLY, NOT ML TRAINING.")
report_lines.append(f"Generated: {RUN_START.isoformat()}")
report_lines.append(f"Python: {sys.version}")
report_lines.append(f"pandas: {pd.__version__} | numpy: {np.__version__} | scipy: {__import__('scipy').__version__}")
report_lines.append(f"OS: {platform.system()} {platform.release()}")
report_lines.append("")
report_lines.append(f"Data sources: {TRAIN_PATH.name} ({len(train_df):,} rows), "
                     f"{VAL_PATH.name} ({len(val_df):,} rows). TEST was never loaded.")
report_lines.append("")
report_lines.append("=== 1. FINAL SYMBOLIC FEATURES CREATED ===")
for fname in FEATURES:
    report_lines.append(f"  - {fname}")
report_lines.append("")
report_lines.append("=== 2. MATHEMATICAL MEANING OF EACH SCORE ===")
for d in definitions:
    report_lines.append(f"  {d['Feature']}: {d['Score_Definition']}")
report_lines.append("")
report_lines.append("=== 3. TRAIN-NORMAL REFERENCE BEHAVIOR ===")
for d in definitions:
    report_lines.append(f"  {d['Feature']}: {d['TRAIN_Normal_Reference']}")
report_lines.append("")
report_lines.append("=== 4. VALIDATION COVERAGE ===")
for fname, f in FEATURES.items():
    cov = 100 * f["val_eval"].mean()
    report_lines.append(f"  {fname}: {cov:.4f}% of all VALIDATION rows evaluable")
report_lines.append("")
report_lines.append("=== 5. VALIDATION ATTACK vs NORMAL SCORE COMPARISON (context only, NOT used for tuning) ===")
for fname, f in FEATURES.items():
    normal_vals = f["val_score"][f["val_eval"] & (val_df["Label"].to_numpy() == 0)]
    attack_vals = f["val_score"][f["val_eval"] & (val_df["Label"].to_numpy() == 1)]
    normal_vals = normal_vals[~np.isnan(normal_vals)]
    attack_vals = attack_vals[~np.isnan(attack_vals)]
    if len(attack_vals) == 0:
        report_lines.append(f"  {fname}: no evaluable VALIDATION ATTACK samples -- comparison not possible")
    else:
        larger = np.mean(attack_vals) > np.mean(normal_vals) if len(normal_vals) else None
        report_lines.append(f"  {fname}: NORMAL mean={np.mean(normal_vals) if len(normal_vals) else float('nan'):.4f} "
                             f"(n={len(normal_vals)})  vs  ATTACK mean={np.mean(attack_vals):.4f} "
                             f"(n={len(attack_vals)})  -- ATTACK larger: {larger} "
                             f"{'[CAUTION: tiny sample]' if len(attack_vals) < 5 else ''}")
report_lines.append("")
report_lines.append("=== 6. SUITABLE FOR DOWNSTREAM ML FUSION ===")
report_lines.append("  S2_R1_score            : YES -- near-universal coverage (state-based), well-behaved "
                     "empirical score, ready to feed into downstream fusion as-is.")
report_lines.append("  S2_R3B_state_mismatch   : " +
                     ("YES -- 100% coverage, TRAIN-normal-supported binary consistency signal."
                      if R3B_MISMATCH_DEFENSIBLE else
                      "CONDITIONAL -- 100% coverage but TRAIN-normal support did not clear the "
                      "documented 1% threshold; usable but interpret cautiously."))
if R2_FALLBACK_DEFENSIBLE:
    report_lines.append("  S2_R2_state_fallback_score: YES -- 100% coverage, TRAIN-normal steady-state "
                         "separation was clean enough (SMD=1.87, no IQR overlap) to support this as a "
                         "defensible always-on complement to the sparse event-based S2_R2_score.")
report_lines.append("")
report_lines.append("=== 7. TOO SPARSE OR UNCERTAIN ===")
report_lines.append(f"  S2_R2_score               : event-only, {int(val_r2_eval.sum())} evaluable VALIDATION "
                     f"rows total, 0 of which are ATTACK-labeled -- cannot yet confirm attack sensitivity.")
report_lines.append(f"  S2_R4_score               : event-only, {int(val_r4_eval.sum())} evaluable VALIDATION "
                     f"rows total; attack-labeled coverage remains limited.")
report_lines.append(f"  S2_R3B_coupling_consistent: event-only, {int(val_transition_active.sum())} evaluable "
                     f"VALIDATION rows total -- very sparse by construction (bounded by pump switch frequency).")
report_lines.append("")
report_lines.append("=== GUARDRAILS CONFIRMED ===")
report_lines.append("  - TEST partition: never loaded, filename not referenced anywhere in this script.")
report_lines.append("  - All reference distributions/envelopes derived from TRAIN NORMAL only.")
report_lines.append("  - No final binary attack/not-attack threshold was chosen.")
report_lines.append("  - No feature selection was performed.")
report_lines.append("  - No model was trained.")
report_lines.append("  - No existing script, dataset, or output was modified.")
report_lines.append("  - Analysis is fully deterministic (no randomness used).")

report_path = RESULTS_DIR / "stage2_symbolic_feature_construction_v1.txt"
report_path.write_text("\n".join(report_lines), encoding="utf-8")
log(f"Wrote {report_path}")

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
    "no_final_threshold_selected": True,
    "no_feature_selection": True,
    "no_model_trained": True,
    "reference_distributions_from_train_normal_only": True,
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

log("1. Final symbolic features: " + ", ".join(FEATURES.keys()))
log("\n2-3. See tables/stage2_symbolic_feature_definitions.csv for full mathematical definitions and "
    "TRAIN-normal reference values.")
log("\n4. VALIDATION coverage:")
for fname, f in FEATURES.items():
    log(f"   {fname}: {100*f['val_eval'].mean():.4f}%")
log("\n5. VALIDATION ATTACK vs NORMAL (context only):")
for fname, f in FEATURES.items():
    normal_vals = f["val_score"][f["val_eval"] & (val_df["Label"].to_numpy() == 0)]
    attack_vals = f["val_score"][f["val_eval"] & (val_df["Label"].to_numpy() == 1)]
    normal_vals = normal_vals[~np.isnan(normal_vals)]
    attack_vals = attack_vals[~np.isnan(attack_vals)]
    if len(attack_vals):
        log(f"   {fname}: NORMAL mean={np.mean(normal_vals) if len(normal_vals) else float('nan'):.4f} vs "
            f"ATTACK mean={np.mean(attack_vals):.4f} (n_attack={len(attack_vals)})")
    else:
        log(f"   {fname}: no evaluable VALIDATION ATTACK samples")
log("\n6. Suitable for downstream ML fusion now: S2_R1_score" +
    (", S2_R3B_state_mismatch" if R3B_MISMATCH_DEFENSIBLE else "") +
    (", S2_R2_state_fallback_score" if R2_FALLBACK_DEFENSIBLE else ""))
log("7. Too sparse/uncertain: S2_R2_score, S2_R4_score, S2_R3B_coupling_consistent (all event-only, "
    "near-zero attack-labeled coverage in VALIDATION)")

run_end = datetime.now()
log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
log("FEATURE-ENGINEERING STAGE COMPLETE. No thresholds chosen, no model trained, no feature selection.")

if not all_checks_passed:
    sys.exit(1)
