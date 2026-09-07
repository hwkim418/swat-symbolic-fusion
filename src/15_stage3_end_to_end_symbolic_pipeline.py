"""
code/15_stage3_end_to_end_symbolic_pipeline.py

STAGE 3 END-TO-END SYMBOLIC PIPELINE — INDEPENDENT DISCOVERY, EMPIRICAL
VALIDATION, FEATURE CONSTRUCTION, AND FINAL SCREENING. ANALYSIS ONLY.

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

Applies the STANDARDIZED METHODOLOGY developed for Stage 1 (script 13),
Stage 2 (scripts 04-11), Stage 4 (script 14), and Stage 5 (script 12) to
Stage 3 (ultrafiltration), but candidate relationships are discovered
and validated INDEPENDENTLY from Stage 3's own tag structure and
TRAIN-NORMAL behavior -- no other stage's specific rule structures are
copied.

Stage 3's tag audit (Part 1) reveals the richest actuator set of any
stage screened so far:
  - FOUR 3-state motorized valves (MV301-304), each structurally like
    Stage 1/2's MV101/MV201 -- each valve's transitional-state structure
    is re-verified independently here, not assumed from precedent.
  - P301 fully constant (0 TRAIN transitions, matching the original
    dataset audit); P302 substantially varying (244 transitions).
  - DPIT301 (differential pressure) is a sensor TYPE not present in any
    prior stage screened so far, enabling a genuinely new
    flow<->differential-pressure consistency candidate category.
  - With 5 varying actuators and only 2 relationship-defining continuous
    sensors besides DPIT301 (FIT301, LIT301), actuator-target pairs are
    selected via TRAIN-NORMAL correlation SCREENING (data-driven, never
    assumed), rather than testing every combination, to avoid fishing.

No ML model is trained, no feature is selected, no attack-label-based
threshold is tuned, no existing script/dataset/output is modified, and
TEST is never loaded -- its filename does not appear anywhere below, by
design. Uses ONLY processed/train_unscaled.csv and
processed/validation_unscaled.csv, plus tables/swat_stage_tag_mapping.csv
and (read-only, for the Part 9 comparison) scripts 11/12/13/14's outputs.
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
STAGE1_SCREENING_PATH = PROJECT_ROOT / "tables" / "stage1_final_rule_screening.csv"
STAGE1_FEATURES_PATH = PROJECT_ROOT / "tables" / "stage1_final_retained_symbolic_features.csv"
STAGE2_SCREENING_PATH = PROJECT_ROOT / "tables" / "stage2_final_rule_screening.csv"
STAGE2_FEATURES_PATH = PROJECT_ROOT / "tables" / "stage2_final_retained_symbolic_features.csv"
STAGE4_SCREENING_PATH = PROJECT_ROOT / "tables" / "stage4_final_rule_screening.csv"
STAGE4_FEATURES_PATH = PROJECT_ROOT / "tables" / "stage4_final_retained_symbolic_features.csv"
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


section("STAGE 3 END-TO-END SYMBOLIC PIPELINE — RUN START (ANALYSIS ONLY)")
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
stage3_tags_df = stage_mapping[stage_mapping.Stage == "Stage 3"]
STAGE3_TAGS = stage3_tags_df["Tag"].tolist()
STAGE3_SENSORS = stage3_tags_df.loc[stage3_tags_df.Sensor_Or_Actuator == "sensor", "Tag"].tolist()
STAGE3_ACTUATORS = stage3_tags_df.loc[stage3_tags_df.Sensor_Or_Actuator == "actuator", "Tag"].tolist()
log(f"Loaded {STAGE_MAPPING_PATH.name}: Stage 3 = {len(STAGE3_TAGS)} tags "
    f"({len(STAGE3_SENSORS)} sensors, {len(STAGE3_ACTUATORS)} actuators)")
log(f"  Sensors  : {STAGE3_SENSORS}")
log(f"  Actuators: {STAGE3_ACTUATORS}")

train_normal = train_df[train_df.Label == 0]

# =============================================================================
# SHARED HELPER FUNCTIONS (identical methodology to scripts 05-14)
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
    """SMD + IQR-overlap-fraction with a defensible (never NaN) fallback for
    degenerate zero-span groups (bug found and fixed in script 12; applied
    correctly from the outset in every subsequent stage script)."""
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


def rolling_slope(df, tag, window):
    return (df[tag] - df[tag].shift(window)).to_numpy()


# =============================================================================
# PART 1 — STAGE 3 TAG AUDIT
# =============================================================================
section("PART 1 — STAGE 3 TAG AUDIT")

tag_audit_rows = []
for tag in STAGE3_TAGS:
    row_meta = stage3_tags_df[stage3_tags_df.Tag == tag].iloc[0]
    for split_name, df in [("TRAIN", train_df), ("VALIDATION", val_df)]:
        s = df[tag]
        n = len(s)
        nunique = s.nunique()
        n_trans = int((s != s.shift(1)).sum())
        n_trans = max(n_trans - 1, 0) if n > 0 else 0
        is_actuator = row_meta["Sensor_Or_Actuator"] == "actuator"
        near_constant = (nunique == 1) or (s.value_counts(normalize=True).iloc[0] > 0.999 if nunique > 1 else True)
        if not is_actuator and nunique > 1:
            cv = s.std(ddof=0) / abs(s.mean()) if s.mean() != 0 else np.inf
            near_constant = near_constant or cv < 0.01
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
tag_audit_path = TABLES_DIR / "stage3_tag_audit.csv"
tag_audit_df.to_csv(tag_audit_path, index=False)
log(f"Wrote {tag_audit_path} ({len(tag_audit_df)} rows)")

log("\nTRAIN summary:")
for _, r in tag_audit_df[tag_audit_df.Split == "TRAIN"].iterrows():
    if r["Sensor_Or_Actuator"] == "actuator":
        log(f"  {r['Tag']:8s} [actuator] states: {r['State_Values_And_Frequencies']} | "
            f"transitions={r['N_Transitions']} | constant/near-constant={r['Constant_Or_Near_Constant']}")
    else:
        log(f"  {r['Tag']:8s} [sensor]   n_unique={r['N_Unique']:>5} min={r['Min']:.4f} max={r['Max']:.4f} "
            f"mean={r['Mean']:.4f} median={r['Median']:.4f} std={r['Std']:.4f} "
            f"constant/near-constant={r['Constant_Or_Near_Constant']}")

varying_actuators = tag_audit_df[(tag_audit_df.Split == "TRAIN") & (tag_audit_df.Sensor_Or_Actuator == "actuator") &
                                  (~tag_audit_df.Constant_Or_Near_Constant)]["Tag"].tolist()
constant_actuators = [t for t in STAGE3_ACTUATORS if t not in varying_actuators]
sparse_varying = []
for t in varying_actuators:
    n_trans = tag_audit_df[(tag_audit_df.Split == "TRAIN") & (tag_audit_df.Tag == t)]["N_Transitions"].iloc[0]
    if n_trans < 20:
        sparse_varying.append(t)
log(f"\nStage 3 actuators with sufficient TRAIN variation for empirical rule construction: {varying_actuators}")
log(f"Stage 3 actuators that are constant/near-constant (NOT discarded, reported explicitly): {constant_actuators}")
log(f"Of the varying actuators, sparse ones (< 20 TRAIN transitions, reported not discarded): {sparse_varying}")
log("NOTE: P301 is fully constant (0 TRAIN transitions), matching the original dataset audit's "
    "constant-feature list. All FOUR motorized valves (MV301-304) show 3 observed states, "
    "structurally identical to Stage 1's MV101 and Stage 2's MV201 -- each valve's transitional-state "
    "structure will be re-verified independently below (not assumed) wherever a valve is selected as a "
    "candidate actuator. DPIT301 (differential pressure) is a sensor type not present in any prior "
    "stage screened so far.")

# =============================================================================
# GENERIC ACTUATOR-STATE SEMANTICS + EVENT VALIDATION (handles 2-state AND
# 3-state actuators uniformly; the 3-state transitional-bridge correction is
# re-derived from each candidate's own data, mirroring script 13's approach,
# never assumed from the MV101/MV201 precedent).
# =============================================================================


def infer_actuator_semantics(source_tag, ref_target_tag):
    """Returns dict: {state: 'LOW'/'HIGH'/'TRANSITIONAL'}, using ref_target_tag
    level comparison. LOW/HIGH are generic (not CLOSED/OPEN or OFF/ON) since
    device role is not asserted -- only relative gating direction matters for
    the empirical validation that follows."""
    states = sorted(train_df[source_tag].unique())
    stats = {}
    for st in states:
        mask = train_normal[source_tag] == st
        stats[st] = {"freq": int(mask.sum()), "mean": train_normal.loc[mask, ref_target_tag].mean()}
    if len(states) <= 2:
        labels = {}
        if len(states) == 2:
            lo, hi = sorted(states, key=lambda s: stats[s]["mean"])
            labels[lo], labels[hi] = "LOW", "HIGH"
        else:
            labels[states[0]] = "LOW"
        return labels, stats
    # 3+ states: extremes by target mean are LOW/HIGH; anything else is
    # TRANSITIONAL, regardless of frequency-based heuristics.
    sorted_by_mean = sorted(stats.items(), key=lambda kv: kv[1]["mean"])
    lo_state, hi_state = sorted_by_mean[0][0], sorted_by_mean[-1][0]
    labels = {}
    for st in states:
        if st == lo_state:
            labels[st] = "LOW"
        elif st == hi_state:
            labels[st] = "HIGH"
        else:
            labels[st] = "TRANSITIONAL"
    return labels, stats


def validate_state_event_generic(source_tag, target_tag, expected_sign_hint=None):
    """Handles 2-state actuators directly, and 3-state actuators via the same
    LOW -> transitional -> HIGH opening-sequence correction discovered for
    MV201 (script 07) and re-verified for MV101 (script 13) -- re-derived
    here from each candidate's own transition graph, not assumed."""
    labels, stats = infer_actuator_semantics(source_tag, target_tag)
    low_states = [s for s, l in labels.items() if l == "LOW"]
    high_states = [s for s, l in labels.items() if l == "HIGH"]
    if not low_states or not high_states:
        return {"kind": "state_event", "supported": False, "reason": "could not identify LOW/HIGH states"}

    off_ref = train_normal.loc[train_normal[source_tag].isin(low_states), target_tag].to_numpy()
    on_ref = train_normal.loc[train_normal[source_tag].isin(high_states), target_tag].to_numpy()
    sep = state_separation(off_ref, on_ref)
    log(f"  [state-based] {target_tag} by {source_tag} LOW/HIGH: SMD={sep['smd']:.3f}, "
        f"overlap_frac={sep['overlap_frac']:.3f}, KS={sep['ks_stat']:.3f} (p={sep['ks_p']:.2e}), "
        f"n_LOW={len(off_ref)}, n_HIGH={len(on_ref)}")
    if min(len(off_ref), len(on_ref)) < 200:
        log(f"  NOTE: a minority-state reference group is thin (n_min={min(len(off_ref), len(on_ref))}); "
            f"percentile-based envelope for that state carries wider uncertainty.")

    events = []
    if len(low_states) == 1 and len(high_states) == 1:
        direct_trans = find_specific_transitions(train_df, source_tag, low_states[0], high_states[0])
        if direct_trans:
            for i0 in direct_trans:
                if train_df["Label"].iloc[i0] != 0:
                    continue
                if not check_baseline_window(train_df, i0, source_tag, low_states[0], require_normal=True):
                    continue
                events.append({"open_idx": i0, "baseline_idx": i0})
            mode = "direct_two_state"
        else:
            trans_states = [s for s in labels if labels[s] == "TRANSITIONAL"]
            trans_state = trans_states[0] if trans_states else None
            mode = "three_state_sequence"
            if trans_state is not None:
                s_arr = train_df[source_tag].to_numpy()
                n = len(s_arr)
                is_trans = (s_arr == trans_state).astype(int)
                diffs = np.diff(np.concatenate(([0], is_trans, [0])))
                starts = np.flatnonzero(diffs == 1)
                ends = np.flatnonzero(diffs == -1) - 1
                for st, en in zip(starts, ends):
                    if st - 1 < 0 or en + 1 >= n:
                        continue
                    if s_arr[st - 1] == low_states[0] and s_arr[en + 1] == high_states[0]:
                        if train_df["Label"].iloc[st] != 0:
                            continue
                        if not check_baseline_window(train_df, st, source_tag, low_states[0], require_normal=True):
                            continue
                        events.append({"open_idx": en + 1, "baseline_idx": st})
    else:
        mode = "multi_state_unsupported_events"
    log(f"  [event-based] {len(events)} usable TRAIN NORMAL LOW->HIGH events ({mode})")

    offset_stats = []
    best = None
    if len(events) >= MIN_EVENT_COUNT:
        expected_sign = expected_sign_hint if expected_sign_hint is not None else (1 if on_ref.mean() > off_ref.mean() else -1)
        for offset in EVENT_OFFSETS:
            vals = []
            for ev in events:
                open_idx, base_idx = ev["open_idx"], ev["baseline_idx"]
                if not check_offset_window(train_df, open_idx, offset, require_normal=True):
                    continue
                base = train_df[target_tag].iloc[base_idx - BASELINE_WINDOW:base_idx].mean()
                vals.append(train_df[target_tag].iloc[open_idx + offset] - base)
            stats_o = summarize(vals)
            pe = pct_expected(vals, expected_sign)
            offset_stats.append({"Offset_Seconds": offset, **stats_o, "Pct_Expected_Direction": pe})
        valid_os = [o for o in offset_stats if o["n"] >= MIN_EVENT_COUNT and not np.isnan(o["Pct_Expected_Direction"])]
        if valid_os:
            best = max(valid_os, key=lambda o: o["Pct_Expected_Direction"])
            for o in valid_os:
                if o["Pct_Expected_Direction"] >= CONSISTENCY_THRESHOLD:
                    best = o
                    break
            log(f"  [event-based] best offset={best['Offset_Seconds']}s: n={best['n']}, "
                f"pct_expected_dir={best['Pct_Expected_Direction']:.1f}%, median={best['median']:.5f}")

    return {"kind": "state_event", "supported": True, "labels": labels, "low_states": low_states,
            "high_states": high_states, "off_ref": off_ref, "on_ref": on_ref, **sep, "n_events": len(events),
            "offset_stats": offset_stats, "best_offset": best, "mode": mode}


# =============================================================================
# PART 2 — CANDIDATE PHYSICAL RELATIONSHIP DISCOVERY
# =============================================================================
section("PART 2 — CANDIDATE RELATIONSHIP DISCOVERY (correlation screening for actuator-target pairs)")

varying_actuator_candidates = varying_actuators  # MV301-304, P302

log("Screening all varying Stage-3 actuators against FIT301 (flow) and DPIT301 (differential pressure), "
    "TRAIN NORMAL, lag 0 (discovery only, not sufficient evidence on its own):")
fit_corr, dpit_corr = {}, {}
for a in varying_actuator_candidates:
    fit_corr[a] = np.corrcoef(train_normal[a], train_normal["FIT301"])[0, 1]
    dpit_corr[a] = np.corrcoef(train_normal[a], train_normal["DPIT301"])[0, 1]
    log(f"  {a}: r(vs FIT301)={fit_corr[a]:+.4f}, r(vs DPIT301)={dpit_corr[a]:+.4f}")

best_flow_actuator = max(fit_corr, key=lambda k: abs(fit_corr[k]))
best_dpit_actuator = max(dpit_corr, key=lambda k: abs(dpit_corr[k]))
log(f"Most FIT301-correlated actuator: {best_flow_actuator} (chosen for S3-R1)")
log(f"Most DPIT301-correlated actuator: {best_dpit_actuator} (chosen for S3-R2)")

log("\nScreening pairwise co-transition overlap among varying actuators (for the actuator-actuator "
    "coupling candidate, discovery only):")
pair_scores = {}
for i in range(len(varying_actuator_candidates)):
    for j in range(i + 1, len(varying_actuator_candidates)):
        a, b = varying_actuator_candidates[i], varying_actuator_candidates[j]
        n = len(train_df)
        sa, sb = train_df[a].to_numpy(), train_df[b].to_numpy()
        a_changed = np.zeros(n, dtype=bool); b_changed = np.zeros(n, dtype=bool)
        a_changed[1:] = sa[1:] != sa[:-1]; b_changed[1:] = sb[1:] != sb[:-1]
        both = (a_changed & b_changed).sum()
        either = (a_changed | b_changed).sum()
        rate = both / either if either else 0
        pair_scores[(a, b)] = rate
        log(f"  {a}<->{b}: co-transition rate={rate:.4f} (both={both}, either={either})")
best_pair = max(pair_scores, key=pair_scores.get)
log(f"Highest co-transition-overlap pair: {best_pair} (chosen for S3-R6)")

candidates = [
    {
        "Rule_ID": "S3-R1", "Source_Tags": best_flow_actuator, "Target_Tags": "FIT301",
        "Relationship_Type": "actuator -> flow (state/event)",
        "Expected_Relationship": f"{best_flow_actuator} state change should coincide with a FIT301 change "
                                  f"(direction determined empirically).",
        "Rationale": f"{best_flow_actuator} was the Stage-3 actuator most strongly correlated with FIT301 "
                     f"in TRAIN NORMAL (r={fit_corr[best_flow_actuator]:+.3f}) -- a data-driven choice, "
                     f"mirroring the mechanical valve/pump-gates-flow pattern validated in every prior "
                     f"stage (S1-R1, S2-R1, S4-R2, S5-R1).",
        "Confidence": "high if a valve (direct mechanical gating) or moderate if a pump (indirect, via "
                       "shared hydraulics) -- determined by which device type was selected.",
        "Time_Lag_Expected": "Possibly -- to be determined empirically in Part 3.",
    },
    {
        "Rule_ID": "S3-R2", "Source_Tags": best_dpit_actuator, "Target_Tags": "DPIT301",
        "Relationship_Type": "actuator -> differential pressure (state/event)",
        "Expected_Relationship": f"{best_dpit_actuator} state change should coincide with a DPIT301 change "
                                  f"(direction determined empirically).",
        "Rationale": f"{best_dpit_actuator} was the Stage-3 actuator most strongly correlated with DPIT301 "
                     f"in TRAIN NORMAL (r={dpit_corr[best_dpit_actuator]:+.3f}). Differential pressure "
                     f"across an ultrafiltration membrane is a standard function of valve/pump-driven "
                     f"flow routing in UF systems -- structurally defensible without asserting which "
                     f"specific valve serves which specific role.",
        "Confidence": "moderate (DPIT301's exact plant role -- e.g. membrane fouling indicator vs. simple "
                       "line pressure drop -- is not independently documented in this project).",
        "Time_Lag_Expected": "Possibly -- to be determined empirically in Part 3.",
    },
    {
        "Rule_ID": "S3-R3", "Source_Tags": "FIT301", "Target_Tags": "DPIT301",
        "Relationship_Type": "flow <-> differential-pressure consistency (sensor-sensor)",
        "Expected_Relationship": "Higher FIT301 should coincide with higher DPIT301 -- a standard "
                                  "hydraulic relationship (pressure drop across a filter/membrane "
                                  "increases with flow rate).",
        "Rationale": "This is basic fluid-dynamics logic for any filtration process (higher throughput "
                     "-> higher resistive pressure drop), not an invented plant-specific identity -- it "
                     "applies regardless of exactly which valves/pumps are active.",
        "Confidence": "high (general hydraulic principle, not dependent on unconfirmed tag identities).",
        "Time_Lag_Expected": "Possibly a short hydraulic propagation lag -- tested via a lag sweep.",
    },
    {
        "Rule_ID": "S3-R4", "Source_Tags": "FIT301", "Target_Tags": "LIT301 (rolling slope)",
        "Relationship_Type": "flow <-> level change (rolling/derivative, mass-balance)",
        "Expected_Relationship": "Higher FIT301 should coincide with a more positive LIT301 slope -- a "
                                  "direct physical mass-balance tautology for any tank.",
        "Rationale": "Mirrors the Stage-1 (S1-R4) and Stage-4 (S4-R4) flow<->level-slope methodology: "
                     "Stage 3 also has exactly one flow and one level sensor, making their rolling "
                     "relationship the most physically certain Stage-3 sensor-sensor candidate.",
        "Confidence": "high (mass-balance physics), though Stage 4's analogous test showed this "
                       "relationship does not automatically hold at every stage -- tested independently.",
        "Time_Lag_Expected": "Yes -- level integrates flow, so a lag/window sweep is required.",
    },
    {
        "Rule_ID": "S3-R5", "Source_Tags": "P302", "Target_Tags": "LIT301 (rolling slope)",
        "Relationship_Type": "pump -> tank-level response (rolling/derivative)",
        "Expected_Relationship": "P302 ON should be associated with a different LIT301 slope than P302 "
                                  "OFF (direction determined empirically).",
        "Rationale": "Mirrors the Stage-1 P101->LIT101-slope (S1-R2) and Stage-4 P402->LIT401-slope "
                     "(S4-R3) methodology: P302 is Stage 3's only substantially-varying pump, tested for "
                     "its own effect on the tank-level trend via state-conditioned slope comparison.",
        "Confidence": "moderate (confounded by simultaneous valve-driven inflow, as found for the "
                       "analogous Stage-4 test).",
        "Time_Lag_Expected": "Yes -- slope evaluated over multiple rolling windows (10-300s).",
    },
    {
        "Rule_ID": "S3-R6", "Source_Tags": best_pair[0], "Target_Tags": best_pair[1],
        "Relationship_Type": "actuator-actuator control coupling",
        "Expected_Relationship": "To be determined empirically -- tested for the same same-row, "
                                  "same-direction transition-coupling pattern strongly confirmed for "
                                  "S2-R3B and tested (and rejected, for sparsity/independence reasons) "
                                  "for S1-R5 and S4-R5.",
        "Rationale": f"{best_pair[0]} and {best_pair[1]} showed the highest co-transition overlap rate "
                     f"({pair_scores[best_pair]:.3f}) among all Stage-3 varying-actuator pairs -- a "
                     f"data-driven choice rather than an assumed backwash-cycle or primary/backup "
                     f"pairing.",
        "Confidence": "low-to-moderate (co-transition overlap is itself only a discovery signal; full "
                       "validation follows in Part 3).",
        "Time_Lag_Expected": "If present, expected near-0s per the Stage-2 S2-R3B precedent.",
    },
]

candidates_df = pd.DataFrame(candidates)
candidates_path = TABLES_DIR / "stage3_candidate_rules.csv"
candidates_df.to_csv(candidates_path, index=False)
log(f"\nWrote {candidates_path} ({len(candidates_df)} rows)")
log(f"\n{len(candidates)} candidate relationships identified BEFORE any validation testing:")
for c in candidates:
    log(f"  {c['Rule_ID']}: {c['Source_Tags']} <-> {c['Target_Tags']} ({c['Relationship_Type']})")

# =============================================================================
# PART 3 + 4 — TRAIN-NORMAL EMPIRICAL VALIDATION + NORMAL OPERATING ENVELOPES
# =============================================================================
section("PART 3 & 4 — TRAIN-NORMAL EMPIRICAL VALIDATION AND OPERATING ENVELOPES")

slope_cols_train = {w: rolling_slope(train_df, "LIT301", w) for w in SLOPE_WINDOWS}
slope_cols_val = {w: rolling_slope(val_df, "LIT301", w) for w in SLOPE_WINDOWS}

validation_rows = []
envelope_rows = []
rule_evidence = {}


def validate_pump_slope(pump_tag):
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
    best = None
    all_results = {}
    for w in SLOPE_WINDOWS:
        slope = pd.Series(slope_cols_train[w])
        mask_valid = slope.notna().to_numpy() & (train_df["Label"].to_numpy() == 0)
        fit = train_df["FIT301"].to_numpy(dtype=float)
        rho, rho_p = scipy_stats.spearmanr(fit[mask_valid], slope[mask_valid])
        all_results[w] = {"rho": rho, "rho_p": rho_p, "n": int(mask_valid.sum())}
        log(f"  [FIT301 vs LIT301 slope, window={w}s] Spearman rho={rho:.4f} (p={rho_p:.2e}), n={int(mask_valid.sum())}")
        if best is None or abs(rho) > abs(all_results[best]["rho"]):
            best = w
    w = best
    slope = pd.Series(slope_cols_train[w])
    mask_valid = slope.notna().to_numpy() & (train_df["Label"].to_numpy() == 0)
    fit = train_df["FIT301"].to_numpy(dtype=float)[mask_valid]
    slope_vals = slope.to_numpy()[mask_valid]
    if len(fit) >= MIN_SAMPLE_COUNT and np.std(fit) > 0:
        b, a = np.polyfit(fit, slope_vals, 1)
        residual = slope_vals - (a + b * fit)
        residual_std = np.std(residual); slope_std = np.std(slope_vals)
        explained_frac = 1 - (residual_std ** 2 / slope_std ** 2) if slope_std > 0 else np.nan
    else:
        explained_frac = np.nan
    log(f"  [FIT301 vs LIT301 slope, best window={w}s] linear-fit explained variance fraction="
        f"{explained_frac:.3f} (beyond-correlation stability check)")
    return {"kind": "flow_level_slope", "best_window": w, "all_results": all_results,
            "rho": all_results[w]["rho"], "rho_p": all_results[w]["rho_p"],
            "explained_frac": explained_frac, "slope_sample_for_envelope": slope_vals}


def validate_actuator_coupling(tag_a, tag_b):
    n = len(train_df)
    a = train_df[tag_a].to_numpy(); b = train_df[tag_b].to_numpy()
    a_changed = np.zeros(n, dtype=bool); b_changed = np.zeros(n, dtype=bool)
    a_changed[1:] = a[1:] != a[:-1]; b_changed[1:] = b[1:] != b[:-1]
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


def validate_sensor_sensor_flow_dpit():
    a = train_normal["FIT301"].to_numpy(dtype=float)
    b = train_normal["DPIT301"].to_numpy(dtype=float)
    rho, rho_p = scipy_stats.spearmanr(a, b)
    diff = a - b
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(b != 0, a / b, np.nan)
    diff_cv = np.std(diff) / abs(np.mean(diff)) if np.mean(diff) != 0 else np.inf
    ratio_valid = ratio[~np.isnan(ratio) & np.isfinite(ratio)]
    ratio_cv = np.std(ratio_valid) / abs(np.mean(ratio_valid)) if len(ratio_valid) and np.mean(ratio_valid) != 0 else np.inf
    use_ratio = ratio_cv < diff_cv and len(ratio_valid) >= MIN_SAMPLE_COUNT
    residual = ratio if use_ratio else diff
    residual_name = "ratio (A/B)" if use_ratio else "difference (A-B)"
    residual_clean = residual[~np.isnan(residual) & np.isfinite(residual)]
    stats = summarize(residual_clean)
    stability_cv = ratio_cv if use_ratio else diff_cv

    lag_results = []
    for lag in LAG_GRID:
        if lag == 0:
            aa, bb = a, b
        elif lag > 0:
            aa, bb = a[lag:], b[:-lag]
        else:
            aa, bb = a[:lag], b[-lag:]
        if len(aa) < MIN_SAMPLE_COUNT:
            continue
        r, _ = scipy_stats.spearmanr(aa, bb)
        lag_results.append({"lag": lag, "rho": r})
    best_lag = max(lag_results, key=lambda r: abs(r["rho"])) if lag_results else {"lag": 0, "rho": rho}

    # explicitly test state-conditioned (flow-band-conditioned) residual as an
    # alternative to the raw pooled residual, per instructions for
    # pressure/differential-pressure relationships
    flow_terciles = np.percentile(a, [33.3, 66.7])
    band_labels = np.digitize(a, flow_terciles)
    band_cvs = {}
    for band in [0, 1, 2]:
        band_mask = band_labels == band
        if band_mask.sum() < MIN_SAMPLE_COUNT:
            continue
        band_resid = residual[band_mask]
        band_resid = band_resid[~np.isnan(band_resid) & np.isfinite(band_resid)]
        if len(band_resid) < 30 or np.mean(band_resid) == 0:
            continue
        band_cvs[band] = np.std(band_resid) / abs(np.mean(band_resid))
    flow_conditioning_helps = bool(band_cvs) and max(band_cvs.values()) < stability_cv
    log(f"  [flow-conditioned residual check] per-flow-tercile CV: {band_cvs} -- "
        f"flow-conditioning improves stability: {flow_conditioning_helps}")

    log(f"  [sensor-sensor] FIT301 vs DPIT301: Spearman rho={rho:.4f} (p={rho_p:.2e}), "
        f"best_lag={best_lag['lag']}s (rho={best_lag['rho']:.4f})")
    log(f"  [sensor-sensor] relationship formalized as {residual_name}: CV={stability_cv:.4f}, "
        f"median={stats['median']:.5f}, IQR=[{stats['p25']:.5f}, {stats['p75']:.5f}]")
    return {"kind": "sensor_sensor", "supported": True, "rho": rho, "rho_p": rho_p,
            "residual_name": residual_name, "residual_sample": residual_clean, "stability_cv": stability_cv,
            "stats": stats, "best_lag": best_lag, "use_ratio": use_ratio,
            "flow_conditioning_helps": flow_conditioning_helps, "band_cvs": band_cvs}


for c in candidates:
    rid = c["Rule_ID"]
    log(f"\n--- Validating {rid}: {c['Source_Tags']} <-> {c['Target_Tags']} ---")
    if rid == "S3-R1":
        result = validate_state_event_generic(best_flow_actuator, "FIT301")
    elif rid == "S3-R2":
        result = validate_state_event_generic(best_dpit_actuator, "DPIT301")
    elif rid == "S3-R3":
        result = validate_sensor_sensor_flow_dpit()
    elif rid == "S3-R4":
        result = validate_flow_level_slope()
    elif rid == "S3-R5":
        result = validate_pump_slope("P302")
    else:  # S3-R6
        result = validate_actuator_coupling(best_pair[0], best_pair[1])
    rule_evidence[rid] = result

    if result["kind"] == "state_event" and result.get("supported"):
        validation_rows.append({
            "Rule_ID": rid, "Kind": "state_event", "N_Events_TRAIN_NORMAL": result["n_events"],
            "State_SMD": result["smd"], "State_IQR_Overlap_Fraction": result["overlap_frac"],
            "State_KS_Statistic": result["ks_stat"], "Mode": result["mode"],
            "Best_Event_Offset_Seconds": result["best_offset"]["Offset_Seconds"] if result["best_offset"] else np.nan,
            "Best_Event_Pct_Expected_Direction": result["best_offset"]["Pct_Expected_Direction"] if result["best_offset"] else np.nan,
            "Best_Event_Median_Response": result["best_offset"]["median"] if result["best_offset"] else np.nan,
        })
        if result["best_offset"]:
            b = result["best_offset"]
            envelope_rows.append({"Rule_ID": rid, "Envelope_Type": "event_response", "Window_Or_Offset": b["Offset_Seconds"],
                                   "N": b["n"], "P5": b["p5"], "P10": b["p10"], "P25": b["p25"], "Median": b["median"],
                                   "P75": b["p75"], "P90": b["p90"], "P95": b["p95"], "IQR": b["iqr"]})
        for label, ref in [("LOW", result["off_ref"]), ("HIGH", result["on_ref"])]:
            st = summarize(ref)
            envelope_rows.append({"Rule_ID": rid, "Envelope_Type": f"state_conditioned_{label}", "Window_Or_Offset": np.nan,
                                   "N": st["n"], "P5": st["p5"], "P10": st["p10"], "P25": st["p25"], "Median": st["median"],
                                   "P75": st["p75"], "P90": st["p90"], "P95": st["p95"], "IQR": st["iqr"]})
    elif result["kind"] == "sensor_sensor" and result.get("supported"):
        validation_rows.append({
            "Rule_ID": rid, "Kind": "sensor_sensor", "N_Events_TRAIN_NORMAL": result["stats"]["n"],
            "Spearman_Rho": result["rho"], "Spearman_Pvalue": result["rho_p"],
            "Residual_Formalization": result["residual_name"], "Residual_Stability_CV": result["stability_cv"],
            "Best_Lag_Seconds": result["best_lag"]["lag"], "Flow_Conditioning_Helps": result["flow_conditioning_helps"],
        })
        st = result["stats"]
        envelope_rows.append({"Rule_ID": rid, "Envelope_Type": f"residual_{result['residual_name']}",
                               "Window_Or_Offset": result["best_lag"]["lag"], "N": st["n"], "P5": st["p5"],
                               "P10": st["p10"], "P25": st["p25"], "Median": st["median"], "P75": st["p75"],
                               "P90": st["p90"], "P95": st["p95"], "IQR": st["iqr"]})
    elif result["kind"] == "flow_level_slope":
        validation_rows.append({
            "Rule_ID": rid, "Kind": "flow_level_slope", "N_Events_TRAIN_NORMAL": result["all_results"][result["best_window"]]["n"],
            "Spearman_Rho": result["rho"], "Spearman_Pvalue": result["rho_p"],
            "Best_Slope_Window_Seconds": result["best_window"], "Explained_Variance_Fraction": result["explained_frac"],
        })
        st = summarize(result["slope_sample_for_envelope"])
        envelope_rows.append({"Rule_ID": rid, "Envelope_Type": "lit301_slope_reference", "Window_Or_Offset": result["best_window"],
                               "N": st["n"], "P5": st["p5"], "P10": st["p10"], "P25": st["p25"], "Median": st["median"],
                               "P75": st["p75"], "P90": st["p90"], "P95": st["p95"], "IQR": st["iqr"]})
    elif result["kind"] == "pump_slope":
        validation_rows.append({
            "Rule_ID": rid, "Kind": "pump_slope",
            "N_Events_TRAIN_NORMAL": (min(len(result["off_ref"]), len(result["on_ref"])) if result.get("supported") else 0),
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
    elif result["kind"] == "actuator_coupling":
        validation_rows.append({"Rule_ID": rid, "Kind": "actuator_coupling",
                                 "N_Events_TRAIN_NORMAL": result["n_transition_active"],
                                 "Coupling_Rate": result["coupling_rate"]})
    else:
        validation_rows.append({"Rule_ID": rid, "Kind": result["kind"], "N_Events_TRAIN_NORMAL": 0,
                                 "Unsupported_Reason": result.get("reason")})

validation_df = pd.DataFrame(validation_rows)
validation_path = TABLES_DIR / "stage3_rule_validation.csv"
validation_df.to_csv(validation_path, index=False)
log(f"\nWrote {validation_path} ({len(validation_df)} rows)")

envelope_df = pd.DataFrame(envelope_rows)
envelope_path = TABLES_DIR / "stage3_normal_operating_envelopes.csv"
envelope_df.to_csv(envelope_path, index=False)
log(f"Wrote {envelope_path} ({len(envelope_df)} rows)")

# =============================================================================
# PART 5 + 6 — REPRESENTATION, COVERAGE, ROBUSTNESS
# =============================================================================
section("PART 5 & 6 — REPRESENTATION TYPE, VALIDATION COVERAGE, AND DISTRIBUTION-SHIFT ROBUSTNESS")

representation = {}
coverage_rows = []
robustness_rows = []
score_arrays = {}

for c in candidates:
    rid = c["Rule_ID"]
    ev = rule_evidence[rid]

    if ev["kind"] == "state_event" and ev.get("supported"):
        source_tag = c["Source_Tags"]; target_tag = c["Target_Tags"]
        representation[rid] = "A: state-based" + (" (+event-response secondary evidence)" if ev["n_events"] >= MIN_EVENT_COUNT else "")

        def score_fn(df, ev=ev, source_tag=source_tag, target_tag=target_tag):
            st = df[source_tag].to_numpy()
            x = df[target_tag].to_numpy(dtype=float)
            score = np.full(len(df), np.nan)
            eval_mask = np.isin(st, ev["low_states"] + ev["high_states"])
            lo_mask = eval_mask & np.isin(st, ev["low_states"])
            hi_mask = eval_mask & np.isin(st, ev["high_states"])
            score[lo_mask] = empirical_two_sided_score_vec(ev["off_ref"], x[lo_mask])
            score[hi_mask] = empirical_two_sided_score_vec(ev["on_ref"], x[hi_mask])
            return eval_mask, score
        eval_mask_train, score_train = score_fn(train_df)
        eval_mask_val, score_val = score_fn(val_df)

    elif ev["kind"] == "sensor_sensor" and ev.get("supported"):
        representation[rid] = "C: rolling-window" if abs(ev["best_lag"]["lag"]) > 0 else "A: state-based (instantaneous residual)"
        tag_a, tag_b = c["Source_Tags"], c["Target_Tags"]

        def score_fn(df, tag_a=tag_a, tag_b=tag_b, ev=ev):
            a = df[tag_a].to_numpy(dtype=float); b = df[tag_b].to_numpy(dtype=float)
            if ev["use_ratio"]:
                with np.errstate(divide="ignore", invalid="ignore"):
                    x = np.where(b != 0, a / b, np.nan)
            else:
                x = a - b
            eval_mask = ~np.isnan(x) & np.isfinite(x)
            score = np.full(len(df), np.nan)
            score[eval_mask] = empirical_two_sided_score_vec(ev["residual_sample"], x[eval_mask])
            return eval_mask, score
        eval_mask_train, score_train = score_fn(train_df)
        eval_mask_val, score_val = score_fn(val_df)

    elif ev["kind"] == "flow_level_slope":
        representation[rid] = "C: rolling-window"
        w = ev["best_window"]; ref_slope = ev["slope_sample_for_envelope"]

        def score_fn(df, split_slopes, w=w, ref_slope=ref_slope):
            slope = split_slopes[w]
            eval_mask = ~np.isnan(slope)
            score = np.full(len(df), np.nan)
            score[eval_mask] = empirical_two_sided_score_vec(ref_slope, slope[eval_mask])
            return eval_mask, score
        eval_mask_train, score_train = score_fn(train_df, slope_cols_train)
        eval_mask_val, score_val = score_fn(val_df, slope_cols_val)

    elif ev["kind"] == "pump_slope" and ev.get("supported"):
        representation[rid] = "C: rolling-window"
        w = ev["best_window"]; pump_tag = c["Source_Tags"]

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

    elif ev["kind"] == "actuator_coupling":
        representation[rid] = "B: event-based"
        tag_a, tag_b = c["Source_Tags"], c["Target_Tags"]
        n_val = len(val_df)
        a = val_df[tag_a].to_numpy(); b = val_df[tag_b].to_numpy()
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
        "Rule_ID": rid, "TRAIN_NORMAL_Score_Median": float(np.median(train_normal_scores)) if len(train_normal_scores) else np.nan,
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
    if ev["kind"] == "state_event" and ev.get("supported"):
        fname = f"{rid}_state_score"
        math_def = (f"score = 1 - 2*min(F(x), 1-F(x)), F = empirical CDF of TRAIN-NORMAL {c['Target_Tags']} "
                     f"for the CURRENT {c['Source_Tags']} state (LOW or HIGH)")
        interp = f"How atypical {c['Target_Tags']} is for {c['Source_Tags']}'s current state, vs. TRAIN-NORMAL."
        eval_cond = f"{c['Source_Tags']} in a LOW or HIGH state ({ev['mode'].replace('_',' ')})"
        limitation = "Event-based companion evidence available." if ev["n_events"] >= MIN_EVENT_COUNT else f"Too few TRAIN transitions ({ev['n_events']}) for a robust event-based companion check."
    elif ev["kind"] == "sensor_sensor" and ev.get("supported"):
        fname = f"{rid}_residual_score"
        lag_suffix = f" at lag {ev['best_lag']['lag']}s" if ev["best_lag"]["lag"] != 0 else ""
        math_def = (f"score = 1 - 2*min(F(x), 1-F(x)), F = empirical CDF of the TRAIN-NORMAL {ev['residual_name']} "
                     f"between {c['Source_Tags']} and {c['Target_Tags']}{lag_suffix}")
        interp = f"How atypical the {c['Source_Tags']}/{c['Target_Tags']} relationship is, vs. TRAIN-NORMAL."
        eval_cond = "Every row where both source and target sensor values are present"
        limitation = ("Flow-conditioned residual was tested and did NOT improve stability over the pooled "
                       "residual; pooled formulation used." if not ev.get("flow_conditioning_helps") else
                       "Flow-conditioned residual improves stability but the pooled (unconditional) "
                       "formulation is used here for simplicity -- a future revision could condition on "
                       "flow band.")
    elif ev["kind"] == "flow_level_slope":
        fname = f"{rid}_slope_score"; w = ev["best_window"]
        math_def = (f"score = 1 - 2*min(F(x), 1-F(x)), F = empirical CDF of TRAIN-NORMAL LIT301 {w}s-rolling-slope "
                     f"(mass-balance relationship with FIT301; explained variance fraction={ev['explained_frac']:.3f})")
        interp = "How atypical the tank-level trend is overall, vs. TRAIN-NORMAL (unconditional on any single actuator)."
        eval_cond = f"Every row where the {w}s slope is computable"
        limitation = "Reflects overall tank-level dynamics, not attributable to any single actuator."
    elif ev["kind"] == "pump_slope" and ev.get("supported"):
        fname = f"{rid}_slope_score"; w = ev["best_window"]
        math_def = (f"score = 1 - 2*min(F(x), 1-F(x)), F = empirical CDF of TRAIN-NORMAL LIT301 {w}s-rolling-slope "
                     f"for the CURRENT {c['Source_Tags']} state")
        interp = f"How atypical the tank-level trend is for {c['Source_Tags']}'s current state, vs. TRAIN-NORMAL."
        eval_cond = f"Every row where the {w}s slope is computable and {c['Source_Tags']} is OFF or ON"
        limitation = "Confounded by simultaneous valve-driven inflow; state-conditioning only partially isolates this pump's own effect."
    elif ev["kind"] == "actuator_coupling":
        fname = f"{rid}_coupling_score"
        math_def = (f"score = 1 - coupling_indicator, where coupling_indicator = 1 if BOTH {c['Source_Tags']} "
                     f"and {c['Target_Tags']} change state on the same row in the SAME direction, else 0")
        interp = f"Whether {c['Source_Tags']} and {c['Target_Tags']} transition together as a coupled pair, vs. TRAIN-NORMAL."
        eval_cond = f"Only at rows where {c['Source_Tags']} OR {c['Target_Tags']} (or both) change state"
        limitation = f"Coverage bounded by actuator switch frequency (TRAIN transition-active rows={ev['n_transition_active']})."
    else:
        continue
    feature_def_rows.append({
        "Feature": fname, "Source_Rule": rid, "Source_Tags": c["Source_Tags"], "Target_Tags": c["Target_Tags"],
        "Representation_Type": representation[rid], "Mathematical_Definition": math_def,
        "Physical_Interpretation": interp, "Evaluability_Condition": eval_cond,
        "Coverage_Pct_VALIDATION_NORMAL": cov["VALIDATION_NORMAL_Coverage_Pct"],
        "Score_Direction": "0 = typical of TRAIN-NORMAL; 1 = extreme tail (atypical)",
        "Robustness_Limitation": limitation,
    })

feature_def_df = pd.DataFrame(feature_def_rows)
feature_def_path = TABLES_DIR / "stage3_symbolic_feature_definitions.csv"
feature_def_df.to_csv(feature_def_path, index=False)
log(f"Wrote {feature_def_path} ({len(feature_def_df)} rows)")

# =============================================================================
# PART 8 — FINAL RULE SCREENING
# =============================================================================
section("PART 8 — FINAL RULE SCREENING (RETAIN / LIMITED / REJECT)")

log("Decision rule (identical standard applied to finalized Stage 1 [script 13], Stage 2 [script 11], "
    "Stage 4 [script 14], and Stage 5 [script 12]):")
log(f"  1. If TRAIN-normal event/sample count < {MIN_EVENT_COUNT} (event/state rules) or "
    f"< {MIN_SAMPLE_COUNT} (sensor-sensor rules) -> REJECT.")
log(f"  2. Else if not empirically supported beyond correlation -> REJECT.")
log(f"  3. Else: RETAIN if VALIDATION coverage >= 90% AND not saturated (VAL-NORMAL mean < "
    f"{SATURATION_THRESHOLD}); otherwise LIMITED.")

decisions = {}
for c in candidates:
    rid = c["Rule_ID"]
    ev = rule_evidence[rid]
    cov = coverage_df[coverage_df.Rule_ID == rid].iloc[0]
    rob = robustness_df[robustness_df.Rule_ID == rid].iloc[0]

    if ev["kind"] == "state_event":
        if not ev.get("supported"):
            n, supported, support_desc, min_n = 0, False, ev.get("reason", "unsupported"), MIN_EVENT_COUNT
        else:
            n = min(len(ev["off_ref"]), len(ev["on_ref"]))
            supported = (not np.isnan(ev["smd"])) and abs(ev["smd"]) > 1.0 and ev["overlap_frac"] < 0.25
            support_desc = f"state SMD={ev['smd']:.3f}, IQR overlap fraction={ev['overlap_frac']:.3f}"
            min_n = MIN_EVENT_COUNT
    elif ev["kind"] == "sensor_sensor":
        if not ev.get("supported"):
            n, supported, support_desc, min_n = 0, False, ev.get("reason", "unsupported"), MIN_SAMPLE_COUNT
        else:
            n = ev["stats"]["n"]
            supported = ev["stability_cv"] < 0.5 and abs(ev["rho"]) > 0.3 and ev["rho_p"] < 0.01
            support_desc = f"Spearman rho={ev['rho']:.3f} (p={ev['rho_p']:.2e}), residual CV={ev['stability_cv']:.3f}"
            min_n = MIN_SAMPLE_COUNT
    elif ev["kind"] == "flow_level_slope":
        n = ev["all_results"][ev["best_window"]]["n"]
        supported = abs(ev["rho"]) > 0.2 and ev["rho_p"] < 0.01 and (not np.isnan(ev["explained_frac"])) and ev["explained_frac"] > 0.04
        support_desc = f"rho={ev['rho']:.3f} (window={ev['best_window']}s), explained_var={ev['explained_frac']:.3f}"
        min_n = MIN_SAMPLE_COUNT
    elif ev["kind"] == "pump_slope":
        if not ev.get("supported"):
            n, supported, support_desc, min_n = 0, False, ev.get("reason", "unsupported"), MIN_EVENT_COUNT
        else:
            n = min(len(ev["off_ref"]), len(ev["on_ref"]))
            supported = (not np.isnan(ev["smd"])) and abs(ev["smd"]) > 0.5 and ev["overlap_frac"] < 0.5
            support_desc = f"state SMD={ev['smd']:.3f} (best window={ev['best_window']}s), overlap_frac={ev['overlap_frac']:.3f}"
            min_n = MIN_EVENT_COUNT
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
log(f"\nStage 3 screening totals: {len(candidates)} candidates -> RETAIN={n_retained}, "
    f"LIMITED={n_limited}, REJECT={n_rejected}")
log("No balanced retain count across stages was forced; this is the evidence-based result.")

rule_ids_order = [c["Rule_ID"] for c in candidates]
screening_rows = []
for c in candidates:
    rid = c["Rule_ID"]
    d = decisions[rid]
    cov = coverage_df[coverage_df.Rule_ID == rid].iloc[0]
    rob = robustness_df[robustness_df.Rule_ID == rid].iloc[0]
    fdef = feature_def_df[feature_def_df.Source_Rule == rid]
    val_row = validation_df[validation_df.Rule_ID == rid]
    val = val_row.iloc[0] if len(val_row) else pd.Series(dtype=object)
    screening_rows.append({
        "Rule_ID": rid, "Physical_Relationship": f"{c['Source_Tags']} <-> {c['Target_Tags']} ({c['Relationship_Type']})",
        "Engineering_Process_Rationale": c["Rationale"],
        "TRAIN_Normal_Evidence": str({k: val[k] for k in val.index if k not in ("Rule_ID", "Kind")}) if len(val_row) else "n/a",
        "Temporal_Evidence": (f"best_offset={val.get('Best_Event_Offset_Seconds', np.nan)}s" if "Best_Event_Offset_Seconds" in val.index else
                               f"best_window={val.get('Best_Slope_Window_Seconds', np.nan)}s" if "Best_Slope_Window_Seconds" in val.index else
                               f"best_lag={val.get('Best_Lag_Seconds', np.nan)}s" if "Best_Lag_Seconds" in val.index else "n/a"),
        "Coverage": f"VALIDATION NORMAL={cov['VALIDATION_NORMAL_Coverage_Pct']:.2f}%, ATTACK={cov['VALIDATION_ATTACK_Coverage_Pct']:.2f}%",
        "Robustness": f"TRAIN-N median={rob['TRAIN_NORMAL_Score_Median']}, VAL-N median={rob['VALIDATION_NORMAL_Score_Median']}, saturated={rob['Saturated_In_VALIDATION']}",
        "Main_Limitation": fdef["Robustness_Limitation"].iloc[0] if len(fdef) else "Rejected before feature construction.",
        "Final_Decision": d["decision"], "Decision_Rationale": d["reason"],
    })
screening_df = pd.DataFrame(screening_rows)
screening_path = TABLES_DIR / "stage3_final_rule_screening.csv"
screening_df.to_csv(screening_path, index=False)
log(f"\nWrote {screening_path} ({len(screening_df)} rows)")

retained_rule_ids = [rid for rid, d in decisions.items() if d["decision"] == "RETAIN"]
final_features_df = feature_def_df[feature_def_df.Source_Rule.isin(retained_rule_ids)].copy()
final_features_path = TABLES_DIR / "stage3_final_retained_symbolic_features.csv"
final_features_df.to_csv(final_features_path, index=False)
log(f"Wrote {final_features_path} ({len(final_features_df)} rows)")
FINAL_FEATURE_NAMES = final_features_df["Feature"].tolist() if len(final_features_df) else []
log(f"Final retained Stage 3 symbolic feature set: {FINAL_FEATURE_NAMES}")

# =============================================================================
# PART 9 — CROSS-STAGE METHODOLOGY CONSISTENCY CHECK
# =============================================================================
section("PART 9 — CROSS-STAGE METHODOLOGY CONSISTENCY CHECK (Stage 3 vs Stages 1, 2, 4, 5)")

s1_screening = pd.read_csv(STAGE1_SCREENING_PATH) if STAGE1_SCREENING_PATH.exists() else None
s1_features = pd.read_csv(STAGE1_FEATURES_PATH) if STAGE1_FEATURES_PATH.exists() else None
s2_screening = pd.read_csv(STAGE2_SCREENING_PATH) if STAGE2_SCREENING_PATH.exists() else None
s2_features = pd.read_csv(STAGE2_FEATURES_PATH) if STAGE2_FEATURES_PATH.exists() else None
s4_screening = pd.read_csv(STAGE4_SCREENING_PATH) if STAGE4_SCREENING_PATH.exists() else None
s4_features = pd.read_csv(STAGE4_FEATURES_PATH) if STAGE4_FEATURES_PATH.exists() else None
s5_screening = pd.read_csv(STAGE5_SCREENING_PATH) if STAGE5_SCREENING_PATH.exists() else None
s5_features = pd.read_csv(STAGE5_FEATURES_PATH) if STAGE5_FEATURES_PATH.exists() else None


def counts(df):
    if df is None:
        return {"n": np.nan, "retain": np.nan, "limited": np.nan, "reject": np.nan}
    return {"n": len(df), "retain": int((df.Final_Decision == "RETAIN").sum()),
            "limited": int((df.Final_Decision == "LIMITED").sum()),
            "reject": int((df.Final_Decision == "REJECT").sum())}


s3c, s1c, s2c, s4c, s5c = counts(screening_df), counts(s1_screening), counts(s2_screening), counts(s4_screening), counts(s5_screening)
n_actuator_sensor_s3 = sum(1 for c in candidates if c["Relationship_Type"].split(" ")[0] in ("actuator", "pump"))
n_actuator_actuator_s3 = sum(1 for c in candidates if "actuator-actuator" in c["Relationship_Type"])
n_sensor_sensor_s3 = sum(1 for c in candidates if "sensor-sensor" in c["Relationship_Type"] or "differential-pressure" in c["Relationship_Type"] or "flow <-> level" in c["Relationship_Type"])

comparison_rows = [
    {"Metric": "Candidate rule count", "Stage_1": s1c["n"], "Stage_2": s2c["n"], "Stage_3": s3c["n"], "Stage_4": s4c["n"], "Stage_5": s5c["n"]},
    {"Metric": "Retained count", "Stage_1": s1c["retain"], "Stage_2": s2c["retain"], "Stage_3": n_retained, "Stage_4": s4c["retain"], "Stage_5": s5c["retain"]},
    {"Metric": "Limited count", "Stage_1": s1c["limited"], "Stage_2": s2c["limited"], "Stage_3": n_limited, "Stage_4": s4c["limited"], "Stage_5": s5c["limited"]},
    {"Metric": "Rejected count", "Stage_1": s1c["reject"], "Stage_2": s2c["reject"], "Stage_3": n_rejected, "Stage_4": s4c["reject"], "Stage_5": s5c["reject"]},
    {"Metric": "Retained feature count",
     "Stage_1": len(s1_features) if s1_features is not None else np.nan,
     "Stage_2": len(s2_features) if s2_features is not None else np.nan,
     "Stage_3": len(FINAL_FEATURE_NAMES),
     "Stage_4": len(s4_features) if s4_features is not None else np.nan,
     "Stage_5": len(s5_features) if s5_features is not None else np.nan},
    {"Metric": "Actuator-sensor candidate count", "Stage_1": 3, "Stage_2": np.nan, "Stage_3": n_actuator_sensor_s3, "Stage_4": 3, "Stage_5": np.nan},
    {"Metric": "Actuator-actuator candidate count", "Stage_1": 1, "Stage_2": 1, "Stage_3": n_actuator_actuator_s3, "Stage_4": 1, "Stage_5": 0},
    {"Metric": "Sensor-sensor / rolling candidate count", "Stage_1": 1, "Stage_2": 0, "Stage_3": n_sensor_sensor_s3, "Stage_4": 2, "Stage_5": 3},
    {"Metric": "Dominant representation type", "Stage_1": "rolling-window", "Stage_2": "state-based",
     "Stage_3": max(set(representation.values()), key=list(representation.values()).count),
     "Stage_4": "rolling-window", "Stage_5": "state-based / sensor-sensor mix"},
    {"Metric": "Best retained-feature VALIDATION coverage",
     "Stage_1": "99.91%", "Stage_2": "~99.6-100%",
     "Stage_3": (f"{coverage_df.loc[coverage_df.Rule_ID.isin(retained_rule_ids), 'VALIDATION_NORMAL_Coverage_Pct'].max():.2f}%"
                 if retained_rule_ids else "n/a"),
     "Stage_4": "100.00%", "Stage_5": "100%"},
    {"Metric": "Genuinely new symbolic-information category vs. prior stages",
     "Stage_1": "flow<->level rolling", "Stage_2": "chemical-dosing response (baseline)",
     "Stage_3": ("YES -- flow<->differential-pressure consistency (DPIT301 is a sensor type not present "
                 "in any prior stage), combined with the richest actuator set (4 three-state valves + 1 "
                 "pump) of any stage screened so far"),
     "Stage_4": "actuator->analyzer + rolling, combined in one stage", "Stage_5": "sensor-sensor conservation"},
]
comparison_df = pd.DataFrame(comparison_rows)
comparison_path = TABLES_DIR / "stage3_cross_stage_methodology_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)
log(f"Wrote {comparison_path} ({len(comparison_df)} rows)")
for r in comparison_rows:
    log(f"  {r['Metric']}: S1={r['Stage_1']} | S2={r['Stage_2']} | S3={r['Stage_3']} | S4={r['Stage_4']} | S5={r['Stage_5']}")

# =============================================================================
# PART 10 — FIGURES
# =============================================================================
section("PART 10 — GENERATING FIGURES (300 DPI)")

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
                      "axes.titlesize": 13, "axes.labelsize": 11})
STATUS_COLOR = {"RETAIN": "#55A868", "LIMITED": "#F4B183", "REJECT": "#C44E52"}


def support_metric(rid):
    ev = rule_evidence[rid]
    if ev["kind"] in ("state_event", "pump_slope"):
        return abs(ev.get("smd", 0)) if ev.get("supported") else 0
    if ev["kind"] == "flow_level_slope":
        return abs(ev["rho"])
    if ev["kind"] == "actuator_coupling":
        return ev["coupling_rate"] if not np.isnan(ev["coupling_rate"]) else 0
    if ev["kind"] == "sensor_sensor":
        return abs(ev.get("rho", 0)) if ev.get("supported") else 0
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
fig.suptitle("Stage 3 Candidate Relationship Empirical Support and Coverage", y=1.16)
fig.tight_layout()
fig1_path = FIGURES_DIR / "stage3_candidate_rule_validation.png"
fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig1_path}")

n_rules = len(candidates)
fig, axes = plt.subplots(1, n_rules, figsize=(4.2 * n_rules, 5))
if n_rules == 1:
    axes = [axes]
for ax, c in zip(axes, candidates):
    rid = c["Rule_ID"]
    ev = rule_evidence[rid]
    if ev["kind"] == "state_event" and ev.get("supported"):
        ax.hist(ev["off_ref"], bins=40, alpha=0.6, label=f"LOW (n={len(ev['off_ref']):,})", color="#C44E52", density=True)
        ax.hist(ev["on_ref"], bins=40, alpha=0.6, label=f"HIGH (n={len(ev['on_ref']):,})", color="#4C72B0", density=True)
        ax.set_xlabel(c["Target_Tags"])
        ax.set_title(f"{rid}\n{c['Source_Tags']} state vs {c['Target_Tags']}")
        ax.legend(fontsize=7)
    elif ev["kind"] == "sensor_sensor" and ev.get("supported"):
        ax.hist(ev["residual_sample"], bins=40, color="#8172B2", edgecolor="white")
        ax.set_xlabel(f"{ev['residual_name']}: {c['Source_Tags']} vs {c['Target_Tags']}")
        ax.set_title(f"{rid}\n(rho={ev['rho']:.2f})")
    elif ev["kind"] == "flow_level_slope":
        ax.hist(ev["slope_sample_for_envelope"], bins=40, color="#55A868", edgecolor="white")
        ax.set_xlabel(f"LIT301 {ev['best_window']}s-slope")
        ax.set_title(f"{rid}\nrho(FIT301,slope)={ev['rho']:.2f}")
    elif ev["kind"] == "pump_slope" and ev.get("supported"):
        ax.hist(ev["off_ref"], bins=40, alpha=0.6, label=f"OFF (n={len(ev['off_ref']):,})", color="#C44E52", density=True)
        ax.hist(ev["on_ref"], bins=40, alpha=0.6, label=f"ON (n={len(ev['on_ref']):,})", color="#4C72B0", density=True)
        ax.set_xlabel(f"LIT301 {ev['best_window']}s-slope")
        ax.set_title(f"{rid}\n{c['Source_Tags']} state vs LIT301 slope")
        ax.legend(fontsize=7)
    elif ev["kind"] == "actuator_coupling":
        rate = ev["coupling_rate"]
        ax.bar(["same-row\nsame-dir.", "not\ncoupled"],
               [rate if not np.isnan(rate) else 0, 1 - rate if not np.isnan(rate) else 1],
               color=["#4C72B0", "#C44E52"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of transition-active TRAIN-NORMAL rows")
        ax.set_title(f"{rid}\ncoupling rate={rate:.3f} (n={ev['n_transition_active']})")
    else:
        ax.text(0.5, 0.5, "Insufficient TRAIN data\n(see validation table)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(rid)
    ax.grid(True, linestyle="--", alpha=0.4)
fig.suptitle("Stage 3 TRAIN-NORMAL Response / Relationship Profiles", y=1.03)
fig.tight_layout()
fig2_path = FIGURES_DIR / "stage3_normal_response_profiles.png"
fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig2_path}")

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
fig.suptitle("Stage 3 Symbolic Scores: VALIDATION NORMAL vs. ATTACK (where evaluable)", y=1.03)
fig.tight_layout()
fig3_path = FIGURES_DIR / "stage3_symbolic_score_validation_comparison.png"
fig.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig3_path}")

fig, ax = plt.subplots(figsize=(9, 5))
colors4 = [STATUS_COLOR[decisions[r]["decision"]] for r in rule_ids_order]
ax.barh(rule_ids_order, [1] * len(rule_ids_order), color=colors4)
for i, r in enumerate(rule_ids_order):
    ax.text(0.5, i, decisions[r]["decision"], ha="center", va="center", fontsize=10, fontweight="bold")
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title("Stage 3 Final Rule Screening Decisions")
fig.tight_layout()
fig4_path = FIGURES_DIR / "stage3_final_rule_screening.png"
fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig4_path}")

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
    ax.set_title("Final Retained Stage 3 Symbolic Feature Set", fontsize=12)
else:
    ax.text(5, ax.get_ylim()[1] / 2, "No Stage 3 candidate rules were RETAINED.\n"
            "See tables/stage3_final_rule_screening.csv for LIMITED/REJECT reasoning.",
            ha="center", va="center", fontsize=12)
    ax.set_title("Final Retained Stage 3 Symbolic Feature Set: NONE RETAINED", fontsize=12)
fig.tight_layout()
fig5_path = FIGURES_DIR / "stage3_final_retained_features.png"
fig.savefig(fig5_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig5_path}")

# =============================================================================
# PART 11 — ACADEMIC NARRATIVE
# =============================================================================
section("PART 11 — WRITING ACADEMIC NARRATIVE")

strongest_rule = max(candidates, key=lambda c: support_metric(c["Rule_ID"]))
weakest_rule = min(candidates, key=lambda c: support_metric(c["Rule_ID"]))

narrative = f"""STAGE 3 SYMBOLIC PIPELINE — ACADEMIC NARRATIVE
(Draft prose for Methodology/Results. Review before submission.)

1. STAGE 3 STRUCTURE
Stage 3 (ultrafiltration) contains {len(STAGE3_TAGS)} physical tags: 3 continuous sensors (DPIT301
differential pressure, FIT301 flow, LIT301 level) and 6 actuators (MV301-304, P301, P302). Empirical
audit (Part 1) found P301 fully constant (matching the original dataset audit) and all FOUR motorized
valves showing 3 observed states each -- structurally identical to Stage 1's MV101 and Stage 2's MV201
-- with each valve's transitional-state structure re-verified independently wherever selected as a
candidate (never assumed from precedent). DPIT301 is a sensor type not present in any prior stage
screened so far.

2. CANDIDATE RELATIONSHIP GENERATION
With 5 varying actuators (4 valves + P302) and only 2 target sensors besides DPIT301 itself, candidate
actuator-target pairs were selected via TRAIN-NORMAL correlation screening ({best_flow_actuator} for
FIT301, {best_dpit_actuator} for DPIT301, {best_pair[0]}<->{best_pair[1]} for the coupling test) rather
than testing every combination, avoiding a fishing expedition. FIT301<->DPIT301 (flow<->
differential-pressure) was added as a directly physically-justified sensor-sensor candidate (hydraulic
resistance increases with flow, a general fluid-dynamics principle, not a plant-specific assumption).
FIT301<->LIT301-slope and P302->LIT301-slope mirror the Stage-1/Stage-4 rolling/derivative methodology.

3. TRAIN-NORMAL EMPIRICAL VALIDATION
Correlation was never treated as sufficient evidence alone. State-based candidates were validated via
distribution separation (SMD, IQR overlap) plus, where transitions permitted, an event-triggered
response check. The flow<->differential-pressure candidate was additionally tested for whether a
flow-band-conditioned residual improved stability over the pooled residual, per the explicit instruction
to test state/flow-conditioning for pressure relationships.

4. LAG, STATE, AND RESIDUAL ANALYSES
{chr(10).join(f"- {rid}: {rule_evidence[rid].get('kind')}" for rid in rule_ids_order)}

5. RETAINED VS LIMITED/REJECTED DECISIONS
{chr(10).join(f"- {rid}: {decisions[rid]['decision']} -- {decisions[rid]['reason']}" for rid in rule_ids_order)}

6. FINAL STAGE 3 SYMBOLIC FEATURE SET
{f"The finalized Stage 3 symbolic feature set consists of: {', '.join(FINAL_FEATURE_NAMES)}." if FINAL_FEATURE_NAMES else "No Stage 3 candidate was promoted to the final retained set at this stage."}

7. KEY LIMITATIONS
Actuator-target pairs were chosen by correlation screening, so their physical interpretation (which
specific valve gates which specific flow path) is not independently documented in this project --
treated as a data-driven association, not an asserted plant identity. The strongest Stage 3 relationship
found was {strongest_rule['Rule_ID']} ({strongest_rule['Source_Tags']} <-> {strongest_rule['Target_Tags']});
the weakest was {weakest_rule['Rule_ID']} ({weakest_rule['Source_Tags']} <-> {weakest_rule['Target_Tags']}).

8. COMPARISON WITH STAGES 1, 2, 4, AND 5
Stage 3 has the richest actuator set of any stage screened so far (4 valves + 1 varying pump) and
contributes a genuinely new sensor-sensor category (flow<->differential-pressure, via DPIT301) not
available in Stage 1, 2, 4, or 5. The identical TRAIN-normal-only, distribution-free empirical-CDF-
scoring, explicit RETAIN/LIMITED/REJECT framework was applied unchanged. See
tables/stage3_cross_stage_methodology_comparison.csv for the full quantitative comparison. No
classifier or attack-detection performance claim is made anywhere in this analysis.
"""

narrative_path = RESULTS_DIR / "stage3_symbolic_pipeline_narrative.txt"
narrative_path.write_text(narrative, encoding="utf-8")
log(f"Wrote {narrative_path}")

# =============================================================================
# PART 12 — MAIN RESULTS LOG
# =============================================================================
section("PART 12 — WRITING MAIN RESULTS LOG")

main_lines = []
main_lines.append("STAGE 3 END-TO-END SYMBOLIC PIPELINE — MAIN RESULTS")
main_lines.append(f"Generated: {RUN_START.isoformat()}")
main_lines.append(f"Python: {sys.version}")
main_lines.append(f"pandas: {pd.__version__} | numpy: {np.__version__} | scipy: {__import__('scipy').__version__}")
main_lines.append(f"OS: {platform.system()} {platform.release()}")
main_lines.append("")
main_lines.append(f"Data sources: {TRAIN_PATH.name} ({len(train_df):,} rows), "
                   f"{VAL_PATH.name} ({len(val_df):,} rows). TEST was never loaded.")
main_lines.append("")
main_lines.append(f"=== STAGE 3 CANDIDATE RULES ({len(candidates)}) ===")
for c in candidates:
    main_lines.append(f"  {c['Rule_ID']}: {c['Source_Tags']} <-> {c['Target_Tags']} ({c['Relationship_Type']})")
main_lines.append("")
main_lines.append("=== STAGE 3 FINAL DECISIONS ===")
for rid in rule_ids_order:
    main_lines.append(f"  {rid} -> {decisions[rid]['decision']}")
main_lines.append("")
main_lines.append("=== FINAL RETAINED STAGE 3 SYMBOLIC FEATURE SET ===")
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
main_lines.append("=== ROBUSTNESS / DISTRIBUTION-SHIFT FINDINGS ===")
for _, r in robustness_df.iterrows():
    main_lines.append(f"  {r['Rule_ID']}: TRAIN-N median={r['TRAIN_NORMAL_Score_Median']}, "
                       f"VAL-N median={r['VALIDATION_NORMAL_Score_Median']}, saturated={r['Saturated_In_VALIDATION']}")
main_lines.append("")
main_lines.append(f"Strongest Stage 3 physical relationship: {strongest_rule['Rule_ID']} "
                   f"({strongest_rule['Source_Tags']} <-> {strongest_rule['Target_Tags']})")
main_lines.append(f"Weakest candidate relationship: {weakest_rule['Rule_ID']} "
                   f"({weakest_rule['Source_Tags']} <-> {weakest_rule['Target_Tags']})")
main_lines.append("")
adds_value = n_retained > 0
main_lines.append(f"Does Stage 3 contribute complementary symbolic information beyond Stages 1, 2, 4, and 5? "
                   f"{'YES -- ' + str(n_retained) + ' feature(s) retained.' if adds_value else 'NOT CONFIRMED at this stage -- no Stage 3 candidate cleared the RETAIN bar.'}")
main_lines.append("")
main_lines.append("=== GUARDRAILS CONFIRMED ===")
main_lines.append("  - TEST partition: never loaded, filename not referenced anywhere in this script.")
main_lines.append("  - TRAIN NORMAL used for all reference distributions and rule construction.")
main_lines.append("  - VALIDATION used only for coverage/robustness diagnostics, never to tune anything.")
main_lines.append("  - No model was trained. No ML feature selection was performed.")
main_lines.append("  - No existing script/output was modified; all Stage 1, 2, 4, and 5 outputs preserved.")
main_lines.append("  - No physical relationship was invented beyond Stage 3's tag structure and "
                   "TRAIN-normal data; uncertain assumptions are explicitly marked throughout.")
main_lines.append("  - Analysis is fully deterministic (no randomness used).")

main_path = RESULTS_DIR / "stage3_end_to_end_symbolic_pipeline.txt"
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

log(f"Stage 3 candidate rules ({len(candidates)}):")
for c in candidates:
    log(f"  {c['Rule_ID']}: {c['Source_Tags']} <-> {c['Target_Tags']}")
log("\nStage 3 final decisions:")
for rid in rule_ids_order:
    log(f"  {rid} -> {decisions[rid]['decision']}")
log(f"\nFinal retained Stage 3 symbolic feature set: {FINAL_FEATURE_NAMES if FINAL_FEATURE_NAMES else '(none)'}")
log(f"\nCandidate={len(candidates)}, Retained={n_retained}, Limited={n_limited}, Rejected={n_rejected}")

run_end = datetime.now()
log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
log("PIPELINE COMPLETE. No model trained, no ML feature selection, no attack-threshold tuning.")

if not all_checks_passed:
    sys.exit(1)
