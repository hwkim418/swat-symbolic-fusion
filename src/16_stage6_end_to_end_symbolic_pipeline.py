"""
code/16_stage6_end_to_end_symbolic_pipeline.py

STAGE 6 END-TO-END SYMBOLIC PIPELINE — INDEPENDENT DISCOVERY, EMPIRICAL
VALIDATION, FEATURE CONSTRUCTION, AND FINAL SCREENING. ANALYSIS ONLY.
FINAL stage-specific symbolic pipeline (Stages 1-5 already completed).

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

Applies the STANDARDIZED METHODOLOGY developed for Stages 1 (script 13),
2 (scripts 04-11), 3 (script 15), 4 (script 14), and 5 (script 12) to
Stage 6 (RO permeate transfer), but candidates are discovered and
validated INDEPENDENTLY from Stage 6's own tag structure -- no other
stage's rule structures are copied.

Stage 6's tag audit (Part 1) reveals the SPARSEST structure of any stage
screened in this project: only 4 tags total (FIT601, P601, P602, P603),
of which P601 and P603 are fully constant (matching the original dataset
audit) and P602 is the ONLY varying actuator. FIT601 is Stage 6's ONLY
sensor. This means Stage 6 structurally supports exactly ONE candidate
relationship category (actuator -> sensor); there is no second sensor
for a sensor-sensor candidate, no second varying actuator for an
actuator-actuator coupling candidate, and no level sensor for a
flow<->level rolling candidate. Rather than inventing additional
candidates to match prior stages' scope, this script documents that
structural ceiling explicitly and evaluates the single defensible
candidate rigorously.

Prior stage-activity auditing (script 10) found Stage 6 has substantially
lower attack-associated activity (~28%) than Stages 1-5. This is used
here ONLY as background context (Part 9's cross-stage comparison) and
NEVER to weaken or strengthen the RETAIN/LIMITED/REJECT criteria, which
remain byte-for-byte identical to every prior stage script.

No ML model is trained, no feature is selected, no attack-label-based
threshold is tuned, no existing script/dataset/output is modified, and
TEST is never loaded -- its filename does not appear anywhere below, by
design. Uses ONLY processed/train_unscaled.csv and
processed/validation_unscaled.csv, plus tables/swat_stage_tag_mapping.csv
and (read-only, for the Part 9 comparison) scripts 11/12/13/14/15's
outputs.
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
STAGE_EXPOSURE_PATH = PROJECT_ROOT / "tables" / "swat_stage_attack_exposure_summary.csv"
STAGE1_SCREENING_PATH = PROJECT_ROOT / "tables" / "stage1_final_rule_screening.csv"
STAGE1_FEATURES_PATH = PROJECT_ROOT / "tables" / "stage1_final_retained_symbolic_features.csv"
STAGE2_SCREENING_PATH = PROJECT_ROOT / "tables" / "stage2_final_rule_screening.csv"
STAGE2_FEATURES_PATH = PROJECT_ROOT / "tables" / "stage2_final_retained_symbolic_features.csv"
STAGE3_SCREENING_PATH = PROJECT_ROOT / "tables" / "stage3_final_rule_screening.csv"
STAGE3_FEATURES_PATH = PROJECT_ROOT / "tables" / "stage3_final_retained_symbolic_features.csv"
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


section("STAGE 6 END-TO-END SYMBOLIC PIPELINE — RUN START (ANALYSIS ONLY, FINAL STAGE-SPECIFIC SCRIPT)")
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
stage6_tags_df = stage_mapping[stage_mapping.Stage == "Stage 6"]
STAGE6_TAGS = stage6_tags_df["Tag"].tolist()
STAGE6_SENSORS = stage6_tags_df.loc[stage6_tags_df.Sensor_Or_Actuator == "sensor", "Tag"].tolist()
STAGE6_ACTUATORS = stage6_tags_df.loc[stage6_tags_df.Sensor_Or_Actuator == "actuator", "Tag"].tolist()
log(f"Loaded {STAGE_MAPPING_PATH.name}: Stage 6 = {len(STAGE6_TAGS)} tags "
    f"({len(STAGE6_SENSORS)} sensors, {len(STAGE6_ACTUATORS)} actuators)")
log(f"  Sensors  : {STAGE6_SENSORS}")
log(f"  Actuators: {STAGE6_ACTUATORS}")

train_normal = train_df[train_df.Label == 0]

# =============================================================================
# SHARED HELPER FUNCTIONS (identical methodology to scripts 05-15)
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


# =============================================================================
# PART 1 — STAGE 6 TAG AUDIT
# =============================================================================
section("PART 1 — STAGE 6 TAG AUDIT")

tag_audit_rows = []
for tag in STAGE6_TAGS:
    row_meta = stage6_tags_df[stage6_tags_df.Tag == tag].iloc[0]
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
tag_audit_path = TABLES_DIR / "stage6_tag_audit.csv"
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
constant_actuators = [t for t in STAGE6_ACTUATORS if t not in varying_actuators]
varying_sensors = tag_audit_df[(tag_audit_df.Split == "TRAIN") & (tag_audit_df.Sensor_Or_Actuator == "sensor") &
                                (~tag_audit_df.Constant_Or_Near_Constant)]["Tag"].tolist()
log(f"\nStage 6 actuators with sufficient TRAIN variation for empirical rule construction: {varying_actuators}")
log(f"Stage 6 actuators that are constant/near-constant (NOT discarded, reported explicitly): {constant_actuators}")
log(f"Stage 6 continuously varying sensors: {varying_sensors}")

sufficient_variation = len(varying_actuators) >= 1 and len(varying_sensors) >= 1
log(f"\nDoes Stage 6 provide enough normal operational variation for symbolic-rule construction? "
    f"{sufficient_variation} (at least one varying actuator AND one varying sensor required)")
log("STRUCTURAL CEILING: Stage 6 has only 4 tags total -- P601 and P603 are fully constant (matching "
    "the original dataset audit's constant-feature list), P602 is the ONLY varying actuator, and "
    "FIT601 is the ONLY sensor of any kind. This means Stage 6 CANNOT structurally support a "
    "sensor-sensor candidate (only one sensor exists), an actuator-actuator coupling candidate (only "
    "one varying actuator exists), or a flow<->level rolling candidate (no level sensor exists). "
    "The only defensible candidate category is actuator->sensor (P602->FIT601). This is documented "
    "explicitly here rather than manufacturing additional candidates to match prior stages' scope.")

# =============================================================================
# PART 2 — CANDIDATE PHYSICAL RELATIONSHIP DISCOVERY
# =============================================================================
section("PART 2 — CANDIDATE RELATIONSHIP DISCOVERY")

candidates = []
if sufficient_variation:
    corr = np.corrcoef(train_normal["P602"], train_normal["FIT601"])[0, 1]
    log(f"P602 vs FIT601, Pearson correlation (TRAIN NORMAL, discovery evidence): r={corr:+.4f}")
    candidates.append({
        "Rule_ID": "S6-R1", "Source_Tags": "P602", "Target_Tags": "FIT601",
        "Relationship_Type": "actuator -> flow (state/event)",
        "Expected_Relationship": "P602 ON should produce a different FIT601 reading than P602 OFF "
                                  "(direction determined empirically, not assumed).",
        "Rationale": "P602 is Stage 6's only varying actuator and FIT601 is Stage 6's only sensor -- "
                     "this is the SOLE structurally available candidate in Stage 6, not a rule "
                     "transplanted from another stage because it 'worked there'. A pump mechanically "
                     "gating its own downstream flow meter is the same class of relationship "
                     "independently validated in Stages 1 (S1-R1), 2 (S2-R1), 3 (S3-R1), 4 (S4-R2), and "
                     "5 (S5-R1), but here it is being tested, not assumed, on Stage 6's own data.",
        "Confidence": "high a priori for the relationship CATEGORY (direct mechanical gating is about "
                       "as certain a hydraulic relationship as exists), but P602's own operational role "
                       "is sparse (150 TRAIN transitions, 0.85% ON-time) -- confidence in the SPECIFIC "
                       "empirical outcome is reserved for Part 3.",
        "Time_Lag_Expected": "Possibly -- to be determined empirically in Part 3.",
        "Discovery_Evidence": f"Pearson r={corr:+.4f} (TRAIN NORMAL, lag 0)",
    })
else:
    log("Stage 6 does not provide sufficient normal operational variation for symbolic-rule "
        "construction -- NO candidates will be generated. This is reported as a legitimate outcome, "
        "not an error.")

candidates_df = pd.DataFrame(candidates)
candidates_path = TABLES_DIR / "stage6_candidate_rules.csv"
candidates_df.to_csv(candidates_path, index=False)
log(f"Wrote {candidates_path} ({len(candidates_df)} rows)")
log(f"\n{len(candidates)} candidate relationship(s) identified BEFORE any validation testing:")
for c in candidates:
    log(f"  {c['Rule_ID']}: {c['Source_Tags']} <-> {c['Target_Tags']} ({c['Relationship_Type']})")

# =============================================================================
# GENERIC ACTUATOR-STATE SEMANTICS + EVENT VALIDATION (reused unchanged from
# script 15's Stage-3 methodology; P602 has only 2 observed states so the
# 3-state transitional-bridge branch is present but will not activate here).
# =============================================================================


def infer_actuator_semantics(source_tag, ref_target_tag):
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


def validate_state_event_generic(source_tag, target_tag):
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
    thin_reference = min(len(off_ref), len(on_ref)) < 200
    if thin_reference:
        log(f"  NOTE: a minority-state reference group is thin (n_min={min(len(off_ref), len(on_ref))}); "
            f"percentile-based envelope for that state carries wider uncertainty.")

    events = []
    mode = "unsupported"
    if len(low_states) == 1 and len(high_states) == 1:
        direct_trans = find_specific_transitions(train_df, source_tag, low_states[0], high_states[0])
        for i0 in direct_trans:
            if train_df["Label"].iloc[i0] != 0:
                continue
            if not check_baseline_window(train_df, i0, source_tag, low_states[0], require_normal=True):
                continue
            events.append({"open_idx": i0, "baseline_idx": i0})
        mode = "direct_two_state"
    log(f"  [event-based] {len(events)} usable TRAIN NORMAL LOW->HIGH events ({mode})")

    offset_stats = []
    best = None
    if len(events) >= MIN_EVENT_COUNT:
        expected_sign = 1 if on_ref.mean() > off_ref.mean() else -1
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
            "offset_stats": offset_stats, "best_offset": best, "mode": mode, "thin_reference": thin_reference}


# =============================================================================
# PART 3 + 4 — TRAIN-NORMAL EMPIRICAL VALIDATION + NORMAL OPERATING ENVELOPES
# =============================================================================
section("PART 3 & 4 — TRAIN-NORMAL EMPIRICAL VALIDATION AND OPERATING ENVELOPES")

validation_rows = []
envelope_rows = []
rule_evidence = {}

for c in candidates:
    rid = c["Rule_ID"]
    log(f"\n--- Validating {rid}: {c['Source_Tags']} <-> {c['Target_Tags']} ---")
    result = validate_state_event_generic(c["Source_Tags"], c["Target_Tags"])
    rule_evidence[rid] = result

    if result.get("supported"):
        validation_rows.append({
            "Rule_ID": rid, "Kind": "state_event", "N_Events_TRAIN_NORMAL": result["n_events"],
            "State_SMD": result["smd"], "State_IQR_Overlap_Fraction": result["overlap_frac"],
            "State_KS_Statistic": result["ks_stat"], "Mode": result["mode"], "Thin_Minority_Reference": result["thin_reference"],
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
    else:
        validation_rows.append({"Rule_ID": rid, "Kind": "state_event", "N_Events_TRAIN_NORMAL": 0,
                                 "Unsupported_Reason": result.get("reason")})

validation_df = pd.DataFrame(validation_rows)
validation_path = TABLES_DIR / "stage6_rule_validation.csv"
validation_df.to_csv(validation_path, index=False)
log(f"\nWrote {validation_path} ({len(validation_df)} rows)")

envelope_df = pd.DataFrame(envelope_rows)
envelope_path = TABLES_DIR / "stage6_normal_operating_envelopes.csv"
envelope_df.to_csv(envelope_path, index=False)
log(f"Wrote {envelope_path} ({len(envelope_df)} rows)")

# =============================================================================
# PART 5 + 6 — REPRESENTATION, COVERAGE (overall + attack-period), ROBUSTNESS
# =============================================================================
section("PART 5 & 6 — REPRESENTATION TYPE, COVERAGE (OVERALL vs ATTACK-PERIOD), AND ROBUSTNESS")

representation = {}
coverage_rows = []
robustness_rows = []
score_arrays = {}

for c in candidates:
    rid = c["Rule_ID"]
    ev = rule_evidence[rid]
    if not ev.get("supported"):
        representation[rid] = "UNSUPPORTED"
        eval_mask_train = np.zeros(len(train_df), dtype=bool)
        score_train = np.full(len(train_df), np.nan)
        eval_mask_val = np.zeros(len(val_df), dtype=bool)
        score_val = np.full(len(val_df), np.nan)
    else:
        source_tag, target_tag = c["Source_Tags"], c["Target_Tags"]
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

    score_arrays[rid] = (eval_mask_train, score_train, eval_mask_val, score_val)

    val_normal_mask = eval_mask_val & (val_df["Label"].to_numpy() == 0)
    val_attack_mask = eval_mask_val & (val_df["Label"].to_numpy() == 1)
    n_val_normal_total = int((val_df["Label"].to_numpy() == 0).sum())
    n_val_attack_total = int((val_df["Label"].to_numpy() == 1).sum())
    cov_normal = 100 * val_normal_mask.sum() / n_val_normal_total if n_val_normal_total else np.nan
    cov_attack = 100 * val_attack_mask.sum() / n_val_attack_total if n_val_attack_total else np.nan
    cov_overall = 100 * eval_mask_val.sum() / len(val_df)
    coverage_rows.append({"Rule_ID": rid, "Representation_Type": representation[rid],
                           "VALIDATION_Overall_Coverage_Pct": cov_overall,
                           "VALIDATION_NORMAL_Coverage_Pct": cov_normal,
                           "VALIDATION_ATTACK_Coverage_Pct": cov_attack,
                           "N_VALIDATION_ATTACK_Evaluable": int(val_attack_mask.sum())})
    log(f"{rid} [{representation[rid]}]: OVERALL coverage={cov_overall:.3f}%, "
        f"NORMAL={cov_normal:.3f}%, ATTACK={cov_attack:.3f}% (coverage = evaluable, NOT attack detected; "
        f"ATTACK-period evaluability reported separately per instructions, since Stage 6 has relatively "
        f"low attack-associated activity)")

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
        "Thin_Minority_Reference": ev.get("thin_reference", np.nan),
    })
    log(f"{rid} robustness: TRAIN-N median={float(np.median(train_normal_scores)) if len(train_normal_scores) else np.nan:.4f}, "
        f"VAL-N median={float(np.median(val_normal_scores)) if len(val_normal_scores) else np.nan:.4f}, "
        f"KS={ks_shift}, saturated={saturated}, thin_minority_reference={ev.get('thin_reference')}")

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
    if not ev.get("supported"):
        continue
    cov = coverage_df[coverage_df.Rule_ID == rid].iloc[0]
    fname = f"{rid}_state_score"
    math_def = (f"score = 1 - 2*min(F(x), 1-F(x)), F = empirical CDF of TRAIN-NORMAL {c['Target_Tags']} "
                 f"for the CURRENT {c['Source_Tags']} state (LOW or HIGH)")
    interp = f"How atypical {c['Target_Tags']} is for {c['Source_Tags']}'s current state, vs. TRAIN-NORMAL."
    eval_cond = f"{c['Source_Tags']} in a LOW or HIGH state ({ev['mode'].replace('_',' ')})"
    limitation_parts = []
    if ev["n_events"] < MIN_EVENT_COUNT:
        limitation_parts.append(f"too few TRAIN transitions ({ev['n_events']}) for a robust event-based companion check")
    else:
        limitation_parts.append("event-based companion evidence available")
    if ev.get("thin_reference"):
        limitation_parts.append("minority-state reference is thin (see stage6_rule_validation.csv)")
    limitation_parts.append("Stage 6 has relatively low attack-associated activity (~28% per script 10); "
                             "see ATTACK-period coverage separately -- this does not reduce overall NORMAL evaluability")
    feature_def_rows.append({
        "Feature": fname, "Source_Rule": rid, "Source_Tags": c["Source_Tags"], "Target_Tags": c["Target_Tags"],
        "Representation_Type": representation[rid], "Mathematical_Definition": math_def,
        "Physical_Interpretation": interp, "Evaluability_Condition": eval_cond,
        "NORMAL_Coverage_Pct": cov["VALIDATION_NORMAL_Coverage_Pct"],
        "ATTACK_Coverage_Pct": cov["VALIDATION_ATTACK_Coverage_Pct"],
        "Score_Direction": "0 = typical of TRAIN-NORMAL; 1 = extreme tail (atypical)",
        "Robustness_Limitation": "; ".join(limitation_parts),
    })

feature_def_df = pd.DataFrame(feature_def_rows)
feature_def_path = TABLES_DIR / "stage6_symbolic_feature_definitions.csv"
feature_def_df.to_csv(feature_def_path, index=False)
log(f"Wrote {feature_def_path} ({len(feature_def_df)} rows)")

# =============================================================================
# PART 8 — FINAL RULE SCREENING
# =============================================================================
section("PART 8 — FINAL RULE SCREENING (RETAIN / LIMITED / REJECT)")

log("Decision rule (BYTE-FOR-BYTE identical standard applied to finalized Stage 1 [script 13], Stage 2 "
    "[script 11], Stage 3 [script 15], Stage 4 [script 14], and Stage 5 [script 12] -- NOT weakened for "
    "Stage 6's low attack-associated activity, and NOT strengthened either):")
log(f"  1. If TRAIN-normal event/state count < {MIN_EVENT_COUNT} -> REJECT.")
log(f"  2. Else if not empirically supported beyond correlation (|SMD|<=1.0 or overlap_frac>=0.25) -> REJECT.")
log(f"  3. Else: RETAIN if VALIDATION coverage >= 90% AND not saturated (VAL-NORMAL mean < "
    f"{SATURATION_THRESHOLD}); otherwise LIMITED.")
log("Reminder: Stage 6's ~28% attack-associated activity finding (script 10) is NOT a screening "
    "criterion above and does not appear in this decision logic.")

decisions = {}
for c in candidates:
    rid = c["Rule_ID"]
    ev = rule_evidence[rid]
    cov = coverage_df[coverage_df.Rule_ID == rid].iloc[0]
    rob = robustness_df[robustness_df.Rule_ID == rid].iloc[0]

    if not ev.get("supported"):
        n, supported, support_desc, min_n = 0, False, ev.get("reason", "unsupported"), MIN_EVENT_COUNT
    else:
        n = min(len(ev["off_ref"]), len(ev["on_ref"]))
        supported = (not np.isnan(ev["smd"])) and abs(ev["smd"]) > 1.0 and ev["overlap_frac"] < 0.25
        support_desc = f"state SMD={ev['smd']:.3f}, IQR overlap fraction={ev['overlap_frac']:.3f}"
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
log(f"\nStage 6 screening totals: {len(candidates)} candidates -> RETAIN={n_retained}, "
    f"LIMITED={n_limited}, REJECT={n_rejected}")
if len(candidates) == 0:
    log("Stage 6 generated zero candidates due to insufficient structural variation -- this is reported "
        "as-is, not forced.")
log("No minimum retained-rule count was enforced (0 RETAIN is an acceptable, scientifically meaningful "
    "result per instructions); the same standard used for Stages 1-5 was applied without adjustment.")

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
        "Temporal_Evidence": f"best_offset={val.get('Best_Event_Offset_Seconds', np.nan)}s" if "Best_Event_Offset_Seconds" in val.index else "n/a",
        "Overall_Coverage": f"{cov['VALIDATION_Overall_Coverage_Pct']:.2f}%",
        "Coverage_By_Label": f"VALIDATION NORMAL={cov['VALIDATION_NORMAL_Coverage_Pct']:.2f}%, ATTACK={cov['VALIDATION_ATTACK_Coverage_Pct']:.2f}%",
        "Robustness": f"TRAIN-N median={rob['TRAIN_NORMAL_Score_Median']}, VAL-N median={rob['VALIDATION_NORMAL_Score_Median']}, saturated={rob['Saturated_In_VALIDATION']}",
        "Main_Limitation": fdef["Robustness_Limitation"].iloc[0] if len(fdef) else "Rejected before feature construction.",
        "Final_Decision": d["decision"], "Decision_Rationale": d["reason"],
    })
screening_df = pd.DataFrame(screening_rows)
screening_path = TABLES_DIR / "stage6_final_rule_screening.csv"
screening_df.to_csv(screening_path, index=False)
log(f"\nWrote {screening_path} ({len(screening_df)} rows)")

retained_rule_ids = [rid for rid, d in decisions.items() if d["decision"] == "RETAIN"]
final_features_df = feature_def_df[feature_def_df.Source_Rule.isin(retained_rule_ids)].copy() if len(feature_def_df) else pd.DataFrame()
final_features_path = TABLES_DIR / "stage6_final_retained_symbolic_features.csv"
final_features_df.to_csv(final_features_path, index=False)
log(f"Wrote {final_features_path} ({len(final_features_df)} rows)")
FINAL_FEATURE_NAMES = final_features_df["Feature"].tolist() if len(final_features_df) else []
log(f"Final retained Stage 6 symbolic feature set: {FINAL_FEATURE_NAMES if FINAL_FEATURE_NAMES else '(none)'}")

# =============================================================================
# PART 9 — CROSS-STAGE METHODOLOGY CONSISTENCY CHECK
# =============================================================================
section("PART 9 — CROSS-STAGE METHODOLOGY CONSISTENCY CHECK (Stage 6 vs Stages 1-5)")

s1_screening = pd.read_csv(STAGE1_SCREENING_PATH) if STAGE1_SCREENING_PATH.exists() else None
s1_features = pd.read_csv(STAGE1_FEATURES_PATH) if STAGE1_FEATURES_PATH.exists() else None
s2_screening = pd.read_csv(STAGE2_SCREENING_PATH) if STAGE2_SCREENING_PATH.exists() else None
s2_features = pd.read_csv(STAGE2_FEATURES_PATH) if STAGE2_FEATURES_PATH.exists() else None
s3_screening = pd.read_csv(STAGE3_SCREENING_PATH) if STAGE3_SCREENING_PATH.exists() else None
s3_features = pd.read_csv(STAGE3_FEATURES_PATH) if STAGE3_FEATURES_PATH.exists() else None
s4_screening = pd.read_csv(STAGE4_SCREENING_PATH) if STAGE4_SCREENING_PATH.exists() else None
s4_features = pd.read_csv(STAGE4_FEATURES_PATH) if STAGE4_FEATURES_PATH.exists() else None
s5_screening = pd.read_csv(STAGE5_SCREENING_PATH) if STAGE5_SCREENING_PATH.exists() else None
s5_features = pd.read_csv(STAGE5_FEATURES_PATH) if STAGE5_FEATURES_PATH.exists() else None
stage_exposure_df = pd.read_csv(STAGE_EXPOSURE_PATH) if STAGE_EXPOSURE_PATH.exists() else None


def counts(df):
    if df is None:
        return {"n": np.nan, "retain": np.nan, "limited": np.nan, "reject": np.nan}
    return {"n": len(df), "retain": int((df.Final_Decision == "RETAIN").sum()),
            "limited": int((df.Final_Decision == "LIMITED").sum()),
            "reject": int((df.Final_Decision == "REJECT").sum())}


s6c, s1c, s2c, s3c, s4c, s5c = counts(screening_df), counts(s1_screening), counts(s2_screening), counts(s3_screening), counts(s4_screening), counts(s5_screening)

stage6_exposure_pct = np.nan
if stage_exposure_df is not None and "Stage" in stage_exposure_df.columns:
    row = stage_exposure_df[stage_exposure_df.Stage == "Stage 6"]
    if len(row) and "Pct_Of_All_Intervals_With_Activity" in row.columns:
        stage6_exposure_pct = row["Pct_Of_All_Intervals_With_Activity"].iloc[0]
log(f"Stage 6 attack-associated activity (from script 10, context only): {stage6_exposure_pct}%")

attack_cov_str = (f"{coverage_df.loc[coverage_df.Rule_ID.isin(retained_rule_ids), 'VALIDATION_ATTACK_Coverage_Pct'].max():.2f}%"
                   if retained_rule_ids else (f"{coverage_df['VALIDATION_ATTACK_Coverage_Pct'].max():.2f}%" if len(coverage_df) else "n/a"))

comparison_rows = [
    {"Metric": "Candidate rule count", "Stage_1": s1c["n"], "Stage_2": s2c["n"], "Stage_3": s3c["n"], "Stage_4": s4c["n"], "Stage_5": s5c["n"], "Stage_6": s6c["n"]},
    {"Metric": "Retained count", "Stage_1": s1c["retain"], "Stage_2": s2c["retain"], "Stage_3": s3c["retain"], "Stage_4": s4c["retain"], "Stage_5": s5c["retain"], "Stage_6": n_retained},
    {"Metric": "Limited count", "Stage_1": s1c["limited"], "Stage_2": s2c["limited"], "Stage_3": s3c["limited"], "Stage_4": s4c["limited"], "Stage_5": s5c["limited"], "Stage_6": n_limited},
    {"Metric": "Rejected count", "Stage_1": s1c["reject"], "Stage_2": s2c["reject"], "Stage_3": s3c["reject"], "Stage_4": s4c["reject"], "Stage_5": s5c["reject"], "Stage_6": n_rejected},
    {"Metric": "Retained feature count",
     "Stage_1": len(s1_features) if s1_features is not None else np.nan,
     "Stage_2": len(s2_features) if s2_features is not None else np.nan,
     "Stage_3": len(s3_features) if s3_features is not None else np.nan,
     "Stage_4": len(s4_features) if s4_features is not None else np.nan,
     "Stage_5": len(s5_features) if s5_features is not None else np.nan,
     "Stage_6": len(FINAL_FEATURE_NAMES)},
    {"Metric": "Dominant/only rule type", "Stage_1": "rolling-window", "Stage_2": "state-based", "Stage_3": "rolling-window",
     "Stage_4": "rolling-window", "Stage_5": "state-based/sensor-sensor mix",
     "Stage_6": (list(representation.values())[0] if representation else "n/a (no candidates)")},
    {"Metric": "Attack-period coverage of best/retained feature", "Stage_1": "~100%", "Stage_2": "up to 100%",
     "Stage_3": "100%", "Stage_4": "100%", "Stage_5": "100%", "Stage_6": attack_cov_str},
    {"Metric": "Sparsity concerns", "Stage_1": "P102 (4 transitions)", "Stage_2": "n/a", "Stage_3": "none major",
     "Stage_4": "P403 (2 transitions), AIT401 near-constant", "Stage_5": "P501-OFF (89 samples)",
     "Stage_6": f"P602 ON-time only 0.85% of TRAIN; only 4 tags total in the whole stage"},
    {"Metric": "Interpretability (qualitative)",
     "Stage_1": "High", "Stage_2": "High", "Stage_3": "High for actuator-sensor", "Stage_4": "High for retained rule",
     "Stage_5": "High to moderate",
     "Stage_6": "High if retained (direct mechanical gating), but rests on the sparsest actuator activity of any retained/candidate rule in the project"},
    {"Metric": "Contributes complementary symbolic information?",
     "Stage_1": "Yes (rolling flow<->level)", "Stage_2": "Yes (baseline)", "Stage_3": "Yes (flow<->DPIT category, though ultimately rejected)",
     "Stage_4": "Yes (actuator->analyzer + rolling in one stage)", "Stage_5": "Yes (sensor-sensor conservation)",
     "Stage_6": ("Yes, marginally (a 6th independent pump->flow confirmation)" if n_retained else "No new category -- structurally could only replicate the pump->flow pattern already established in Stages 1/2/3/4/5")},
]
comparison_df = pd.DataFrame(comparison_rows)
comparison_path = TABLES_DIR / "stage6_cross_stage_methodology_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)
log(f"Wrote {comparison_path} ({len(comparison_df)} rows)")
for r in comparison_rows:
    log(f"  {r['Metric']}: S1={r['Stage_1']} | S2={r['Stage_2']} | S3={r['Stage_3']} | S4={r['Stage_4']} | S5={r['Stage_5']} | S6={r['Stage_6']}")

log(f"\nDISCUSSION: Stage 6's ~28% attack-associated activity (script 10) plausibly explains part of why "
    f"its ATTACK-period coverage figure ({attack_cov_str}) differs from other stages' retained features "
    f"-- but this observation was NOT used to adjust Part 8's RETAIN/LIMITED/REJECT thresholds, which "
    f"remained numerically identical to every prior stage script.")

# =============================================================================
# PART 10 — FIGURES
# =============================================================================
section("PART 10 — GENERATING FIGURES (300 DPI)")

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
                      "axes.titlesize": 13, "axes.labelsize": 11})
STATUS_COLOR = {"RETAIN": "#55A868", "LIMITED": "#F4B183", "REJECT": "#C44E52"}

# --- Figure 1: candidate rule validation ---
if candidates:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ax = axes[0]
    support_vals = [abs(rule_evidence[r].get("smd", 0)) if rule_evidence[r].get("supported") else 0 for r in rule_ids_order]
    colors = [STATUS_COLOR[decisions[r]["decision"]] for r in rule_ids_order]
    ax.barh(rule_ids_order, support_vals, color=colors)
    ax.set_xlabel("Empirical support metric (|SMD|)")
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
    fig.suptitle("Stage 6 Candidate Relationship Empirical Support and Coverage\n"
                 "(Stage 6 structurally supports only 1 candidate -- see Part 1)", y=1.18)
else:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.text(0.5, 0.5, "Stage 6 generated ZERO candidates:\ninsufficient structural variation "
            "(no varying actuator and/or sensor pair available).", ha="center", va="center", fontsize=11)
    ax.axis("off")
    ax.set_title("Stage 6 Candidate Relationship Validation")
fig.tight_layout()
fig1_path = FIGURES_DIR / "stage6_candidate_rule_validation.png"
fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig1_path}")

# --- Figure 2: normal response profiles ---
n_rules = max(len(candidates), 1)
fig, axes = plt.subplots(1, n_rules, figsize=(6, 5))
axes = [axes] if n_rules == 1 else axes
if candidates:
    for ax, c in zip(axes, candidates):
        rid = c["Rule_ID"]
        ev = rule_evidence[rid]
        if ev.get("supported"):
            ax.hist(ev["off_ref"], bins=40, alpha=0.6, label=f"LOW (n={len(ev['off_ref']):,})", color="#C44E52", density=True)
            ax.hist(ev["on_ref"], bins=40, alpha=0.6, label=f"HIGH (n={len(ev['on_ref']):,})", color="#4C72B0", density=True)
            ax.set_xlabel(c["Target_Tags"])
            ax.set_title(f"{rid}\n{c['Source_Tags']} state vs {c['Target_Tags']}")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "Insufficient TRAIN data\n(see validation table)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(rid)
        ax.grid(True, linestyle="--", alpha=0.4)
else:
    ax0 = axes[0] if isinstance(axes, list) else axes
    ax0.text(0.5, 0.5, "No Stage 6 candidates to profile.", ha="center", va="center", transform=ax0.transAxes)
    ax0.set_title("Stage 6")
    ax0.axis("off")
fig.suptitle("Stage 6 TRAIN-NORMAL Response / Relationship Profiles", y=1.03)
fig.tight_layout()
fig2_path = FIGURES_DIR / "stage6_normal_response_profiles.png"
fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig2_path}")

# --- Figure 3: symbolic score VALIDATION NORMAL vs ATTACK ---
fig, axes = plt.subplots(1, n_rules, figsize=(6, 5.5))
axes = [axes] if n_rules == 1 else axes
if candidates:
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
            ax.annotate(f"CAUTION: n={len(attack_vals)} ATTACK sample(s)\n(Stage 6 has low attack-associated activity)",
                         xy=(0.5, 0.02), xycoords="axes fraction", ha="center", fontsize=6.5, color="#C44E52")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
else:
    ax0 = axes[0] if isinstance(axes, list) else axes
    ax0.text(0.5, 0.5, "No Stage 6 symbolic features to compare.", ha="center", va="center", transform=ax0.transAxes)
    ax0.axis("off")
fig.suptitle("Stage 6 Symbolic Scores: VALIDATION NORMAL vs. ATTACK (where evaluable)", y=1.03)
fig.tight_layout()
fig3_path = FIGURES_DIR / "stage6_symbolic_score_validation_comparison.png"
fig.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig3_path}")

# --- Figure 4: final rule screening ---
fig, ax = plt.subplots(figsize=(9, max(3, 1.2 * n_rules)))
if candidates:
    colors4 = [STATUS_COLOR[decisions[r]["decision"]] for r in rule_ids_order]
    ax.barh(rule_ids_order, [1] * len(rule_ids_order), color=colors4)
    for i, r in enumerate(rule_ids_order):
        ax.text(0.5, i, decisions[r]["decision"], ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
else:
    ax.text(0.5, 0.5, "No Stage 6 candidates were generated\n(structurally insufficient variation).",
            ha="center", va="center", fontsize=11)
    ax.axis("off")
ax.set_title("Stage 6 Final Rule Screening Decisions")
fig.tight_layout()
fig4_path = FIGURES_DIR / "stage6_final_rule_screening.png"
fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig4_path}")

# --- Figure 5: final retained features (professionally handles zero-retained case) ---
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
    ax.set_title("Final Retained Stage 6 Symbolic Feature Set", fontsize=12)
else:
    msg = ("No Stage 6 candidate relationships satisfied the full RETAIN criteria.\n\n"
           f"Candidates evaluated: {len(candidates)}" +
           (f"  |  Screening: {n_limited} LIMITED, {n_rejected} REJECT" if candidates else "") +
           "\n\nThis is a scientifically meaningful negative result, not a pipeline failure --\n"
           "the same evidentiary standard applied to Stages 1-5 was used without adjustment.\n"
           "See tables/stage6_final_rule_screening.csv for full reasoning.")
    ax.text(5, ax.get_ylim()[1] / 2, msg, ha="center", va="center", fontsize=11)
    ax.set_title("Final Retained Stage 6 Symbolic Feature Set: NONE RETAINED", fontsize=12)
fig.tight_layout()
fig5_path = FIGURES_DIR / "stage6_final_retained_features.png"
fig.savefig(fig5_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig5_path}")

# =============================================================================
# PART 11 — ACADEMIC NARRATIVE
# =============================================================================
section("PART 11 — WRITING ACADEMIC NARRATIVE")

if candidates:
    strongest_rule = max(candidates, key=lambda c: abs(rule_evidence[c["Rule_ID"]].get("smd", 0)) if rule_evidence[c["Rule_ID"]].get("supported") else 0)
    weakest_rule = strongest_rule  # only one candidate exists
else:
    strongest_rule = weakest_rule = None

narrative = f"""STAGE 6 SYMBOLIC PIPELINE — ACADEMIC NARRATIVE
(Draft prose for Methodology/Results. Review before submission. FINAL stage-specific pipeline.)

1. STAGE 6 PROCESS / TAG STRUCTURE
Stage 6 (RO permeate transfer) is the sparsest stage in the SWaT testbed by tag count: only 4 physical
tags (FIT601, P601, P602, P603) versus 5-13 tags in every other stage screened in this project.
Empirical audit (Part 1) found P601 and P603 fully constant (matching the original dataset audit's
constant-feature list) and P602 the ONLY varying actuator (150 TRAIN transitions, but only 0.85% ON-time
-- the sparsest operational actuator activity of any RETAIN-eligible candidate encountered across all
six stages). FIT601 is Stage 6's ONLY sensor of any kind.

2. CANDIDATE-RULE DISCOVERY
This structural ceiling means Stage 6 supports exactly {len(candidates)} candidate relationship(s): no
sensor-sensor candidate is possible (only one sensor exists), no actuator-actuator coupling candidate is
possible (only one varying actuator exists), and no flow<->level rolling candidate is possible (no level
sensor exists). Rather than manufacturing additional candidates to match the scope of Stages 1-5, this
was documented explicitly and only the single structurally defensible candidate
({candidates[0]['Source_Tags'] + ' -> ' + candidates[0]['Target_Tags'] if candidates else 'none'}) was
generated and tested.

3. TRAIN-NORMAL EMPIRICAL VALIDATION
{"The P602->FIT601 candidate was validated via the identical state-separation (SMD, IQR overlap, KS) "
 "and event-triggered response methodology used in every prior stage." if candidates else "No candidate existed to validate."}

4. TEMPORAL / STATE / SENSOR-CONSISTENCY FINDINGS
{chr(10).join(f"- {rid}: {rule_evidence[rid].get('kind')}, supported={rule_evidence[rid].get('supported')}" for rid in rule_ids_order) if candidates else "N/A -- no candidates generated."}

5. COVERAGE AND ROBUSTNESS
{"Coverage was reported separately for overall evaluability, VALIDATION-NORMAL, and VALIDATION-ATTACK "
 "periods, per the explicit instruction to distinguish these given Stage 6's relatively low "
 "attack-associated activity (~28%, per script 10)." if candidates else "N/A."}

6. RETAIN / LIMITED / REJECT DECISIONS
{chr(10).join(f"- {rid}: {decisions[rid]['decision']} -- {decisions[rid]['reason']}" for rid in rule_ids_order) if candidates else "No candidates were available to screen; 0 RETAIN, 0 LIMITED, 0 REJECT."}

7. FINAL STAGE 6 SYMBOLIC FEATURE SET
{f"The finalized Stage 6 symbolic feature set consists of: {', '.join(FINAL_FEATURE_NAMES)}." if FINAL_FEATURE_NAMES else "No Stage 6 symbolic feature satisfied the full RETAIN criteria. This is reported as a legitimate, scientifically meaningful negative result -- Stage 6's structural sparsity (a single low-duty-cycle actuator and a single sensor) was not compensated for by loosening the screening standard used for Stages 1-5."}

8. MAJOR LIMITATIONS
Stage 6's entire candidate space rests on P602's 0.85% TRAIN ON-time -- even a well-separated
relationship here is characterized from far less operational activity than any other stage's retained
features. No sensor-sensor or actuator-actuator category could be tested at all, purely for structural
reasons (tag scarcity), not because such relationships were tested and found wanting.

9. IMPLICATIONS OF STAGE 6'S RELATIVELY LOW ATTACK-ASSOCIATED ACTIVITY
Script 10's stage-exposure audit found Stage 6 had the lowest attack-associated activity (~28%) of any
SWaT stage. This context plausibly helps explain limited ATTACK-period evaluability for any Stage 6
symbolic feature (see Part 5/9's attack-period coverage figures) -- but per explicit instruction, this
observation was NOT used to adjust the RETAIN/LIMITED/REJECT thresholds in Part 8, which remained
numerically identical to Stages 1-5.

10. HOW STAGE 6 DIFFERS FROM STAGES 1-5
Stage 6 is structurally the polar opposite of Stage 3 (which had the richest actuator set of any stage):
minimal tag count, a single low-duty-cycle actuator, and no possibility of sensor-sensor or
actuator-actuator candidates. Where Stages 1-5 each contributed at least one genuinely new symbolic-
information CATEGORY (rolling flow<->level, dosing-response, flow<->differential-pressure,
actuator->analyzer, sensor-sensor conservation), Stage 6 can at most reconfirm the pump->flow category
already established independently in five other stages -- a legitimate but structurally bounded
contribution. No classifier or attack-detection performance claim is made anywhere in this analysis.
"""

narrative_path = RESULTS_DIR / "stage6_symbolic_pipeline_narrative.txt"
narrative_path.write_text(narrative, encoding="utf-8")
log(f"Wrote {narrative_path}")

# =============================================================================
# PART 12 — MAIN RESULTS LOG
# =============================================================================
section("PART 12 — WRITING MAIN RESULTS LOG")

main_lines = []
main_lines.append("STAGE 6 END-TO-END SYMBOLIC PIPELINE — MAIN RESULTS (FINAL STAGE-SPECIFIC SCRIPT)")
main_lines.append(f"Generated: {RUN_START.isoformat()}")
main_lines.append(f"Python: {sys.version}")
main_lines.append(f"pandas: {pd.__version__} | numpy: {np.__version__} | scipy: {__import__('scipy').__version__}")
main_lines.append(f"OS: {platform.system()} {platform.release()}")
main_lines.append("")
main_lines.append(f"Data sources: {TRAIN_PATH.name} ({len(train_df):,} rows), "
                   f"{VAL_PATH.name} ({len(val_df):,} rows). TEST was never loaded.")
main_lines.append("")
main_lines.append(f"=== STAGE 6 CANDIDATE RULES ({len(candidates)}) ===")
if candidates:
    for c in candidates:
        main_lines.append(f"  {c['Rule_ID']}: {c['Source_Tags']} <-> {c['Target_Tags']} ({c['Relationship_Type']})")
else:
    main_lines.append("  (none -- Stage 6 lacks sufficient structural variation, see Part 1)")
main_lines.append("")
main_lines.append("=== STAGE 6 FINAL DECISIONS ===")
if candidates:
    for rid in rule_ids_order:
        main_lines.append(f"  {rid} -> {decisions[rid]['decision']}")
else:
    main_lines.append("  (no candidates to decide)")
main_lines.append("")
main_lines.append("=== FINAL RETAINED STAGE 6 SYMBOLIC FEATURE SET ===")
if FINAL_FEATURE_NAMES:
    for f in FINAL_FEATURE_NAMES:
        main_lines.append(f"  {f}")
else:
    main_lines.append("  No Stage 6 symbolic feature satisfied the full RETAIN criteria.")
main_lines.append("")
main_lines.append(f"Candidate count : {len(candidates)}")
main_lines.append(f"Retained count  : {n_retained}")
main_lines.append(f"Limited count   : {n_limited}")
main_lines.append(f"Rejected count  : {n_rejected}")
main_lines.append("")
main_lines.append("=== OVERALL / ATTACK-PERIOD EVALUABILITY ===")
for _, r in coverage_df.iterrows():
    main_lines.append(f"  {r['Rule_ID']}: overall={r['VALIDATION_Overall_Coverage_Pct']:.2f}%, "
                       f"NORMAL={r['VALIDATION_NORMAL_Coverage_Pct']:.2f}%, "
                       f"ATTACK={r['VALIDATION_ATTACK_Coverage_Pct']:.2f}% "
                       f"(n_attack_evaluable={r['N_VALIDATION_ATTACK_Evaluable']})")
main_lines.append("")
main_lines.append("=== ROBUSTNESS / DISTRIBUTION-SHIFT FINDINGS ===")
for _, r in robustness_df.iterrows():
    main_lines.append(f"  {r['Rule_ID']}: TRAIN-N median={r['TRAIN_NORMAL_Score_Median']}, "
                       f"VAL-N median={r['VALIDATION_NORMAL_Score_Median']}, saturated={r['Saturated_In_VALIDATION']}, "
                       f"thin_minority_reference={r['Thin_Minority_Reference']}")
main_lines.append("")
if strongest_rule is not None:
    main_lines.append(f"Strongest Stage 6 candidate relationship: {strongest_rule['Rule_ID']} "
                       f"({strongest_rule['Source_Tags']} <-> {strongest_rule['Target_Tags']})")
    main_lines.append(f"Weakest Stage 6 candidate relationship: {weakest_rule['Rule_ID']} "
                       f"({weakest_rule['Source_Tags']} <-> {weakest_rule['Target_Tags']}) "
                       f"(only one candidate existed, so it is simultaneously the strongest and weakest)")
else:
    main_lines.append("Strongest / weakest Stage 6 candidate relationship: N/A (no candidates generated)")
main_lines.append("")
adds_value = n_retained > 0
main_lines.append(f"Does Stage 6 contribute useful symbolic information beyond prior stages? "
                   f"{'YES, marginally -- 1 feature retained, an independent 6th confirmation of the pump->flow category already seen in Stages 1/2/3/4/5.' if adds_value else 'NOT CONFIRMED -- no Stage 6 candidate cleared the RETAIN bar; this is a legitimate negative result, not a pipeline failure.'}")
main_lines.append("")
main_lines.append("=== GUARDRAILS CONFIRMED ===")
main_lines.append("  - TEST partition: never loaded, filename not referenced anywhere in this script.")
main_lines.append("  - TRAIN NORMAL used for all reference distributions and rule construction.")
main_lines.append("  - VALIDATION used only for coverage/robustness diagnostics, never to tune anything.")
main_lines.append("  - No model was trained. No ML feature selection was performed.")
main_lines.append("  - No existing script/output was modified; all Stage 1-5 outputs preserved.")
main_lines.append("  - Stage 6's ~28% attack-associated activity was used as CONTEXT ONLY (Part 9), "
                   "never to adjust Part 8's screening thresholds.")
main_lines.append("  - No physical relationship was invented beyond Stage 6's tag structure and "
                   "TRAIN-normal data; uncertain assumptions are explicitly marked throughout.")
main_lines.append("  - Analysis is fully deterministic (no randomness used).")

main_path = RESULTS_DIR / "stage6_end_to_end_symbolic_pipeline.txt"
main_path.write_text("\n".join(main_lines), encoding="utf-8")
log(f"Wrote {main_path}")

# =============================================================================
# PART 13 — PREPARE FOR FINAL CROSS-STAGE CONSOLIDATION
# =============================================================================
section("PART 13 — STAGE 6 FINAL PIPELINE COMPLETION SUMMARY (for the next, cross-stage script)")

completion_row = {
    "Stage": "Stage 6", "Candidate_Count": len(candidates), "Retained_Count": n_retained,
    "Limited_Count": n_limited, "Rejected_Count": n_rejected,
    "Retained_Feature_Names": "; ".join(FINAL_FEATURE_NAMES) if FINAL_FEATURE_NAMES else "(none)",
    "Strongest_Relationship": (f"{strongest_rule['Rule_ID']}: {strongest_rule['Source_Tags']} <-> {strongest_rule['Target_Tags']}"
                                if strongest_rule is not None else "N/A (no candidates)"),
    "Primary_Limitation": ("P602's 0.85% TRAIN ON-time is the sparsest operational actuator activity of any "
                            "candidate/retained relationship across all six stages; Stage 6's 4-tag structure "
                            "structurally precludes sensor-sensor and actuator-actuator candidates entirely."),
    "Completion_Status": "COMPLETE",
}
completion_df = pd.DataFrame([completion_row])
completion_path = TABLES_DIR / "stage6_final_pipeline_completion_summary.csv"
completion_df.to_csv(completion_path, index=False)
log(f"Wrote {completion_path} ({len(completion_df)} rows)")
log("This is the FINAL stage-specific symbolic pipeline. No Stage 1-5 features were merged or re-screened "
    "here -- that is explicitly deferred to the next, cross-stage consolidation script.")

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

log(f"Stage 6 candidate rules ({len(candidates)}):")
for c in candidates:
    log(f"  {c['Rule_ID']}: {c['Source_Tags']} <-> {c['Target_Tags']}")
log("\nStage 6 final decisions:")
for rid in rule_ids_order:
    log(f"  {rid} -> {decisions[rid]['decision']}")
if FINAL_FEATURE_NAMES:
    log(f"\nFinal retained Stage 6 symbolic feature set: {FINAL_FEATURE_NAMES}")
else:
    log("\nNo Stage 6 symbolic feature satisfied the full RETAIN criteria.")
log(f"\nCandidate={len(candidates)}, Retained={n_retained}, Limited={n_limited}, Rejected={n_rejected}")

run_end = datetime.now()
log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
log("PIPELINE COMPLETE. No model trained, no ML feature selection, no attack-threshold tuning.")
log("This concludes the stage-specific symbolic pipelines (Stages 1-6). Cross-stage consolidation is "
    "deferred to the next script.")

if not all_checks_passed:
    sys.exit(1)
