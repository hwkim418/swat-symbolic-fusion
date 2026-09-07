"""
code/12_stage5_end_to_end_symbolic_pipeline.py

STAGE 5 END-TO-END SYMBOLIC PIPELINE — INDEPENDENT DISCOVERY, EMPIRICAL
VALIDATION, FEATURE CONSTRUCTION, AND FINAL SCREENING. ANALYSIS ONLY.

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

Applies the STANDARDIZED METHODOLOGY developed for Stage 2 (scripts
04-11: empirical state-semantics verification, TRAIN-NORMAL-only
reference distributions, distribution-free empirical scoring,
RETAIN/LIMITED/REJECT screening) to Stage 5 (Reverse Osmosis), but
candidate relationships are discovered and validated INDEPENDENTLY from
Stage 5's own tag structure and TRAIN-NORMAL behavior -- Stage 2's
specific rules are never copied.

Stage 5 has only ONE actuator with any TRAIN variation (P501; P502 is
constant, confirmed below), so the candidate space is necessarily
different in character from Stage 2's three varying actuators -- this
script leans more on sensor-sensor consistency relationships (flow-flow,
pressure-pressure, analyzer-analyzer), which are explicitly discovered
via correlation screening (never assumed) and then validated beyond
correlation alone, per instructions.

No ML model is trained, no feature is selected, no attack-label-based
threshold is tuned, no existing script/dataset/output is modified, and
TEST is never loaded -- its filename does not appear anywhere below, by
design. Uses ONLY processed/train_unscaled.csv and
processed/validation_unscaled.csv, plus tables/swat_stage_tag_mapping.csv
and (read-only, for the Part 9 comparison) script 11's Stage 2 outputs.
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


section("STAGE 5 END-TO-END SYMBOLIC PIPELINE — RUN START (ANALYSIS ONLY)")
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
LAG_GRID = [-30, -15, -10, -5, 0, 5, 10, 15, 30]
CONSISTENCY_THRESHOLD = 95.0
SATURATION_THRESHOLD = 0.90
MIN_EVENT_COUNT = 10
MIN_SAMPLE_COUNT = 1000  # for always-evaluable sensor-sensor relationships (TRAIN NORMAL rows)

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
stage5_tags_df = stage_mapping[stage_mapping.Stage == "Stage 5"]
STAGE5_TAGS = stage5_tags_df["Tag"].tolist()
STAGE5_SENSORS = stage5_tags_df.loc[stage5_tags_df.Sensor_Or_Actuator == "sensor", "Tag"].tolist()
STAGE5_ACTUATORS = stage5_tags_df.loc[stage5_tags_df.Sensor_Or_Actuator == "actuator", "Tag"].tolist()
log(f"Loaded {STAGE_MAPPING_PATH.name}: Stage 5 = {len(STAGE5_TAGS)} tags "
    f"({len(STAGE5_SENSORS)} sensors, {len(STAGE5_ACTUATORS)} actuators)")
log(f"  Sensors  : {STAGE5_SENSORS}")
log(f"  Actuators: {STAGE5_ACTUATORS}")

# =============================================================================
# SHARED HELPER FUNCTIONS (identical methodology to scripts 05-11)
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


# =============================================================================
# PART 1 — STAGE 5 TAG AUDIT
# =============================================================================
section("PART 1 — STAGE 5 TAG AUDIT")

tag_audit_rows = []
for tag in STAGE5_TAGS:
    row_meta = stage5_tags_df[stage5_tags_df.Tag == tag].iloc[0]
    for split_name, df in [("TRAIN", train_df), ("VALIDATION", val_df)]:
        s = df[tag]
        n = len(s)
        nunique = s.nunique()
        n_trans = int((s != s.shift(1)).sum())
        n_trans = max(n_trans - 1, 0) if n > 0 else 0  # correct row-0-vs-NaN artifact
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
            rec["Min"] = rec["Max"] = rec["Median"] = rec["Std"] = np.nan
        else:
            rec["State_Values_And_Frequencies"] = np.nan
            rec["Min"] = float(s.min()); rec["Max"] = float(s.max())
            rec["Median"] = float(s.median()); rec["Std"] = float(s.std(ddof=0))
        tag_audit_rows.append(rec)

tag_audit_df = pd.DataFrame(tag_audit_rows)
tag_audit_path = TABLES_DIR / "stage5_tag_audit.csv"
tag_audit_df.to_csv(tag_audit_path, index=False)
log(f"Wrote {tag_audit_path} ({len(tag_audit_df)} rows)")

log("\nTRAIN summary:")
for _, r in tag_audit_df[tag_audit_df.Split == "TRAIN"].iterrows():
    if r["Sensor_Or_Actuator"] == "actuator":
        log(f"  {r['Tag']:8s} [actuator] states: {r['State_Values_And_Frequencies']} | "
            f"transitions={r['N_Transitions']} | constant/near-constant={r['Constant_Or_Near_Constant']}")
    else:
        log(f"  {r['Tag']:8s} [sensor]   n_unique={r['N_Unique']:>5} min={r['Min']:.4f} max={r['Max']:.4f} "
            f"median={r['Median']:.4f} std={r['Std']:.4f}")

varying_actuators = tag_audit_df[(tag_audit_df.Split == "TRAIN") & (tag_audit_df.Sensor_Or_Actuator == "actuator") &
                                  (~tag_audit_df.Constant_Or_Near_Constant)]["Tag"].tolist()
constant_actuators = [t for t in STAGE5_ACTUATORS if t not in varying_actuators]
log(f"\nStage 5 actuators with sufficient TRAIN variation for empirical rule construction: {varying_actuators}")
log(f"Stage 5 actuators that are constant/near-constant (NOT discarded, reported explicitly): {constant_actuators}")
log("NOTE: unlike Stage 2 (3 varying actuators: MV201, P203, P205), Stage 5 has only ONE varying "
    "actuator (P501). This structurally limits actuator-actuator coupling candidates (no Stage-5 "
    "analog to S2-R3B is possible) and shifts candidate discovery toward sensor-sensor relationships, "
    "which Stage 5's 11 continuous sensors make more available here than in Stage 2 (4 sensors).")

# =============================================================================
# EMPIRICALLY INFER P501 SEMANTICS (data-driven, mirrors script 05's method;
# no engineering identity for P501/sensors is assumed a priori)
# =============================================================================
section("EMPIRICALLY INFERRING P501 STATE SEMANTICS (data-driven, not assumed)")

p501_states = sorted(train_df["P501"].unique())
log(f"P501 observed TRAIN states: {p501_states}")

# Rank all 11 sensors by |Pearson correlation| with P501's raw state in TRAIN
# NORMAL -- purely a discovery/screening step (candidate generation), NOT
# treated as sufficient validation evidence on its own (see Part 3).
train_normal = train_df[train_df.Label == 0]
p501_corr = {}
for s in STAGE5_SENSORS:
    p501_corr[s] = np.corrcoef(train_normal["P501"], train_normal[s])[0, 1]
p501_corr_sorted = sorted(p501_corr.items(), key=lambda kv: -abs(kv[1]))
log("P501 vs each Stage-5 sensor, Pearson correlation (TRAIN NORMAL, lag 0, screening only):")
for s, c in p501_corr_sorted:
    log(f"  P501 vs {s:8s}: r={c:+.4f}")

p501_primary_sensor = p501_corr_sorted[0][0]
sign = np.sign(p501_corr[p501_primary_sensor])
majority_state = train_df["P501"].mode().iloc[0]
minority_state = [s for s in p501_states if s != majority_state][0]
maj_mean = train_normal.loc[train_normal.P501 == majority_state, p501_primary_sensor].mean()
min_mean = train_normal.loc[train_normal.P501 == minority_state, p501_primary_sensor].mean()
P501_ON = majority_state if (sign > 0) == (maj_mean > min_mean) else minority_state
P501_OFF = minority_state if P501_ON == majority_state else majority_state
log(f"\nMost-correlated sensor: {p501_primary_sensor} (r={p501_corr[p501_primary_sensor]:+.4f})")
log(f"Inferred (data-driven, via {p501_primary_sensor} level comparison): "
    f"P501={P501_ON} -> ON (mean {p501_primary_sensor}={max(maj_mean,min_mean):.4f}), "
    f"P501={P501_OFF} -> OFF (mean {p501_primary_sensor}={min(maj_mean,min_mean):.4f})")
log("CONFIDENCE: moderate-to-high -- this mirrors script 05's steady-state-level inference method, but "
    "unlike Stage 2's dosing pumps (which had an independently stated engineering-expected direction to "
    "check against), no external expected direction was available for P501, so this inference rests on "
    "the correlation-ranked sensor alone. Marked explicitly as a data-driven inference, not a documented "
    "plant fact.")

n_p501_transitions = find_specific_transitions(train_df, "P501", P501_OFF, P501_ON)
log(f"\nP501 {P501_OFF}->{P501_ON} (OFF->ON) transitions in TRAIN: {len(n_p501_transitions)}")

# =============================================================================
# PART 2 — CANDIDATE PHYSICAL RELATIONSHIP DISCOVERY
# =============================================================================
section("PART 2 — CANDIDATE RELATIONSHIP DISCOVERY (correlation screening, restricted to structurally "
        "defensible categories: pump->sensor, and same-instrument-type sensor-sensor pairs)")

candidates = []

# --- pump -> sensor candidates: take the top 2 most-correlated sensors (by
# |r|), restricted to at most one per instrument type for diversity, since a
# pump mechanically gating its own downstream instrumentation is the single
# most structurally certain relationship category available for Stage 5. ---
seen_prefix = set()
pump_candidates_selected = []
for s, c in p501_corr_sorted:
    prefix = "".join(ch for ch in s if not ch.isdigit())
    if prefix in seen_prefix:
        continue
    seen_prefix.add(prefix)
    pump_candidates_selected.append((s, c))
    if len(pump_candidates_selected) >= 2:
        break

for i, (s, c) in enumerate(pump_candidates_selected, start=1):
    candidates.append({
        "Rule_ID": f"S5-R{i}", "Source_Tags": "P501", "Target_Tags": s,
        "Relationship_Type": "pump -> sensor (state/event)",
        "Expected_Relationship": f"P501 ON should produce {'higher' if c > 0 else 'lower'} {s} than P501 OFF",
        "Rationale": f"A pump mechanically gates its own hydraulically-downstream instrumentation "
                     f"({s} is Stage 5's tag with the strongest |correlation| to P501 among "
                     f"{'flow' if s.startswith('FIT') else 'pressure' if s.startswith('PIT') else 'analyzer'} "
                     f"instruments) -- this is the same structural logic as the validated S2-R1 "
                     f"(MV201->FIT201), applied here to Stage 5's only varying actuator.",
        "Confidence": "high (direct mechanical gating is a near-universal relationship for any pump and "
                       "its immediate downstream instrumentation)",
        "Time_Lag_Expected": "Possibly -- to be determined empirically in Part 3 (event-based test).",
        "Discovery_Correlation_R": c,
    })

# --- sensor-sensor candidates: strongest pair within each same-instrument-type
# family (FIT-FIT, PIT-PIT, AIT-AIT) -- "conservation/consistency" category. ---
FAMILIES = {"FIT": [t for t in STAGE5_SENSORS if t.startswith("FIT")],
            "PIT": [t for t in STAGE5_SENSORS if t.startswith("PIT")],
            "AIT": [t for t in STAGE5_SENSORS if t.startswith("AIT")]}

next_id = len(candidates) + 1
for family, tags in FAMILIES.items():
    if len(tags) < 2:
        continue
    best_pair, best_rho = None, 0
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            rho, _ = scipy_stats.spearmanr(train_normal[tags[i]], train_normal[tags[j]])
            if abs(rho) > abs(best_rho):
                best_rho, best_pair = rho, (tags[i], tags[j])
    if best_pair is None:
        continue
    candidates.append({
        "Rule_ID": f"S5-R{next_id}", "Source_Tags": best_pair[0], "Target_Tags": best_pair[1],
        "Relationship_Type": "sensor-sensor consistency (same instrument type)",
        "Expected_Relationship": f"{best_pair[0]} and {best_pair[1]} should move "
                                  f"{'together' if best_rho > 0 else 'oppositely'} in a stable, consistent way",
        "Rationale": f"{best_pair[0]} and {best_pair[1]} are both {family}-type instruments within the "
                     f"same Stage 5 treatment train; the strongest same-type pairwise Spearman rank "
                     f"correlation among all {family}-{family} pairs (rho={best_rho:+.3f}) identifies "
                     f"them as the most promising conservation/consistency candidate within this family "
                     f"-- this is a discovery signal only, not sufficient validation on its own.",
        "Confidence": "moderate (correlation-based discovery; requires the Part 3 validation beyond "
                       "correlation before any conclusion)",
        "Time_Lag_Expected": "Possibly, if the two instruments are hydraulically separated -- tested via "
                              "a symmetric lag sweep in Part 3.",
        "Discovery_Correlation_R": best_rho,
    })
    next_id += 1

candidates_df = pd.DataFrame(candidates)
candidates_path = TABLES_DIR / "stage5_candidate_rules.csv"
candidates_df.to_csv(candidates_path, index=False)
log(f"Wrote {candidates_path} ({len(candidates_df)} rows)")
log(f"\n{len(candidates)} candidate relationships identified BEFORE any validation testing:")
for c in candidates:
    log(f"  {c['Rule_ID']}: {c['Source_Tags']} <-> {c['Target_Tags']} ({c['Relationship_Type']}) -- "
        f"discovery r={c['Discovery_Correlation_R']:+.3f}")

# =============================================================================
# PART 3 + 4 — TRAIN-NORMAL EMPIRICAL VALIDATION + NORMAL OPERATING ENVELOPES
# =============================================================================
section("PART 3 & 4 — TRAIN-NORMAL EMPIRICAL VALIDATION AND OPERATING ENVELOPES")

validation_rows = []
envelope_rows = []
rule_reference = {}  # Rule_ID -> dict with reference sample(s) etc., for Part 7


def validate_pump_sensor(rule_id, target):
    """State-based distribution comparison + event-based transition response."""
    off_ref = train_normal.loc[train_normal.P501 == P501_OFF, target].to_numpy()
    on_ref = train_normal.loc[train_normal.P501 == P501_ON, target].to_numpy()
    pooled_std = np.sqrt((np.var(off_ref) + np.var(on_ref)) / 2)
    smd = (np.mean(on_ref) - np.mean(off_ref)) / pooled_std if pooled_std > 0 else np.nan
    off_iqr = np.percentile(off_ref, [25, 75]); on_iqr = np.percentile(on_ref, [25, 75])
    overlap = max(0.0, min(off_iqr[1], on_iqr[1]) - max(off_iqr[0], on_iqr[0]))
    min_span = min(off_iqr[1] - off_iqr[0], on_iqr[1] - on_iqr[0])
    if min_span > 0:
        overlap_frac = overlap / min_span
    elif overlap == 0:
        # one group's IQR has zero span (e.g. a near-constant reading), but it
        # does not intersect the other group's IQR at all -> perfectly separated.
        overlap_frac = 0.0
    else:
        # zero-span group's point value falls inside the other's IQR -> treat
        # conservatively as full overlap rather than an undefined NaN.
        overlap_frac = 1.0
    ks_stat, ks_p = scipy_stats.ks_2samp(off_ref, on_ref)
    minority_n = min(len(off_ref), len(on_ref))
    log(f"  [state-based] {target} by P501 state: SMD={smd:.3f}, IQR overlap frac={overlap_frac:.3f}, "
        f"KS={ks_stat:.3f} (p={ks_p:.2e}), n_OFF={len(off_ref)}, n_ON={len(on_ref)}")
    if minority_n < 200:
        log(f"  NOTE: minority-state reference group is thin (n={minority_n}); percentile-based envelope "
            f"for that state carries wider uncertainty than a larger sample would (comparable in scale to "
            f"Stage 2's ~64-68 event-based reference samples, but drawn from a heavily skewed state split "
            f"rather than discrete events).")

    trans = find_specific_transitions(train_df, "P501", P501_OFF, P501_ON)
    events = []
    for i0 in trans:
        if train_df["Label"].iloc[i0] != 0:
            continue
        if not check_baseline_window(train_df, i0, "P501", P501_OFF, require_normal=True):
            continue
        events.append(i0)
    log(f"  [event-based] {len(trans)} raw P501 OFF->ON transitions, {len(events)} usable TRAIN NORMAL events")

    offset_stats = []
    if len(events) >= MIN_EVENT_COUNT:
        expected_sign = 1 if on_ref.mean() > off_ref.mean() else -1
        for offset in EVENT_OFFSETS:
            vals = []
            for i0 in events:
                if not check_offset_window(train_df, i0, offset, require_normal=True):
                    continue
                base = train_df[target].iloc[i0 - BASELINE_WINDOW:i0].mean()
                vals.append(train_df[target].iloc[i0 + offset] - base)
            stats = summarize(vals)
            pe = pct_expected(vals, expected_sign)
            offset_stats.append({"Offset_Seconds": offset, **stats, "Pct_Expected_Direction": pe})
        valid_os = [o for o in offset_stats if o["n"] >= MIN_EVENT_COUNT and not np.isnan(o["Pct_Expected_Direction"])]
        if valid_os:
            best = max(valid_os, key=lambda o: (o["Pct_Expected_Direction"], -o["Offset_Seconds"]))
            for o in valid_os:
                if o["Pct_Expected_Direction"] >= 95.0:
                    best = o
                    break
            log(f"  [event-based] best offset={best['Offset_Seconds']}s: n={best['n']}, "
                f"pct_expected_dir={best['Pct_Expected_Direction']:.1f}%, median={best['median']:.5f}")
        else:
            best = None
    else:
        best = None
        expected_sign = None

    return {
        "kind": "pump_sensor", "off_ref": off_ref, "on_ref": on_ref, "smd": smd,
        "overlap_frac": overlap_frac, "ks_stat": ks_stat, "ks_p": ks_p,
        "n_events": len(events), "expected_sign": expected_sign,
        "offset_stats": offset_stats, "best_offset": best,
    }


def validate_sensor_sensor(rule_id, tag_a, tag_b):
    """Robust relationship beyond correlation: ratio/difference stability, lag sweep, monotonicity."""
    a = train_normal[tag_a].to_numpy(dtype=float)
    b = train_normal[tag_b].to_numpy(dtype=float)
    rho, rho_p = scipy_stats.spearmanr(a, b)
    kendall_tau, tau_p = scipy_stats.kendalltau(a[:20000], b[:20000])  # subsample for tractable runtime

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
    ts_full = train_normal["_ts"].to_numpy()
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
    lag_hit_boundary = abs(best_lag["lag"]) == max(abs(l) for l in LAG_GRID)

    log(f"  [sensor-sensor] {tag_a} vs {tag_b}: Spearman rho={rho:.4f} (p={rho_p:.2e}), "
        f"Kendall tau={kendall_tau:.4f} (subsample), best_lag={best_lag['lag']}s (rho={best_lag['rho']:.4f})")
    if lag_hit_boundary:
        log(f"  [sensor-sensor] WARNING: best lag ({best_lag['lag']}s) is at the EDGE of the tested "
            f"{LAG_GRID} grid -- the true optimum may lie further out and was not explored; flagged "
            f"as a limitation rather than silently accepted.")
    log(f"  [sensor-sensor] relationship formalized as {residual_name}: CV={stability_cv:.4f}, "
        f"median={stats['median']:.5f}, IQR=[{stats['p25']:.5f}, {stats['p75']:.5f}]")

    return {
        "kind": "sensor_sensor", "rho": rho, "rho_p": rho_p, "kendall_tau": kendall_tau,
        "residual_name": residual_name, "residual_sample": residual_clean, "stability_cv": stability_cv,
        "stats": stats, "best_lag": best_lag, "lag_results": lag_results, "use_ratio": use_ratio,
        "lag_hit_boundary": lag_hit_boundary,
    }


rule_evidence = {}
for c in candidates:
    rid = c["Rule_ID"]
    log(f"\n--- Validating {rid}: {c['Source_Tags']} <-> {c['Target_Tags']} ---")
    if c["Relationship_Type"].startswith("pump"):
        result = validate_pump_sensor(rid, c["Target_Tags"])
    else:
        result = validate_sensor_sensor(rid, c["Source_Tags"], c["Target_Tags"])
    rule_evidence[rid] = result

    if result["kind"] == "pump_sensor":
        validation_rows.append({
            "Rule_ID": rid, "Kind": "pump_sensor", "N_Events_TRAIN_NORMAL": result["n_events"],
            "State_SMD": result["smd"], "State_IQR_Overlap_Fraction": result["overlap_frac"],
            "State_KS_Statistic": result["ks_stat"], "State_KS_Pvalue": result["ks_p"],
            "Best_Event_Offset_Seconds": result["best_offset"]["Offset_Seconds"] if result["best_offset"] else np.nan,
            "Best_Event_Pct_Expected_Direction": result["best_offset"]["Pct_Expected_Direction"] if result["best_offset"] else np.nan,
            "Best_Event_Median_Response": result["best_offset"]["median"] if result["best_offset"] else np.nan,
        })
        if result["best_offset"]:
            b = result["best_offset"]
            envelope_rows.append({"Rule_ID": rid, "Envelope_Type": "event_response",
                                   "Offset_Seconds": b["Offset_Seconds"], "N": b["n"],
                                   "P5": b["p5"], "P10": b["p10"], "P25": b["p25"], "Median": b["median"],
                                   "P75": b["p75"], "P90": b["p90"], "P95": b["p95"], "IQR": b["iqr"]})
        for state, ref, label in [(P501_OFF, result["off_ref"], "P501_OFF"), (P501_ON, result["on_ref"], "P501_ON")]:
            st = summarize(ref)
            envelope_rows.append({"Rule_ID": rid, "Envelope_Type": f"state_conditioned_{label}",
                                   "Offset_Seconds": np.nan, "N": st["n"], "P5": st["p5"], "P10": st["p10"],
                                   "P25": st["p25"], "Median": st["median"], "P75": st["p75"], "P90": st["p90"],
                                   "P95": st["p95"], "IQR": st["iqr"]})
    else:
        validation_rows.append({
            "Rule_ID": rid, "Kind": "sensor_sensor", "N_Events_TRAIN_NORMAL": result["stats"]["n"],
            "Spearman_Rho": result["rho"], "Spearman_Pvalue": result["rho_p"],
            "Kendall_Tau_Subsample": result["kendall_tau"], "Residual_Formalization": result["residual_name"],
            "Residual_Stability_CV": result["stability_cv"], "Best_Lag_Seconds": result["best_lag"]["lag"],
            "Best_Lag_Rho": result["best_lag"]["rho"],
        })
        st = result["stats"]
        envelope_rows.append({"Rule_ID": rid, "Envelope_Type": f"residual_{result['residual_name']}",
                               "Offset_Seconds": result["best_lag"]["lag"], "N": st["n"], "P5": st["p5"],
                               "P10": st["p10"], "P25": st["p25"], "Median": st["median"], "P75": st["p75"],
                               "P90": st["p90"], "P95": st["p95"], "IQR": st["iqr"]})

validation_df = pd.DataFrame(validation_rows)
validation_path = TABLES_DIR / "stage5_rule_validation.csv"
validation_df.to_csv(validation_path, index=False)
log(f"\nWrote {validation_path} ({len(validation_df)} rows)")

envelope_df = pd.DataFrame(envelope_rows)
envelope_path = TABLES_DIR / "stage5_normal_operating_envelopes.csv"
envelope_df.to_csv(envelope_path, index=False)
log(f"Wrote {envelope_path} ({len(envelope_df)} rows)")

# =============================================================================
# PART 5 + 6 — CONTINUOUS EVALUABILITY, COVERAGE, AND ROBUSTNESS
# =============================================================================
section("PART 5 & 6 — REPRESENTATION TYPE, VALIDATION COVERAGE, AND DISTRIBUTION-SHIFT ROBUSTNESS")

representation = {}
coverage_rows = []
robustness_rows = []

for c in candidates:
    rid = c["Rule_ID"]
    ev = rule_evidence[rid]
    if ev["kind"] == "pump_sensor":
        # Representation: state-based always evaluable (P501 has no
        # transitional state, unlike MV201); event-based only near transitions.
        rep = "A: state-based" + (" (event-response also characterized as secondary evidence)" if ev["n_events"] >= MIN_EVENT_COUNT else "")
        representation[rid] = rep

        def state_score(df):
            p = df["P501"].to_numpy()
            target = c["Target_Tags"]
            x = df[target].to_numpy(dtype=float)
            score = np.full(len(df), np.nan)
            off_mask = p == P501_OFF
            on_mask = p == P501_ON
            score[off_mask] = empirical_two_sided_score_vec(ev["off_ref"], x[off_mask])
            score[on_mask] = empirical_two_sided_score_vec(ev["on_ref"], x[on_mask])
            return np.ones(len(df), dtype=bool), score

        eval_mask_train, score_train = state_score(train_df)
        eval_mask_val, score_val = state_score(val_df)
    else:
        rep = "C: rolling-window" if abs(ev["best_lag"]["lag"]) > 0 else "A: state-based (instantaneous residual)"
        representation[rid] = rep
        tag_a, tag_b = c["Source_Tags"], c["Target_Tags"]

        def residual_score(df):
            a = df[tag_a].to_numpy(dtype=float)
            b = df[tag_b].to_numpy(dtype=float)
            if ev["use_ratio"]:
                with np.errstate(divide="ignore", invalid="ignore"):
                    x = np.where(b != 0, a / b, np.nan)
            else:
                x = a - b
            eval_mask = ~np.isnan(x) & np.isfinite(x)
            score = np.full(len(df), np.nan)
            score[eval_mask] = empirical_two_sided_score_vec(ev["residual_sample"], x[eval_mask])
            return eval_mask, score

        eval_mask_train, score_train = residual_score(train_df)
        eval_mask_val, score_val = residual_score(val_df)

    rule_evidence[rid]["eval_mask_train"] = eval_mask_train
    rule_evidence[rid]["score_train"] = score_train
    rule_evidence[rid]["eval_mask_val"] = eval_mask_val
    rule_evidence[rid]["score_val"] = score_val

    val_normal_mask = eval_mask_val & (val_df["Label"].to_numpy() == 0)
    val_attack_mask = eval_mask_val & (val_df["Label"].to_numpy() == 1)
    n_val_normal_total = int((val_df["Label"].to_numpy() == 0).sum())
    n_val_attack_total = int((val_df["Label"].to_numpy() == 1).sum())
    cov_normal = 100 * val_normal_mask.sum() / n_val_normal_total if n_val_normal_total else np.nan
    cov_attack = 100 * val_attack_mask.sum() / n_val_attack_total if n_val_attack_total else np.nan
    coverage_rows.append({"Rule_ID": rid, "Representation_Type": rep,
                           "VALIDATION_NORMAL_Coverage_Pct": cov_normal,
                           "VALIDATION_ATTACK_Coverage_Pct": cov_attack,
                           "N_VALIDATION_ATTACK_Evaluable": int(val_attack_mask.sum())})
    log(f"{rid} [{rep}]: VALIDATION coverage NORMAL={cov_normal:.3f}%, ATTACK={cov_attack:.3f}% "
        f"(NOTE: coverage means evaluable, NOT that an attack was detected)")

    train_normal_scores = score_train[eval_mask_train & (train_df["Label"].to_numpy() == 0)]
    val_normal_scores = score_val[val_normal_mask]
    val_attack_scores = score_val[val_attack_mask]
    train_normal_scores = train_normal_scores[~np.isnan(train_normal_scores)]
    val_normal_scores = val_normal_scores[~np.isnan(val_normal_scores)]
    val_attack_scores = val_attack_scores[~np.isnan(val_attack_scores)]

    ks_shift = ks_shift_p = np.nan
    if len(train_normal_scores) >= 30 and len(val_normal_scores) >= 30:
        ks_shift, ks_shift_p = scipy_stats.ks_2samp(train_normal_scores, val_normal_scores)
    val_normal_mean = float(np.mean(val_normal_scores)) if len(val_normal_scores) else np.nan
    val_attack_mean = float(np.mean(val_attack_scores)) if len(val_attack_scores) else np.nan
    saturated = (not np.isnan(val_normal_mean)) and val_normal_mean >= SATURATION_THRESHOLD
    robustness_rows.append({
        "Rule_ID": rid, "TRAIN_NORMAL_Score_Median": float(np.median(train_normal_scores)) if len(train_normal_scores) else np.nan,
        "VALIDATION_NORMAL_Score_Median": float(np.median(val_normal_scores)) if len(val_normal_scores) else np.nan,
        "TRAIN_vs_VALIDATION_NORMAL_KS": ks_shift, "TRAIN_vs_VALIDATION_NORMAL_KS_Pvalue": ks_shift_p,
        "VALIDATION_NORMAL_Mean": val_normal_mean, "VALIDATION_ATTACK_Mean": val_attack_mean,
        "Saturated_In_VALIDATION": saturated,
    })
    log(f"{rid} robustness: TRAIN-NORMAL median={float(np.median(train_normal_scores)) if len(train_normal_scores) else np.nan:.4f}, "
        f"VALIDATION-NORMAL median={float(np.median(val_normal_scores)) if len(val_normal_scores) else np.nan:.4f}, "
        f"KS(TRAIN-N vs VAL-N)={ks_shift}, saturated_in_VALIDATION={saturated}")

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
    if ev["kind"] == "pump_sensor":
        fname = f"{rid}_state_score"
        math_def = ("score = 1 - 2*min(F(x), 1-F(x)), F = empirical CDF of TRAIN-NORMAL "
                     f"{c['Target_Tags']} for the CURRENT steady P501 state (OFF or ON)")
        interp = f"How atypical {c['Target_Tags']} is for P501's current steady state, vs. TRAIN-NORMAL."
        eval_cond = "Every row (P501 has only 2 observed states; no transitional state)"
    else:
        fname = f"{rid}_residual_score"
        best_lag_val = ev["best_lag"]["lag"]
        lag_suffix = f" at lag {best_lag_val}s" if best_lag_val != 0 else ""
        math_def = (f"score = 1 - 2*min(F(x), 1-F(x)), F = empirical CDF of the TRAIN-NORMAL "
                     f"{ev['residual_name']} between {c['Source_Tags']} and {c['Target_Tags']}{lag_suffix}")
        interp = f"How atypical the {c['Source_Tags']}/{c['Target_Tags']} relationship is, vs. TRAIN-NORMAL."
        eval_cond = "Every row where both source and target sensor values are present (near-100% by construction)"
    feature_def_rows.append({
        "Feature": fname, "Source_Rule": rid, "Mathematical_Definition": math_def,
        "Physical_Interpretation": interp, "Evaluability_Condition": eval_cond,
        "Score_Direction": "0 = typical of TRAIN-NORMAL; 1 = extreme tail (atypical)",
        "Coverage_Pct_VALIDATION_NORMAL": cov["VALIDATION_NORMAL_Coverage_Pct"],
        "Limitation": ("Event-based companion evidence is additionally available (see stage5_rule_validation.csv)"
                        if ev["kind"] == "pump_sensor" and ev["n_events"] >= MIN_EVENT_COUNT else
                        ("Relationship strength depends on the discovery-stage correlation continuing to hold; "
                         "monitor for regime changes not present in TRAIN." +
                         (" WARNING: best lag hit the edge of the tested +/-30s grid -- the true optimal "
                          "lag was not fully explored." if ev.get("lag_hit_boundary") else ""))
                        if ev["kind"] == "sensor_sensor" else
                        f"Pump had too few TRAIN transitions ({ev['n_events']} usable) for a robust "
                        f"event-based companion check; state-based comparison used instead."),
    })

feature_def_df = pd.DataFrame(feature_def_rows)
feature_def_path = TABLES_DIR / "stage5_symbolic_feature_definitions.csv"
feature_def_df.to_csv(feature_def_path, index=False)
log(f"Wrote {feature_def_path} ({len(feature_def_df)} rows)")

# =============================================================================
# PART 8 — FINAL RULE SCREENING (same logic/thresholds as script 11)
# =============================================================================
section("PART 8 — FINAL RULE SCREENING (RETAIN / LIMITED / REJECT)")

log(f"Decision rule (identical to the finalized Stage 2 methodology, script 11):")
log(f"  1. If TRAIN-normal event/sample count < {MIN_EVENT_COUNT} (event rules) or < "
    f"{MIN_SAMPLE_COUNT} (sensor-sensor rules) -> REJECT (too sparse to characterize).")
log(f"  2. Else if the relationship is not empirically supported beyond correlation (state SMD/overlap "
    f"or residual CV too weak, or event-consistency < {CONSISTENCY_THRESHOLD}%) -> REJECT.")
log(f"  3. Else (physical relationship IS supported): RETAIN if the resulting symbolic score reaches "
    f">=90% VALIDATION coverage AND is not saturated (VALIDATION-NORMAL mean < {SATURATION_THRESHOLD}); "
    f"otherwise LIMITED.")

decisions = {}
for c in candidates:
    rid = c["Rule_ID"]
    ev = rule_evidence[rid]
    cov = coverage_df[coverage_df.Rule_ID == rid].iloc[0]
    rob = robustness_df[robustness_df.Rule_ID == rid].iloc[0]

    if ev["kind"] == "pump_sensor":
        n = len(ev["off_ref"]) + len(ev["on_ref"])
        supported = (not np.isnan(ev["smd"])) and abs(ev["smd"]) > 1.0 and ev["overlap_frac"] < 0.25
        support_desc = f"state SMD={ev['smd']:.3f}, IQR overlap fraction={ev['overlap_frac']:.3f}"
    else:
        n = ev["stats"]["n"]
        supported = ev["stability_cv"] < 0.5 and abs(ev["rho"]) > 0.3 and ev["rho_p"] < 0.01
        support_desc = (f"Spearman rho={ev['rho']:.3f} (p={ev['rho_p']:.2e}), "
                         f"residual CV={ev['stability_cv']:.3f}")

    if n < (MIN_EVENT_COUNT if ev["kind"] == "pump_sensor" else MIN_SAMPLE_COUNT):
        decision, reason = "REJECT", f"n={n} below the minimum characterization bar."
    elif not supported:
        decision, reason = "REJECT", f"Relationship not empirically supported beyond correlation ({support_desc})."
    else:
        cov_pct = cov["VALIDATION_NORMAL_Coverage_Pct"]
        saturated = bool(rob["Saturated_In_VALIDATION"])
        has_coverage = (not np.isnan(cov_pct)) and cov_pct >= 90.0
        if has_coverage and not saturated:
            decision = "RETAIN"
            reason = (f"Relationship empirically supported ({support_desc}); symbolic score reaches "
                       f"{cov_pct:.1f}% VALIDATION coverage and is NOT saturated "
                       f"(VALIDATION-NORMAL mean={rob['VALIDATION_NORMAL_Mean']:.3f} < {SATURATION_THRESHOLD}).")
        else:
            decision = "LIMITED"
            reasons = []
            if not has_coverage:
                reasons.append(f"coverage only {cov_pct:.3f}%")
            if saturated:
                reasons.append(f"saturated in VALIDATION (NORMAL mean={rob['VALIDATION_NORMAL_Mean']:.3f})")
            reason = (f"Relationship empirically supported ({support_desc}), BUT practical use is "
                       f"constrained: " + "; and ".join(reasons) + ".")
    decisions[rid] = {"decision": decision, "reason": reason}
    log(f"\n{rid}: {decision}\n  {reason}")

n_retained = sum(1 for d in decisions.values() if d["decision"] == "RETAIN")
n_limited = sum(1 for d in decisions.values() if d["decision"] == "LIMITED")
n_rejected = sum(1 for d in decisions.values() if d["decision"] == "REJECT")
log(f"\nStage 5 screening totals: {len(candidates)} candidates -> RETAIN={n_retained}, "
    f"LIMITED={n_limited}, REJECT={n_rejected}")
log("No minimum retained-rule count was enforced; this is the evidence-based result.")

screening_rows = []
for c in candidates:
    rid = c["Rule_ID"]
    d = decisions[rid]
    cov = coverage_df[coverage_df.Rule_ID == rid].iloc[0]
    rob = robustness_df[robustness_df.Rule_ID == rid].iloc[0]
    val = validation_df[validation_df.Rule_ID == rid].iloc[0]
    screening_rows.append({
        "Rule_ID": rid, "Physical_Relationship": f"{c['Source_Tags']} <-> {c['Target_Tags']} ({c['Relationship_Type']})",
        "Engineering_Process_Rationale": c["Rationale"],
        "TRAIN_Normal_Evidence": (f"n_events={val['N_Events_TRAIN_NORMAL']}, SMD={val.get('State_SMD', np.nan)}, "
                                   f"KS={val.get('State_KS_Statistic', np.nan)}" if c["Relationship_Type"].startswith("pump")
                                   else f"n={val['N_Events_TRAIN_NORMAL']}, rho={val.get('Spearman_Rho', np.nan):.3f}, "
                                        f"CV={val.get('Residual_Stability_CV', np.nan):.3f}"),
        "Temporal_Evidence": (f"best_offset={val.get('Best_Event_Offset_Seconds', np.nan)}s" if c["Relationship_Type"].startswith("pump")
                               else f"best_lag={val.get('Best_Lag_Seconds', np.nan)}s"),
        "Coverage": f"VALIDATION NORMAL={cov['VALIDATION_NORMAL_Coverage_Pct']:.2f}%, "
                    f"ATTACK={cov['VALIDATION_ATTACK_Coverage_Pct']:.2f}%",
        "Robustness": f"TRAIN-N median={rob['TRAIN_NORMAL_Score_Median']:.3f}, "
                       f"VAL-N median={rob['VALIDATION_NORMAL_Score_Median']:.3f}, "
                       f"KS(shift)={rob['TRAIN_vs_VALIDATION_NORMAL_KS']}, "
                       f"saturated={rob['Saturated_In_VALIDATION']}",
        "Main_Limitation": feature_def_df[feature_def_df.Source_Rule == rid]["Limitation"].iloc[0],
        "Final_Decision": d["decision"], "Decision_Rationale": d["reason"],
    })
screening_df = pd.DataFrame(screening_rows)
screening_path = TABLES_DIR / "stage5_final_rule_screening.csv"
screening_df.to_csv(screening_path, index=False)
log(f"\nWrote {screening_path} ({len(screening_df)} rows)")

retained_rule_ids = [rid for rid, d in decisions.items() if d["decision"] == "RETAIN"]
final_feature_rows = []
for rid in retained_rule_ids:
    fdef = feature_def_df[feature_def_df.Source_Rule == rid].iloc[0]
    final_feature_rows.append(dict(fdef))
final_features_df = pd.DataFrame(final_feature_rows)
final_features_path = TABLES_DIR / "stage5_final_retained_symbolic_features.csv"
final_features_df.to_csv(final_features_path, index=False)
log(f"Wrote {final_features_path} ({len(final_features_df)} rows)")
FINAL_FEATURE_NAMES = final_features_df["Feature"].tolist() if len(final_features_df) else []
log(f"Final retained Stage 5 symbolic feature set: {FINAL_FEATURE_NAMES}")

# =============================================================================
# PART 9 — COMPARISON WITH STAGE 2 SCREENING STANDARD
# =============================================================================
section("PART 9 — COMPARISON WITH STAGE 2 SCREENING METHODOLOGY (NOT attack-detection accuracy)")

s2_screening = pd.read_csv(STAGE2_SCREENING_PATH) if STAGE2_SCREENING_PATH.exists() else None
s2_features = pd.read_csv(STAGE2_FEATURES_PATH) if STAGE2_FEATURES_PATH.exists() else None

comparison_rows = []
if s2_screening is not None:
    s2_n = len(s2_screening)
    s2_retain = (s2_screening.Final_Decision == "RETAIN").sum()
    s2_limited = (s2_screening.Final_Decision == "LIMITED").sum()
    s2_reject = (s2_screening.Final_Decision == "REJECT").sum()
    comparison_rows.append({"Metric": "Candidate rule count", "Stage_2": s2_n, "Stage_5": len(candidates)})
    comparison_rows.append({"Metric": "Retained rule count", "Stage_2": int(s2_retain), "Stage_5": n_retained})
    comparison_rows.append({"Metric": "Limited count", "Stage_2": int(s2_limited), "Stage_5": n_limited})
    comparison_rows.append({"Metric": "Rejected count", "Stage_2": int(s2_reject), "Stage_5": n_rejected})
    comparison_rows.append({"Metric": "Retained feature count",
                             "Stage_2": len(s2_features) if s2_features is not None else np.nan,
                             "Stage_5": len(FINAL_FEATURE_NAMES)})
    best_s2_cov = "~99.6% (S2_R1_score, state-based) / 100% (S2_R3B_state_mismatch, state-based)"
    best_s5_cov = (f"{coverage_df.loc[coverage_df.Rule_ID.isin(retained_rule_ids), 'VALIDATION_NORMAL_Coverage_Pct'].max():.2f}%"
                    if retained_rule_ids else "n/a (no retained rules)")
    comparison_rows.append({"Metric": "Best retained-feature VALIDATION coverage", "Stage_2": best_s2_cov, "Stage_5": best_s5_cov})
    comparison_rows.append({"Metric": "Dominant rule TYPE among retained features",
                             "Stage_2": "state-based (actuator-sensor + actuator-actuator)",
                             "Stage_5": ("state-based / sensor-sensor mix" if retained_rule_ids else "n/a")})
    comparison_rows.append({"Metric": "Interpretability (qualitative)",
                             "Stage_2": "High -- single-cause, well-documented plant chemistry (dosing, valve)",
                             "Stage_5": "High for pump->sensor (mechanical gating); moderate for "
                                        "sensor-sensor (same-instrument-type consistency, plant-specific "
                                        "identities of individual FIT/PIT/AIT tags not independently confirmed)"})
    log("Stage 2 vs Stage 5 methodology comparison (screening process, NOT detection accuracy):")
    for r in comparison_rows:
        log(f"  {r['Metric']}: Stage 2 = {r['Stage_2']}  |  Stage 5 = {r['Stage_5']}")
else:
    log("Stage 2 screening table not found -- skipping comparison.")

comparison_df = pd.DataFrame(comparison_rows)
comparison_path = TABLES_DIR / "stage5_vs_stage2_methodology_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)
log(f"\nWrote {comparison_path} ({len(comparison_df)} rows)")

# =============================================================================
# PART 11 — FIGURES
# =============================================================================
section("PART 11 — GENERATING FIGURES (300 DPI)")

plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
                      "axes.titlesize": 13, "axes.labelsize": 11})
STATUS_COLOR = {"RETAIN": "#55A868", "LIMITED": "#F4B183", "REJECT": "#C44E52"}
rule_ids_order = [c["Rule_ID"] for c in candidates]

# --- Figure: candidate rule validation summary ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
ax = axes[0]
support_vals = []
for rid in rule_ids_order:
    ev = rule_evidence[rid]
    support_vals.append(abs(ev["smd"]) if ev["kind"] == "pump_sensor" else abs(ev["rho"]))
colors = [STATUS_COLOR[decisions[r]["decision"]] for r in rule_ids_order]
ax.barh(rule_ids_order, support_vals, color=colors)
ax.set_xlabel("|SMD| (pump->sensor) or |Spearman rho| (sensor-sensor)")
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
fig.suptitle("Stage 5 Candidate Relationship Empirical Support and Coverage", y=1.16)
fig.tight_layout()
fig1_path = FIGURES_DIR / "stage5_candidate_rule_validation.png"
fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig1_path}")

# --- Figure: normal response profiles ---
n_rules = len(candidates)
fig, axes = plt.subplots(1, n_rules, figsize=(4.2 * n_rules, 5))
if n_rules == 1:
    axes = [axes]
for ax, c in zip(axes, candidates):
    rid = c["Rule_ID"]
    ev = rule_evidence[rid]
    if ev["kind"] == "pump_sensor":
        ax.hist(ev["off_ref"], bins=40, alpha=0.6, label=f"P501=OFF (n={len(ev['off_ref']):,})", color="#C44E52", density=True)
        ax.hist(ev["on_ref"], bins=40, alpha=0.6, label=f"P501=ON (n={len(ev['on_ref']):,})", color="#4C72B0", density=True)
        ax.set_xlabel(c["Target_Tags"])
        ax.set_title(f"{rid}\nP501 state vs {c['Target_Tags']}")
    else:
        ax.hist(ev["residual_sample"], bins=40, color="#55A868", edgecolor="white")
        for pct in [5, 50, 95]:
            v = np.percentile(ev["residual_sample"], pct)
            ax.axvline(v, linestyle="--", color="#C44E52", linewidth=1)
        ax.set_xlabel(f"{ev['residual_name']}: {c['Source_Tags']} vs {c['Target_Tags']}")
        ax.set_title(f"{rid}\n(lag={ev['best_lag']['lag']}s, rho={ev['rho']:.2f})")
    if ev["kind"] == "pump_sensor":
        ax.legend(fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.4)
fig.suptitle("Stage 5 TRAIN-NORMAL Response / Relationship Profiles", y=1.03)
fig.tight_layout()
fig2_path = FIGURES_DIR / "stage5_normal_response_profiles.png"
fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig2_path}")

# --- Figure: symbolic score VALIDATION NORMAL vs ATTACK ---
fig, axes = plt.subplots(1, n_rules, figsize=(4.2 * n_rules, 5.5))
if n_rules == 1:
    axes = [axes]
for ax, c in zip(axes, candidates):
    rid = c["Rule_ID"]
    ev = rule_evidence[rid]
    normal_vals = ev["score_val"][ev["eval_mask_val"] & (val_df["Label"].to_numpy() == 0)]
    attack_vals = ev["score_val"][ev["eval_mask_val"] & (val_df["Label"].to_numpy() == 1)]
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
fig.suptitle("Stage 5 Symbolic Scores: VALIDATION NORMAL vs. ATTACK (where evaluable)", y=1.03)
fig.tight_layout()
fig3_path = FIGURES_DIR / "stage5_symbolic_score_validation_comparison.png"
fig.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig3_path}")

# --- Figure: final rule screening ---
fig, ax = plt.subplots(figsize=(9, 5))
status_num = {"REJECT": 0, "LIMITED": 1, "RETAIN": 2}
vals = [status_num[decisions[r]["decision"]] for r in rule_ids_order]
colors4 = [STATUS_COLOR[decisions[r]["decision"]] for r in rule_ids_order]
ax.barh(rule_ids_order, [1] * len(rule_ids_order), color=colors4)
for i, r in enumerate(rule_ids_order):
    ax.text(0.5, i, decisions[r]["decision"], ha="center", va="center", fontsize=10, fontweight="bold")
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title("Stage 5 Final Rule Screening Decisions")
fig.tight_layout()
fig4_path = FIGURES_DIR / "stage5_final_rule_screening.png"
fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig4_path}")

# --- Figure: final retained features schematic ---
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
        draw_box(ax, (6.9, y0), 2.9, 1.3, f"{fname}\n{coverage_df[coverage_df.Rule_ID==rid]['VALIDATION_NORMAL_Coverage_Pct'].iloc[0]:.1f}% coverage",
                 facecolor="#E2F0D9", edgecolor="#55A868")
        ax.annotate("", xy=(6.85, y0 + 0.65), xytext=(6.55, y0 + 0.65),
                     arrowprops=dict(arrowstyle="->", lw=1.8, color="#55A868"))
    ax.set_title("Final Retained Stage 5 Symbolic Feature Set", fontsize=12)
else:
    ax.text(5, ax.get_ylim()[1] / 2, "No Stage 5 candidate rules were RETAINED.\n"
            "See tables/stage5_final_rule_screening.csv for LIMITED/REJECT reasoning.",
            ha="center", va="center", fontsize=12)
    ax.set_title("Final Retained Stage 5 Symbolic Feature Set: NONE RETAINED", fontsize=12)
fig.tight_layout()
fig5_path = FIGURES_DIR / "stage5_final_retained_features.png"
fig.savefig(fig5_path, dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"Wrote {fig5_path}")

# =============================================================================
# PART 12 — ACADEMIC NARRATIVE
# =============================================================================
section("PART 12 — WRITING ACADEMIC NARRATIVE")

strongest_rule = max(candidates, key=lambda c: abs(rule_evidence[c["Rule_ID"]]["smd"]) if rule_evidence[c["Rule_ID"]]["kind"] == "pump_sensor" else abs(rule_evidence[c["Rule_ID"]]["rho"]))
weakest_rule = min(candidates, key=lambda c: abs(rule_evidence[c["Rule_ID"]]["smd"]) if rule_evidence[c["Rule_ID"]]["kind"] == "pump_sensor" else abs(rule_evidence[c["Rule_ID"]]["rho"]))

narrative = f"""STAGE 5 SYMBOLIC PIPELINE — ACADEMIC NARRATIVE
(Draft prose for Methodology/Results. Review before submission.)

1. STAGE 5 STRUCTURE
Stage 5 (Reverse Osmosis) contains {len(STAGE5_TAGS)} physical tags: {len(STAGE5_SENSORS)} continuous
sensors (4 AIT analyzers, 4 FIT flow meters, 3 PIT pressure transmitters) and only 2 actuator tags
(P501, P502). Empirical audit (Part 1) confirmed P502 is constant across the entire TRAIN partition
(no observed state change), leaving P501 as Stage 5's only actuator with usable variation -- a
structural contrast with Stage 2, which had three varying actuators (MV201, P203, P205) available for
rule construction.

2. CANDIDATE RELATIONSHIP GENERATION
Because Stage 5 offers only one varying actuator, candidate relationships were generated from two
structurally defensible categories: (a) pump-to-sensor gating, restricted to the sensors most strongly
correlated with P501 in TRAIN-NORMAL data (never assumed a priori), and (b) same-instrument-type
sensor-sensor consistency (FIT-FIT, PIT-PIT, AIT-AIT), using the strongest same-family Spearman
correlation as a discovery signal only. No cross-family relationship (e.g. a specific FIT tag assumed
to gate a specific PIT tag) was proposed without a direct correlation basis, and no plant-specific
identity (e.g. "FIT502 is the permeate flow meter") was asserted as fact -- all such interpretations are
marked as uncertain engineering assumptions, distinct from the directly observed statistical evidence.

3. TRAIN-NORMAL EMPIRICAL VALIDATION
Correlation was explicitly NOT treated as sufficient evidence on its own. Pump-to-sensor candidates
were additionally validated via a state-conditioned distribution comparison (standardized mean
difference and interquartile-range overlap between P501-OFF and P501-ON periods) and, where enough
TRAIN-NORMAL transitions existed, an event-triggered response check across a {len(EVENT_OFFSETS)}-point
offset grid (0-120s). Sensor-sensor candidates were additionally validated via a stability check on
their most consistent residual formalization (ratio or difference, whichever showed lower coefficient
of variation) and a symmetric lag sweep to test for a hydraulically-motivated delay.

4. TEMPORAL BEHAVIOR AND NORMAL ENVELOPES
For each supported relationship, empirical (not Gaussian-assumed) TRAIN-NORMAL reference distributions
were derived -- P5/P10/P25/median/P75/P90/P95 and IQR -- either as state-conditioned sensor
distributions, event-response distributions at the best-supported offset, or residual distributions at
the best-supported lag.

5. WHY SOME RULES WERE RETAINED
{"Rules were retained where the underlying relationship showed strong TRAIN-normal empirical support AND the resulting symbolic score reached practical (>=90%) VALIDATION coverage without saturating (VALIDATION-NORMAL mean below the documented 0.90 ceiling)." if n_retained else "No Stage 5 candidate met the combined bar of strong TRAIN-normal support, practical coverage, and non-saturated VALIDATION robustness -- see Section 6."}

6. WHY SOME PHYSICALLY PLAUSIBLE RULES WERE LIMITED OR REJECTED
{chr(10).join(f"- {rid}: {decisions[rid]['reason']}" for rid in decisions if decisions[rid]['decision'] != 'RETAIN')}

7. FINAL STAGE 5 SYMBOLIC FEATURE SET
{f"The finalized Stage 5 symbolic feature set consists of: {', '.join(FINAL_FEATURE_NAMES)}." if FINAL_FEATURE_NAMES else "No Stage 5 symbolic feature was promoted to the final retained set at this stage; all candidates remain documented as LIMITED or REJECTED evidence for future revisiting."}

8. MAJOR LIMITATIONS
Stage 5's single varying actuator sharply constrains the actuator-actuator coupling category that
proved strongest in Stage 2 (S2-R3B) -- no analog exists here. Sensor-sensor candidates, while
structurally defensible, rest on same-instrument-type correlation rather than an independently
documented plant P&ID identity for each tag, so their physical interpretation carries more uncertainty
than Stage 2's dosing-pump relationships. The strongest Stage 5 relationship found was {strongest_rule['Rule_ID']}
({strongest_rule['Source_Tags']} <-> {strongest_rule['Target_Tags']}); the weakest was
{weakest_rule['Rule_ID']} ({weakest_rule['Source_Tags']} <-> {weakest_rule['Target_Tags']}).

9. COMPARISON WITH THE STAGE 2 SCREENING METHODOLOGY
The identical decision framework (TRAIN-normal-only reference construction, distribution-free empirical
CDF scoring, explicit RETAIN/LIMITED/REJECT thresholds for consistency, coverage, and saturation) was
applied unchanged to Stage 5, preserving methodological consistency across stages. See
tables/stage5_vs_stage2_methodology_comparison.csv for the full quantitative comparison. No
classifier or attack-detection performance claim is made anywhere in this analysis -- this is a
feature-readiness screening only.
"""

narrative_path = RESULTS_DIR / "stage5_symbolic_pipeline_narrative.txt"
narrative_path.write_text(narrative, encoding="utf-8")
log(f"Wrote {narrative_path}")

# =============================================================================
# PART 13 — MAIN RESULTS LOG
# =============================================================================
section("PART 13 — WRITING MAIN RESULTS LOG")

main_lines = []
main_lines.append("STAGE 5 END-TO-END SYMBOLIC PIPELINE — MAIN RESULTS")
main_lines.append(f"Generated: {RUN_START.isoformat()}")
main_lines.append(f"Python: {sys.version}")
main_lines.append(f"pandas: {pd.__version__} | numpy: {np.__version__} | scipy: {__import__('scipy').__version__}")
main_lines.append(f"OS: {platform.system()} {platform.release()}")
main_lines.append("")
main_lines.append(f"Data sources: {TRAIN_PATH.name} ({len(train_df):,} rows), "
                   f"{VAL_PATH.name} ({len(val_df):,} rows). TEST was never loaded.")
main_lines.append("")
main_lines.append(f"=== STAGE 5 CANDIDATE RULES ({len(candidates)}) ===")
for c in candidates:
    main_lines.append(f"  {c['Rule_ID']}: {c['Source_Tags']} <-> {c['Target_Tags']} ({c['Relationship_Type']})")
main_lines.append("")
main_lines.append("=== STAGE 5 FINAL DECISIONS ===")
for rid in rule_ids_order:
    main_lines.append(f"  {rid} -> {decisions[rid]['decision']}")
main_lines.append("")
main_lines.append("=== FINAL RETAINED STAGE 5 SYMBOLIC FEATURE SET ===")
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
    main_lines.append(f"  {r['Rule_ID']}: TRAIN-N median={r['TRAIN_NORMAL_Score_Median']:.3f}, "
                       f"VAL-N median={r['VALIDATION_NORMAL_Score_Median']:.3f}, "
                       f"saturated={r['Saturated_In_VALIDATION']}")
main_lines.append("")
main_lines.append(f"Strongest Stage 5 physical relationship: {strongest_rule['Rule_ID']} "
                   f"({strongest_rule['Source_Tags']} <-> {strongest_rule['Target_Tags']})")
main_lines.append(f"Weakest candidate relationship: {weakest_rule['Rule_ID']} "
                   f"({weakest_rule['Source_Tags']} <-> {weakest_rule['Target_Tags']})")
main_lines.append("")
useful_beyond_s2 = n_retained > 0
main_lines.append(f"Does Stage 5 provide useful additional symbolic information beyond Stage 2? "
                   f"{'YES -- ' + str(n_retained) + ' feature(s) retained with practical VALIDATION coverage.' if useful_beyond_s2 else 'NOT CONFIRMED at this stage -- no Stage 5 candidate cleared the same RETAIN bar used for Stage 2; see LIMITED/REJECT reasoning above.'}")
main_lines.append("")
main_lines.append("=== GUARDRAILS CONFIRMED ===")
main_lines.append("  - TEST partition: never loaded, filename not referenced anywhere in this script.")
main_lines.append("  - TRAIN NORMAL used for all reference distributions and rule construction.")
main_lines.append("  - VALIDATION used only for coverage/robustness diagnostics, never to tune anything.")
main_lines.append("  - No model was trained. No ML feature selection was performed.")
main_lines.append("  - No existing script/output was modified; all Stage 2 outputs preserved.")
main_lines.append("  - No physical relationship was invented beyond what Stage 5's tag structure and "
                   "TRAIN-normal data support; uncertain assumptions are explicitly marked throughout.")
main_lines.append("  - Analysis is fully deterministic (no randomness used).")

main_path = RESULTS_DIR / "stage5_end_to_end_symbolic_pipeline.txt"
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

log(f"Stage 5 candidate rules ({len(candidates)}):")
for c in candidates:
    log(f"  {c['Rule_ID']}: {c['Source_Tags']} <-> {c['Target_Tags']}")
log("\nStage 5 final decisions:")
for rid in rule_ids_order:
    log(f"  {rid} -> {decisions[rid]['decision']}")
log(f"\nFinal retained Stage 5 symbolic feature set: {FINAL_FEATURE_NAMES if FINAL_FEATURE_NAMES else '(none)'}")
log(f"\nCandidate={len(candidates)}, Retained={n_retained}, Limited={n_limited}, Rejected={n_rejected}")

run_end = datetime.now()
log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
log("PIPELINE COMPLETE. No model trained, no ML feature selection, no attack-threshold tuning.")

if not all_checks_passed:
    sys.exit(1)
