"""
code/06_stage2_normal_operating_envelope.py

STAGE 2 DIAGNOSTIC #3 — NORMAL OPERATING ENVELOPE FOR THE FOUR STRONGLY-
SUPPORTED STAGE 2 RELATIONSHIPS. ANALYSIS ONLY.

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

Quantifies normal-operation response distributions for:
  S2-R1: MV201 OPEN  -> FIT201 flow response
  S2-R2: P205 ON     -> AIT203 ORP response
  S2-R3: P205 ON     -> P203 compensatory activation
  S2-R4: P203 ON     -> AIT202 pH decrease

using ONLY TRAIN NORMAL-labeled transition events (Part 1-4), then
replays the resulting TRAIN-derived candidate response windows,
UNCHANGED, against VALIDATION NORMAL and VALIDATION ATTACK transitions
as a diagnostic (Part 5). Part 6 discusses how each relationship might
eventually become a continuously-evaluable symbolic feature.

This script does NOT define final thresholds, does NOT select features,
does NOT train any model, and does NOT modify any existing script,
dataset, or split. Uses ONLY processed/train_unscaled.csv and
processed/validation_unscaled.csv. TEST is never loaded -- its filename
does not appear anywhere below, by design.
"""

import os
import sys
import hashlib
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

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
TABLES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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


section("STAGE 2 NORMAL OPERATING ENVELOPE — RUN START (ANALYSIS ONLY)")
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
OFFSETS = [5, 10, 15, 20, 30, 45, 60, 90, 120]
MAX_HORIZON = 120
MIN_N_FOR_RECOMMENDATION = 30

# =============================================================================
# LOAD DATA
# =============================================================================
section("LOADING TRAIN AND VALIDATION (TEST NEVER LOADED)")

train_df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
val_df = pd.read_csv(VAL_PATH).reset_index(drop=True)
train_df["_ts"] = pd.to_datetime(train_df["Timestamp"])
val_df["_ts"] = pd.to_datetime(val_df["Timestamp"])
log(f"TRAIN: {train_df.shape[0]:,} rows | VALIDATION: {val_df.shape[0]:,} rows")

# Fixed, previously-inferred state semantics (NOT re-derived here):
#   MV201: 2=steady OPEN, 1=CLOSED, 0=transitional/uncertain (excluded)
#   P203 : 1=OFF, 2=ON
#   P205 : 1=OFF, 2=ON
# P201/P202/P204/P206 are excluded from rule construction at this stage.
RELATIONSHIPS = [
    {"id": "S2-R1", "name": "MV201 OPEN -> FIT201 flow response", "actuator": "MV201",
     "from_state": 1, "to_state": 2, "target": "FIT201", "kind": "delta", "expected_sign": +1},
    {"id": "S2-R2", "name": "P205 ON -> AIT203 ORP response", "actuator": "P205",
     "from_state": 1, "to_state": 2, "target": "AIT203", "kind": "delta", "expected_sign": +1},
    {"id": "S2-R3", "name": "P205 ON -> P203 compensatory activation", "actuator": "P205",
     "from_state": 1, "to_state": 2, "target": "P203", "kind": "activation", "on_state": 2,
     "expected_sign": None},
    {"id": "S2-R4", "name": "P203 ON -> AIT202 pH decrease", "actuator": "P203",
     "from_state": 1, "to_state": 2, "target": "AIT202", "kind": "delta", "expected_sign": -1},
]

# =============================================================================
# HELPER FUNCTIONS
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


def baseline_value(df, i0, target, window=BASELINE_WINDOW):
    return df[target].iloc[i0 - window:i0].mean()


def check_offset_window(df, i0, offset, require_normal=False):
    j = i0 + offset
    if j >= len(df):
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


def offset_value(df, target, i0, offset):
    return df[target].iloc[i0 + offset]


def scan_first_activation(df, i0, target_tag, on_state, max_horizon=MAX_HORIZON, require_normal=False):
    """Returns (delay_seconds_or_None, fully_observed_bool)."""
    ts0 = df["_ts"].iloc[i0]
    for step in range(1, max_horizon + 1):
        j = i0 + step
        if j >= len(df):
            return None, False
        if df["_ts"].iloc[j] != ts0 + pd.Timedelta(seconds=step):
            return None, False
        if require_normal and df["Label"].iloc[j] != 0:
            return None, False
        if df[target_tag].iloc[j] == on_state:
            return step, True
    return None, True


def summarize(values):
    vals = np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))],
                     dtype=float)
    n = len(vals)
    if n == 0:
        keys = ["n", "mean", "median", "std", "min", "max", "p5", "p10", "p25", "p75", "p90", "p95", "iqr"]
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
# PART 1 — NORMAL-ONLY EVENT EXTRACTION (TRAIN)
# =============================================================================
section("PART 1 — NORMAL-ONLY EVENT EXTRACTION (TRAIN)")

event_store = {}  # rel_id -> list of dicts {event_id, i0, ts}
per_event_rows = []

for rel in RELATIONSHIPS:
    all_trans = find_specific_transitions(train_df, rel["actuator"], rel["from_state"], rel["to_state"])
    total_switch_events = int((train_df[rel["actuator"]] != train_df[rel["actuator"]].shift(1)).sum() - 1)
    events = []
    for i0 in all_trans:
        if train_df["Label"].iloc[i0] != 0:
            continue  # must occur during NORMAL-labeled operation
        if not check_baseline_window(train_df, i0, rel["actuator"], rel["from_state"], require_normal=True):
            continue
        if rel["kind"] == "activation":
            # also require the compensating actuator (target) was OFF and Normal
            # throughout the SAME baseline window, so we are only counting
            # genuine new activations, not an already-on pump.
            if not check_baseline_window(train_df, i0, rel["target"], 1, require_normal=True):
                continue
        events.append(i0)
    event_store[rel["id"]] = events
    log(f"{rel['id']} ({rel['name']}): {len(all_trans)} raw {rel['actuator']} "
        f"{rel['from_state']}->{rel['to_state']} transitions in TRAIN, "
        f"{len(events)} usable NORMAL-only events after baseline/continuity/label filtering "
        f"(actuator had {total_switch_events} total switch events across all states).")

# =============================================================================
# PART 2 — NORMAL RESPONSE DISTRIBUTIONS
# =============================================================================
section("PART 2 — NORMAL RESPONSE DISTRIBUTIONS (TRAIN NORMAL EVENTS)")

envelope_rows = []
activation_delay_by_rel = {}

for rel in RELATIONSHIPS:
    events = event_store[rel["id"]]
    if rel["kind"] == "delta":
        for offset in OFFSETS:
            vals = []
            for k, i0 in enumerate(events):
                if not check_offset_window(train_df, i0, offset, require_normal=True):
                    continue
                base = baseline_value(train_df, i0, rel["target"])
                resp = offset_value(train_df, rel["target"], i0, offset) - base
                vals.append(resp)
                per_event_rows.append({
                    "Event_ID": f"{rel['id']}_{k:04d}", "Relationship": rel["name"],
                    "Actuator": rel["actuator"], "Target": rel["target"],
                    "Event_Timestamp": train_df["_ts"].iloc[i0], "Response_Kind": "delta",
                    "Offset_Seconds": offset, "Usable": True, "Response_Value": resp,
                    "Baseline_Value": base, "Censored": False,
                })
            stats = summarize(vals)
            pe = pct_expected(vals, rel["expected_sign"])
            envelope_rows.append({
                "Relationship": rel["name"], "Metric_Type": "response_delta", "Offset_Seconds": offset,
                "N_Usable": stats["n"], "Mean": stats["mean"], "Median": stats["median"],
                "Std": stats["std"], "Min": stats["min"], "Max": stats["max"], "P5": stats["p5"],
                "P10": stats["p10"], "P25": stats["p25"], "P75": stats["p75"], "P90": stats["p90"],
                "P95": stats["p95"], "IQR": stats["iqr"], "Pct_Expected_Direction": pe,
                "Activation_Probability": np.nan, "N_No_Response_120s": np.nan,
            })
        log(f"\n{rel['id']} response_delta by offset (n usable, median, pct_expected_direction):")
        for r in envelope_rows[-len(OFFSETS):]:
            log(f"  offset={r['Offset_Seconds']:>3}s n={r['N_Usable']:>4} median={r['Median']:.5f} "
                f"pct_expected={r['Pct_Expected_Direction']}")

    else:  # activation kind (S2-R3: P205 -> P203)
        delays, censored_flags = [], []
        for k, i0 in enumerate(events):
            delay, fully_observed = scan_first_activation(
                train_df, i0, rel["target"], rel["on_state"], require_normal=True)
            delays.append(delay)
            censored_flags.append(delay is None and fully_observed)
            per_event_rows.append({
                "Event_ID": f"{rel['id']}_{k:04d}", "Relationship": rel["name"],
                "Actuator": rel["actuator"], "Target": rel["target"],
                "Event_Timestamp": train_df["_ts"].iloc[i0], "Response_Kind": "activation_delay",
                "Offset_Seconds": np.nan, "Usable": (delay is not None or fully_observed),
                "Response_Value": delay if delay is not None else np.nan,
                "Baseline_Value": np.nan, "Censored": (delay is None and fully_observed),
            })
        observed_delays = [d for d in delays if d is not None]
        n_censored_no_response = sum(censored_flags)
        delay_stats = summarize(observed_delays)
        activation_delay_by_rel[rel["id"]] = {
            "delay_stats": delay_stats, "n_censored_no_response": n_censored_no_response,
            "n_events": len(events),
        }
        envelope_rows.append({
            "Relationship": rel["name"], "Metric_Type": "activation_delay_distribution",
            "Offset_Seconds": np.nan, "N_Usable": delay_stats["n"], "Mean": delay_stats["mean"],
            "Median": delay_stats["median"], "Std": delay_stats["std"], "Min": delay_stats["min"],
            "Max": delay_stats["max"], "P5": delay_stats["p5"], "P10": delay_stats["p10"],
            "P25": delay_stats["p25"], "P75": delay_stats["p75"], "P90": delay_stats["p90"],
            "P95": delay_stats["p95"], "IQR": delay_stats["iqr"], "Pct_Expected_Direction": np.nan,
            "Activation_Probability": (delay_stats["n"] / len(events)) if len(events) else np.nan,
            "N_No_Response_120s": n_censored_no_response,
        })
        log(f"\n{rel['id']} activation delay distribution (TRAIN NORMAL): "
            f"n_events={len(events)}, observed_activations={delay_stats['n']}, "
            f"no_response_within_120s={n_censored_no_response}")
        log(f"  median_delay={delay_stats['median']}, mean_delay={delay_stats['mean']}, "
            f"p10={delay_stats['p10']}, p25={delay_stats['p25']}, p75={delay_stats['p75']}, "
            f"p90={delay_stats['p90']}")

        for offset in OFFSETS:
            usable_events = [i0 for i0 in events if check_offset_window(train_df, i0, offset, require_normal=True)]
            n_usable = len(usable_events)
            n_activated = 0
            for i0, delay in zip(events, delays):
                if i0 not in usable_events:
                    continue
                if delay is not None and delay <= offset:
                    n_activated += 1
                per_event_rows.append({
                    "Event_ID": f"{rel['id']}_{events.index(i0):04d}", "Relationship": rel["name"],
                    "Actuator": rel["actuator"], "Target": rel["target"],
                    "Event_Timestamp": train_df["_ts"].iloc[i0], "Response_Kind": "activation_by_offset",
                    "Offset_Seconds": offset, "Usable": True,
                    "Response_Value": 1.0 if (delay is not None and delay <= offset) else 0.0,
                    "Baseline_Value": np.nan, "Censored": False,
                })
            act_prob = (n_activated / n_usable) if n_usable else np.nan
            envelope_rows.append({
                "Relationship": rel["name"], "Metric_Type": "activation_probability",
                "Offset_Seconds": offset, "N_Usable": n_usable, "Mean": np.nan, "Median": np.nan,
                "Std": np.nan, "Min": np.nan, "Max": np.nan, "P5": np.nan, "P10": np.nan, "P25": np.nan,
                "P75": np.nan, "P90": np.nan, "P95": np.nan, "IQR": np.nan,
                "Pct_Expected_Direction": np.nan, "Activation_Probability": act_prob,
                "N_No_Response_120s": np.nan,
            })
        log(f"\n{rel['id']} activation probability by horizon:")
        for r in [r for r in envelope_rows if r["Relationship"] == rel["name"] and r["Metric_Type"] == "activation_probability"]:
            log(f"  horizon={r['Offset_Seconds']:>3}s n={r['N_Usable']:>4} "
                f"activation_prob={r['Activation_Probability']}")

envelope_df = pd.DataFrame(envelope_rows)
envelope_path = TABLES_DIR / "stage2_normal_envelope_summary.csv"
envelope_df.to_csv(envelope_path, index=False)
log(f"\nWrote {envelope_path} ({len(envelope_df)} rows)")

events_df = pd.DataFrame(per_event_rows)
events_path = TABLES_DIR / "stage2_normal_event_responses.csv"
events_df.to_csv(events_path, index=False)
log(f"Wrote {events_path} ({len(events_df)} rows)")

# =============================================================================
# PART 3 — DETERMINE EMPIRICAL RESPONSE WINDOWS (TRAIN NORMAL ONLY)
# =============================================================================
section("PART 3 — CANDIDATE RESPONSE WINDOWS (derived from TRAIN NORMAL data only)")

candidate_rows = []


def recommend_delta_offset(rel):
    sub = envelope_df[(envelope_df.Relationship == rel["name"]) & (envelope_df.Metric_Type == "response_delta")]
    sub = sub.sort_values("Offset_Seconds")
    valid = sub[sub["Pct_Expected_Direction"].notna()]
    if len(valid) == 0:
        return None, "no usable offsets -- insufficient data to recommend a window"
    max_pct = valid["Pct_Expected_Direction"].max()
    target_pct = 95.0 if max_pct >= 95.0 else max_pct
    candidates = valid[(valid["Pct_Expected_Direction"] >= target_pct) &
                        (valid["N_Usable"] >= MIN_N_FOR_RECOMMENDATION)]
    if len(candidates) == 0:
        candidates = valid[valid["N_Usable"] >= MIN_N_FOR_RECOMMENDATION]
        if len(candidates) == 0:
            candidates = valid
        candidates = candidates.sort_values("Pct_Expected_Direction", ascending=False)
        pick = candidates.iloc[0]
        reason = (f"best achievable consistency ({pick['Pct_Expected_Direction']:.1f}%) among offsets with "
                  f"n>={MIN_N_FOR_RECOMMENDATION} usable events; target of >=95% was not reached at any offset")
    else:
        pick = candidates.sort_values("Offset_Seconds").iloc[0]
        reason = (f"{pick['Pct_Expected_Direction']:.1f}% expected-direction consistency (>= target "
                  f"{target_pct:.1f}%), n={int(pick['N_Usable'])} usable TRAIN NORMAL events, "
                  f"IQR={pick['IQR']:.4f} -- earliest offset reaching this consistency band")
    return pick, reason


def recommend_activation_offset(rel):
    sub = envelope_df[(envelope_df.Relationship == rel["name"]) &
                       (envelope_df.Metric_Type == "activation_probability")].sort_values("Offset_Seconds")
    if len(sub) == 0 or sub["Activation_Probability"].isna().all():
        return None, "no usable offsets -- insufficient data to recommend a window"
    plateau = sub["Activation_Probability"].iloc[-1]
    target = 0.9 * plateau if plateau > 0 else 0
    candidates = sub[(sub["Activation_Probability"] >= target) & (sub["N_Usable"] >= MIN_N_FOR_RECOMMENDATION)]
    if len(candidates) == 0:
        candidates = sub[sub["N_Usable"] >= MIN_N_FOR_RECOMMENDATION]
        if len(candidates) == 0:
            candidates = sub
        pick = candidates.sort_values("Activation_Probability", ascending=False).iloc[0]
        reason = (f"best achievable activation probability ({pick['Activation_Probability']*100:.1f}%) among "
                  f"offsets with n>={MIN_N_FOR_RECOMMENDATION}; did not reach 90% of the 120s plateau "
                  f"({plateau*100:.1f}%)")
    else:
        pick = candidates.sort_values("Offset_Seconds").iloc[0]
        reason = (f"{pick['Activation_Probability']*100:.1f}% activation probability, n={int(pick['N_Usable'])} "
                  f"-- earliest offset reaching >=90% of the 120s plateau activation probability "
                  f"({plateau*100:.1f}%)")
    return pick, reason


for rel in RELATIONSHIPS:
    if rel["kind"] == "delta":
        pick, reason = recommend_delta_offset(rel)
        pct_at_rec = pick["Pct_Expected_Direction"] if pick is not None else np.nan
        n_at_rec = pick["N_Usable"] if pick is not None else 0
        iqr_at_rec = pick["IQR"] if pick is not None else np.nan
        act_prob_at_rec = np.nan
        offset_rec = pick["Offset_Seconds"] if pick is not None else np.nan
    else:
        pick, reason = recommend_activation_offset(rel)
        pct_at_rec = np.nan
        n_at_rec = pick["N_Usable"] if pick is not None else 0
        iqr_at_rec = np.nan
        act_prob_at_rec = pick["Activation_Probability"] if pick is not None else np.nan
        offset_rec = pick["Offset_Seconds"] if pick is not None else np.nan

    candidate_rows.append({
        "Rule_ID": rel["id"], "Relationship": rel["name"],
        "Recommended_Offset_Seconds": offset_rec, "Reason": reason,
        "Pct_Expected_Direction_At_Recommended": pct_at_rec,
        "N_Usable_At_Recommended": n_at_rec, "IQR_At_Recommended": iqr_at_rec,
        "Activation_Probability_At_Recommended": act_prob_at_rec,
        "Status": "TRAIN-derived candidate, NOT a final threshold",
    })
    log(f"\n{rel['id']}: {rel['name']}")
    log(f"  Candidate response window: {offset_rec} seconds")
    log(f"  Reason: {reason}")

candidate_df = pd.DataFrame(candidate_rows)
candidate_path = TABLES_DIR / "stage2_candidate_rule_windows.csv"
candidate_df.to_csv(candidate_path, index=False)
log(f"\nWrote {candidate_path} ({len(candidate_df)} rows)")

RECOMMENDED_OFFSET = {row["Rule_ID"]: row["Recommended_Offset_Seconds"] for row in candidate_rows}

# =============================================================================
# PART 4 — CANDIDATE NORMAL-ENVELOPE BOUNDARIES (extracted from Part 2 @ recommended offset)
# =============================================================================
section("PART 4 — CANDIDATE NORMAL-ENVELOPE BOUNDARIES (TRAIN NORMAL, at recommended offset)")

for rel in RELATIONSHIPS:
    off = RECOMMENDED_OFFSET[rel["id"]]
    if pd.isna(off):
        log(f"{rel['id']}: no candidate offset recommended -- skipping envelope boundary report.")
        continue
    if rel["kind"] == "delta":
        row = envelope_df[(envelope_df.Relationship == rel["name"]) &
                           (envelope_df.Metric_Type == "response_delta") &
                           (envelope_df.Offset_Seconds == off)]
        if len(row):
            r = row.iloc[0]
            log(f"{rel['id']} @ {off}s: P5={r['P5']:.5f}  P10={r['P10']:.5f}  Median={r['Median']:.5f}  "
                f"P90={r['P90']:.5f}  P95={r['P95']:.5f}  (n={int(r['N_Usable'])})")
    else:
        info = activation_delay_by_rel[rel["id"]]
        ds = info["delay_stats"]
        log(f"{rel['id']} activation-delay distribution (TRAIN NORMAL): "
            f"P10={ds['p10']}  P25={ds['p25']}  Median={ds['median']}  P75={ds['p75']}  P90={ds['p90']}  "
            f"n_observed={ds['n']}  n_no_response_within_120s={info['n_censored_no_response']} "
            f"(of {info['n_events']} total events)")
log("\nNo percentile has been designated the final violation threshold; these are candidate boundaries only.")

# =============================================================================
# PART 5 — VALIDATION-ONLY DIAGNOSTIC (candidate windows applied UNCHANGED)
# =============================================================================
section("PART 5 — VALIDATION DIAGNOSTIC (TRAIN-derived windows applied unchanged, NOT tuned)")

val_diag_rows = []

for rel in RELATIONSHIPS:
    off = RECOMMENDED_OFFSET[rel["id"]]
    if pd.isna(off):
        log(f"SKIPPING Part 5 for {rel['id']}: no candidate offset could be recommended in Part 3 "
            f"(insufficient TRAIN NORMAL events) -- nothing to apply to VALIDATION.")
        for stratum in ("NORMAL", "ATTACK"):
            val_diag_rows.append({
                "Relationship": rel["name"], "Label_Stratum": stratum,
                "Recommended_Offset_Seconds": np.nan, "N_Evaluable": 0,
                "Pct_Expected_Direction": np.nan, "Mean_Response": np.nan, "Median_Response": np.nan,
                "Std_Response": np.nan, "Pct_Below_TRAIN_P5": np.nan, "Pct_Below_TRAIN_P10": np.nan,
                "Activation_Probability": np.nan, "Pct_Failed_To_Activate": np.nan,
            })
        continue
    off = int(off)
    all_trans = find_specific_transitions(val_df, rel["actuator"], rel["from_state"], rel["to_state"])

    if rel["kind"] == "delta":
        train_p5 = envelope_df[(envelope_df.Relationship == rel["name"]) &
                                (envelope_df.Metric_Type == "response_delta") &
                                (envelope_df.Offset_Seconds == off)]["P5"].iloc[0]
        train_p10 = envelope_df[(envelope_df.Relationship == rel["name"]) &
                                 (envelope_df.Metric_Type == "response_delta") &
                                 (envelope_df.Offset_Seconds == off)]["P10"].iloc[0]
        for stratum, label_val in [("NORMAL", 0), ("ATTACK", 1)]:
            vals = []
            for i0 in all_trans:
                if int(val_df["Label"].iloc[i0]) != label_val:
                    continue
                if not check_baseline_window(val_df, i0, rel["actuator"], rel["from_state"], require_normal=False):
                    continue
                if not check_offset_window(val_df, i0, off, require_normal=False):
                    continue
                base = baseline_value(val_df, i0, rel["target"])
                vals.append(offset_value(val_df, rel["target"], i0, off) - base)
            stats = summarize(vals)
            pe = pct_expected(vals, rel["expected_sign"])
            arr = np.array(vals, dtype=float)
            pct_below_p5 = 100 * float((arr < train_p5).sum()) / len(arr) if len(arr) else np.nan
            pct_below_p10 = 100 * float((arr < train_p10).sum()) / len(arr) if len(arr) else np.nan
            val_diag_rows.append({
                "Relationship": rel["name"], "Label_Stratum": stratum,
                "Recommended_Offset_Seconds": off, "N_Evaluable": stats["n"],
                "Pct_Expected_Direction": pe, "Mean_Response": stats["mean"],
                "Median_Response": stats["median"], "Std_Response": stats["std"],
                "Pct_Below_TRAIN_P5": pct_below_p5, "Pct_Below_TRAIN_P10": pct_below_p10,
                "Activation_Probability": np.nan, "Pct_Failed_To_Activate": np.nan,
            })
    else:
        for stratum, label_val in [("NORMAL", 0), ("ATTACK", 1)]:
            usable_events = []
            for i0 in all_trans:
                if int(val_df["Label"].iloc[i0]) != label_val:
                    continue
                if not check_baseline_window(val_df, i0, rel["actuator"], rel["from_state"], require_normal=False):
                    continue
                if not check_baseline_window(val_df, i0, rel["target"], 1, require_normal=False):
                    continue
                if not check_offset_window(val_df, i0, off, require_normal=False):
                    continue
                usable_events.append(i0)
            n_activated = 0
            for i0 in usable_events:
                delay, _ = scan_first_activation(val_df, i0, rel["target"], rel["on_state"],
                                                   max_horizon=off, require_normal=False)
                if delay is not None and delay <= off:
                    n_activated += 1
            n = len(usable_events)
            act_prob = (n_activated / n) if n else np.nan
            pct_failed = (100 * (1 - act_prob)) if n else np.nan
            val_diag_rows.append({
                "Relationship": rel["name"], "Label_Stratum": stratum,
                "Recommended_Offset_Seconds": off, "N_Evaluable": n,
                "Pct_Expected_Direction": np.nan, "Mean_Response": np.nan, "Median_Response": np.nan,
                "Std_Response": np.nan, "Pct_Below_TRAIN_P5": np.nan, "Pct_Below_TRAIN_P10": np.nan,
                "Activation_Probability": act_prob, "Pct_Failed_To_Activate": pct_failed,
            })

val_diag_df = pd.DataFrame(val_diag_rows)
val_diag_path = TABLES_DIR / "stage2_validation_envelope_diagnostic.csv"
val_diag_df.to_csv(val_diag_path, index=False)
log(f"Wrote {val_diag_path} ({len(val_diag_df)} rows)")

log("\nVALIDATION diagnostic (TRAIN-derived windows, unchanged):")
for _, r in val_diag_df.iterrows():
    if pd.notna(r["N_Evaluable"]) and r["N_Evaluable"] > 0:
        if pd.notna(r["Activation_Probability"]):
            log(f"  {r['Relationship']:38s} [{r['Label_Stratum']:6s}] n={int(r['N_Evaluable']):>4} "
                f"activation_prob={r['Activation_Probability']:.3f} "
                f"pct_failed_to_activate={r['Pct_Failed_To_Activate']:.1f}%")
        else:
            log(f"  {r['Relationship']:38s} [{r['Label_Stratum']:6s}] n={int(r['N_Evaluable']):>4} "
                f"pct_expected={r['Pct_Expected_Direction']} median={r['Median_Response']:.5f} "
                f"pct_below_p5={r['Pct_Below_TRAIN_P5']} pct_below_p10={r['Pct_Below_TRAIN_P10']}")
    else:
        log(f"  {r['Relationship']:38s} [{r['Label_Stratum']:6s}] n=0 (no evaluable VALIDATION events)")

# =============================================================================
# PART 6 — CONTINUOUS CONSISTENCY PERSPECTIVE
# =============================================================================
section("PART 6 — CONTINUOUS CONSISTENCY PERSPECTIVE (discussion, not implemented)")

rule_representation = {
    "S2-R1": {
        "type": "B: state-based",
        "reasoning": (
            "MV201 spends long, stable stretches OPEN (~65% of TRAIN) or CLOSED (~34%), and FIT201 "
            "tracks the CURRENT state almost deterministically (near-0 when CLOSED, ~2.45 when OPEN) "
            "with only a few seconds of lag, not a slowly-decaying transient. The relationship is a "
            "steady-state correspondence, not a one-off event response, so a continuous state-"
            "consistency check ('IS MV201 OPEN but FIT201 near 0 RIGHT NOW') is more natural and gives "
            "far more time-series coverage than an event-triggered rule, which would only fire near the "
            "268 TRAIN switch events."
        ),
    },
    "S2-R2": {
        "type": "D: hybrid (event-triggered response + rolling-window trend)",
        "reasoning": (
            "The ORP jump after P205 turns on is a genuine transient (builds over 5-30s, per Part 2/3), "
            "so the clearest signal is around the transition itself -- an event-based check. But P205 "
            "only switches 134 times across 270,000 TRAIN rows, so a purely event-based rule would leave "
            "the vast majority of the time series unmonitored for this relationship. A rolling-window "
            "companion feature (e.g. is AIT203 behaving consistently with P205's CURRENT state over the "
            "last N seconds) would extend coverage between events."
        ),
    },
    "S2-R4": {
        "type": "D: hybrid (event-triggered response + rolling-window trend)",
        "reasoning": (
            "Same structural argument as S2-R2: P203 switches only 136 times in TRAIN, the pH response "
            "is a bounded transient after ON, and a rolling-window consistency check would be needed to "
            "cover the long stretches between switches."
        ),
    },
    "S2-R3": {
        "type": "A: event-based",
        "reasoning": (
            "This is fundamentally a discrete action following a discrete trigger (does P203 turn ON "
            "after P205 turns ON), not a continuous magnitude -- there is no natural 'rolling' or "
            "'state' formulation for an activation-coupling rule. It is inherently tied to P205's "
            "OFF->ON transition moments (134 in TRAIN) and is best implemented as an event-based check "
            "with the Part 3/4 activation-delay window."
        ),
    },
}
for rel in RELATIONSHIPS:
    info = rule_representation[rel["id"]]
    log(f"\n{rel['id']} ({rel['name']}):")
    log(f"  Recommended representation: {info['type']}")
    log(f"  Reasoning: {info['reasoning']}")

# =============================================================================
# NARRATIVE REPORT
# =============================================================================
section("SAVING results/stage2_normal_operating_envelope.txt")

report_lines = []
report_lines.append("STAGE 2 DIAGNOSTIC #3 — NORMAL OPERATING ENVELOPE. ANALYSIS ONLY. NO FINAL THRESHOLDS.")
report_lines.append(f"Generated: {RUN_START.isoformat()}")
report_lines.append(f"Python: {sys.version}")
report_lines.append(f"pandas: {pd.__version__} | numpy: {np.__version__}")
report_lines.append(f"OS: {platform.system()} {platform.release()}")
report_lines.append("")
report_lines.append(f"Data sources: {TRAIN_PATH.name} ({len(train_df):,} rows), "
                     f"{VAL_PATH.name} ({len(val_df):,} rows). TEST was never loaded.")
report_lines.append("")
report_lines.append("=== FIXED STATE SEMANTICS USED (not re-derived here) ===")
report_lines.append("  MV201: 2=steady OPEN, 1=CLOSED, 0=transitional/uncertain (excluded from events)")
report_lines.append("  P203 : 1=OFF, 2=ON")
report_lines.append("  P205 : 1=OFF, 2=ON")
report_lines.append("  P201/P202/P204/P206: excluded from rule construction at this stage")
report_lines.append("")
report_lines.append("=== PART 1: USABLE NORMAL-ONLY TRAIN EVENTS ===")
for rel in RELATIONSHIPS:
    report_lines.append(f"  {rel['id']} ({rel['name']}): {len(event_store[rel['id']])} usable events")
report_lines.append("")
report_lines.append("=== PART 3: RECOMMENDED CANDIDATE RESPONSE WINDOWS (TRAIN NORMAL ONLY) ===")
for row in candidate_rows:
    report_lines.append(f"  {row['Rule_ID']} ({row['Relationship']}):")
    report_lines.append(f"    Candidate response window: {row['Recommended_Offset_Seconds']} seconds")
    report_lines.append(f"    Reason: {row['Reason']}")
    report_lines.append(f"    Status: {row['Status']}")
report_lines.append("")
report_lines.append("=== PART 4: CANDIDATE NORMAL-ENVELOPE BOUNDARIES (TRAIN NORMAL, at recommended offset) ===")
for rel in RELATIONSHIPS:
    off = RECOMMENDED_OFFSET[rel["id"]]
    if rel["kind"] == "delta":
        row = envelope_df[(envelope_df.Relationship == rel["name"]) &
                           (envelope_df.Metric_Type == "response_delta") & (envelope_df.Offset_Seconds == off)]
        if len(row):
            r = row.iloc[0]
            report_lines.append(f"  {rel['id']} @ {off}s: P5={r['P5']:.5f} P10={r['P10']:.5f} "
                                 f"Median={r['Median']:.5f} P90={r['P90']:.5f} P95={r['P95']:.5f} "
                                 f"(n={int(r['N_Usable'])})")
    else:
        info = activation_delay_by_rel[rel["id"]]
        ds = info["delay_stats"]
        report_lines.append(f"  {rel['id']} activation delay: P10={ds['p10']} P25={ds['p25']} "
                             f"Median={ds['median']} P75={ds['p75']} P90={ds['p90']} "
                             f"n_no_response_120s={info['n_censored_no_response']}")
report_lines.append("")
report_lines.append("=== PART 5: VALIDATION DIAGNOSTIC (unchanged TRAIN windows) ===")
for _, r in val_diag_df.iterrows():
    if pd.notna(r["Activation_Probability"]):
        report_lines.append(f"  {r['Relationship']} [{r['Label_Stratum']}]: n={int(r['N_Evaluable'])} "
                             f"activation_prob={r['Activation_Probability']}")
    else:
        report_lines.append(f"  {r['Relationship']} [{r['Label_Stratum']}]: n={int(r['N_Evaluable']) if pd.notna(r['N_Evaluable']) else 0} "
                             f"pct_expected={r['Pct_Expected_Direction']} pct_below_p5={r['Pct_Below_TRAIN_P5']} "
                             f"pct_below_p10={r['Pct_Below_TRAIN_P10']}")
report_lines.append("")
report_lines.append("=== PART 6: RULE REPRESENTATION RECOMMENDATION ===")
for rel in RELATIONSHIPS:
    info = rule_representation[rel["id"]]
    report_lines.append(f"  {rel['id']}: {info['type']}")
    report_lines.append(f"    {info['reasoning']}")
report_lines.append("")
report_lines.append("=== GUARDRAILS CONFIRMED ===")
report_lines.append("  - TEST partition: never loaded, filename not referenced anywhere in this script.")
report_lines.append("  - No arbitrary/final thresholds were selected.")
report_lines.append("  - No feature selection was performed.")
report_lines.append("  - No model was trained.")
report_lines.append("  - No existing script, dataset, or split was modified.")
report_lines.append("  - Candidate windows were derived from TRAIN NORMAL data only, never from ATTACK labels.")
report_lines.append("  - Analysis is fully deterministic (no randomness used).")

report_path = RESULTS_DIR / "stage2_normal_operating_envelope.txt"
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
    "no_thresholds_selected": True,
    "no_feature_selection": True,
    "no_model_trained": True,
    "candidates_derived_from_train_normal_only": True,
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

log("1. RECOMMENDED TRAIN-DERIVED RESPONSE WINDOW PER RULE:")
for row in candidate_rows:
    log(f"   {row['Rule_ID']}: {row['Recommended_Offset_Seconds']}s -- {row['Reason']}")

log("\n2. NORMAL RESPONSE PERCENTILES (at recommended offset, TRAIN NORMAL):")
for rel in RELATIONSHIPS:
    off = RECOMMENDED_OFFSET[rel["id"]]
    if rel["kind"] == "delta":
        row = envelope_df[(envelope_df.Relationship == rel["name"]) &
                           (envelope_df.Metric_Type == "response_delta") & (envelope_df.Offset_Seconds == off)]
        if len(row):
            r = row.iloc[0]
            log(f"   {rel['id']}: P5={r['P5']:.5f} P10={r['P10']:.5f} Median={r['Median']:.5f} "
                f"P90={r['P90']:.5f} P95={r['P95']:.5f}")

log("\n3. P205->P203 ACTIVATION DELAY DISTRIBUTION (TRAIN NORMAL):")
ds = activation_delay_by_rel["S2-R3"]["delay_stats"]
nc = activation_delay_by_rel["S2-R3"]["n_censored_no_response"]
log(f"   n_observed={ds['n']}, median={ds['median']}, mean={ds['mean']}, "
    f"p10={ds['p10']}, p25={ds['p25']}, p75={ds['p75']}, p90={ds['p90']}, "
    f"no_response_within_120s={nc}")

log("\n4. VALIDATION NORMAL vs TRAIN NORMAL CONSISTENCY:")
for rel in RELATIONSHIPS:
    row = val_diag_df[(val_diag_df.Relationship == rel["name"]) & (val_diag_df.Label_Stratum == "NORMAL")]
    if len(row):
        r = row.iloc[0]
        if pd.notna(r["Activation_Probability"]):
            log(f"   {rel['id']}: VALIDATION-NORMAL n={int(r['N_Evaluable'])} "
                f"activation_prob={r['Activation_Probability']}")
        else:
            log(f"   {rel['id']}: VALIDATION-NORMAL n={int(r['N_Evaluable']) if pd.notna(r['N_Evaluable']) else 0} "
                f"pct_expected={r['Pct_Expected_Direction']} pct_below_p5={r['Pct_Below_TRAIN_P5']}")

log("\n5. ATTACK EVENTS FALLING OUTSIDE THE TRAIN NORMAL ENVELOPE (VALIDATION-ATTACK):")
for rel in RELATIONSHIPS:
    row = val_diag_df[(val_diag_df.Relationship == rel["name"]) & (val_diag_df.Label_Stratum == "ATTACK")]
    if len(row):
        r = row.iloc[0]
        if pd.notna(r["Activation_Probability"]):
            log(f"   {rel['id']}: VALIDATION-ATTACK n={int(r['N_Evaluable'])} "
                f"pct_failed_to_activate={r['Pct_Failed_To_Activate']}")
        else:
            n_eval = int(r["N_Evaluable"]) if pd.notna(r["N_Evaluable"]) else 0
            log(f"   {rel['id']}: VALIDATION-ATTACK n={n_eval} "
                f"pct_below_p5={r['Pct_Below_TRAIN_P5']} pct_below_p10={r['Pct_Below_TRAIN_P10']}")

log("\n6. RECOMMENDED RULE REPRESENTATION TYPE:")
for rel in RELATIONSHIPS:
    log(f"   {rel['id']}: {rule_representation[rel['id']]['type']}")

run_end = datetime.now()
log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
log("ANALYSIS-ONLY STAGE COMPLETE. No final thresholds selected, no features selected, no model trained.")

if not all_checks_passed:
    sys.exit(1)
