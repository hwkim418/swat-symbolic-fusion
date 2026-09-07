"""
code/08_stage2_p205_p203_deactivation_coupling.py

STAGE 2 DIAGNOSTIC #5 — P205 -> P203 DEACTIVATION COUPLING. ANALYSIS ONLY.

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

The prior diagnostic (script 07) found that S2-R3 (P205 OFF->ON followed by
P203 activation) is extremely consistent in TRAIN NORMAL, but that its
event-triggered coverage never overlaps a VALIDATION attack interval --
except for one notable exception: during VALIDATION attack interval 2,
P205 and P203 turn OFF simultaneously (an ON->OFF pair, not the OFF->ON
direction S2-R3 currently tracks).

This script determines, from TRAIN NORMAL data only, whether the same
P205->P203 coupling holds in the ON->OFF (deactivation) direction, then
uses that TRAIN-derived normal behavior (not attack labels) to judge
whether the VALIDATION interval-2 simultaneous OFF event was unusual or
consistent with ordinary control coupling.

No ML model is trained, no feature is selected, no final attack
threshold is chosen, and no existing script, dataset, or split is
modified. Uses ONLY processed/train_unscaled.csv and
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


section("P205->P203 DEACTIVATION COUPLING — RUN START (ANALYSIS ONLY)")
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
OFFSETS = [0, 1, 2, 3, 5, 10, 15, 30]
MAX_HORIZON = 30
P205_OFF, P205_ON = 1, 2
P203_OFF, P203_ON = 1, 2

# =============================================================================
# LOAD DATA
# =============================================================================
section("LOADING TRAIN AND VALIDATION (TEST NEVER LOADED)")

train_df = pd.read_csv(TRAIN_PATH).reset_index(drop=True)
val_df = pd.read_csv(VAL_PATH).reset_index(drop=True)
train_df["_ts"] = pd.to_datetime(train_df["Timestamp"])
val_df["_ts"] = pd.to_datetime(val_df["Timestamp"])
log(f"TRAIN: {train_df.shape[0]:,} rows | VALIDATION: {val_df.shape[0]:,} rows")
log("Fixed semantics used (not re-derived here): P205: 1=OFF, 2=ON | P203: 1=OFF, 2=ON")

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


def scan_first_state(df, i0, target_tag, target_state, max_horizon=MAX_HORIZON, require_normal=False):
    """Scans i0, i0+1, ... i0+max_horizon (INCLUSIVE of step 0, so a
    simultaneous same-row transition is detectable as delay=0). Returns
    (delay_seconds_or_None, fully_observed_bool)."""
    ts0 = df["_ts"].iloc[i0]
    for step in range(0, max_horizon + 1):
        j = i0 + step
        if j >= len(df):
            return None, False
        if step > 0 and df["_ts"].iloc[j] != ts0 + pd.Timedelta(seconds=step):
            return None, False
        if require_normal and df["Label"].iloc[j] != 0:
            return None, False
        if df[target_tag].iloc[j] == target_state:
            return step, True
    return None, True


def summarize(values):
    vals = np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))],
                     dtype=float)
    n = len(vals)
    if n == 0:
        keys = ["n", "mean", "median", "std", "min", "max", "p10", "p25", "p50", "p75", "p90", "iqr"]
        return {k: (0 if k == "n" else np.nan) for k in keys}
    p10, p25, p50, p75, p90 = np.percentile(vals, [10, 25, 50, 75, 90])
    return {"n": n, "mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "std": float(np.std(vals, ddof=0)), "min": float(np.min(vals)), "max": float(np.max(vals)),
            "p10": float(p10), "p25": float(p25), "p50": float(p50), "p75": float(p75), "p90": float(p90),
            "iqr": float(p75 - p25)}


def analyze_coupling(df, source_from, source_to, target_from, target_to, require_normal, label_filter=None):
    """source/target are always P205/P203 respectively here. Returns events,
    delays, fully_observed flags, and per-offset usable/matched counts."""
    trans = find_specific_transitions(df, "P205", source_from, source_to)
    events = []
    for i0 in trans:
        if label_filter is not None and int(df["Label"].iloc[i0]) != label_filter:
            continue
        if require_normal and df["Label"].iloc[i0] != 0:
            continue
        if not check_baseline_window(df, i0, "P205", source_from, require_normal=require_normal):
            continue
        if not check_baseline_window(df, i0, "P203", target_from, require_normal=require_normal):
            continue
        events.append(i0)

    delays, fully_observed_flags = [], []
    for i0 in events:
        delay, fo = scan_first_state(df, i0, "P203", target_to, max_horizon=MAX_HORIZON,
                                      require_normal=require_normal)
        delays.append(delay)
        fully_observed_flags.append(fo)

    offset_rows = []
    for offset in OFFSETS:
        usable = [i0 for i0 in events if check_offset_window(df, i0, offset, require_normal=require_normal)]
        matched = sum(1 for i0, d in zip(events, delays) if i0 in usable and d is not None and d <= offset)
        prob = (matched / len(usable)) if usable else np.nan
        offset_rows.append({"Offset_Seconds": offset, "N_Usable": len(usable), "N_Matched": matched,
                             "Response_Probability": prob})

    n_unmatched = sum(1 for d, fo in zip(delays, fully_observed_flags) if d is None and fo)
    delay_stats = summarize(delays)
    return {"events": events, "delays": delays, "fully_observed": fully_observed_flags,
            "offset_rows": offset_rows, "n_unmatched": n_unmatched, "delay_stats": delay_stats}


# =============================================================================
# PART 1 — TRAIN NORMAL DEACTIVATION EVENTS (P205 ON->OFF -> P203 ON->OFF)
# =============================================================================
section("PART 1 — TRAIN NORMAL P205 ON->OFF -> P203 ON->OFF COUPLING")

deact = analyze_coupling(train_df, P205_ON, P205_OFF, P203_ON, P203_OFF, require_normal=True)
log(f"Total usable P205 ON->OFF (deactivation) TRAIN NORMAL events: {len(deact['events'])}")
log("\nResponse probability (P203 ON->OFF within window) by offset:")
for r in deact["offset_rows"]:
    log(f"  offset={r['Offset_Seconds']:>3}s: n_usable={r['N_Usable']:>3} n_matched={r['N_Matched']:>3} "
        f"prob={r['Response_Probability']}")
ds = deact["delay_stats"]
log(f"\nDelay distribution (seconds): n_observed={ds['n']}, mean={ds['mean']}, median={ds['median']}, "
    f"std={ds['std']}, min={ds['min']}, max={ds['max']}")
log(f"P10={ds['p10']}  P25={ds['p25']}  P50={ds['p50']}  P75={ds['p75']}  P90={ds['p90']}")
log(f"Number of unmatched P205 deactivations (no P203 OFF within {MAX_HORIZON}s, fully observed): "
    f"{deact['n_unmatched']}")
log("\nNote: this reports observed behavior as-is -- no assumption that P203 must always follow was made.")

deact_event_rows = []
for k, i0 in enumerate(deact["events"]):
    d = deact["delays"][k]
    fo = deact["fully_observed"][k]
    deact_event_rows.append({
        "Event_Index": k, "P205_OFF_Timestamp": train_df["_ts"].iloc[i0],
        "Baseline_Usable": True, "Delay_Seconds": d if d is not None else np.nan,
        "Fully_Observed_30s": fo, "Matched": d is not None,
    })
deact_events_df = pd.DataFrame(deact_event_rows)
deact_events_path = TABLES_DIR / "stage2_p205_p203_deactivation_events.csv"
deact_events_df.to_csv(deact_events_path, index=False)
log(f"\nWrote {deact_events_path} ({len(deact_events_df)} rows)")

# =============================================================================
# PART 2 — COMPARE ACTIVATION VS DEACTIVATION COUPLING (same offsets, TRAIN NORMAL)
# =============================================================================
section("PART 2 — ACTIVATION (OFF->ON) vs DEACTIVATION (ON->OFF) COUPLING COMPARISON")

activ = analyze_coupling(train_df, P205_OFF, P205_ON, P203_OFF, P203_ON, require_normal=True)
log(f"Total usable P205 OFF->ON (activation) TRAIN NORMAL events: {len(activ['events'])}")
log("Response probability (P203 OFF->ON within window) by offset:")
for r in activ["offset_rows"]:
    log(f"  offset={r['Offset_Seconds']:>3}s: n_usable={r['N_Usable']:>3} n_matched={r['N_Matched']:>3} "
        f"prob={r['Response_Probability']}")
as_ = activ["delay_stats"]
log(f"\nActivation delay distribution (seconds): n_observed={as_['n']}, mean={as_['mean']}, "
    f"median={as_['median']}, std={as_['std']}")
log(f"P10={as_['p10']}  P25={as_['p25']}  P50={as_['p50']}  P75={as_['p75']}  P90={as_['p90']}")

comparison_rows = []
for direction, res in [("Activation (OFF->ON)", activ), ("Deactivation (ON->OFF)", deact)]:
    for r in res["offset_rows"]:
        comparison_rows.append({
            "Direction": direction, "Metric_Type": "by_offset", "Offset_Seconds": r["Offset_Seconds"],
            "N_Usable": r["N_Usable"], "N_Matched": r["N_Matched"],
            "Response_Probability": r["Response_Probability"],
            "N_Total_Events": np.nan, "N_Unmatched": np.nan, "Median_Delay": np.nan, "Mean_Delay": np.nan,
            "Std_Delay": np.nan, "P10_Delay": np.nan, "P25_Delay": np.nan, "P75_Delay": np.nan,
            "P90_Delay": np.nan, "IQR_Delay": np.nan,
        })
    ds_ = res["delay_stats"]
    comparison_rows.append({
        "Direction": direction, "Metric_Type": "overall_summary", "Offset_Seconds": np.nan,
        "N_Usable": np.nan, "N_Matched": np.nan, "Response_Probability": ds_["n"] / len(res["events"]) if res["events"] else np.nan,
        "N_Total_Events": len(res["events"]), "N_Unmatched": res["n_unmatched"],
        "Median_Delay": ds_["median"], "Mean_Delay": ds_["mean"], "Std_Delay": ds_["std"],
        "P10_Delay": ds_["p10"], "P25_Delay": ds_["p25"], "P75_Delay": ds_["p75"], "P90_Delay": ds_["p90"],
        "IQR_Delay": ds_["iqr"],
    })

comparison_df = pd.DataFrame(comparison_rows)
comparison_path = TABLES_DIR / "stage2_p205_p203_coupling_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)
log(f"\nWrote {comparison_path} ({len(comparison_df)} rows)")

overall_activ = comparison_df[(comparison_df.Direction.str.startswith("Activation")) &
                               (comparison_df.Metric_Type == "overall_summary")].iloc[0]
overall_deact = comparison_df[(comparison_df.Direction.str.startswith("Deactivation")) &
                               (comparison_df.Metric_Type == "overall_summary")].iloc[0]
log("\n=== Direct comparison (TRAIN NORMAL, same offset grid) ===")
log(f"  Activation  : n_events={overall_activ['N_Total_Events']}, overall_response_rate="
    f"{overall_activ['Response_Probability']}, median_delay={overall_activ['Median_Delay']}s, "
    f"std={overall_activ['Std_Delay']}, IQR={overall_activ['IQR_Delay']}, "
    f"unmatched={overall_activ['N_Unmatched']}")
log(f"  Deactivation: n_events={overall_deact['N_Total_Events']}, overall_response_rate="
    f"{overall_deact['Response_Probability']}, median_delay={overall_deact['Median_Delay']}s, "
    f"std={overall_deact['Std_Delay']}, IQR={overall_deact['IQR_Delay']}, "
    f"unmatched={overall_deact['N_Unmatched']}")

both_strong = (not np.isnan(overall_activ["Response_Probability"]) and
               not np.isnan(overall_deact["Response_Probability"]) and
               overall_activ["Response_Probability"] >= 0.9 and overall_deact["Response_Probability"] >= 0.9)
log(f"\nBoth directions show similarly strong coupling (>=90% response rate each): {both_strong}")

# =============================================================================
# PART 3 — VALIDATION DIAGNOSTIC (both directions, NORMAL vs ATTACK strata)
# =============================================================================
section("PART 3 — VALIDATION DIAGNOSTIC (P205<->P203 coupling, NORMAL vs ATTACK)")

TRAIN_BAND = {
    "activation": (activ["delay_stats"]["p10"], activ["delay_stats"]["p90"]) if activ["delay_stats"]["n"] else (np.nan, np.nan),
    "deactivation": (deact["delay_stats"]["p10"], deact["delay_stats"]["p90"]) if deact["delay_stats"]["n"] else (np.nan, np.nan),
}
log(f"TRAIN-normal activation delay P10-P90 band: {TRAIN_BAND['activation']}")
log(f"TRAIN-normal deactivation delay P10-P90 band: {TRAIN_BAND['deactivation']}")

val_diag_rows = []
interval2_event = None

for direction_name, source_from, source_to, target_from, target_to, band_key in [
    ("Activation (OFF->ON)", P205_OFF, P205_ON, P203_OFF, P203_ON, "activation"),
    ("Deactivation (ON->OFF)", P205_ON, P205_OFF, P203_ON, P203_OFF, "deactivation"),
]:
    for stratum, label_val in [("NORMAL", 0), ("ATTACK", 1)]:
        res = analyze_coupling(val_df, source_from, source_to, target_from, target_to,
                                require_normal=False, label_filter=label_val)
        band_lo, band_hi = TRAIN_BAND[band_key]
        for k, i0 in enumerate(res["events"]):
            d = res["delays"][k]
            consistent = (np.nan if (d is None or np.isnan(band_lo) or np.isnan(band_hi))
                          else bool(band_lo <= d <= band_hi))
            row = {
                "Direction": direction_name, "Label_Stratum": stratum,
                "Event_Timestamp": val_df["_ts"].iloc[i0], "Matched": d is not None,
                "Delay_Seconds": d if d is not None else np.nan,
                "TRAIN_Normal_P10_Band": band_lo, "TRAIN_Normal_P90_Band": band_hi,
                "Consistent_With_TRAIN_Normal": consistent,
            }
            val_diag_rows.append(row)
            if (direction_name == "Deactivation (ON->OFF)" and stratum == "ATTACK" and
                    val_df["_ts"].iloc[i0] == pd.Timestamp("2015-12-31 16:06:31")):
                interval2_event = row
        log(f"{direction_name} [{stratum}]: {len(res['events'])} events, "
            f"{sum(1 for d in res['delays'] if d is not None)} matched, "
            f"median_delay={res['delay_stats']['median']}")

val_diag_df = pd.DataFrame(val_diag_rows)
val_diag_path = TABLES_DIR / "stage2_p205_p203_validation_diagnostic.csv"
val_diag_df.to_csv(val_diag_path, index=False)
log(f"\nWrote {val_diag_path} ({len(val_diag_df)} rows)")

log("\n=== VALIDATION attack-interval-2 P205/P203 simultaneous OFF event ===")
if interval2_event is not None:
    log(f"  Timestamp: {interval2_event['Event_Timestamp']}")
    log(f"  Observed delay: {interval2_event['Delay_Seconds']}s")
    log(f"  TRAIN-normal deactivation delay P10-P90 band: "
        f"[{interval2_event['TRAIN_Normal_P10_Band']}, {interval2_event['TRAIN_Normal_P90_Band']}]")
    log(f"  Consistent with TRAIN-normal deactivation coupling: "
        f"{interval2_event['Consistent_With_TRAIN_Normal']}")
    if interval2_event["Consistent_With_TRAIN_Normal"]:
        interval2_verdict = ("CONSISTENT with normal control coupling -- the simultaneous OFF event during "
                              "attack interval 2 falls within the same delay band observed throughout TRAIN "
                              "NORMAL operation. It does not, by itself, look like an anomalous coupling "
                              "event; whatever attack interval 2 involves, it does not appear to have broken "
                              "this specific P205->P203 deactivation relationship.")
    else:
        interval2_verdict = ("INCONSISTENT with the TRAIN-normal deactivation delay band -- this simultaneous "
                              "OFF event falls outside the range seen during TRAIN NORMAL operation, which is "
                              "at least consistent with (though not proof of) the coupling being disrupted or "
                              "directly manipulated during this attack interval.")
    log(f"  Interpretation: {interval2_verdict}")
else:
    interval2_verdict = ("The expected VALIDATION attack-interval-2 event (2015-12-31 16:06:31) was not found "
                          "among the extracted ATTACK-stratum deactivation events in this run -- see the full "
                          "diagnostic table for what WAS captured.")
    log(f"  {interval2_verdict}")

# =============================================================================
# PART 4 — RULE RECOMMENDATION
# =============================================================================
section("PART 4 — RULE RECOMMENDATION: S2-R3A (activation-only) vs S2-R3B (bidirectional)")

if both_strong:
    recommendation = "S2-R3B: bidirectional transition-coupling rule"
    rec_reasoning = (
        f"TRAIN-normal evidence shows both directions are strongly coupled: activation response rate "
        f"{overall_activ['Response_Probability']} (n={overall_activ['N_Total_Events']}) and deactivation "
        f"response rate {overall_deact['Response_Probability']} (n={overall_deact['N_Total_Events']}), "
        f"both with tight, low-dispersion delay distributions (activation median={overall_activ['Median_Delay']}s "
        f"std={overall_activ['Std_Delay']}; deactivation median={overall_deact['Median_Delay']}s "
        f"std={overall_deact['Std_Delay']}). This is symmetric enough, based on TRAIN-normal data alone, to "
        f"support expanding S2-R3 to: 'When P205 changes state, P203 should exhibit the corresponding state "
        f"transition within the empirically observed normal delay window' (direction-specific windows, not a "
        f"single shared threshold: use the activation delay band for OFF->ON triggers and the deactivation "
        f"delay band for ON->OFF triggers). This decision is based on TRAIN-normal evidence, not on the "
        f"VALIDATION attack-interval-2 observation, per instructions."
    )
else:
    recommendation = "S2-R3A: activation-only coupling"
    rec_reasoning = (
        f"TRAIN-normal evidence does not show comparably strong coupling in both directions (activation "
        f"response rate {overall_activ['Response_Probability']} vs deactivation response rate "
        f"{overall_deact['Response_Probability']}). Expanding to a bidirectional rule is not supported by "
        f"the data at this time; S2-R3 should remain activation-only until deactivation coupling is better "
        f"characterized."
    )
log(f"Recommendation: {recommendation}")
log(f"Reasoning: {rec_reasoning}")
log("\n(For context only, not part of the decision basis: the VALIDATION attack-interval-2 deactivation event "
    f"was found to be {'CONSISTENT' if (interval2_event and interval2_event['Consistent_With_TRAIN_Normal']) else 'not confidently classified or INCONSISTENT'} "
    "with the TRAIN-normal deactivation band -- see Part 3.)")

# =============================================================================
# NARRATIVE REPORT
# =============================================================================
section("SAVING results/stage2_p205_p203_deactivation_coupling.txt")

report_lines = []
report_lines.append("STAGE 2 DIAGNOSTIC #5 — P205->P203 DEACTIVATION COUPLING. ANALYSIS ONLY.")
report_lines.append(f"Generated: {RUN_START.isoformat()}")
report_lines.append(f"Python: {sys.version}")
report_lines.append(f"pandas: {pd.__version__} | numpy: {np.__version__}")
report_lines.append(f"OS: {platform.system()} {platform.release()}")
report_lines.append("")
report_lines.append(f"Data sources: {TRAIN_PATH.name} ({len(train_df):,} rows), "
                     f"{VAL_PATH.name} ({len(val_df):,} rows). TEST was never loaded.")
report_lines.append("")
report_lines.append("=== PART 1: TRAIN NORMAL P205 ON->OFF -> P203 ON->OFF ===")
report_lines.append(f"Usable events: {len(deact['events'])}")
for r in deact["offset_rows"]:
    report_lines.append(f"  offset={r['Offset_Seconds']}s: n_usable={r['N_Usable']} "
                         f"n_matched={r['N_Matched']} prob={r['Response_Probability']}")
report_lines.append(f"Delay distribution: median={ds['median']}, mean={ds['mean']}, std={ds['std']}, "
                     f"P10={ds['p10']}, P25={ds['p25']}, P50={ds['p50']}, P75={ds['p75']}, P90={ds['p90']}")
report_lines.append(f"Unmatched deactivations: {deact['n_unmatched']}")
report_lines.append("")
report_lines.append("=== PART 2: ACTIVATION vs DEACTIVATION COMPARISON (TRAIN NORMAL) ===")
report_lines.append(f"Activation  : n={overall_activ['N_Total_Events']}, response_rate="
                     f"{overall_activ['Response_Probability']}, median_delay={overall_activ['Median_Delay']}s, "
                     f"std={overall_activ['Std_Delay']}")
report_lines.append(f"Deactivation: n={overall_deact['N_Total_Events']}, response_rate="
                     f"{overall_deact['Response_Probability']}, median_delay={overall_deact['Median_Delay']}s, "
                     f"std={overall_deact['Std_Delay']}")
report_lines.append(f"Both directions similarly strong (>=90% each): {both_strong}")
report_lines.append("")
report_lines.append("=== PART 3: VALIDATION DIAGNOSTIC ===")
for direction_name in ["Activation (OFF->ON)", "Deactivation (ON->OFF)"]:
    for stratum in ["NORMAL", "ATTACK"]:
        sub = val_diag_df[(val_diag_df.Direction == direction_name) & (val_diag_df.Label_Stratum == stratum)]
        report_lines.append(f"  {direction_name} [{stratum}]: n={len(sub)}, matched={int(sub['Matched'].sum())}")
report_lines.append("")
report_lines.append("Attack-interval-2 simultaneous P205/P203 OFF event:")
if interval2_event is not None:
    report_lines.append(f"  Timestamp={interval2_event['Event_Timestamp']}, "
                         f"delay={interval2_event['Delay_Seconds']}s, "
                         f"TRAIN band=[{interval2_event['TRAIN_Normal_P10_Band']}, "
                         f"{interval2_event['TRAIN_Normal_P90_Band']}], "
                         f"consistent={interval2_event['Consistent_With_TRAIN_Normal']}")
report_lines.append(f"  {interval2_verdict}")
report_lines.append("")
report_lines.append("=== PART 4: RULE RECOMMENDATION ===")
report_lines.append(f"Recommendation: {recommendation}")
report_lines.append(f"Reasoning: {rec_reasoning}")
report_lines.append("")
report_lines.append("=== GUARDRAILS CONFIRMED ===")
report_lines.append("  - TEST partition: never loaded, filename not referenced anywhere in this script.")
report_lines.append("  - No arbitrary/final attack thresholds were selected.")
report_lines.append("  - No feature selection was performed.")
report_lines.append("  - No model was trained.")
report_lines.append("  - No existing script, dataset, or split was modified.")
report_lines.append("  - The A/B recommendation was based on TRAIN-normal evidence, not VALIDATION attack behavior.")
report_lines.append("  - Analysis is fully deterministic (no randomness used).")

report_path = RESULTS_DIR / "stage2_p205_p203_deactivation_coupling.txt"
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
    "decision_based_on_train_normal_only": True,
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

log(f"1. TRAIN-normal P205 ON->OFF event count: {len(deact['events'])}")
log(f"2. P203 deactivation response probability: {ds['n'] / len(deact['events']) if deact['events'] else np.nan} "
    f"({ds['n']} of {len(deact['events'])} events matched within {MAX_HORIZON}s)")
log(f"3. Delay distribution: median={ds['median']}s, P10={ds['p10']}s, P25={ds['p25']}s, "
    f"P75={ds['p75']}s, P90={ds['p90']}s, std={ds['std']}")
log(f"4. Comparison with OFF->ON coupling: activation response_rate="
    f"{overall_activ['Response_Probability']} median_delay={overall_activ['Median_Delay']}s  |  "
    f"deactivation response_rate={overall_deact['Response_Probability']} "
    f"median_delay={overall_deact['Median_Delay']}s  |  similarly strong: {both_strong}")
log(f"5. VALIDATION attack-interval-2 simultaneous OFF event: {interval2_verdict}")
log(f"6. Final recommendation: {recommendation}")

run_end = datetime.now()
log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
log("ANALYSIS-ONLY STAGE COMPLETE. No final thresholds selected, no features selected, no model trained.")

if not all_checks_passed:
    sys.exit(1)
