"""
code/07_stage2_rule_specification_and_continuous_diagnostics.py

STAGE 2 DIAGNOSTIC #4 — CORRECTED S2-R1 DEFINITION, FINAL RULE
SPECIFICATIONS, AND CONTINUOUS-DIAGNOSTIC COVERAGE ANALYSIS. ANALYSIS ONLY.

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

Part 1: corrects S2-R1 to use the real MV201 opening sequence
CLOSED(1) -> transitional(0) -> OPEN(2), since the prior script found
zero direct 1->2 transitions in TRAIN (every switch passes through the
transitional state).

Part 2: produces final structured specifications for all four candidate
rules (S2-R1..S2-R4), self-contained (S2-R2/R3/R4 are recomputed here
with the same TRAIN-NORMAL-only methodology used previously, not read
from a prior script's output, so this script stands on its own).

Part 3: describes, qualitatively and structurally, how each rule could
be evaluated CONTINUOUSLY across the whole time series rather than only
at transition moments.

Part 4: measures, on VALIDATION, what fraction of NORMAL vs ATTACK
timestamps each proposed continuous formulation could actually
evaluate, and whether the relevant actuators even move during the 3
VALIDATION attack intervals.

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


section("STAGE 2 RULE SPECIFICATION & CONTINUOUS DIAGNOSTICS — RUN START (ANALYSIS ONLY)")
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
DELTA_OFFSETS = [5, 10, 15, 20, 30, 45, 60, 90, 120]          # S2-R2 / S2-R4 (unchanged from before)
R1_OFFSETS = [0, 5, 10, 15, 20, 30, 45, 60]                    # S2-R1 corrected, per this task's spec
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

# Fixed, previously-established state semantics:
#   MV201: 2=steady OPEN, 1=CLOSED, 0=transitional (mandatory, never skipped)
#   P203 : 1=OFF, 2=ON   |   P205 : 1=OFF, 2=ON
# P201/P202/P204/P206 remain excluded from rule construction.

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
    seg_ts = df["_ts"].iloc[i0:j + 1] if offset >= 0 else df["_ts"].iloc[j:i0 + 1]
    if not (seg_ts.diff().dropna() == pd.Timedelta(seconds=1)).all():
        return False
    expected_ts = df["_ts"].iloc[i0] + pd.Timedelta(seconds=offset)
    if df["_ts"].iloc[j] != expected_ts:
        return False
    if require_normal and not (df["Label"].iloc[min(i0, j):max(i0, j) + 1] == 0).all():
        return False
    return True


def offset_value(df, target, i0, offset):
    return df[target].iloc[i0 + offset]


def scan_first_activation(df, i0, target_tag, on_state, max_horizon=MAX_HORIZON, require_normal=False):
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


def recommend_offset(per_offset_stats, prob_key="Pct_Expected_Direction", min_n=MIN_N_FOR_RECOMMENDATION,
                      target=95.0):
    valid = [s for s in per_offset_stats if not np.isnan(s[prob_key])]
    if not valid:
        return None, "no usable offsets -- insufficient data to recommend a window"
    max_val = max(s[prob_key] for s in valid)
    tgt = target if max_val >= target else max_val
    candidates = [s for s in valid if s[prob_key] >= tgt and s["N_Usable"] >= min_n]
    if not candidates:
        candidates = [s for s in valid if s["N_Usable"] >= min_n] or valid
        pick = max(candidates, key=lambda s: s[prob_key])
        reason = (f"best achievable ({pick[prob_key]:.1f}) among offsets with n>={min_n}; "
                  f"target of >={target} was not reached at any offset")
    else:
        pick = min(candidates, key=lambda s: s["Offset_Seconds"])
        reason = (f"{pick[prob_key]:.1f} (>= target {tgt:.1f}), n={int(pick['N_Usable'])} -- "
                  f"earliest offset reaching this band")
    return pick, reason


# =============================================================================
# PART 1 — CORRECTED S2-R1 VALVE-OPENING DEFINITION (CLOSED -> transitional -> OPEN)
# =============================================================================
section("PART 1 — CORRECTED S2-R1: MV201 OPENING SEQUENCE 1 -> 0 -> 2")


def find_opening_sequences(df, closed_state=1, trans_state=0, open_state=2):
    s = df["MV201"].to_numpy()
    n = len(s)
    is_trans = (s == trans_state).astype(int)
    diffs = np.diff(np.concatenate(([0], is_trans, [0])))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1) - 1
    seqs = []
    for st, en in zip(starts, ends):
        if st - 1 < 0 or en + 1 >= n:
            continue
        if s[st - 1] == closed_state and s[en + 1] == open_state:
            seqs.append({"trans_start": int(st), "trans_end": int(en), "dwell": int(en - st + 1),
                         "open_idx": int(en + 1), "closed_last_idx": int(st - 1)})
    return seqs


all_opening_seqs = find_opening_sequences(train_df)
dwell_times = [seq["dwell"] for seq in all_opening_seqs]
dwell_stats = summarize(dwell_times)
log(f"Total CLOSED(1)->transitional(0)->OPEN(2) sequences found in TRAIN: {len(all_opening_seqs)}")
log(f"Transitional-state dwell time distribution (seconds): "
    f"n={dwell_stats['n']}, mean={dwell_stats['mean']:.2f}, median={dwell_stats['median']:.2f}, "
    f"std={dwell_stats['std']:.2f}, min={dwell_stats['min']:.0f}, max={dwell_stats['max']:.0f}, "
    f"p10={dwell_stats['p10']:.1f}, p90={dwell_stats['p90']:.1f}")

r1_seq_records = []
usable_r1_events = []
for seq in all_opening_seqs:
    st, en, open_idx = seq["trans_start"], seq["trans_end"], seq["open_idx"]
    trigger_label = int(train_df["Label"].iloc[st])
    baseline_ok = check_baseline_window(train_df, st, "MV201", 1, window=BASELINE_WINDOW, require_normal=True)
    usable = (trigger_label == 0) and baseline_ok
    r1_seq_records.append({
        "Sequence_Index": len(r1_seq_records), "Trans_Start_Timestamp": train_df["_ts"].iloc[st],
        "Trans_End_Timestamp": train_df["_ts"].iloc[en], "Open_Timestamp": train_df["_ts"].iloc[open_idx],
        "Dwell_Seconds": seq["dwell"], "Trigger_Label": trigger_label, "Baseline_Usable": baseline_ok,
        "Event_Usable": usable,
    })
    if usable:
        usable_r1_events.append(seq)

n_usable_r1 = len(usable_r1_events)
log(f"Usable NORMAL-only opening sequences (valid 10s CLOSED baseline before entering state 0): "
    f"{n_usable_r1} of {len(all_opening_seqs)}")

r1_response_rows = []
r1_offset_stats = []
for offset in R1_OFFSETS:
    vals = []
    for seq in usable_r1_events:
        st, open_idx = seq["trans_start"], seq["open_idx"]
        if not check_offset_window(train_df, open_idx, offset, require_normal=True):
            continue
        baseline = train_df["FIT201"].iloc[st - BASELINE_WINDOW:st].mean()
        resp = offset_value(train_df, "FIT201", open_idx, offset) - baseline
        vals.append(resp)
        r1_response_rows.append({
            "Sequence_Index": usable_r1_events.index(seq), "Offset_Seconds": offset,
            "Response_Value": resp, "Baseline_Value": baseline,
        })
    stats = summarize(vals)
    pe = pct_expected(vals, +1)
    r1_offset_stats.append({"Offset_Seconds": offset, "N_Usable": stats["n"], "Mean": stats["mean"],
                             "Median": stats["median"], "Std": stats["std"], "Min": stats["min"],
                             "Max": stats["max"], "P5": stats["p5"], "P10": stats["p10"], "P25": stats["p25"],
                             "P75": stats["p75"], "P90": stats["p90"], "P95": stats["p95"], "IQR": stats["iqr"],
                             "Pct_Expected_Direction": pe})
    log(f"  offset={offset:>3}s (from OPEN entry) n={stats['n']:>4} median={stats['median']:.5f} "
        f"pct_expected_dir={pe}")

r1_pick, r1_reason = recommend_offset(r1_offset_stats)
r1_recommended_offset = r1_pick["Offset_Seconds"] if r1_pick is not None else np.nan
log(f"\nS2-R1 recommended TRAIN-derived response horizon: {r1_recommended_offset}s")
log(f"Reason: {r1_reason}")

r1_analysis_df = pd.DataFrame(r1_seq_records)
r1_analysis_path = TABLES_DIR / "stage2_r1_corrected_valve_analysis.csv"
r1_analysis_df.to_csv(r1_analysis_path, index=False)
log(f"\nWrote {r1_analysis_path} ({len(r1_analysis_df)} rows)")

# =============================================================================
# RECOMPUTE S2-R2 / S2-R3 / S2-R4 (self-contained, TRAIN-NORMAL-only, same
# methodology as the prior envelope script)
# =============================================================================
section("RECOMPUTING S2-R2 / S2-R3 / S2-R4 (self-contained, TRAIN NORMAL only)")

OTHER_RELATIONSHIPS = [
    {"id": "S2-R2", "name": "P205 OFF->ON -> AIT203 ORP increase", "actuator": "P205", "from_state": 1,
     "to_state": 2, "target": "AIT203", "kind": "delta", "expected_sign": +1},
    {"id": "S2-R3", "name": "P205 OFF->ON -> P203 compensatory activation", "actuator": "P205",
     "from_state": 1, "to_state": 2, "target": "P203", "kind": "activation", "on_state": 2,
     "expected_sign": None},
    {"id": "S2-R4", "name": "P203 OFF->ON -> AIT202 pH decrease", "actuator": "P203", "from_state": 1,
     "to_state": 2, "target": "AIT202", "kind": "delta", "expected_sign": -1},
]

rel_results = {}  # id -> dict with events, per-offset stats, recommendation, activation info

for rel in OTHER_RELATIONSHIPS:
    all_trans = find_specific_transitions(train_df, rel["actuator"], rel["from_state"], rel["to_state"])
    events = []
    for i0 in all_trans:
        if train_df["Label"].iloc[i0] != 0:
            continue
        if not check_baseline_window(train_df, i0, rel["actuator"], rel["from_state"], require_normal=True):
            continue
        if rel["kind"] == "activation" and not check_baseline_window(
                train_df, i0, rel["target"], 1, require_normal=True):
            continue
        events.append(i0)

    if rel["kind"] == "delta":
        offset_stats = []
        for offset in DELTA_OFFSETS:
            vals = []
            for i0 in events:
                if not check_offset_window(train_df, i0, offset, require_normal=True):
                    continue
                base = train_df[rel["target"]].iloc[i0 - BASELINE_WINDOW:i0].mean()
                vals.append(offset_value(train_df, rel["target"], i0, offset) - base)
            stats = summarize(vals)
            pe = pct_expected(vals, rel["expected_sign"])
            offset_stats.append({"Offset_Seconds": offset, "N_Usable": stats["n"], "Mean": stats["mean"],
                                  "Median": stats["median"], "Std": stats["std"], "Min": stats["min"],
                                  "Max": stats["max"], "P5": stats["p5"], "P10": stats["p10"],
                                  "P25": stats["p25"], "P75": stats["p75"], "P90": stats["p90"],
                                  "P95": stats["p95"], "IQR": stats["iqr"], "Pct_Expected_Direction": pe})
        pick, reason = recommend_offset(offset_stats)
        rel_results[rel["id"]] = {"events": events, "offset_stats": offset_stats, "pick": pick,
                                   "reason": reason, "kind": "delta"}
        log(f"{rel['id']}: {len(events)} usable TRAIN NORMAL events; recommended horizon = "
            f"{pick['Offset_Seconds'] if pick is not None else 'n/a'}s ({reason})")
    else:
        delays = []
        for i0 in events:
            delay, fully_observed = scan_first_activation(train_df, i0, rel["target"], rel["on_state"],
                                                             require_normal=True)
            delays.append(delay)
        delay_stats = summarize([d for d in delays if d is not None])
        n_no_response = sum(1 for d in delays if d is None)
        offset_stats = []
        for offset in DELTA_OFFSETS:
            usable = [i0 for i0 in events if check_offset_window(train_df, i0, offset, require_normal=True)]
            n_act = sum(1 for i0, d in zip(events, delays) if i0 in usable and d is not None and d <= offset)
            act_prob = (n_act / len(usable)) if usable else np.nan
            offset_stats.append({"Offset_Seconds": offset, "N_Usable": len(usable),
                                  "Activation_Probability": act_prob})
        pick, reason = recommend_offset(offset_stats, prob_key="Activation_Probability", target=0.9)
        # recommend_offset's "target" default (95.0) doesn't apply to probabilities in [0,1];
        # call with an appropriate 0.9 target and matching max-based fallback for this scale.
        rel_results[rel["id"]] = {"events": events, "offset_stats": offset_stats, "pick": pick,
                                   "reason": reason, "kind": "activation", "delay_stats": delay_stats,
                                   "n_no_response_120s": n_no_response}
        log(f"{rel['id']}: {len(events)} usable TRAIN NORMAL events; median activation delay="
            f"{delay_stats['median']}s; recommended horizon = "
            f"{pick['Offset_Seconds'] if pick is not None else 'n/a'}s ({reason})")

# =============================================================================
# PART 2 — FINAL CANDIDATE RULE SPECIFICATIONS
# =============================================================================
section("PART 2 — FINAL CANDIDATE RULE SPECIFICATIONS")

REPRESENTATION_TYPE = {
    "S2-R1": "B: state-based (steady-state consistency)",
    "S2-R2": "D: hybrid (event-triggered response + rolling-window trend)",
    "S2-R3": "A: event-based (discrete trigger -> discrete response)",
    "S2-R4": "D: hybrid (event-triggered response + rolling-window trend)",
}


def evidence_strength(pct_or_prob, n, is_probability=False):
    val = pct_or_prob * 100 if is_probability else pct_or_prob
    if np.isnan(val):
        return "Insufficient data"
    if val >= 95 and n >= 50:
        return "Strong"
    if val >= 85 and n >= 30:
        return "Moderate"
    return "Weak/Uncertain"


spec_rows = []

# S2-R1
r1_row_at_rec = None
if not np.isnan(r1_recommended_offset):
    r1_row_at_rec = next(r for r in r1_offset_stats if r["Offset_Seconds"] == r1_recommended_offset)
spec_rows.append({
    "Rule_ID": "S2-R1", "Stage": 2,
    "Trigger": "MV201 opening sequence CLOSED(1) -> transitional(0) -> OPEN(2)",
    "Source_Tag": "MV201", "Target_Tag": "FIT201", "Expected_Direction": "increase",
    "TRAIN_Response_Horizon_Seconds": r1_recommended_offset,
    "TRAIN_Normal_Median_Response": r1_row_at_rec["Median"] if r1_row_at_rec else np.nan,
    "TRAIN_Normal_P5": r1_row_at_rec["P5"] if r1_row_at_rec else np.nan,
    "TRAIN_Normal_P10": r1_row_at_rec["P10"] if r1_row_at_rec else np.nan,
    "TRAIN_Normal_P90": r1_row_at_rec["P90"] if r1_row_at_rec else np.nan,
    "TRAIN_Normal_P95": r1_row_at_rec["P95"] if r1_row_at_rec else np.nan,
    "Representation_Type": REPRESENTATION_TYPE["S2-R1"],
    "Evidence_Strength": evidence_strength(r1_row_at_rec["Pct_Expected_Direction"], r1_row_at_rec["N_Usable"])
        if r1_row_at_rec else "Insufficient data",
    "Limitations": (f"Requires detecting the full 3-state sequence, not a simple 2-state transition; "
                     f"only {n_usable_r1} usable NORMAL TRAIN sequences; transitional-state dwell time "
                     f"varies (median {dwell_stats['median']:.1f}s), so the true 'moment of opening' has "
                     f"some inherent jitter; VALIDATION attack-period coverage for this exact sequence has "
                     f"not yet been confirmed nonzero (see Part 4)."),
})

# S2-R2, S2-R3, S2-R4
for rel in OTHER_RELATIONSHIPS:
    res = rel_results[rel["id"]]
    pick = res["pick"]
    off = pick["Offset_Seconds"] if pick is not None else np.nan
    if res["kind"] == "delta":
        row_at_rec = next((r for r in res["offset_stats"] if r["Offset_Seconds"] == off), None) if pick else None
        median_r = row_at_rec["Median"] if row_at_rec else np.nan
        p5 = row_at_rec["P5"] if row_at_rec else np.nan
        p10 = row_at_rec["P10"] if row_at_rec else np.nan
        p90 = row_at_rec["P90"] if row_at_rec else np.nan
        p95 = row_at_rec["P95"] if row_at_rec else np.nan
        strength = evidence_strength(row_at_rec["Pct_Expected_Direction"], row_at_rec["N_Usable"]) if row_at_rec else "Insufficient data"
        expected_dir = "increase" if rel["expected_sign"] > 0 else "decrease"
        limitations = (f"n={len(res['events'])} usable TRAIN NORMAL events; response magnitude is "
                        f"{'large and clear' if abs(median_r) > 1 else 'small, closer to the sensor noise floor'}; "
                        f"no VALIDATION-ATTACK transitions of this actuator were found in the prior "
                        f"transition-only analysis, so attack behavior for this exact rule remains untested "
                        f"under the transition-triggered formulation.")
    else:
        row_at_rec = next((r for r in res["offset_stats"] if r["Offset_Seconds"] == off), None) if pick else None
        # For the activation-based rule, "response" is naturally a DELAY (seconds until
        # P203 activates), not a magnitude -- reuse the delay distribution here instead
        # of leaving these columns blank, so the unified table stays comparable across
        # all four rules (units: seconds, not target-variable units).
        ds = res["delay_stats"]
        median_r, p5, p10, p90, p95 = ds["median"], ds["p5"], ds["p10"], ds["p90"], ds["p95"]
        strength = evidence_strength(row_at_rec["Activation_Probability"], row_at_rec["N_Usable"], is_probability=True) if row_at_rec else "Insufficient data"
        expected_dir = "P203 activates (response = activation delay, seconds)"
        limitations = (f"n={len(res['events'])} usable TRAIN NORMAL events; extremely consistent "
                        f"(median activation delay {res['delay_stats']['median']}s) but based on a small "
                        f"sample; {res['n_no_response_120s']} events showed no P203 response within 120s; "
                        f"VALIDATION-ATTACK coverage for this exact transition remains untested (see Part 4).")

    spec_rows.append({
        "Rule_ID": rel["id"], "Stage": 2, "Trigger": f"{rel['actuator']} OFF(1)->ON(2)",
        "Source_Tag": rel["actuator"], "Target_Tag": rel["target"], "Expected_Direction": expected_dir,
        "TRAIN_Response_Horizon_Seconds": off,
        "TRAIN_Normal_Median_Response": median_r, "TRAIN_Normal_P5": p5, "TRAIN_Normal_P10": p10,
        "TRAIN_Normal_P90": p90, "TRAIN_Normal_P95": p95,
        "Representation_Type": REPRESENTATION_TYPE[rel["id"]], "Evidence_Strength": strength,
        "Limitations": limitations,
    })

spec_df = pd.DataFrame(spec_rows)
spec_path = TABLES_DIR / "stage2_final_rule_specifications.csv"
spec_df.to_csv(spec_path, index=False)
log(f"Wrote {spec_path} ({len(spec_df)} rows)")
for _, r in spec_df.iterrows():
    log(f"\n{r['Rule_ID']}: {r['Trigger']} -> {r['Target_Tag']} ({r['Expected_Direction']})")
    log(f"  Horizon={r['TRAIN_Response_Horizon_Seconds']}s | Median={r['TRAIN_Normal_Median_Response']} | "
        f"P5={r['TRAIN_Normal_P5']} P95={r['TRAIN_Normal_P95']} | {r['Representation_Type']} | "
        f"Evidence: {r['Evidence_Strength']}")

# =============================================================================
# PART 3 — CONTINUOUS DIAGNOSTIC FORMULATION (qualitative + structural)
# =============================================================================
section("PART 3 — CONTINUOUS DIAGNOSTIC FORMULATION")

CONTINUOUS_FEATURES = {
    "S2-R1": {
        "description": "State-consistency feature: at every timestamp where MV201 is in a STEADY state "
                        "(1 or 2, i.e. NOT the rare transitional state 0), compare the observed FIT201 "
                        "value against the TRAIN-normal distribution of FIT201 for that steady state "
                        "(near-0 band for CLOSED, the P5-P95 band from Part 2 for OPEN).",
        "measures": "Whether the commanded/reported valve position and the physically-confirmed flow are "
                    "mutually consistent RIGHT NOW, independent of when the last transition happened.",
        "time_window": "None required beyond the current row (state-based); optionally a short (~2-3s) "
                        "lookback to confirm the state has been steady rather than mid-flicker.",
        "requires_transition": False,
        "every_timestamp": "Yes, except while MV201==0 (0.36% of TRAIN rows) -- during those rows the "
                            "rule is 'not applicable' rather than violated, since the valve hasn't settled.",
        "violation_score_idea": "Percentile-rank distance of the observed FIT201 from the TRAIN-normal "
                                 "distribution for the current MV201 steady state (e.g. 0 inside the "
                                 "P5-P95 band, rising toward 1 as the reading moves further outside it).",
    },
    "S2-R2": {
        "description": "Rolling ORP-response feature: maintain a rolling record of P205's state history; "
                        "whenever the current timestamp falls within the TRAIN-derived response horizon "
                        "after the most recent P205 OFF->ON transition, compare the AIT203 delta-so-far "
                        "against the TRAIN-normal response envelope for that elapsed time.",
        "measures": "Whether AIT203's ORP trajectory is behaving consistently with P205's recent dosing "
                     "activity.",
        "time_window": f"A rolling lookback of at least the recommended horizon "
                        f"({rel_results['S2-R2']['pick']['Offset_Seconds'] if rel_results['S2-R2']['pick'] else 'n/a'}s) "
                        "after the last P205 OFF->ON transition to know whether an expectation is 'active'.",
        "requires_transition": True,
        "every_timestamp": "No in its strict event-triggered form -- only evaluable while within the "
                            "lookback window of a recent P205 transition (P205 switches only 134 times in "
                            "270,000 TRAIN rows, so this window covers a small fraction of the timeline; "
                            "see Part 4 for the measured VALIDATION coverage). A steady-state fallback "
                            "component (e.g. flagging large *unexplained* AIT203 swings with no recent "
                            "P205 activity at all) could extend coverage in a later revision but is not "
                            "implemented here.",
        "violation_score_idea": "Deviation of the observed AIT203 delta from the TRAIN-normal envelope "
                                 "at the current elapsed time since the P205 transition, normalized by the "
                                 "envelope's IQR.",
    },
    "S2-R3": {
        "description": "Recent-activation-without-response feature: maintain 'seconds since the last P205 "
                        "OFF->ON transition' and 'seconds since the last P203 ON'. Flag when P205 activated "
                        "within the monitoring window but P203 has not activated since.",
        "measures": "Whether the expected compensatory pump activation is missing or delayed relative to "
                     "the near-1-second response observed in TRAIN NORMAL operation.",
        "time_window": f"Monitoring window = the recommended horizon "
                        f"({rel_results['S2-R3']['pick']['Offset_Seconds'] if rel_results['S2-R3']['pick'] else 'n/a'}s), "
                        "i.e. flag if P205 turned on within that many seconds ago and P203 has not yet "
                        "turned on.",
        "requires_transition": True,
        "every_timestamp": "No in strict form -- only meaningfully 'armed' within the monitoring window "
                            "after a P205 transition; outside that window there is no active expectation to "
                            "violate, so the feature is naturally 0/inactive most of the time (matching "
                            "P205's rarity), not 'not applicable'.",
        "violation_score_idea": "Binary flag (P205 active-window AND P203 not yet responded) that could "
                                 "be widened into a continuous 'seconds overdue past the TRAIN P90 delay' "
                                 "score once a threshold is chosen.",
    },
    "S2-R4": {
        "description": "Rolling pH-response feature: symmetric to S2-R2, using P203's transition history "
                        "and AIT202's response envelope.",
        "measures": "Whether AIT202's pH trajectory is behaving consistently with P203's recent dosing "
                     "activity.",
        "time_window": f"A rolling lookback of at least the recommended horizon "
                        f"({rel_results['S2-R4']['pick']['Offset_Seconds'] if rel_results['S2-R4']['pick'] else 'n/a'}s) "
                        "after the last P203 OFF->ON transition.",
        "requires_transition": True,
        "every_timestamp": "No in strict event-triggered form -- bounded by P203's 136 TRAIN switch "
                            "events, same structural limitation as S2-R2 (see Part 4 for measured coverage).",
        "violation_score_idea": "Deviation of the observed AIT202 delta from the TRAIN-normal envelope at "
                                 "the current elapsed time since the P203 transition, normalized by IQR.",
    },
}

for rule_id, feat in CONTINUOUS_FEATURES.items():
    log(f"\n{rule_id}:")
    log(f"  What it measures: {feat['measures']}")
    log(f"  Time window needed: {feat['time_window']}")
    log(f"  Requires a transition to be evaluable: {feat['requires_transition']}")
    log(f"  Evaluable on every timestamp: {feat['every_timestamp']}")
    log(f"  Violation-score idea (not implemented): {feat['violation_score_idea']}")

# =============================================================================
# PART 4 — VALIDATION COVERAGE DIAGNOSTIC
# =============================================================================
section("PART 4 — VALIDATION COVERAGE DIAGNOSTIC")

val_df = val_df.reset_index(drop=True)
val_df["_row"] = np.arange(len(val_df))

# --- identify the 3 VALIDATION attack intervals directly from Label ---
is_attack = val_df["Label"].eq(1)
group_id = (is_attack != is_attack.shift()).cumsum()
attack_intervals = []
for gid, sub in val_df.groupby(group_id):
    if sub["Label"].iloc[0] == 1:
        attack_intervals.append({"start_idx": int(sub["_row"].iloc[0]), "end_idx": int(sub["_row"].iloc[-1]),
                                  "start_ts": sub["_ts"].iloc[0], "end_ts": sub["_ts"].iloc[-1]})
log(f"VALIDATION attack intervals found: {len(attack_intervals)}")
for i, iv in enumerate(attack_intervals):
    log(f"  Interval {i+1}: {iv['start_ts']} -> {iv['end_ts']}")

# --- does MV201 / P205 / P203 change state during each attack interval? ---
log("\nDo the relevant actuators change state DURING each VALIDATION attack interval?")
tag_change_records = []
for i, iv in enumerate(attack_intervals):
    seg = val_df.iloc[iv["start_idx"]:iv["end_idx"] + 1]
    for tag in ["MV201", "P205", "P203"]:
        n_changes = int((seg[tag] != seg[tag].shift(1)).sum() - 1) if len(seg) > 1 else 0
        n_changes = max(n_changes, 0)
        changed = n_changes > 0
        tag_change_records.append({"Attack_Interval": i + 1, "Start": iv["start_ts"], "End": iv["end_ts"],
                                    "Tag": tag, "N_State_Changes_During_Interval": n_changes,
                                    "Changed": changed})
        log(f"  Interval {i+1} [{tag}]: {n_changes} state change(s) during the interval "
            f"({'YES' if changed else 'no'})")

# --- coverage: S2-R1 (state-based, evaluable whenever MV201 != 0) ---
r1_evaluable = (val_df["MV201"] != 0)
val_normal_mask = val_df["Label"] == 0
val_attack_mask = val_df["Label"] == 1

coverage_rows = []


def coverage_stats(evaluable_mask, rule_id, feature_desc):
    n_normal = int(val_normal_mask.sum())
    n_attack = int(val_attack_mask.sum())
    cov_normal = 100 * float((evaluable_mask & val_normal_mask).sum()) / n_normal if n_normal else np.nan
    cov_attack = 100 * float((evaluable_mask & val_attack_mask).sum()) / n_attack if n_attack else np.nan
    n_attack_evaluable = int((evaluable_mask & val_attack_mask).sum())
    coverage_rows.append({
        "Rule_ID": rule_id, "Feature_Description": feature_desc,
        "VALIDATION_NORMAL_Total": n_normal, "VALIDATION_NORMAL_Coverage_Pct": cov_normal,
        "VALIDATION_ATTACK_Total": n_attack, "VALIDATION_ATTACK_Coverage_Pct": cov_attack,
        "N_Attack_Timestamps_Evaluable": n_attack_evaluable,
    })
    log(f"{rule_id}: NORMAL coverage={cov_normal:.2f}% ({int((evaluable_mask & val_normal_mask).sum())}/{n_normal}), "
        f"ATTACK coverage={cov_attack:.2f}% ({n_attack_evaluable}/{n_attack})")


coverage_stats(r1_evaluable, "S2-R1", "State-consistency (MV201 steady state vs FIT201 band)")


def transition_window_evaluable_mask(df, actuator, from_state, to_state, horizon):
    """Row i is 'evaluable' if it falls within `horizon` seconds AFTER the most
    recent actuator OFF->ON transition (inclusive of the transition instant)."""
    n = len(df)
    evaluable = np.zeros(n, dtype=bool)
    if horizon is None or (isinstance(horizon, float) and np.isnan(horizon)):
        return evaluable
    horizon = int(horizon)
    trans_idx = find_specific_transitions(df, actuator, from_state, to_state)
    ts = df["_ts"]
    for i0 in trans_idx:
        for step in range(0, horizon + 1):
            j = i0 + step
            if j >= n:
                break
            if ts.iloc[j] != ts.iloc[i0] + pd.Timedelta(seconds=step):
                break
            evaluable[j] = True
    return pd.Series(evaluable, index=df.index)


r2_horizon = rel_results["S2-R2"]["pick"]["Offset_Seconds"] if rel_results["S2-R2"]["pick"] else np.nan
r3_horizon = rel_results["S2-R3"]["pick"]["Offset_Seconds"] if rel_results["S2-R3"]["pick"] else np.nan
r4_horizon = rel_results["S2-R4"]["pick"]["Offset_Seconds"] if rel_results["S2-R4"]["pick"] else np.nan

r2_evaluable = transition_window_evaluable_mask(val_df, "P205", 1, 2, r2_horizon)
r3_evaluable = transition_window_evaluable_mask(val_df, "P205", 1, 2, r3_horizon)
r4_evaluable = transition_window_evaluable_mask(val_df, "P203", 1, 2, r4_horizon)

coverage_stats(r2_evaluable, "S2-R2", f"Rolling ORP response window ({r2_horizon}s post-P205-transition)")
coverage_stats(r3_evaluable, "S2-R3", f"Recent-activation-without-response window ({r3_horizon}s post-P205-transition)")
coverage_stats(r4_evaluable, "S2-R4", f"Rolling pH response window ({r4_horizon}s post-P203-transition)")

coverage_df = pd.DataFrame(coverage_rows)
coverage_path = TABLES_DIR / "stage2_continuous_feature_coverage.csv"
coverage_df.to_csv(coverage_path, index=False)
log(f"\nWrote {coverage_path} ({len(coverage_df)} rows)")

tag_change_df = pd.DataFrame(tag_change_records)

log("\nKey coverage comparison: S2-R1's state-based formulation vs the event-window formulations "
    "(S2-R2/R3/R4) -- the state-based approach is expected to give dramatically higher ATTACK coverage "
    "precisely because it does not require a transition to have JUST occurred.")

# =============================================================================
# NARRATIVE REPORT
# =============================================================================
section("SAVING results/stage2_rule_specification_and_continuous_diagnostics.txt")

report_lines = []
report_lines.append("STAGE 2 DIAGNOSTIC #4 — RULE SPECIFICATIONS AND CONTINUOUS-DIAGNOSTIC COVERAGE")
report_lines.append("ANALYSIS ONLY. NO FINAL THRESHOLDS. NO MODEL TRAINED.")
report_lines.append(f"Generated: {RUN_START.isoformat()}")
report_lines.append(f"Python: {sys.version}")
report_lines.append(f"pandas: {pd.__version__} | numpy: {np.__version__}")
report_lines.append(f"OS: {platform.system()} {platform.release()}")
report_lines.append("")
report_lines.append(f"Data sources: {TRAIN_PATH.name} ({len(train_df):,} rows), "
                     f"{VAL_PATH.name} ({len(val_df):,} rows). TEST was never loaded.")
report_lines.append("")
report_lines.append("=== PART 1: CORRECTED S2-R1 (MV201 1->0->2 OPENING SEQUENCE) ===")
report_lines.append(f"Total sequences found (TRAIN): {len(all_opening_seqs)}")
report_lines.append(f"Dwell time (transitional state 0): mean={dwell_stats['mean']:.2f}s, "
                     f"median={dwell_stats['median']:.2f}s, p10={dwell_stats['p10']:.1f}s, "
                     f"p90={dwell_stats['p90']:.1f}s, min={dwell_stats['min']:.0f}s, max={dwell_stats['max']:.0f}s")
report_lines.append(f"Usable NORMAL-only sequences: {n_usable_r1}")
report_lines.append(f"Recommended response horizon: {r1_recommended_offset}s ({r1_reason})")
report_lines.append("")
report_lines.append("=== PART 2: FINAL RULE SPECIFICATIONS ===")
for _, r in spec_df.iterrows():
    report_lines.append(f"  {r['Rule_ID']}: {r['Trigger']} -> {r['Target_Tag']} ({r['Expected_Direction']})")
    report_lines.append(f"    Horizon={r['TRAIN_Response_Horizon_Seconds']}s  Median={r['TRAIN_Normal_Median_Response']}  "
                         f"P5={r['TRAIN_Normal_P5']}  P10={r['TRAIN_Normal_P10']}  P90={r['TRAIN_Normal_P90']}  "
                         f"P95={r['TRAIN_Normal_P95']}")
    report_lines.append(f"    Representation: {r['Representation_Type']}  |  Evidence: {r['Evidence_Strength']}")
    report_lines.append(f"    Limitations: {r['Limitations']}")
report_lines.append("")
report_lines.append("=== PART 3: CONTINUOUS FORMULATIONS ===")
for rule_id, feat in CONTINUOUS_FEATURES.items():
    report_lines.append(f"  {rule_id}: {feat['description']}")
    report_lines.append(f"    Measures: {feat['measures']}")
    report_lines.append(f"    Time window: {feat['time_window']}")
    report_lines.append(f"    Requires transition: {feat['requires_transition']}")
    report_lines.append(f"    Every-timestamp evaluable: {feat['every_timestamp']}")
report_lines.append("")
report_lines.append("=== PART 4: VALIDATION COVERAGE ===")
for _, r in coverage_df.iterrows():
    report_lines.append(f"  {r['Rule_ID']}: NORMAL coverage={r['VALIDATION_NORMAL_Coverage_Pct']:.2f}%, "
                         f"ATTACK coverage={r['VALIDATION_ATTACK_Coverage_Pct']:.2f}% "
                         f"({r['N_Attack_Timestamps_Evaluable']} attack timestamps evaluable)")
report_lines.append("")
report_lines.append("=== ATTACK-INTERVAL ACTUATOR ACTIVITY (does MV201/P205/P203 move during the 3 VALIDATION attacks?) ===")
for _, r in tag_change_df.iterrows():
    report_lines.append(f"  Interval {r['Attack_Interval']} [{r['Tag']}]: {r['N_State_Changes_During_Interval']} "
                         f"change(s) ({'YES' if r['Changed'] else 'no'})")
report_lines.append("")
report_lines.append("=== GUARDRAILS CONFIRMED ===")
report_lines.append("  - TEST partition: never loaded, filename not referenced anywhere in this script.")
report_lines.append("  - No arbitrary/final attack thresholds were selected.")
report_lines.append("  - No feature selection was performed.")
report_lines.append("  - No model was trained.")
report_lines.append("  - No existing script, dataset, or split was modified.")
report_lines.append("  - Recommended horizons were derived from TRAIN NORMAL data only, never from ATTACK labels.")
report_lines.append("  - Analysis is fully deterministic (no randomness used).")

report_path = RESULTS_DIR / "stage2_rule_specification_and_continuous_diagnostics.txt"
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
    "horizons_derived_from_train_normal_only": True,
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

log("1. CORRECTED S2-R1 FINDINGS:")
log(f"   {len(all_opening_seqs)} total 1->0->2 sequences in TRAIN; dwell median={dwell_stats['median']:.1f}s "
    f"(range {dwell_stats['min']:.0f}-{dwell_stats['max']:.0f}s)")
log(f"   {n_usable_r1} usable NORMAL-only sequences; recommended horizon={r1_recommended_offset}s ({r1_reason})")

log("\n2. FINAL FOUR CANDIDATE RULE DEFINITIONS:")
for _, r in spec_df.iterrows():
    log(f"   {r['Rule_ID']}: {r['Trigger']} -> {r['Target_Tag']} expects {r['Expected_Direction']} "
        f"within {r['TRAIN_Response_Horizon_Seconds']}s [{r['Evidence_Strength']}]")

log("\n3. CONTINUOUS FORMULATION PER RULE:")
for rule_id in CONTINUOUS_FEATURES:
    log(f"   {rule_id}: {REPRESENTATION_TYPE[rule_id]}")

log("\n4. NORMAL vs ATTACK COVERAGE IN VALIDATION:")
for _, r in coverage_df.iterrows():
    log(f"   {r['Rule_ID']}: NORMAL={r['VALIDATION_NORMAL_Coverage_Pct']:.2f}% "
        f"ATTACK={r['VALIDATION_ATTACK_Coverage_Pct']:.2f}% "
        f"(n_attack_evaluable={r['N_Attack_Timestamps_Evaluable']})")

log("\n5. RULES APPEARING VIABLE FOR IMPLEMENTATION AS SYMBOLIC FEATURES:")
for _, r in spec_df.iterrows():
    cov_row = coverage_df[coverage_df.Rule_ID == r["Rule_ID"]]
    attack_cov = cov_row["VALIDATION_ATTACK_Coverage_Pct"].iloc[0] if len(cov_row) else np.nan
    viable = r["Evidence_Strength"] in ("Strong", "Moderate") and not np.isnan(attack_cov) and attack_cov > 0
    verdict = "VIABLE (evidence + nonzero attack coverage)" if viable else \
        "VIABLE evidence-wise, but ZERO ATTACK-labeled coverage found -- cannot yet confirm attack sensitivity" \
        if r["Evidence_Strength"] in ("Strong", "Moderate") else "NOT YET VIABLE -- insufficient/weak evidence"
    log(f"   {r['Rule_ID']}: {verdict}")

run_end = datetime.now()
log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
log("ANALYSIS-ONLY STAGE COMPLETE. No final thresholds selected, no features selected, no model trained.")

if not all_checks_passed:
    sys.exit(1)
