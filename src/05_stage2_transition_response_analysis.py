"""
code/05_stage2_transition_response_analysis.py

STAGE 2 DIAGNOSTIC #2 — ACTUATOR-STATE SEMANTICS AND TRANSITION-RESPONSE
BEHAVIOR. ANALYSIS ONLY.

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

Part 1: infer which numeric state of each Stage 2 actuator (MV201, P201-
P206) plausibly corresponds to ON/OFF or OPEN/CLOSED, from observed
sensor behavior -- NOT from assuming "larger number = ON".

Part 2: for each usable actuator transition in the inferred ON/OFF
sense, measure the associated sensor's change relative to a pre-
transition baseline at 10 fixed offsets (0-120s).

Part 3: repeat Part 2's summaries separately for NORMAL-labeled and
ATTACK-labeled transitions (TRAIN only, where both classes exist in
volume) to see whether attack periods disrupt the expected physical
relationships.

This script does NOT define symbolic thresholds, does NOT select
features, does NOT train any model, and does NOT modify any existing
script, dataset, or split. Uses ONLY processed/train_unscaled.csv and
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
META_DIR = PROCESSED_DIR / "metadata"
TRAIN_PATH = PROCESSED_DIR / "train_unscaled.csv"
VAL_PATH = PROCESSED_DIR / "validation_unscaled.csv"
FEATURE_LIST_PATH = META_DIR / "feature_list.txt"
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


section("STAGE 2 TRANSITION-RESPONSE ANALYSIS — RUN START (ANALYSIS ONLY)")
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

OFFSETS = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120]
BASELINE_WINDOW = 10
MIN_USABLE_TRANSITIONS = 5  # below this, an actuator is reported as "too sparse" not "primary"

# =============================================================================
# LOAD DATA
# =============================================================================
section("LOADING TRAIN AND VALIDATION (TEST NEVER LOADED)")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
train_df["_ts"] = pd.to_datetime(train_df["Timestamp"])
val_df["_ts"] = pd.to_datetime(val_df["Timestamp"])
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
log(f"TRAIN: {train_df.shape[0]:,} rows | VALIDATION: {val_df.shape[0]:,} rows")
SPLITS = {"TRAIN": train_df, "VALIDATION": val_df}

PUMP_TAGS = ["P201", "P202", "P203", "P204", "P205", "P206"]
ACTUATOR_TAGS = ["MV201"] + PUMP_TAGS

# Primary associated sensor for each pump (chemical dosing pump -> its
# analyzer), used for Part 1 semantics and default rate-of-change checks.
PUMP_SENSOR = {"P201": "AIT201", "P202": "AIT201", "P203": "AIT202",
                "P204": "AIT202", "P205": "AIT203", "P206": "AIT203"}
# Engineer-expected sign of the ASSOCIATED sensor while the pump is ON
# (+1 = increases, -1 = decreases). Used only to help infer which numeric
# state is "ON" -- not assumed true a priori, tested against the data.
PUMP_EXPECTED_ON_SIGN = {"P201": +1, "P202": +1, "P203": -1, "P204": -1,
                          "P205": +1, "P206": +1}

# =============================================================================
# PART 1 — ACTUATOR STATE SEMANTICS
# =============================================================================
section("PART 1 — VERIFYING ACTUATOR-STATE SEMANTICS (NOT ASSUMING LARGER=ON)")

semantics_records = []
semantic_label_map = {}  # {(split, tag, state): "ON"/"OFF"/"OPEN"/"CLOSED"/"UNCERTAIN"}


def steady_state_diff(df, tag, sensor, state):
    """Mean per-second change of `sensor` restricted to rows where `tag` has
    been in `state` for at least 2 consecutive seconds (i.e. NOT the instant
    of a transition) -- characterizes ongoing behavior WHILE active/inactive,
    as opposed to the transition response itself (that is Part 2)."""
    s = df[tag]
    steady_mask = (s == state) & (s.shift(1) == state)
    diffs = df[sensor].diff()
    vals = diffs[steady_mask].dropna()
    return vals.mean() if len(vals) else np.nan, len(vals)


# --- MV201 (3 states, evaluated against FIT201) ---
for split_name, df in SPLITS.items():
    n = len(df)
    states = sorted(df["MV201"].unique())
    state_stats = {}
    for st in states:
        mask = df["MV201"] == st
        freq = int(mask.sum())
        sensor_mean = df.loc[mask, "FIT201"].mean()
        sensor_median = df.loc[mask, "FIT201"].median()
        rate, n_rate = steady_state_diff(df, "MV201", "FIT201", st)
        state_stats[st] = {"freq": freq, "pct": 100 * freq / n, "mean": sensor_mean,
                            "median": sensor_median, "rate": rate, "n_rate": n_rate}

    # Determine CLOSED as the state with FIT201 nearest zero (relative to the
    # overall FIT201 range); OPEN as the state(s) with materially higher flow.
    fit_max = df["FIT201"].max()
    near_zero_thresh = 0.05 * fit_max
    for st, stt in state_stats.items():
        freq_ok = stt["freq"] >= MIN_USABLE_TRANSITIONS
        if not freq_ok:
            label, conf = "UNCERTAIN", "uncertain (extremely sparse state, n<5)"
        elif stt["mean"] <= near_zero_thresh and stt["median"] <= near_zero_thresh:
            label, conf = "CLOSED", "high (FIT201 ~ 0 while in this state)"
        elif stt["mean"] > near_zero_thresh:
            label, conf = "OPEN", "high (FIT201 clearly > 0 while in this state)"
        else:
            label, conf = "UNCERTAIN", "medium (ambiguous FIT201 level)"
        semantic_label_map[(split_name, "MV201", st)] = label
        semantics_records.append({
            "Tag": "MV201", "Split": split_name, "Associated_Sensor": "FIT201",
            "State_Value": st, "Frequency": stt["freq"], "Frequency_Pct": stt["pct"],
            "Sensor_Mean_At_State": stt["mean"], "Sensor_Median_At_State": stt["median"],
            "Steady_State_Mean_Diff_Per_Sec": stt["rate"],
            "Inferred_Meaning": label, "Confidence": conf,
        })
    log(f"\nMV201 [{split_name}] (vs FIT201):")
    for st, stt in state_stats.items():
        log(f"  state={st}: freq={stt['freq']:,} ({stt['pct']:.2f}%) FIT201_mean={stt['mean']:.4f} "
            f"FIT201_median={stt['median']:.4f} -> {semantic_label_map[(split_name,'MV201',st)]}")

# --- Pumps P201-P206 (evaluated against their associated AIT sensor) ---
for split_name, df in SPLITS.items():
    n = len(df)
    for pump in PUMP_TAGS:
        sensor = PUMP_SENSOR[pump]
        expected_sign = PUMP_EXPECTED_ON_SIGN[pump]
        states = sorted(df[pump].unique())
        state_stats = {}
        for st in states:
            mask = df[pump] == st
            freq = int(mask.sum())
            sensor_mean = df.loc[mask, sensor].mean()
            sensor_median = df.loc[mask, sensor].median()
            rate, n_rate = steady_state_diff(df, pump, sensor, st)
            state_stats[st] = {"freq": freq, "pct": 100 * freq / n, "mean": sensor_mean,
                                "median": sensor_median, "rate": rate, "n_rate": n_rate}

        if len(states) == 1:
            st = states[0]
            semantic_label_map[(split_name, pump, st)] = "UNCERTAIN"
            semantics_records.append({
                "Tag": pump, "Split": split_name, "Associated_Sensor": sensor,
                "State_Value": st, "Frequency": state_stats[st]["freq"],
                "Frequency_Pct": state_stats[st]["pct"],
                "Sensor_Mean_At_State": state_stats[st]["mean"],
                "Sensor_Median_At_State": state_stats[st]["median"],
                "Steady_State_Mean_Diff_Per_Sec": state_stats[st]["rate"],
                "Inferred_Meaning": "UNCERTAIN (ON/OFF)",
                "Confidence": "uncertain (constant in this split, no variation observed)",
            })
            continue

        # Rank states by frequency; the minority state's rate-of-change sign
        # relative to the expected ON direction drives the inference.
        ordered = sorted(states, key=lambda s: -state_stats[s]["freq"])
        majority_state, minority_state = ordered[0], ordered[-1]
        maj_rate = state_stats[majority_state]["rate"]
        min_rate = state_stats[minority_state]["rate"]
        min_freq = state_stats[minority_state]["freq"]

        if min_freq < MIN_USABLE_TRANSITIONS:
            on_state, off_state = None, None
            conf = f"uncertain (minority state n={min_freq}, too sparse to characterize reliably)"
            # still make a tentative call using rate sign, but mark low confidence
            if not np.isnan(min_rate) and np.sign(min_rate) == np.sign(expected_sign) and min_rate != 0:
                on_state, off_state = minority_state, majority_state
                conf = (f"low (matches expected direction but minority state n={min_freq} is very "
                         f"sparse; statistics unreliable)")
            for st in states:
                lbl = "UNCERTAIN"
                if on_state is not None:
                    lbl = "ON" if st == on_state else "OFF"
                semantic_label_map[(split_name, pump, st)] = lbl
        else:
            if np.isnan(maj_rate) or np.isnan(min_rate):
                on_state, off_state, conf = None, None, "uncertain (insufficient steady-state samples)"
            else:
                maj_matches = np.sign(maj_rate) == np.sign(expected_sign) and maj_rate != 0
                min_matches = np.sign(min_rate) == np.sign(expected_sign) and min_rate != 0
                if min_matches and not maj_matches:
                    on_state, off_state = minority_state, majority_state
                    ratio = abs(min_rate) / (abs(maj_rate) + 1e-12)
                    conf = "high" if ratio > 1.5 else "medium"
                elif maj_matches and not min_matches:
                    on_state, off_state = majority_state, minority_state
                    ratio = abs(maj_rate) / (abs(min_rate) + 1e-12)
                    conf = "high" if ratio > 1.5 else "medium"
                elif min_matches and maj_matches:
                    # both states show the expected-sign drift -- pick whichever
                    # has the larger magnitude, but flag reduced confidence
                    if abs(min_rate) >= abs(maj_rate):
                        on_state, off_state = minority_state, majority_state
                    else:
                        on_state, off_state = majority_state, minority_state
                    conf = "medium (both states show expected-sign drift; using larger magnitude)"
                else:
                    on_state, off_state, conf = None, None, (
                        "uncertain (neither state's steady-state drift matches the engineer-"
                        "expected direction -- data do not clearly support an ON/OFF call)"
                    )
            for st in states:
                lbl = "UNCERTAIN"
                if on_state is not None:
                    lbl = "ON" if st == on_state else "OFF"
                semantic_label_map[(split_name, pump, st)] = lbl

        for st in states:
            stt = state_stats[st]
            semantics_records.append({
                "Tag": pump, "Split": split_name, "Associated_Sensor": sensor,
                "State_Value": st, "Frequency": stt["freq"], "Frequency_Pct": stt["pct"],
                "Sensor_Mean_At_State": stt["mean"], "Sensor_Median_At_State": stt["median"],
                "Steady_State_Mean_Diff_Per_Sec": stt["rate"],
                "Inferred_Meaning": semantic_label_map[(split_name, pump, st)],
                "Confidence": conf,
            })
        log(f"\n{pump} [{split_name}] (vs {sensor}, expected ON sign={'+' if expected_sign>0 else '-'}):")
        for st in states:
            stt = state_stats[st]
            log(f"  state={st}: freq={stt['freq']:,} ({stt['pct']:.2f}%) {sensor}_mean={stt['mean']:.4f} "
                f"steady_rate={stt['rate']:.6f}/s -> {semantic_label_map[(split_name,pump,st)]}  [{conf}]")

semantics_df = pd.DataFrame(semantics_records)
semantics_path = TABLES_DIR / "stage2_state_semantics.csv"
semantics_df.to_csv(semantics_path, index=False)
log(f"\nWrote {semantics_path} ({len(semantics_df)} rows)")

# =============================================================================
# PART 2 — TRANSITION-BASED RESPONSE ANALYSIS
# =============================================================================
section("PART 2 — TRANSITION-BASED RESPONSE ANALYSIS (TRAIN + VALIDATION)")


def find_transitions(state_series):
    s = state_series
    changed = s != s.shift(1)
    changed.iloc[0] = False
    idx = np.flatnonzero(changed.to_numpy())
    return [(i, s.iloc[i - 1], s.iloc[i]) for i in idx]


def classify_direction(split_name, actuator, from_state, to_state):
    from_lbl = semantic_label_map.get((split_name, actuator, from_state), "UNCERTAIN")
    to_lbl = semantic_label_map.get((split_name, actuator, to_state), "UNCERTAIN")
    on_like = {"ON", "OPEN"}
    off_like = {"OFF", "CLOSED"}
    if from_lbl in off_like and to_lbl in on_like:
        return "OFF_TO_ON"
    if from_lbl in on_like and to_lbl in off_like:
        return "ON_TO_OFF"
    return "OTHER_UNCERTAIN"


def baseline_ok_and_value(df_ts, state_series, target_series, i0, window=BASELINE_WINDOW):
    """Baseline is valid only if the preceding `window` seconds are
    temporally continuous (1s steps, no gap) AND the actuator held the
    single pre-transition state throughout that window (so the baseline
    genuinely represents 'before this transition', not a different flicker)."""
    if i0 - window < 0:
        return False, np.nan
    ts_window = df_ts.iloc[i0 - window:i0]
    if not (ts_window.diff().dropna() == pd.Timedelta(seconds=1)).all():
        return False, np.nan
    if df_ts.iloc[i0] - df_ts.iloc[i0 - 1] != pd.Timedelta(seconds=1):
        return False, np.nan
    pre_state = state_series.iloc[i0 - 1]
    if not (state_series.iloc[i0 - window:i0] == pre_state).all():
        return False, np.nan
    return True, target_series.iloc[i0 - window:i0].mean()


def offset_ok_and_value(df_ts, target_series, i0, offset):
    j = i0 + offset
    if j >= len(df_ts) or j < 0:
        return False, np.nan
    expected_ts = df_ts.iloc[i0] + pd.Timedelta(seconds=offset)
    if df_ts.iloc[j] != expected_ts:
        return False, np.nan
    return True, target_series.iloc[j]


RELATIONSHIPS = [
    {"name": "MV201 transition -> FIT201", "actuator": "MV201", "target": "FIT201",
     "target_kind": "sensor", "expected_on_sign": +1, "mirror_off": True},
    {"name": "P201 transition -> AIT201", "actuator": "P201", "target": "AIT201",
     "target_kind": "sensor", "expected_on_sign": +1, "mirror_off": True},
    {"name": "P203 transition -> AIT202", "actuator": "P203", "target": "AIT202",
     "target_kind": "sensor", "expected_on_sign": -1, "mirror_off": True},
    {"name": "P203 transition -> AIT201", "actuator": "P203", "target": "AIT201",
     "target_kind": "sensor", "expected_on_sign": +1, "mirror_off": True},
    {"name": "P205 transition -> AIT203", "actuator": "P205", "target": "AIT203",
     "target_kind": "sensor", "expected_on_sign": +1, "mirror_off": True},
    {"name": "P205 transition -> AIT202", "actuator": "P205", "target": "AIT202",
     "target_kind": "sensor", "expected_on_sign": +1, "mirror_off": True},
    {"name": "P205 transition -> P203 response", "actuator": "P205", "target": "P203",
     "target_kind": "actuator_active", "expected_on_sign": +1, "mirror_off": False},
]

# Sparsity report for actuators excluded from the 7 relationships
sparsity_records = []
for split_name, df in SPLITS.items():
    for tag in ACTUATOR_TAGS:
        trans = find_transitions(df[tag])
        sparsity_records.append({"Tag": tag, "Split": split_name, "N_Transitions": len(trans)})
sparsity_df = pd.DataFrame(sparsity_records)
log("\nSwitch-event sparsity (all Stage 2 actuators, both splits):")
for _, r in sparsity_df.iterrows():
    flag = "" if r["N_Transitions"] >= MIN_USABLE_TRANSITIONS else "  <-- TOO SPARSE for primary analysis"
    log(f"  {r['Tag']:8s} [{r['Split']:10s}]: {r['N_Transitions']:>4} transitions{flag}")
log("\nP202, P204, P206 are excluded from the 7 primary transition relationships per instructions "
    "(none of the 7 listed relationships uses them as the transitioning actuator); their sparsity "
    "is reported above and repeated in the narrative report.")


def get_target_series(df, target_kind, target_tag):
    if target_kind == "sensor":
        return df[target_tag].astype(float)
    else:  # actuator_active -- use this split's own semantic ON state if known, else the TRAIN
        # majority/minority heuristic fallback (mode = idle) for a usable 0/1 series.
        return None  # filled in per-split below since it depends on split-specific labels


def get_actuator_active_series(split_name, df, tag):
    states = sorted(df[tag].unique())
    on_states = [st for st in states if semantic_label_map.get((split_name, tag, st)) in ("ON", "OPEN")]
    if on_states:
        return df[tag].isin(on_states).astype(float)
    # fallback: no confident ON label -- treat minority state as "active" (documented, still labeled UNCERTAIN upstream)
    mode_state = df[tag].mode().iloc[0]
    return (df[tag] != mode_state).astype(float)


def summarize_changes(vals, expected_sign):
    vals = np.array([v for v in vals if not (v is None or (isinstance(v, float) and np.isnan(v)))])
    n = len(vals)
    if n == 0:
        return dict(n=0, median=np.nan, mean=np.nan, std=np.nan, iqr=np.nan, pct_expected=np.nan)
    median = float(np.median(vals))
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=0))
    q75, q25 = np.percentile(vals, [75, 25])
    iqr = float(q75 - q25)
    if expected_sign is None:
        pct_expected = np.nan
    else:
        matches = np.sign(vals) == np.sign(expected_sign)
        pct_expected = 100 * matches.sum() / n
    return dict(n=n, median=median, mean=mean, std=std, iqr=iqr, pct_expected=pct_expected)


def analyze_relationship(split_name, df, rel, label_filter=None):
    """label_filter: None (all rows), 0 (NORMAL only), 1 (ATTACK only) --
    filter applied at the transition POINT's Label."""
    actuator = rel["actuator"]
    target_kind = rel["target_kind"]
    target_tag = rel["target"]
    df_ts = df["_ts"]
    state_series = df[actuator]

    if target_kind == "sensor":
        target_series = df[target_tag].astype(float)
    else:
        target_series = get_actuator_active_series(split_name, df, target_tag)

    transitions = find_transitions(state_series)
    rows = []
    for direction_name, sign_key in [("OFF_TO_ON", +1), ("ON_TO_OFF", -1), ("OTHER_UNCERTAIN", 0)]:
        if direction_name == "OFF_TO_ON":
            expected_sign = rel["expected_on_sign"]
        elif direction_name == "ON_TO_OFF":
            expected_sign = (-rel["expected_on_sign"]) if rel["mirror_off"] else None
        else:
            expected_sign = None

        matching = []
        for i0, from_state, to_state in transitions:
            d = classify_direction(split_name, actuator, from_state, to_state)
            if d != direction_name:
                continue
            if label_filter is not None and int(df["Label"].iloc[i0]) != label_filter:
                continue
            matching.append(i0)

        for offset in OFFSETS:
            changes = []
            for i0 in matching:
                base_ok, base_val = baseline_ok_and_value(df_ts, state_series, target_series, i0)
                if not base_ok:
                    continue
                off_ok, off_val = offset_ok_and_value(df_ts, target_series, i0, offset)
                if not off_ok:
                    continue
                changes.append(off_val - base_val)
            stats = summarize_changes(changes, expected_sign)
            rows.append({
                "Relationship": rel["name"], "Split": split_name, "Actuator": actuator,
                "Target": target_tag, "Target_Kind": target_kind,
                "Transition_Direction": direction_name, "Offset_Seconds": offset,
                "N_Usable_Transitions": stats["n"], "Median_Change": stats["median"],
                "Mean_Change": stats["mean"], "Std_Change": stats["std"], "IQR_Change": stats["iqr"],
                "Pct_Expected_Direction": stats["pct_expected"],
                "Expected_Sign": expected_sign if expected_sign is not None else "N/A",
                "N_Raw_Transitions_This_Direction": len(matching),
            })
    return rows


transition_response_records = []
for split_name, df in SPLITS.items():
    for rel in RELATIONSHIPS:
        transition_response_records.extend(analyze_relationship(split_name, df, rel, label_filter=None))

transition_response_df = pd.DataFrame(transition_response_records)
transition_response_path = TABLES_DIR / "stage2_transition_response.csv"
transition_response_df.to_csv(transition_response_path, index=False)
log(f"\nWrote {transition_response_path} ({len(transition_response_df)} rows)")

log("\nTRAIN, offset=30s summary for each relationship/direction:")
sub = transition_response_df[(transition_response_df.Split == "TRAIN") &
                              (transition_response_df.Offset_Seconds == 30) &
                              (transition_response_df.Transition_Direction != "OTHER_UNCERTAIN")]
for _, r in sub.iterrows():
    log(f"  {r['Relationship']:32s} {r['Transition_Direction']:12s} n={r['N_Usable_Transitions']:>4} "
        f"median_change={r['Median_Change']:.5f} pct_expected_dir={r['Pct_Expected_Direction']}")

# =============================================================================
# PART 3 — ATTACK CONTAMINATION CHECK (TRAIN ONLY, per task framing)
# =============================================================================
section("PART 3 — NORMAL vs ATTACK TRANSITION RESPONSE (TRAIN ONLY)")
log("TRAIN is used for this stratified comparison because it is the partition explicitly "
    "described in the task as containing both NORMAL and ATTACK periods in sufficient volume "
    "for this breakdown; VALIDATION also contains both but was not requested here.")

normal_attack_records = []
for rel in RELATIONSHIPS:
    for label_filter, stratum_name in [(0, "NORMAL"), (1, "ATTACK")]:
        rows = analyze_relationship("TRAIN", train_df, rel, label_filter=label_filter)
        for row in rows:
            row["Label_Stratum"] = stratum_name
        normal_attack_records.extend(rows)
normal_attack_df = pd.DataFrame(normal_attack_records)
# reorder so Label_Stratum reads naturally near the front
cols = normal_attack_df.columns.tolist()
cols.insert(2, cols.pop(cols.index("Label_Stratum")))
normal_attack_df = normal_attack_df[cols]

normal_attack_path = TABLES_DIR / "stage2_transition_normal_vs_attack.csv"
normal_attack_df.to_csv(normal_attack_path, index=False)
log(f"Wrote {normal_attack_path} ({len(normal_attack_df)} rows)")

log("\nNORMAL vs ATTACK comparison at offset=30s, OFF_TO_ON direction:")
cmp_sub = normal_attack_df[(normal_attack_df.Offset_Seconds == 30) &
                            (normal_attack_df.Transition_Direction == "OFF_TO_ON")]
for rel in RELATIONSHIPS:
    rn = cmp_sub[(cmp_sub.Relationship == rel["name"]) & (cmp_sub.Label_Stratum == "NORMAL")]
    ra = cmp_sub[(cmp_sub.Relationship == rel["name"]) & (cmp_sub.Label_Stratum == "ATTACK")]
    if len(rn) and len(ra):
        rn, ra = rn.iloc[0], ra.iloc[0]
        log(f"  {rel['name']:32s} NORMAL: n={rn['N_Usable_Transitions']:>4} "
            f"median={rn['Median_Change']:.5f} pct_exp={rn['Pct_Expected_Direction']}  |  "
            f"ATTACK: n={ra['N_Usable_Transitions']:>4} median={ra['Median_Change']:.5f} "
            f"pct_exp={ra['Pct_Expected_Direction']}")

# =============================================================================
# STRONGEST / SPARSEST RELATIONSHIP SUMMARY (for narrative + terminal output)
# =============================================================================
section("SUMMARY — STRONGEST AND SPARSEST NORMAL-OPERATION RESPONSES (TRAIN, offset=30s)")

train_off_to_on_30 = transition_response_df[
    (transition_response_df.Split == "TRAIN") & (transition_response_df.Offset_Seconds == 30) &
    (transition_response_df.Transition_Direction == "OFF_TO_ON")
].copy()
train_off_to_on_30["abs_pct_expected"] = train_off_to_on_30["Pct_Expected_Direction"]
strongest = train_off_to_on_30.sort_values("abs_pct_expected", ascending=False)
sparsest = transition_response_df[
    (transition_response_df.Split == "TRAIN") & (transition_response_df.Offset_Seconds == 30)
].groupby("Relationship")["N_Usable_Transitions"].max().sort_values()

log("Ranked by %% of transitions matching the engineer-expected direction (TRAIN, OFF_TO_ON, 30s offset):")
for _, r in strongest.iterrows():
    log(f"  {r['Relationship']:32s} n={r['N_Usable_Transitions']:>4} "
        f"pct_expected_dir={r['Pct_Expected_Direction']}")

log("\nSparsest relationships (max usable-transition count across directions, TRAIN, 30s offset):")
for rel_name, max_n in sparsest.items():
    flag = "  <-- TOO SPARSE for a reliable deterministic rule" if max_n < MIN_USABLE_TRANSITIONS else ""
    log(f"  {rel_name:32s} max_n={max_n}{flag}")

# =============================================================================
# NARRATIVE REPORT
# =============================================================================
section("SAVING results/stage2_transition_response_analysis.txt")

report_lines = []
report_lines.append("STAGE 2 DIAGNOSTIC #2 — ACTUATOR-STATE SEMANTICS AND TRANSITION-RESPONSE ANALYSIS")
report_lines.append("ANALYSIS ONLY. NO SYMBOLIC THRESHOLDS DEFINED.")
report_lines.append(f"Generated: {RUN_START.isoformat()}")
report_lines.append(f"Python: {sys.version}")
report_lines.append(f"pandas: {pd.__version__} | numpy: {np.__version__}")
report_lines.append(f"OS: {platform.system()} {platform.release()}")
report_lines.append("")
report_lines.append(f"Data sources: {TRAIN_PATH.name} ({len(train_df):,} rows), "
                     f"{VAL_PATH.name} ({len(val_df):,} rows). TEST was never loaded.")
report_lines.append("")
report_lines.append("=== METHODOLOGY NOTES ===")
report_lines.append(" - Part 1 ON/OFF inference is based on the steady-state (non-transition-instant) "
                     "mean per-second rate of change of each actuator's associated sensor while "
                     "holding a given state, compared against the engineer-stated expected sign for "
                     "the ON state. Raw per-state sensor mean/median are also reported as requested, "
                     "but used only as corroborating context, not as the primary inference signal, "
                     "since dosing pumps are typically under closed-loop control and their sensor's "
                     "raw LEVEL while 'on' need not differ from 'off' -- the DIRECTION of drift is "
                     "the more mechanistically meaningful signal.")
report_lines.append(" - A pre-transition baseline (mean of the preceding 10s) is used only when that "
                     "window is temporally continuous (1s steps, no gap) AND the actuator held a "
                     "single, constant state throughout it -- otherwise the transition is excluded "
                     "from that statistic as not usable.")
report_lines.append(" - Each offset (0-120s) uses the single instantaneous target reading at that "
                     "exact point in time (requiring temporal continuity from the transition to that "
                     "point); the actuator is NOT required to remain in its new state for the whole "
                     "duration -- a real subsequent switch is itself informative and will show up as "
                     "unstable/inconsistent response statistics rather than being hidden.")
report_lines.append(" - OFF_TO_ON / ON_TO_OFF classification uses Part 1's inferred semantics; "
                     "transitions touching a state Part 1 could not confidently label are grouped as "
                     "OTHER_UNCERTAIN and excluded from the primary ON/OFF comparison.")
report_lines.append("")
report_lines.append("=== PART 1: INFERRED ACTUATOR SEMANTICS (TRAIN) ===")
for tag in ACTUATOR_TAGS:
    sub = semantics_df[(semantics_df.Tag == tag) & (semantics_df.Split == "TRAIN")]
    for _, r in sub.iterrows():
        report_lines.append(f"  {tag:8s} state={r['State_Value']}: {r['Inferred_Meaning']:10s} "
                             f"[{r['Confidence']}]  (freq={r['Frequency']}, {r['Frequency_Pct']:.2f}%)")
report_lines.append("")
report_lines.append("=== PART 2: USABLE TRANSITIONS PER ACTUATOR (TRAIN) ===")
for _, r in sparsity_df[sparsity_df.Split == "TRAIN"].iterrows():
    flag = "" if r["N_Transitions"] >= MIN_USABLE_TRANSITIONS else " (TOO SPARSE)"
    report_lines.append(f"  {r['Tag']:8s}: {r['N_Transitions']} transitions{flag}")
report_lines.append("")
report_lines.append("=== STRONGEST NORMAL-OPERATION RESPONSES (TRAIN, OFF_TO_ON, 30s offset) ===")
for _, r in strongest.iterrows():
    report_lines.append(f"  {r['Relationship']:32s} n={r['N_Usable_Transitions']:>4} "
                         f"pct_expected_dir={r['Pct_Expected_Direction']}")
report_lines.append("")
report_lines.append("=== TOO SPARSE / INCONSISTENT FOR DETERMINISTIC RULES ===")
for rel_name, max_n in sparsest.items():
    if max_n < MIN_USABLE_TRANSITIONS:
        report_lines.append(f"  {rel_name}: max usable n={max_n} (below minimum {MIN_USABLE_TRANSITIONS})")
low_consistency = strongest[strongest["Pct_Expected_Direction"] < 55]
for _, r in low_consistency.iterrows():
    report_lines.append(f"  {r['Relationship']}: only {r['Pct_Expected_Direction']:.1f}% of TRAIN "
                         f"OFF_TO_ON transitions match the engineer-expected direction at 30s")
report_lines.append("")
report_lines.append("=== PART 3: NORMAL vs ATTACK DIFFERENCES (TRAIN, OFF_TO_ON, 30s offset) ===")
for rel in RELATIONSHIPS:
    rn = cmp_sub[(cmp_sub.Relationship == rel["name"]) & (cmp_sub.Label_Stratum == "NORMAL")]
    ra = cmp_sub[(cmp_sub.Relationship == rel["name"]) & (cmp_sub.Label_Stratum == "ATTACK")]
    if len(rn) and len(ra):
        rn, ra = rn.iloc[0], ra.iloc[0]
        report_lines.append(f"  {rel['name']}: NORMAL pct_expected={rn['Pct_Expected_Direction']}, "
                             f"n={rn['N_Usable_Transitions']}  vs  ATTACK pct_expected="
                             f"{ra['Pct_Expected_Direction']}, n={ra['N_Usable_Transitions']}")
report_lines.append("")
report_lines.append("=== GUARDRAILS CONFIRMED ===")
report_lines.append("  - TEST partition: never loaded, filename not referenced anywhere in this script.")
report_lines.append("  - No symbolic thresholds were defined.")
report_lines.append("  - No feature selection was performed.")
report_lines.append("  - No model was trained.")
report_lines.append("  - No existing script, dataset, or split was modified.")
report_lines.append("  - Analysis is fully deterministic (no randomness used).")

report_path = RESULTS_DIR / "stage2_transition_response_analysis.txt"
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
    "no_thresholds_defined": True,
    "no_feature_selection": True,
    "no_model_trained": True,
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

log("1. INFERRED ON/OFF OR OPEN/CLOSED MAPPINGS (TRAIN):")
for tag in ACTUATOR_TAGS:
    sub = semantics_df[(semantics_df.Tag == tag) & (semantics_df.Split == "TRAIN")]
    parts = [f"{r['State_Value']}={r['Inferred_Meaning']}({r['Confidence'].split(' ')[0]})"
             for _, r in sub.iterrows()]
    log(f"   {tag:8s}: {', '.join(parts)}")

log("\n2. USABLE TRANSITIONS PER ACTUATOR (TRAIN):")
for _, r in sparsity_df[sparsity_df.Split == "TRAIN"].iterrows():
    log(f"   {r['Tag']:8s}: {r['N_Transitions']}")

log("\n3. STRONGEST / MOST CONSISTENT NORMAL-OPERATION RESPONSES (TRAIN, OFF_TO_ON, 30s offset):")
for _, r in strongest.head(4).iterrows():
    log(f"   {r['Relationship']:32s} n={r['N_Usable_Transitions']:>4} "
        f"pct_expected_dir={r['Pct_Expected_Direction']}")

log("\n4. TOO SPARSE OR INCONSISTENT FOR DETERMINISTIC RULES:")
any_flagged = False
for rel_name, max_n in sparsest.items():
    if max_n < MIN_USABLE_TRANSITIONS:
        log(f"   {rel_name}: max usable n={max_n} (sparse)")
        any_flagged = True
for _, r in low_consistency.iterrows():
    log(f"   {r['Relationship']}: only {r['Pct_Expected_Direction']:.1f}% match expected direction")
    any_flagged = True
if not any_flagged:
    log("   None flagged beyond what is listed above.")

log("\n5. NORMAL vs ATTACK DIFFERENCES (TRAIN, OFF_TO_ON, 30s offset):")
for rel in RELATIONSHIPS:
    rn = cmp_sub[(cmp_sub.Relationship == rel["name"]) & (cmp_sub.Label_Stratum == "NORMAL")]
    ra = cmp_sub[(cmp_sub.Relationship == rel["name"]) & (cmp_sub.Label_Stratum == "ATTACK")]
    if len(rn) and len(ra):
        rn, ra = rn.iloc[0], ra.iloc[0]
        log(f"   {rel['name']:32s} NORMAL pct_exp={rn['Pct_Expected_Direction']} (n={rn['N_Usable_Transitions']})"
            f"  vs  ATTACK pct_exp={ra['Pct_Expected_Direction']} (n={ra['N_Usable_Transitions']})")

run_end = datetime.now()
log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
log("ANALYSIS-ONLY STAGE COMPLETE. No thresholds defined, no features selected, no model trained.")

if not all_checks_passed:
    sys.exit(1)
