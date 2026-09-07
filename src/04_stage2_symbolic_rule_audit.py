"""
code/04_stage2_symbolic_rule_audit.py

STAGE 2 SYMBOLIC RULE AUDIT — ANALYSIS ONLY.

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

Purpose: characterize the Stage 2 SWaT tags (MV201, FIT201, P201-P206,
AIT201, AIT202, AIT203) needed to later convert engineering knowledge
into deterministic symbolic features. This script only describes the
variables and their timing behavior -- it does NOT define symbolic
thresholds, does NOT select features, does NOT train any model, and
does NOT modify any existing script, dataset, or split.

Uses ONLY processed/train_unscaled.csv and processed/validation_unscaled.csv
(the already-approved TRAIN and VALIDATION partitions, in original
physical units so states/thresholds stay interpretable). TEST is never
loaded -- its filename does not appear anywhere below, by design.
"""

import os
import sys
import hashlib
import platform
from pathlib import Path
from datetime import datetime
from itertools import combinations

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


section("STAGE 2 SYMBOLIC RULE AUDIT — RUN START (ANALYSIS ONLY)")
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

STAGE2_TAGS = ["MV201", "FIT201", "P201", "P202", "P203", "P204", "P205", "P206",
               "AIT201", "AIT202", "AIT203"]
PUMP_VALVE_TAGS = ["MV201", "P201", "P202", "P203", "P204", "P205", "P206"]
LABEL_TAGS = ["FIT201", "AIT201", "AIT202", "AIT203"]
PUMP_PAIRS = [("P201", "P202"), ("P203", "P204"), ("P205", "P206")]

# =============================================================================
# LOAD DATA
# =============================================================================
section("LOADING TRAIN AND VALIDATION (TEST NEVER LOADED)")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
train_df["_ts"] = pd.to_datetime(train_df["Timestamp"])
val_df["_ts"] = pd.to_datetime(val_df["Timestamp"])
log(f"TRAIN: {train_df.shape[0]:,} rows | VALIDATION: {val_df.shape[0]:,} rows")

SPLITS = {"TRAIN": train_df, "VALIDATION": val_df}

# =============================================================================
# TASK 1 — CONFIRM ALL TAGS EXIST
# =============================================================================
section("TASK 1 — CONFIRMING STAGE 2 TAGS EXIST")

feature_list = FEATURE_LIST_PATH.read_text(encoding="utf-8").strip().split("\n")
missing_from_featurelist = [t for t in STAGE2_TAGS if t not in feature_list]
missing_from_train = [t for t in STAGE2_TAGS if t not in train_df.columns]
missing_from_val = [t for t in STAGE2_TAGS if t not in val_df.columns]

log(f"Stage 2 tags requested ({len(STAGE2_TAGS)}): {STAGE2_TAGS}")
log(f"Missing from processed/metadata/feature_list.txt: {missing_from_featurelist}")
log(f"Missing from TRAIN columns: {missing_from_train}")
log(f"Missing from VALIDATION columns: {missing_from_val}")
assert not missing_from_featurelist and not missing_from_train and not missing_from_val, \
    "One or more Stage 2 tags are missing -- aborting."
log("CONFIRMED: all 11 Stage 2 tags exist in the feature list, TRAIN, and VALIDATION.")

# Sampling continuity check (relevant to transition/lag analyses below)
gap_report = {}
for split_name, df in SPLITS.items():
    diffs = df["_ts"].diff().dropna()
    gaps = diffs[diffs != pd.Timedelta(seconds=1)]
    gap_report[split_name] = gaps
    if len(gaps):
        log(f"\n{split_name}: {len(gaps)} non-1-second timestamp gap(s) detected "
            f"(may introduce minor edge effects in transition/lag counts near the gap):")
        for idx, g in gaps.items():
            loc = df.loc[idx, "_ts"]
            log(f"    gap of {g} ending at {loc}")
    else:
        log(f"\n{split_name}: perfectly uniform 1-second sampling, no gaps.")

# =============================================================================
# TASK 2 (+ TASK 5) — PER-TAG SUMMARY STATISTICS
# =============================================================================
section("TASK 2 & 5 — PER-TAG SUMMARY STATISTICS (incl. NORMAL/ATTACK for FIT201/AIT201-203)")


def top_values_str(series, n=5):
    vc = series.value_counts().head(n)
    return " | ".join(f"{v}:{c}" for v, c in vc.items())


def summarize(series):
    s = series.dropna()
    return {
        "N": len(s),
        "dtype": str(series.dtype),
        "min": s.min(), "max": s.max(), "mean": s.mean(), "median": s.median(),
        "std": s.std(ddof=0), "n_unique": s.nunique(),
        "top_values": top_values_str(s),
    }


summary_records = []
for tag in STAGE2_TAGS:
    label_filters = ["ALL", "NORMAL", "ATTACK"] if tag in LABEL_TAGS else ["ALL"]
    for split_name, df in SPLITS.items():
        for lf in label_filters:
            if lf == "ALL":
                sub = df
            elif lf == "NORMAL":
                sub = df[df["Label"] == 0]
            else:
                sub = df[df["Label"] == 1]
            stats = summarize(sub[tag])
            rec = {"Tag": tag, "Split": split_name, "Label_Filter": lf}
            rec.update(stats)
            summary_records.append(rec)

summary_df = pd.DataFrame(summary_records)
summary_path = TABLES_DIR / "stage2_tag_summary.csv"
summary_df.to_csv(summary_path, index=False)
log(f"Wrote {summary_path} ({len(summary_df)} rows)")

log("\nTRAIN / ALL summary for all 11 tags:")
for _, r in summary_df[(summary_df["Split"] == "TRAIN") & (summary_df["Label_Filter"] == "ALL")].iterrows():
    log(f"  {r['Tag']:8s} dtype={r['dtype']:8s} n_unique={int(r['n_unique']):>3} "
        f"min={r['min']:.4f} max={r['max']:.4f} mean={r['mean']:.4f} "
        f"median={r['median']:.4f} std={r['std']:.4f}  top={r['top_values']}")

# =============================================================================
# TASK 3 — VARIABLE TYPE CLASSIFICATION
# =============================================================================
section("TASK 3 — VARIABLE TYPE CLASSIFICATION (based on TRAIN, cross-checked vs VALIDATION)")

var_types = {}
type_notes = {}
for tag in STAGE2_TAGS:
    tr_nunique = train_df[tag].nunique()
    val_nunique = val_df[tag].nunique()
    is_integer_valued = np.all(np.mod(train_df[tag].dropna(), 1) == 0)
    if tr_nunique <= 2:
        vtype = "binary actuator/state"
    elif tr_nunique <= 10 and is_integer_valued:
        vtype = "discrete multi-state"
    else:
        vtype = "continuous sensor"
    var_types[tag] = vtype
    note = ""
    if tr_nunique != val_nunique:
        note = (f"NOTE: TRAIN shows {tr_nunique} unique value(s) but VALIDATION shows "
                 f"{val_nunique} -- classification below is TRAIN-based and may not capture "
                 f"states only observed in VALIDATION.")
    type_notes[tag] = note
    log(f"  {tag:8s} -> {vtype:24s} (TRAIN n_unique={tr_nunique}, VALIDATION n_unique={val_nunique})"
        + (f"  {note}" if note else ""))

# =============================================================================
# TASK 4 — PUMP/VALVE STATES, FREQUENCIES, TRANSITIONS
# =============================================================================
section("TASK 4 — PUMP/VALVE STATES, FREQUENCIES, AND TRANSITION COUNTS")

actuator_records = []
for split_name, df in SPLITS.items():
    for tag in PUMP_VALVE_TAGS:
        s = df[tag]
        n = len(s)
        vc = s.value_counts().sort_index()
        for state, count in vc.items():
            actuator_records.append({
                "Tag": tag, "Split": split_name, "Record_Type": "state_frequency",
                "Key": str(state), "Count": int(count), "Percentage": 100 * count / n,
            })
        # transitions: genuine state CHANGES only (self-transitions where the
        # state is unchanged from the previous second are excluded -- at 1 Hz
        # sampling those would simply reflect "stayed the same" and would
        # swamp the table without characterizing switching behavior)
        changed = s != s.shift(1)
        changed.iloc[0] = False  # first row has no prior state to compare to
        from_state = s.shift(1)[changed]
        to_state = s[changed]
        trans_pairs = pd.DataFrame({"from": from_state.values, "to": to_state.values})
        trans_counts = trans_pairs.value_counts()
        n_switch_events = int(changed.sum())
        for (fr, to), count in trans_counts.items():
            actuator_records.append({
                "Tag": tag, "Split": split_name, "Record_Type": "transition",
                "Key": f"{fr}->{to}", "Count": int(count),
                "Percentage": 100 * count / n_switch_events if n_switch_events else float("nan"),
            })
        actuator_records.append({
            "Tag": tag, "Split": split_name, "Record_Type": "total_switch_events",
            "Key": "any_change", "Count": n_switch_events, "Percentage": 100 * n_switch_events / n,
        })

actuator_df = pd.DataFrame(actuator_records)
actuator_path = TABLES_DIR / "stage2_actuator_states.csv"
actuator_df.to_csv(actuator_path, index=False)
log(f"Wrote {actuator_path} ({len(actuator_df)} rows)")

log("\nTRAIN state frequencies and switch-event totals:")
for tag in PUMP_VALVE_TAGS:
    freqs = actuator_df[(actuator_df.Tag == tag) & (actuator_df.Split == "TRAIN") &
                         (actuator_df.Record_Type == "state_frequency")]
    switches = actuator_df[(actuator_df.Tag == tag) & (actuator_df.Split == "TRAIN") &
                            (actuator_df.Record_Type == "total_switch_events")]["Count"].iloc[0]
    states_str = ", ".join(f"{r.Key}={r.Count}({r.Percentage:.2f}%)" for r in freqs.itertuples())
    log(f"  {tag:8s} states: {states_str}  |  switch events: {switches}")

# =============================================================================
# TASK 6 — PUMP PAIR PRIMARY/BACKUP BEHAVIOR
# =============================================================================
section("TASK 6 — PUMP PAIR (P201/P202, P203/P204, P205/P206) PRIMARY/BACKUP CHECK")

# Data-driven "active" heuristic (TRAIN-derived idle baseline applied to both
# splits for consistency): a pump's single most frequent TRAIN state is
# treated as its idle/off baseline; any other observed state counts as
# "active". This matches the common SWaT engineering convention (1=Off,
# 2=On) wherever a pump actually has two states, but is derived from the
# data rather than hard-coded, so it degrades gracefully (e.g. for a
# TRAIN-constant tag it correctly yields "always idle, never active").
idle_state = {}
for tag in PUMP_VALVE_TAGS:
    idle_state[tag] = train_df[tag].mode().iloc[0]
    log(f"  TRAIN-derived idle/baseline state for {tag}: {idle_state[tag]}")

pair_records = []
for a, b in PUMP_PAIRS:
    for split_name, df in SPLITS.items():
        active_a = (df[a] != idle_state[a])
        active_b = (df[b] != idle_state[b])
        n = len(df)
        combos = {
            "both_off": int((~active_a & ~active_b).sum()),
            f"{a}_only_on": int((active_a & ~active_b).sum()),
            f"{b}_only_on": int((~active_a & active_b).sum()),
            "both_on": int((active_a & active_b).sum()),
        }
        combos["one_on_either"] = combos[f"{a}_only_on"] + combos[f"{b}_only_on"]
        for combo_name, count in combos.items():
            pair_records.append({
                "Pair": f"{a}/{b}", "Split": split_name, "Combo": combo_name,
                "Count": count, "Percentage": 100 * count / n,
            })
        primary_backup_like = (combos["one_on_either"] / n) > 0.5 and combos["both_on"] / n < 0.05
        log(f"  {a}/{b} [{split_name}]: both_off={combos['both_off']} "
            f"({100*combos['both_off']/n:.2f}%), one_on={combos['one_on_either']} "
            f"({100*combos['one_on_either']/n:.2f}%), both_on={combos['both_on']} "
            f"({100*combos['both_on']/n:.2f}%) -- "
            f"{'consistent with primary/backup operation' if primary_backup_like else 'NOT clearly primary/backup (see percentages)'}")

pair_df = pd.DataFrame(pair_records)
pair_path = TABLES_DIR / "stage2_pump_pair_states.csv"
pair_df.to_csv(pair_path, index=False)
log(f"\nWrote {pair_path} ({len(pair_df)} rows)")

# =============================================================================
# TASK 7 — LAGGED EXPLORATORY ASSOCIATION (TRAIN ONLY, LAG 0-60s)
# =============================================================================
section("TASK 7 — LAGGED EXPLORATORY ASSOCIATION, TRAIN ONLY, LAGS 0-60s (NO CAUSAL CLAIM)")

RELATIONSHIPS = [
    (("P201", "P202"), "AIT201"),
    (("P203", "P204"), "AIT202"),
    (("P205", "P206"), "AIT203"),
    (("P203", "P204"), "AIT201"),
    (("P205", "P206"), "AIT202"),
    (("MV201",), "FIT201"),
]

MAX_LAG = 60
lag_records = []
best_lag_summary = []

for predictors, target in RELATIONSHIPS:
    if len(predictors) == 2:
        a, b = predictors
        pred_signal = ((train_df[a] != idle_state[a]) | (train_df[b] != idle_state[b])).astype(float)
        pred_label = f"{a}_or_{b}_active"
    else:
        (a,) = predictors
        pred_signal = (train_df[a] != idle_state[a]).astype(float)
        pred_label = f"{a}_active"

    target_signal = train_df[target].astype(float)
    rel_name = f"{'/'.join(predictors)} -> {target}"

    best = {"lag": None, "abs_corr": -1, "corr": np.nan}
    for lag in range(0, MAX_LAG + 1):
        shifted_pred = pred_signal.shift(lag)
        valid = shifted_pred.notna() & target_signal.notna()
        if shifted_pred[valid].nunique() < 2:
            corr = np.nan  # constant predictor after shifting -- correlation undefined
        else:
            corr = np.corrcoef(shifted_pred[valid], target_signal[valid])[0, 1]
        abs_corr = abs(corr) if not np.isnan(corr) else -1
        lag_records.append({
            "Relationship": rel_name, "Predictor_Signal": pred_label, "Target_Tag": target,
            "Lag_Seconds": lag, "Pearson_Correlation": corr, "Abs_Correlation": abs_corr,
            "Is_Best_Lag": False,
        })
        if abs_corr > best["abs_corr"]:
            best = {"lag": lag, "abs_corr": abs_corr, "corr": corr}

    for rec in lag_records:
        if rec["Relationship"] == rel_name and rec["Lag_Seconds"] == best["lag"]:
            rec["Is_Best_Lag"] = True

    best_lag_summary.append({
        "Relationship": rel_name, "Best_Lag_Seconds": best["lag"],
        "Correlation_At_Best_Lag": best["corr"], "Abs_Correlation_At_Best_Lag": best["abs_corr"],
    })
    log(f"  {rel_name:28s} strongest |corr| = {best['abs_corr']:.4f} at lag = {best['lag']}s "
        f"(signed corr = {best['corr']:.4f})")

lag_df = pd.DataFrame(lag_records)
lag_path = TABLES_DIR / "stage2_lag_analysis.csv"
lag_df.to_csv(lag_path, index=False)
log(f"\nWrote {lag_path} ({len(lag_df)} rows)")
log("\nReminder: these are simple lagged Pearson correlations between a binary activity "
    "indicator and a continuous sensor reading. They characterize TIMING/ASSOCIATION only. "
    "No causal claim is made or implied.")

# =============================================================================
# UNEXPECTED FINDINGS
# =============================================================================
section("UNEXPECTED FINDINGS RELEVANT TO SYMBOLIC RULE DESIGN")

unexpected = []
if train_df["P202"].nunique() == 1:
    unexpected.append(
        f"P202 is CONSTANT in TRAIN (single value = {train_df['P202'].iloc[0]}), confirming the "
        f"earlier preprocessing/RF audit finding. Any symbolic rule referencing a P202 state "
        f"CHANGE, or the P201/P202 pair's 'backup pump activation' behavior, cannot be "
        f"characterized from TRAIN alone -- P202 never activates in TRAIN, so the pair's "
        f"apparent 'primary/backup' behavior in TRAIN is really just P201 acting alone. "
        f"Check VALIDATION's P202 behavior (see stage2_actuator_states.csv) before relying on "
        f"this pair for a symbolic feature."
    )
for tag, note in type_notes.items():
    if note:
        unexpected.append(f"{tag}: {note}")
for split_name, gaps in gap_report.items():
    if len(gaps):
        unexpected.append(
            f"{split_name} contains {len(gaps)} timestamp gap(s) (non-1-second step). Transition "
            f"counts and lag correlations spanning a gap may reflect an artificial jump rather "
            f"than a real fast state change; this was not corrected here (analysis-only stage)."
        )
if not unexpected:
    unexpected.append("None beyond what is already itemized in the per-task sections above.")
for u in unexpected:
    log(f"  - {u}")

# =============================================================================
# SAVE NARRATIVE AUDIT REPORT
# =============================================================================
section("SAVING results/stage2_symbolic_rule_audit.txt")

report_lines = []
report_lines.append("STAGE 2 SYMBOLIC RULE AUDIT — ANALYSIS ONLY, NO THRESHOLDS DEFINED")
report_lines.append(f"Generated: {RUN_START.isoformat()}")
report_lines.append(f"Python: {sys.version}")
report_lines.append(f"pandas: {pd.__version__} | numpy: {np.__version__}")
report_lines.append(f"OS: {platform.system()} {platform.release()}")
report_lines.append("")
report_lines.append(f"Data sources: {TRAIN_PATH.name} ({len(train_df):,} rows), "
                     f"{VAL_PATH.name} ({len(val_df):,} rows). TEST was never loaded.")
report_lines.append("")
report_lines.append("=== TASK 1: TAG EXISTENCE ===")
report_lines.append(f"All {len(STAGE2_TAGS)} Stage 2 tags confirmed present in feature_list.txt, "
                     f"TRAIN, and VALIDATION: {STAGE2_TAGS}")
report_lines.append("")
report_lines.append("=== TASK 3: VARIABLE TYPE CLASSIFICATION (TRAIN-based) ===")
for tag in STAGE2_TAGS:
    report_lines.append(f"  {tag:8s}: {var_types[tag]}" + (f"  [{type_notes[tag]}]" if type_notes[tag] else ""))
report_lines.append("")
report_lines.append("=== TASK 4: PUMP/VALVE STATES (TRAIN) ===")
for tag in PUMP_VALVE_TAGS:
    freqs = actuator_df[(actuator_df.Tag == tag) & (actuator_df.Split == "TRAIN") &
                         (actuator_df.Record_Type == "state_frequency")]
    switches = actuator_df[(actuator_df.Tag == tag) & (actuator_df.Split == "TRAIN") &
                            (actuator_df.Record_Type == "total_switch_events")]["Count"].iloc[0]
    states_str = ", ".join(f"{r.Key}={r.Count}({r.Percentage:.2f}%)" for r in freqs.itertuples())
    report_lines.append(f"  {tag:8s} states: {states_str} | switch events: {switches}")
report_lines.append("")
report_lines.append("=== TASK 6: PUMP PAIR PRIMARY/BACKUP CHECK ===")
for a, b in PUMP_PAIRS:
    for split_name in SPLITS:
        row = pair_df[(pair_df.Pair == f"{a}/{b}") & (pair_df.Split == split_name)]
        both_off = row[row.Combo == "both_off"]["Percentage"].iloc[0]
        one_on = row[row.Combo == "one_on_either"]["Percentage"].iloc[0]
        both_on = row[row.Combo == "both_on"]["Percentage"].iloc[0]
        report_lines.append(f"  {a}/{b} [{split_name}]: both_off={both_off:.2f}%, "
                             f"one_on={one_on:.2f}%, both_on={both_on:.2f}%")
report_lines.append("")
report_lines.append("=== TASK 7: STRONGEST LAG PER RELATIONSHIP (TRAIN ONLY, NO CAUSAL CLAIM) ===")
for rec in best_lag_summary:
    report_lines.append(f"  {rec['Relationship']:28s} best lag = {rec['Best_Lag_Seconds']:>2}s, "
                         f"corr = {rec['Correlation_At_Best_Lag']:.4f}")
report_lines.append("")
report_lines.append("=== UNEXPECTED FINDINGS ===")
for u in unexpected:
    report_lines.append(f"  - {u}")
report_lines.append("")
report_lines.append("=== GUARDRAILS CONFIRMED ===")
report_lines.append("  - TEST partition: never loaded, filename not referenced anywhere in this script.")
report_lines.append("  - No symbolic thresholds were defined.")
report_lines.append("  - No feature selection was performed.")
report_lines.append("  - No model was trained.")
report_lines.append("  - No existing script, dataset, or split was modified.")

report_path = RESULTS_DIR / "stage2_symbolic_rule_audit.txt"
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
    "no_thresholds_defined": True,   # structural: no threshold-defining code path exists
    "no_feature_selection": True,    # structural: no selection code path exists
    "no_model_trained": True,        # structural: no fit() call anywhere in this script
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

log("1. DETECTED VARIABLE TYPES:")
for tag in STAGE2_TAGS:
    log(f"   {tag:8s}: {var_types[tag]}")

log("\n2. PUMP / VALVE STATES (TRAIN):")
for tag in PUMP_VALVE_TAGS:
    freqs = actuator_df[(actuator_df.Tag == tag) & (actuator_df.Split == "TRAIN") &
                         (actuator_df.Record_Type == "state_frequency")]
    states_str = ", ".join(f"{r.Key}({r.Percentage:.1f}%)" for r in freqs.itertuples())
    log(f"   {tag:8s}: {states_str}")

log("\n3. STRONGEST LAG PER CANDIDATE RELATIONSHIP (TRAIN ONLY):")
for rec in best_lag_summary:
    log(f"   {rec['Relationship']:28s} lag={rec['Best_Lag_Seconds']:>2}s  "
        f"corr={rec['Correlation_At_Best_Lag']:.4f}")

log("\n4. UNEXPECTED FINDINGS AFFECTING SYMBOLIC RULE DESIGN:")
for u in unexpected:
    log(f"   - {u}")

run_end = datetime.now()
log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
log("ANALYSIS-ONLY STAGE COMPLETE. No thresholds defined, no features selected, no model trained.")

if not all_checks_passed:
    sys.exit(1)
