"""
code/01_split_candidate_analysis.py

CHRONOLOGICAL TRAIN/VALIDATION/TEST SPLIT DESIGN — ANALYSIS ONLY

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

This script does NOT train Random Forest, does NOT train LSTM, does NOT
perform feature selection, does NOT fit a scaler, does NOT generate
symbolic features, and does NOT modify either source CSV. It only reads
the raw Attack CSV (for exact label/interval counts, as permitted) and
the existing audit outputs, evaluates three candidate chronological
split boundaries, and writes analysis files. No split is materialized
to disk as a dataset -- only descriptive statistics ABOUT candidate
splits are written.
"""

import os
import sys
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
DATASET_DIR = Path(os.environ.get("SWAT_DATASET_DIR", str(PROJECT_ROOT / "dataset")))  # portability fix: actual directory is lowercase; overridable via env var
ATTACK_PATH = DATASET_DIR / "SWaT_Dataset_Attack_v0.csv"

AUDIT_DIR = PROJECT_ROOT / "results" / "audit"
TABLES_DIR = PROJECT_ROOT / "tables"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

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


# ---------------------------------------------------------------------------
# 1. LOAD RAW ATTACK DATASET (read-only; same handling verified in Stage 0)
# ---------------------------------------------------------------------------
section("LOADING RAW ATTACK DATASET (READ-ONLY)")
df = pd.read_csv(ATTACK_PATH, encoding="utf-8-sig", low_memory=False)
ts_col = df.columns[0]
label_col = df.columns[-1]
raw_ts = df[ts_col].astype(str).str.strip()
df["_ts"] = pd.to_datetime(raw_ts, format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
assert df["_ts"].isna().sum() == 0, "Timestamp parse failure detected -- aborting."
assert df["_ts"].is_monotonic_increasing, "Timestamps not monotonic -- aborting."
df = df.reset_index(drop=True)
n_total = len(df)
file_start = df["_ts"].iloc[0]
file_end = df["_ts"].iloc[-1]
log(f"Loaded {n_total:,} rows. Range: {file_start} -> {file_end}")

df["_is_attack"] = df[label_col].ne("Normal")

# ---------------------------------------------------------------------------
# 2. RECONSTRUCT THE 35 CONTIGUOUS ATTACK INTERVALS (cross-checked against
#    results/audit/attack_intervals.csv) AND ADD "normal data immediately
#    before / after" FOR EACH INTERVAL (Task 1).
# ---------------------------------------------------------------------------
section("TASK 1 — RECONSTRUCTING 35 ATTACK INTERVALS WITH SURROUNDING NORMAL SPANS")

group_id = (df["_is_attack"] != df["_is_attack"].shift()).cumsum()
groups = df.groupby(group_id)

intervals = []
for gid, sub_idx in groups.groups.items():
    sub = df.loc[sub_idx]
    if not sub["_is_attack"].iloc[0]:
        continue
    intervals.append({
        "start_pos": sub_idx[0],
        "end_pos": sub_idx[-1],
        "start_timestamp": sub["_ts"].iloc[0],
        "end_timestamp": sub["_ts"].iloc[-1],
        "n_observations": len(sub),
        "n_label_Attack": int((sub[label_col] == "Attack").sum()),
        "n_label_A_ttack_typo": int((sub[label_col] == "A ttack").sum()),
    })

intervals_df = pd.DataFrame(intervals).reset_index(drop=True)
intervals_df.insert(0, "interval_id", intervals_df.index + 1)
intervals_df["duration_seconds"] = (
    intervals_df["end_timestamp"] - intervals_df["start_timestamp"]
).dt.total_seconds()

assert len(intervals_df) == 35, f"Expected 35 attack intervals, found {len(intervals_df)}"
log(f"Reconstructed {len(intervals_df)} contiguous attack intervals (matches audit: 35).")

# normal data immediately BEFORE each interval
normal_before = []
prev_end = file_start  # for interval 1, "before" is measured from file start
for i, row in intervals_df.iterrows():
    normal_before.append((row["start_timestamp"] - prev_end).total_seconds())
    prev_end = row["end_timestamp"]
intervals_df["normal_seconds_immediately_before"] = normal_before

# normal data immediately AFTER each interval
normal_after = []
for i, row in intervals_df.iterrows():
    if i + 1 < len(intervals_df):
        nxt_start = intervals_df.loc[i + 1, "start_timestamp"]
    else:
        nxt_start = file_end + pd.Timedelta(seconds=1)  # end of file (last row is inclusive)
    normal_after.append((nxt_start - row["end_timestamp"]).total_seconds())
intervals_df["normal_seconds_immediately_after"] = normal_after

# Verify the tail after interval 35 truly is all Normal
tail = df[df["_ts"] > intervals_df["end_timestamp"].iloc[-1]]
n_tail_attack = int(tail["_is_attack"].sum())
log(f"Rows after interval 35 end: {len(tail):,}; of which attack-labeled: {n_tail_attack} "
    f"(expected 0)")
assert n_tail_attack == 0, "Unexpected attack-labeled rows after the last detected interval."

extended_path = AUDIT_DIR / "attack_intervals_extended.csv"
intervals_df.to_csv(extended_path, index=False)
log(f"Wrote {extended_path}")

log("")
log("Chronological summary of all 35 attack intervals:")
for _, r in intervals_df.iterrows():
    log(f"  [{int(r['interval_id']):>2}] {r['start_timestamp']} -> {r['end_timestamp']}  "
        f"dur={r['duration_seconds']:>7.0f}s  obs={r['n_observations']:>5}  "
        f"(Attack={r['n_label_Attack']}, typo={r['n_label_A_ttack_typo']})  "
        f"normal_before={r['normal_seconds_immediately_before']:>6.0f}s  "
        f"normal_after={r['normal_seconds_immediately_after']:>6.0f}s")

giant = intervals_df.loc[intervals_df["duration_seconds"].idxmax()]
total_attack_rows = int(intervals_df["n_observations"].sum())
log("")
log(f"NOTE: interval {int(giant['interval_id'])} ({giant['start_timestamp']} -> "
    f"{giant['end_timestamp']}) is a major outlier: {int(giant['n_observations']):,} "
    f"observations = {100*giant['n_observations']/total_attack_rows:.1f}% of ALL "
    f"attack-labeled rows in the entire Attack file, by itself. This single interval "
    f"dominates any split it is placed in and is treated as a first-class design "
    f"consideration below, not an incidental detail.")

# ---------------------------------------------------------------------------
# 3. CANDIDATE SPLIT DEFINITIONS (Task 2)
# ---------------------------------------------------------------------------
section("TASK 2 — CANDIDATE CHRONOLOGICAL SPLIT DEFINITIONS")

CANDIDATES = {
    "Candidate_1_GapOptimal": {
        "description": (
            "Gap-optimal, target-matching split. Boundaries placed in the two "
            "largest Normal gaps nearest the 60-70%/10-20%/20-25% guideline "
            "targets. The giant interval 22 (~66% of all attack rows) falls "
            "entirely in TRAIN."
        ),
        "train_val_boundary": pd.Timestamp("2015-12-31 13:00:00"),
        "val_test_boundary": pd.Timestamp("2016-01-01 08:00:00"),
    },
    "Candidate_2_CalendarAligned": {
        "description": (
            "Calendar-day-aligned split: TRAIN = everything through 2015-12-31, "
            "VALIDATION = all of 2016-01-01, TEST = all of 2016-01-02. Simple, "
            "reviewer-legible narrative. Giant interval 22 falls in TRAIN."
        ),
        "train_val_boundary": pd.Timestamp("2016-01-01 00:00:00"),
        "val_test_boundary": pd.Timestamp("2016-01-02 00:00:00"),
    },
    "Candidate_3_GiantIntervalIsolated": {
        "description": (
            "The giant interval 22 is deliberately isolated INSIDE VALIDATION "
            "only (not TRAIN, not TEST), so it can never bias supervised "
            "training toward one dominant scenario and never inflate TEST "
            "metrics with one easy-to-detect sustained anomaly. TEST is "
            "consequently larger than the 20-25% guideline as a direct, "
            "documented structural consequence."
        ),
        "train_val_boundary": pd.Timestamp("2015-12-30 20:00:00"),
        "val_test_boundary": pd.Timestamp("2015-12-31 14:00:00"),
    },
}

for name, spec in CANDIDATES.items():
    log(f"{name}: train/val @ {spec['train_val_boundary']}  |  val/test @ {spec['val_test_boundary']}")
    log(f"    {spec['description']}")


# ---------------------------------------------------------------------------
# 4. PER-CANDIDATE STATISTICS (Task 3) + BOUNDARY CHECK (Task 4)
# ---------------------------------------------------------------------------
section("TASK 3 & 4 — PER-CANDIDATE STATISTICS AND BOUNDARY CHECKS")


def split_stats(sub_df, split_name):
    n = len(sub_df)
    if n == 0:
        return {
            "split": split_name, "start_timestamp": None, "end_timestamp": None,
            "n_rows": 0, "n_normal": 0, "n_attack_exact": 0, "n_attack_typo": 0,
            "n_attack_total": 0, "attack_pct": None,
        }
    n_normal = int((sub_df[label_col] == "Normal").sum())
    n_attack_exact = int((sub_df[label_col] == "Attack").sum())
    n_attack_typo = int((sub_df[label_col] == "A ttack").sum())
    n_attack_total = n_attack_exact + n_attack_typo
    return {
        "split": split_name,
        "start_timestamp": sub_df["_ts"].iloc[0],
        "end_timestamp": sub_df["_ts"].iloc[-1],
        "n_rows": n,
        "n_normal": n_normal,
        "n_attack_exact": n_attack_exact,
        "n_attack_typo": n_attack_typo,
        "n_attack_total": n_attack_total,
        "attack_pct": round(100 * n_attack_total / n, 4),
    }


def intervals_fully_within(lo, hi):
    """Count attack intervals fully contained in [lo, hi)."""
    mask = (intervals_df["start_timestamp"] >= lo) & (intervals_df["end_timestamp"] < hi)
    return intervals_df.loc[mask]


def nearest_gap_context(boundary_ts):
    """Find which Normal gap a boundary falls in, and the distance in seconds
    to the nearest attack interval edge on each side."""
    before = intervals_df[intervals_df["end_timestamp"] < boundary_ts]
    after = intervals_df[intervals_df["start_timestamp"] > boundary_ts]
    prev_iv = before.iloc[-1] if len(before) else None
    next_iv = after.iloc[0] if len(after) else None
    sec_to_prev = (boundary_ts - prev_iv["end_timestamp"]).total_seconds() if prev_iv is not None else None
    sec_to_next = (next_iv["start_timestamp"] - boundary_ts).total_seconds() if next_iv is not None else None
    return prev_iv, next_iv, sec_to_prev, sec_to_next


all_candidate_rows = []
comparison_rows = []
boundary_check_lines = []

for cname, spec in CANDIDATES.items():
    tv = spec["train_val_boundary"]
    vt = spec["val_test_boundary"]
    assert tv < vt, f"{cname}: train/val boundary must precede val/test boundary"

    train_df = df[df["_ts"] < tv]
    val_df = df[(df["_ts"] >= tv) & (df["_ts"] < vt)]
    test_df = df[df["_ts"] >= vt]
    assert len(train_df) + len(val_df) + len(test_df) == n_total

    train_iv = intervals_fully_within(file_start, tv)
    val_iv = intervals_fully_within(tv, vt)
    test_iv = intervals_fully_within(vt, file_end + pd.Timedelta(seconds=1))
    n_iv_accounted = len(train_iv) + len(val_iv) + len(test_iv)

    # Split-crossing check: any interval whose start and end straddle a boundary?
    def straddles(boundary):
        return intervals_df[(intervals_df["start_timestamp"] < boundary) &
                             (intervals_df["end_timestamp"] >= boundary)]

    split_tv = straddles(tv)
    split_vt = straddles(vt)

    stats = {
        "candidate": cname,
        "train": split_stats(train_df, "train"),
        "val": split_stats(val_df, "validation"),
        "test": split_stats(test_df, "test"),
        "n_intervals_train": len(train_iv),
        "n_intervals_val": len(val_iv),
        "n_intervals_test": len(test_iv),
        "n_intervals_accounted": n_iv_accounted,
        "interval_split_at_tv_boundary": len(split_tv) > 0,
        "interval_split_at_vt_boundary": len(split_vt) > 0,
    }
    all_candidate_rows.append(stats)

    log(f"\n--- {cname} ---")
    for split_name, s in [("TRAIN", stats["train"]), ("VAL", stats["val"]), ("TEST", stats["test"])]:
        log(f"  {split_name:5s}: {s['start_timestamp']} -> {s['end_timestamp']} | "
            f"rows={s['n_rows']:,} ({100*s['n_rows']/n_total:.2f}% of file) | "
            f"normal={s['n_normal']:,} | attack={s['n_attack_total']:,} "
            f"(exact={s['n_attack_exact']}, typo={s['n_attack_typo']}) | "
            f"attack%={s['attack_pct']}")
    log(f"  Attack intervals -> train={len(train_iv)}, val={len(val_iv)}, test={len(test_iv)} "
        f"(sum={n_iv_accounted}/35)")
    assert n_iv_accounted == 35, f"{cname}: interval accounting mismatch ({n_iv_accounted}/35)"

    # Boundary check detail (Task 4)
    for boundary_name, boundary_ts in [("train/val", tv), ("val/test", vt)]:
        prev_iv, next_iv, sec_prev, sec_next = nearest_gap_context(boundary_ts)
        is_split = len(straddles(boundary_ts)) > 0
        line = (
            f"{cname} | {boundary_name} boundary = {boundary_ts} | "
            f"splits an attack interval: {is_split} | "
            f"located in Normal gap between interval "
            f"{int(prev_iv['interval_id']) if prev_iv is not None else 'START-OF-FILE'} "
            f"(ends {prev_iv['end_timestamp'] if prev_iv is not None else file_start}) and interval "
            f"{int(next_iv['interval_id']) if next_iv is not None else 'END-OF-FILE'} "
            f"(starts {next_iv['start_timestamp'] if next_iv is not None else file_end}) | "
            f"separation from previous attack: {sec_prev}s | "
            f"separation from next attack: {sec_next}s"
        )
        boundary_check_lines.append(line)
        log(f"  BOUNDARY CHECK: {line}")

for l in boundary_check_lines:
    pass  # already logged above


# ---------------------------------------------------------------------------
# 5. BUILD COMPARISON TABLE (Task 5) AND OUTPUT CSVs (Task 9)
# ---------------------------------------------------------------------------
section("TASK 5 — BUILDING COMPARISON TABLE")

comparison_records = []
for stats in all_candidate_rows:
    cname = stats["candidate"]
    tr, va, te = stats["train"], stats["val"], stats["test"]
    comparison_records.append({
        "candidate": cname,
        "train_start": tr["start_timestamp"], "train_end": tr["end_timestamp"],
        "train_rows": tr["n_rows"], "train_pct_of_file": round(100 * tr["n_rows"] / n_total, 2),
        "train_normal": tr["n_normal"], "train_attack": tr["n_attack_total"],
        "train_attack_pct": tr["attack_pct"], "train_n_intervals": stats["n_intervals_train"],
        "val_start": va["start_timestamp"], "val_end": va["end_timestamp"],
        "val_rows": va["n_rows"], "val_pct_of_file": round(100 * va["n_rows"] / n_total, 2),
        "val_normal": va["n_normal"], "val_attack": va["n_attack_total"],
        "val_attack_pct": va["attack_pct"], "val_n_intervals": stats["n_intervals_val"],
        "test_start": te["start_timestamp"], "test_end": te["end_timestamp"],
        "test_rows": te["n_rows"], "test_pct_of_file": round(100 * te["n_rows"] / n_total, 2),
        "test_normal": te["n_normal"], "test_attack": te["n_attack_total"],
        "test_attack_pct": te["attack_pct"], "test_n_intervals": stats["n_intervals_test"],
        "any_interval_split": stats["interval_split_at_tv_boundary"] or stats["interval_split_at_vt_boundary"],
        "giant_interval_22_location": (
            "train" if "Candidate_1" in cname or "Candidate_2" in cname else "validation"
        ),
    })

comparison_df = pd.DataFrame(comparison_records)
comparison_path = AUDIT_DIR / "split_candidate_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)
log(f"Wrote {comparison_path}")

# Publication-facing compact table
table_records = []
for stats in all_candidate_rows:
    cname = stats["candidate"]
    for split_name, s in [("Train", stats["train"]), ("Validation", stats["val"]), ("Test", stats["test"])]:
        table_records.append({
            "Candidate": cname,
            "Split": split_name,
            "Start": str(s["start_timestamp"]),
            "End": str(s["end_timestamp"]),
            "Rows": s["n_rows"],
            "% of Attack File": round(100 * s["n_rows"] / n_total, 2),
            "Normal": s["n_normal"],
            "Attack": s["n_attack_total"],
            "Attack %": s["attack_pct"],
        })
table_split_df = pd.DataFrame(table_records)
table_path = TABLES_DIR / "table_split_candidates.csv"
table_split_df.to_csv(table_path, index=False)
log(f"Wrote {table_path}")

# ---------------------------------------------------------------------------
# 6. WRITE split_protocol_analysis.txt (Tasks 1 summary, 2, 4, 5 narrative)
# ---------------------------------------------------------------------------
section("WRITING split_protocol_analysis.txt")

lines = []
lines.append("SWaT CHRONOLOGICAL SPLIT PROTOCOL ANALYSIS")
lines.append(f"Generated: {RUN_START.isoformat()}")
lines.append("Status: ANALYSIS ONLY. No split implemented. No model trained.")
lines.append("")
lines.append("=" * 78)
lines.append("1. ATTACK INTERVAL LANDSCAPE (see attack_intervals_extended.csv for full table)")
lines.append("=" * 78)
lines.append(f"35 contiguous attack-state intervals span {file_start} to {file_end}.")
lines.append(f"Total attack-labeled rows: {total_attack_rows:,} "
             f"({100*total_attack_rows/n_total:.2f}% of the Attack file).")
lines.append(f"Interval {int(giant['interval_id'])} is a structural outlier: "
             f"{giant['start_timestamp']} -> {giant['end_timestamp']} "
             f"({giant['duration_seconds']:.0f}s, {int(giant['n_observations']):,} obs) "
             f"= {100*giant['n_observations']/total_attack_rows:.1f}% of ALL attack rows in the "
             f"file, by itself. Every candidate below treats this interval as a first-class "
             f"design variable rather than incidental detail.")
lines.append(f"Interval 19 ({intervals_df.loc[18,'start_timestamp']} -> "
             f"{intervals_df.loc[18,'end_timestamp']}) is the only interval containing the raw "
             f"'A ttack' label typo (37 of its 321 rows).")
lines.append("")
lines.append("=" * 78)
lines.append("2. CANDIDATE DEFINITIONS")
lines.append("=" * 78)
for cname, spec in CANDIDATES.items():
    lines.append(f"\n{cname}")
    lines.append(f"  Train/Val boundary : {spec['train_val_boundary']}")
    lines.append(f"  Val/Test boundary  : {spec['val_test_boundary']}")
    lines.append(f"  Rationale: {spec['description']}")

lines.append("")
lines.append("=" * 78)
lines.append("3. PER-CANDIDATE STATISTICS")
lines.append("=" * 78)
for stats in all_candidate_rows:
    cname = stats["candidate"]
    lines.append(f"\n{cname}")
    for split_name, s in [("TRAIN", stats["train"]), ("VALIDATION", stats["val"]), ("TEST", stats["test"])]:
        lines.append(f"  {split_name}:")
        lines.append(f"    start={s['start_timestamp']}  end={s['end_timestamp']}")
        lines.append(f"    rows={s['n_rows']:,} ({100*s['n_rows']/n_total:.2f}% of Attack file)")
        lines.append(f"    normal={s['n_normal']:,}  attack={s['n_attack_total']:,} "
                     f"(exact={s['n_attack_exact']}, typo={s['n_attack_typo']})  "
                     f"attack%={s['attack_pct']}")
    lines.append(f"  Complete attack intervals: train={stats['n_intervals_train']}, "
                 f"val={stats['n_intervals_val']}, test={stats['n_intervals_test']} "
                 f"(sum={stats['n_intervals_accounted']}/35)")
    lines.append(f"  Any interval split across a boundary: "
                 f"{stats['interval_split_at_tv_boundary'] or stats['interval_split_at_vt_boundary']}")

lines.append("")
lines.append("=" * 78)
lines.append("4. BOUNDARY CHECK DETAIL")
lines.append("=" * 78)
for l in boundary_check_lines:
    lines.append(l)
lines.append("")
lines.append("All six boundaries (2 per candidate x 3 candidates) fall inside a contiguous")
lines.append("Normal span between two attack intervals; none cuts an attack interval in half.")
lines.append("Separation-from-nearest-attack figures above should be reviewed against the")
lines.append("chosen sequence length once LSTM windowing is designed (Task 9 of the prior")
lines.append("audit) -- a boundary with only tens of seconds of separation could still let a")
lines.append("sliding window straddle it even though no single row is mislabeled.")

lines.append("")
lines.append("=" * 78)
lines.append("5. CANDIDATE COMPARISON")
lines.append("=" * 78)
lines.append("See results/audit/split_candidate_comparison.csv for the full numeric table.")
lines.append("Qualitative comparison:")
lines.append("")
lines.append("Candidate_1_GapOptimal")
lines.append("  + Closest to the 60-70/10-20/20-25 guideline of any candidate.")
lines.append("  + Giant interval 22 fully in TRAIN: RF/LSTM get to learn from the richest,")
lines.append("    longest sustained-attack signal available.")
lines.append("  + VAL and TEST each contain multiple, diverse, shorter/medium intervals,")
lines.append("    which is more representative of typical attack duration than the outlier.")
lines.append("  - VAL and TEST attack% will be noticeably lower than TRAIN's, since TRAIN")
lines.append("    absorbs 66% of all attack rows via interval 22 alone -- reviewers may ask")
lines.append("    whether TRAIN's class balance is representative of deployment conditions.")
lines.append("  - A likely reviewer question: 'does performance depend on having seen a")
lines.append("    10-hour attack in training?' should be pre-empted in the paper's limitations.")
lines.append("")
lines.append("Candidate_2_CalendarAligned")
lines.append("  + Simplest possible narrative for a paper/reviewer: whole calendar days per")
lines.append("    split, trivially reproducible and auditable by any reader.")
lines.append("  + Also keeps interval 22 in TRAIN; VAL and TEST both fully self-contained")
lines.append("    calendar days with multiple attack intervals each.")
lines.append("  - TEST is only ~12% of the file -- below the 20-25% guideline -- so per-split")
lines.append("    confidence intervals on test metrics will be wider than Candidate 1's.")
lines.append("  - Calendar-day framing is convenient but arbitrary with respect to the")
lines.append("    process; a reviewer could ask why days rather than attack-driven boundaries")
lines.append("    were used (answer: both criteria happen to coincide here, which is why this")
lines.append("    candidate is viable at all -- it is not guaranteed in general).")
lines.append("")
lines.append("Candidate_3_GiantIntervalIsolated")
lines.append("  + Directly addresses the interval-22 confound: TRAIN never sees the giant")
lines.append("    interval (so RF/LSTM cannot simply memorize one long, easy-to-detect")
lines.append("    anomaly and get credit for 66% of all attack rows) and TEST never sees it")
lines.append("    either (so headline test metrics cannot be dominated by one trivially")
lines.append("    detectable long anomaly).")
lines.append("  + VALIDATION becomes scientifically useful for a specific purpose: tuning an")
lines.append("    LSTM/RF decision threshold's behavior on a sustained, high-magnitude event.")
lines.append("  - TEST balloons to ~39% of the file, well above the 20-25% guideline, as a")
lines.append("    direct, unavoidable structural consequence of interval 22 sitting near the")
lines.append("    temporal midpoint of the whole Attack file. TRAIN correspondingly shrinks to")
lines.append("    ~46%, below the 60-70% guideline.")
lines.append("  - VALIDATION's class balance is extremely atypical (dominated almost entirely")
lines.append("    by interval 22), so validation-set metrics must not be read as a proxy for")
lines.append("    'typical' detector performance -- only for sustained-event sensitivity.")
lines.append("")
lines.append("Comparison is NOT decided by which candidate has the most balanced class")
lines.append("percentages; Candidate 3 is included specifically because its imbalance is")
lines.append("informative, not despite it. See split_candidate_comparison.csv columns")
lines.append("train_attack_pct / val_attack_pct / test_attack_pct for exact figures.")

protocol_path = AUDIT_DIR / "split_protocol_analysis.txt"
protocol_path.write_text("\n".join(lines), encoding="utf-8")
log(f"Wrote {protocol_path}")

# ---------------------------------------------------------------------------
# 7. Section 6 — Normal-only dataset role analysis
# ---------------------------------------------------------------------------
section("WRITING normal_dataset_role_analysis.txt")

normal_lines = []
normal_lines.append("ROLE OF THE NORMAL-ONLY DATASET (SWaT_Dataset_Normal_v0.csv) IN THE")
normal_lines.append("SUPERVISED EXPERIMENT -- ANALYSIS ONLY, NOT IMPLEMENTED")
normal_lines.append(f"Generated: {RUN_START.isoformat()}")
normal_lines.append("")
normal_lines.append("Recap from the raw-dataset audit: SWaT_Dataset_Normal_v0.csv contains")
normal_lines.append("496,800 rows, 100% labeled 'Normal', spanning 2015-12-22 16:00:00 to")
normal_lines.append("2015-12-28 09:59:59 -- immediately and contiguously BEFORE the Attack file")
normal_lines.append("begins (2015-12-28 10:00:00, 1 second later). Because it is 100% one class,")
normal_lines.append("pooling it directly into a supervised train set recreates the exact")
normal_lines.append("file-vs-class confound flagged in the prior audit: a classifier could learn")
normal_lines.append("'which file did this row come from' as a proxy for 'is this Normal', since")
normal_lines.append("file membership and class label would be almost perfectly correlated for")
normal_lines.append("every row contributed by this file.")
normal_lines.append("")


def role_block(title, benefit, risk, recommend):
    normal_lines.append(f"--- Role: {title} ---")
    normal_lines.append(f"1. Scientific benefit: {benefit}")
    normal_lines.append(f"2. Leakage/confounding risk: {risk}")
    normal_lines.append(f"3. Recommendation: {recommend}")
    normal_lines.append("")


role_block(
    "Establishing normal operating statistics (baseline ranges, expected sensor "
    "bounds for symbolic-rule thresholds)",
    "Gives a large, attack-free reference for 'what does normal ICS operation look "
    "like' independent of the Attack file's own Normal rows -- useful cross-check "
    "and a natural source for symbolic/explainable rule thresholds the paper "
    "proposes.",
    "Low, PROVIDED the derived statistics are treated as fixed domain knowledge "
    "computed once, upfront, and are not re-derived per split or tuned against "
    "Attack-file performance. If thresholds are iteratively adjusted based on Test "
    "results, that reintroduces leakage through a back door.",
    "RECOMMENDED, with the threshold-derivation step documented as a one-time, "
    "pre-registered step decided before any split evaluation is viewed.",
)

role_block(
    "LSTM pretraining / normal sequence-pattern learning (e.g. autoencoder or "
    "next-step prediction pretext task, later fine-tuned on the supervised task)",
    "Large volume (496,800 extra timesteps) of clean sequential Normal data is "
    "exactly the kind of data an unsupervised/self-supervised pretraining "
    "objective wants, and could improve the LSTM's representation of normal "
    "process dynamics before it ever sees Attack-file rows.",
    "Moderate. Pretraining itself does not see labels, so it cannot directly leak "
    "the Attack file's Test labels. But the pretrained weights ARE a channel: if "
    "pretraining hyperparameters (epochs, architecture) are tuned by watching "
    "downstream Test performance, that is leakage through model selection, not "
    "through the data itself. Also note the file-vs-class confound does not apply "
    "here in the same way, because pretraining is label-free -- but a smooth "
    "process-drift signal specific to the Dec 22-28 window could still bias early "
    "representations in ways that don't transfer to the Attack-file period.",
    "CONDITIONALLY RECOMMENDED: acceptable as an unsupervised pretraining corpus "
    "IF pretraining hyperparameters are fixed before looking at any downstream "
    "split's performance, and IF this choice is reported explicitly as a "
    "pretraining source in the paper.",
)

role_block(
    "Scaler fitting (mean/std or min/max for normalization)",
    "A large, purely-Normal reference could seem like a principled place to fit a "
    "scaler, since 'normal operating range' is a natural normalization basis.",
    "HIGH. Fitting a scaler on Normal-only data still means the scaler has never "
    "seen the value ranges attacks push sensors into -- attack rows could clip, "
    "saturate, or sit as extreme outliers under a scaler fit only on Normal data, "
    "distorting exactly the signal the classifier most needs. It also does not "
    "solve the actual leakage question, which is whether TEST rows influenced the "
    "scaler -- using the Normal-only file sidesteps that but at the cost of a "
    "scaler that is not representative of the full input distribution the model "
    "must handle.",
    "NOT RECOMMENDED as the sole scaler-fitting source. If used at all, fit the "
    "scaler on the designated TRAIN portion of the Attack file (which contains "
    "both classes), optionally concatenated with the Normal-only file for "
    "additional Normal-range coverage -- but never on Normal-only data alone, and "
    "never on anything that includes VALIDATION or TEST rows.",
)

role_block(
    "Symbolic-rule threshold derivation (for the paper's symbolic feature fusion "
    "component)",
    "Same benefit as role 1: a large, independent Normal-only reference is a "
    "clean, defensible basis for defining 'normal range' thresholds that symbolic "
    "rules test against, without needing to touch the Attack file's Normal rows "
    "(which are entangled with the same file as the Attack rows and the eventual "
    "split boundaries).",
    "Low-to-moderate. Risk appears only if thresholds are tuned iteratively "
    "against downstream detection performance rather than fixed from descriptive "
    "statistics alone.",
    "RECOMMENDED, with thresholds fixed once (e.g. percentile bounds of each "
    "sensor's Normal-only distribution) before any split's labels are used for "
    "evaluation.",
)

role_block(
    "Additional Normal training observations pooled directly into the supervised "
    "classifier's TRAIN split",
    "Superficially attractive: more Normal examples could improve the "
    "Normal-class decision boundary and reduce false positives.",
    "HIGH -- this is the file-vs-class confound in its most direct form. Every "
    "row contributed by the Normal-only file has label=Normal AND file-origin="
    "Normal-only simultaneously; a sufficiently flexible model (especially "
    "Random Forest, which can split on any feature interaction) can partially "
    "learn to recognize 'Normal-only-file-era process drift' rather than "
    "'Normal class' per se, then fail to generalize that recognition to Normal "
    "rows drawn from the Attack file's own Normal spans (different date range, "
    "different ambient drift). This would inflate apparent Normal-class recall "
    "without genuinely improving attack discrimination.",
    "NOT RECOMMENDED as direct supervised training rows unless the paper "
    "explicitly models and reports file-origin as a potential confound (e.g. via "
    "a domain-adaptation framing) -- which is substantial extra scope beyond the "
    "current design.",
)

role_block(
    "Not using it in the supervised classifier at all -- reserve it purely for "
    "documentation/EDA and the roles above",
    "Cleanest option with respect to the audit's core finding: the supervised "
    "RF/LSTM pipeline is trained and evaluated exclusively on chronologically "
    "split portions of the single Attack file, which is the only file containing "
    "both classes, eliminating the file-vs-class confound entirely by "
    "construction.",
    "None from this choice itself. The only 'cost' is not using ~497K rows of "
    "available data for supervised training, which is a reasonable trade for "
    "avoiding a confound that would otherwise need to be explicitly modeled and "
    "defended to reviewers.",
    "RECOMMENDED as the default stance for the SUPERVISED classifier. Combine "
    "with roles 1, 3 and (conditionally) 2 above for a scientifically clean "
    "reservation-of-purpose story: Normal-only file supports baseline "
    "statistics / symbolic thresholds / optional label-free pretraining, but "
    "contributes zero rows to supervised TRAIN, VALIDATION, or TEST labels.",
)

normal_path = AUDIT_DIR / "normal_dataset_role_analysis.txt"
normal_path.write_text("\n".join(normal_lines), encoding="utf-8")
log(f"Wrote {normal_path}")

# ---------------------------------------------------------------------------
# 8. Sections 7 & 8 — feature-selection protocol + preprocessing order
# ---------------------------------------------------------------------------
section("WRITING preprocessing_order_analysis.txt")

prep_lines = []
prep_lines.append("FEATURE-SELECTION PROTOCOL AND PREPROCESSING ORDER -- ANALYSIS ONLY")
prep_lines.append(f"Generated: {RUN_START.isoformat()}")
prep_lines.append("")
prep_lines.append("=" * 78)
prep_lines.append("PART A -- FEATURE-SELECTION PROTOCOL, PER CANDIDATE (Task 7)")
prep_lines.append("=" * 78)
prep_lines.append("")
prep_lines.append("For every candidate defined above, whenever Random Forest feature ranking")
prep_lines.append("is eventually run (NOT in this step), it is allowed to see ONLY the rows in")
prep_lines.append("that candidate's TRAIN split -- i.e., strictly the rows with timestamp <")
prep_lines.append("the candidate's train/val boundary. Concretely:")
prep_lines.append("")
for cname, spec in CANDIDATES.items():
    prep_lines.append(f"  {cname}: RF may see rows in "
                       f"[{file_start}, {spec['train_val_boundary']}) only.")
prep_lines.append("")
prep_lines.append("RF feature ranking must NOT see, directly or indirectly:")
prep_lines.append("  - any VALIDATION row (timestamp in [train/val boundary, val/test boundary))")
prep_lines.append("  - any TEST row (timestamp >= val/test boundary)")
prep_lines.append("  - any statistic computed by pooling TRAIN with VALIDATION or TEST (e.g. a")
prep_lines.append("    global mean/std computed over the whole Attack file before splitting)")
prep_lines.append("  - the label or timestamp columns as candidate input features (both are")
prep_lines.append("    excluded per the leakage assessment from the prior audit stage)")
prep_lines.append("")
prep_lines.append("VALIDATION DATA AND FEATURE RANKING -- explicit answer:")
prep_lines.append("VALIDATION must remain SEPARATE from Random Forest feature ranking. It is")
prep_lines.append("legitimate to use VALIDATION for choosing Top-k (i.e. comparing downstream")
prep_lines.append("LSTM performance across a few candidate k values and picking the best on")
prep_lines.append("VALIDATION), but the RF importance scores themselves, and the TRAIN-only")
prep_lines.append("fitting of the RF model that produces them, must be computed from TRAIN")
prep_lines.append("rows exclusively. If VALIDATION rows influence the importance ranking itself")
prep_lines.append("(rather than just the choice of k given a fixed ranking), VALIDATION is no")
prep_lines.append("longer an independent check on Top-k selection and TEST becomes the only")
prep_lines.append("remaining unbiased evaluation -- which then cannot also be used to sanity")
prep_lines.append("check Top-k after the fact.")
prep_lines.append("")
prep_lines.append("Choice of Top-k, and any hyperparameter (RF n_estimators/depth, LSTM units/")
prep_lines.append("layers/learning rate/sequence length), must be selected by comparing")
prep_lines.append("performance on VALIDATION across candidate settings -- never on TEST. TEST")
prep_lines.append("is evaluated exactly once, after every modeling decision (features, k,")
prep_lines.append("hyperparameters) is frozen.")
prep_lines.append("")
prep_lines.append("=" * 78)
prep_lines.append("PART B -- CORRECT PREPROCESSING ORDER FOR THE EVENTUAL CHOSEN SPLIT (Task 8)")
prep_lines.append("=" * 78)
prep_lines.append("")
prep_lines.append("Order of operations, with each step tagged by what data it is allowed to use:")
prep_lines.append("")
prep_lines.append("1. RAW-FORMAT CORRECTIONS  [MAY use the entire dataset -- deterministic,")
prep_lines.append("   label-independent structural fixes]")
prep_lines.append("   - Skip the Normal file's process-stage header row; handle the Attack")
prep_lines.append("     file's UTF-8 BOM. Both are fixed, mechanical parsing corrections that")
prep_lines.append("     do not depend on any row's label or on train/val/test membership.")
prep_lines.append("   - Reconcile the 7 column-name whitespace mismatches between files (e.g.")
prep_lines.append("     Normal's 'MV101' vs Attack's ' MV101') to a single canonical name.")
prep_lines.append("     Purely a string-rewrite of a column NAME, not a value; identical result")
prep_lines.append("     regardless of split, so may be applied dataset-wide.")
prep_lines.append("   - Parse the raw timestamp strings into datetime objects. Deterministic")
prep_lines.append("     format conversion; identical for every row regardless of split.")
prep_lines.append("")
prep_lines.append("2. LABEL NORMALIZATION  [MAY use the entire dataset -- deterministic mapping")
prep_lines.append("   decided a priori, not learned from data]")
prep_lines.append("   - Decide, once and explicitly (document in the paper), how 'A ttack' (the")
prep_lines.append("     37-row typo) is mapped -- almost certainly to 'Attack' given it is a")
prep_lines.append("     clear data-entry artifact, not a third class. This is a fixed rule, not")
prep_lines.append("     a statistic fit from data, so applying it dataset-wide before splitting")
prep_lines.append("     does not leak anything -- the rule does not depend on which split a row")
prep_lines.append("     later falls into.")
prep_lines.append("")
prep_lines.append("3. CHRONOLOGICAL SPLIT  [defines TRAIN/VALIDATION/TEST membership; every")
prep_lines.append("   subsequent step must respect it]")
prep_lines.append("   - Apply the chosen candidate's boundaries to assign every row to TRAIN,")
prep_lines.append("     VALIDATION, or TEST. From this point forward, no statistic, model fit,")
prep_lines.append("     or selection may be computed by pooling across these three groups.")
prep_lines.append("")
prep_lines.append("4. FEATURE EXCLUSION  [MAY use the entire dataset -- deterministic, decided")
prep_lines.append("   a priori from the leakage assessment, not learned from data]")
prep_lines.append("   - Drop the label column and the timestamp column from the model INPUT "
                   "feature set (timestamp is retained separately for ordering/windowing "
                   "only). This is a fixed column-list decision, not a fitted statistic.")
prep_lines.append("")
prep_lines.append("5. SCALER FITTING  [MUST use TRAIN ONLY]")
prep_lines.append("   - Fit mean/std (or min/max) exclusively on TRAIN rows' physical process")
prep_lines.append("     variables. VALIDATION and TEST must never contribute to the fitted")
prep_lines.append("     scaler parameters.")
prep_lines.append("")
prep_lines.append("6. SCALER TRANSFORMATION  [Applies the TRAIN-fitted scaler to all three")
prep_lines.append("   splits, but the fit itself came from TRAIN only in step 5]")
prep_lines.append("   - Transform TRAIN, VALIDATION, and TEST using the SAME parameters learned")
prep_lines.append("     in step 5. Applying a fixed, already-fitted transform to VALIDATION/TEST")
prep_lines.append("     is not leakage; RE-FITTING it on VALIDATION/TEST would be.")
prep_lines.append("")
prep_lines.append("7. RANDOM FOREST FEATURE RANKING  [MUST use TRAIN ONLY]")
prep_lines.append("   - Fit RF and derive importances using only scaled TRAIN rows, per Part A")
prep_lines.append("     above.")
prep_lines.append("")
prep_lines.append("8. TOP-K SELECTION  [decide using VALIDATION performance, not TEST]")
prep_lines.append("   - Given the TRAIN-derived ranking, choose k by comparing downstream")
prep_lines.append("     (e.g. LSTM) performance on VALIDATION across a small number of")
prep_lines.append("     candidate k values fixed in advance.")
prep_lines.append("")
prep_lines.append("9. LSTM SEQUENCE CONSTRUCTION  [structural windowing MAY reference the full")
prep_lines.append("   timeline for ordering, but windows must not straddle split boundaries or")
prep_lines.append("   Normal/Attack-file boundaries, and must be BUILT from already scaled,")
prep_lines.append("   feature-selected, split-respecting data]")
prep_lines.append("   - Build sliding windows independently within TRAIN, within VALIDATION, and")
prep_lines.append("     within TEST, using only the Top-k features selected in step 8 and the")
prep_lines.append("     TRAIN-fitted scaling from step 6. No window may include timesteps from")
prep_lines.append("     two different splits, per the boundary-check analysis above.")
prep_lines.append("")
prep_lines.append("SUMMARY -- steps that may use the WHOLE dataset (1, 2, 4) are all")
prep_lines.append("deterministic, a-priori, label/split-independent structural corrections.")
prep_lines.append("Steps that MUST be learned/decided from TRAIN (or TRAIN+VALIDATION for")
prep_lines.append("hyperparameter choice) only are 5, 6 (fit), 7, 8. Step 3 is the hinge point:")
prep_lines.append("nothing after it may pool across TRAIN/VALIDATION/TEST again.")

prep_path = AUDIT_DIR / "preprocessing_order_analysis.txt"
prep_path.write_text("\n".join(prep_lines), encoding="utf-8")
log(f"Wrote {prep_path}")

# ---------------------------------------------------------------------------
# Wrap up
# ---------------------------------------------------------------------------
section("ANALYSIS COMPLETE — NO SPLIT IMPLEMENTED, NO MODEL TRAINED")
run_end = datetime.now()
log(f"Run end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")

log_path = LOGS_DIR / f"01_split_candidate_analysis_{RUN_START.strftime('%Y%m%d_%H%M%S')}.log"
log_path.write_text("\n".join(_LOG_LINES), encoding="utf-8")
print(f"\nFull run log written to: {log_path}")
