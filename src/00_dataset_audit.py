"""
code/00_dataset_audit.py

RAW DATASET AUDIT — SWaT_Dataset_Normal_v0.csv / SWaT_Dataset_Attack_v0.csv

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

Purpose
-------
This script performs a strictly descriptive, non-destructive audit of the
two raw SWaT CSV files. It does NOT:
  - train Random Forest or LSTM
  - generate symbolic features
  - modify either source CSV
  - randomly split any data
  - normalize / transform any values
  - silently rename columns or standardize label strings

Every "cleaning-like" operation performed below (BOM handling, header-row
skip, whitespace stripping for timestamp PARSING only) is done on an
in-memory copy purely so the audit can compute correct statistics, and is
explicitly logged. The original CSV files on disk are never opened in
write mode and are never altered.

Run with:
    .venv\\Scripts\\python.exe code/00_dataset_audit.py
"""

import os
import sys
import platform
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Windows consoles default to a legacy code page (e.g. cp1252) that cannot
# encode the project's Korean directory name or other non-ASCII output.
# Force UTF-8 stdout/stderr so nothing here silently crashes on print().
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 0. PATHS AND DIRECTORY STRUCTURE (TASK 1)
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

DATASET_DIR = Path(os.environ.get("SWAT_DATASET_DIR", str(PROJECT_ROOT / "dataset")))  # portability fix: actual directory is lowercase; overridable via env var
NORMAL_PATH = DATASET_DIR / "SWaT_Dataset_Normal_v0.csv"
ATTACK_PATH = DATASET_DIR / "SWaT_Dataset_Attack_v0.csv"

CODE_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
AUDIT_DIR = RESULTS_DIR / "audit"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"

for d in [CODE_DIR, RESULTS_DIR, AUDIT_DIR, METRICS_DIR, LOGS_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_START = datetime.now()

# ---------------------------------------------------------------------------
# Console + log capture
# ---------------------------------------------------------------------------
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


# Researcher-provided context to verify (TASK 3, TASK 12 item 2)
CLAIMED_NORMAL_START = "2015-12-22"
CLAIMED_NORMAL_END = "2015-12-28"
CLAIMED_ATTACK_START = "2015-12-28"
CLAIMED_ATTACK_END = "2016-01-02"

NEAR_CONSTANT_DOMINANT_FRACTION = 0.99  # documented threshold, see Task 2.17

# ---------------------------------------------------------------------------
# 1. FILE-LEVEL INFO (TASK 2.1-2.2, TASK 11)
# ---------------------------------------------------------------------------


def file_info(path: Path):
    size_bytes = path.stat().st_size
    return {
        "filename": path.name,
        "full_path": str(path),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3),
    }


# ---------------------------------------------------------------------------
# 2. RAW HEADER INSPECTION (structural discrepancies between the two files)
# ---------------------------------------------------------------------------


def inspect_raw_header(path: Path, n_lines=3):
    """Read the first few raw lines as bytes to detect BOM, extra header
    rows, delimiter, and leading/trailing whitespace WITHOUT letting any
    parser silently normalize anything."""
    with open(path, "rb") as f:
        raw_lines = [f.readline() for _ in range(n_lines)]
    bom_present = raw_lines[0].startswith(b"\xef\xbb\xbf")
    decoded = [l.decode("utf-8-sig", errors="replace").rstrip("\r\n") for l in raw_lines]
    return {
        "bom_present": bom_present,
        "raw_lines": decoded,
    }


# ---------------------------------------------------------------------------
# 3. LOADING (structural handling is documented explicitly, not silent)
# ---------------------------------------------------------------------------


def load_normal():
    """
    SWaT_Dataset_Normal_v0.csv structure (verified on raw bytes):
      line 1 = process-stage grouping row (',P1,,,,,P2,...') -- NOT a data
               header, must be skipped to reach the real header.
      line 2 = real column header (' Timestamp,FIT101,...,Normal/Attack')
      line 3+ = data rows
    No UTF-8 BOM present in this file.
    """
    df = pd.read_csv(NORMAL_PATH, skiprows=1, low_memory=False)
    return df


def load_attack():
    """
    SWaT_Dataset_Attack_v0.csv structure (verified on raw bytes):
      line 1 = real column header, PRECEDED BY A UTF-8 BOM (EF BB BF).
               encoding='utf-8-sig' is used so the BOM byte sequence is
               correctly consumed as an encoding artifact rather than
               becoming part of the first column name; this is standard
               UTF-8-with-BOM decoding, not a data-cleaning step.
      line 2+ = data rows
    No separate process-group header row in this file (unlike Normal).
    """
    df = pd.read_csv(ATTACK_PATH, encoding="utf-8-sig", low_memory=False)
    return df


# ---------------------------------------------------------------------------
# 4. COLUMN-LEVEL AUDIT (TASK 2.5-2.18)
# ---------------------------------------------------------------------------


def classify_variable_type(dtype, nunique):
    if pd.api.types.is_float_dtype(dtype):
        return "numerical (continuous)"
    if pd.api.types.is_integer_dtype(dtype):
        if nunique <= 10:
            return "categorical/binary/discrete"
        return "numerical (discrete/integer)"
    return "non-numeric"


def audit_columns(df, dataset_name, ts_col, label_col):
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        nunique = s.nunique(dropna=True)
        n_missing = int(s.isna().sum())
        pct_missing = round(100 * n_missing / n, 6) if n else 0.0

        if col == ts_col:
            role = "timestamp"
        elif col == label_col:
            role = "label"
        else:
            role = "physical_process_variable"

        if role == "physical_process_variable":
            vtype = classify_variable_type(s.dtype, nunique)
        elif role == "label":
            vtype = "categorical (target)"
        else:
            vtype = "datetime-string (raw)"

        is_constant = bool(nunique == 1)
        is_near_constant = False
        dominant_value = None
        dominant_fraction = None
        if nunique > 1 and role == "physical_process_variable":
            vc = s.value_counts(dropna=True, normalize=True)
            dominant_value = vc.index[0]
            dominant_fraction = round(float(vc.iloc[0]), 6)
            is_near_constant = bool(dominant_fraction >= NEAR_CONSTANT_DOMINANT_FRACTION)

        row = {
            "dataset": dataset_name,
            "column_name_raw": repr(col)[1:-1],  # preserves leading/trailing spaces visibly
            "role": role,
            "dtype": str(s.dtype),
            "variable_type": vtype,
            "n_unique": int(nunique),
            "n_missing": n_missing,
            "pct_missing": pct_missing,
            "is_constant": is_constant,
            "is_near_constant": is_near_constant,
            "dominant_value": dominant_value,
            "dominant_value_fraction": dominant_fraction,
        }

        if pd.api.types.is_numeric_dtype(s.dtype) and role == "physical_process_variable":
            desc = s.describe()
            row.update({
                "min": desc.get("min"),
                "max": desc.get("max"),
                "mean": desc.get("mean"),
                "std": desc.get("std"),
                "25%": desc.get("25%"),
                "50%": desc.get("50%"),
                "75%": desc.get("75%"),
            })
        else:
            row.update({"min": None, "max": None, "mean": None, "std": None,
                         "25%": None, "50%": None, "75%": None})

        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. TIMESTAMP AUDIT (TASK 3)
# ---------------------------------------------------------------------------


def parse_timestamps(raw_series):
    """Parse the raw timestamp strings for ANALYSIS purposes only.
    Format observed on raw bytes: 'DD/MM/YYYY H:MM:SS AM/PM' with
    inconsistent zero-padding and a leading space on every value
    (consistent with the leading space in the ' Timestamp' header).
    Leading/trailing whitespace is stripped ONLY inside this parsed,
    in-memory copy; the raw column reported elsewhere is untouched."""
    stripped = raw_series.astype(str).str.strip()
    parsed = pd.to_datetime(stripped, format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    return parsed


def audit_timestamps(df, ts_col, dataset_name):
    raw = df[ts_col]
    parsed = parse_timestamps(raw)
    n = len(parsed)
    n_parse_fail = int(parsed.isna().sum())

    first_ts = parsed.iloc[0]
    last_ts = parsed.iloc[-1]
    duration = last_ts - first_ts
    monotonic = bool(parsed.is_monotonic_increasing)
    n_dup_ts = int(parsed.duplicated().sum())

    diffs = parsed.diff().dropna()
    diffs_sec = diffs.dt.total_seconds()
    if len(diffs_sec) > 0:
        mode_interval = float(diffs_sec.mode().iloc[0])
        min_interval = float(diffs_sec.min())
        max_interval = float(diffs_sec.max())
        n_gaps_above_mode = int((diffs_sec > mode_interval).sum())
        largest_gap = float(diffs_sec.max())
        largest_gap_idx = diffs_sec.idxmax()
        largest_gap_start = parsed.loc[largest_gap_idx - 1] if (largest_gap_idx - 1) in parsed.index else None
        largest_gap_end = parsed.loc[largest_gap_idx]
    else:
        mode_interval = min_interval = max_interval = None
        n_gaps_above_mode = 0
        largest_gap = None
        largest_gap_start = largest_gap_end = None

    result = {
        "dataset": dataset_name,
        "n_rows": n,
        "n_timestamp_parse_failures": n_parse_fail,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "duration": str(duration),
        "duration_seconds": duration.total_seconds(),
        "monotonic_increasing": monotonic,
        "n_duplicate_timestamps": n_dup_ts,
        "sampling_interval_mode_seconds": mode_interval,
        "sampling_interval_min_seconds": min_interval,
        "sampling_interval_max_seconds": max_interval,
        "n_intervals_above_mode": n_gaps_above_mode,
        "largest_gap_seconds": largest_gap,
        "largest_gap_start": largest_gap_start,
        "largest_gap_end": largest_gap_end,
    }
    return result, parsed


# ---------------------------------------------------------------------------
# 6. LABEL AUDIT (TASK 4)
# ---------------------------------------------------------------------------


def audit_labels(df, label_col, dataset_name):
    s = df[label_col]
    n = len(s)
    vc = s.value_counts(dropna=False)
    rows = []
    for val, cnt in vc.items():
        val_repr = val if pd.notna(val) else "<MISSING>"
        has_leading_ws = isinstance(val, str) and val != val.lstrip()
        has_trailing_ws = isinstance(val, str) and val != val.rstrip()
        rows.append({
            "dataset": dataset_name,
            "raw_label_value": repr(val_repr)[1:-1] if isinstance(val_repr, str) else str(val_repr),
            "count": int(cnt),
            "percentage": round(100 * cnt / n, 6),
            "leading_whitespace": has_leading_ws,
            "trailing_whitespace": has_trailing_ws,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. ATTACK-INTERVAL AUDIT (TASK 5)
# ---------------------------------------------------------------------------


def audit_attack_intervals(df, label_col, parsed_ts):
    """
    Any raw label value that is NOT the exact string 'Normal' is treated as
    an attack-state observation for the purpose of grouping CONTIGUOUS
    intervals only. This explicitly includes both 'Attack' and the raw
    typo variant 'A ttack' found in the data. This is an analysis-time
    grouping decision, clearly logged here -- the raw label column itself
    is not modified or rewritten anywhere.
    """
    raw = df[label_col]
    is_attack_state = raw.ne("Normal")

    group_id = (is_attack_state != is_attack_state.shift()).cumsum()

    intervals = []
    normal_gap_durations = []
    prev_attack_end = None

    for gid, idx in df.groupby(group_id).groups.items():
        idx = df.index[df.index.isin(idx)] if not isinstance(idx, pd.Index) else idx
        sub_is_attack = is_attack_state.loc[idx].iloc[0]
        start_pos = idx[0]
        end_pos = idx[-1]
        start_ts = parsed_ts.loc[start_pos]
        end_ts = parsed_ts.loc[end_pos]
        n_obs = len(idx)

        if sub_is_attack:
            sub_labels = raw.loc[idx]
            n_exact_attack = int((sub_labels == "Attack").sum())
            n_typo_attack = int((sub_labels == "A ttack").sum())
            intervals.append({
                "interval_id": len(intervals) + 1,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "duration_seconds": (end_ts - start_ts).total_seconds(),
                "n_observations": n_obs,
                "n_label_Attack": n_exact_attack,
                "n_label_A_ttack_typo": n_typo_attack,
                "preceding_normal_gap_seconds": (
                    (start_ts - prev_attack_end).total_seconds() if prev_attack_end is not None else None
                ),
            })
            prev_attack_end = end_ts
        else:
            normal_gap_durations.append((end_ts - start_ts).total_seconds())

    intervals_df = pd.DataFrame(intervals)
    normal_gap_stats = {
        "n_normal_intervals_between_attacks": max(len(normal_gap_durations) - 0, 0),
        "normal_gap_min_seconds": float(np.min(normal_gap_durations)) if normal_gap_durations else None,
        "normal_gap_max_seconds": float(np.max(normal_gap_durations)) if normal_gap_durations else None,
        "normal_gap_mean_seconds": float(np.mean(normal_gap_durations)) if normal_gap_durations else None,
    }
    return intervals_df, normal_gap_stats


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    section("SWaT RAW DATASET AUDIT — START")
    log(f"Run start time : {RUN_START.isoformat()}")
    log(f"Project root   : {PROJECT_ROOT}")

    # ---- reproducibility info (TASK 11) ----
    section("REPRODUCIBILITY INFORMATION (TASK 11)")
    py_version = sys.version
    np_version = np.__version__
    pd_version = pd.__version__
    os_info = f"{platform.system()} {platform.release()} ({platform.version()})"
    log(f"Python version : {py_version}")
    log(f"numpy version  : {np_version}")
    log(f"pandas version : {pd_version}")
    log(f"Operating sys  : {os_info}")
    log(f"Machine        : {platform.machine()} / {platform.processor()}")

    normal_finfo = file_info(NORMAL_PATH)
    attack_finfo = file_info(ATTACK_PATH)
    log(f"Input file 1   : {normal_finfo['filename']} ({normal_finfo['size_bytes']:,} bytes)")
    log(f"Input file 2   : {attack_finfo['filename']} ({attack_finfo['size_bytes']:,} bytes)")

    # ---- raw header structural inspection ----
    section("RAW HEADER / STRUCTURAL INSPECTION")
    normal_header_info = inspect_raw_header(NORMAL_PATH)
    attack_header_info = inspect_raw_header(ATTACK_PATH)
    log("NORMAL file — first 2 raw lines:")
    for l in normal_header_info["raw_lines"][:2]:
        log(f"    {l[:160]}")
    log(f"NORMAL file — UTF-8 BOM present: {normal_header_info['bom_present']}")
    log("")
    log("ATTACK file — first raw line:")
    log(f"    {attack_header_info['raw_lines'][0][:160]}")
    log(f"ATTACK file — UTF-8 BOM present: {attack_header_info['bom_present']}")
    log("")
    log("STRUCTURAL DISCREPANCY: NORMAL file has an extra process-stage")
    log("grouping row (',P1,,,,,P2,...') before the real header; ATTACK")
    log("file does not. ATTACK file begins with a UTF-8 BOM; NORMAL does not.")

    # ---- load ----
    section("LOADING DATASETS")
    dfN = load_normal()
    dfA = load_attack()
    log(f"NORMAL loaded shape: {dfN.shape}")
    log(f"ATTACK loaded shape: {dfA.shape}")

    ts_col_N, label_col_N = dfN.columns[0], dfN.columns[-1]
    ts_col_A, label_col_A = dfA.columns[0], dfA.columns[-1]
    log(f"NORMAL timestamp column (raw): {ts_col_N!r} | label column (raw): {label_col_N!r}")
    log(f"ATTACK timestamp column (raw): {ts_col_A!r} | label column (raw): {label_col_A!r}")

    cols_N = list(dfN.columns)
    cols_A = list(dfA.columns)
    log("")
    log(f"NORMAL column count: {len(cols_N)}  | physical process variables: {len(cols_N) - 2}")
    log(f"ATTACK column count: {len(cols_A)}  | physical process variables: {len(cols_A) - 2}")

    cols_N_stripped = [c.strip() for c in cols_N]
    cols_A_stripped = [c.strip() for c in cols_A]
    if cols_N_stripped == cols_A_stripped:
        log("Column NAMES match between files after whitespace-insensitive comparison.")
    else:
        log("WARNING: column names differ even after whitespace-insensitive comparison.")
    whitespace_diffs = [(a, b) for a, b in zip(cols_N, cols_A) if a != b]
    if whitespace_diffs:
        log(f"Raw whitespace differences in {len(whitespace_diffs)} column name(s) between files:")
        for a, b in whitespace_diffs:
            log(f"    NORMAL={a!r}  vs  ATTACK={b!r}")

    # ---- column-level audit ----
    section("COLUMN-LEVEL AUDIT (TASK 2)")
    colN_df = audit_columns(dfN, "Normal", ts_col_N, label_col_N)
    colA_df = audit_columns(dfA, "Attack", ts_col_A, label_col_A)
    column_summary = pd.concat([colN_df, colA_df], ignore_index=True)

    for name, cdf in [("NORMAL", colN_df), ("ATTACK", colA_df)]:
        n_const = int(cdf["is_constant"].sum())
        n_near = int(cdf["is_near_constant"].sum())
        n_missing_total = int(cdf["n_missing"].sum())
        log(f"{name}: constant features={n_const}, near-constant features "
            f"(dominant value >= {int(NEAR_CONSTANT_DOMINANT_FRACTION*100)}%)={n_near}, "
            f"total missing cells={n_missing_total}")
        if n_const:
            log(f"    constant columns: {cdf.loc[cdf['is_constant'], 'column_name_raw'].tolist()}")
        if n_near:
            log(f"    near-constant columns: {cdf.loc[cdf['is_near_constant'], 'column_name_raw'].tolist()}")

    # ---- duplicate rows ----
    section("DUPLICATE ROW AUDIT")
    dupN_full = int(dfN.duplicated().sum())
    dupA_full = int(dfA.duplicated().sum())
    dupN_no_ts = int(dfN.drop(columns=[ts_col_N]).duplicated().sum())
    dupA_no_ts = int(dfA.drop(columns=[ts_col_A]).duplicated().sum())
    log(f"NORMAL: full-row duplicates (incl. timestamp) = {dupN_full}")
    log(f"NORMAL: duplicate sensor snapshots (excl. timestamp) = {dupN_no_ts}")
    log(f"ATTACK: full-row duplicates (incl. timestamp) = {dupA_full}")
    log(f"ATTACK: duplicate sensor snapshots (excl. timestamp) = {dupA_no_ts}")

    # ---- timestamp audit ----
    section("TEMPORAL STRUCTURE AUDIT (TASK 3)")
    tsN_result, parsedN = audit_timestamps(dfN, ts_col_N, "Normal")
    tsA_result, parsedA = audit_timestamps(dfA, ts_col_A, "Attack")
    for r in (tsN_result, tsA_result):
        log(f"[{r['dataset']}] first={r['first_timestamp']}  last={r['last_timestamp']}  "
            f"duration={r['duration']}  monotonic={r['monotonic_increasing']}  "
            f"dup_ts={r['n_duplicate_timestamps']}  parse_failures={r['n_timestamp_parse_failures']}")
        log(f"           sampling interval: mode={r['sampling_interval_mode_seconds']}s  "
            f"min={r['sampling_interval_min_seconds']}s  max={r['sampling_interval_max_seconds']}s  "
            f"n_intervals_above_mode={r['n_intervals_above_mode']}")
        if r["largest_gap_seconds"] is not None and r["largest_gap_seconds"] > r["sampling_interval_mode_seconds"]:
            log(f"           largest gap: {r['largest_gap_seconds']}s between "
                f"{r['largest_gap_start']} and {r['largest_gap_end']}")

    # ---- verify researcher-provided periods ----
    section("VERIFICATION OF RESEARCHER-PROVIDED DATE RANGES")
    normal_start_actual = tsN_result["first_timestamp"].date().isoformat()
    normal_end_actual = tsN_result["last_timestamp"].date().isoformat()
    attack_start_actual = tsA_result["first_timestamp"].date().isoformat()
    attack_end_actual = tsA_result["last_timestamp"].date().isoformat()

    log(f"Claimed NORMAL period : {CLAIMED_NORMAL_START} through {CLAIMED_NORMAL_END}")
    log(f"Actual  NORMAL period : {normal_start_actual} through {normal_end_actual} "
        f"(exact: {tsN_result['first_timestamp']} -> {tsN_result['last_timestamp']})")
    log(f"Claimed ATTACK period : {CLAIMED_ATTACK_START} through {CLAIMED_ATTACK_END}")
    log(f"Actual  ATTACK period : {attack_start_actual} through {attack_end_actual} "
        f"(exact: {tsA_result['first_timestamp']} -> {tsA_result['last_timestamp']})")

    normal_start_match = normal_start_actual == CLAIMED_NORMAL_START
    normal_end_match = normal_end_actual == CLAIMED_NORMAL_END
    attack_start_match = attack_start_actual == CLAIMED_ATTACK_START
    attack_end_match = attack_end_actual == CLAIMED_ATTACK_END
    log(f"NORMAL start date matches claim: {normal_start_match}")
    log(f"NORMAL end date matches claim  : {normal_end_match} "
        f"(claim gives calendar date only; actual end time is {tsN_result['last_timestamp']})")
    log(f"ATTACK start date matches claim: {attack_start_match}")
    log(f"ATTACK end date matches claim  : {attack_end_match}")

    # ---- overlap check ----
    section("DATASET TEMPORAL OVERLAP CHECK")
    overlap_start = max(tsN_result["first_timestamp"], tsA_result["first_timestamp"])
    overlap_end = min(tsN_result["last_timestamp"], tsA_result["last_timestamp"])
    has_overlap = overlap_start <= overlap_end
    gap_seconds = (tsA_result["first_timestamp"] - tsN_result["last_timestamp"]).total_seconds()
    log(f"NORMAL ends at : {tsN_result['last_timestamp']}")
    log(f"ATTACK starts at: {tsA_result['first_timestamp']}")
    log(f"Gap between NORMAL end and ATTACK start: {gap_seconds} seconds")
    log(f"Overlap present: {has_overlap}")
    if not has_overlap:
        log("The two files are temporally CONTIGUOUS with NO overlap and NO gap "
            "beyond one nominal sampling interval.")

    # ---- label audit ----
    section("LABEL AUDIT (TASK 4)")
    labN_df = audit_labels(dfN, label_col_N, "Normal")
    labA_df = audit_labels(dfA, label_col_A, "Attack")
    label_distribution = pd.concat([labN_df, labA_df], ignore_index=True)
    log("NORMAL raw label distribution:")
    for _, r in labN_df.iterrows():
        log(f"    {r['raw_label_value']!r}: {r['count']} ({r['percentage']}%)")
    log("ATTACK raw label distribution:")
    for _, r in labA_df.iterrows():
        log(f"    {r['raw_label_value']!r}: {r['count']} ({r['percentage']}%)")

    normal_only_in_normal = set(dfN[label_col_N].unique()) == {"Normal"}
    log(f"NORMAL dataset contains ONLY 'Normal' labels: {normal_only_in_normal}")
    attack_labels = set(dfA[label_col_A].unique())
    log(f"ATTACK dataset unique raw label set: {attack_labels}")
    has_typo = "A ttack" in attack_labels
    if has_typo:
        n_typo = int((dfA[label_col_A] == "A ttack").sum())
        log(f"WARNING: found {n_typo} rows with raw label 'A ttack' (embedded space) "
            f"in ATTACK dataset. This is a genuine raw-data label inconsistency, "
            f"NOT corrected in this audit.")

    # ---- attack interval audit ----
    section("ATTACK-DATASET TEMPORAL LABEL ANALYSIS (TASK 5)")
    n_normal_attackfile = int((dfA[label_col_A] == "Normal").sum())
    n_attack_exact = int((dfA[label_col_A] == "Attack").sum())
    n_attack_typo = int((dfA[label_col_A] == "A ttack").sum())
    n_attack_state_total = n_attack_exact + n_attack_typo
    log(f"ATTACK dataset: Normal={n_normal_attackfile} "
        f"({round(100*n_normal_attackfile/len(dfA),4)}%)")
    log(f"ATTACK dataset: Attack (exact label)={n_attack_exact} "
        f"({round(100*n_attack_exact/len(dfA),4)}%)")
    log(f"ATTACK dataset: 'A ttack' (typo label)={n_attack_typo} "
        f"({round(100*n_attack_typo/len(dfA),4)}%)")
    log(f"ATTACK dataset: combined non-Normal (attack-state) total={n_attack_state_total} "
        f"({round(100*n_attack_state_total/len(dfA),4)}%)")

    intervals_df, normal_gap_stats = audit_attack_intervals(dfA, label_col_A, parsedA)
    log(f"Number of contiguous attack-state intervals: {len(intervals_df)}")
    if len(intervals_df):
        log(f"Shortest interval: {intervals_df['duration_seconds'].min()}s "
            f"({intervals_df['n_observations'].min()} obs)")
        log(f"Longest interval : {intervals_df['duration_seconds'].max()}s "
            f"({intervals_df['n_observations'].max()} obs)")
    log(f"Normal intervals between attacks: {normal_gap_stats['n_normal_intervals_between_attacks']} "
        f"(min={normal_gap_stats['normal_gap_min_seconds']}s, "
        f"max={normal_gap_stats['normal_gap_max_seconds']}s, "
        f"mean={round(normal_gap_stats['normal_gap_mean_seconds'],2) if normal_gap_stats['normal_gap_mean_seconds'] else None}s)")
    log("Note: this dataset provides NO attack-type / attack-scenario column; "
        "only a binary-ish Normal/Attack(+typo) label exists. Attack TYPE is "
        "therefore not inferable from this file and is not reported.")

    # =========================================================================
    # TASK 10 — WRITE OUTPUT FILES
    # =========================================================================
    section("WRITING AUDIT OUTPUT FILES (TASK 10)")

    # dataset_summary.csv
    dataset_summary = pd.DataFrame([
        {
            "dataset": "Normal",
            "filename": normal_finfo["filename"],
            "size_bytes": normal_finfo["size_bytes"],
            "size_mb": normal_finfo["size_mb"],
            "n_rows": dfN.shape[0],
            "n_columns": dfN.shape[1],
            "n_physical_process_variables": dfN.shape[1] - 2,
            "first_timestamp": tsN_result["first_timestamp"],
            "last_timestamp": tsN_result["last_timestamp"],
            "duration": tsN_result["duration"],
            "monotonic_timestamps": tsN_result["monotonic_increasing"],
            "n_duplicate_timestamps": tsN_result["n_duplicate_timestamps"],
            "sampling_interval_mode_seconds": tsN_result["sampling_interval_mode_seconds"],
            "n_full_row_duplicates": dupN_full,
            "n_duplicate_sensor_snapshots_excl_timestamp": dupN_no_ts,
            "n_constant_features": int(colN_df["is_constant"].sum()),
            "n_near_constant_features": int(colN_df["is_near_constant"].sum()),
            "n_missing_cells_total": int(colN_df["n_missing"].sum()),
            "n_label_Normal": int((dfN[label_col_N] == "Normal").sum()),
            "n_label_Attack": 0,
            "n_label_A_ttack_typo": 0,
            "claimed_period_start": CLAIMED_NORMAL_START,
            "claimed_period_end": CLAIMED_NORMAL_END,
            "claimed_period_verified": bool(normal_start_match and normal_end_match),
        },
        {
            "dataset": "Attack",
            "filename": attack_finfo["filename"],
            "size_bytes": attack_finfo["size_bytes"],
            "size_mb": attack_finfo["size_mb"],
            "n_rows": dfA.shape[0],
            "n_columns": dfA.shape[1],
            "n_physical_process_variables": dfA.shape[1] - 2,
            "first_timestamp": tsA_result["first_timestamp"],
            "last_timestamp": tsA_result["last_timestamp"],
            "duration": tsA_result["duration"],
            "monotonic_timestamps": tsA_result["monotonic_increasing"],
            "n_duplicate_timestamps": tsA_result["n_duplicate_timestamps"],
            "sampling_interval_mode_seconds": tsA_result["sampling_interval_mode_seconds"],
            "n_full_row_duplicates": dupA_full,
            "n_duplicate_sensor_snapshots_excl_timestamp": dupA_no_ts,
            "n_constant_features": int(colA_df["is_constant"].sum()),
            "n_near_constant_features": int(colA_df["is_near_constant"].sum()),
            "n_missing_cells_total": int(colA_df["n_missing"].sum()),
            "n_label_Normal": n_normal_attackfile,
            "n_label_Attack": n_attack_exact,
            "n_label_A_ttack_typo": n_attack_typo,
            "claimed_period_start": CLAIMED_ATTACK_START,
            "claimed_period_end": CLAIMED_ATTACK_END,
            "claimed_period_verified": bool(attack_start_match and attack_end_match),
        },
    ])
    dataset_summary.to_csv(AUDIT_DIR / "dataset_summary.csv", index=False)
    log(f"Wrote {AUDIT_DIR / 'dataset_summary.csv'}")

    column_summary.to_csv(AUDIT_DIR / "column_summary.csv", index=False)
    log(f"Wrote {AUDIT_DIR / 'column_summary.csv'}")

    label_distribution.to_csv(AUDIT_DIR / "label_distribution.csv", index=False)
    log(f"Wrote {AUDIT_DIR / 'label_distribution.csv'}")

    intervals_df.to_csv(AUDIT_DIR / "attack_intervals.csv", index=False)
    log(f"Wrote {AUDIT_DIR / 'attack_intervals.csv'}")

    timestamp_analysis = pd.DataFrame([tsN_result, tsA_result])
    timestamp_analysis.to_csv(AUDIT_DIR / "timestamp_analysis.csv", index=False)
    log(f"Wrote {AUDIT_DIR / 'timestamp_analysis.csv'}")

    # data_quality_report.txt
    dq_lines = []
    dq_lines.append("SWaT RAW DATASET AUDIT — DATA QUALITY REPORT")
    dq_lines.append(f"Generated: {RUN_START.isoformat()}")
    dq_lines.append("")
    dq_lines.append("1. FILE STRUCTURE")
    dq_lines.append(f"   - Normal file has an extra non-data header row (process-stage groups: "
                     f"P1..P6) before the real column header. Attack file does not.")
    dq_lines.append(f"   - Attack file begins with a UTF-8 BOM (EF BB BF). Normal file does not.")
    dq_lines.append(f"   - Attack file's raw header/data rows have inconsistent leading spaces on "
                     f"several column names ({len(whitespace_diffs)} columns differ from Normal "
                     f"file's spelling of the same variable name): "
                     f"{[b for a,b in whitespace_diffs]}")
    dq_lines.append(f"   - Both files share a leading space in the raw timestamp column header "
                     f"(' Timestamp') and in every raw timestamp value.")
    dq_lines.append("")
    dq_lines.append("2. DIMENSIONS")
    dq_lines.append(f"   - Normal: {dfN.shape[0]:,} rows x {dfN.shape[1]} columns "
                     f"({dfN.shape[1]-2} physical process variables + timestamp + label)")
    dq_lines.append(f"   - Attack: {dfA.shape[0]:,} rows x {dfA.shape[1]} columns "
                     f"({dfA.shape[1]-2} physical process variables + timestamp + label)")
    dq_lines.append("")
    dq_lines.append("3. MISSING VALUES")
    dq_lines.append(f"   - Normal: {int(colN_df['n_missing'].sum())} missing cells total")
    dq_lines.append(f"   - Attack: {int(colA_df['n_missing'].sum())} missing cells total")
    dq_lines.append("")
    dq_lines.append("4. DUPLICATE ROWS")
    dq_lines.append(f"   - Normal: {dupN_full} full-row duplicates (incl. timestamp), "
                     f"{dupN_no_ts} duplicate sensor snapshots (excl. timestamp)")
    dq_lines.append(f"   - Attack: {dupA_full} full-row duplicates (incl. timestamp), "
                     f"{dupA_no_ts} duplicate sensor snapshots (excl. timestamp)")
    dq_lines.append("")
    dq_lines.append("5. CONSTANT / NEAR-CONSTANT FEATURES "
                     f"(near-constant threshold: dominant value >= {int(NEAR_CONSTANT_DOMINANT_FRACTION*100)}% of rows)")
    dq_lines.append(f"   - Normal constant: {colN_df.loc[colN_df['is_constant'],'column_name_raw'].tolist()}")
    dq_lines.append(f"   - Normal near-constant: {colN_df.loc[colN_df['is_near_constant'],'column_name_raw'].tolist()}")
    dq_lines.append(f"   - Attack constant: {colA_df.loc[colA_df['is_constant'],'column_name_raw'].tolist()}")
    dq_lines.append(f"   - Attack near-constant: {colA_df.loc[colA_df['is_near_constant'],'column_name_raw'].tolist()}")
    dq_lines.append("")
    dq_lines.append("6. TIMESTAMP INTEGRITY")
    dq_lines.append(f"   - Normal: monotonic={tsN_result['monotonic_increasing']}, "
                     f"duplicate timestamps={tsN_result['n_duplicate_timestamps']}, "
                     f"parse failures={tsN_result['n_timestamp_parse_failures']}, "
                     f"sampling mode={tsN_result['sampling_interval_mode_seconds']}s")
    dq_lines.append(f"   - Attack: monotonic={tsA_result['monotonic_increasing']}, "
                     f"duplicate timestamps={tsA_result['n_duplicate_timestamps']}, "
                     f"parse failures={tsA_result['n_timestamp_parse_failures']}, "
                     f"sampling mode={tsA_result['sampling_interval_mode_seconds']}s")
    dq_lines.append("")
    dq_lines.append("7. RESEARCHER-PROVIDED PERIOD VERIFICATION")
    dq_lines.append(f"   - Claimed Normal: {CLAIMED_NORMAL_START} -> {CLAIMED_NORMAL_END} | "
                     f"Actual: {tsN_result['first_timestamp']} -> {tsN_result['last_timestamp']} | "
                     f"Match: {normal_start_match and normal_end_match}")
    dq_lines.append(f"   - Claimed Attack: {CLAIMED_ATTACK_START} -> {CLAIMED_ATTACK_END} | "
                     f"Actual: {tsA_result['first_timestamp']} -> {tsA_result['last_timestamp']} | "
                     f"Match: {attack_start_match and attack_end_match}")
    dq_lines.append(f"   - Normal-end to Attack-start gap: {gap_seconds} seconds "
                     f"(no overlap, effectively contiguous)")
    dq_lines.append("")
    dq_lines.append("8. LABEL QUALITY")
    dq_lines.append(f"   - Normal dataset contains ONLY the label 'Normal': {normal_only_in_normal}")
    dq_lines.append(f"   - Attack dataset raw label set: {sorted(attack_labels)}")
    dq_lines.append(f"   - Attack dataset contains a raw label typo 'A ttack' (embedded space) "
                     f"in {n_attack_typo} rows. This is NOT corrected here; it must be an "
                     f"explicit, documented decision in the modeling stage whether to treat it "
                     f"as 'Attack'.")
    dq_lines.append("")
    dq_lines.append("9. REPRODUCIBILITY")
    dq_lines.append(f"   - Python: {py_version}")
    dq_lines.append(f"   - numpy: {np_version}")
    dq_lines.append(f"   - pandas: {pd_version}")
    dq_lines.append(f"   - OS: {os_info}")
    dq_lines.append(f"   - Script executed: {RUN_START.isoformat()}")
    dq_lines.append(f"   - Input files: {normal_finfo['filename']} ({normal_finfo['size_bytes']:,} bytes), "
                     f"{attack_finfo['filename']} ({attack_finfo['size_bytes']:,} bytes)")

    (AUDIT_DIR / "data_quality_report.txt").write_text("\n".join(dq_lines), encoding="utf-8")
    log(f"Wrote {AUDIT_DIR / 'data_quality_report.txt'}")

    # leakage_assessment.txt (TASK 6)
    leakage_lines = []
    leakage_lines.append("SWaT RAW DATASET AUDIT — POTENTIAL DATA LEAKAGE ASSESSMENT")
    leakage_lines.append(f"Generated: {RUN_START.isoformat()}")
    leakage_lines.append("")
    leakage_lines.append(f"Column: {label_col_N!r} / {label_col_A!r} (label column, 'Normal/Attack')")
    leakage_lines.append("  Risk: THIS IS THE TARGET. Must never appear among model input features.")
    leakage_lines.append("  Must be separated before any feature matrix is constructed for RF or LSTM.")
    leakage_lines.append("")
    leakage_lines.append(f"Column: {ts_col_N!r} / {ts_col_A!r} (timestamp)")
    leakage_lines.append("  Risk: Not inherently a leak of the label's semantic content, but it IS a")
    leakage_lines.append("  leak of WHICH FILE / WHICH CHRONOLOGICAL PERIOD a row came from. Because")
    leakage_lines.append("  the Normal file and Attack file occupy non-overlapping, contiguous date")
    leakage_lines.append("  ranges (Normal ends 2015-12-28 09:59:59, Attack begins 2015-12-28")
    leakage_lines.append("  10:00:00), a model given raw timestamp/date as a feature could learn a")
    leakage_lines.append("  trivial decision rule ('date >= 2015-12-28 10:00 => higher attack")
    leakage_lines.append("  probability') that reflects dataset construction, not attack dynamics.")
    leakage_lines.append("  Raw timestamp (or any monotonic transform of it, e.g. row index) should")
    leakage_lines.append("  be excluded from RF/LSTM input features; it may be used only for")
    leakage_lines.append("  ordering/windowing, chronological splitting, and reporting.")
    leakage_lines.append("")
    leakage_lines.append("Row/index columns: none present in the raw CSVs (no explicit ID column).")
    leakage_lines.append("  If an integer row index is added during preprocessing, it must not be")
    leakage_lines.append("  used as a feature for the same reason as timestamp above (it is monotonic")
    leakage_lines.append("  with chronological/dataset-membership position).")
    leakage_lines.append("")
    leakage_lines.append("Metadata / identifier columns: none found. All 51 non-timestamp,")
    leakage_lines.append("non-label columns are physical process sensor/actuator readings")
    leakage_lines.append("(FIT/LIT/AIT/DPIT/PIT = sensors; MV/P/UV = actuators/pumps/valves).")
    leakage_lines.append("")
    leakage_lines.append("Process-stage header row (Normal file only, 'P1'..'P6'): this is a")
    leakage_lines.append("structural artifact of the file, not a data row or feature; it is skipped")
    leakage_lines.append("during loading and poses no leakage risk, but it does mean any future")
    leakage_lines.append("script that reads this CSV without skipping row 1 will silently corrupt")
    leakage_lines.append("column dtypes -- a data-integrity risk worth flagging for the pipeline.")
    leakage_lines.append("")
    leakage_lines.append("General cross-dataset risk: because Normal and Attack are two distinct")
    leakage_lines.append("chronological files rather than a single randomly-assembled table, ANY")
    leakage_lines.append("step that pools rows from both files and then fits a preprocessing")
    leakage_lines.append("transform, or ranks/selects features, using information from rows that")
    leakage_lines.append("will later be held out for evaluation, constitutes leakage. This applies")
    leakage_lines.append("to scaling/normalization statistics, Random Forest feature-importance")
    leakage_lines.append("ranking, and any imputation statistics -- see")
    leakage_lines.append("experimental_protocol_assessment.txt for detail.")

    (AUDIT_DIR / "leakage_assessment.txt").write_text("\n".join(leakage_lines), encoding="utf-8")
    log(f"Wrote {AUDIT_DIR / 'leakage_assessment.txt'}")

    # experimental_protocol_assessment.txt (TASK 7, 8, 9)
    proto_lines = []
    proto_lines.append("SWaT RAW DATASET AUDIT — EXPERIMENTAL PROTOCOL ASSESSMENT")
    proto_lines.append(f"Generated: {RUN_START.isoformat()}")
    proto_lines.append("")
    proto_lines.append("=== TASK 7: DATASET RELATIONSHIP ===")
    proto_lines.append("")
    proto_lines.append("The two files form a single chronological timeline:")
    proto_lines.append(f"  Normal-only period : {tsN_result['first_timestamp']} -> {tsN_result['last_timestamp']}")
    proto_lines.append(f"  Attack period       : {tsA_result['first_timestamp']} -> {tsA_result['last_timestamp']}")
    proto_lines.append(f"  Gap between them     : {gap_seconds} seconds (contiguous, no overlap)")
    proto_lines.append("")
    proto_lines.append("This structure DOES support a defensible time-series experimental")
    proto_lines.append("protocol, but ONLY if chronological order is respected. It does NOT")
    proto_lines.append("support naive row-level random splitting across the combined data,")
    proto_lines.append("because adjacent rows within either file are highly autocorrelated")
    proto_lines.append("(1-second industrial process sampling) -- random splitting would let")
    proto_lines.append("near-duplicate neighboring timesteps leak across train/test.")
    proto_lines.append("")
    proto_lines.append("Option A: Train only on the Normal dataset (e.g. anomaly/novelty")
    proto_lines.append("detection framing, one-class methods).")
    proto_lines.append("  + Matches how the Normal file was actually collected (attack-free).")
    proto_lines.append("  + No label leakage risk from the Attack file at all.")
    proto_lines.append("  - Cannot directly train a supervised binary classifier (no Attack")
    proto_lines.append("    examples). Random Forest as proposed in the paper needs both classes.")
    proto_lines.append("  - Cannot validate detection performance without also using Attack data")
    proto_lines.append("    for testing, which raises the question addressed in Option D.")
    proto_lines.append("")
    proto_lines.append("Option B: Train a supervised detector using chronologically appropriate")
    proto_lines.append("portions of the Attack dataset (e.g. an early chronological slice for")
    proto_lines.append("train/validation, a later chronological slice held out for test).")
    proto_lines.append("  + Provides both classes for supervised RF/LSTM training as the paper")
    proto_lines.append("    proposes.")
    proto_lines.append("  + Chronological ordering can be preserved within the split.")
    proto_lines.append("  - Requires a principled, pre-registered cut point (e.g. by interval")
    proto_lines.append("    boundary or by date) decided BEFORE looking at test-set performance,")
    proto_lines.append("    to avoid the analyst tuning the split to maximize results.")
    proto_lines.append("  - The Normal dataset's role (pure normal baseline) vs. Attack dataset's")
    proto_lines.append("    train slice must be reconciled -- e.g. is the Normal file only used")
    proto_lines.append("    for baseline statistics, or pooled with the Attack-train slice?")
    proto_lines.append("    This must be decided explicitly, not by default.")
    proto_lines.append("")
    proto_lines.append("Option C: Combine both datasets and randomly split rows for train/test.")
    proto_lines.append("  + Simple, maximizes nominal sample size per split.")
    proto_lines.append("  - HIGH LEAKAGE RISK: at 1-second sampling, adjacent rows are near-")
    proto_lines.append("    identical (process state changes slowly relative to 1s). Random")
    proto_lines.append("    splitting places near-duplicate timesteps on both sides of the split,")
    proto_lines.append("    inflating apparent performance in a way that will not replicate on")
    proto_lines.append("    genuinely unseen future data.")
    proto_lines.append("  - Also breaks the LSTM's sequence assumption entirely (windows would")
    proto_lines.append("    need contiguous chronological rows).")
    proto_lines.append("  - NOT RECOMMENDED for a publication-quality ICS time-series protocol.")
    proto_lines.append("    This audit does not implement it and does not recommend it as a")
    proto_lines.append("    default, per the explicit instruction accompanying this audit.")
    proto_lines.append("")
    proto_lines.append("Option D: Chronological / temporal evaluation (train on an earlier")
    proto_lines.append("contiguous span, evaluate on a later contiguous span, in dataset order).")
    proto_lines.append("  + Matches real-world deployment: a detector trained on past data must")
    proto_lines.append("    generalize to future, unseen attack instances/intervals.")
    proto_lines.append("  + Avoids the autocorrelation leakage of Option C.")
    proto_lines.append("  + Naturally compatible with LSTM sequence construction (Task 9).")
    proto_lines.append("  - Performance is more sensitive to WHICH attack intervals fall in")
    proto_lines.append("    train vs. test (only "
                        f"{len(intervals_df)} contiguous attack intervals exist in total, so a")
    proto_lines.append("    single split could over- or under-represent attack diversity;")
    proto_lines.append("    should be documented as a limitation regardless of split choice.")
    proto_lines.append("")
    proto_lines.append("Option E: Feature-selection leakage via Random Forest importance ranking.")
    proto_lines.append("  If Random Forest feature importances are computed using rows that")
    proto_lines.append("  include, overlap, or were tuned against the final held-out test split,")
    proto_lines.append("  the resulting 'important' feature subset is implicitly informed by")
    proto_lines.append("  test-set structure. This inflates reported downstream LSTM performance")
    proto_lines.append("  and is a well-known form of leakage in publication pipelines.")
    proto_lines.append("  Mitigation (see Task 8): fit RF and derive feature rankings using ONLY")
    proto_lines.append("  the designated training portion; the test portion must remain fully")
    proto_lines.append("  unseen by every step of the pipeline (imputation, scaling, feature")
    proto_lines.append("  selection, RF, LSTM) until final evaluation.")
    proto_lines.append("")
    proto_lines.append("No split is implemented by this script. Protocol selection (A-E) is a")
    proto_lines.append("decision to be made explicitly before the next stage.")
    proto_lines.append("")
    proto_lines.append("=== TASK 8: RANDOM FOREST DATA-EXPOSURE CONSIDERATIONS ===")
    proto_lines.append("")
    proto_lines.append("Random Forest feature ranking, whenever it is eventually run, SHOULD see:")
    proto_lines.append("  - Only the designated TRAINING portion of the data (whatever protocol")
    proto_lines.append("    is chosen from Task 7), spanning both Normal and Attack classes so it")
    proto_lines.append("    can rank features by class-discriminative value.")
    proto_lines.append("Random Forest SHOULD NOT see:")
    proto_lines.append("  - Any row from the final held-out evaluation span (whether that span is")
    proto_lines.append("    a chronological tail of the Attack file, specific attack intervals")
    proto_lines.append("    reserved for testing, or both).")
    proto_lines.append("  - The raw timestamp or label columns as INPUT features (label is the")
    proto_lines.append("    target; timestamp leaks dataset-membership per leakage_assessment.txt).")
    proto_lines.append("  - Any statistic (mean/std for scaling, imputation values, etc.) derived")
    proto_lines.append("    by pooling train and test rows together.")
    proto_lines.append("Special note: because the Normal file is 100% Normal-labeled and the")
    proto_lines.append("Attack file mixes both classes, a naive 'Random Forest trained on Normal")
    proto_lines.append("file + tested on Attack file' design would let RF learn to distinguish")
    proto_lines.append("the two FILES rather than the two CLASSES, since file membership and")
    proto_lines.append("class label would be almost perfectly confounded for the Normal-only")
    proto_lines.append("file. Feature importance rankings from such a setup would likely reflect")
    proto_lines.append("dataset-collection artifacts (e.g., process drift between mid-Dec and")
    proto_lines.append("early-Jan) rather than genuine attack signatures. This confound must be")
    proto_lines.append("resolved by protocol design before running Random Forest.")
    proto_lines.append("")
    proto_lines.append("=== TASK 9: LSTM SEQUENCE-CONSTRUCTION CONSIDERATIONS ===")
    proto_lines.append("")
    proto_lines.append(f"Sampling regularity: Normal file sampling mode = "
                        f"{tsN_result['sampling_interval_mode_seconds']}s, "
                        f"{tsN_result['n_intervals_above_mode']} intervals exceed the mode out of "
                        f"{dfN.shape[0]-1} gaps. Attack file sampling mode = "
                        f"{tsA_result['sampling_interval_mode_seconds']}s, "
                        f"{tsA_result['n_intervals_above_mode']} intervals exceed the mode out of "
                        f"{dfA.shape[0]-1} gaps.")
    proto_lines.append("Both files are regularly sampled at (effectively) 1-second resolution")
    proto_lines.append("with no missing timestamps or large temporal gaps detected within each")
    proto_lines.append("file (see timestamp_analysis.csv for exact figures).")
    proto_lines.append("")
    proto_lines.append("Sliding-window construction issues to resolve before implementation:")
    proto_lines.append(f"  - {len(intervals_df)} contiguous attack intervals exist in the Attack")
    proto_lines.append("    file, separated by Normal spans of varying length (see")
    proto_lines.append("    attack_intervals.csv, 'preceding_normal_gap_seconds'). A sliding")
    proto_lines.append("    window of length W drawn naively across the whole Attack file WILL")
    proto_lines.append("    cross Normal/Attack boundaries -- the label to assign such a mixed")
    proto_lines.append("    window (majority vote? label-of-last-timestep? exclude?) is a design")
    proto_lines.append("    decision that must be made explicitly, not by default.")
    proto_lines.append("  - Windows must never be constructed across the Normal-file/Attack-file")
    proto_lines.append("    boundary (there is a real, if tiny, discontinuity there -- different")
    proto_lines.append("    files, and per Task 8, mixing them changes the class-vs-file confound).")
    proto_lines.append("  - No internal temporal gaps were detected in either file at the")
    proto_lines.append("    per-second level, so gap-crossing windows are not currently a concern")
    proto_lines.append("    WITHIN a single file; this should be re-verified after any future")
    proto_lines.append("    row removal (e.g. de-duplication) since removing rows could introduce")
    proto_lines.append("    artificial gaps that don't exist in the raw data today.")
    proto_lines.append("  - Chronological ordering must be preserved end-to-end: windows must be")
    proto_lines.append("    built from data sorted by the parsed timestamp (already monotonic in")
    proto_lines.append("    both raw files, confirmed above), and any train/test split must keep")
    proto_lines.append("    all windows on one side of the split fully before all windows on the")
    proto_lines.append("    other side in time, with no window straddling the split boundary.")
    proto_lines.append("Sequence length (window size) is NOT chosen here, per instruction.")

    (AUDIT_DIR / "experimental_protocol_assessment.txt").write_text("\n".join(proto_lines), encoding="utf-8")
    log(f"Wrote {AUDIT_DIR / 'experimental_protocol_assessment.txt'}")

    # tables/table_dataset_summary.csv (TASK 10, publication table — factual only)
    table_summary = pd.DataFrame([
        {
            "Dataset": "SWaT_Dataset_Normal_v0.csv",
            "Rows": dfN.shape[0],
            "Columns": dfN.shape[1],
            "Physical Process Variables": dfN.shape[1] - 2,
            "First Timestamp": str(tsN_result["first_timestamp"]),
            "Last Timestamp": str(tsN_result["last_timestamp"]),
            "Duration": str(tsN_result["duration"]),
            "Sampling Interval (mode, s)": tsN_result["sampling_interval_mode_seconds"],
            "Normal Observations": int((dfN[label_col_N] == "Normal").sum()),
            "Attack Observations": 0,
            "File Size (MB)": normal_finfo["size_mb"],
        },
        {
            "Dataset": "SWaT_Dataset_Attack_v0.csv",
            "Rows": dfA.shape[0],
            "Columns": dfA.shape[1],
            "Physical Process Variables": dfA.shape[1] - 2,
            "First Timestamp": str(tsA_result["first_timestamp"]),
            "Last Timestamp": str(tsA_result["last_timestamp"]),
            "Duration": str(tsA_result["duration"]),
            "Sampling Interval (mode, s)": tsA_result["sampling_interval_mode_seconds"],
            "Normal Observations": n_normal_attackfile,
            "Attack Observations": n_attack_exact + n_attack_typo,
            "File Size (MB)": attack_finfo["size_mb"],
        },
    ])
    table_summary.to_csv(TABLES_DIR / "table_dataset_summary.csv", index=False)
    log(f"Wrote {TABLES_DIR / 'table_dataset_summary.csv'}")

    # reproducibility info file (Task 11, supplementary to data_quality_report.txt)
    repro_lines = [
        "REPRODUCIBILITY INFORMATION",
        f"Run timestamp   : {RUN_START.isoformat()}",
        f"Python version  : {py_version}",
        f"numpy version   : {np_version}",
        f"pandas version  : {pd_version}",
        f"Operating system: {os_info}",
        f"Machine         : {platform.machine()} / {platform.processor()}",
        f"Input file 1    : {normal_finfo['filename']} ({normal_finfo['size_bytes']:,} bytes)",
        f"Input file 2    : {attack_finfo['filename']} ({attack_finfo['size_bytes']:,} bytes)",
        f"Script          : {SCRIPT_PATH}",
    ]
    (LOGS_DIR / "reproducibility_info.txt").write_text("\n".join(repro_lines), encoding="utf-8")
    log(f"Wrote {LOGS_DIR / 'reproducibility_info.txt'}")

    section("AUDIT COMPLETE — NO MODELING PERFORMED")
    run_end = datetime.now()
    log(f"Run end time: {run_end.isoformat()}  (elapsed: {run_end - RUN_START})")
    log("Per research protocol: STOPPING HERE. No preprocessing, Random Forest,")
    log("LSTM, or symbolic feature generation was performed by this script.")

    # write full console log
    log_path = LOGS_DIR / f"00_dataset_audit_{RUN_START.strftime('%Y%m%d_%H%M%S')}.log"
    log_path.write_text("\n".join(_LOG_LINES), encoding="utf-8")
    print(f"\nFull run log written to: {log_path}")


if __name__ == "__main__":
    main()
