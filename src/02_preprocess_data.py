"""
code/02_preprocess_data.py

PREPROCESSING + MATERIALIZATION OF THE APPROVED CANDIDATE-1 CHRONOLOGICAL
TRAIN / VALIDATION / TEST SPLIT.

Paper: "Explainable Cyberattack Detection for Industrial Control Systems
Using Random Forest, LSTM, and Symbolic Feature Fusion"

This script:
  - loads Dataset/SWaT_Dataset_Attack_v0.csv READ-ONLY
  - applies deterministic, logged raw-format corrections
  - normalizes labels via an explicit fixed mapping
  - applies the APPROVED Candidate-1 chronological boundaries (no random
    splitting)
  - defines predictive input as the 51 physical process variables only
  - fits StandardScaler on TRAIN ONLY, applies it to TRAIN/VAL/TEST
  - saves scaled + unscaled processed splits and full reproducibility
    metadata
  - runs a battery of post-hoc validation checks

This script does NOT train Random Forest, does NOT train LSTM, does NOT
perform feature ranking or Top-k selection, does NOT generate symbolic
features, and NEVER opens either raw source CSV in write mode.
"""

import os
import sys
import platform
import hashlib
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.preprocessing import StandardScaler

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
NORMAL_PATH = DATASET_DIR / "SWaT_Dataset_Normal_v0.csv"

PROCESSED_DIR = PROJECT_ROOT / "processed"
META_DIR = PROCESSED_DIR / "metadata"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
for d in (PROCESSED_DIR, META_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RUN_START = datetime.now()
_LOG_LINES = []
_CORRECTIONS_LOG = []


def log(msg=""):
    msg = str(msg)
    print(msg)
    _LOG_LINES.append(msg)


def section(title):
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


def record_correction(msg):
    _CORRECTIONS_LOG.append(msg)
    log(f"[CORRECTION] {msg}")


def sha256_of(path: Path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# APPROVED CANDIDATE 1 BOUNDARIES (from reviewed/approved split analysis)
# ---------------------------------------------------------------------------
VAL_START = pd.Timestamp("2015-12-31 13:00:00")
TEST_START = pd.Timestamp("2016-01-01 08:00:00")

EXPECTED = {
    "train": {"rows": 270000, "normal": 222343, "attack": 47657, "intervals": 22},
    "val": {"rows": 68319, "normal": 66660, "attack": 1659, "intervals": 3},
    "test": {"rows": 111600, "normal": 106295, "attack": 5305, "intervals": 10},
}

section("PREPROCESSING RUN START")
log(f"Run start: {RUN_START.isoformat()}")
log(f"Project root: {PROJECT_ROOT}")

# ---------------------------------------------------------------------------
# Pre-run integrity snapshot of the raw source files (verified again at the
# very end to prove neither file was modified by this script).
# ---------------------------------------------------------------------------
section("PRE-RUN SOURCE FILE INTEGRITY SNAPSHOT")
attack_size_before = ATTACK_PATH.stat().st_size
normal_size_before = NORMAL_PATH.stat().st_size
attack_hash_before = sha256_of(ATTACK_PATH)
log(f"Attack file size (bytes): {attack_size_before:,}")
log(f"Attack file SHA-256     : {attack_hash_before}")
log(f"Normal file size (bytes): {normal_size_before:,}  (not modified by this script)")

# =============================================================================
# 2. RAW STRUCTURAL CORRECTIONS
# =============================================================================
section("TASK 2 — RAW STRUCTURAL CORRECTIONS")

# Correction 1: UTF-8 BOM handling on the Attack file.
with open(ATTACK_PATH, "rb") as f:
    first_bytes = f.read(3)
bom_present = first_bytes == b"\xef\xbb\xbf"
record_correction(
    f"Attack CSV UTF-8 BOM present at byte offset 0: {bom_present}. "
    f"Loaded with encoding='utf-8-sig' so the BOM is consumed as an encoding "
    f"artifact rather than becoming part of the first column name."
)
df = pd.read_csv(ATTACK_PATH, encoding="utf-8-sig", low_memory=False)
raw_columns = list(df.columns)
log(f"Loaded Attack CSV: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Correction 2: normalize leading/trailing whitespace in column names.
stripped_columns = [c.strip() for c in raw_columns]
changed = [(a, b) for a, b in zip(raw_columns, stripped_columns) if a != b]
df.columns = stripped_columns
record_correction(
    f"Stripped leading/trailing whitespace from {len(changed)} column name(s): "
    f"{[repr(a) + ' -> ' + repr(b) for a, b in changed]}"
)

ts_col = df.columns[0]
label_col = df.columns[-1]
assert ts_col == "Timestamp", f"Unexpected timestamp column name after stripping: {ts_col!r}"
assert label_col == "Normal/Attack", f"Unexpected label column name: {label_col!r}"
physical_cols = list(df.columns[1:-1])
log(f"Timestamp column (post-correction): {ts_col!r}")
log(f"Label column (post-correction)    : {label_col!r}")
log(f"Physical process variable columns : {len(physical_cols)}")
assert len(physical_cols) == 51, f"Expected 51 physical process variables, found {len(physical_cols)}"

# Correction 3: verify the 51 physical variable names are consistent with the
# Normal-only file's header (cross-file structural check; header line only,
# no Normal-file data rows are loaded or used).
with open(NORMAL_PATH, "rb") as f:
    f.readline()  # skip process-stage grouping row
    normal_header_line = f.readline().decode("utf-8").rstrip("\r\n")
normal_cols_stripped = [c.strip() for c in normal_header_line.split(",")]
normal_physical_cols = normal_cols_stripped[1:-1]
names_consistent = physical_cols == normal_physical_cols
record_correction(
    f"Cross-checked stripped Attack-file physical variable names against the "
    f"Normal-file header (structural check only, no Normal-file data rows "
    f"read): identical order and names = {names_consistent}."
)
assert names_consistent, "Physical variable name/order mismatch between Attack and Normal files."

# Correction 4: no duplicate column names after stripping.
dup_check = pd.Series(df.columns).duplicated().sum()
record_correction(f"Duplicate column names after whitespace normalization: {int(dup_check)} (expected 0).")
assert dup_check == 0

# Correction 5: parse timestamps.
raw_ts = df[ts_col].astype(str).str.strip()
df["_ts"] = pd.to_datetime(raw_ts, format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
n_parse_fail = int(df["_ts"].isna().sum())
record_correction(
    f"Parsed raw timestamp strings (format 'DD/MM/YYYY H:MM:SS AM/PM', leading "
    f"whitespace stripped for parsing only) into datetime64. Parse failures: "
    f"{n_parse_fail} (expected 0)."
)
assert n_parse_fail == 0
assert df["_ts"].is_monotonic_increasing, "Timestamps not monotonically increasing."
log(f"Parsed timestamp range: {df['_ts'].iloc[0]} -> {df['_ts'].iloc[-1]}")

log("")
log("No corrections beyond the five logged above were applied. In particular: "
    "physical process variable VALUES were not altered, reordered, imputed, or "
    "clipped in this section.")

# =============================================================================
# 3. LABEL NORMALIZATION
# =============================================================================
section("TASK 3 — LABEL NORMALIZATION")

raw_label_values = set(df[label_col].unique())
allowed = {"Normal", "Attack", "A ttack"}
unexpected = raw_label_values - allowed
log(f"Raw unique label values found: {sorted(raw_label_values)}")
if unexpected:
    raise ValueError(f"Unexpected raw label value(s) found, refusing to map silently: {unexpected}")
log(f"Verified: no raw label values outside the expected set {sorted(allowed)}.")

LABEL_MAP = {"Normal": 0, "Attack": 1, "A ttack": 1}
n_typo = int((df[label_col] == "A ttack").sum())
n_attack_exact = int((df[label_col] == "Attack").sum())
n_normal_raw = int((df[label_col] == "Normal").sum())
df["_label"] = df[label_col].map(LABEL_MAP).astype(int)

log(f"Applied fixed mapping: 'Normal' -> 0, 'Attack' -> 1, 'A ttack' -> 1")
log(f"  'Normal' rows mapped to 0        : {n_normal_raw:,}")
log(f"  'Attack' rows mapped to 1        : {n_attack_exact:,}")
log(f"  'A ttack' (malformed) mapped to 1: {n_typo:,}")
assert n_typo == 37, f"Expected exactly 37 'A ttack' malformed rows, found {n_typo}"
log(f"CONFIRMED: exactly 37 malformed 'A ttack' observations were mapped to Attack (1). "
    f"No rows were discarded -- row count before mapping ({len(df):,}) equals row count "
    f"after ({df['_label'].notna().sum():,}).")
assert df["_label"].isna().sum() == 0
assert set(df["_label"].unique()) <= {0, 1}

# =============================================================================
# 4. CHRONOLOGICAL SPLIT (APPROVED CANDIDATE 1)
# =============================================================================
section("TASK 4 — CHRONOLOGICAL SPLIT (APPROVED CANDIDATE 1)")

df = df.sort_values("_ts").reset_index(drop=True)
assert df["_ts"].is_monotonic_increasing

train_mask = df["_ts"] < VAL_START
val_mask = (df["_ts"] >= VAL_START) & (df["_ts"] < TEST_START)
test_mask = df["_ts"] >= TEST_START

train_df = df.loc[train_mask].reset_index(drop=True)
val_df = df.loc[val_mask].reset_index(drop=True)
test_df = df.loc[test_mask].reset_index(drop=True)

# --- verify no row omitted / no row duplicated across splits ---
n_total = len(df)
n_split_sum = len(train_df) + len(val_df) + len(test_df)
assert n_split_sum == n_total, f"Row accounting mismatch: {n_split_sum} != {n_total}"
overlap_check = (train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int))
assert (overlap_check == 1).all(), "Some row belongs to zero or more than one split."
log(f"Row accounting verified: train({len(train_df):,}) + val({len(val_df):,}) + "
    f"test({len(test_df):,}) = {n_split_sum:,} = total({n_total:,}). "
    f"Every row assigned to exactly one split.")

# --- verify chronological ordering within and across splits ---
for name, sub in [("train", train_df), ("val", val_df), ("test", test_df)]:
    assert sub["_ts"].is_monotonic_increasing, f"{name} split not chronologically sorted"
assert train_df["_ts"].iloc[-1] < val_df["_ts"].iloc[0]
assert val_df["_ts"].iloc[-1] < test_df["_ts"].iloc[0]
log("Chronological ordering verified: train fully precedes val, val fully precedes test, "
    "each split internally sorted.")

# --- verify no attack interval crosses a split boundary ---
is_attack = df["_label"].eq(1)
group_id = (is_attack != is_attack.shift()).cumsum()
n_boundary_crossings = 0
for gid, sub_idx in df.groupby(group_id).groups.items():
    sub = df.loc[sub_idx]
    if not sub["_label"].iloc[0] == 1:
        continue
    splits_touched = set()
    if (sub["_ts"].iloc[0] < VAL_START) != (sub["_ts"].iloc[-1] < VAL_START):
        splits_touched.add("train/val boundary")
    if (sub["_ts"].iloc[0] < TEST_START) != (sub["_ts"].iloc[-1] < TEST_START):
        splits_touched.add("val/test boundary")
    if splits_touched:
        n_boundary_crossings += 1
        log(f"  WARNING: attack interval {sub['_ts'].iloc[0]} -> {sub['_ts'].iloc[-1]} "
            f"crosses {splits_touched}")
assert n_boundary_crossings == 0, "An attack interval crosses a split boundary -- aborting."
log("Verified: no contiguous attack interval crosses the train/val or val/test boundary.")


def interval_count(sub):
    is_atk = sub["_label"].eq(1)
    if is_atk.sum() == 0:
        return 0
    gid = (is_atk != is_atk.shift()).cumsum()
    n = 0
    for _, idx in sub.groupby(gid).groups.items():
        if sub.loc[idx, "_label"].iloc[0] == 1:
            n += 1
    return n


results = {}
for name, sub in [("train", train_df), ("val", val_df), ("test", test_df)]:
    n_rows = len(sub)
    n_normal = int((sub["_label"] == 0).sum())
    n_attack = int((sub["_label"] == 1).sum())
    n_iv = interval_count(sub)
    results[name] = {
        "start": sub["_ts"].iloc[0], "end": sub["_ts"].iloc[-1],
        "rows": n_rows, "normal": n_normal, "attack": n_attack,
        "attack_pct": round(100 * n_attack / n_rows, 4), "intervals": n_iv,
    }
    log(f"{name.upper():5s}: {sub['_ts'].iloc[0]} -> {sub['_ts'].iloc[-1]} | "
        f"rows={n_rows:,} | normal={n_normal:,} | attack={n_attack:,} "
        f"({results[name]['attack_pct']}%) | complete attack intervals={n_iv}")

section("TASK 4b — VERIFYING AGAINST RESEARCHER-APPROVED EXPECTED STATISTICS")
all_match = True
for name in ["train", "val", "test"]:
    exp = EXPECTED[name]
    got = results[name]
    ok = (got["rows"] == exp["rows"] and got["normal"] == exp["normal"] and
          got["attack"] == exp["attack"] and got["intervals"] == exp["intervals"])
    all_match = all_match and ok
    log(f"{name.upper():5s}: expected rows={exp['rows']:,} normal={exp['normal']:,} "
        f"attack={exp['attack']:,} intervals={exp['intervals']} | "
        f"actual rows={got['rows']:,} normal={got['normal']:,} attack={got['attack']:,} "
        f"intervals={got['intervals']} | MATCH={ok}")
if not all_match:
    raise AssertionError("Computed split statistics do not match the researcher-approved "
                          "expected values -- aborting before any data is written.")
log("\nAll split statistics exactly match the approved audit's expected values.")

# =============================================================================
# 5. FEATURE DEFINITION
# =============================================================================
section("TASK 5 — FEATURE DEFINITION")
FEATURE_COLS = list(physical_cols)  # exactly the 51 physical process variables
log(f"Predictive feature set ({len(FEATURE_COLS)} columns): {FEATURE_COLS}")
log("Excluded from predictive input: Timestamp, Label, row index, any metadata.")
log("Timestamp is preserved separately (for later sequence construction / explainability).")
log("Label is preserved separately as the prediction target.")
assert len(FEATURE_COLS) == 51
for sub, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
    assert list(sub.columns[1:1 + 51]) == FEATURE_COLS or set(FEATURE_COLS).issubset(sub.columns), \
        f"{name}: feature columns missing"

# =============================================================================
# 6. SCALING (TRAIN-ONLY FIT)
# =============================================================================
section("TASK 6 — STANDARDSCALER FITTING (TRAIN ONLY)")

X_train_raw = train_df[FEATURE_COLS].to_numpy(dtype=np.float64)
X_val_raw = val_df[FEATURE_COLS].to_numpy(dtype=np.float64)
X_test_raw = test_df[FEATURE_COLS].to_numpy(dtype=np.float64)

scaler = StandardScaler()
scaler.fit(X_train_raw)  # <-- TRAIN ONLY. Never called on val/test.
log(f"StandardScaler.fit() called on TRAIN feature matrix only: shape {X_train_raw.shape}")
log("StandardScaler was NOT fit or refit on VALIDATION or TEST data at any point.")

# Independent numeric cross-check that the fitted parameters equal a manual
# TRAIN-only mean/std computation (proves no other rows influenced the fit).
# NOTE: sklearn's StandardScaler documented behavior substitutes scale_=1.0
# for any zero-variance ("constant") feature, to avoid a divide-by-zero
# during transform -- a naive manual std() gives exactly 0.0 for those same
# columns instead. This is expected, not a bug; the check below accounts for
# it explicitly rather than asserting a raw numeric match everywhere.
manual_mean = X_train_raw.mean(axis=0)
manual_std = X_train_raw.std(axis=0, ddof=0)
zero_var_mask = manual_std < 1e-10
zero_var_features = [f for f, z in zip(FEATURE_COLS, zero_var_mask) if z]
expected_scale = manual_std.copy()
expected_scale[zero_var_mask] = 1.0  # mirrors sklearn's _handle_zeros_in_scale
mean_match = np.allclose(scaler.mean_, manual_mean, rtol=1e-10, atol=1e-10)
std_match = np.allclose(scaler.scale_, expected_scale, rtol=1e-10, atol=1e-10)
log(f"Constant (zero-variance) TRAIN features detected: {zero_var_features} "
    f"({len(zero_var_features)} of {len(FEATURE_COLS)}). sklearn sets scale_=1.0 for "
    f"these (documented behavior) rather than 0.0, to avoid divide-by-zero on transform.")
log(f"Independent verification: scaler.mean_ matches manual TRAIN-only mean: {mean_match}")
log(f"Independent verification: scaler.scale_ matches manual TRAIN-only std "
    f"(zero-variance columns expected as 1.0, per sklearn): {std_match}")
assert mean_match and std_match, "Scaler parameters do not match TRAIN-only statistics!"

# Also prove it does NOT match if val/test rows were included (sanity contrast).
combined_mean = np.concatenate([X_train_raw, X_val_raw, X_test_raw], axis=0).mean(axis=0)
contaminated_would_differ = not np.allclose(scaler.mean_, combined_mean, rtol=1e-8, atol=1e-8)
log(f"Sanity check: scaler.mean_ DIFFERS from a train+val+test pooled mean "
    f"(confirms val/test did not leak into the fit): {contaminated_would_differ}")

X_train_scaled = scaler.transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)
log("Applied the SAME TRAIN-fitted scaler to transform TRAIN, VALIDATION, and TEST.")

scaler_path = PROCESSED_DIR / "scaler.joblib"
joblib.dump(scaler, scaler_path)
log(f"Saved fitted StandardScaler to {scaler_path}")

# =============================================================================
# 7. SAVE PROCESSED DATA
# =============================================================================
section("TASK 7 — SAVING PROCESSED SPLIT OUTPUTS")


def build_output_df(sub_df, X_scaled):
    out = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
    out.insert(0, "Timestamp", sub_df["_ts"].values)
    out["Label"] = sub_df["_label"].values
    return out


train_scaled_df = build_output_df(train_df, X_train_scaled)
val_scaled_df = build_output_df(val_df, X_val_scaled)
test_scaled_df = build_output_df(test_df, X_test_scaled)

train_scaled_path = PROCESSED_DIR / "train_scaled.csv"
val_scaled_path = PROCESSED_DIR / "validation_scaled.csv"
test_scaled_path = PROCESSED_DIR / "test_scaled.csv"
train_scaled_df.to_csv(train_scaled_path, index=False)
val_scaled_df.to_csv(val_scaled_path, index=False)
test_scaled_df.to_csv(test_scaled_path, index=False)
log(f"Wrote {train_scaled_path} ({train_scaled_df.shape})")
log(f"Wrote {val_scaled_path} ({val_scaled_df.shape})")
log(f"Wrote {test_scaled_path} ({test_scaled_df.shape})")

# unscaled versions preserved for later explainability (SHAP/symbolic rules
# need original physical units, not standardized values)


def build_unscaled_df(sub_df):
    out = sub_df[FEATURE_COLS].copy()
    out.insert(0, "Timestamp", sub_df["_ts"].values)
    out["Label"] = sub_df["_label"].values
    out["Label_raw"] = sub_df[label_col].values
    return out


train_unscaled_path = PROCESSED_DIR / "train_unscaled.csv"
val_unscaled_path = PROCESSED_DIR / "validation_unscaled.csv"
test_unscaled_path = PROCESSED_DIR / "test_unscaled.csv"
build_unscaled_df(train_df).to_csv(train_unscaled_path, index=False)
build_unscaled_df(val_df).to_csv(val_unscaled_path, index=False)
build_unscaled_df(test_df).to_csv(test_unscaled_path, index=False)
log(f"Wrote {train_unscaled_path}")
log(f"Wrote {val_unscaled_path}")
log(f"Wrote {test_unscaled_path}")
log("Unscaled splits retain the original raw label string (Label_raw) alongside the "
    "binary Label, for full explainability traceability.")

# =============================================================================
# 8. REPRODUCIBILITY METADATA
# =============================================================================
section("TASK 8 — SAVING REPRODUCIBILITY METADATA")

feature_list_path = META_DIR / "feature_list.txt"
feature_list_path.write_text("\n".join(FEATURE_COLS), encoding="utf-8")
log(f"Wrote {feature_list_path}")

label_mapping_path = META_DIR / "label_mapping.txt"
label_mapping_lines = [
    "LABEL MAPPING (fixed, applied deterministically before any split-specific step)",
    "'Normal'  -> 0",
    "'Attack'  -> 1",
    "'A ttack' -> 1   (37 malformed/typo raw label rows, confirmed and logged, NOT discarded)",
    "",
    f"Counts: Normal={n_normal_raw:,}, Attack(exact)={n_attack_exact:,}, "
    f"A_ttack(typo)={n_typo:,}, Total mapped to 1 (Attack)={n_attack_exact + n_typo:,}",
]
label_mapping_path.write_text("\n".join(label_mapping_lines), encoding="utf-8")
log(f"Wrote {label_mapping_path}")

scaler_stats_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "train_mean": scaler.mean_,
    "train_std": scaler.scale_,
    "train_var": scaler.var_,
})
scaler_stats_path = META_DIR / "scaler_statistics.csv"
scaler_stats_df.to_csv(scaler_stats_path, index=False)
log(f"Wrote {scaler_stats_path}")

split_stats_records = []
for name in ["train", "val", "test"]:
    r = results[name]
    split_stats_records.append({
        "split": name,
        "start_timestamp": r["start"], "end_timestamp": r["end"],
        "n_rows": r["rows"], "n_normal": r["normal"], "n_attack": r["attack"],
        "attack_pct": r["attack_pct"], "n_complete_attack_intervals": r["intervals"],
        "n_features": len(FEATURE_COLS),
    })
split_stats_df = pd.DataFrame(split_stats_records)
split_stats_path = META_DIR / "split_statistics.csv"
split_stats_df.to_csv(split_stats_path, index=False)
log(f"Wrote {split_stats_path}")

py_version = sys.version
np_version = np.__version__
pd_version = pd.__version__
sk_version = sklearn.__version__
os_info = f"{platform.system()} {platform.release()} ({platform.version()})"

# ---------------------------------------------------------------------------
# TASK 9 — VALIDATION CHECKS (run before writing the final summary so results
# can be embedded in it)
# ---------------------------------------------------------------------------
section("TASK 9 — POST-HOC VALIDATION CHECKS")

checks = {}

# 1. 51 predictive features exist in every split
checks["51_features_per_split"] = all(
    len(set(FEATURE_COLS) - set(d.columns)) == 0
    for d in (train_scaled_df, val_scaled_df, test_scaled_df)
) and len(FEATURE_COLS) == 51

# 2. feature ordering identical
cols_train = [c for c in train_scaled_df.columns if c in FEATURE_COLS]
cols_val = [c for c in val_scaled_df.columns if c in FEATURE_COLS]
cols_test = [c for c in test_scaled_df.columns if c in FEATURE_COLS]
checks["feature_ordering_identical"] = (cols_train == cols_val == cols_test == FEATURE_COLS)

# 3. no NaN / infinite values introduced by scaling
all_scaled = np.concatenate([X_train_scaled, X_val_scaled, X_test_scaled], axis=0)
checks["no_nan"] = not np.isnan(all_scaled).any()
checks["no_inf"] = not np.isinf(all_scaled).any()

# 4. binary targets contain only 0 and 1
all_labels = pd.concat([train_scaled_df["Label"], val_scaled_df["Label"], test_scaled_df["Label"]])
checks["labels_binary_only"] = set(all_labels.unique()) <= {0, 1}

# 5. timestamps remain chronologically sorted (per split)
checks["timestamps_sorted"] = all(
    d["Timestamp"].is_monotonic_increasing
    for d in (train_scaled_df, val_scaled_df, test_scaled_df)
)

# 6. no timestamp overlap between splits
checks["no_timestamp_overlap"] = (
    train_scaled_df["Timestamp"].max() < val_scaled_df["Timestamp"].min() and
    val_scaled_df["Timestamp"].max() < test_scaled_df["Timestamp"].min()
)

# 7. TEST statistics were never used to fit preprocessing parameters
test_mean = X_test_raw.mean(axis=0)
checks["test_not_used_in_fit"] = not np.allclose(scaler.mean_, test_mean, rtol=1e-6, atol=1e-6)

# 8. fitted scaler parameters came only from TRAIN
checks["scaler_from_train_only"] = mean_match and std_match

# 9. raw source files remain unchanged
attack_size_after = ATTACK_PATH.stat().st_size
attack_hash_after = sha256_of(ATTACK_PATH)
normal_size_after = NORMAL_PATH.stat().st_size
checks["raw_attack_file_unchanged"] = (
    attack_size_after == attack_size_before and attack_hash_after == attack_hash_before
)
checks["raw_normal_file_unchanged_size"] = (normal_size_after == normal_size_before)

for k, v in checks.items():
    log(f"  [{'PASS' if v else 'FAIL'}] {k}")

all_checks_passed = all(checks.values())
if not all_checks_passed:
    log("\nWARNING: one or more validation checks FAILED. See above.")
else:
    log("\nAll validation checks PASSED.")

# ---------------------------------------------------------------------------
# preprocessing_summary.txt
# ---------------------------------------------------------------------------
summary_lines = []
summary_lines.append("PREPROCESSING SUMMARY — SWaT Candidate-1 Chronological Split")
summary_lines.append(f"Generated: {RUN_START.isoformat()}")
summary_lines.append(f"Script: {SCRIPT_PATH}")
summary_lines.append("")
summary_lines.append("=== APPROVED SPLIT BOUNDARIES ===")
summary_lines.append(f"TRAIN : [{results['train']['start']}, {VAL_START}) exclusive upper bound")
summary_lines.append(f"VAL   : [{VAL_START}, {TEST_START}) exclusive upper bound")
summary_lines.append(f"TEST  : [{TEST_START}, {results['test']['end']}] inclusive")
summary_lines.append("")
summary_lines.append("=== SPLIT STATISTICS ===")
for name in ["train", "val", "test"]:
    r = results[name]
    summary_lines.append(
        f"{name.upper():5s}: {r['start']} -> {r['end']} | rows={r['rows']:,} | "
        f"normal={r['normal']:,} | attack={r['attack']:,} | attack%={r['attack_pct']} | "
        f"complete attack intervals={r['intervals']}"
    )
summary_lines.append("")
summary_lines.append(f"Total rows processed: {n_total:,} (train+val+test, fully accounted)")
summary_lines.append(f"Predictive feature count: {len(FEATURE_COLS)} (51 physical process variables)")
summary_lines.append("")
summary_lines.append("=== CORRECTIONS APPLIED (raw-format, deterministic) ===")
for c in _CORRECTIONS_LOG:
    summary_lines.append(f"  - {c}")
summary_lines.append("")
summary_lines.append("=== LABEL NORMALIZATION ===")
summary_lines.append("Fixed mapping: 'Normal'->0, 'Attack'->1, 'A ttack'->1 (37 malformed rows, "
                      "not discarded, confirmed).")
summary_lines.append("")
summary_lines.append("=== SCALING ===")
summary_lines.append("StandardScaler fit on TRAIN physical features ONLY "
                      f"(shape {X_train_raw.shape}); the identical fitted scaler was used to "
                      "transform TRAIN, VALIDATION, and TEST. Saved to processed/scaler.joblib.")
summary_lines.append(f"Constant (zero-variance) TRAIN features: {zero_var_features} "
                      f"({len(zero_var_features)} of 51). These were NOT removed (feature "
                      "selection is out of scope for this step); StandardScaler maps them to a "
                      "constant 0.0 in every split per its documented zero-variance handling. "
                      "They remain present as columns of zeros and will contribute no variance "
                      "unless removed or handled during the later Random Forest / feature-"
                      "ranking stage.")
summary_lines.append("")
summary_lines.append("=== VALIDATION CHECKS ===")
for k, v in checks.items():
    summary_lines.append(f"  [{'PASS' if v else 'FAIL'}] {k}")
summary_lines.append("")
summary_lines.append("=== REPRODUCIBILITY ===")
summary_lines.append(f"Python : {py_version}")
summary_lines.append(f"numpy  : {np_version}")
summary_lines.append(f"pandas : {pd_version}")
summary_lines.append(f"scikit-learn : {sk_version}")
summary_lines.append(f"joblib : {joblib.__version__}")
summary_lines.append(f"OS     : {os_info}")
summary_lines.append(f"Execution timestamp : {RUN_START.isoformat()}")
summary_lines.append(f"Input file (Attack) : {ATTACK_PATH.name} "
                      f"({attack_size_before:,} bytes, SHA-256 {attack_hash_before})")
summary_lines.append(f"Raw source files unchanged after run: "
                      f"attack={checks['raw_attack_file_unchanged']}, "
                      f"normal_size_unchanged={checks['raw_normal_file_unchanged_size']}")

summary_path = META_DIR / "preprocessing_summary.txt"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
log(f"\nWrote {summary_path}")

# ---------------------------------------------------------------------------
# Wrap-up
# ---------------------------------------------------------------------------
section("PREPROCESSING COMPLETE — NO MODEL TRAINED")
run_end = datetime.now()
log(f"Run end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
log("Per instructions: STOPPING HERE. No Random Forest, no LSTM, no feature "
    "ranking, no Top-k selection, no symbolic feature generation was performed.")

log_path = LOGS_DIR / f"02_preprocess_data_{RUN_START.strftime('%Y%m%d_%H%M%S')}.log"
log_path.write_text("\n".join(_LOG_LINES), encoding="utf-8")
print(f"\nFull run log written to: {log_path}")

if not all_checks_passed:
    sys.exit(1)
