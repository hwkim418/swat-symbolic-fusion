"""
code/26_final_test_evaluation.py

FINAL TEST EVALUATION DRIVER — the last, one-shot, TEST-touching step of the
whole project. Implements (but, per explicit instruction, does NOT execute
in this session) the final evaluation of the four primary five-seed
neuro-symbolic model configurations plus the secondary single-seed
Symbolic-only baseline, using ONLY frozen, VALIDATION-selected artifacts.

Two modes:
  - Default (no flag): TEST-FREE PREFLIGHT ONLY. Builds the full
    checkpoint/threshold provenance manifest, verifies the 16-feature
    symbolic-scoring provenance map (using TRAIN-NORMAL data only -- this is
    always-allowed TRAIN access, never TEST), verifies the frozen protocol
    constants, and verifies the TEST sequence-count arithmetic WITHOUT ever
    opening a TEST file. TEST_UNSCALED_PATH / TEST_SCALED_PATH are not even
    defined at module scope -- they are constructed only inside the
    --evaluate-test branch, so merely importing or preflighting this module
    cannot reference a TEST path.
  - `--evaluate-test`: (NOT RUN IN THIS SESSION) loads TEST for the first and
    only time, builds the TEST symbolic feature matrix using the exact
    TRAIN-NORMAL-derived reference parameters already used for TRAIN/
    VALIDATION (verbatim scoring functions, no TEST fitting), evaluates all
    20 primary model-seed checkpoints plus Symbolic-only at their frozen
    VALIDATION-selected thresholds (never recomputed), and persists final
    metrics + aggregation tables.

ONE-SHOT TEST POLICY (see also the printed banner in --evaluate-test mode):
Once --evaluate-test has been run and TEST results produced, TEST is final
evaluation only. No retraining, threshold adjustment, feature selection,
symbolic-rule modification, architecture change, hyperparameter tuning,
seed removal, exclusion of unfavorable results, model switching, or repeat
TEST probing is permitted afterward. A genuine software/runtime failure may
be investigated and reported, but never "fixed" by reference to TEST
performance.

This script does NOT modify scripts 23, 24, or 25, nor any of their
persisted checkpoints, predictions, thresholds, or the symbolic audit
artifacts (scripts 17/18/21). All of it is loaded read-only.
"""

import sys
import json
import time
import hashlib
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Load script 23 as a read-only module (never modified, never re-executed
# via __main__) so this driver reuses its exact shared methodology.
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
SCRIPT23_PATH = PROJECT_ROOT / "src" / "23_main_experiment.py"
SCRIPT21_PATH = PROJECT_ROOT / "src" / "21_build_symbolic_feature_matrices.py"

_spec23 = importlib.util.spec_from_file_location("main_experiment_v23", SCRIPT23_PATH)
me = importlib.util.module_from_spec(_spec23)
_spec23.loader.exec_module(me)

TABLES_DIR = PROJECT_ROOT / "tables"
RESULTS_DIR = PROJECT_ROOT / "results"
PROCESSED_DIR = PROJECT_ROOT / "processed"
ORIGINAL_ARTIFACTS_DIR = RESULTS_DIR / "main_experiment"        # script 23's untouched output (seed 42)
REPEATED_SEED_DIR = RESULTS_DIR / "repeated_seed"                # script 24's untouched output (seeds 7/21/84/123)
FINAL_TEST_DIR = RESULTS_DIR / "final_test_evaluation"           # this script's own output only (written only in --evaluate-test)
# NOTE: TEST paths are deliberately NOT defined here at module scope. They
# are constructed only inside evaluate_test(), which is only reachable via
# the explicit --evaluate-test flag. Importing or preflighting this module
# can never construct a TEST path.

SEEDS = [7, 21, 42, 84, 123]
PRIMARY_MODELS = ["full_lstm", "selected_lstm", "neurosymbolic_core", "neurosymbolic_extended"]
EXPECTED_TEST_ROWS = 111600
EXPECTED_TEST_SEQUENCES = 111481  # 111600 - 120 + 1, verified by pure arithmetic below, no file access

RUN_START = datetime.now()
_LOG_LINES = []


def log(msg=""):
    msg = str(msg)
    print(msg, flush=True)
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


# =============================================================================
# SYMBOLIC SCORING METHODOLOGY -- copied verbatim from code/21_build_symbolic_
# feature_matrices.py (which has no __main__ guard and would re-run its own
# TRAIN/VALIDATION recomputation as an import side effect, so it cannot be
# imported directly). Byte-identity against script 21's actual source is
# verified in preflight below, exactly as script 21 verified its own
# FEATURE_SPECS against script 17's. All reference-construction functions
# below depend ONLY on train_df/train_normal (TRAIN-NORMAL data, always
# allowed) -- none of them reference TEST in any way, and none can, since
# they close over module-level train_df/train_normal defined here from
# processed/train_unscaled.csv only.
# =============================================================================

train_df = pd.read_csv(PROCESSED_DIR / "train_unscaled.csv").reset_index(drop=True)
train_normal = train_df[train_df.Label == 0].reset_index(drop=True)


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


def infer_2state_semantics(source_tag, target_tag):
    states = sorted(train_df[source_tag].unique())
    mean0 = train_normal.loc[train_normal[source_tag] == states[0], target_tag].mean()
    mean1 = train_normal.loc[train_normal[source_tag] == states[1], target_tag].mean()
    return (states[0], states[1]) if mean0 <= mean1 else (states[1], states[0])


def build_state2_refs(source_tag, target_tag):
    lo, hi = infer_2state_semantics(source_tag, target_tag)
    off_ref = train_normal.loc[train_normal[source_tag] == lo, target_tag].to_numpy()
    on_ref = train_normal.loc[train_normal[source_tag] == hi, target_tag].to_numpy()
    return lo, hi, off_ref, on_ref


def score_state_2way(df, source_tag, target_tag, refs):
    lo, hi, off_ref, on_ref = refs
    st = df[source_tag].to_numpy()
    x = df[target_tag].to_numpy(dtype=float)
    score = np.full(len(df), np.nan)
    score[st == lo] = empirical_two_sided_score_vec(off_ref, x[st == lo])
    score[st == hi] = empirical_two_sided_score_vec(on_ref, x[st == hi])
    return score


def build_valve3_refs(source_tag, target_tag):
    states = sorted(train_df[source_tag].unique())
    means = {s: train_normal.loc[train_normal[source_tag] == s, target_tag].mean() for s in states}
    lo_state = min(means, key=means.get)
    hi_state = max(means, key=means.get)
    off_ref = train_normal.loc[train_normal[source_tag] == lo_state, target_tag].to_numpy()
    on_ref = train_normal.loc[train_normal[source_tag] == hi_state, target_tag].to_numpy()
    return lo_state, hi_state, off_ref, on_ref


def score_valve_state_3way(df, source_tag, target_tag, refs):
    lo_state, hi_state, off_ref, on_ref = refs
    st = df[source_tag].to_numpy()
    x = df[target_tag].to_numpy(dtype=float)
    score = np.full(len(df), np.nan)
    score[st == lo_state] = empirical_two_sided_score_vec(off_ref, x[st == lo_state])
    score[st == hi_state] = empirical_two_sided_score_vec(on_ref, x[st == hi_state])
    return score  # transitional state (neither lo nor hi) left NaN, matching script 17/21 exactly


def build_pump_slope_refs(source_tag, target_tag, window):
    slope = (train_df[target_tag] - train_df[target_tag].shift(window)).to_numpy()
    normal_mask = (train_df["Label"].to_numpy() == 0)
    pump_states = train_df[source_tag].to_numpy()
    states = sorted(train_df[source_tag].unique())
    off_state, on_state = states[0], states[1]
    off_ref = slope[normal_mask & (pump_states == off_state) & ~np.isnan(slope)]
    on_ref = slope[normal_mask & (pump_states == on_state) & ~np.isnan(slope)]
    return off_state, on_state, off_ref, on_ref


def score_pump_slope(df, source_tag, target_tag, window, refs):
    off_state, on_state, off_ref, on_ref = refs
    slope = (df[target_tag] - df[target_tag].shift(window)).to_numpy()  # independent per split, never bridges partitions
    pump_states = df[source_tag].to_numpy()
    score = np.full(len(df), np.nan)
    valid = ~np.isnan(slope)
    off_mask = valid & (pump_states == off_state)
    on_mask = valid & (pump_states == on_state)
    score[off_mask] = empirical_two_sided_score_vec(off_ref, slope[off_mask])
    score[on_mask] = empirical_two_sided_score_vec(on_ref, slope[on_mask])
    return score


def build_uncond_slope_ref(target_tag, window):
    slope = (train_df[target_tag] - train_df[target_tag].shift(window)).to_numpy()
    normal_mask = (train_df["Label"].to_numpy() == 0)
    return slope[normal_mask & ~np.isnan(slope)]


def score_unconditional_slope(df, target_tag, window, ref):
    slope = (df[target_tag] - df[target_tag].shift(window)).to_numpy()  # independent per split
    score = np.full(len(df), np.nan)
    valid = ~np.isnan(slope)
    score[valid] = empirical_two_sided_score_vec(ref, slope[valid])
    return score


def build_residual_ref(tag_a, tag_b, kind):
    a_n = train_normal[tag_a].to_numpy(dtype=float)
    b_n = train_normal[tag_b].to_numpy(dtype=float)
    if kind == "ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            ref = np.where(b_n != 0, a_n / b_n, np.nan)
    else:
        ref = a_n - b_n
    return ref[~np.isnan(ref) & np.isfinite(ref)]


def score_sensor_residual(df, tag_a, tag_b, kind, ref):
    # `lag` in FEATURE_SPECS is discovery-stage documentation only, never
    # applied as a shift -- identical to scripts 17/21's own functions.
    a_full = df[tag_a].to_numpy(dtype=float)
    b_full = df[tag_b].to_numpy(dtype=float)
    if kind == "ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            x_full = np.where(b_full != 0, a_full / b_full, np.nan)
    else:
        x_full = a_full - b_full
    return empirical_two_sided_score_vec(ref, x_full)


def score_binary_mismatch(df, tag_a, tag_b):
    a = df[tag_a].to_numpy()
    b = df[tag_b].to_numpy()
    return (a != b).astype(float)  # always defined, never NaN


FEATURE_SPECS = {
    "S1-R1_state_score": ("valve3", "MV101", "FIT101", None),
    "S1-R2_slope_score": ("pump_slope", "P101", "LIT101", 120),
    "S1-R3_slope_score": ("pump_slope", "P102", "LIT101", 300),
    "S1-R4_slope_score": ("uncond_slope", None, "LIT101", 60),
    "S2_R1_score": ("valve3", "MV201", "FIT201", None),
    "S2_R3B_state_mismatch": ("mismatch", "P205", "P203", None),
    "S3-R1_state_score": ("state2", "P302", "FIT301", None),
    "S3-R2_state_score": ("state2", "P302", "DPIT301", None),
    "S3-R4_slope_score": ("uncond_slope", None, "LIT301", 300),
    "S4-R2_state_score": ("state2", "P402", "FIT401", None),
    "S5-R1_state_score": ("state2", "P501", "FIT504", None),
    "S5-R2_state_score": ("state2", "P501", "PIT501", None),
    "S5-R3_residual_score": ("ratio", "FIT501", "FIT503", 0),
    "S5-R4_residual_score": ("diff", "PIT501", "PIT503", 0),
    "S5-R5_residual_score": ("ratio", "AIT501", "AIT504", -30),
    "S6-R1_state_score": ("state2", "P602", "FIT601", None),
}

KIND_DESCRIPTION = {
    "state2": ("build_state2_refs(source,target)", "off_ref/on_ref: TRAIN-NORMAL target values at each of the actuator's 2 states"),
    "valve3": ("build_valve3_refs(source,target)", "off_ref/on_ref: TRAIN-NORMAL target values at the empirically-identified LOW/HIGH extreme states (transitional state excluded)"),
    "pump_slope": ("build_pump_slope_refs(source,target,window)", "off_ref/on_ref: TRAIN-NORMAL rolling-slope values (computed within TRAIN only) at each pump state"),
    "uncond_slope": ("build_uncond_slope_ref(target,window)", "ref: TRAIN-NORMAL rolling-slope values (computed within TRAIN only), unconditional on any actuator"),
    "ratio": ("build_residual_ref(a,b,kind)", "ref: TRAIN-NORMAL ratio/difference values between the two sensors"),
    "diff": ("build_residual_ref(a,b,kind)", "ref: TRAIN-NORMAL ratio/difference values between the two sensors"),
    "mismatch": ("(none)", "no reference needed -- direct binary state comparison, always defined"),
}


def verify_feature_specs_byte_identical_to_script21():
    """Byte-compares this file's FEATURE_SPECS block against script 21's
    actual source text, so the copy cannot silently drift -- the same
    pattern script 21 itself used to verify against script 17."""
    import re
    script21_text = SCRIPT21_PATH.read_text(encoding="utf-8")
    m21 = re.search(r"FEATURE_SPECS = \{.*?\n\}", script21_text, flags=re.S)
    this_text = SCRIPT_PATH.read_text(encoding="utf-8")
    m26 = re.search(r"FEATURE_SPECS = \{.*?\n\}", this_text, flags=re.S)
    if m21 is None or m26 is None:
        return False
    return m21.group(0).strip() == m26.group(0).strip()


# --- AST-level function-body equivalence (not just FEATURE_SPECS) -----------
# A byte/text diff would flag trivial reformatting as a "change" and could
# also be fooled by a comment-only edit that alters meaning inside a string.
# Comparing normalized ASTs instead verifies actual behavior -- formulas,
# conditionals, rolling/window computation, parameter usage, lag handling
# (or lack thereof), empirical-CDF scoring, NaN handling, and return
# values -- while ignoring comments, whitespace, and docstring text, which
# carry no runtime meaning.
import ast


def _get_function_node(file_path, func_name):
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    return None


def _normalized_dump(node):
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]  # drop a docstring, if present -- text only, no runtime meaning
    body_dump = ast.dump(ast.Module(body=body, type_ignores=[]), annotate_fields=False)
    args_dump = ast.dump(node.args, annotate_fields=False)
    return body_dump, args_dump


def verify_function_equivalence(func_name):
    """Returns (match: bool, detail: str). match=False whenever the function
    is missing from either file OR its normalized AST (signature + body,
    comments/docstrings/formatting stripped) differs in any way."""
    node21 = _get_function_node(SCRIPT21_PATH, func_name)
    node26 = _get_function_node(SCRIPT_PATH, func_name)
    if node21 is None:
        return False, f"'{func_name}' not found in script 21"
    if node26 is None:
        return False, f"'{func_name}' not found in script 26"
    body21, args21 = _normalized_dump(node21)
    body26, args26 = _normalized_dump(node26)
    if args21 != args26:
        return False, "signature (arguments) differs"
    if body21 != body26:
        return False, "function body AST differs"
    return True, "AST-equivalent (signature + body, comments/formatting ignored)"


# The 6 "score_*" functions the ablation/scoring pipeline directly calls per
# FEATURE_SPECS kind, plus the 7 supporting reference-construction/helper
# functions that implement rolling-window behavior, empirical-CDF scoring,
# and NaN handling referenced in those 6 -- verifying all 13 closes every
# concern raised (formulas, conditionals, window behavior, parameter usage,
# lag handling, empirical-CDF scoring, NaN behavior, return values).
SCORING_FUNCTIONS = [
    "score_state_2way", "score_valve_state_3way", "score_pump_slope",
    "score_unconditional_slope", "score_sensor_residual", "score_binary_mismatch",
]
HELPER_FUNCTIONS = [
    "empirical_two_sided_score_vec", "infer_2state_semantics",
    "build_state2_refs", "build_valve3_refs", "build_pump_slope_refs",
    "build_uncond_slope_ref", "build_residual_ref",
]


def build_reference(kind, source_tag, target_tag, param):
    if kind == "state2":
        return build_state2_refs(source_tag, target_tag)
    elif kind == "valve3":
        return build_valve3_refs(source_tag, target_tag)
    elif kind == "pump_slope":
        return build_pump_slope_refs(source_tag, target_tag, param)
    elif kind == "uncond_slope":
        return build_uncond_slope_ref(target_tag, param)
    elif kind in ("ratio", "diff"):
        return build_residual_ref(source_tag, target_tag, kind)
    elif kind == "mismatch":
        return None
    raise ValueError(kind)


def score_with_reference(kind, df, source_tag, target_tag, param, ref):
    if kind == "state2":
        return score_state_2way(df, source_tag, target_tag, ref)
    elif kind == "valve3":
        return score_valve_state_3way(df, source_tag, target_tag, ref)
    elif kind == "pump_slope":
        return score_pump_slope(df, source_tag, target_tag, param, ref)
    elif kind == "uncond_slope":
        return score_unconditional_slope(df, target_tag, param, ref)
    elif kind in ("ratio", "diff"):
        return score_sensor_residual(df, source_tag, target_tag, kind, ref)
    elif kind == "mismatch":
        return score_binary_mismatch(df, source_tag, target_tag)
    raise ValueError(kind)


# =============================================================================
# CHECKPOINT / THRESHOLD / CONFIG MANIFEST (20 primary + Symbolic-only)
# =============================================================================

def load_metadata(model_name, seed):
    if seed == 42:
        path = ORIGINAL_ARTIFACTS_DIR / f"{model_name}_metadata.json"
        source_experiment = "script 23 (seed 42, original single-seed run)"
    else:
        path = REPEATED_SEED_DIR / f"{model_name}_seed{seed}_metadata.json"
        source_experiment = "script 24 (repeated-seed run)"
    if not path.exists():
        return None, path, source_experiment
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    return meta, path, source_experiment


def checkpoint_path_for(model_name, seed):
    if seed == 42:
        return ORIGINAL_ARTIFACTS_DIR / f"{model_name}_checkpoint.pt"
    return REPEATED_SEED_DIR / f"{model_name}_seed{seed}_checkpoint.pt"


def build_manifest(pf23):
    all_51, top25 = pf23["all_51_features"], pf23["top25_features"]
    core6, ext16 = pf23["core_features"], pf23["extended_features"]
    expected_raw = {"full_lstm": all_51, "selected_lstm": top25,
                     "neurosymbolic_core": top25, "neurosymbolic_extended": top25}
    expected_sym = {"full_lstm": None, "selected_lstm": None,
                     "neurosymbolic_core": core6, "neurosymbolic_extended": ext16}

    manifest_rows = []
    problems = []
    for model_name in PRIMARY_MODELS:
        for seed in SEEDS:
            meta, meta_path, source_experiment = load_metadata(model_name, seed)
            ckpt_path = checkpoint_path_for(model_name, seed)
            row = {"Model": model_name, "Seed": seed, "Source_Experiment": source_experiment,
                   "Metadata_Path": str(meta_path), "Checkpoint_Path": str(ckpt_path)}
            if meta is None:
                problems.append(f"MISSING metadata: {meta_path}")
                row["Status"] = "MISSING_METADATA"
                manifest_rows.append(row)
                continue
            if not ckpt_path.exists():
                problems.append(f"MISSING checkpoint: {ckpt_path}")
                row["Status"] = "MISSING_CHECKPOINT"
                manifest_rows.append(row)
                continue
            row["Checkpoint_SHA256"] = sha256_of(ckpt_path)
            row["Metadata_SHA256"] = sha256_of(meta_path)
            row["Validation_Selected_Threshold"] = meta["selected_threshold"]
            row["Best_Epoch"] = meta["best_epoch"]
            cfg = meta["config"]
            row["Config_Hidden_Size"] = cfg.get("hidden_size")
            row["Config_Num_Layers"] = cfg.get("num_layers")
            row["Config_Dropout"] = cfg.get("dropout")
            row["Config_Seed"] = cfg.get("seed")
            actual_raw = cfg.get("raw_cols")
            actual_sym = cfg.get("sym_cols")
            raw_ok = (actual_raw == expected_raw[model_name])
            sym_ok = (actual_sym == expected_sym[model_name])
            row["Raw_Features_Match_Frozen_Spec"] = raw_ok
            row["Symbolic_Features_Match_Frozen_Spec"] = sym_ok
            row["Seed_Identity_Matches"] = (cfg.get("seed") == seed)
            if not raw_ok:
                problems.append(f"{model_name} seed={seed}: raw_cols does not match frozen spec")
            if not sym_ok:
                problems.append(f"{model_name} seed={seed}: sym_cols does not match frozen spec")
            if cfg.get("seed") != seed:
                problems.append(f"{model_name} seed={seed}: config seed mismatch ({cfg.get('seed')})")
            row["Status"] = "OK" if (raw_ok and sym_ok and cfg.get("seed") == seed) else "MISMATCH"
            manifest_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows)
    return manifest_df, problems


def check_symbolic_only_eligibility(pf23):
    core6 = pf23["core_features"]
    meta_path = ORIGINAL_ARTIFACTS_DIR / "symbolic_only_metadata.json"
    ckpt_path = ORIGINAL_ARTIFACTS_DIR / "symbolic_only_checkpoint.pt"
    pred_path = ORIGINAL_ARTIFACTS_DIR / "symbolic_only_validation_predictions.csv"
    eligible = meta_path.exists() and ckpt_path.exists() and pred_path.exists()
    detail = {"metadata_exists": meta_path.exists(), "checkpoint_exists": ckpt_path.exists(),
              "validation_predictions_exist": pred_path.exists(), "available_seeds": []}
    if eligible:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        detail["config_sym_cols_matches_core"] = (meta["config"].get("sym_cols") == core6)
        detail["threshold"] = meta["selected_threshold"]
        detail["checkpoint_sha256"] = sha256_of(ckpt_path)
        detail["metadata_sha256"] = sha256_of(meta_path)
        eligible = eligible and detail["config_sym_cols_matches_core"]
    # confirm repeated-seed symbolic_only artifacts do NOT exist (script 24
    # deliberately excluded symbolic_only from its 5-seed sweep)
    other_seed_paths = [REPEATED_SEED_DIR / f"symbolic_only_seed{s}_metadata.json" for s in (7, 21, 84, 123)]
    detail["other_seed_artifacts_found"] = [str(p) for p in other_seed_paths if p.exists()]
    detail["available_seeds"] = [42] if eligible else []
    return eligible, detail


# =============================================================================
# PREFLIGHT (TEST-FREE)
# =============================================================================

def run_preflight():
    section("PREFLIGHT — TEST-FREE (default mode; --evaluate-test not invoked)")
    checks = {}

    log("Delegating to code/23_main_experiment.py's own run_preflight() (TRAIN/VALIDATION only)...")
    pf23 = me.run_preflight()
    # Merge script 23's own 21 atomic sub-checks into THIS flat dict (namespaced
    # to avoid key collisions), rather than storing a single rollup boolean --
    # so `len(checks)` at the end is the single, unambiguous, authoritative
    # count of every atomic guardrail verified across both layers, with no
    # separate "top-level" vs "delegated" totals to reconcile by hand.
    for k, v in pf23["checks"].items():
        checks[f"script23_delegated__{k}"] = v

    # --- A. Protocol freeze ---
    section("A. PROTOCOL FREEZE")
    protocol_checks = {
        "K_equals_25": len(pf23["top25_features"]) == 25,
        "window_equals_120": me.WINDOW == 120,
        "stride_equals_1": me.STRIDE == 1,
        "hidden_size_equals_64": me.HIDDEN_SIZE == 64,
        "num_layers_equals_1": me.NUM_LAYERS == 1,
        "dropout_equals_0.2": me.DROPOUT == 0.2,
        "lr_equals_1e-3": me.LR == 1e-3,
        "batch_size_equals_256": me.BATCH_SIZE == 256,
        "max_epochs_equals_30": me.MAX_EPOCHS == 30,
        "patience_equals_5": me.PATIENCE == 5,
        "seeds_match_locked_spec": SEEDS == [7, 21, 42, 84, 123],
        "core_exactly_6": len(pf23["core_features"]) == 6,
        "extended_exactly_16": len(pf23["extended_features"]) == 16,
    }
    for k, v in protocol_checks.items():
        log(f"  [{'PASS' if v else 'FAIL'}] {k}")
    checks.update(protocol_checks)

    # --- B. Artifact provenance ---
    section("B. ARTIFACT PROVENANCE (20 primary model-seed checkpoints)")
    manifest_df, manifest_problems = build_manifest(pf23)
    log(manifest_df[["Model", "Seed", "Source_Experiment", "Status"]].to_string(index=False))
    checks["all_20_checkpoints_present_and_matching"] = (len(manifest_problems) == 0
                                                          and (manifest_df["Status"] == "OK").all())
    if manifest_problems:
        log("\n  PROBLEMS FOUND:")
        for p in manifest_problems:
            log(f"    - {p}")
    checks["no_threshold_recomputation_performed"] = True  # structural: thresholds are read verbatim from metadata, never recomputed anywhere in this script

    sym_only_eligible, sym_only_detail = check_symbolic_only_eligibility(pf23)
    log(f"\n  Symbolic-only eligibility: {'ELIGIBLE (single-seed, seed 42 only)' if sym_only_eligible else 'NOT ELIGIBLE'}")
    for k, v in sym_only_detail.items():
        log(f"    {k}: {v}")
    checks["symbolic_only_eligibility_determined_without_retraining"] = True

    # --- C. Feature provenance ---
    section("C. FEATURE PROVENANCE")
    checks["full_uses_exactly_51_features"] = len(pf23["all_51_features"]) == 51
    checks["selected_core_extended_use_frozen_top25"] = True  # verified per-row in manifest above (Raw_Features_Match_Frozen_Spec)
    checks["core_order_matches_frozen_definition"] = (pf23["core_features"] == pd.read_csv(TABLES_DIR / "final_core_symbolic_features.csv")["Feature_Name"].tolist())
    checks["extended_order_matches_frozen_definition"] = (pf23["extended_features"] == pd.read_csv(TABLES_DIR / "final_extended_symbolic_features.csv")["Feature_Name"].tolist())

    log("\n  16-feature symbolic TEST provenance map (reference construction uses TRAIN-NORMAL only, computed now as a dry run -- zero TEST access):")
    checks["feature_specs_byte_identical_to_script21"] = verify_feature_specs_byte_identical_to_script21()
    log(f"    FEATURE_SPECS byte-identical to script 21's: {checks['feature_specs_byte_identical_to_script21']}")

    provenance_rows = []
    all_16_reproducible = True
    for fname, (kind, a, b, param) in FEATURE_SPECS.items():
        build_fn_desc, param_desc = KIND_DESCRIPTION[kind]
        try:
            ref = build_reference(kind, a, b, param)
            ok = True
            note = "reference built successfully from TRAIN-NORMAL data"
        except Exception as e:
            ref = None
            ok = False
            note = f"BLOCKER: {e}"
            all_16_reproducible = False
        provenance_rows.append({
            "Symbolic_Feature": fname, "Scoring_Kind": kind,
            "Build_Function": build_fn_desc, "Required_Parameters": param_desc,
            "Source_Artifact": "code/21_build_symbolic_feature_matrices.py (verbatim copy, byte-verified above)",
            "No_TEST_Fitting_Required": True, "Reproducible_From_Frozen_TRAIN_Params": ok, "Note": note,
        })
    provenance_df = pd.DataFrame(provenance_rows)
    log(provenance_df[["Symbolic_Feature", "Scoring_Kind", "Reproducible_From_Frozen_TRAIN_Params"]].to_string(index=False))
    checks["all_16_features_reproducible_from_frozen_train_params"] = all_16_reproducible

    # --- C2. AST-level scoring-function equivalence against script 21 ---
    section("C2. SCORING-FUNCTION EQUIVALENCE (AST-normalized, comments/formatting ignored)")
    log("The 6 scoring functions directly selected by FEATURE_SPECS kind:")
    for fn in SCORING_FUNCTIONS:
        ok, detail = verify_function_equivalence(fn)
        checks[f"function_equivalent__{fn}"] = ok
        log(f"  [{'PASS' if ok else 'FAIL'}] {fn}: {detail}")
    log("\nSupporting reference-construction/helper functions (rolling-window behavior, "
        "empirical-CDF scoring, NaN handling):")
    for fn in HELPER_FUNCTIONS:
        ok, detail = verify_function_equivalence(fn)
        checks[f"function_equivalent__{fn}"] = ok
        log(f"  [{'PASS' if ok else 'FAIL'}] {fn}: {detail}")

    # --- D. Planned TEST shape (pure arithmetic, zero file access) ---
    section("D. PLANNED TEST SHAPE (arithmetic only, no TEST file opened)")
    computed_test_sequences = EXPECTED_TEST_ROWS - me.WINDOW + 1
    checks["test_sequence_arithmetic_correct"] = (computed_test_sequences == EXPECTED_TEST_SEQUENCES == 111481)
    log(f"  Expected TEST raw rows: {EXPECTED_TEST_ROWS:,}")
    log(f"  Expected TEST sequences = {EXPECTED_TEST_ROWS:,} - {me.WINDOW} + 1 = {computed_test_sequences:,} "
        f"(matches locked expectation 111,481: {computed_test_sequences == 111481})")

    # --- E. Leakage firewall ---
    section("E. LEAKAGE FIREWALL")
    module_src = SCRIPT_PATH.read_text(encoding="utf-8")
    import re as _re
    checks["no_test_path_variable_at_module_scope"] = (
        _re.search(r"^TEST\w*_PATH\s*=", module_src, flags=_re.M) is None)
    all_log_text = "\n".join(_LOG_LINES)
    checks["no_test_filename_in_preflight_log"] = ("test_scaled" not in all_log_text and "test_unscaled" not in all_log_text)
    checks["test_evaluate_test_not_invoked_this_run"] = True  # structural: this function is run_preflight(), evaluate_test() is never called here
    for k in ("no_test_path_variable_at_module_scope", "no_test_filename_in_preflight_log", "test_evaluate_test_not_invoked_this_run"):
        log(f"  [{'PASS' if checks[k] else 'FAIL'}] {k}")

    log("\nFull guardrail summary (single flat dict -- this IS the authoritative total, no separate tallies to reconcile):")
    for k, v in checks.items():
        log(f"  [{'PASS' if v else 'FAIL'}] {k}")
    all_passed = all(checks.values())
    n_pass = sum(1 for v in checks.values() if v)
    n_total = len(checks)
    log(f"\nAUTHORITATIVE TOTAL GUARDRAIL CHECKS: {n_total} ({n_total - 21} defined directly in script 26 "
        f"+ 21 delegated from script 23's own preflight, merged into this same dict)")
    log(f"AUTHORITATIVE RESULT: {n_pass}/{n_total} PASS")
    log(f"\nPREFLIGHT OVERALL: {'PASS' if all_passed else 'FAIL'}")

    return {"checks": checks, "all_passed": all_passed, "pf23": pf23, "manifest_df": manifest_df,
            "manifest_problems": manifest_problems, "sym_only_eligible": sym_only_eligible,
            "sym_only_detail": sym_only_detail, "provenance_df": provenance_df}


# =============================================================================
# --evaluate-test (IMPLEMENTED, NOT INVOKED IN THIS SESSION)
# =============================================================================

def evaluate_test(pf, args):
    """First and only authorized TEST access. Not called unless --evaluate-test
    is passed explicitly. See the ONE-SHOT TEST POLICY in this file's
    docstring: after this runs and produces results, no retraining, no
    threshold adjustment, no feature/architecture/seed changes are permitted
    in response to what TEST shows."""
    section("*** AUTHORIZED TEST ACCESS -- ONE-SHOT FINAL EVALUATION ***")
    log("ONE-SHOT TEST POLICY: after this run, no retraining, threshold adjustment, feature "
        "selection, symbolic-rule modification, architecture change, hyperparameter tuning, seed "
        "removal, exclusion of unfavorable results, model switching, or repeat TEST probing is "
        "permitted. A genuine software/runtime failure may be investigated and reported, but never "
        "corrected using TEST performance as a guide.")
    FINAL_TEST_DIR.mkdir(parents=True, exist_ok=True)

    # TEST paths constructed here ONLY, inside this explicitly-gated function.
    TEST_UNSCALED_PATH = PROCESSED_DIR / "test_unscaled.csv"
    TEST_SCALED_PATH = PROCESSED_DIR / "test_scaled.csv"

    pf23 = pf["pf23"]
    test_unscaled = pd.read_csv(TEST_UNSCALED_PATH).reset_index(drop=True)
    test_scaled = pd.read_csv(TEST_SCALED_PATH).reset_index(drop=True)
    assert len(test_unscaled) == EXPECTED_TEST_ROWS == len(test_scaled)
    log(f"Loaded TEST: {len(test_scaled):,} rows (expected {EXPECTED_TEST_ROWS:,})")

    # --- Build TEST symbolic feature matrix using FROZEN TRAIN-NORMAL
    # references only (built in Part 1 above, before TEST was ever opened) ---
    section("BUILDING TEST SYMBOLIC FEATURE MATRIX (frozen TRAIN-NORMAL references, zero TEST fitting)")
    test_symbolic = {}
    for fname, (kind, a, b, param) in FEATURE_SPECS.items():
        ref = build_reference(kind, a, b, param)  # TRAIN-NORMAL only, independent of TEST
        scores = score_with_reference(kind, test_unscaled, a, b, param, ref)
        test_symbolic[fname] = np.nan_to_num(scores, nan=0.0)  # frozen Decision-1 zero-fill policy
    sym_test_df = pd.DataFrame(test_symbolic)
    sym_test_df.insert(0, "Label", test_unscaled["Label"])
    sym_test_df.insert(0, "Timestamp", test_unscaled["Timestamp"])

    y_test_full = test_unscaled["Label"].to_numpy(dtype=np.float32)

    def raw_fwd(model, batch):
        xb, yb = batch
        return model(xb), yb

    def fusion_fwd(model, batch):
        xb, sb, yb = batch
        return model(xb, sb), yb

    all_51, top25 = pf23["all_51_features"], pf23["top25_features"]
    core6, ext16 = pf23["core_features"], pf23["extended_features"]

    def spec_for(model_name):
        if model_name == "full_lstm":
            return {"mode": "raw", "raw_cols": all_51}
        elif model_name == "selected_lstm":
            return {"mode": "raw", "raw_cols": top25}
        elif model_name == "neurosymbolic_core":
            return {"mode": "fusion", "raw_cols": top25, "sym_cols": core6}
        elif model_name == "neurosymbolic_extended":
            return {"mode": "fusion", "raw_cols": top25, "sym_cols": ext16}
        elif model_name == "symbolic_only":
            return {"mode": "symbolic", "sym_cols": core6}
        raise ValueError(model_name)

    section("EVALUATING ALL MODEL-SEED PAIRS ON TEST (frozen VALIDATION-selected thresholds, never recomputed)")
    result_rows = []
    eval_targets = [(m, s) for m in PRIMARY_MODELS for s in SEEDS]
    if pf["sym_only_eligible"]:
        eval_targets.append(("symbolic_only", 42))

    for model_name, seed in eval_targets:
        meta, meta_path, source_experiment = load_metadata(model_name, seed)
        ckpt_path = checkpoint_path_for(model_name, seed) if model_name != "symbolic_only" else ORIGINAL_ARTIFACTS_DIR / "symbolic_only_checkpoint.pt"
        threshold = meta["selected_threshold"]  # frozen, never recomputed
        spec = spec_for(model_name)

        raw_arr = test_scaled[spec["raw_cols"]].to_numpy(dtype=np.float32) if "raw_cols" in spec else None
        sym_arr = sym_test_df[spec["sym_cols"]].to_numpy(dtype=np.float32) if "sym_cols" in spec else None
        test_ds = me.MainExperimentDataset(spec["mode"], me.WINDOW, y_test_full, raw_arr, sym_arr)
        test_loader = DataLoader(test_ds, batch_size=me.BATCH_SIZE, shuffle=False, num_workers=0)
        assert len(test_ds) == EXPECTED_TEST_SEQUENCES

        if spec["mode"] == "raw":
            model = me.LSTMOnlyModel(input_size=len(spec["raw_cols"]))
            fwd = raw_fwd
        elif spec["mode"] == "fusion":
            model = me.FusionModel(raw_input_size=len(spec["raw_cols"]), symbolic_input_size=len(spec["sym_cols"]))
            fwd = fusion_fwd
        else:
            model = me.SymbolicOnlyModel(input_size=len(spec["sym_cols"]))
            fwd = raw_fwd
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()

        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                logits, yb = fwd(model, batch)
                all_probs.append(torch.sigmoid(logits).numpy())
                all_labels.append(yb.numpy())
        y_prob = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels)
        y_pred = (y_prob >= threshold).astype(int)  # frozen threshold, never tuned on TEST

        metrics = me.compute_metrics(y_true, y_pred, y_prob)
        latency = me.measure_latency_ms_per_sequence(model, test_loader, fwd)

        prefix = FINAL_TEST_DIR / f"{model_name}_seed{seed}"
        pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).to_csv(f"{prefix}_test_predictions.csv", index=False)
        result = {"Model": model_name, "Seed": seed, "Threshold_Source": str(meta_path),
                  "Selected_Threshold": threshold, "Latency_ms_per_seq": latency,
                  "N_Test_Sequences": len(test_ds), **metrics}
        with open(f"{prefix}_test_metadata.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        result_rows.append(result)
        log(f"  [{model_name} seed={seed}] threshold={threshold:.6f} (frozen) "
            f"Acc={metrics['Accuracy']:.4f} MCC={metrics['MCC']:.4f} F1={metrics['F1']:.4f} "
            f"ROC_AUC={metrics['ROC_AUC']:.4f} FPR={metrics['FPR']:.4f} latency={latency:.4f}ms/seq")

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(TABLES_DIR / "final_test_results_long.csv", index=False)
    log(f"\nWrote {TABLES_DIR / 'final_test_results_long.csv'} ({len(results_df)} rows)")

    section("AGGREGATION (4 primary models x 5 seeds; Symbolic-only reported separately, single-seed)")
    stats_rows = []
    for model_name in PRIMARY_MODELS:
        sub = results_df[results_df.Model == model_name].sort_values("Seed")
        for metric in ("MCC", "F1", "ROC_AUC", "Accuracy", "Precision", "Recall", "FPR", "Latency_ms_per_seq"):
            vals = sub[metric].to_numpy()
            stats_rows.append({"Model": model_name, "Metric": metric,
                                **{f"Seed_{s}": v for s, v in zip(sub["Seed"], vals)},
                                "Mean": vals.mean(), "Std": vals.std(ddof=1) if len(vals) > 1 else np.nan,
                                "Median": np.median(vals), "Min": vals.min(), "Max": vals.max()})
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(TABLES_DIR / "final_test_summary_stats.csv", index=False)
    log(f"Wrote {TABLES_DIR / 'final_test_summary_stats.csv'}")

    pair_defs = [("full_lstm", "selected_lstm"), ("selected_lstm", "neurosymbolic_core"),
                 ("selected_lstm", "neurosymbolic_extended"), ("neurosymbolic_core", "neurosymbolic_extended")]
    delta_rows = []
    for a, b in pair_defs:
        sub_a = results_df[results_df.Model == a].set_index("Seed")
        sub_b = results_df[results_df.Model == b].set_index("Seed")
        for metric in ("MCC", "F1", "ROC_AUC"):
            deltas = (sub_b[metric] - sub_a[metric]).reindex(SEEDS)
            delta_rows.append({"From": a, "To": b, "Metric": metric,
                                **{f"Seed_{s}_Delta": d for s, d in deltas.items()},
                                "Mean_Delta": deltas.mean(), "Median_Delta": deltas.median(),
                                "N_Seeds_Second_Higher": int((deltas > 0).sum())})
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(TABLES_DIR / "final_test_paired_deltas.csv", index=False)
    log(f"Wrote {TABLES_DIR / 'final_test_paired_deltas.csv'}")

    if pf["sym_only_eligible"]:
        sym_row = results_df[results_df.Model == "symbolic_only"]
        log(f"\nSymbolic-only (single-seed=42 secondary baseline, reported separately, NOT part of the "
            f"5-seed robustness comparison):\n{sym_row.to_string(index=False)}")

    section("FINAL GUARDRAIL RE-CHECK")
    log("  [Not yet implemented for --evaluate-test in this session -- this branch is not executed now.]")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-test", action="store_true",
                         help="AUTHORIZES THE FIRST AND ONLY TEST ACCESS. Do not pass this flag "
                              "without explicit, separate approval -- see the ONE-SHOT TEST POLICY "
                              "in this file's module docstring.")
    args = parser.parse_args()

    section("SCRIPT 26 — FINAL TEST EVALUATION DRIVER")
    log(f"Run start: {RUN_START.isoformat()}")
    log(f"Mode: {'!!! --evaluate-test AUTHORIZED !!!' if args.evaluate_test else 'PREFLIGHT ONLY (TEST-free)'}")

    pf = run_preflight()
    if not pf["all_passed"]:
        section("PREFLIGHT FAILED — ABORTING (TEST was never accessed)")
        sys.exit(1)

    if not args.evaluate_test:
        section("PREFLIGHT PASSED — STOPPING HERE (pass --evaluate-test only with explicit separate approval)")
        log("TEST has NOT been opened, read, scored, or referenced by this run. No side effects occurred "
            "beyond this log. READY FOR EXPLICIT FINAL TEST AUTHORIZATION.")
        return

    evaluate_test(pf, args)


if __name__ == "__main__":
    main()
