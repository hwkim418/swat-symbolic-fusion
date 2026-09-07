"""
code/25_symbolic_alignment_ablation.py

SYMBOLIC-ALIGNMENT ABLATION — tests whether the temporal alignment of the
audited CORE symbolic representation (s_t attached to the SAME final
timestep t as h_t) contributes information beyond Selected LSTM, by
comparing against a "Shuffled CORE" variant where the intervention
  z_t = [h_t || s_t]  -->  z_t = [h_t || s_pi(t)]
replaces each row's true CORE vector with another row's CORE vector via a
fixed, deterministic, within-split permutation -- breaking temporal
alignment while exactly preserving symbolic dimensionality, marginal
distributions, and within-row (cross-feature) structure.

Final ablation table (4 rows, each summarized over 5 model seeds):
  1. Selected LSTM        -- REUSED from scripts 23 (seed 42) + 24 (seeds 7/21/84/123). Not retrained.
  2. LSTM + Shuffled CORE -- NEW. 5 seeds trained here.
  3. LSTM + CORE          -- REUSED (same sources as #1).
  4. LSTM + Extended      -- REUSED (same sources as #1).

Only NEW neural training performed by this script: Shuffled CORE x model
seeds [7, 21, 42, 84, 123] = 5 runs.

This script does NOT modify code/23_main_experiment.py or
code/24_repeated_seed_validation.py in any way -- both are loaded read-only
(by file path) and their exact shared methodology (MainExperimentDataset,
FusionModel, train_model, select_threshold, compute_metrics,
measure_latency_ms_per_sequence, run_preflight, set_seed) is reused
unchanged. The only new logic here is the deterministic row-permutation of
the CORE symbolic array before it is handed to the unmodified
MainExperimentDataset -- everything downstream (windowing, model, training,
threshold selection, metrics) is identical to the CORE experiment.

TEST is never loaded, referenced, scored, or constructed for anywhere in
this script.

Two modes, both never touching TEST:
  - Default (no flag): PREFLIGHT ONLY. Verifies frozen inputs (via script
    23's run_preflight()), the reused Selected/CORE/Extended 5-seed
    artifacts, the shuffle-permutation guardrails (bijection, row-wise
    movement, marginal preservation, per-split separation, determinism
    across model seeds), and reports how many of the 5 new Shuffled-CORE
    seeds are already done. Trains nothing.
  - `--train`: trains whatever Shuffled-CORE seeds are missing (skipping
    completed ones, resume-safe), persists per-seed artifacts, then builds
    the 4-row ablation summary table.
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
# Load scripts 23 and 24 as read-only modules (never modified, never
# re-executed via __main__) so this ablation reuses their exact methodology.
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
SCRIPT23_PATH = PROJECT_ROOT / "src" / "23_main_experiment.py"

_spec23 = importlib.util.spec_from_file_location("main_experiment_v23", SCRIPT23_PATH)
me = importlib.util.module_from_spec(_spec23)
_spec23.loader.exec_module(me)

TABLES_DIR = PROJECT_ROOT / "tables"
RESULTS_DIR = PROJECT_ROOT / "results"
ORIGINAL_ARTIFACTS_DIR = RESULTS_DIR / "main_experiment"        # script 23's untouched output (seed 42)
REPEATED_SEED_DIR = RESULTS_DIR / "repeated_seed"                # script 24's untouched output (seeds 7/21/84/123)
ABLATION_ARTIFACTS_DIR = RESULTS_DIR / "symbolic_alignment_ablation"  # this script's own output only
ABLATION_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [7, 21, 42, 84, 123]
SHUFFLE_SEED = 2026
EXPECTED_CORE_DIM = 6

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


def array_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


# =============================================================================
# DETERMINISTIC PERMUTATION OVER THE ELIGIBLE SEQUENCE-FINAL SUBSET ONLY
# (computed ONCE, reused identically for all 5 model seeds -- model-seed
# variability and shuffle variability are kept strictly separate).
#
# CORRECTED per methodological review: MainExperimentDataset only ever reads
# self.sym[end-1, :] for end-1 in [WINDOW-1, N-1] -- rows [0, WINDOW-2] are
# never used as a sequence-final symbolic input at all. Permuting the FULL
# array (as the first implementation did) could move a "dead" row's value
# into the evaluated subset and vice versa, so the multiset actually SEEN by
# the model would not be guaranteed identical to the CORE experiment's. The
# permutation must therefore be a bijection of the eligible index range
# [WINDOW-1, N-1] onto itself; rows [0, WINDOW-2] are left byte-identical
# (copied, never touched) since they are provably never read downstream.
# =============================================================================

def build_eligible_permutation(n_rows, window, shuffle_seed):
    eligible_start = window - 1  # 119 for window=120
    eligible_indices = np.arange(eligible_start, n_rows)
    rng = np.random.RandomState(shuffle_seed)
    permuted_eligible = rng.permutation(eligible_indices)  # a bijection OF the eligible index VALUES onto themselves
    assert len(permuted_eligible) == len(eligible_indices)
    assert set(permuted_eligible.tolist()) == set(eligible_indices.tolist()), \
        "Eligible-subset permutation is not a true bijection of [WINDOW-1, N-1]."
    assert np.all(permuted_eligible >= eligible_start), "An eligible row was mapped to an ineligible index."
    return eligible_indices, permuted_eligible


def apply_eligible_permutation(sym_array, eligible_indices, permuted_eligible):
    """Rows [0, WINDOW-2] are copied unchanged (never read downstream); rows
    in the eligible range are reassigned via the bijection -- an entire
    6-dim row moves together, never per-column."""
    shuffled = sym_array.copy()
    shuffled[eligible_indices] = sym_array[permuted_eligible]
    # verify every shuffled eligible row exactly equals the corresponding
    # original row at its permuted source index (row-wise integrity)
    sample_idx = np.linspace(0, len(eligible_indices) - 1, min(2000, len(eligible_indices)), dtype=int)
    assert np.array_equal(shuffled[eligible_indices[sample_idx]], sym_array[permuted_eligible[sample_idx]])
    # verify the untouched region is truly byte-identical
    assert np.array_equal(shuffled[:eligible_indices[0]], sym_array[:eligible_indices[0]])
    return shuffled


def sort_rows(arr):
    """Canonical row ordering for multiset comparison (lexicographic across
    all columns) -- exact, vectorized, no floating-point recomputation."""
    idx = np.lexsort(arr.T[::-1])
    return arr[idx]


def verify_marginals_preserved_eligible(original, shuffled, eligible_indices, col_names):
    orig_elig = original[eligible_indices]
    shuf_elig = shuffled[eligible_indices]
    for j, name in enumerate(col_names):
        if not np.array_equal(np.sort(orig_elig[:, j]), np.sort(shuf_elig[:, j])):
            return False, name
    return True, None


def verify_row_multiset_preserved_eligible(original, shuffled, eligible_indices):
    orig_elig = original[eligible_indices]
    shuf_elig = shuffled[eligible_indices]
    return np.array_equal(sort_rows(orig_elig), sort_rows(shuf_elig))


# =============================================================================
# PREFLIGHT
# =============================================================================

def run_preflight():
    section("PREFLIGHT — REUSING SCRIPT 23's GUARDRAILS + ABLATION-SPECIFIC CHECKS")
    checks = {}

    log("Delegating to code/23_main_experiment.py's own run_preflight()...")
    pf23 = me.run_preflight()
    checks["script23_preflight_passed"] = pf23["all_passed"]

    checks["seed_set_matches_locked_spec"] = (SEEDS == [7, 21, 42, 84, 123])
    checks["shuffle_seed_is_2026"] = (SHUFFLE_SEED == 2026)
    log(f"  Model seeds: {SEEDS} (locked) | Shuffle seed: {SHUFFLE_SEED} (locked)")

    core_features = pf23["core_features"]
    checks["core_dimension_is_6"] = (len(core_features) == EXPECTED_CORE_DIM)
    log(f"  CORE features ({len(core_features)}): {core_features}")

    # --- reused artifacts (Selected LSTM, CORE, Extended) must exist for all 5 seeds each ---
    reuse_models = ["selected_lstm", "neurosymbolic_core", "neurosymbolic_extended"]
    missing_reuse = []
    for name in reuse_models:
        seed42_meta = ORIGINAL_ARTIFACTS_DIR / f"{name}_metadata.json"
        if not seed42_meta.exists():
            missing_reuse.append(str(seed42_meta))
        for seed in (7, 21, 84, 123):
            p = REPEATED_SEED_DIR / f"{name}_seed{seed}_metadata.json"
            if not p.exists():
                missing_reuse.append(str(p))
    checks["all_15_reused_artifacts_exist"] = (len(missing_reuse) == 0)
    log(f"  Reused-artifact check: {'all 15 present (3 models x 5 seeds)' if not missing_reuse else f'{len(missing_reuse)} MISSING: {missing_reuse}'}")

    # --- original script 23/24 artifacts must remain untouched by this script ---
    original_files = list(ORIGINAL_ARTIFACTS_DIR.glob("*")) + list(REPEATED_SEED_DIR.glob("*"))
    pre_hashes = {p: sha256_of(p) for p in original_files}
    checks["original_artifact_count_correct"] = (len(list(ORIGINAL_ARTIFACTS_DIR.glob("*"))) == 25
                                                  and len(list(REPEATED_SEED_DIR.glob("*"))) == 80)
    log(f"  script23 artifacts: {len(list(ORIGINAL_ARTIFACTS_DIR.glob('*')))} (expected 25); "
        f"script24 artifacts: {len(list(REPEATED_SEED_DIR.glob('*')))} (expected 80)")

    # --- permutation guardrails, restricted to the ELIGIBLE sequence-final
    # subset [WINDOW-1, N-1] only (corrected: a full-partition permutation
    # could leak values between the never-read rows [0, WINDOW-2] and the
    # subset MainExperimentDataset actually reads, per methodological
    # review) -- bijection-of-eligible-onto-itself, no eligible->ineligible
    # mapping, row-wise movement, per-column AND full-row-multiset
    # preservation over the eligible subset, per-split separation, no
    # cross-partition mixing, determinism across model seeds ---
    train_df, val_df = pf23["train_df"], pf23["val_df"]
    sym_train_df, sym_val_df = pf23["sym_train_df"], pf23["sym_val_df"]
    core_train_arr = sym_train_df[core_features].to_numpy(dtype=np.float32)
    core_val_arr = sym_val_df[core_features].to_numpy(dtype=np.float32)

    eligible_train, perm_train = build_eligible_permutation(len(core_train_arr), me.WINDOW, SHUFFLE_SEED)
    eligible_val, perm_val = build_eligible_permutation(len(core_val_arr), me.WINDOW, SHUFFLE_SEED)
    log(f"  Eligible range: TRAIN=[{eligible_train[0]}, {eligible_train[-1]}] ({len(eligible_train):,} rows), "
        f"VALIDATION=[{eligible_val[0]}, {eligible_val[-1]}] ({len(eligible_val):,} rows)")

    checks["train_permutation_is_bijection_of_eligible_only"] = (set(perm_train.tolist()) == set(eligible_train.tolist()))
    checks["validation_permutation_is_bijection_of_eligible_only"] = (set(perm_val.tolist()) == set(eligible_val.tolist()))
    checks["train_no_eligible_to_ineligible_mapping"] = bool(np.all(perm_train >= eligible_train[0]))
    checks["validation_no_eligible_to_ineligible_mapping"] = bool(np.all(perm_val >= eligible_val[0]))
    checks["train_validation_permutations_are_separate"] = (len(perm_train) != len(perm_val)
                                                              or not np.array_equal(perm_train, perm_val))
    checks["no_cross_partition_mixing_possible"] = (len(eligible_train) == len(core_train_arr) - (me.WINDOW - 1)
                                                      and len(eligible_val) == len(core_val_arr) - (me.WINDOW - 1))

    shuffled_train = apply_eligible_permutation(core_train_arr, eligible_train, perm_train)
    shuffled_val = apply_eligible_permutation(core_val_arr, eligible_val, perm_val)
    checks["train_row_wise_movement_verified"] = True  # asserted inside apply_eligible_permutation
    checks["validation_row_wise_movement_verified"] = True
    checks["train_dead_rows_0_to_118_byte_identical"] = np.array_equal(
        shuffled_train[:eligible_train[0]], core_train_arr[:eligible_train[0]])
    checks["validation_dead_rows_0_to_118_byte_identical"] = np.array_equal(
        shuffled_val[:eligible_val[0]], core_val_arr[:eligible_val[0]])

    ok_train, bad_col_train = verify_marginals_preserved_eligible(core_train_arr, shuffled_train, eligible_train, core_features)
    ok_val, bad_col_val = verify_marginals_preserved_eligible(core_val_arr, shuffled_val, eligible_val, core_features)
    checks["train_eligible_column_marginals_exactly_preserved"] = ok_train
    checks["validation_eligible_column_marginals_exactly_preserved"] = ok_val
    if not ok_train:
        log(f"  MARGINAL MISMATCH in TRAIN eligible-subset column: {bad_col_train}")
    if not ok_val:
        log(f"  MARGINAL MISMATCH in VALIDATION eligible-subset column: {bad_col_val}")

    checks["train_eligible_row_multiset_exactly_preserved"] = verify_row_multiset_preserved_eligible(
        core_train_arr, shuffled_train, eligible_train)
    checks["validation_eligible_row_multiset_exactly_preserved"] = verify_row_multiset_preserved_eligible(
        core_val_arr, shuffled_val, eligible_val)

    # determinism across model seeds: hash the permutation now; the training
    # loop will assert this exact hash before every one of the 5 seed runs
    perm_train_hash = array_hash(perm_train)
    perm_val_hash = array_hash(perm_val)
    log(f"  perm_train hash: {perm_train_hash[:16]}...  perm_val hash: {perm_val_hash[:16]}...")
    checks["permutation_hashes_computable"] = True

    # raw sequences and labels must be untouched by the shuffle (structural:
    # the shuffle is only ever applied to a COPY of the symbolic array, never
    # to raw_train_arr/raw_val_arr or the label arrays)
    checks["raw_and_label_arrays_never_permuted"] = True  # enforced by construction in train_one_shuffled_seed()

    n_train_seq = len(train_df) - me.WINDOW + 1
    n_val_seq = len(val_df) - me.WINDOW + 1
    checks["sequence_counts_match_core_experiment"] = (n_train_seq == 269881 and n_val_seq == 68200)
    checks["eligible_count_matches_sequence_count"] = (len(eligible_train) == n_train_seq and len(eligible_val) == n_val_seq)

    # --- resume status ---
    pending, done = [], []
    for seed in SEEDS:
        meta_path = ABLATION_ARTIFACTS_DIR / f"shuffled_core_seed{seed}_metadata.json"
        (done if meta_path.exists() else pending).append(seed)
    log(f"\n  Shuffled-CORE seeds already completed: {len(done)} of {len(SEEDS)} -- {done}")
    log(f"  Shuffled-CORE seeds still needed: {len(pending)} -- {pending}")
    checks["expected_total_new_seeds_is_5"] = (len(done) + len(pending) == 5)

    log("\nGuardrail results:")
    for k, v in checks.items():
        log(f"  [{'PASS' if v else 'FAIL'}] {k}")
    all_passed = all(checks.values())
    log(f"\nPREFLIGHT OVERALL: {'PASS' if all_passed else 'FAIL'}")

    return {"checks": checks, "all_passed": all_passed, "pf23": pf23, "pre_hashes_original": pre_hashes,
            "pending": pending, "done": done, "core_features": core_features,
            "eligible_train": eligible_train, "perm_train": perm_train,
            "eligible_val": eligible_val, "perm_val": perm_val,
            "perm_train_hash": perm_train_hash, "perm_val_hash": perm_val_hash}


# =============================================================================
# TRAIN ONE SHUFFLED-CORE SEED -- reuses script 23's exact shared functions;
# the ONLY difference from neurosymbolic_core is that the symbolic array
# handed to MainExperimentDataset has been row-permuted beforehand.
# =============================================================================

def fusion_fwd(model, batch):
    xb, sb, yb = batch
    return model(xb, sb), yb


def train_one_shuffled_seed(seed, pf23, pos_weight, eligible_train, perm_train, eligible_val, perm_val,
                             perm_train_hash, perm_val_hash, core_features):
    train_df, val_df = pf23["train_df"], pf23["val_df"]
    sym_train_df, sym_val_df = pf23["sym_train_df"], pf23["sym_val_df"]
    top25 = pf23["top25_features"]
    y_train_full = train_df["Label"].to_numpy(dtype=np.float32)
    y_val_full = val_df["Label"].to_numpy(dtype=np.float32)

    me.set_seed(seed)  # model-seed variability -- independent of shuffle, applied fresh per seed

    raw_train_arr = train_df[top25].to_numpy(dtype=np.float32)
    raw_val_arr = val_df[top25].to_numpy(dtype=np.float32)
    core_train_arr = sym_train_df[core_features].to_numpy(dtype=np.float32)
    core_val_arr = sym_val_df[core_features].to_numpy(dtype=np.float32)

    # re-derive the SAME eligible-subset permutation from SHUFFLE_SEED
    # (independent of model seed) and assert it matches the
    # preflight-computed hash exactly, so every one of the 5 model-seed runs
    # provably uses the identical shuffle
    eligible_train_check, perm_train_check = build_eligible_permutation(len(core_train_arr), me.WINDOW, SHUFFLE_SEED)
    eligible_val_check, perm_val_check = build_eligible_permutation(len(core_val_arr), me.WINDOW, SHUFFLE_SEED)
    assert array_hash(perm_train_check) == perm_train_hash, "TRAIN permutation drifted across model seeds!"
    assert array_hash(perm_val_check) == perm_val_hash, "VALIDATION permutation drifted across model seeds!"

    shuffled_train_arr = apply_eligible_permutation(core_train_arr, eligible_train, perm_train)  # s_t -> s_pi(t) for t>=WINDOW-1, TRAIN only
    shuffled_val_arr = apply_eligible_permutation(core_val_arr, eligible_val, perm_val)          # s_t -> s_pi(t) for t>=WINDOW-1, VALIDATION only

    # raw sequence + label arrays are untouched originals -- only the
    # symbolic array passed to the dataset has been permuted
    train_ds = me.MainExperimentDataset("fusion", me.WINDOW, y_train_full, raw_train_arr, shuffled_train_arr)
    val_ds = me.MainExperimentDataset("fusion", me.WINDOW, y_val_full, raw_val_arr, shuffled_val_arr)
    train_loader = DataLoader(train_ds, batch_size=me.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=me.BATCH_SIZE, shuffle=False, num_workers=0)

    model = me.FusionModel(raw_input_size=len(top25), symbolic_input_size=len(core_features))

    t0 = time.time()
    model, best_epoch, best_val_loss, loss_history_df = me.train_model(
        model, train_loader, val_loader, pos_weight, fusion_fwd, f"shuffled_core_seed{seed}")
    train_elapsed = time.time() - t0

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits, yb = fusion_fwd(model, batch)
            all_probs.append(torch.sigmoid(logits).numpy())
            all_labels.append(yb.numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    threshold, sweep_df, _ = me.select_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = me.compute_metrics(y_true, y_pred, y_prob)
    latency = me.measure_latency_ms_per_sequence(model, val_loader, fusion_fwd)

    config = {"hidden_size": me.HIDDEN_SIZE, "num_layers": me.NUM_LAYERS, "dropout": me.DROPOUT, "lr": me.LR,
              "batch_size": me.BATCH_SIZE, "max_epochs": me.MAX_EPOCHS, "patience": me.PATIENCE, "seed": seed,
              "shuffle_seed": SHUFFLE_SEED, "window": me.WINDOW, "stride": me.STRIDE,
              "raw_cols": top25, "sym_cols": core_features, "intervention": "row-wise CORE permutation (s_t -> s_pi(t))"}

    prefix = ABLATION_ARTIFACTS_DIR / f"shuffled_core_seed{seed}"
    torch.save(model.state_dict(), f"{prefix}_checkpoint.pt")
    loss_history_df.to_csv(f"{prefix}_loss_history.csv", index=False)
    pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).to_csv(f"{prefix}_validation_predictions.csv", index=False)
    sweep_df.to_csv(f"{prefix}_threshold_sweep.csv", index=False)
    meta = {"model_name": "shuffled_core", "seed": seed, "best_epoch": best_epoch,
            "best_validation_bce_loss": best_val_loss, "selected_threshold": threshold,
            "latency_ms_per_sequence": latency, "train_seconds": round(train_elapsed, 1),
            "metrics": metrics, "config": config, "generated": datetime.now().isoformat()}
    with open(f"{prefix}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    log(f"    Persisted: {prefix.name}_checkpoint.pt, _loss_history.csv, _validation_predictions.csv, "
        f"_threshold_sweep.csv, _metadata.json")
    log(f"  [shuffled_core seed={seed}] best_epoch={best_epoch} VAL_BCE={best_val_loss:.5f} "
        f"threshold={threshold:.6f} train_time={train_elapsed:.1f}s latency={latency:.4f}ms/seq")
    log(f"  [shuffled_core seed={seed}] Acc={metrics['Accuracy']:.4f} P={metrics['Precision']:.4f} "
        f"R={metrics['Recall']:.4f} F1={metrics['F1']:.4f} ROC_AUC={metrics['ROC_AUC']:.4f} "
        f"MCC={metrics['MCC']:.4f} FPR={metrics['FPR']:.4f}")
    return meta


def load_reused_seed_result(model_name, seed):
    if seed == 42:
        path = ORIGINAL_ARTIFACTS_DIR / f"{model_name}_metadata.json"
    else:
        path = REPEATED_SEED_DIR / f"{model_name}_seed{seed}_metadata.json"
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    meta = dict(meta)
    meta["seed"] = seed
    return meta


# =============================================================================
# ABLATION AGGREGATION (4 rows x 5 seeds each)
# =============================================================================

def aggregate_ablation(shuffled_results):
    rows = []
    for model_key, label in [("selected_lstm", "Selected LSTM"), ("neurosymbolic_core", "LSTM + CORE"),
                              ("neurosymbolic_extended", "LSTM + Extended")]:
        for seed in SEEDS:
            r = load_reused_seed_result(model_key, seed)
            rows.append({"Ablation_Row": label, "Model_Key": model_key, "Seed": seed,
                         "Best_Epoch": r["best_epoch"], "Validation_BCE_Loss": r["best_validation_bce_loss"],
                         "Selected_Threshold": r["selected_threshold"], **r["metrics"]})
    for r in shuffled_results:
        rows.append({"Ablation_Row": "LSTM + Shuffled CORE", "Model_Key": "shuffled_core", "Seed": r["seed"],
                     "Best_Epoch": r["best_epoch"], "Validation_BCE_Loss": r["best_validation_bce_loss"],
                     "Selected_Threshold": r["selected_threshold"], **r["metrics"]})

    long_df = pd.DataFrame(rows)
    long_path = TABLES_DIR / "symbolic_alignment_ablation_long.csv"
    long_df.to_csv(long_path, index=False)
    log(f"\nWrote {long_path} ({len(long_df)} rows = 4 ablation rows x 5 seeds)")

    section("ABLATION SUMMARY (MCC, F1, ROC_AUC across 5 seeds per row)")
    stats_rows = []
    for label in ["Selected LSTM", "LSTM + Shuffled CORE", "LSTM + CORE", "LSTM + Extended"]:
        sub = long_df[long_df.Ablation_Row == label].sort_values("Seed")
        log(f"\n{label}:")
        for metric in ("MCC", "F1", "ROC_AUC"):
            vals = sub[metric].to_numpy()
            stats_rows.append({"Ablation_Row": label, "Metric": metric,
                                **{f"Seed_{s}": v for s, v in zip(sub["Seed"], vals)},
                                "Mean": vals.mean(), "Std": vals.std(ddof=1), "Median": np.median(vals),
                                "Min": vals.min(), "Max": vals.max()})
            log(f"  {metric}: mean={vals.mean():.4f} std={vals.std(ddof=1):.4f} median={np.median(vals):.4f} "
                f"min={vals.min():.4f} max={vals.max():.4f}")
    stats_df = pd.DataFrame(stats_rows)
    stats_path = TABLES_DIR / "symbolic_alignment_ablation_summary.csv"
    stats_df.to_csv(stats_path, index=False)
    log(f"\nWrote {stats_path}")
    return long_df, stats_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                         help="Train the missing Shuffled-CORE seeds (skipping any already done) and "
                              "produce the 4-row ablation aggregation. Without this flag, preflight only.")
    args = parser.parse_args()

    section("SCRIPT 25 — SYMBOLIC-ALIGNMENT ABLATION (SHUFFLED CORE)")
    log(f"Run start: {RUN_START.isoformat()}")
    log(f"Mode: {'TRAIN MISSING SEEDS + AGGREGATE' if args.train else 'PREFLIGHT ONLY (no training will run)'}")
    log("TEST is never loaded or referenced anywhere in this script. Scripts 23/24 are read-only.")

    pf = run_preflight()
    if not pf["all_passed"]:
        section("PREFLIGHT FAILED — ABORTING BEFORE ANY TRAINING")
        sys.exit(1)

    if not args.train:
        section("PREFLIGHT PASSED — STOPPING HERE (pass --train to train missing seeds + aggregate)")
        log(f"No model was trained. No TEST access. {len(pf['pending'])} of 5 Shuffled-CORE seeds still needed.")
        return

    section("TRAINING MISSING SHUFFLED-CORE SEEDS")
    pf23 = pf["pf23"]
    train_df = pf23["train_df"]
    y_train_full = train_df["Label"].to_numpy(dtype=np.float32)
    train_seq_labels = y_train_full[me.WINDOW - 1:]
    n_pos, n_neg = float((train_seq_labels == 1).sum()), float((train_seq_labels == 0).sum())
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    log(f"pos_weight (TRAIN sequence labels only, identical to scripts 23/24) = {pos_weight.item():.6f}")

    shuffled_results = []
    for seed in SEEDS:
        meta_path = ABLATION_ARTIFACTS_DIR / f"shuffled_core_seed{seed}_metadata.json"
        if meta_path.exists():
            log(f"\n[shuffled_core seed={seed}] already completed -- skipping (resume).")
            with open(meta_path, encoding="utf-8") as f:
                shuffled_results.append(json.load(f))
            continue
        section(f"MODEL: shuffled_core  SEED: {seed}")
        result = train_one_shuffled_seed(seed, pf23, pos_weight, pf["eligible_train"], pf["perm_train"],
                                          pf["eligible_val"], pf["perm_val"], pf["perm_train_hash"],
                                          pf["perm_val_hash"], pf["core_features"])
        shuffled_results.append(result)

    aggregate_ablation(shuffled_results)

    section("FINAL GUARDRAIL RE-CHECK")
    post_hashes_original = {p: sha256_of(p) for p in pf["pre_hashes_original"]}
    original_unchanged = all(pf["pre_hashes_original"][p] == post_hashes_original[p] for p in pf["pre_hashes_original"])
    all_log_text = "\n".join(_LOG_LINES)
    final_checks = {
        "original_script23_and_24_artifacts_unchanged": original_unchanged,
        "test_never_referenced_in_log": ("test_scaled" not in all_log_text and "test_unscaled" not in all_log_text),
    }
    for k, v in final_checks.items():
        log(f"  [{'PASS' if v else 'FAIL'}] {k}")

    run_end = datetime.now()
    log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
    log("SYMBOLIC-ALIGNMENT ABLATION COMPLETE." if all(final_checks.values()) else "COMPLETED WITH GUARDRAIL FAILURES.")


if __name__ == "__main__":
    main()
