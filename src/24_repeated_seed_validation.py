"""
code/24_repeated_seed_validation.py

REPEATED-SEED VALIDATION ROBUSTNESS EXPERIMENT — follows the Script 23
diagnostic verdict (B: REPEAT-SEED VALIDATION NEEDED). Repeats Full LSTM,
Selected LSTM, Neuro-symbolic CORE, and Neuro-symbolic Extended (NOT
Symbolic-only) across SEEDS = [7, 21, 42, 84, 123].

This script does NOT modify code/23_main_experiment.py in any way. It loads
it as a read-only module (by file path, since its filename cannot be
`import`ed normally) and reuses its exact shared methodology --
MainExperimentDataset, LSTMOnlyModel, FusionModel, train_model,
select_threshold, compute_metrics, measure_latency_ms_per_sequence,
run_preflight -- so the repeated-seed runs cannot silently diverge from the
already-frozen single-seed protocol. Only the seed changes between runs;
every architecture/optimizer/loss/threshold-rule/data setting is identical.

Seed 42 is NOT retrained: its four models' results are read directly from
the existing, unmodified code/23_main_experiment.py artifacts in
results/main_experiment/. Only seeds 7, 21, 84, 123 are newly trained here,
written to results/repeated_seed/, never touching or duplicating the
original seed-42 files.

Resume-safe: before training a given (model, seed) pair, this script checks
whether its metadata file already exists in results/repeated_seed/ and
skips it if so -- an interrupted run can be restarted with --train and will
only train whatever is still missing.

Two modes, both never touching TEST:
  - Default (no flag): PREFLIGHT ONLY. Verifies the frozen inputs (via
    script 23's own run_preflight()), the frozen seed list, that seed-42
    artifacts exist and are readable, and reports how many of the 16
    (model, new-seed) pairs are already done vs. still needed. Trains
    nothing.
  - `--train`: trains whatever (model, seed) pairs are missing (skipping
    completed ones), persists per-pair artifacts, then aggregates the full
    5-seed summary (per-model seed tables + descriptive stats + paired
    seed-level deltas) across all 20 model-seed results (4 reused + 16 new).
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
# Load script 23 as a read-only module (never modified, never re-executed via
# __main__) so this driver reuses its exact shared methodology.
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
SCRIPT23_PATH = PROJECT_ROOT / "src" / "23_main_experiment.py"

_spec = importlib.util.spec_from_file_location("main_experiment_v23", SCRIPT23_PATH)
me = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(me)  # runs only module-level defs (functions/classes/constants); main() is not invoked

TABLES_DIR = PROJECT_ROOT / "tables"
RESULTS_DIR = PROJECT_ROOT / "results"
ORIGINAL_ARTIFACTS_DIR = RESULTS_DIR / "main_experiment"       # script 23's untouched output
SEED_ARTIFACTS_DIR = RESULTS_DIR / "repeated_seed"              # this script's own output only
SEED_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [7, 21, 42, 84, 123]
REUSED_SEED = 42
NEW_SEEDS = [s for s in SEEDS if s != REUSED_SEED]
MODEL_NAMES = ["full_lstm", "selected_lstm", "neurosymbolic_core", "neurosymbolic_extended"]

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
# PREFLIGHT
# =============================================================================

def run_preflight():
    section("PREFLIGHT — REUSING SCRIPT 23's GUARDRAILS + REPEATED-SEED-SPECIFIC CHECKS")
    checks = {}

    log("Delegating to code/23_main_experiment.py's own run_preflight() (frozen inputs, "
        "alignment, NaN, TEST-absence, RF/CORE/Extended feature-count checks)...")
    pf = me.run_preflight()
    checks["script23_preflight_passed"] = pf["all_passed"]

    checks["seed_set_matches_locked_spec"] = (SEEDS == [7, 21, 42, 84, 123])
    log(f"  Seed set: {SEEDS} (locked)")

    # --- seed-42 reused artifacts must exist and be readable for all 4 models ---
    reuse_ok = {}
    for name in MODEL_NAMES:
        meta_path = ORIGINAL_ARTIFACTS_DIR / f"{name}_metadata.json"
        pred_path = ORIGINAL_ARTIFACTS_DIR / f"{name}_validation_predictions.csv"
        ok = meta_path.exists() and pred_path.exists()
        reuse_ok[name] = ok
        log(f"  seed-42 reuse source for {name}: metadata={'found' if meta_path.exists() else 'MISSING'}, "
            f"predictions={'found' if pred_path.exists() else 'MISSING'}")
    checks["all_seed42_reuse_artifacts_exist"] = all(reuse_ok.values())

    # --- original script-23 artifacts must remain untouched by this script ---
    original_files = list(ORIGINAL_ARTIFACTS_DIR.glob("*"))
    pre_hashes = {p: sha256_of(p) for p in original_files}
    checks["original_script23_artifacts_present"] = (len(original_files) == 25)  # 5 models x 5 files each
    log(f"  Original script-23 artifacts found: {len(original_files)} (expected 25)")

    # --- resume status: how many of the 16 (model, new-seed) pairs are already done ---
    pending, done = [], []
    for name in MODEL_NAMES:
        for seed in NEW_SEEDS:
            meta_path = SEED_ARTIFACTS_DIR / f"{name}_seed{seed}_metadata.json"
            (done if meta_path.exists() else pending).append((name, seed))
    log(f"\n  Repeated-seed pairs already completed: {len(done)} of {len(MODEL_NAMES) * len(NEW_SEEDS)}")
    for name, seed in done:
        log(f"    done: {name} seed={seed}")
    log(f"  Repeated-seed pairs still needed: {len(pending)}")
    for name, seed in pending:
        log(f"    pending: {name} seed={seed}")
    checks["expected_total_new_pairs_is_16"] = (len(done) + len(pending) == 16)

    log("\nGuardrail results:")
    for k, v in checks.items():
        log(f"  [{'PASS' if v else 'FAIL'}] {k}")
    all_passed = all(checks.values())
    log(f"\nPREFLIGHT OVERALL: {'PASS' if all_passed else 'FAIL'}")

    return {"checks": checks, "all_passed": all_passed, "pf23": pf, "pre_hashes_original": pre_hashes,
            "pending": pending, "done": done}


# =============================================================================
# TRAIN ONE (model, seed) PAIR -- reuses script 23's exact shared functions
# =============================================================================

def build_spec(name, pf23):
    all_51, top25, core6, ext16 = pf23["all_51_features"], pf23["top25_features"], pf23["core_features"], pf23["extended_features"]
    if name == "full_lstm":
        return {"mode": "raw", "raw_cols": all_51}
    elif name == "selected_lstm":
        return {"mode": "raw", "raw_cols": top25}
    elif name == "neurosymbolic_core":
        return {"mode": "fusion", "raw_cols": top25, "sym_cols": core6}
    elif name == "neurosymbolic_extended":
        return {"mode": "fusion", "raw_cols": top25, "sym_cols": ext16}
    raise ValueError(name)


def raw_fwd(model, batch):
    xb, yb = batch
    return model(xb), yb


def fusion_fwd(model, batch):
    xb, sb, yb = batch
    return model(xb, sb), yb


def train_one_pair(name, seed, pf23, pos_weight):
    spec = build_spec(name, pf23)
    train_df, val_df = pf23["train_df"], pf23["val_df"]
    sym_train_df, sym_val_df = pf23["sym_train_df"], pf23["sym_val_df"]
    y_train_full = train_df["Label"].to_numpy(dtype=np.float32)
    y_val_full = val_df["Label"].to_numpy(dtype=np.float32)

    me.set_seed(seed)

    raw_train_arr = train_df[spec["raw_cols"]].to_numpy(dtype=np.float32) if "raw_cols" in spec else None
    raw_val_arr = val_df[spec["raw_cols"]].to_numpy(dtype=np.float32) if "raw_cols" in spec else None
    sym_train_arr = sym_train_df[spec["sym_cols"]].to_numpy(dtype=np.float32) if "sym_cols" in spec else None
    sym_val_arr = sym_val_df[spec["sym_cols"]].to_numpy(dtype=np.float32) if "sym_cols" in spec else None

    train_ds = me.MainExperimentDataset(spec["mode"], me.WINDOW, y_train_full, raw_train_arr, sym_train_arr)
    val_ds = me.MainExperimentDataset(spec["mode"], me.WINDOW, y_val_full, raw_val_arr, sym_val_arr)
    train_loader = DataLoader(train_ds, batch_size=me.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=me.BATCH_SIZE, shuffle=False, num_workers=0)

    if spec["mode"] == "raw":
        model = me.LSTMOnlyModel(input_size=len(spec["raw_cols"]))
        fwd = raw_fwd
    else:
        model = me.FusionModel(raw_input_size=len(spec["raw_cols"]), symbolic_input_size=len(spec["sym_cols"]))
        fwd = fusion_fwd

    t0 = time.time()
    model, best_epoch, best_val_loss, loss_history_df = me.train_model(
        model, train_loader, val_loader, pos_weight, fwd, f"{name}_seed{seed}")
    train_elapsed = time.time() - t0

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits, yb = fwd(model, batch)
            all_probs.append(torch.sigmoid(logits).numpy())
            all_labels.append(yb.numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    threshold, sweep_df, _ = me.select_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = me.compute_metrics(y_true, y_pred, y_prob)
    latency = me.measure_latency_ms_per_sequence(model, val_loader, fwd)

    config = {"hidden_size": me.HIDDEN_SIZE, "num_layers": me.NUM_LAYERS, "dropout": me.DROPOUT, "lr": me.LR,
              "batch_size": me.BATCH_SIZE, "max_epochs": me.MAX_EPOCHS, "patience": me.PATIENCE, "seed": seed,
              "window": me.WINDOW, "stride": me.STRIDE, "raw_cols": spec.get("raw_cols"), "sym_cols": spec.get("sym_cols")}

    prefix = SEED_ARTIFACTS_DIR / f"{name}_seed{seed}"
    torch.save(model.state_dict(), f"{prefix}_checkpoint.pt")
    loss_history_df.to_csv(f"{prefix}_loss_history.csv", index=False)
    pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).to_csv(f"{prefix}_validation_predictions.csv", index=False)
    sweep_df.to_csv(f"{prefix}_threshold_sweep.csv", index=False)
    meta = {"model_name": name, "seed": seed, "best_epoch": best_epoch, "best_validation_bce_loss": best_val_loss,
            "selected_threshold": threshold, "latency_ms_per_sequence": latency, "train_seconds": round(train_elapsed, 1),
            "metrics": metrics, "config": config, "generated": datetime.now().isoformat()}
    with open(f"{prefix}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    log(f"    Persisted: {prefix.name}_checkpoint.pt, _loss_history.csv, _validation_predictions.csv, "
        f"_threshold_sweep.csv, _metadata.json")
    log(f"  [{name} seed={seed}] best_epoch={best_epoch} VAL_BCE={best_val_loss:.5f} threshold={threshold:.6f} "
        f"train_time={train_elapsed:.1f}s latency={latency:.4f}ms/seq")
    log(f"  [{name} seed={seed}] Acc={metrics['Accuracy']:.4f} P={metrics['Precision']:.4f} R={metrics['Recall']:.4f} "
        f"F1={metrics['F1']:.4f} ROC_AUC={metrics['ROC_AUC']:.4f} MCC={metrics['MCC']:.4f} FPR={metrics['FPR']:.4f}")
    return meta


def load_seed42_result(name):
    """Reads seed-42's result directly from script 23's untouched artifacts --
    never retrains, never copies/duplicates the files."""
    with open(ORIGINAL_ARTIFACTS_DIR / f"{name}_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    meta = dict(meta)
    meta["seed"] = 42
    meta.setdefault("train_seconds", None)  # script 23's metadata didn't store this field separately
    return meta


# =============================================================================
# AGGREGATION: per-model seed table, descriptive stats, paired seed-level deltas
# =============================================================================

def aggregate(all_results):
    rows = []
    for r in all_results:
        rows.append({"Model": r["model_name"], "Seed": r["seed"], "Best_Epoch": r["best_epoch"],
                     "Validation_BCE_Loss": r["best_validation_bce_loss"], "Selected_Threshold": r["selected_threshold"],
                     **r["metrics"]})
    long_df = pd.DataFrame(rows)
    long_path = TABLES_DIR / "repeated_seed_results_long.csv"
    long_df.to_csv(long_path, index=False)
    log(f"\nWrote {long_path} ({len(long_df)} rows = {len(MODEL_NAMES)} models x {len(SEEDS)} seeds)")

    section("PER-MODEL DESCRIPTIVE STATISTICS ACROSS 5 SEEDS (MCC, F1, ROC_AUC)")
    stats_rows = []
    for name in MODEL_NAMES:
        sub = long_df[long_df.Model == name].sort_values("Seed")
        for metric in ("MCC", "F1", "ROC_AUC"):
            vals = sub[metric].to_numpy()
            stats_rows.append({"Model": name, "Metric": metric,
                                **{f"Seed_{s}": v for s, v in zip(sub["Seed"], vals)},
                                "Mean": vals.mean(), "Std": vals.std(ddof=1), "Median": np.median(vals),
                                "Min": vals.min(), "Max": vals.max()})
        log(f"\n{name}:")
        for metric in ("MCC", "F1", "ROC_AUC"):
            vals = sub[metric].to_numpy()
            log(f"  {metric}: seeds={dict(zip(sub['Seed'], np.round(vals, 4)))} "
                f"mean={vals.mean():.4f} std={vals.std(ddof=1):.4f} median={np.median(vals):.4f} "
                f"min={vals.min():.4f} max={vals.max():.4f}")
    stats_df = pd.DataFrame(stats_rows)
    stats_path = TABLES_DIR / "repeated_seed_summary_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    log(f"\nWrote {stats_path}")

    section("PAIRED SEED-LEVEL DELTAS")
    pair_defs = [("full_lstm", "selected_lstm"), ("selected_lstm", "neurosymbolic_core"),
                 ("selected_lstm", "neurosymbolic_extended"), ("neurosymbolic_core", "neurosymbolic_extended")]
    delta_rows = []
    for a, b in pair_defs:
        sub_a = long_df[long_df.Model == a].set_index("Seed")
        sub_b = long_df[long_df.Model == b].set_index("Seed")
        log(f"\n{a} -> {b}:")
        for metric in ("MCC", "F1", "ROC_AUC"):
            deltas = (sub_b[metric] - sub_a[metric]).reindex(SEEDS)
            n_higher = int((deltas > 0).sum())
            delta_rows.append({"From": a, "To": b, "Metric": metric,
                                **{f"Seed_{s}_Delta": d for s, d in deltas.items()},
                                "Mean_Delta": deltas.mean(), "Median_Delta": deltas.median(),
                                "N_Seeds_Second_Higher": n_higher, "N_Seeds_Total": len(SEEDS)})
            log(f"  {metric}: deltas={dict(zip(SEEDS, np.round(deltas.values, 4)))} "
                f"mean={deltas.mean():.4f} median={deltas.median():.4f} "
                f"({n_higher}/{len(SEEDS)} seeds: {b} > {a})")
    delta_df = pd.DataFrame(delta_rows)
    delta_path = TABLES_DIR / "repeated_seed_paired_deltas.csv"
    delta_df.to_csv(delta_path, index=False)
    log(f"\nWrote {delta_path}")

    return long_df, stats_df, delta_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                         help="Train the 16 missing (model, new-seed) pairs (skipping any already done) "
                              "and produce the full 5-seed aggregation. Without this flag, preflight only.")
    args = parser.parse_args()

    section("SCRIPT 24 — REPEATED-SEED VALIDATION ROBUSTNESS EXPERIMENT")
    log(f"Run start: {RUN_START.isoformat()}")
    log(f"Mode: {'TRAIN MISSING PAIRS + AGGREGATE' if args.train else 'PREFLIGHT ONLY (no training will run)'}")
    log("TEST is never loaded or referenced anywhere in this script. code/23_main_experiment.py is read-only.")

    pf = run_preflight()
    if not pf["all_passed"]:
        section("PREFLIGHT FAILED — ABORTING BEFORE ANY TRAINING")
        sys.exit(1)

    if not args.train:
        section("PREFLIGHT PASSED — STOPPING HERE (pass --train to train missing pairs + aggregate)")
        log(f"No model was trained. No TEST access. {len(pf['pending'])} of 16 new (model, seed) pairs still needed.")
        return

    section("TRAINING MISSING (model, seed) PAIRS")
    pf23 = pf["pf23"]
    train_df = pf23["train_df"]
    y_train_full = train_df["Label"].to_numpy(dtype=np.float32)
    train_seq_labels = y_train_full[me.WINDOW - 1:]
    n_pos, n_neg = float((train_seq_labels == 1).sum()), float((train_seq_labels == 0).sum())
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    log(f"pos_weight (TRAIN sequence labels only, identical for every seed) = {pos_weight.item():.6f}")

    all_results = []
    for name in MODEL_NAMES:
        all_results.append(load_seed42_result(name))  # reused, not retrained
        log(f"\nReused seed-42 result for {name} from results/main_experiment/ (not retrained).")
        for seed in NEW_SEEDS:
            meta_path = SEED_ARTIFACTS_DIR / f"{name}_seed{seed}_metadata.json"
            if meta_path.exists():
                log(f"\n[{name} seed={seed}] already completed -- skipping (resume).")
                with open(meta_path, encoding="utf-8") as f:
                    all_results.append(json.load(f))
                continue
            section(f"MODEL: {name}  SEED: {seed}")
            result = train_one_pair(name, seed, pf23, pos_weight)
            all_results.append(result)

    aggregate(all_results)

    section("FINAL GUARDRAIL RE-CHECK")
    post_hashes_original = {p: sha256_of(p) for p in pf["pre_hashes_original"]}
    original_unchanged = all(pf["pre_hashes_original"][p] == post_hashes_original[p] for p in pf["pre_hashes_original"])
    all_log_text = "\n".join(_LOG_LINES)
    final_checks = {
        "original_script23_artifacts_unchanged": original_unchanged,
        "test_never_referenced_in_log": ("test_scaled" not in all_log_text and "test_unscaled" not in all_log_text),
    }
    for k, v in final_checks.items():
        log(f"  [{'PASS' if v else 'FAIL'}] {k}")

    run_end = datetime.now()
    log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
    log("REPEATED-SEED EXPERIMENT COMPLETE." if all(final_checks.values()) else "COMPLETED WITH GUARDRAIL FAILURES.")


if __name__ == "__main__":
    main()
