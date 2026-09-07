"""
code/23_main_experiment.py

MAIN FIVE-MODEL EXPERIMENT — Full LSTM / Selected LSTM / Symbolic-only /
Neuro-symbolic CORE / Neuro-symbolic Extended. VALIDATION-only training and
evaluation. TEST is never loaded or referenced anywhere in this script.

This script has two modes, controlled by the --train flag:
  - Default (no flag): PREFLIGHT ONLY. Loads all frozen inputs, runs every
    guardrail check, and exits without training anything. Safe to run
    freely -- this is the mode used to validate the pipeline before any
    long training run is approved.
  - `--train`: runs the full five-model experiment (after preflight passes).

Frozen decisions implemented here (all previously approved, none re-derived):
  - RF top-25 subset (script 22's selected K=25, re-derived from the frozen,
    hash-verified table_rf_feature_ranking.csv -- never re-ranked).
  - CORE (6) and Extended (16) symbolic feature sets (scripts 17/18's frozen
    definitions) and their already zero-filled per-timestep matrices (script
    21's output) -- read directly, never recomputed.
  - Sequence window=120, stride=1, last-timestep label, windows never cross
    the TRAIN/VALIDATION boundary, symbolic vector s_t aligned to the same
    final timestep t as h_t.
  - LSTM: hidden_size=64, 1 layer, external dropout=0.2, Adam(lr=1e-3),
    batch_size=256, max_epochs=30, early stopping (patience=5, monitors
    VALIDATION BCE loss, restores best checkpoint), seed=42,
    BCEWithLogitsLoss with pos_weight from TRAIN sequence labels only.
  - Neuro-symbolic CORE/Extended: joint end-to-end training -- each gets its
    own independently-initialized Selected-LSTM-architecture encoder, not a
    frozen/reused one.
  - Threshold selection: per model, sweep every unique VALIDATION predicted
    probability; select by (1) highest MCC, (2) highest F1, (3) lowest FPR,
    (4) threshold closest to 0.5, (5) higher threshold -- identical rule for
    all five models, fixed before any result is observed.
  - Full persistence per model: checkpoint, loss history, VALIDATION
    predictions, threshold sweep, confusion matrix, metrics, latency, config.
"""

import sys
import json
import time
import random
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              matthews_corrcoef, roc_auc_score, confusion_matrix)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Paths (TEST is never defined anywhere in this file)
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
TABLES_DIR = PROJECT_ROOT / "tables"
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACTS_DIR = RESULTS_DIR / "main_experiment"
for d in (TABLES_DIR, RESULTS_DIR, ARTIFACTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TRAIN_SCALED_PATH = PROCESSED_DIR / "train_scaled.csv"
VAL_SCALED_PATH = PROCESSED_DIR / "validation_scaled.csv"
RF_RANKING_PATH = TABLES_DIR / "table_rf_feature_ranking.csv"
RF_SWEEP_PATH = TABLES_DIR / "rf_k_sweep_results.csv"
CORE_FEATURES_PATH = TABLES_DIR / "final_core_symbolic_features.csv"
EXTENDED_FEATURES_PATH = TABLES_DIR / "final_extended_symbolic_features.csv"
SYMBOLIC_TRAIN_PATH = TABLES_DIR / "symbolic_feature_matrix_train.csv"
SYMBOLIC_VAL_PATH = TABLES_DIR / "symbolic_feature_matrix_validation.csv"

# ---------------------------------------------------------------------------
# Locked configuration
# ---------------------------------------------------------------------------
WINDOW = 120
STRIDE = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.2
LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 30
PATIENCE = 5
SEED = 42
EXPECTED_TRAIN_ROWS = 270000
EXPECTED_VAL_ROWS = 68319
EXPECTED_TRAIN_SEQUENCES = EXPECTED_TRAIN_ROWS - WINDOW + 1
EXPECTED_VAL_SEQUENCES = EXPECTED_VAL_ROWS - WINDOW + 1
EXPECTED_N_CORE = 6
EXPECTED_N_EXTENDED = 16
EXPECTED_SELECTED_K = 25

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


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================================
# PREFLIGHT: LOAD FROZEN INPUTS + GUARDRAIL CHECKS (always runs, cheap, safe)
# =============================================================================

def run_preflight():
    section("PREFLIGHT — LOADING FROZEN INPUTS AND RUNNING GUARDRAIL CHECKS")
    checks = {}

    frozen_paths = [RF_RANKING_PATH, RF_SWEEP_PATH, CORE_FEATURES_PATH, EXTENDED_FEATURES_PATH,
                    SYMBOLIC_TRAIN_PATH, SYMBOLIC_VAL_PATH]
    pre_hashes = {p: sha256_of(p) for p in frozen_paths if p.exists()}
    checks["all_frozen_input_files_exist"] = all(p.exists() for p in frozen_paths)
    for p in frozen_paths:
        log(f"  {'found' if p.exists() else 'MISSING'}: {p.relative_to(PROJECT_ROOT)}"
            f"{' sha256=' + pre_hashes[p][:12] + '...' if p in pre_hashes else ''}")

    # --- TEST-absence: (a) no TEST path variable ever assigned in this module's
    # actual code (static check), and (b) no TEST filename ever appears in the
    # runtime log (dynamic check, evaluated at the very end in main() once all
    # log lines exist). Note: (a) deliberately does NOT grep the source text for
    # the TEST filenames themselves, since this very check's own code would then
    # match its own forbidden-token list -- a self-referential false positive.
    import re as _re
    module_src = SCRIPT_PATH.read_text(encoding="utf-8")
    checks["no_test_path_variable_assigned"] = (_re.search(r"^\s*TEST\w*_PATH\s*=", module_src, flags=_re.M) is None)

    # --- Load TRAIN / VALIDATION (scaled), never TEST ---
    train_df = pd.read_csv(TRAIN_SCALED_PATH)
    val_df = pd.read_csv(VAL_SCALED_PATH)
    assert "test" not in TRAIN_SCALED_PATH.name and "test" not in VAL_SCALED_PATH.name
    checks["train_row_count_correct"] = (len(train_df) == EXPECTED_TRAIN_ROWS)
    checks["validation_row_count_correct"] = (len(val_df) == EXPECTED_VAL_ROWS)
    log(f"  TRAIN rows: {len(train_df):,} (expected {EXPECTED_TRAIN_ROWS:,})")
    log(f"  VALIDATION rows: {len(val_df):,} (expected {EXPECTED_VAL_ROWS:,})")

    n_train_seq = len(train_df) - WINDOW + 1
    n_val_seq = len(val_df) - WINDOW + 1
    checks["train_sequence_count_correct"] = (n_train_seq == EXPECTED_TRAIN_SEQUENCES)
    checks["validation_sequence_count_correct"] = (n_val_seq == EXPECTED_VAL_SEQUENCES)
    log(f"  TRAIN sequences (window={WINDOW}): {n_train_seq:,} (expected {EXPECTED_TRAIN_SEQUENCES:,})")
    log(f"  VALIDATION sequences (window={WINDOW}): {n_val_seq:,} (expected {EXPECTED_VAL_SEQUENCES:,})")

    # --- Frozen RF ranking -> all 51, and top-25 (re-derived, never re-ranked) ---
    rf_ranking = pd.read_csv(RF_RANKING_PATH).sort_values("Rank").reset_index(drop=True)
    all_51_features = rf_ranking["Feature"].tolist()
    top25_features = rf_ranking.loc[rf_ranking.Rank <= 25, "Feature"].tolist()
    checks["exactly_51_total_features"] = (len(all_51_features) == 51)
    checks["exactly_25_selected_features"] = (len(top25_features) == 25)

    # cross-check K=25 against script 22's own frozen selection artifact,
    # rather than trusting the hardcoded "25" in isolation
    sweep_df = pd.read_csv(RF_SWEEP_PATH)
    selected_row = sweep_df[sweep_df.Selected == True]
    checks["rf_sweep_selected_k_is_25"] = (len(selected_row) == 1 and int(selected_row.iloc[0]["K"]) == EXPECTED_SELECTED_K)
    log(f"  RF ranking: {len(all_51_features)} total features; top-25 (frozen K from script 22's "
        f"Selected row = {int(selected_row.iloc[0]['K']) if len(selected_row) else 'MISSING'}): {top25_features}")

    # --- Frozen CORE (6) / Extended (16) symbolic feature definitions ---
    core_features = pd.read_csv(CORE_FEATURES_PATH)["Feature_Name"].tolist()
    extended_features = pd.read_csv(EXTENDED_FEATURES_PATH)["Feature_Name"].tolist()
    checks["exactly_6_core_features"] = (len(core_features) == EXPECTED_N_CORE)
    checks["exactly_16_extended_features"] = (len(extended_features) == EXPECTED_N_EXTENDED)
    checks["core_is_subset_of_extended"] = set(core_features).issubset(set(extended_features))
    log(f"  CORE features ({len(core_features)}): {core_features}")
    log(f"  Extended features ({len(extended_features)}): {extended_features}")

    # --- Symbolic feature matrices (already zero-filled, script 21's output) ---
    sym_train_df = pd.read_csv(SYMBOLIC_TRAIN_PATH)
    sym_val_df = pd.read_csv(SYMBOLIC_VAL_PATH)
    checks["symbolic_train_row_count_correct"] = (len(sym_train_df) == EXPECTED_TRAIN_ROWS)
    checks["symbolic_validation_row_count_correct"] = (len(sym_val_df) == EXPECTED_VAL_ROWS)

    # raw/symbolic row alignment: Timestamp columns must match exactly, row for row
    checks["train_timestamp_alignment"] = bool((train_df["Timestamp"].to_numpy() == sym_train_df["Timestamp"].to_numpy()).all())
    checks["validation_timestamp_alignment"] = bool((val_df["Timestamp"].to_numpy() == sym_val_df["Timestamp"].to_numpy()).all())
    # label alignment: Label columns must match exactly, row for row
    checks["train_label_alignment"] = bool((train_df["Label"].to_numpy() == sym_train_df["Label"].to_numpy()).all())
    checks["validation_label_alignment"] = bool((val_df["Label"].to_numpy() == sym_val_df["Label"].to_numpy()).all())

    # --- No NaNs anywhere in the columns this experiment actually consumes ---
    checks["no_nan_in_raw_51_features"] = not (train_df[all_51_features].isna().any().any()
                                                or val_df[all_51_features].isna().any().any())
    checks["no_nan_in_symbolic_extended_16"] = not (sym_train_df[extended_features].isna().any().any()
                                                      or sym_val_df[extended_features].isna().any().any())

    # dynamic TEST-absence check: everything logged in preflight so far must
    # never mention a TEST filename (this is the log itself, not the source --
    # no self-reference risk).
    checks["no_test_filename_in_preflight_log"] = not any(
        ("test_scaled" in line or "test_unscaled" in line) for line in _LOG_LINES
    )

    log("\nGuardrail results:")
    for k, v in checks.items():
        log(f"  [{'PASS' if v else 'FAIL'}] {k}")
    all_passed = all(checks.values())
    log(f"\nPREFLIGHT OVERALL: {'PASS' if all_passed else 'FAIL'}")

    return {
        "checks": checks, "all_passed": all_passed, "pre_hashes": pre_hashes,
        "train_df": train_df, "val_df": val_df, "rf_ranking": rf_ranking,
        "all_51_features": all_51_features, "top25_features": top25_features,
        "core_features": core_features, "extended_features": extended_features,
        "sym_train_df": sym_train_df, "sym_val_df": sym_val_df,
    }


# =============================================================================
# DATASETS (one shared class, three modes -- cannot silently diverge)
# =============================================================================

class MainExperimentDataset(Dataset):
    """
    mode='raw'      -> (x_window[WINDOW, n_raw], y)            Full LSTM, Selected LSTM
    mode='symbolic' -> (s_t[n_symbolic], y)                    Symbolic-only (no window; single
                                                                 row at the same aligned index t
                                                                 as the windowed models, so all
                                                                 five models are compared on the
                                                                 identical set of timesteps)
    mode='fusion'   -> (x_window[WINDOW, n_raw], s_t[n_symbolic], y)   Neuro-symbolic CORE/Extended
    """

    def __init__(self, mode, window, y_array, raw_array=None, symbolic_array=None):
        assert mode in ("raw", "symbolic", "fusion")
        self.mode = mode
        self.window = window
        self.y = y_array.astype(np.float32)
        self.raw = raw_array.astype(np.float32) if raw_array is not None else None
        self.sym = symbolic_array.astype(np.float32) if symbolic_array is not None else None
        self.n = len(self.y) - window + 1
        assert self.n > 0

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        end = idx + self.window  # exclusive end; last included row index = end-1
        label = self.y[end - 1]  # last-timestep label
        if self.mode == "raw":
            x = self.raw[idx:end, :]
            return torch.from_numpy(x), torch.tensor(label, dtype=torch.float32)
        elif self.mode == "symbolic":
            s = self.sym[end - 1, :]  # s_t aligned to the SAME final timestep t
            return torch.from_numpy(s), torch.tensor(label, dtype=torch.float32)
        else:  # fusion
            x = self.raw[idx:end, :]
            s = self.sym[end - 1, :]
            return torch.from_numpy(x), torch.from_numpy(s), torch.tensor(label, dtype=torch.float32)


# =============================================================================
# MODELS (shared LSTM architecture everywhere it appears)
# =============================================================================

class LSTMOnlyModel(nn.Module):
    """Full LSTM / Selected LSTM: LSTM encoder -> external dropout -> linear head."""

    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h_t = out[:, -1, :]
        h_t = self.dropout(h_t)
        return self.fc(h_t).squeeze(-1)


class SymbolicOnlyModel(nn.Module):
    """Symbolic-only: a single linear classifier directly on s_t. Same classifier
    complexity philosophy as every other model -- only the input differs."""

    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, s):
        return self.fc(s).squeeze(-1)


class FusionModel(nn.Module):
    """Neuro-symbolic CORE/Extended: its OWN independently-initialized LSTM
    encoder (identical architecture to LSTMOnlyModel), trained jointly with a
    linear head over [h_t ; s_t]. Not a frozen/reused Selected-LSTM encoder."""

    def __init__(self, raw_input_size, symbolic_input_size, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size=raw_input_size, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size + symbolic_input_size, 1)

    def forward(self, x, s):
        out, _ = self.lstm(x)
        h_t = out[:, -1, :]
        h_t = self.dropout(h_t)
        combined = torch.cat([h_t, s], dim=-1)
        return self.fc(combined).squeeze(-1)


# =============================================================================
# SHARED TRAINING LOOP (identical logic for every model; only forward_fn differs)
# =============================================================================

def train_model(model, train_loader, val_loader, pos_weight, forward_fn, model_name):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0
    loss_history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, yb = forward_fn(model, batch)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * yb.size(0)
            train_n += yb.size(0)
        train_loss = train_loss_sum / train_n

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                logits, yb = forward_fn(model, batch)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * yb.size(0)
                val_n += yb.size(0)
        val_loss = val_loss_sum / val_n

        improved = val_loss < best_val_loss - 1e-5
        log(f"    [{model_name}] epoch {epoch:2d}/{MAX_EPOCHS}: TRAIN loss={train_loss:.5f}  "
            f"VAL loss={val_loss:.5f}{'  (best so far)' if improved else ''}")
        loss_history.append({"Epoch": epoch, "Train_Loss": train_loss, "Validation_Loss": val_loss,
                              "Improved": improved})
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                log(f"    [{model_name}] early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_val_loss, pd.DataFrame(loss_history)


# =============================================================================
# THRESHOLD SELECTION (vectorized, exact -- every unique probability tested)
# =============================================================================

def select_threshold(y_true, y_prob):
    """Locked rule: (1) highest MCC, (2) highest F1, (3) lowest FPR,
    (4) threshold closest to 0.5, (5) higher threshold. Fixed before any
    result is observed; identical for all five models."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    total_pos = int(y_true.sum())
    total_neg = n - total_pos

    order = np.argsort(y_prob)  # ascending
    y_sorted = y_true[order]
    prob_sorted = y_prob[order]

    pos_suffix = np.cumsum(y_sorted[::-1])[::-1]        # positives at index >= i
    total_suffix = np.arange(n, 0, -1)                  # count of indices >= i
    tp_arr = pos_suffix
    fp_arr = total_suffix - pos_suffix
    fn_arr = total_pos - tp_arr
    tn_arr = total_neg - fp_arr

    unique_vals, first_idx = np.unique(prob_sorted, return_index=True)
    tp_u, fp_u, fn_u, tn_u = tp_arr[first_idx], fp_arr[first_idx], fn_arr[first_idx], tn_arr[first_idx]

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp_u + fp_u) > 0, tp_u / (tp_u + fp_u), 0.0)
        recall = np.where((tp_u + fn_u) > 0, tp_u / (tp_u + fn_u), 0.0)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
        fpr = np.where((fp_u + tn_u) > 0, fp_u / (fp_u + tn_u), 0.0)
        mcc_denom = np.sqrt((tp_u + fp_u).astype(float) * (tp_u + fn_u) * (tn_u + fp_u) * (tn_u + fn_u))
        mcc = np.where(mcc_denom > 0, (tp_u * tn_u - fp_u * fn_u) / mcc_denom, 0.0)

    sweep_df = pd.DataFrame({
        "Threshold": unique_vals, "TP": tp_u, "FP": fp_u, "FN": fn_u, "TN": tn_u,
        "Precision": precision, "Recall": recall, "F1": f1, "MCC": mcc, "FPR": fpr,
        "Dist_From_0.5": np.abs(unique_vals - 0.5),
    })
    sweep_sorted = sweep_df.sort_values(
        by=["MCC", "F1", "FPR", "Dist_From_0.5", "Threshold"],
        ascending=[False, False, True, True, False]).reset_index(drop=True)
    selected_threshold = float(sweep_sorted.iloc[0]["Threshold"])
    return selected_threshold, sweep_df, sweep_sorted


# =============================================================================
# METRICS + PERSISTENCE (shared across all five models)
# =============================================================================

def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred), "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0), "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "FPR": fp / (fp + tn) if (fp + tn) > 0 else float("nan"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def persist_model_artifacts(model_name, model, best_epoch, best_val_loss, loss_history_df,
                             y_true, y_prob, threshold, metrics, sweep_df, latency_ms_per_seq, config):
    prefix = ARTIFACTS_DIR / model_name
    torch.save(model.state_dict(), f"{prefix}_checkpoint.pt")
    loss_history_df.to_csv(f"{prefix}_loss_history.csv", index=False)
    pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).to_csv(f"{prefix}_validation_predictions.csv", index=False)
    sweep_df.to_csv(f"{prefix}_threshold_sweep.csv", index=False)
    meta = {
        "model_name": model_name, "best_epoch": best_epoch, "best_validation_bce_loss": best_val_loss,
        "selected_threshold": threshold, "latency_ms_per_sequence": latency_ms_per_seq,
        "metrics": metrics, "config": config, "generated": datetime.now().isoformat(),
    }
    with open(f"{prefix}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    log(f"    Persisted: {prefix}_checkpoint.pt, _loss_history.csv, _validation_predictions.csv, "
        f"_threshold_sweep.csv, _metadata.json")


def measure_latency_ms_per_sequence(model, loader, forward_fn, n_repeats=3):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t0 = time.time()
            n_seen = 0
            for batch in loader:
                _ = forward_fn(model, batch)
                n_seen += batch[-1].size(0)
            times.append((time.time() - t0) / n_seen * 1000.0)
    return float(np.mean(times))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                         help="Run the full five-model training experiment. Without this flag, "
                              "only preflight guardrail checks run and the script exits.")
    args = parser.parse_args()

    section("SCRIPT 23 — MAIN FIVE-MODEL EXPERIMENT")
    log(f"Run start: {RUN_START.isoformat()}")
    log(f"Mode: {'FULL TRAINING' if args.train else 'PREFLIGHT ONLY (no training will run)'}")
    log("TEST is never loaded or referenced anywhere in this script.")

    pf = run_preflight()
    if not pf["all_passed"]:
        section("PREFLIGHT FAILED — ABORTING BEFORE ANY TRAINING")
        sys.exit(1)

    if not args.train:
        section("PREFLIGHT PASSED — STOPPING HERE (pass --train to run the full experiment)")
        log("No model was trained. No TEST access. No artifacts other than this log were created.")
        return

    # ------------------------------------------------------------------
    # Everything below only executes with --train (not run in this session yet)
    # ------------------------------------------------------------------
    section("RUNNING THE FULL FIVE-MODEL EXPERIMENT")

    train_df, val_df = pf["train_df"], pf["val_df"]
    sym_train_df, sym_val_df = pf["sym_train_df"], pf["sym_val_df"]
    all_51, top25, core6, ext16 = pf["all_51_features"], pf["top25_features"], pf["core_features"], pf["extended_features"]

    y_train_full = train_df["Label"].to_numpy(dtype=np.float32)
    y_val_full = val_df["Label"].to_numpy(dtype=np.float32)
    train_seq_labels = y_train_full[WINDOW - 1:]
    n_pos, n_neg = float((train_seq_labels == 1).sum()), float((train_seq_labels == 0).sum())
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    log(f"pos_weight (TRAIN sequence labels only) = {pos_weight.item():.6f}")

    def raw_fwd(model, batch):
        xb, yb = batch
        return model(xb), yb

    def fusion_fwd(model, batch):
        xb, sb, yb = batch
        return model(xb, sb), yb

    specs = [
        {"name": "full_lstm", "mode": "raw", "raw_cols": all_51},
        {"name": "selected_lstm", "mode": "raw", "raw_cols": top25},
        {"name": "symbolic_only", "mode": "symbolic", "sym_cols": core6},
        {"name": "neurosymbolic_core", "mode": "fusion", "raw_cols": top25, "sym_cols": core6},
        {"name": "neurosymbolic_extended", "mode": "fusion", "raw_cols": top25, "sym_cols": ext16},
    ]

    comparison_rows = []
    for spec in specs:
        name, mode = spec["name"], spec["mode"]
        section(f"MODEL: {name} (mode={mode})")
        set_seed(SEED)

        raw_train_arr = train_df[spec["raw_cols"]].to_numpy(dtype=np.float32) if "raw_cols" in spec else None
        raw_val_arr = val_df[spec["raw_cols"]].to_numpy(dtype=np.float32) if "raw_cols" in spec else None
        sym_train_arr = sym_train_df[spec["sym_cols"]].to_numpy(dtype=np.float32) if "sym_cols" in spec else None
        sym_val_arr = sym_val_df[spec["sym_cols"]].to_numpy(dtype=np.float32) if "sym_cols" in spec else None

        train_ds = MainExperimentDataset(mode, WINDOW, y_train_full, raw_train_arr, sym_train_arr)
        val_ds = MainExperimentDataset(mode, WINDOW, y_val_full, raw_val_arr, sym_val_arr)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        if mode == "raw":
            model = LSTMOnlyModel(input_size=len(spec["raw_cols"]))
            fwd = raw_fwd
        elif mode == "symbolic":
            model = SymbolicOnlyModel(input_size=len(spec["sym_cols"]))
            fwd = raw_fwd
        else:
            model = FusionModel(raw_input_size=len(spec["raw_cols"]), symbolic_input_size=len(spec["sym_cols"]))
            fwd = fusion_fwd

        t0 = time.time()
        model, best_epoch, best_val_loss, loss_history_df = train_model(
            model, train_loader, val_loader, pos_weight, fwd, name)
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

        threshold, sweep_df, sweep_sorted = select_threshold(y_true, y_prob)
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        latency = measure_latency_ms_per_sequence(model, val_loader, fwd)

        config = {"hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS, "dropout": DROPOUT, "lr": LR,
                  "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "seed": SEED,
                  "window": WINDOW, "stride": STRIDE, "raw_cols": spec.get("raw_cols"), "sym_cols": spec.get("sym_cols")}
        persist_model_artifacts(name, model, best_epoch, best_val_loss, loss_history_df,
                                 y_true, y_prob, threshold, metrics, sweep_df, latency, config)

        log(f"  [{name}] best_epoch={best_epoch} VAL_BCE={best_val_loss:.5f} threshold={threshold:.6f} "
            f"train_time={train_elapsed:.1f}s latency={latency:.4f}ms/seq")
        log(f"  [{name}] Acc={metrics['Accuracy']:.4f} P={metrics['Precision']:.4f} R={metrics['Recall']:.4f} "
            f"F1={metrics['F1']:.4f} ROC_AUC={metrics['ROC_AUC']:.4f} MCC={metrics['MCC']:.4f} FPR={metrics['FPR']:.4f}")

        comparison_rows.append({"Model": name, "Best_Epoch": best_epoch, "Validation_BCE_Loss": best_val_loss,
                                 "Selected_Threshold": threshold, "Train_Seconds": round(train_elapsed, 1),
                                 "Latency_ms_per_seq": round(latency, 4), **metrics})

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = TABLES_DIR / "five_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    log(f"\nWrote {comparison_path}")

    section("FINAL GUARDRAIL RE-CHECK (post-training)")
    post_hashes = {p: sha256_of(p) for p in pf["pre_hashes"]}
    frozen_unchanged = all(pf["pre_hashes"][p] == post_hashes[p] for p in pf["pre_hashes"])
    all_log_text = "\n".join(_LOG_LINES)
    final_checks = {
        "frozen_inputs_unchanged": frozen_unchanged,
        "test_never_referenced_in_log": ("test_scaled" not in all_log_text and "test_unscaled" not in all_log_text),
    }
    for k, v in final_checks.items():
        log(f"  [{'PASS' if v else 'FAIL'}] {k}")

    run_end = datetime.now()
    log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
    log("FIVE-MODEL EXPERIMENT COMPLETE." if all(final_checks.values()) else "COMPLETED WITH GUARDRAIL FAILURES.")


if __name__ == "__main__":
    main()
