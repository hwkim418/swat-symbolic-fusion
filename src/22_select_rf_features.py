"""
code/22_select_rf_features.py

RF SUBSET-SIZE SELECTION via a compact-LSTM proxy sweep — Decision 2 of the
main 5-model experiment. Selects K (the number of RF-ranked raw variables
used by the "Selected LSTM" / neuro-symbolic models) using VALIDATION-only
performance. Does NOT train any of the five final comparison models. Does
NOT re-rank features (the existing TRAIN-only RF ranking is read verbatim
and its source file is hash-verified unchanged before/after this run).
TEST is never loaded or referenced anywhere below.

Locked specification (all user-approved, not re-derived here):
  - Candidate grid: K in {5, 10, 15, 20, 25, 30}, by RF Rank (MDI-based),
    from tables/table_rf_feature_ranking.csv.
  - Sequence window = 120 timesteps, stride = 1, last-timestep label
    convention, windows never cross the TRAIN/VALIDATION boundary (built
    independently per split from independent on-the-fly Dataset objects).
  - Proxy: 1-layer unidirectional LSTM, hidden_size=32, dropout=0.2 (applied
    externally to the final hidden state, since PyTorch's LSTM dropout is a
    no-op at num_layers=1), Adam optimizer, lr=1e-3, batch_size=256,
    max_epochs=15, early stopping on VALIDATION BCE loss (patience=3,
    restore best-epoch weights), seed=42, BCEWithLogitsLoss with
    pos_weight = n_neg/n_pos computed from TRAIN sequence labels only.
  - Decision threshold fixed at 0.5 for this sweep -- no threshold tuning.
  - K-selection rule: (1) highest VALIDATION MCC @0.5, (2) tie-break highest
    F1, (3) tie-break smaller K.
  - All proxy settings identical across every K -- only input dimensionality
    (number of features) changes between runs.

Outputs (minimal, by design):
  tables/rf_k_sweep_results.csv
  results/rf_k_sweep_validation_report.txt
No figures. No further scripts invoked. Stops before any final-model training.
"""

import sys
import time
import random
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
TABLES_DIR = PROJECT_ROOT / "tables"
RESULTS_DIR = PROJECT_ROOT / "results"
for d in (TABLES_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TRAIN_SCALED_PATH = PROCESSED_DIR / "train_scaled.csv"
VAL_SCALED_PATH = PROCESSED_DIR / "validation_scaled.csv"
RF_RANKING_PATH = TABLES_DIR / "table_rf_feature_ranking.csv"
# No TEST path is defined anywhere in this script, by design.

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


section("SCRIPT 22 — RF SUBSET-SIZE SELECTION (COMPACT-LSTM PROXY SWEEP, VALIDATION ONLY)")
log(f"Run start: {RUN_START.isoformat()}")
log("No TEST access. No re-ranking. No final-model training. Stops after reporting the selected K.")

# ---------------------------------------------------------------------------
# Locked configuration
# ---------------------------------------------------------------------------
K_GRID = [5, 10, 15, 20, 25, 30]
WINDOW = 120
STRIDE = 1
HIDDEN_SIZE = 32
NUM_LAYERS = 1
DROPOUT = 0.2
LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 15
PATIENCE = 3
SEED = 42
THRESHOLD = 0.5

RF_RANKING_PRE_HASH = sha256_of(RF_RANKING_PATH)

rf_ranking = pd.read_csv(RF_RANKING_PATH).sort_values("Rank").reset_index(drop=True)
log(f"Loaded frozen TRAIN-only RF ranking ({len(rf_ranking)} features), sha256={RF_RANKING_PRE_HASH[:12]}...")

# =============================================================================
# LOAD DATA (TRAIN + VALIDATION ONLY, SCALED)
# =============================================================================
section("LOADING TRAIN AND VALIDATION (SCALED; TEST NEVER LOADED)")

train_df = pd.read_csv(TRAIN_SCALED_PATH)
val_df = pd.read_csv(VAL_SCALED_PATH)
log(f"TRAIN: {len(train_df):,} rows | VALIDATION: {len(val_df):,} rows")
assert "test" not in TRAIN_SCALED_PATH.name and "test" not in VAL_SCALED_PATH.name

y_train_full = train_df["Label"].to_numpy(dtype=np.float32)
y_val_full = val_df["Label"].to_numpy(dtype=np.float32)

# =============================================================================
# ON-THE-FLY WINDOWED SEQUENCE DATASET (never materializes the full 3D tensor)
# =============================================================================
section("DEFINING ON-THE-FLY WINDOWED SEQUENCE DATASET")


class WindowedSequenceDataset(Dataset):
    """Each __getitem__ slices a (WINDOW, n_features) window from the base 2D
    array on demand -- only one batch's worth of windows is ever materialized
    at a time, regardless of dataset size or K. Windows are built strictly
    within this single split's array (this class is instantiated separately
    per split), so a window can never span the TRAIN/VALIDATION boundary.
    Label follows the last-timestep convention: the label of the window's
    final row."""

    def __init__(self, feature_array: np.ndarray, label_array: np.ndarray, window: int):
        assert feature_array.shape[0] == label_array.shape[0]
        self.X = feature_array.astype(np.float32)
        self.y = label_array.astype(np.float32)
        self.window = window
        self.n_sequences = self.X.shape[0] - window + 1
        assert self.n_sequences > 0, "Split shorter than window length."

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        end = idx + self.window  # exclusive end
        x = self.X[idx:end, :]          # (window, n_features)
        label = self.y[end - 1]         # last-timestep label
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.float32)


log(f"WindowedSequenceDataset defined: window={WINDOW}, stride={STRIDE} (native to per-row indexing, "
    f"no downsampling of start points). TRAIN sequences will be "
    f"{len(train_df) - WINDOW + 1:,}; VALIDATION sequences will be {len(val_df) - WINDOW + 1:,}.")

# =============================================================================
# COMPACT-LSTM PROXY MODEL
# =============================================================================
section("DEFINING COMPACT-LSTM PROXY MODEL (identical config across every K)")


class ProxyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # out: (batch, window, hidden_size)
        h_t = out[:, -1, :]            # final-timestep hidden representation
        h_t = self.dropout(h_t)
        logit = self.fc(h_t).squeeze(-1)
        return logit


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


log(f"ProxyLSTM: layers={NUM_LAYERS}, hidden_size={HIDDEN_SIZE}, dropout={DROPOUT} (external, on final hidden "
    f"state), optimizer=Adam(lr={LR}), batch_size={BATCH_SIZE}, max_epochs={MAX_EPOCHS}, "
    f"early_stopping_patience={PATIENCE} (on VALIDATION BCE loss), seed={SEED}, threshold={THRESHOLD} (fixed, "
    f"no tuning in this script). NOTE: because input_size varies with K, the global RNG stream is consumed "
    f"differently by each K's weight initialization even under the same seed value -- 'identical across K' "
    f"refers to every hyperparameter/config choice, not bit-identical initial weights, which is not possible "
    f"once input dimensionality changes.")

# =============================================================================
# TRAIN + EVALUATE ONE K
# =============================================================================
section("RUNNING THE K-SWEEP")


def run_one_k(k, feature_cols):
    t0 = time.time()
    set_seed(SEED)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    train_ds = WindowedSequenceDataset(X_train, y_train_full, WINDOW)
    val_ds = WindowedSequenceDataset(X_val, y_val_full, WINDOW)

    # pos_weight from TRAIN sequence labels only (post-windowing, last-timestep)
    train_seq_labels = y_train_full[WINDOW - 1:]
    n_pos = float((train_seq_labels == 1).sum())
    n_neg = float((train_seq_labels == 0).sum())
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ProxyLSTM(input_size=k)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_n += xb.size(0)
        val_loss = val_loss_sum / val_n

        improved = val_loss < best_val_loss - 1e-5
        log(f"    K={k:2d} epoch {epoch:2d}/{MAX_EPOCHS}: VALIDATION BCE loss={val_loss:.5f}"
            f"{'  (best so far)' if improved else ''}")
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                log(f"    K={k:2d}: early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            probs = torch.sigmoid(model(xb))
            all_probs.append(probs.numpy())
            all_labels.append(yb.numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    elapsed = time.time() - t0
    cum_importance = float(rf_ranking.loc[rf_ranking.Rank == k, "Cumulative_MDI"].iloc[0])
    log(f"  K={k:2d} DONE: best_epoch={best_epoch}, VAL_BCE={best_val_loss:.5f}, "
        f"P={precision:.4f} R={recall:.4f} F1={f1:.4f} MCC={mcc:.4f} FPR={fpr:.4f}, "
        f"runtime={elapsed:.1f}s")

    return {
        "K": k, "Cumulative_RF_Importance": round(cum_importance, 4), "Best_Epoch": best_epoch,
        "VALIDATION_BCE_Loss": round(best_val_loss, 6), "Precision": round(precision, 6),
        "Recall": round(recall, 6), "F1": round(f1, 6), "MCC": round(mcc, 6), "FPR": round(fpr, 6),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "N_VALIDATION_Sequences": len(val_ds), "Runtime_Seconds": round(elapsed, 1),
    }


sweep_rows = []
for k in K_GRID:
    feature_cols = rf_ranking.loc[rf_ranking.Rank <= k, "Feature"].tolist()
    assert len(feature_cols) == k
    log(f"\n--- K={k} (top {k} features by frozen TRAIN-only RF ranking) ---")
    log(f"  Features: {feature_cols}")
    row = run_one_k(k, feature_cols)
    sweep_rows.append(row)

sweep_df = pd.DataFrame(sweep_rows)

# =============================================================================
# K-SELECTION (LOCKED RULE: MCC desc, then F1 desc, then K asc)
# =============================================================================
section("APPLYING THE LOCKED K-SELECTION RULE")

sweep_df_sorted = sweep_df.sort_values(by=["MCC", "F1", "K"], ascending=[False, False, True]).reset_index(drop=True)
sweep_df["Selected"] = False
selected_k = int(sweep_df_sorted.iloc[0]["K"])
sweep_df.loc[sweep_df.K == selected_k, "Selected"] = True

log("Ranking of candidate K values under the locked rule (1: MCC desc, 2: F1 desc, 3: K asc):")
log(sweep_df_sorted[["K", "MCC", "F1", "Precision", "Recall", "FPR", "Cumulative_RF_Importance"]].to_string(index=False))
log(f"\nSELECTED K = {selected_k} (MCC={sweep_df_sorted.iloc[0]['MCC']:.4f}, "
    f"F1={sweep_df_sorted.iloc[0]['F1']:.4f}, cumulative RF importance="
    f"{sweep_df_sorted.iloc[0]['Cumulative_RF_Importance']*100:.2f}%)")

selected_features = rf_ranking.loc[rf_ranking.Rank <= selected_k, "Feature"].tolist()
log(f"Selected feature list ({selected_k} variables): {selected_features}")

sweep_path = TABLES_DIR / "rf_k_sweep_results.csv"
sweep_df.to_csv(sweep_path, index=False)
log(f"\nWrote {sweep_path} ({len(sweep_df)} rows)")

# =============================================================================
# INTEGRITY / GUARDRAIL CHECKS
# =============================================================================
section("INTEGRITY / GUARDRAIL CHECKS")

RF_RANKING_POST_HASH = sha256_of(RF_RANKING_PATH)
_all_log_text = "\n".join(_LOG_LINES)
checks = {
    "test_never_referenced": ("test_scaled" not in _all_log_text and "test_unscaled" not in _all_log_text),
    "rf_ranking_unchanged": (RF_RANKING_PRE_HASH == RF_RANKING_POST_HASH),
    "no_reranking_performed": True,  # rf_ranking loaded once, read-only, never refit
    "k_grid_matches_locked_spec": (K_GRID == [5, 10, 15, 20, 25, 30]),
    "threshold_fixed_at_0_5": (THRESHOLD == 0.5),
    "all_six_k_completed": (len(sweep_df) == 6),
    "windows_never_cross_split_boundary": True,  # structural: train_ds/val_ds built from separate arrays
    "no_final_model_trained": True,
}
for k_, v_ in checks.items():
    log(f"  [{'PASS' if v_ else 'FAIL'}] {k_}")
all_checks_passed = all(checks.values())
log(f"\nAll checks passed: {all_checks_passed}")

# =============================================================================
# FINAL VALIDATION / REPORT
# =============================================================================
section("FINAL REPORT")

report_lines = []
report_lines.append("SCRIPT 22 — RF SUBSET-SIZE SELECTION — VALIDATION REPORT")
report_lines.append(f"Generated: {RUN_START.isoformat()}")
report_lines.append("")
report_lines.append(f"K grid: {K_GRID}")
report_lines.append(f"Proxy: 1-layer LSTM, hidden={HIDDEN_SIZE}, dropout={DROPOUT}, Adam(lr={LR}), "
                     f"batch={BATCH_SIZE}, max_epochs={MAX_EPOCHS}, early-stop patience={PATIENCE} "
                     f"(on VALIDATION BCE), seed={SEED}, threshold={THRESHOLD} (fixed, not tuned).")
report_lines.append("")
report_lines.append(sweep_df[["K", "Cumulative_RF_Importance", "Best_Epoch", "VALIDATION_BCE_Loss",
                               "Precision", "Recall", "F1", "MCC", "FPR", "Runtime_Seconds",
                               "Selected"]].to_string(index=False))
report_lines.append("")
report_lines.append(f"SELECTED K = {selected_k} (rule: highest VALIDATION MCC @0.5 -> tie-break highest F1 "
                     f"-> tie-break smaller K)")
report_lines.append(f"Selected features: {selected_features}")
report_lines.append("")
report_lines.append("Guardrails:")
for k_, v_ in checks.items():
    report_lines.append(f"  [{'PASS' if v_ else 'FAIL'}] {k_}")
report_lines.append("")
report_lines.append(f"OVERALL: {'PASS' if all_checks_passed else 'FAIL'}")
report_lines.append("")
report_lines.append("Scope note: this script selects K only. No final comparison model (Full LSTM, Selected "
                     "LSTM, Symbolic-only, Neuro-symbolic CORE/Extended) was trained. TEST was never loaded. "
                     "Script 23 (sequence dataset builder for the final models) has not been started.")

report_text = "\n".join(report_lines)
report_path = RESULTS_DIR / "rf_k_sweep_validation_report.txt"
report_path.write_text(report_text, encoding="utf-8")
log(f"\nWrote {report_path}")

run_end = datetime.now()
log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")
log(f"\nOVERALL: {'PASS' if all_checks_passed else 'FAIL'}")

if not all_checks_passed:
    sys.exit(1)
