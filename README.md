# SWaT Symbolic Feature Fusion — Reproducibility Repository

Code and result artifacts accompanying the paper *"Explainable Cyberattack
Detection for Industrial Control Systems Using Random Forest, LSTM, and
Symbolic Feature Fusion"*, evaluated on the SWaT (Secure Water Treatment)
testbed dataset.

## Research objective

This project studies whether fusing a small set of **audited, physically
interpretable symbolic features** with an LSTM sequence encoder improves
cyberattack detection on the SWaT dataset relative to raw-sensor-only LSTM
baselines, and whether the audit that selects those symbolic features (27
candidate physical relationships → 16 retained → 6 "CORE") is doing more
than simply reducing feature count. Reported TEST performance is modest in
absolute terms (MCC 0.14–0.25 across the primary models); this repository
reproduces those numbers as reported, not a claim of production-ready
detection.

## License

The original source code, scripts, tables, figures, and documentation in
this repository are released under the **MIT License** (see `LICENSE`).
That license covers this repository's own authored content only.

**It does not apply to, and this repository claims no ownership or
licensing rights over, the SWaT dataset.** SWaT is a third-party dataset
distributed solely by its own provider under its own access terms — see
below.

## Required dataset acquisition (not included)

**Raw SWaT data is NOT distributed with this repository** and never will
be. SWaT is released by iTrust, Centre for Research in Cyber Security,
Singapore University of Technology and Design, under its own data-request
process (https://itrust.sutd.edu.sg/itrust-labs_datasets/). SWaT remains
subject exclusively to iTrust's own access terms and conditions; this
repository neither redistributes it nor grants any rights over it. **Every
user must obtain SWaT independently, directly from iTrust**, before any
script here can be run. Once obtained, place the two files here:

```
dataset/SWaT_Dataset_Normal_v0.csv
dataset/SWaT_Dataset_Attack_v0.csv
```

The directory name is **lowercase** (`dataset/`) and every script resolves
it relative to the repository root; override the location with the
`SWAT_DATASET_DIR` environment variable if you keep the files elsewhere.

## Pipeline overview

```
raw SWaT collection (449,919 obs, 51 physical variables)
  -> chronological TRAIN/VALIDATION/TEST split
  -> Random Forest feature ranking (500 trees, MDI importance)
  -> compact-LSTM proxy K-selection (K=25 selected)
  -> symbolic relationship audit, Stages 1-6 (27 candidates)
  -> cross-stage audit -> 16 retained (6 CORE + 10 retained-with-caveat), 2 LIMITED, 9 REJECT
  -> symbolic feature matrix construction (TRAIN/VALIDATION)
  -> five-model experiment (Full LSTM / Selected LSTM / Symbolic-only / NS CORE / NS Extended)
  -> repeated-seed VALIDATION robustness (5 seeds)
  -> symbolic-alignment (shuffled-CORE) ablation
  -> one-shot final TEST evaluation
  -> Statistical-Top-6 matched control (post-hoc, addresses a feature-count-vs-quality confound)
```

## Data split protocol

Chronological, non-overlapping, no attack interval crosses a partition
boundary:

| Split | Rows | Attack events | Role |
|---|---|---|---|
| TRAIN | 270,000 | 22 | Model fitting, RF ranking, symbolic reference construction (TRAIN-NORMAL only) |
| VALIDATION | 68,319 | 3 | All model selection, K selection, threshold selection, symbolic audit decisions |
| TEST | 111,600 | 10 | Evaluated exactly once, after every other decision was frozen |

**TEST is reserved for final evaluation only.** No script in this
repository re-opens TEST for tuning; `src/26_final_test_evaluation.py`'s
`--evaluate-test` flag is the sole TEST-touching entry point, gated behind
an explicit CLI flag, and was run once.

## Random Forest feature selection

`src/03_random_forest_feature_ranking.py` fits a single
`RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1,
class_weight="balanced")` on unscaled TRAIN data and ranks all 51 variables
by Mean Decrease in Impurity (MDI). `src/22_select_rf_features.py` then
sweeps candidate subset sizes K ∈ {5,10,15,20,25,30} using a **compact LSTM
proxy** (hidden size 32, max epochs 15, patience 3, fixed decision
threshold 0.5 — used only for this K-selection sweep, not the final model)
and selects **K=25** by the locked rule (highest VALIDATION MCC, tie-break
F1, tie-break smaller K).

## Symbolic relationship audit

27 candidate physical relationships across the plant's 6 process stages are
screened (`src/04_stage2_symbolic_rule_audit.py` through
`src/17_final_cross_stage_symbolic_audit.py`) against category-specific,
code-defined empirical criteria (state/valve separation, rolling-slope
mass-balance, sensor-sensor consistency, actuator-actuator coupling) —
never against TEST. This yields:

- **9 REJECT**, **2 LIMITED** (physically plausible but not usable at the
  required VALIDATION coverage/saturation bar)
- **16 retained**, of which **6 are CORE** (no cross-stage caveat) and 10
  are retained-with-caveat

CORE features:

| Feature | Relationship |
|---|---|
| S1-R1 | MV101 ↔ FIT101 |
| S1-R2 | P101 ↔ LIT101 |
| S1-R4 | FIT101 ↔ LIT101 |
| S2-R1 | MV201 ↔ FIT201 |
| S2-R3B | P205 ↔ P203 |
| S5-R3 | FIT501 ↔ FIT503 |

Full criteria, per-candidate evidence, and the exact reject/caveat
reasoning are in `tables/final_cross_stage_symbolic_audit.csv`,
`tables/manuscript_symbolic_rejection_rationale.csv`, and
`docs/final_paper_results_package.md`.

## Final models

All five models in `src/23_main_experiment.py` share one architecture and
training protocol: 1-layer LSTM, hidden size 64, dropout 0.2 (applied
externally to the final hidden state, not via PyTorch's inter-layer LSTM
dropout — a deliberate choice, since that mechanism is a no-op at
`num_layers=1`), Adam (lr=1e-3), batch size 256, max 30 epochs, early
stopping on VALIDATION BCE loss (patience 5), weighted BCE with
positive-class weight = n_negative/n_positive from TRAIN sequence labels.
Window length 120, stride 1, last-timestep label convention.

- **Full LSTM** — all 51 raw variables
- **Selected LSTM** — RF top-25 raw variables
- **Symbolic-only** — 6 CORE symbolic features, no LSTM (single seed only)
- **NS CORE** — Selected-LSTM 64-dim representation late-fused with the 6 CORE symbolic features (at the sequence-final timestep)
- **NS Extended** — same fusion with all 16 retained symbolic features

Decision thresholds are selected on VALIDATION only: every unique predicted
probability is swept, ranked by (1) highest MCC, (2) highest F1, (3) lowest
FPR, (4) closest to 0.5, (5) higher threshold — frozen before TEST is ever
opened (`src/23_main_experiment.py`'s `select_threshold`).

Full LSTM, Selected LSTM, NS CORE, and NS Extended are repeated across
**five seeds {7, 21, 42, 84, 123}** (`src/24_repeated_seed_validation.py`)
for VALIDATION robustness; Symbolic-only is single-seed only (documented
as such, never presented as a five-seed aggregate).

## Statistical-Top-6 matched control

`src/27_matched_control_statistical_top6.py` trains the identical
architecture/protocol/25-raw-variable configuration as NS CORE, across the
same 5 seeds, but with 6 symbolic features chosen by a deterministic,
TRAIN/VALIDATION-only univariate separation criterion instead of the
physical audit — isolating whether NS CORE's result is explained simply by
using six symbolic features rather than sixteen (it is not, per
`tables/core_vs_stat_top6_comparison.csv`; the difference is directional
but not statistically significant at n=5).

## Shuffled-symbolic diagnostic

`src/25_symbolic_alignment_ablation.py` retrains the CORE fusion model with
the CORE symbolic vectors deterministically permuted across
sequence-eligible rows (VALIDATION only) — a control for whether the
symbolic branch's contribution depends on correct temporal alignment to the
LSTM's final timestep, rather than just adding six extra numeric inputs.

## Execution order

```
00_dataset_audit.py
01_split_candidate_analysis.py
02_preprocess_data.py
03_random_forest_feature_ranking.py
04_rf_generalization_diagnostic.py
04_stage2_symbolic_rule_audit.py   (two independent 04-numbered scripts; see file docstrings)
05_stage2_transition_response_analysis.py
06_stage2_normal_operating_envelope.py
07_stage2_rule_specification_and_continuous_diagnostics.py
08_stage2_p205_p203_deactivation_coupling.py
09_stage2_symbolic_feature_construction_v1.py
10_swat_stage_attack_exposure_and_s2_justification.py
11_finalize_stage2_symbolic_feature_set.py
12_stage5_end_to_end_symbolic_pipeline.py
13_stage1_end_to_end_symbolic_pipeline.py
14_stage4_end_to_end_symbolic_pipeline.py
15_stage3_end_to_end_symbolic_pipeline.py
16_stage6_end_to_end_symbolic_pipeline.py
17_final_cross_stage_symbolic_audit.py
18_write_symbolic_branch_manuscript.py
19_symbolic_rule_funnel_figure.py
20_overall_pipeline_figure.py
21_build_symbolic_feature_matrices.py
22_select_rf_features.py
23_main_experiment.py --train
24_repeated_seed_validation.py --train
25_symbolic_alignment_ablation.py --train
26_final_test_evaluation.py --evaluate-test   (one-shot; do not rerun casually)
27_matched_control_statistical_top6.py --run
```

Every script defaults to a read-only preflight/guardrail check; training or
TEST access requires the explicit flag shown above.

## Expected outputs

Running the pipeline populates `tables/`, `figures/`, and `results/` with
the same file names already present in this repository (those files here
are the authors' own frozen run, provided so results can be inspected
without re-executing anything). `processed_metadata/` is the small
reference-metadata copy shipped with the repository; a fresh pipeline run
instead regenerates preprocessing outputs under `processed/metadata/`
(and the full `processed/` tree) locally — the two are equivalent in
content but not the same path. Model checkpoints (`*.pt`) are **not
included** — see "Model weights" below.

## Mapping from artifacts to paper results

| Paper item | Source table/figure |
|---|---|
| Data split statistics | `processed_metadata/split_statistics.csv` |
| RF ranking / K-sweep | `tables/table_rf_feature_ranking.csv`, `tables/rf_k_sweep_results.csv` |
| Symbolic audit funnel (27→16→6) | `tables/final_stage_symbolic_summary.csv`, `tables/final_cross_stage_symbolic_audit.csv`, `figures/symbolic_rule_funnel.png` |
| Overall architecture figure | `figures/overall_system_architecture.png` |
| Five-model VALIDATION comparison | `tables/five_model_comparison.csv`, `tables/repeated_seed_summary_stats.csv` |
| Final TEST results (all seeds) | `tables/final_test_results_long.csv`, `tables/final_test_summary_stats.csv`, `tables/final_test_paired_deltas.csv` |
| Statistical-Top-6 control | `tables/stat_top6_results_long.csv`, `tables/core_vs_stat_top6_comparison.csv` |
| Shuffled-symbolic ablation | `tables/symbolic_alignment_ablation_long.csv`, `tables/symbolic_alignment_ablation_summary.csv` |
| VAL→TEST generalization / attack-interval diagnostics | `diagnostics/` |
| Consolidated results narrative | `docs/final_paper_results_package.md` |

## Model weights

Trained checkpoints (`*.pt`) are intentionally **not published in this
version** of the repository. Whether to release trained weights is a
separate decision pending review against the SWaT data-use agreement,
since weights are derived from restricted data. All non-checkpoint
per-seed artifacts (metadata, predictions, loss histories) needed to verify
every reported number are included.

## Installation

```
pip install -r requirements.txt
```

Developed and run under Python 3.13.x.
