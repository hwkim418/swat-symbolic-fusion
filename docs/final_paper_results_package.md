# Final Paper Results Package

Compiled entirely from already-persisted artifacts. No new experiments, no retraining, no TEST re-access. Every number below was cross-checked directly against its source CSV/report at compilation time; any discrepancy found is called out explicitly rather than silently resolved (none were found).

---

## A. RF Feature-Selection Facts

**Source:** `tables/table_rf_feature_ranking.csv`, `tables/rf_k_sweep_results.csv`

- **Top-5 RF variables (ordered by rank):** PIT501, PIT503, FIT401, AIT201, FIT504
- **Top-25 RF variables (ordered by rank):** PIT501, PIT503, FIT401, AIT201, FIT504, FIT501, AIT402, PIT502, FIT503, AIT502, AIT501, LIT401, DPIT301, AIT202, LIT101, FIT502, UV401, P501, LIT301, FIT201, AIT503, AIT504, AIT203, P102, FIT301
- **Cumulative RF (MDI) importance at K=25:** 0.9459 (94.59%)

**Full K-sweep (VALIDATION, proxy-model diagnostic, fixed threshold=0.5):**

| K | Cumulative RF Importance | MCC | F1 | FPR | Selected |
|---|---|---|---|---|---|
| 5 | 0.3452 | 0.0000 | 0.0000 | 0.0000 | No |
| 10 | 0.5537 | 0.0000 | 0.0000 | 0.0000 | No |
| 15 | 0.7224 | −0.0399 | 0.0247 | 0.2489 | No |
| 20 | 0.8524 | −0.0493 | 0.0235 | 0.3019 | No |
| **25** | **0.9459** | **0.0641** | **0.0651** | **0.4979** | **Yes** |
| 30 | 0.9815 | 0.0431 | 0.0577 | 0.5675 | No |

**Selection statement for the paper:** K=25 was selected via the pre-specified, pre-registered rule — highest VALIDATION MCC, then F1, then smaller K — evaluated with a compact single-layer LSTM proxy at a fixed 0.5 threshold. **K=25 was never selected using, or informed by, TEST data.** The proxy's absolute VALIDATION performance at K=25 was weak in absolute terms (MCC=0.064, FPR≈50%) — it should be described only as the best-ranked candidate under the pre-registered rule, never as a strong detector in its own right.

---

## B. Symbolic Audit Facts

**Source:** `tables/manuscript_symbolic_candidate_summary.csv`, `tables/final_stage_symbolic_summary.csv`, `tables/final_core_symbolic_features.csv`

**Confirmed totals:** 27 candidate physical relationships → 16 retained at stage level → 6 CORE + 10 RETAIN_WITH_CAVEAT (0 EXCLUDE) at cross-stage audit. 2 candidates LIMITED, 9 REJECTED at the stage level. 27 = 16 + 2 + 9 ✓; 16 = 6 + 10 ✓.

**Six CORE feature names (exact, in file order):**
1. `S1-R1_state_score`
2. `S1-R2_slope_score`
3. `S1-R4_slope_score`
4. `S2_R1_score`
5. `S2_R3B_state_mismatch`
6. `S5-R3_residual_score`

**Stage-level summary:**

| Stage | Candidates | Retained | Limited | Rejected | CORE | Caveat |
|---|---|---|---|---|---|---|
| Stage 1 | 5 | 4 | 0 | 1 | 3 | 1 |
| Stage 2 | 4 | 2 | 2 | 0 | 2 | 0 |
| Stage 3 | 6 | 3 | 0 | 3 | 0 | 3 |
| Stage 4 | 6 | 1 | 0 | 5 | 0 | 1 |
| Stage 5 | 5 | 5 | 0 | 0 | 1 | 4 |
| Stage 6 | 1 | 1 | 0 | 0 | 0 | 1 |
| **Total** | **27** | **16** | **2** | **9** | **6** | **10** |

---

## C. Final TEST Primary Table (five-seed aggregate, seeds 7/21/42/84/123)

**Source:** `tables/final_test_summary_stats.csv`

| Model | Raw vars | Symbolic vars | MCC (mean±std) | F1 (mean±std) | ROC-AUC (mean±std) |
|---|---|---|---|---|---|
| Full LSTM | 51 | 0 | 0.135 ± 0.082 | 0.066 ± 0.043 | 0.621 ± 0.084 |
| Selected LSTM | 25 | 0 | 0.183 ± 0.101 | 0.137 ± 0.111 | 0.520 ± 0.049 |
| **Neuro-symbolic CORE** | 25 | 6 | **0.247 ± 0.063** | **0.161 ± 0.063** | 0.600 ± 0.052 |
| Neuro-symbolic Extended | 25 | 16 | 0.155 ± 0.123 | 0.084 ± 0.098 | 0.577 ± 0.041 |

CORE has both the highest mean MCC/F1 **and** the lowest cross-seed MCC standard deviation among the four — the most consistently-performing configuration on final TEST.

---

## D. Symbolic-only Result (single seed = 42 only — NOT a five-seed aggregate)

**Source:** `tables/final_test_results_long.csv`

| Model | Seed(s) | MCC | F1 | ROC-AUC |
|---|---|---|---|---|
| Symbolic-only | 42 only | 0.099 | 0.150 | 0.611 |

Reported separately, as a single-seed secondary baseline, exactly as required — not combined into Table C's five-seed statistics.

---

## E. Repeated-Seed VALIDATION Robustness Table (same four primary models)

**Source:** `tables/repeated_seed_summary_stats.csv`

| Model | MCC (mean±std) | F1 (mean±std) | ROC-AUC (mean±std) |
|---|---|---|---|
| Full LSTM | 0.821 ± 0.001 | 0.809 ± 0.001 | 0.816 ± 0.046 |
| Selected LSTM | 0.820 ± 0.002 | 0.808 ± 0.002 | 0.890 ± 0.063 |
| Neuro-symbolic CORE | 0.821 ± 0.002 | 0.809 ± 0.002 | 0.821 ± 0.082 |
| Neuro-symbolic Extended | 0.819 ± 0.005 | 0.807 ± 0.006 | 0.862 ± 0.067 |

**VALIDATION mean MCC → TEST mean MCC per model** (computed only from already-persisted summary values in this table and Table C — no new predictions):

| Model | VALIDATION mean MCC | TEST mean MCC | Gap (TEST − VALIDATION) |
|---|---|---|---|
| Full LSTM | 0.821 | 0.135 | **−0.686** |
| Selected LSTM | 0.820 | 0.183 | **−0.637** |
| Neuro-symbolic CORE | 0.821 | 0.247 | **−0.574** |
| Neuro-symbolic Extended | 0.819 | 0.155 | **−0.664** |

All four models show a severe (≈0.57–0.69 absolute MCC) VALIDATION-to-TEST drop. CORE has the smallest gap of the four, consistent with its stronger relative TEST showing in Table C.

---

## F. Symbolic-Alignment Ablation Table (five-seed VALIDATION)

**Source:** `tables/symbolic_alignment_ablation_summary.csv`

| Configuration | MCC (mean±std) | F1 (mean±std) | ROC-AUC (mean±std) |
|---|---|---|---|
| Selected LSTM | 0.8197 ± 0.0015 | 0.8079 ± 0.0022 | 0.8901 ± 0.0631 |
| LSTM + Shuffled CORE | 0.8206 ± 0.0018 | 0.8088 ± 0.0024 | 0.8455 ± 0.0798 |
| LSTM + aligned CORE | 0.8212 ± 0.0021 | 0.8093 ± 0.0025 | 0.8208 ± 0.0820 |
| LSTM + Extended | 0.8193 ± 0.0046 | 0.8073 ± 0.0055 | 0.8620 ± 0.0672 |

**Aligned CORE − Shuffled CORE, five paired seed-level deltas (source: `tables/symbolic_alignment_ablation_long.csv`):**

| Metric | Mean delta | Seeds where aligned CORE strictly higher |
|---|---|---|
| MCC | +0.0006 | 2 / 5 |
| F1 | +0.0005 | 2 / 5 |
| ROC-AUC | −0.0248 | 2 / 5 |

**Required interpretation, preserved exactly:** aligned CORE did **not** consistently outperform Shuffled CORE on VALIDATION (2/5 seeds on every metric — worse than a coin flip). Therefore the TEST-side finding that CORE has the strongest aggregate MCC/F1 among the four primary configurations (Table C) **must not** be presented as evidence that the audited symbolic temporal alignment specifically caused that improvement — this ablation shows the VALIDATION-side signal is indistinguishable from random symbolic input at the descriptive five-seed level, and no TEST-side shuffled-control replication was run (by design, per the final-evaluation protocol).

---

## G. Paired Final TEST Deltas

**Source:** `tables/final_test_paired_deltas.csv`

**Full → Selected:** MCC mean=+0.048 (median=−0.032, 2/5 higher) · F1 mean=+0.071 (median=−0.009, 2/5 higher) · ROC-AUC mean=−0.101 (median=−0.126, 1/5 higher)

**Selected → CORE:** MCC mean=+0.064 (median=+0.058, 3/5 higher) · F1 mean=+0.024 (median=+0.041, 3/5 higher) · **ROC-AUC mean=+0.080 (median=+0.074, 5/5 higher)**

**Selected → Extended:** MCC mean=−0.028 (1/5 higher) · F1 mean=−0.053 (**0/5 higher**) · ROC-AUC mean=+0.058 (4/5 higher)

**CORE → Extended:** MCC mean=−0.092 (2/5 higher) · F1 mean=−0.077 (2/5 higher) · ROC-AUC mean=−0.023 (2/5 higher)

**Minimum facts supporting the four target claims:**
- *CORE has the highest mean TEST MCC and F1* → directly read from Table C (MCC=0.247, F1=0.161, both the maximum across all four rows).
- *CORE improves TEST ROC-AUC over Selected in 5/5 seeds* → Selected→CORE ROC-AUC row above, N_Seeds_Second_Higher=5/5, all five individual seed deltas positive (+0.133, +0.069, +0.025, +0.101, +0.074).
- *Extended does not improve MCC/F1 over CORE* → CORE→Extended row above: mean MCC delta −0.092 (2/5 higher, i.e. worse in 3/5), mean F1 delta −0.077 (2/5 higher, worse in 3/5).
- *51→25 reduction does not cleanly preserve TEST performance* → Full→Selected row above: ROC-AUC mean=−0.101 with only 1/5 seeds favoring Selected; MCC/F1 means are positive only due to two large-swing seeds while medians are negative.

No significance claims are made anywhere in this section.

---

## H. Latency

**Source:** `tables/final_test_summary_stats.csv` (Metric="Latency_ms_per_seq") and `tables/final_test_results_long.csv` (Symbolic-only)

All values below are **TEST-side inference latency, mean ± std across the 5 model seeds** (matching Table C's aggregation), except Symbolic-only which is **seed-42 only** (its sole available seed) — these two aggregation levels are kept explicitly separate below and must not be blended into one column in the paper.

| Model | Latency (ms/sequence) | Aggregation |
|---|---|---|
| Full LSTM | 0.0790 ± 0.0031 | mean±std across 5 TEST seeds |
| Selected LSTM | 0.0641 ± 0.0011 | mean±std across 5 TEST seeds |
| Neuro-symbolic CORE | 0.0672 ± 0.0007 | mean±std across 5 TEST seeds |
| Neuro-symbolic Extended | 0.0685 ± 0.0010 | mean±std across 5 TEST seeds |
| Symbolic-only | 0.0063 | **seed 42 only**, no repeated-seed figure exists |

---

## I. Supported vs. Not-Supported Claims

**SUPPORTED (verified numerically above):**
- The symbolic audit compresses 27 candidate physical relationships to 16 retained and 6 CORE (Section B).
- Symbolic-only retains non-degenerate TEST detection signal (MCC=0.099, F1=0.150, ROC-AUC=0.611 — all clearly above a degenerate/zero baseline; Section D).
- CORE has the strongest aggregate held-out TEST MCC (0.247) and F1 (0.161) among the four primary configurations, and the lowest cross-seed MCC variance (Section C).
- CORE's TEST ROC-AUC exceeds Selected LSTM's in all five paired seeds (Section G).
- Extended provides no MCC/F1 advantage over CORE on TEST — the opposite direction dominates (Section G).
- A severe chronological generalization degradation occurs from VALIDATION to TEST, consistently across all four models (≈0.57–0.69 absolute MCC drop; Section E).

**NOT SUPPORTED — must not be claimed:**
- Production-ready detection performance for any model (best mean TEST MCC ≈0.25, far below VALIDATION's ≈0.82).
- Statistical significance for any comparison (no significance procedure was pre-registered; 5 seeds is descriptive only).
- Clean preservation of performance from 51→25 raw variables (Section G: ROC-AUC favors Full LSTM in 4/5 seeds).
- That correct symbolic temporal alignment caused the TEST-side CORE improvement (Section F: aligned CORE was indistinguishable from Shuffled CORE on VALIDATION, 2/5 seeds each metric).
- That Extended fusion improves over CORE (Section G: opposite direction on MCC/F1 in the majority of seeds).
- Any TEST-guided model-selection claim — K=25, the four model architectures, and all thresholds were fixed before TEST was ever opened (Sections A and the final-TEST-evaluation provenance manifest).

---

## J. Recommended Final Paper Figures/Tables (≈7-page budget)

**Table 1 — Symbolic audit summary** (Section B's stage-level table). Preferred over a separate dataset/protocol table since the 27→16→6/10 story is the more novel, paper-specific quantitative result; standard dataset/split facts (270,000/68,319/111,600 rows) can be stated in one sentence of prose instead of a dedicated table.

**Table 2 — Final TEST primary results** (Section C + Section D's one-line Symbolic-only addendum). This is the paper's central quantitative claim and must appear in full.

**Table 3 — Merged VALIDATION-robustness-and-ablation table**: combine Section E (VALIDATION MCC/F1/ROC-AUC per model) with Section F's aligned-vs-Shuffled-CORE MCC delta as a single additional row/column, rather than two separate tables — this communicates both the VALIDATION→TEST generalization gap and the alignment-ablation caveat in one compact table, preserving page budget.

**Figure 1 — Overall system architecture** (already exists: `figures/overall_system_architecture.png`).

**Figure 2 — Symbolic rule funnel** (already exists: `figures/symbolic_rule_funnel.png`, the 27→16→6/10 retention diagram) — pairs directly with Table 1, reinforcing the same numbers visually rather than duplicating a new figure for numbers the table already states clearly.

**Optional Figure 3 — recommend omitting.** A third figure (e.g., a VALIDATION-vs-TEST MCC bar chart) would visualize Section E's gap table, but that table alone communicates the same six numbers (four models × VALIDATION/TEST MCC) more compactly than a figure would within a 7-page budget. Include only if reviewer feedback specifically requests a visual generalization-gap illustration.
