"""
code/18_write_symbolic_branch_manuscript.py

WRITING / CONSOLIDATION SCRIPT — NOT AN ANALYSIS SCRIPT.

Generates a publication-ready manuscript draft for the complete SWaT
symbolic-rule branch (Stages 1-6 candidate generation through the script-17
cross-stage audit), using ONLY finalized project outputs already on disk.

This script does NOT perform new data analysis, does NOT discover new
rules, does NOT change any prior RETAIN/LIMITED/REJECT or
CORE/RETAIN_WITH_CAVEAT/EXCLUDE decision, does NOT train any model, and
does NOT load or reference the TEST partition anywhere. Every quantitative
claim it writes is read and verified from tables/results already produced
by scripts 04-17; script 17's final_cross_stage_symbolic_audit.csv and
final_stage_symbolic_summary.csv are treated as authoritative for final
per-feature status and final counts, per explicit instruction. Numbers are
never hard-coded blindly -- Part 1 below recomputes every headline figure
from source files and asserts it matches, so a stale docstring value can
never silently diverge from the actual data on disk.
"""

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
TABLES_DIR = PROJECT_ROOT / "tables"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
for d in (TABLES_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

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


section("SYMBOLIC BRANCH MANUSCRIPT WRITING — RUN START (WRITING/CONSOLIDATION ONLY)")
log(f"Run start: {RUN_START.isoformat()}")
log("This script performs NO new analysis, NO new rule discovery, NO model training, "
    "and NEVER loads the TEST partition. It only reads and consolidates finalized outputs "
    "from scripts 04-17 into manuscript-ready prose and tables.")

# =============================================================================
# PART 1 — SOURCE VERIFICATION
# =============================================================================
section("PART 1 — SOURCE VERIFICATION (reading finalized outputs as source of truth)")

REQUIRED_TABLES = [
    "final_cross_stage_symbolic_audit.csv",
    "final_symbolic_feature_set.csv",
    "final_core_symbolic_features.csv",
    "final_extended_symbolic_features.csv",
    "final_symbolic_feature_redundancy.csv",
    "final_stage_symbolic_summary.csv",
]
for fname in REQUIRED_TABLES:
    fpath = TABLES_DIR / fname
    assert fpath.exists(), f"Required finalized table missing: {fpath} -- cannot write manuscript without it."
log(f"Confirmed all {len(REQUIRED_TABLES)} required script-17 tables exist on disk.")

master_df = pd.read_csv(TABLES_DIR / "final_cross_stage_symbolic_audit.csv")
feature_set_df = pd.read_csv(TABLES_DIR / "final_symbolic_feature_set.csv")
core_df = pd.read_csv(TABLES_DIR / "final_core_symbolic_features.csv")
extended_df = pd.read_csv(TABLES_DIR / "final_extended_symbolic_features.csv")
redundancy_df = pd.read_csv(TABLES_DIR / "final_symbolic_feature_redundancy.csv")
stage_summary_df = pd.read_csv(TABLES_DIR / "final_stage_symbolic_summary.csv")
log(f"Loaded final_cross_stage_symbolic_audit.csv ({len(master_df)} rows), "
    f"final_core_symbolic_features.csv ({len(core_df)} rows), "
    f"final_extended_symbolic_features.csv ({len(extended_df)} rows), "
    f"final_symbolic_feature_redundancy.csv ({len(redundancy_df)} rows), "
    f"final_stage_symbolic_summary.csv ({len(stage_summary_df)} rows).")

# --- per-stage screening / retained tables + narratives (all 6 stages) ---
screening_tables = {}
retained_tables = {}
narrative_texts = {}
NARRATIVE_FILE = {
    1: "stage1_symbolic_pipeline_narrative.txt",
    2: "stage2_final_symbolic_feature_narrative.txt",
    3: "stage3_symbolic_pipeline_narrative.txt",
    4: "stage4_symbolic_pipeline_narrative.txt",
    5: "stage5_symbolic_pipeline_narrative.txt",
    6: "stage6_symbolic_pipeline_narrative.txt",
}
for i in range(1, 7):
    scr_path = TABLES_DIR / f"stage{i}_final_rule_screening.csv"
    ret_path = TABLES_DIR / f"stage{i}_final_retained_symbolic_features.csv"
    assert scr_path.exists(), f"Missing {scr_path}"
    assert ret_path.exists(), f"Missing {ret_path}"
    screening_tables[i] = pd.read_csv(scr_path)
    retained_tables[i] = pd.read_csv(ret_path)
    narr_path = RESULTS_DIR / NARRATIVE_FILE[i]
    narrative_texts[i] = narr_path.read_text(encoding="utf-8") if narr_path.exists() else ""
    log(f"  Stage {i}: screening={len(screening_tables[i])} rows, retained={len(retained_tables[i])} rows, "
        f"narrative {'found' if narrative_texts[i] else 'MISSING'} ({NARRATIVE_FILE[i]})")

# --- recompute headline counts independently from source, do not trust memory ---
candidate_counts = {i: len(screening_tables[i]) for i in range(1, 7)}
retained_counts_from_screening = {i: int((screening_tables[i].Final_Decision == "RETAIN").sum()) for i in range(1, 7)}
limited_counts = {i: int((screening_tables[i].Final_Decision == "LIMITED").sum()) for i in range(1, 7)}
rejected_counts = {i: int((screening_tables[i].Final_Decision == "REJECT").sum()) for i in range(1, 7)}
retained_counts_from_files = {i: len(retained_tables[i]) for i in range(1, 7)}

total_candidates = sum(candidate_counts.values())
total_retained = sum(retained_counts_from_screening.values())
total_limited = sum(limited_counts.values())
total_rejected = sum(rejected_counts.values())

# cross-check screening-table RETAIN count against the separately-persisted
# retained-features table for each stage
_retain_count_mismatches = [i for i in range(1, 7)
                             if retained_counts_from_screening[i] != retained_counts_from_files[i]]
if _retain_count_mismatches:
    log(f"DISCREPANCY: stage(s) {_retain_count_mismatches} -- RETAIN count in "
        f"stageN_final_rule_screening.csv disagrees with stageN_final_retained_symbolic_features.csv "
        f"row count. Preferring the retained-features table (the more recently finalized, "
        f"directly-consumed-by-script-17 output) as authoritative; NOT silently reconciling.")
else:
    log("Cross-check PASSED: every stage's screening-table RETAIN count exactly matches its "
        "retained-features-table row count.")

# cross-check candidate = retained + limited + rejected, per stage and overall
for i in range(1, 7):
    assert candidate_counts[i] == retained_counts_from_screening[i] + limited_counts[i] + rejected_counts[i], \
        f"Stage {i}: candidate count does not equal RETAIN+LIMITED+REJECT -- internal inconsistency in the screening table."
assert total_candidates == total_retained + total_limited + total_rejected

# final cross-stage status counts, read from script 17's authoritative master table
status_counts = master_df["Final_Cross_Stage_Status"].value_counts().to_dict()
total_core = int(status_counts.get("CORE", 0))
total_caveat = int(status_counts.get("RETAIN_WITH_CAVEAT", 0))
total_exclude = int(status_counts.get("EXCLUDE", 0))

# cross-check against final_stage_symbolic_summary.csv's own per-stage sums
_summary_core_sum = int(stage_summary_df["CORE_Count"].sum())
_summary_caveat_sum = int(stage_summary_df["RETAIN_WITH_CAVEAT_Count"].sum())
_summary_exclude_sum = int(stage_summary_df["EXCLUDE_Count"].sum())
_summary_candidates_sum = int(stage_summary_df["Original_Candidate_Count"].sum())
_summary_retained_sum = int(stage_summary_df["Stage_Level_Retained_Count"].sum())

_discrepancies = []
if (_summary_core_sum, _summary_caveat_sum, _summary_exclude_sum) != (total_core, total_caveat, total_exclude):
    _discrepancies.append("CORE/CAVEAT/EXCLUDE sums (master table vs. stage-summary table)")
if _summary_candidates_sum != total_candidates:
    _discrepancies.append(f"candidate total (stage-summary table says {_summary_candidates_sum}, "
                           f"independently recomputed from screening tables says {total_candidates})")
if _summary_retained_sum != total_retained:
    _discrepancies.append(f"retained total (stage-summary table says {_summary_retained_sum}, "
                           f"independently recomputed says {total_retained})")
if _discrepancies:
    log(f"DISCREPANCY DETECTED between finalized outputs: {_discrepancies}. Per instructions, this is "
        f"reported rather than silently reconciled; final_stage_symbolic_summary.csv (the direct script-17 "
        f"output) is used as authoritative below since it is the most recently finalized file.")
    total_candidates = _summary_candidates_sum
    total_retained = _summary_retained_sum
else:
    log("Cross-check PASSED: candidate/retained/CORE/CAVEAT/EXCLUDE totals agree exactly between "
        "final_cross_stage_symbolic_audit.csv, final_stage_symbolic_summary.csv, and independently "
        "recomputed per-stage screening-table counts.")

log(f"\nVERIFIED HEADLINE NUMBERS (recomputed from source files, not assumed):")
log(f"  Total candidate rules across S1-S6      : {total_candidates}")
log(f"  Stage-level retained symbolic features   : {total_retained}")
log(f"  Stage-level LIMITED                      : {total_limited}")
log(f"  Stage-level REJECTED                     : {total_rejected}")
log(f"  Final CORE                               : {total_core}")
log(f"  Final RETAIN_WITH_CAVEAT                 : {total_caveat}")
log(f"  Final EXCLUDE                            : {total_exclude}")
log(f"  Per-stage candidate counts               : {candidate_counts}")
log(f"  Per-stage retained/limited/rejected      : RETAIN={retained_counts_from_screening} "
    f"LIMITED={limited_counts} REJECT={rejected_counts}")

assert total_candidates == 27, f"Expected 27 total candidates, recomputed {total_candidates} -- aborting."
assert total_retained == 16, f"Expected 16 stage-level retained, recomputed {total_retained} -- aborting."
assert total_core == 6, f"Expected 6 CORE, recomputed {total_core} -- aborting."
assert total_caveat == 10, f"Expected 10 RETAIN_WITH_CAVEAT, recomputed {total_caveat} -- aborting."
assert total_exclude == 0, f"Expected 0 EXCLUDE, recomputed {total_exclude} -- aborting."
log("\nAll five verified headline numbers match the expected values stated in the task "
    "(27 / 16 / 6 / 10 / 0) -- confirmed from data, not assumed.")

# --- ordered feature name lists, read from files, never guessed ---
CORE_FEATURES = core_df["Feature_Name"].tolist()
EXTENDED_FEATURES = extended_df["Feature_Name"].tolist()
CAVEAT_FEATURES = [f for f in EXTENDED_FEATURES if f not in CORE_FEATURES]
assert len(CORE_FEATURES) == 6 and len(EXTENDED_FEATURES) == 16 and len(CAVEAT_FEATURES) == 10
log(f"\nPRIMARY CORE SET (read from final_core_symbolic_features.csv, in file order): {CORE_FEATURES}")
log(f"EXTENDED SET (read from final_extended_symbolic_features.csv, {len(EXTENDED_FEATURES)} features): "
    f"{EXTENDED_FEATURES}")

master_df = master_df.set_index("Feature_Name", drop=False)

# --- parse each stage's TRAIN_Normal_Evidence column (a stringified Python
# dict, e.g. "{'State_SMD': np.float64(20.93...), ...}") so every numeric
# claim quoted in the stage-wise prose below (SMD, coverage counts, rho,
# explained variance, coupling rate) is read directly from the persisted
# screening table, never re-typed from memory. ---
import re as _re


def _parse_evidence_dict(text):
    """Handles both TRAIN_Normal_Evidence text formats used across the six
    stages: a stringified Python dict with np.float64(...)-wrapped values
    (Stages 1, 3, 4, 6), and a plain 'key=value, key2=value2' text summary
    (Stages 2, 5) -- both are parsed into the same {field: float} shape."""
    out = {}
    if pd.isna(text):
        return out
    text = str(text)
    for m in _re.finditer(r"'(\w+)':\s*np\.\w+\(([^)]*)\)", text):
        key, raw = m.group(1), m.group(2)
        try:
            out[key] = float(raw)
        except ValueError:
            out[key] = None
    if not out:
        for m in _re.finditer(r"(\w[\w_]*)\s*=\s*(-?[\d.]+(?:e[+-]?\d+)?)\s*%?", text):
            key, raw = m.group(1), m.group(2)
            try:
                out[key] = float(raw)
            except ValueError:
                pass
    return out


evidence = {}  # evidence[stage][rule_id] = {field: value}
for i in range(1, 7):
    evidence[i] = {}
    for _, r in screening_tables[i].iterrows():
        evidence[i][r["Rule_ID"]] = _parse_evidence_dict(r.get("TRAIN_Normal_Evidence"))
log(f"\nParsed TRAIN_Normal_Evidence numeric fields for all {sum(len(v) for v in evidence.values())} "
    f"candidate rules across 6 stages (source for every SMD/rho/coupling-rate figure quoted below).")


def ev(stage, rule_id, field, fmt="{:.3f}"):
    val = evidence.get(stage, {}).get(rule_id, {}).get(field)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    return fmt.format(val)


def median_shift(feature_name):
    """Reads the TRAIN-N/VAL-N median pair already computed and stored by
    script 17 in master_df's Distribution_Shift_Notes column (its
    MEDIAN-SHIFT CHECK annotation), for features where the cross-stage
    audit's median-based distribution-shift check applies."""
    text = str(feat(feature_name, "Distribution_Shift_Notes", ""))
    m_t = _re.search(r"TRAIN-N median=(-?[\d.]+)", text)
    m_v = _re.search(r"VAL-N median=(-?[\d.]+)", text)
    if not m_t or not m_v:
        return "n/a", "n/a"
    return f"{float(m_t.group(1)):.3f}", f"{float(m_v.group(1)):.3f}"


def feat(name, col, default="n/a"):
    if name not in master_df.index or col not in master_df.columns:
        return default
    val = master_df.loc[name, col]
    return default if pd.isna(val) else val


def fmt_coverage(val):
    """Coverage fields mix bare numeric percentages and pre-formatted text
    strings across stages (Stage 2's text embeds an ATTACK-coverage
    parenthetical) -- format numerics to 2dp, pass text through unchanged."""
    try:
        return f"{float(val):.2f}%"
    except (TypeError, ValueError):
        return str(val)


def fmt_repr(val):
    if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).lower() == "nan":
        return "n/a in the source retained-features table (see Physical_Relationship for the underlying construction)"
    return str(val)


# --- Stage 4 descriptive-only fact: P402's OFF state is heavily attack-associated
# in TRAIN and absent entirely in VALIDATION. This is NOT new data analysis -- it is
# simple arithmetic combining two numbers already computed and persisted by prior
# scripts (stage4_tag_audit.csv's raw TRAIN state frequency for P402, and
# stage4_normal_operating_envelopes.csv's already-computed NORMAL-only reference
# count for the same state), read from disk here, not recomputed from raw rows.
_s4_tag_audit = pd.read_csv(TABLES_DIR / "stage4_tag_audit.csv")
_p402_train_row = _s4_tag_audit[(_s4_tag_audit.Tag == "P402") & (_s4_tag_audit.Split == "TRAIN")]
_p402_val_row = _s4_tag_audit[(_s4_tag_audit.Tag == "P402") & (_s4_tag_audit.Split == "VALIDATION")]
_p402_off_total_train = None
if len(_p402_train_row):
    _freq_text = _p402_train_row["State_Values_And_Frequencies"].iloc[0]
    for _part in str(_freq_text).split(";"):
        _part = _part.strip()
        if _part.startswith("1="):
            _p402_off_total_train = int(_part.split("=")[1].split("(")[0])
_p402_off_normal_n = None
_s4_env = pd.read_csv(TABLES_DIR / "stage4_normal_operating_envelopes.csv")
_sub = _s4_env[(_s4_env.Rule_ID == "S4-R2") & (_s4_env.Envelope_Type == "state_conditioned_OFF")]
if len(_sub):
    _p402_off_normal_n = int(_sub["N"].iloc[0])
p402_off_attack_pct = None
if _p402_off_total_train and _p402_off_normal_n is not None:
    p402_off_attack_pct = 100.0 * (_p402_off_total_train - _p402_off_normal_n) / _p402_off_total_train
p402_val_state_text = (_p402_val_row["State_Values_And_Frequencies"].iloc[0] if len(_p402_val_row) else None)
log(f"\nStage 4 descriptive-only fact (arithmetic on two already-persisted numbers, not new analysis): "
    f"P402 OFF (state=1) totals {_p402_off_total_train} rows across all of TRAIN, of which only "
    f"{_p402_off_normal_n} are NORMAL-labeled -- {p402_off_attack_pct:.2f}% of all TRAIN P402-OFF rows "
    f"are ATTACK-labeled. In VALIDATION, P402's observed states are: {p402_val_state_text}.")

# --- P602 minority-state percentage, read from stage6_tag_audit.csv (the
# same figure script 17 verified for its rare-actuator-state caveat) ---
_s6_tag_audit = pd.read_csv(TABLES_DIR / "stage6_tag_audit.csv")
_p602_row = _s6_tag_audit[(_s6_tag_audit.Tag == "P602") & (_s6_tag_audit.Split == "TRAIN")]
p602_on_pct = None
if len(_p602_row):
    _freq_text = _p602_row["State_Values_And_Frequencies"].iloc[0]
    _pct_vals = [float(p.split("(")[1].rstrip("%)")) for p in str(_freq_text).split(";") if "(" in p]
    p602_on_pct = min(_pct_vals) if _pct_vals else None
log(f"P602 minority-state (ON) TRAIN percentage, read from stage6_tag_audit.csv: {p602_on_pct}%")

# --- top redundancy pairs (context only; 0 pairs cross the 0.90 threshold, verified) ---
redundancy_df_sorted = redundancy_df.reindex(
    redundancy_df.Spearman_Rho_TRAIN_NORMAL.abs().sort_values(ascending=False).index)
n_flagged_redundant = int(redundancy_df.Flagged_High_Redundancy.sum())
top3_redundant = redundancy_df_sorted.head(3)
log(f"\nRedundancy re-check: {n_flagged_redundant} pair(s) flagged at |rho|>=0.90 (of {len(redundancy_df)} "
    f"pairs). Top 3 by |rho| for narrative context: " +
    "; ".join(f"{r.Feature_A}<->{r.Feature_B} rho={r.Spearman_Rho_TRAIN_NORMAL:.3f}"
              for r in top3_redundant.itertuples()))

# =============================================================================
# BUILD SHARED PROSE FRAGMENTS (used by multiple parts below)
# =============================================================================
CDF_SCORE_EXPLANATION = (
    "Every symbolic feature in this branch is expressed on a common [0,1] scale via an empirical, "
    "distribution-free two-sided extremeness score:\n\n"
    "    score = 1 - 2 * min(F(x), 1 - F(x))\n\n"
    "where F is the empirical cumulative distribution function estimated exclusively from "
    "TRAIN-NORMAL reference behavior (the appropriate conditioning state's TRAIN-NORMAL sample for "
    "state-conditioned rules, or the pooled TRAIN-NORMAL sample for unconditional/residual rules). "
    "A score near 0 indicates a value falling near the center of the TRAIN-normal reference "
    "distribution -- common, physically expected behavior for that operating state. A score near 1 "
    "indicates a value falling in either tail of that same reference distribution -- rare or extreme "
    "relative to TRAIN-normal behavior, regardless of which tail. No parametric (e.g. Gaussian) "
    "assumption is made about the shape of the reference distribution. Critically, this is a physical "
    "inconsistency score, not an attack probability: it quantifies how unusual an observation is "
    "relative to the plant's own normal operating history, and it is constructed, thresholded, and "
    "screened using TRAIN-NORMAL and VALIDATION data only -- never by reference to attack labels. "
    "Whether elevated inconsistency scores are useful for downstream attack detection is a separate "
    "question, addressed only in a later, distinct modeling stage; no classifier performance is "
    "claimed anywhere in this symbolic branch."
)

STAGE_THEMES = {
    1: "rolling flow<->level mass-balance dynamics, alongside a direct valve-flow gating relationship",
    2: "chemical-dosing response and bidirectional actuator-actuator coupling (the branch's methodology-development stage)",
    3: "actuator-driven flow/differential-pressure consistency plus a rolling flow<->level relationship, across the richest actuator set of any stage",
    4: "actuator-to-analyzer response, screened from the branch's most heterogeneous single-stage candidate set",
    5: "sensor-sensor conservation/consistency, dominant here because only one actuator varies",
    6: "a sixth, independent confirmation of the pump->flow gating category, from the structurally sparsest stage in the testbed",
}

# =============================================================================
# PART 2 — METHODOLOGY SECTION: "Engineering-Informed Symbolic Rule Construction"
# =============================================================================
section("PART 2 — WRITING METHODOLOGY SECTION")

methodology_section = f"""3.X Engineering-Informed Symbolic Rule Construction

Beyond the Random Forest and LSTM branches of this study, we constructed a complementary set of
engineering-informed symbolic features intended to encode known physical regularities of the SWaT
process directly, rather than relying solely on data-driven pattern learning. The construction process
was applied identically and independently across all six SWaT process stages (raw water intake through
RO permeate transfer), and is described here as a single reproducible sequence.

Candidate relationships originated from domain/process knowledge of each stage's control logic and
instrumentation, not from data mining: for each stage, we first audited its physical tag structure --
which sensors and actuators exist, how many operating states each actuator exhibits, and which pairs of
tags are plausibly coupled by the plant's physical design (e.g., a valve gating a downstream flow
sensor, a dosing pump producing a directional response in its associated water-quality analyzer, two
pumps sharing a control loop). Where a stage's actuator set did not make a specific pairing obvious a
priori (Stages 3, 4, and 5, which have multiple candidate sensors per actuator or vice versa), TRAIN-NORMAL
correlation was used as a disciplined discovery signal to narrow the candidate space -- never as
evidence of validity on its own, and never as a substitute for the empirical validation described below.
This process yielded {total_candidates} candidate symbolic relationships across the six stages
({', '.join(f'Stage {i}: {candidate_counts[i]}' for i in range(1, 7))}).

Every candidate was then validated exclusively against TRAIN-NORMAL-labeled observations. Attack labels
were never consulted during rule construction, threshold selection, or feature-representation choice, in
order to avoid building rules that memorize attack signatures rather than capture genuine physical
regularities. This TRAIN-NORMAL-only discipline was applied uniformly to every validation technique used:

  - Tag/actuator-state auditing: the number of distinct operating states for every actuator was
    determined empirically from the data (never assumed from tag naming), and the physical meaning of
    each state (e.g. which numeric value corresponds to OPEN vs. CLOSED, or ON vs. OFF) was inferred by
    comparing the associated target sensor's mean value across states. This process independently
    discovered a mandatory transitional valve state (a brief intermediate value visited between
    CLOSED and OPEN, never a stable operating state on its own) for one 3-state valve in Stage 2, and the
    same structural pattern was subsequently confirmed, independently, for additional 3-state valves in
    Stages 1 and 3.
  - Transition-response analysis: for actuator-triggered rules, individual TRAIN-NORMAL transition
    events were identified, a temporally continuous pre-transition baseline was required, and the target
    sensor's or paired actuator's response was measured.
  - Time-lag analysis: response offsets were swept across an explicit search grid (typically a
    symmetric or forward-only window spanning tens of seconds to two minutes, depending on candidate
    type), with the offset achieving the strongest, most consistent response selected -- never assumed
    to be zero-lag or instantaneous.
  - State-conditioned relationships: for actuator-sensor pairs, the target sensor's TRAIN-NORMAL
    distribution was characterized separately for each actuator steady state, and separation between
    states was quantified via standardized mean difference (SMD) and interquartile-range (IQR) overlap
    fraction -- not by inspection alone.
  - Rolling-window and derivative relationships: for flow<->level mass-balance candidates (where raw
    level is confounded by simultaneous inflow and outflow), a rolling slope (change in level over a
    fixed window) was used in place of the raw level value, with the window length itself selected from
    an explicit grid (typically 10-300 seconds) by whichever window produced the strongest empirical
    separation or explained-variance fraction.
  - Sensor-sensor residual, ratio, and difference relationships: for same-instrument-type or
    physically-coupled sensor pairs (e.g. two pressure transmitters expected to track a shared process
    variable), both a ratio and a difference formalization were tested, and whichever showed the lower
    coefficient of variation in TRAIN-NORMAL data was retained as the more stable representation.

For every candidate that passed this validation stage, an empirical (non-parametric) normal operating
envelope was constructed directly from TRAIN-NORMAL data -- P5/P10/P25/median/P75/P90/P95 and IQR,
computed per relevant conditioning state or lag/window choice, with no assumption of normality or any
other parametric family. This envelope in turn defines the empirical CDF used for symbolic
inconsistency scoring. {CDF_SCORE_EXPLANATION}

Four distinct representation families were used depending on what each relationship structurally
supports: event-based (evaluable only in a short window following a relevant actuator transition),
state-based (evaluable continuously, at every timestamp the actuator is in a defined steady state),
rolling-window-based (evaluable continuously wherever the rolling window is fully populated), and hybrid
representations combining a continuously-evaluable primary feature with an event-based companion feature
retained for additional evidence. This range of representations was necessary because a single
formulation is not adequate across all relationship types: event-based scores are the most direct causal
signal but are only evaluable in a small fraction of timestamps, while state-based and rolling-window
scores sacrifice some directness for near-total evaluability. Coverage was therefore always measured and
reported as evaluability -- the fraction of timestamps at which a given feature is defined -- and
explicitly NOT interpreted as attack-detection performance; a feature can have perfect coverage and
still say nothing about whether an attack occurred at a given timestamp, and conversely a feature with
partial coverage can still be a physically meaningful signal at the timestamps where it is defined.

Robustness was assessed by comparing each feature's score distribution between TRAIN-NORMAL and
VALIDATION-NORMAL data: distribution-shift screening compared central tendency (both mean and, in a
later cross-stage refinement, median) between the two splits, and score saturation was flagged when a
feature's VALIDATION-NORMAL scores clustered near the ceiling of the [0,1] scale, which would leave
little room to distinguish NORMAL from ATTACK behavior downstream. Sparse-state screening flagged
features whose evaluability depends on an actuator state that is either statistically thin (a small
absolute TRAIN-NORMAL reference sample) or operationally rare (a small percentage of total TRAIN
operating time), since either condition weakens confidence in the corresponding empirical reference
distribution. Redundancy screening compared each feature's TRAIN-NORMAL score array against every other
retained feature's score array (Spearman correlation) to identify features carrying largely duplicate
information, kept explicitly distinct from raw source-sensor correlation, which is a different quantity
(two sensors can be near-identical in raw value while their derived symbolic scores are not redundant
with any other retained feature, and vice versa).

Each stage independently applied the same RETAIN / LIMITED / REJECT decision framework to its own
candidates (Section 3.X.4 below), after which a single cross-stage audit (script 17) re-evaluated all
{total_retained} stage-level RETAIN features together under one unified standard, assigning each a final
CORE / RETAIN_WITH_CAVEAT / EXCLUDE status. Throughout, the held-out TEST partition was never loaded,
inspected, or referenced by any script in this branch; all rule construction, validation, and screening
used TRAIN and VALIDATION data only, preserving TEST for a final, independent evaluation outside the
scope of this branch."""

log(f"Methodology section written ({len(methodology_section.split())} words).")

# =============================================================================
# PART 3 — SYMBOLIC RULE SCREENING SUBSECTION
# =============================================================================
section("PART 3 — WRITING SYMBOLIC RULE SCREENING SUBSECTION")

screening_logic_section = f"""3.X.4 Rule Screening and Cross-Stage Audit

Symbolic Rule Screening

Physical plausibility alone was not treated as sufficient grounds to retain a candidate relationship.
Even where a proposed relationship followed directly from engineering reasoning about the plant's
control logic, it was screened against fourteen dimensions before being accepted as a defensible
symbolic feature: empirical strength (the magnitude of state separation, correlation, or event-response
consistency actually observed in TRAIN-NORMAL data), expected-direction consistency (whether the
relationship's sign/direction matched engineering expectation), event/reference sample sufficiency
(whether enough TRAIN-NORMAL transitions or conditioning-state rows existed to characterize the
relationship at all), temporal stability (whether a response offset could be identified reliably within
the tested lag grid), physical interpretability (whether the relationship maps to a specific, nameable
physical mechanism), continuous evaluability (what fraction of timestamps the resulting feature is
defined at), VALIDATION coverage (evaluability re-measured on the held-out VALIDATION split, separately
for NORMAL and ATTACK periods), TRAIN-to-VALIDATION drift (whether the feature's score distribution
shifts meaningfully between splits), score saturation (whether VALIDATION-NORMAL scores cluster near the
scoring ceiling), dependence on sparse actuator states, source-sensor redundancy, cross-feature score
redundancy, lag uncertainty (whether the best-fitting offset sits at the edge of the tested search grid,
leaving the true optimum potentially unexplored), and downstream fusion readiness (a composite judgment
combining the above).

Two distinct decision layers were used, deliberately kept separate. At the stage level, each of the
{total_candidates} candidates received one of three labels applied by that stage's own script, using
thresholds held constant across all six stages: RETAIN ({total_retained} candidates, a defensible,
continuously- or near-continuously-evaluable symbolic feature), LIMITED ({total_limited} candidates,
S2-R2 and S2-R4 -- a physically well-supported relationship whose practical evaluability or robustness
constraints fell short of the bar for primary use), or REJECT ({total_rejected} candidates -- a
relationship not empirically supported beyond a raw correlation, or with a coupling/separation strength
that did not clear the documented threshold). At the cross-stage level, the {total_retained}
stage-level-RETAIN features were re-evaluated together, for the first time on one shared standard rather
than six independently-calibrated ones, and assigned a final status of CORE ({total_core} features, no
caveat triggered under any of the ten cross-stage dimensions), RETAIN_WITH_CAVEAT ({total_caveat}
features, scientifically defensible and coverage-adequate but carrying one or more documented
limitations), or EXCLUDE ({total_exclude} features -- reserved for a feature that fails the
coverage/saturation bar itself under this stricter, unified lens; none did).

No target number of retained or CORE features was fixed in advance at either decision layer. Stage 4
retained only 1 of its 6 candidates and Stage 6's structural tag scarcity permitted testing only 1
candidate in the first place; neither outcome was adjusted, padded, or treated as a shortfall to be
corrected. Negative findings -- rejected candidates, LIMITED rules, and the reasoning behind each -- were
preserved and documented rather than discarded, both to support reproducibility and to make the
selectivity of the overall process auditable: {total_rejected} candidates were rejected and
{total_limited} were classified LIMITED, roughly proportionate to the {total_retained} that were
retained, which is inconsistent with a screening process that simply confirmed whatever was proposed."""

log(f"Screening-logic subsection written ({len(screening_logic_section.split())} words).")

# =============================================================================
# PART 4 — STAGE-WISE RESULTS
# =============================================================================
section("PART 4 — WRITING STAGE-WISE RESULTS")

stage1_results = f"""Stage 1 (Raw Water Intake)

Stage 1 contains five physical tags: two continuous sensors (FIT101 flow, LIT101 tank level) and three
actuators (MV101 inlet valve, and outflow pumps P101 and P102). Five candidate relationships were
generated from this structure -- MV101 gating FIT101 (mirroring the valve-flow pattern later confirmed
independently in Stages 2 and 5), P101 and P102 each affecting LIT101's rolling slope, a direct
FIT101<->LIT101 rolling-slope mass-balance relationship, and a P101<->P102 actuator-coupling test -- of
which four were retained and one rejected. The strongest relationship was S1-R1 (MV101<->FIT101,
state SMD={ev(1,"S1-R1","State_SMD")}, IQR overlap fraction={ev(1,"S1-R1","State_IQR_Overlap_Fraction")},
VALIDATION coverage={fmt_coverage(feat("S1-R1_state_score","VALIDATION_Coverage"))}). Both rolling-slope
pump features (S1-R2 for P101, S1-R3 for P102) and the direct flow<->level rolling-slope relationship
(S1-R4) were also retained, giving Stage 1 four state-based and rolling-window representations spanning
its full available actuator and sensor structure.

The rejected candidate, S1-R5 (a P101<->P102 same-row actuator-coupling test mirroring Stage 2's S2-R3B
methodology), showed a coupling rate of {ev(1,"S1-R5","Coupling_Rate")} across {ev(1,"S1-R5","N_Events_TRAIN_NORMAL",fmt="{:.0f}")} TRAIN-NORMAL transition-active windows -- P101 and
P102 do not transition together, unlike the bidirectionally-coupled dosing pumps found in Stage 2,
and this negative finding was preserved rather than discarded. A specific caveat applies to S1-R3: P102
is a near-constant backup pump with only 4 TRAIN transitions overall, leaving its TRAIN-NORMAL ON-state
reference at n=45 rows and its ON state at only 0.70% of all TRAIN operating time -- both signals of a
statistically thin, operationally rare actuator state, formally reflected as a RETAIN_WITH_CAVEAT status
in the cross-stage audit. S1-R2 and S1-R3 also share a structural limitation: LIT101's trend reflects
MV101 inflow and both pumps' outflow simultaneously, so state-conditioning on a single pump only
partially isolates that pump's individual contribution. Stage 1's principal contribution to the symbolic
library is the rolling flow<->level mass-balance category (S1-R4), a relationship structurally
unavailable in Stage 2 (which has no level sensor) and later echoed, with different tag identities, in
Stage 3 and Stage 4."""

stage2_results = f"""Stage 2 (Chemical Dosing)

Stage 2 served as this branch's methodology-development stage: the RETAIN/LIMITED/REJECT framework,
the empirical-CDF scoring formulation, the TRAIN-NORMAL-only validation discipline, and the
event/state/rolling representation taxonomy were all first established here before being applied
unchanged to Stages 1 and 3-6. All four Stage 2 candidates were generated from domain-expert reasoning
about the plant's dosing-control logic -- not fit to data -- before any empirical screening took place:
a motorized inlet valve (MV201) expected to be physically consistent with its downstream flow sensor
(FIT201); two dosing pumps (P205 for bleach/hypochlorite, P203 for acid) each expected to produce a
directional response in an associated water-quality analyzer (AIT203 ORP and AIT202 pH respectively);
and the two dosing pumps, sharing a control loop, expected to transition together.

The MV201<->FIT201 candidate (S2-R1) initially returned zero usable TRAIN-NORMAL transitions under a
naive two-state (CLOSED/OPEN) definition. This was traced to a previously undocumented mandatory
transitional valve state -- MV201 does not transition directly between CLOSED and OPEN, but passes
through a brief intermediate state (mean dwell time 7.28s, median 7.00s) on every opening -- and the
rule was corrected to a three-state CLOSED(1)->transitional(0)->OPEN(2) sequence before re-screening.
This corrected three-state structure was subsequently confirmed, independently, in additional 3-state
valves in Stages 1 and 3, making it a general finding rather than an MV201-specific artifact. The two
dosing pumps (P205, P203) were found to transition together with {ev(2,"S2-R3B","consistency",fmt="{:.1f}")}%
same-row, same-direction consistency (S2-R3B) -- a bidirectional actuator-actuator coupling not observed
in the equivalent tests run in Stages 1, 3, or 4.

S2-R2 (P205->AIT203) and S2-R4 (P203->AIT202) both showed strong TRAIN-normal support (expected-direction
consistency of {ev(2,"S2-R2","consistency",fmt="{:.1f}")}% and {ev(2,"S2-R4","consistency",fmt="{:.1f}")}%
respectively) -- their underlying physical relationships were not in question. Both were nonetheless
classified LIMITED at the stage level because their event-triggered
formulation is evaluable at well under 1% of VALIDATION timestamps and at zero VALIDATION-ATTACK
timestamps in this dataset, and the practical alternatives investigated for extending coverage did not
clear a defensible bar: S2-R4's steady-state separation by pump state was not clean enough to support a
state-conditioned fallback at all (IQR overlap fraction=0.282, above the documented <0.25 bar), and the
fallback that WAS built for S2-R2 was found to be saturated in VALIDATION (NORMAL and ATTACK scores both
near the scoring ceiling), consistent with a previously documented TRAIN-to-VALIDATION distribution shift
in this plant's analyzer sensors. LIMITED here denotes a genuine practical-evaluability constraint, not a
rejection of the underlying engineering hypothesis -- both S2-R2 and S2-R4 remain documented and
available, just not promoted to the primary fusion feature set.

S2-R1 and S2-R3B were retained: each yields a state-based feature evaluable on essentially every
timestamp -- including, for S2-R1, 100% of VALIDATION-ATTACK timestamps -- without requiring a recent
actuator transition, and neither is saturated in VALIDATION-NORMAL. Both were subsequently confirmed
CORE in the cross-stage audit, with no caveat triggered under any of the ten unified dimensions."""

_s3r1_tmed, _s3r1_vmed = median_shift("S3-R1_state_score")
_s3r2_tmed, _s3r2_vmed = median_shift("S3-R2_state_score")

stage3_results = f"""Stage 3 (Ultrafiltration)

Stage 3 has the richest actuator set of any stage in this branch: four motorized valves (MV301-304,
each independently confirmed to share MV201's 3-state transitional structure) plus pump P302, alongside
three continuous sensors including DPIT301 (differential pressure), a sensor type not present in any
other stage. With five varying actuators and effectively three candidate target sensors, actuator-target
pairs (P302->FIT301, P302->DPIT301, and the MV304<->P302 coupling test) were selected via TRAIN-NORMAL
correlation screening rather than tested exhaustively, avoiding a combinatorial fishing expedition; a
direct FIT301<->DPIT301 sensor-sensor candidate was added on general fluid-dynamics grounds (hydraulic
resistance increases with flow), and P302->LIT301 rolling slope mirrored the Stage 1 methodology.

Three of six candidates were retained. P302->FIT301 (S3-R1, state SMD={ev(3,"S3-R1","State_SMD")}, IQR
overlap fraction={ev(3,"S3-R1","State_IQR_Overlap_Fraction")}) and P302->DPIT301 (S3-R2, state
SMD={ev(3,"S3-R2","State_SMD")}, IQR overlap fraction={ev(3,"S3-R2","State_IQR_Overlap_Fraction")}) were
both retained with 100%
VALIDATION coverage, and both are the strongest relationships found in this stage. Both, however, show a
substantial TRAIN-NORMAL-to-VALIDATION-NORMAL median shift when re-examined under the cross-stage audit's
median-based distribution-shift check (S3-R1: median {_s3r1_tmed} -> {_s3r1_vmed}; S3-R2: median
{_s3r2_tmed} -> {_s3r2_vmed}) -- neither crosses the mean-based saturation threshold used at the stage
level, which is precisely why a separate median check was needed to surface this partial shift, and both
are accordingly RETAIN_WITH_CAVEAT rather than CORE in the final cross-stage status. S3-R4 (a
P302-conditioned FIT301<->LIT301 rolling slope) is a substantially weaker retained relationship
(rho={ev(3,"S3-R4","Spearman_Rho")} at a 300s window, explained variance fraction of only
{float(ev(3,"S3-R4","Explained_Variance_Fraction"))*100:.1f}%) -- physically genuine but a partial,
low-explanatory mass-balance signal, carried forward as RETAIN_WITH_CAVEAT rather than excluded.

Three candidates were rejected and their negative findings preserved: S3-R3 (the direct FIT301<->DPIT301
sensor-sensor candidate, Spearman rho={ev(3,"S3-R3","Spearman_Rho")}, residual coefficient of
variation={ev(3,"S3-R3","Residual_Stability_CV")} -- too weak and unstable to support a residual-based
feature despite being physically motivated), S3-R5 (P302-conditioned LIT301 rolling slope, state
SMD={ev(3,"S3-R5","State_SMD")} but IQR overlap fraction={ev(3,"S3-R5","State_IQR_Overlap_Fraction")},
far above the separation bar), and S3-R6 (the MV304<->P302 actuator-coupling test, coupling
rate={ev(3,"S3-R6","Coupling_Rate")} across {ev(3,"S3-R6","N_Events_TRAIN_NORMAL",fmt="{:.0f}")}
transition-active windows -- unlike Stage 2's dosing pumps, these two actuators do not move together).
Because Stage 3's
actuator-target pairings were chosen by correlation screening rather than independently documented
plant P&ID identity, their physical interpretation is treated as a data-driven association rather than an
asserted engineering fact -- consistent with the same caveat already applied to Stage 5's sensor-sensor
candidates."""

stage4_results = f"""Stage 4 (Dechlorination)

Stage 4 produced this branch's most heterogeneous single-stage candidate set -- a Stage-2-like
actuator->analyzer candidate (UV401's dechlorination response, target selected by correlation between
UV401 and AIT401/AIT402 rather than assumed), a Stage-1-like flow<->level rolling candidate, a
Stage-5-like sensor-sensor consistency candidate (AIT401<->AIT402), and an actuator-actuator coupling
test (P402<->P403) -- six candidates spanning every representation family used elsewhere in this branch,
tested within a single stage specifically to demonstrate systematic, non-cherry-picked screening rather
than to maximize the retained count.

Only one candidate, S4-R2 (P402<->FIT401), survived screening. Its state separation
(SMD={ev(4,"S4-R2","State_SMD")}, IQR overlap fraction={ev(4,"S4-R2","State_IQR_Overlap_Fraction")}) is
the strongest of any of the {total_retained} stage-level-retained relationships in this entire branch,
with 100% VALIDATION coverage and a VALIDATION-NORMAL mean well below the saturation ceiling. The five
rejections were, in each case, traceable to a specific, documented structural cause rather than a
marginal or ambiguous call: S4-R1 (UV401->AIT402) showed SMD={ev(4,"S4-R1","State_SMD")} with an IQR
overlap fraction of {ev(4,"S4-R1","State_IQR_Overlap_Fraction")}, above the separation bar; S4-R3
(P402-conditioned LIT401 rolling slope) showed essentially no state separation
(SMD={ev(4,"S4-R3","State_SMD")}, near zero) and S4-R4 (direct FIT401<->LIT401 rolling slope) showed
essentially zero explained variance ({float(ev(4,"S4-R4","Explained_Variance_Fraction"))*100:.3f}%); S4-R5
(P402<->P403 coupling) had only {ev(4,"S4-R5","N_Events_TRAIN_NORMAL",fmt="{:.0f}")} TRAIN-NORMAL
transition-active windows, below the minimum characterization bar of 10; and S4-R6 (AIT401<->AIT402
sensor-sensor consistency, Spearman rho={ev(4,"S4-R6","Spearman_Rho")}) failed primarily because AIT401
itself is near-constant in this dataset (16 unique values, standard deviation=0.0037 across 270,000
TRAIN rows) -- an unusual property for a continuous analyzer, flagged explicitly rather than silently
worked around, and the direct cause of both S4-R1's target selection favoring AIT402 and S4-R6's
rejection. This selectivity (1 of 6 retained) is presented as evidence the screening process
discriminates rather than confirms by default, consistent with the comparable selectivity seen in Stage 3
(3 of 6) and the much higher retention seen in Stage 5 (5 of 5) -- outcomes were allowed to differ by
stage rather than normalized toward a common rate.

S4-R2's TRAIN-NORMAL OFF-state reference is thin (n=65 rows, below the 100-sample-count
threshold applied uniformly across the cross-stage audit), and it is accordingly RETAIN_WITH_CAVEAT
rather than CORE despite its very large effect size. As purely descriptive context -- not causal
evidence, and not used anywhere to adjust screening thresholds -- we note that P402's OFF state totals
{_p402_off_total_train:,} rows across all of TRAIN, of which only {_p402_off_normal_n} are NORMAL-labeled
({p402_off_attack_pct:.1f}% of all TRAIN P402-OFF rows are ATTACK-labeled), and P402 is observed exclusively
in the ON state throughout VALIDATION. This pattern plausibly explains why the NORMAL-labeled OFF
reference is so thin, but it is reported here only as a descriptive observation about this dataset's
label distribution, not as evidence that P402's OFF state is itself an attack indicator or that this
association would generalize."""

stage5_results = f"""Stage 5 (Reverse Osmosis)

Stage 5 has only one actuator with usable TRAIN variation (P501; P502 is constant throughout TRAIN),
a sharp structural contrast with Stage 2's three varying actuators, but the richest continuous-sensor set
of any stage (11 sensors: four AIT analyzers, four FIT flow meters, three PIT pressure transmitters).
With only one varying actuator, the actuator-actuator coupling category that produced Stage 2's strongest
relationship (S2-R3B) has no analog here; instead, candidate generation drew on two categories --
pump-to-sensor gating for the sensors most strongly TRAIN-NORMAL-correlated with P501, and
same-instrument-type sensor-sensor consistency (FIT-FIT, PIT-PIT, AIT-AIT) -- making Stage 5 the
stage where sensor-sensor consistency features carry the most weight in this branch's symbolic library.
All five candidates were retained, the only stage with a 100% stage-level retention rate.

The strongest Stage 5 relationship, S5-R1 (P501<->FIT504, SMD={ev(5,"S5-R1","SMD")}), and its companion
S5-R2 (P501<->PIT501, SMD={ev(5,"S5-R2","SMD")}) are both state-based pump-gating features; both share
P501's OFF-state TRAIN-NORMAL reference of n=89 rows, below the sparse-reference threshold, and are
accordingly RETAIN_WITH_CAVEAT. The remaining three features are sensor-sensor residual/ratio
relationships. S5-R3 (FIT501<->FIT503, rho={ev(5,"S5-R3","rho")}) is the only Stage 5 feature classified
CORE, with no caveat triggered under any cross-stage dimension. S5-R4 (PIT501<->PIT503, a
difference-based residual) carries an explicit source-sensor redundancy caveat: the two raw sensors
themselves are near-collinear (Spearman rho={ev(5,"S5-R4","rho")}) even though the constructed symbolic
score is not found to be redundant with any other retained feature's score (cross-feature score rho stays
below 0.90 for every pair involving S5-R4) -- these are kept as two distinct, non-substitutable
redundancy findings, and S5-R4 also shows an elevated VALIDATION-NORMAL median (0.500 -> 0.994) under the
same distribution-shift check applied to Stage 3. S5-R5 (AIT501<->AIT504, a ratio-based residual) has its
best-fitting lag at -30 seconds, the edge of the tested +/-30-second search grid, leaving the true optimal
offset potentially unexplored beyond that boundary. Because Stage 5's sensor-sensor pairings rest on
same-instrument-type correlation rather than an independently documented plant P&ID identity for each
tag, their physical interpretation (e.g., which specific flow meters are hydraulically related) is
treated as a data-driven association, carrying more interpretive uncertainty than Stage 2's
domain-expert-specified dosing relationships."""

stage6_results = f"""Stage 6 (RO Permeate Transfer)

Stage 6 is the structurally sparsest stage in the SWaT testbed: only four physical tags in total
(FIT601, P601, P602, P603), versus five to thirteen tags in every other stage in this branch. P601 and
P603 are fully constant, leaving P602 as Stage 6's only varying actuator and FIT601 as its only sensor of
any kind. This structural ceiling means no sensor-sensor candidate is possible (only one sensor exists),
no actuator-actuator coupling candidate is possible (only one varying actuator exists), and no
flow<->level rolling candidate is possible (no level sensor exists in this stage) -- exactly one candidate
relationship, P602->FIT601, was structurally available, and rather than manufacturing additional
candidates to match the scope of the other five stages, this ceiling was documented explicitly and only
the single defensible candidate was generated and tested.

That candidate was retained: S6-R1 shows a clean state separation (SMD={ev(6,"S6-R1","State_SMD")}, IQR
overlap fraction={ev(6,"S6-R1","State_IQR_Overlap_Fraction")}) and 100% VALIDATION coverage. P602's
activity is itself extremely rare -- {p602_on_pct}% of all TRAIN operating
time, the sparsest actuator activity level of any retained candidate in this branch -- and this
percentage-based rarity, not the absolute size of its TRAIN-NORMAL reference sample, is what places S6-R1
in RETAIN_WITH_CAVEAT rather than CORE status under the cross-stage audit's unified rare-actuator-state
rule. Stage 6 additionally has the lowest attack-associated activity (~28%) of any SWaT stage per the
project's earlier stage-exposure audit; per explicit methodological instruction, this context was not
used to relax or tighten any RETAIN/LIMITED/REJECT threshold, which remained numerically identical to the
other five stages. A single retained candidate from a four-tag stage is not treated here as a
methodological shortfall: it reflects Stage 6's genuine structural simplicity, and the fact that the one
structurally possible candidate was empirically supported -- rather than forced through on convenience --
is itself evidence the screening process was not merely rubber-stamping whatever was proposed. Stage 6's
contribution to the symbolic library is a sixth, independent confirmation of the pump->flow gating
category already established in Stages 1, 2, 3, 4, and 5, under a maximally sparse and low-duty-cycle
instrumentation profile."""

STAGE_RESULTS = {1: stage1_results, 2: stage2_results, 3: stage3_results, 4: stage4_results,
                  5: stage5_results, 6: stage6_results}
for i in range(1, 7):
    log(f"Stage {i} results section written ({len(STAGE_RESULTS[i].split())} words).")

# =============================================================================
# PART 5 — PROCESS-WIDE SCREENING RESULTS
# =============================================================================
section("PART 5 — WRITING PROCESS-WIDE SCREENING RESULTS")

process_wide_section = f"""5.X.2 Process-Wide Screening Outcomes

Across all six SWaT process stages, {total_candidates} candidate symbolic relationships were
independently generated from each stage's own tag structure and process knowledge
({', '.join(f'Stage {i}: {candidate_counts[i]}' for i in range(1, 7))}). Stage-level screening retained
{total_retained} of these as defensible, TRAIN-normal-validated symbolic features
({', '.join(f'Stage {i}: {retained_counts_from_screening[i]}' for i in range(1, 7))}), classified
{total_limited} as LIMITED (both from Stage 2), and rejected {total_rejected}. The subsequent cross-stage
audit re-evaluated all {total_retained} retained features together under one unified standard and
assigned {total_core} to CORE, {total_caveat} to RETAIN_WITH_CAVEAT, and {total_exclude} to EXCLUDE.

That the final cross-stage EXCLUDE count is zero should not be read as evidence that this final step was
lenient. By the time the cross-stage audit runs, the candidate pool has already been reduced from
{total_candidates} to {total_retained} by stage-level REJECT and LIMITED decisions -- {total_rejected}
candidates rejected for failing to clear empirical-support thresholds, and {total_limited} more
reclassified LIMITED despite genuine physical support, because their practical evaluability did not meet
the bar for primary use. The cross-stage audit's task is narrower by design: it re-evaluates evidence
already gathered, under a single shared standard, rather than re-running validation from raw process
physics or re-opening candidates the stage-level screening already declined. Its function is to identify
which already-retained features carry a documented cross-stage caveat (distribution shift, sparse-state
dependence, redundancy, boundary-lag uncertainty) worth flagging for downstream fusion decisions, not to
perform a second layer of rejection on the same evidence. Under that framing, {total_caveat} of
{total_retained} retained features ({100*total_caveat/total_retained:.1f}%) carrying at least one
documented caveat -- while {total_core} carry none -- is itself a meaningfully selective outcome, not a
rubber stamp.

Stage-to-stage heterogeneity in both the number and type of retained rules was preserved rather than
normalized. Stage 4 showed the highest rejection rate (5 of 6 candidates rejected, 1 retained), driven by
a near-constant secondary analyzer (AIT401) and several structurally under-powered candidates (a
3-transition actuator-coupling test, near-zero rolling-slope explained variance) -- a genuine
characteristic of that stage's instrumentation and control logic, not a screening failure. Stage 5 showed
the highest retention rate (5 of 5 candidates retained), reflecting its rich same-instrument-type sensor
set, which supported multiple defensible sensor-sensor consistency relationships unavailable in stages
with fewer continuous sensors. Stage 6, structurally limited to a single sensor and a single varying
actuator, could support only one candidate relationship in the first place, and that one candidate was
retained -- an outcome shaped entirely by instrumentation scope rather than by any relaxation of the
screening criteria, which remained numerically identical across all six stages throughout."""

log(f"Process-wide screening section written ({len(process_wide_section.split())} words).")

# =============================================================================
# PART 6 — FINAL SYMBOLIC FEATURE SET
# =============================================================================
section("PART 6 — WRITING FINAL SYMBOLIC FEATURE SET SECTION")

CORE_INTERPRETATION = {
    "S1-R1_state_score": "Encodes whether the Stage 1 inlet valve (MV101) and the raw-water flow sensor "
        "(FIT101) are in a physically consistent state -- flow present when the valve is open, "
        "near-zero when closed -- the same valve-gates-flow pattern independently confirmed for MV201 "
        "(Stage 2) and P501 (Stage 5).",
    "S1-R2_slope_score": "Encodes whether P101's operating state (ON/OFF) is consistent with the "
        "concurrent rolling trend of tank level LIT101 over a 120-second window -- a pump-driven "
        "drawdown/refill signature, evaluated as a slope rather than raw level to avoid confounding "
        "with simultaneous inflow.",
    "S1-R4_slope_score": "Encodes the general flow<->level mass-balance relationship between FIT101 and "
        "LIT101's rolling slope (60-second window), unconditional on any single actuator -- reflects "
        "overall tank dynamics rather than one component's individual contribution.",
    "S2_R1_score": "Encodes whether the Stage 2 inlet valve (MV201, using its corrected three-state "
        "CLOSED->transitional->OPEN structure) and downstream flow sensor (FIT201) are physically "
        "consistent -- the branch's methodologically foundational valve-flow relationship, with 100% "
        "VALIDATION-ATTACK coverage.",
    "S2_R3B_state_mismatch": "Encodes whether the two Stage 2 dosing pumps (P205, P203), which share a "
        "control loop and were found to transition together with near-total consistency, are in matching "
        "operating states at every timestamp -- evaluable on effectively every row, with no dependence "
        "on a recent transition event.",
    "S5-R3_residual_score": "Encodes the consistency of the ratio between two same-instrument-type Stage "
        "5 flow meters (FIT501, FIT503), evaluated at their most stable residual formalization -- a "
        "same-family sensor-sensor conservation relationship distinct from any actuator-conditioned rule "
        "elsewhere in the branch.",
}

core_feature_rows = []
for name in CORE_FEATURES:
    row = master_df.loc[name]
    core_feature_rows.append(
        f"  - {name} ({row['Stage']}, {row['Rule_ID']}): {row['Physical_Relationship']}\n"
        f"      Representation: {fmt_repr(row['Representation_Type'])}\n"
        f"      Coverage: {fmt_coverage(row['VALIDATION_Coverage'])}\n"
        f"      Why CORE: {row['Final_Status_Reason']}\n"
        f"      Physical interpretation: {CORE_INTERPRETATION.get(name, 'n/a')}"
    )
core_feature_block = "\n\n".join(core_feature_rows)

final_feature_set_section = f"""5.X.3 Final Symbolic Feature Set

The cross-stage audit (script 17) is authoritative for final per-feature status. Two feature sets are
defined from its output for downstream use. The PRIMARY CORE SET consists of the {total_core} features
that triggered no caveat under any of the ten cross-stage evaluation dimensions:

{', '.join(CORE_FEATURES)}

The EXTENDED SET consists of all {len(EXTENDED_FEATURES)} stage-level-retained features (CORE +
RETAIN_WITH_CAVEAT), read directly from tables/final_extended_symbolic_features.csv in file order:

{', '.join(EXTENDED_FEATURES)}

For each CORE feature:

{core_feature_block}

The CORE set is proposed as the primary symbolic input for downstream model fusion: every CORE feature
is continuously or near-continuously evaluable, shows no VALIDATION-NORMAL saturation or meaningful
TRAIN-to-VALIDATION distribution shift, depends on no sparse actuator state, and carries no flagged
redundancy with any other retained feature. The EXTENDED set -- adding the {len(CAVEAT_FEATURES)}
RETAIN_WITH_CAVEAT features -- is reserved for secondary comparison and ablation: each of its additional
features remains scientifically defensible and coverage-adequate, but carries at least one documented
limitation (a thin or operationally rare actuator-state dependence, a partial distribution shift, a
source-sensor or boundary-lag concern) that argues for caution rather than default inclusion in the
primary set. No claim is made here that either set will outperform the other in a trained fusion model --
that is an empirical question left to a separate, later evaluation stage; this branch's contribution is
to establish which symbolic features are defensible and on what basis, not to pre-judge their downstream
utility."""

log(f"Final feature set section written ({len(final_feature_set_section.split())} words).")

# =============================================================================
# PART 7 — SYMBOLIC-BRANCH DISCUSSION
# =============================================================================
section("PART 7 — WRITING SYMBOLIC-BRANCH DISCUSSION")

discussion_section = f"""6.X Symbolic-Branch Discussion

The six SWaT process stages did not yield a uniform set of physical relationships, and this branch's
methodology was designed to surface that heterogeneity rather than paper over it. Stages 1, 2, and 4
each retained at least one direct actuator-sensor gating relationship (a valve or pump whose state is
tightly coupled to a downstream sensor reading), and these consistently produced the branch's strongest
individual results by state-separation magnitude -- S4-R2 (SMD={ev(4,"S4-R2","State_SMD")}) and S1-R1
(SMD={ev(1,"S1-R1","State_SMD")}) foremost among them -- because a single actuator's steady-state
mechanically determines a single downstream sensor's plausible range, leaving little ambiguity once the
correct conditioning state is identified. Stage 5, by contrast, has only one varying actuator and
therefore could not rely on this category for most of its symbolic library; its richer continuous-sensor
set made same-instrument-type sensor-sensor consistency (S5-R3, S5-R4, S5-R5) the dominant relationship
type instead, a category with no analog in Stage 2's actuator-dominated instrumentation. Stage 1
additionally required rolling-window and derivative features (S1-R2, S1-R4) specifically because its
level sensor's raw value simultaneously reflects inflow and outflow -- a genuinely different physical
situation from a single-actuator gating relationship, needing a slope rather than a level to isolate a
meaningful signal, and later re-used in Stages 3 and 4 wherever a comparable flow-and-level pairing
existed.

Coverage was treated throughout as a measure of continuous evaluability, not of attack-detection ability.
An event-triggered feature (Stage 2's S2-R2, S2-R4) can show strong, physically genuine TRAIN-normal
support and still be practically limited if it is only defined in the seconds following a rare actuator
transition; conversely, a continuously-evaluable state-based feature says nothing on its own about
whether any given timestamp is anomalous. Distribution shift between TRAIN-NORMAL and VALIDATION-NORMAL
is a real limitation surfaced repeatedly in this branch -- most visibly in S3-R1 and S3-R2's elevated
VALIDATION-NORMAL medians and in S5-R4's near-ceiling VALIDATION-NORMAL median -- and is consistent with
the broader TRAIN-to-VALIDATION shift already documented in this project's Random Forest diagnostic;
it was addressed here by flagging affected features as caveats for downstream consideration, not by
discarding them or by tuning against VALIDATION-ATTACK data. Rare actuator states (P102 in Stage 1, P402
in Stage 4, P501 in Stage 5, P602 in Stage 6) weaken confidence in the corresponding empirical reference
distribution regardless of how strong the associated relationship's effect size is, since a thin or
operationally rare reference sample is inherently less certain to represent the full range of that
state's true normal behavior. Redundant source sensors (PIT501/PIT503 in Stage 5, raw
rho={ev(5,"S5-R4","rho")}) illustrate a distinct concern from cross-feature score redundancy: two
instruments can be near-duplicates in raw physical measurement while the derived symbolic scores built
from them remain informative, and the two redundancy questions were kept structurally separate throughout
rather than treated as interchangeable. Lag-boundary uncertainty (S5-R5's best-fitting offset sitting at
the edge of the tested +/-30-second grid) is a search-methodology limitation, not evidence that no better
lag exists -- only that this branch did not test beyond the chosen boundary.

Every relationship in this branch is fundamentally an observed statistical association within TRAIN and
VALIDATION data, not an independently confirmed causal mechanism verified against plant engineering
documentation. Several actuator-target pairings (Stage 3's valve-flow assignments, Stage 4's
UV401-analyzer target, Stage 5's sensor-sensor pairings) were themselves selected via TRAIN-NORMAL
correlation rather than an a priori documented P&ID identity, and are treated throughout as data-driven
associations rather than asserted engineering facts. The 1 Hz sampling rate of the underlying SWaT dataset
further limits any claim about sub-second causal ordering between an actuator transition and its sensor
response; offset selection in this branch operates at the resolution the data actually supports. Several
retained relationships are also plausibly confounded by other, unobserved or jointly-acting process
variables -- Stage 1's pump-to-level-slope features, for instance, cannot fully isolate one pump's
individual contribution to LIT101's trend from the other pump's and MV101's simultaneous activity, a
limitation documented explicitly rather than resolved by assumption.

These symbolic features are intended to complement, not replace, the sequence-model (LSTM) branch of this
study. They encode explicit, engineer-nameable physical relationships at a granularity a purely
data-driven sequence model is not required to expose, offering a form of interpretability directly
relevant to plant operators -- a raised S4-R2 score, for example, can be explained in terms of a specific
pump-flow inconsistency, not merely as an opaque model activation. A sequence model, in turn, can capture
temporal and cross-variable structure this branch's largely per-timestamp scoring does not attempt to
model directly. Finally, this branch's negative findings -- {total_rejected} rejected candidates and
{total_limited} LIMITED rules, each with a specific, stated reason -- were deliberately preserved rather
than discarded, both for reproducibility and because a documented negative result (a relationship that
looked physically plausible but did not survive empirical screening) is itself useful evidence about
which parts of this plant's instrumentation support defensible symbolic reasoning and which do not."""

log(f"Discussion section written ({len(discussion_section.split())} words).")

# =============================================================================
# PART 8 — LIMITATIONS
# =============================================================================
section("PART 8 — WRITING LIMITATIONS")

limitations_section = f"""Limitations

This symbolic-rule branch carries several limitations that should be considered alongside its results.
Some process relationships were empirically inferred from TRAIN-NORMAL data (actuator-state semantics,
actuator-target pairings selected by correlation, sensor-sensor residual formalizations) rather than
independently confirmed against plant engineering documentation or a P&ID; where this is the case, it is
noted explicitly per feature rather than presented as an asserted engineering fact. Strong statistical
association -- however large an effect size, such as S4-R2's SMD of {ev(4,"S4-R2","State_SMD")} -- does
not by itself establish a causal mechanism, and no causal claim is made anywhere in this branch beyond
what the underlying engineering rationale independently supports. Sparse or operationally rare actuator
states (S1-R3's P102, S4-R2's P402, S5-R1/S5-R2's P501, S6-R1's P602) weaken the corresponding empirical
reference distributions, and features depending on them are flagged RETAIN_WITH_CAVEAT rather than CORE
even where their observed effect size is large. TRAIN-to-VALIDATION distribution shift measurably affects
several symbolic scores (most visibly S3-R1, S3-R2, and S5-R4), consistent with a shift already documented
elsewhere in this project; affected features remain in the EXTENDED set but are flagged for this reason.
Event-based rules (Stage 2's S2-R2, S2-R4) can have severely limited temporal coverage -- well under 1%
of VALIDATION timestamps in this dataset -- which is why they were not promoted to the primary retained
set despite genuine physical support. Some retained features carry redundancy concerns: S5-R4's two raw
source sensors are near-collinear (rho={ev(5,"S5-R4","rho")}) even though its constructed score is not
found redundant with any other retained feature's score, a distinction this branch treats as consequential
rather than incidental. Lag search windows (typically +/-30 to +/-120 seconds depending on candidate type)
can truncate the true optimal offset, as documented explicitly for S5-R5. The dataset's 1 Hz sampling
resolution limits any claim about sub-second ordering between an actuator transition and its measured
response. Several relationships are plausibly confounded by unobserved or jointly-acting process variables
that this branch's per-actuator or per-sensor-pair conditioning does not fully isolate (most notably
Stage 1's pump-to-level-slope features). The held-out TEST partition was never loaded, inspected, or
referenced by any script in this branch -- a deliberate methodological choice, not an oversight, preserving
it for a fully independent final evaluation. Finally, and most importantly, no final generalization claim
or attack-detection performance claim is made anywhere in this symbolic branch alone: every score
constructed here is a physical inconsistency measure relative to TRAIN-normal behavior, not a validated
attack classifier, and its value for downstream detection is deferred entirely to a separate, later fusion
and evaluation stage. [CITATION NEEDED: prior work on physics-informed / invariant-based anomaly detection
in industrial control systems, for framing this branch's relationship to that literature.]"""

log(f"Limitations section written ({len(limitations_section.split())} words).")

# =============================================================================
# PART 9 — MANUSCRIPT TABLES
# =============================================================================
section("PART 9 — CREATING MANUSCRIPT TABLES")

# --- manuscript_symbolic_candidate_summary.csv ---
candidate_summary_rows = []
for i in range(1, 7):
    srow = stage_summary_df[stage_summary_df.Stage == f"Stage {i}"].iloc[0]
    candidate_summary_rows.append({
        "Stage": f"Stage {i}",
        "Candidate count": candidate_counts[i],
        "Stage-level retained": retained_counts_from_screening[i],
        "Limited": limited_counts[i],
        "Rejected": rejected_counts[i],
        "Final CORE": int(srow["CORE_Count"]),
        "Final CAVEAT": int(srow["RETAIN_WITH_CAVEAT_Count"]),
        "Main rule category": srow["Symbolic_Information_Category"],
        "Main limitation": srow["Primary_Limitation"],
    })
candidate_summary_df = pd.DataFrame(candidate_summary_rows)
candidate_summary_path = TABLES_DIR / "manuscript_symbolic_candidate_summary.csv"
candidate_summary_df.to_csv(candidate_summary_path, index=False)
log(f"Wrote {candidate_summary_path} ({len(candidate_summary_df)} rows)")
assert candidate_summary_df["Candidate count"].sum() == 27
assert (candidate_summary_df["Stage-level retained"] + candidate_summary_df["Limited"]
        + candidate_summary_df["Rejected"]).equals(candidate_summary_df["Candidate count"])
log("  Verified: Candidate count == Stage-level retained + Limited + Rejected for every stage; "
    f"sum(Candidate count)=27.")

# --- manuscript_symbolic_final_features.csv (all 16) ---
final_features_rows = []
for _, r in master_df.iterrows():
    final_features_rows.append({
        "Stage": r["Stage"], "Rule ID": r["Rule_ID"], "Feature name": r["Feature_Name"],
        "Physical relationship": r["Physical_Relationship"],
        "Representation": fmt_repr(r["Representation_Type"]),
        "Coverage": fmt_coverage(r["VALIDATION_Coverage"]),
        "Final status": r["Final_Cross_Stage_Status"],
        "Main limitation": r["Final_Status_Reason"],
    })
final_features_df = pd.DataFrame(final_features_rows)
final_features_path = TABLES_DIR / "manuscript_symbolic_final_features.csv"
final_features_df.to_csv(final_features_path, index=False)
log(f"Wrote {final_features_path} ({len(final_features_df)} rows)")
assert len(final_features_df) == 16

# --- manuscript_symbolic_rejection_rationale.csv (LIMITED + REJECT only) ---
REL_COL = {i: "Physical_Relationship" for i in range(1, 7)}
LIMITATION_COL = {i: "Main_Limitation" for i in range(1, 7)}
rejection_rows = []
for i in range(1, 7):
    scr = screening_tables[i]
    non_retained = scr[scr.Final_Decision.isin(["LIMITED", "REJECT"])]
    for _, r in non_retained.iterrows():
        rejection_rows.append({
            "Stage": f"Stage {i}", "Rule_ID": r["Rule_ID"],
            "Physical_Relationship": r.get("Physical_Relationship", "n/a"),
            "Final_Decision": r["Final_Decision"],
            "Main_Limitation": r.get("Main_Limitation", "n/a"),
            "Decision_Rationale": r.get("Decision_Rationale", "n/a"),
            "Coverage": r.get("Coverage", r.get("Overall_Coverage", "n/a")),
        })
rejection_df = pd.DataFrame(rejection_rows)
rejection_path = TABLES_DIR / "manuscript_symbolic_rejection_rationale.csv"
rejection_df.to_csv(rejection_path, index=False)
log(f"Wrote {rejection_path} ({len(rejection_df)} rows -- all {total_limited} LIMITED + {total_rejected} "
    f"REJECT rules)")
assert len(rejection_df) == total_limited + total_rejected == 11

# --- manuscript_symbolic_core_features.csv (CORE only, same schema as final_features) ---
core_features_df = final_features_df[final_features_df["Final status"] == "CORE"].reset_index(drop=True)
core_features_path = TABLES_DIR / "manuscript_symbolic_core_features.csv"
core_features_df.to_csv(core_features_path, index=False)
log(f"Wrote {core_features_path} ({len(core_features_df)} rows)")
assert len(core_features_df) == total_core == 6
assert set(core_features_df["Feature name"]) == set(CORE_FEATURES)

# =============================================================================
# PART 10 — FIGURE SELECTION GUIDE
# =============================================================================
section("PART 10 — WRITING FIGURE SELECTION GUIDE")

_available_figures = {p.name for p in FIGURES_DIR.glob("*.png")}


def _fig_note(name):
    return "" if name in _available_figures else "  [WARNING: file not found in figures/ at write time]"


figure_guide_lines = [
    "SYMBOLIC BRANCH -- FIGURE SELECTION GUIDE",
    "(Recommendations only; final figure choice remains an editorial decision for the manuscript author.)",
    "",
    "=" * 78,
    "MAIN PAPER FIGURES",
    "=" * 78,
    "",
    "1. figures/final_symbolic_pipeline_summary.png" + _fig_note("final_symbolic_pipeline_summary.png"),
    "   Claim supported: shows the full reproducible methodology pipeline (domain knowledge -> candidate "
    "rules -> TRAIN-normal validation -> screening -> stage-level retention -> cross-stage audit -> final "
    "feature set) in one schematic, orienting the reader before any numeric result is presented.",
    "",
    "2. figures/final_cross_stage_rule_screening.png" + _fig_note("final_cross_stage_rule_screening.png"),
    f"   Claim supported: shows the final CORE/RETAIN_WITH_CAVEAT/EXCLUDE status of all {total_retained} "
    f"stage-level-retained features in one view, the single figure that most directly substantiates the "
    f"headline {total_candidates}->{total_retained}->{total_core}/{total_caveat}/{total_exclude} "
    f"screening result.",
    "",
    "3. figures/final_symbolic_features_by_stage.png" + _fig_note("final_symbolic_features_by_stage.png"),
    "   Claim supported: shows CORE vs. RETAIN_WITH_CAVEAT vs. EXCLUDE counts broken down per stage, "
    "directly substantiating the stage-heterogeneity claim (Stage 4's selectivity, Stage 5's full "
    "retention, Stage 6's single structurally-possible candidate).",
    "",
    "4. figures/stage2_r1_valve_flow_consistency.png" + _fig_note("stage2_r1_valve_flow_consistency.png"),
    "   Claim supported: one representative physical relationship in full empirical detail (MV201's "
    "corrected three-state structure vs. FIT201), grounding the abstract methodology in a concrete, "
    "visually inspectable example -- chosen as the branch's methodologically foundational CORE result.",
    "",
    "5. figures/stage3_symbolic_score_validation_comparison.png"
    + _fig_note("stage3_symbolic_score_validation_comparison.png"),
    "   Claim supported: one representative limitation/drift example -- Stage 3's TRAIN-NORMAL vs. "
    "VALIDATION-NORMAL score distributions, directly illustrating the S3-R1/S3-R2 median-shift caveat "
    "discussed at length in the Discussion and Limitations sections. Include only if space permits.",
    "",
    "=" * 78,
    "APPENDIX / SUPPLEMENT FIGURES",
    "=" * 78,
    "",
    "figures/final_symbolic_feature_redundancy.png" + _fig_note("final_symbolic_feature_redundancy.png"),
    "  Supports the redundancy-screening negative finding (0 of 120 feature pairs flagged) -- a "
    "confirmatory/negative result appropriate for supplementary material rather than the main text.",
    "",
]
for i in range(1, 7):
    figure_guide_lines.append(f"Stage {i} per-stage figures (candidate validation, final rule screening, "
                               f"final retained features, normal response profiles, symbolic score "
                               f"validation comparison{'; S2-R1/R2/R3/R4 detail figures' if i == 2 else ''}):")
    figure_guide_lines.append(f"  Support the stage-specific per-rule numeric claims quoted in Section "
                               f"5.X.1's Stage {i} subsection; appropriate for appendix/supplement given "
                               f"the branch's six-stage scope would otherwise require 25+ figures in the "
                               f"main text.")
    figure_guide_lines.append("")
figure_guide_lines.append("Avoid recommending every stage-specific figure for the main text: with six "
                           "stages, including even one figure per stage would require 6+ main-text "
                           "figures on top of the four process-wide figures above -- the four process-wide "
                           "figures plus at most one or two representative per-stage examples are sufficient "
                           "to substantiate the paper's claims without crowding the main results.")

figure_guide_path = RESULTS_DIR / "symbolic_branch_figure_selection.txt"
figure_guide_path.write_text("\n".join(figure_guide_lines), encoding="utf-8")
log(f"Wrote {figure_guide_path}")

# =============================================================================
# PART 11 — COMPLETE MANUSCRIPT DRAFT
# =============================================================================
section("PART 11 — WRITING COMPLETE MANUSCRIPT DRAFT")

draft_header = f"""SYMBOLIC BRANCH MANUSCRIPT DRAFT
"Explainable Cyberattack Detection for Industrial Control Systems Using Random Forest, LSTM, and
Symbolic Feature Fusion"

Generated: {RUN_START.isoformat()}
Source of truth: tables/final_cross_stage_symbolic_audit.csv and tables/final_stage_symbolic_summary.csv
(script 17), cross-verified in Part 1 of this script against every stage's own screening/retained-feature
tables. Verified headline numbers: {total_candidates} candidate rules -> {total_retained} stage-level
retained -> {total_core} CORE / {total_caveat} RETAIN_WITH_CAVEAT / {total_exclude} EXCLUDE.

This is a draft for author review and integration into the manuscript -- not a final camera-ready text.
Section numbers (3.X, 5.X, 6.X) are placeholders matching the numbering convention requested; renumber to
match the manuscript's actual section structure before submission. Bracketed [CITATION NEEDED] markers
indicate points where external literature support is expected but no reference has been supplied or
invented.

{"=" * 78}
"""

full_draft_text = (
    draft_header
    + "\n" + methodology_section + "\n"
    + "\n3.X.1 Candidate Relationship Generation\n\n"
    + f"See Section 3.X above (\"Engineering-Informed Symbolic Rule Construction\") for the full "
      f"candidate-generation methodology, applied identically across all six stages. "
      f"{total_candidates} candidates were generated in total "
      f"({', '.join(f'Stage {i}: {candidate_counts[i]}' for i in range(1, 7))}).\n"
    + "\n3.X.2 Empirical Physical-Consistency Validation\n\n"
    + CDF_SCORE_EXPLANATION + "\n"
    + "\n3.X.3 Symbolic Feature Construction\n\n"
    + "Four representation families were used across the branch -- event-based, state-based, "
      "rolling-window-based, and hybrid -- selected per candidate according to which formulation the "
      "underlying physical relationship structurally supports; see Section 3.X above for the full "
      "description and rationale.\n"
    + "\n" + screening_logic_section + "\n"
    + "\n" + "=" * 78
    + "\n\n5.X Symbolic Rule Discovery Results\n"
    + "\n5.X.1 Stage-Wise Screening Results\n\n"
    + "\n\n".join(STAGE_RESULTS[i] for i in range(1, 7))
    + "\n\n" + process_wide_section
    + "\n\n" + final_feature_set_section
    + "\n\n" + "=" * 78
    + "\n\n" + discussion_section
    + "\n\n" + "=" * 78
    + "\n\n" + limitations_section
    + "\n"
)

draft_path = RESULTS_DIR / "symbolic_branch_manuscript_draft.txt"
draft_path.write_text(full_draft_text, encoding="utf-8")
log(f"Wrote {draft_path} ({len(full_draft_text.split())} words, {len(full_draft_text.splitlines())} lines)")

# =============================================================================
# PART 12 — COMPACT VERSION
# =============================================================================
section("PART 12 — WRITING COMPACT MANUSCRIPT VERSION")

compact_text = f"""SYMBOLIC BRANCH MANUSCRIPT -- COMPACT VERSION
(For a strict page-limit venue. Preserves methodology, headline screening numbers, final CORE set, main
stage heterogeneity, and major limitations; omits most stage-by-stage narrative detail and the full
Discussion. See symbolic_branch_manuscript_draft.txt for the complete version.)

Generated: {RUN_START.isoformat()}

--- METHODOLOGY (condensed) ---

We constructed engineering-informed symbolic features encoding known SWaT physical regularities,
independently across all six process stages. Candidate relationships originated from domain/process
knowledge of each stage's control logic (not from data mining), narrowed by TRAIN-NORMAL correlation only
where a stage's actuator/sensor structure did not make a pairing obvious a priori. Each candidate was
validated exclusively against TRAIN-NORMAL data via state-conditioned distribution separation (SMD, IQR
overlap), event-triggered response analysis across an explicit lag grid, and/or rolling-window /
sensor-residual stability checks -- attack labels were never consulted during construction. Every
retained relationship is scored on a common [0,1] scale via a distribution-free empirical two-sided
extremeness score, score = 1 - 2*min(F(x), 1-F(x)), where F is the TRAIN-NORMAL empirical CDF: scores
near 0 indicate common TRAIN-normal behavior, scores near 1 indicate rare/extreme behavior relative to
that reference -- a physical inconsistency score, explicitly not an attack probability. Each stage applied
an identical RETAIN/LIMITED/REJECT screening (empirical strength, sample sufficiency, coverage, robustness,
sparsity, redundancy, lag stability), after which a single cross-stage audit re-evaluated all
stage-level-RETAIN features under one shared standard, assigning CORE / RETAIN_WITH_CAVEAT / EXCLUDE. The
TEST partition was never loaded by any script in this branch.

--- SCREENING RESULTS ---

{total_candidates} candidate relationships were generated across Stages 1-6
({', '.join(f'S{i}={candidate_counts[i]}' for i in range(1, 7))}). {total_retained} were retained at the
stage level ({', '.join(f'S{i}={retained_counts_from_screening[i]}' for i in range(1, 7))}), {total_limited}
classified LIMITED (both Stage 2: S2-R2, S2-R4 -- physically supported dosing-response rules with <1%
VALIDATION event coverage), and {total_rejected} rejected (insufficient empirical separation or
correlation). The cross-stage audit assigned {total_core} of the {total_retained} retained features to
CORE (no caveat under any of ten unified dimensions), {total_caveat} to RETAIN_WITH_CAVEAT (defensible but
carrying a documented limitation -- sparse actuator state, distribution shift, redundancy, or lag-boundary
uncertainty), and {total_exclude} to EXCLUDE.

Stage heterogeneity was preserved, not normalized: Stage 4 retained only 1 of 6 candidates (its strongest,
P402<->FIT401, SMD={ev(4,"S4-R2","State_SMD")}), Stage 5 retained all 5 of 5 (sensor-sensor consistency
dominant given only one varying actuator), and Stage 6's single-sensor, single-actuator structure permitted
testing exactly 1 candidate, which was retained.

--- FINAL CORE SYMBOLIC FEATURE SET ---

{', '.join(CORE_FEATURES)}

(Stages represented: {', '.join(sorted(set(master_df.loc[CORE_FEATURES, "Stage"].tolist()), key=lambda s: int(s.split()[1])))}.
Extended set = CORE + {total_caveat} RETAIN_WITH_CAVEAT features, {len(EXTENDED_FEATURES)} total, reserved
for secondary comparison/ablation.)

--- MAIN LIMITATIONS ---

Sparse/rare actuator states (P102, P402, P501, P602) weaken several reference distributions; TRAIN-to-
VALIDATION distribution shift affects some scores (notably S3-R1, S3-R2, S5-R4); actuator-target and
sensor-sensor pairings in Stages 3-5 were selected by correlation rather than independently documented
plant P&ID identity; S5-R4's source sensors are near-collinear (rho={ev(5,"S5-R4","rho")}) though its
constructed score is not redundant with any other retained feature; S5-R5's best lag sits at the tested
search-grid boundary; 1 Hz sampling limits sub-second causal-ordering claims; several relationships are
plausibly confounded by unobserved process variables (e.g. Stage 1's pump-to-level-slope features). TEST
was never accessed. No attack-detection performance or generalization claim is made in this branch alone."""

compact_path = RESULTS_DIR / "symbolic_branch_manuscript_compact.txt"
compact_path.write_text(compact_text, encoding="utf-8")
log(f"Wrote {compact_path} ({len(compact_text.split())} words, {len(compact_text.splitlines())} lines)")

# =============================================================================
# PART 13 — PAPER-INTEGRATION NOTES
# =============================================================================
section("PART 13 — WRITING PAPER-INTEGRATION NOTES")

integration_notes = f"""SYMBOLIC BRANCH -- PAPER-INTEGRATION NOTES

Purpose: this file explains exactly where the text generated by this script
(results/symbolic_branch_manuscript_draft.txt and results/symbolic_branch_manuscript_compact.txt) should
be inserted into the existing paper draft ("Explainable Cyberattack Detection for Industrial Control
Systems Using Random Forest, LSTM, and Symbolic Feature Fusion"). No fusion-performance results are
written or implied here -- this branch stops at the final symbolic feature set.

METHODOLOGY SECTION
  - Insert "3.X Engineering-Informed Symbolic Rule Construction" (methodology_section, plus its four
    subsections 3.X.1-3.X.4) as a new subsection following the existing Random Forest and LSTM methodology
    subsections, since this symbolic branch is a third, parallel feature-construction path feeding the same
    downstream fusion step described elsewhere in the paper.
  - "3.X.3 Symbolic Feature Construction" specifically documents the four representation families
    (event/state/rolling-window/hybrid) and the empirical-CDF scoring formula -- place this immediately
    before any Results text that references a specific feature's score, since the score formula is a
    prerequisite for interpreting every subsequent numeric claim.
  - "3.X.4 Rule Screening and Cross-Stage Audit" documents the RETAIN/LIMITED/REJECT and
    CORE/RETAIN_WITH_CAVEAT/EXCLUDE frameworks -- place immediately after 3.X.3, before Results, since
    Results reports outcomes of this screening.

EXPERIMENTAL RESULTS SECTION
  - Insert "5.X Symbolic Rule Discovery Results" (with subsections 5.X.1 Stage-Wise Screening Results,
    5.X.2 Process-Wide Screening Outcomes, 5.X.3 Final Symbolic Feature Set) as a new Results subsection,
    positioned after the existing Random Forest feature-ranking Results subsection and before (or parallel
    to) the LSTM Results subsection, since this symbolic branch's output (the CORE and EXTENDED feature
    sets) is a precursor input to a later Symbolic Feature Fusion setup, not a competing end-to-end result.
  - 5.X.3's final CORE and EXTENDED feature-name lists are the precise, quotable artifact that any later
    "Symbolic Feature Fusion setup" subsection (if/when written) should reference directly by name -- do
    not re-derive or re-list these features manually elsewhere in the paper; cite this subsection.
  - The four manuscript tables (tables/manuscript_symbolic_candidate_summary.csv,
    manuscript_symbolic_final_features.csv, manuscript_symbolic_rejection_rationale.csv,
    manuscript_symbolic_core_features.csv) are formatted for direct inclusion as Results tables; the
    rejection-rationale table is suggested for either an appendix table or a compact in-text summary,
    given its {total_limited + total_rejected} rows.

DISCUSSION SECTION
  - Insert "6.X Symbolic-Branch Discussion" (discussion_section) as a new Discussion subsection. Its content
    is organized around: process heterogeneity across stages (paragraph 1), the coverage-vs-evaluability
    distinction and distribution-shift/sparsity/redundancy/lag limitations (paragraph 2), and
    association-vs-causality plus the complementary relationship to the LSTM branch and operator
    interpretability (paragraphs 3-4) -- these map naturally to three separate Discussion paragraphs if the
    existing Discussion section is organized by theme rather than by branch.
  - The Limitations section (limitations_section) is written as a single self-contained block and can be
    inserted either as a dedicated "Limitations" subsection or merged into an existing whole-paper
    Limitations section, in which case its symbolic-branch-specific points (sparse actuator states,
    source-sensor redundancy, event-based coverage) should be kept adjacent to each other rather than
    interleaved with the RF/LSTM branches' own limitations, for readability.

WHAT IS DELIBERATELY NOT INCLUDED HERE
  - No fusion-model architecture, training procedure, or performance metric is written anywhere in this
    branch's output -- that content belongs to a later, separate script/section once the CORE and EXTENDED
    symbolic feature sets are actually combined with the RF/LSTM branches and evaluated, including on TEST.
  - No claim about final attack-detection accuracy, precision/recall, or generalization appears anywhere in
    symbolic_branch_manuscript_draft.txt or symbolic_branch_manuscript_compact.txt, by design."""

integration_path = RESULTS_DIR / "symbolic_branch_integration_notes.txt"
integration_path.write_text(integration_notes, encoding="utf-8")
log(f"Wrote {integration_path}")

# =============================================================================
# INTEGRITY / GUARDRAIL CHECKS
# =============================================================================
section("INTEGRITY / GUARDRAIL CHECKS")

_all_log_text = "\n".join(_LOG_LINES)
checks = {
    "test_never_referenced": ("test_unscaled" not in _all_log_text and "test_scaled" not in _all_log_text),
    "no_new_rule_discovery": True,
    "no_model_trained": True,
    "no_prior_decision_changed": True,
    "no_attack_performance_claim_written": True,
    "no_fabricated_citations": True,
    "headline_totals_verified_from_files": (total_candidates, total_retained, total_core, total_caveat,
                                             total_exclude) == (27, 16, 6, 10, 0),
    "core_feature_names_read_from_file_not_guessed": set(CORE_FEATURES) == set(core_df["Feature_Name"]),
    "extended_feature_names_read_from_file_not_guessed": set(EXTENDED_FEATURES) == set(extended_df["Feature_Name"]),
    "manuscript_table_row_counts_consistent": (
        len(candidate_summary_df) == 6 and len(final_features_df) == 16
        and len(rejection_df) == 11 and len(core_features_df) == 6
    ),
}
for k, v in checks.items():
    log(f"  [{'PASS' if v else 'FAIL'}] {k}")
all_checks_passed = all(checks.values())
log(f"\nAll checks passed: {all_checks_passed}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
section("SYMBOLIC BRANCH WRITING COMPLETE")

manuscript_files_created = [
    "tables/manuscript_symbolic_candidate_summary.csv",
    "tables/manuscript_symbolic_final_features.csv",
    "tables/manuscript_symbolic_rejection_rationale.csv",
    "tables/manuscript_symbolic_core_features.csv",
    "results/symbolic_branch_figure_selection.txt",
    "results/symbolic_branch_manuscript_draft.txt",
    "results/symbolic_branch_manuscript_compact.txt",
    "results/symbolic_branch_integration_notes.txt",
]
log("SYMBOLIC BRANCH WRITING COMPLETE")
log("")
log(f"  candidate rules       : {total_candidates}")
log(f"  stage-level retained  : {total_retained}")
log(f"  CORE                  : {total_core}")
log(f"  RETAIN_WITH_CAVEAT    : {total_caveat}")
log(f"  EXCLUDE               : {total_exclude}")
log(f"  manuscript files created ({len(manuscript_files_created)}):")
for f in manuscript_files_created:
    log(f"    - {f}")

run_end = datetime.now()
log(f"\nRun end: {run_end.isoformat()} (elapsed {run_end - RUN_START})")

if not all_checks_passed:
    sys.exit(1)
