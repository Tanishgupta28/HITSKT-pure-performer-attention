"""Render the ASSIST2012 methods and verified dataset counts."""
from pathlib import Path

from scripts.aggregate_algebra2005 import read_json

evidence = Path("reports/datasets/assist2012")
data = read_json(evidence / "summary.json")
raw = data["raw"]
prov = data["provenance"]
arch = read_json(evidence / "historical_architecture.json")
split = "\n".join(
    f"| {name.capitalize()} | {data['split_sessions'][name]:,} | {size:,} | "
    f"{size / data['retained_interactions']:.4%} | {data['split_correct'][name]:,} |"
    for name, size in data["split_interactions"].items()
)
dists = "\n".join(
    f"| {name.replace('_', ' ')} | {item['count']:,} | {item['mean']:.2f} | "
    + " | ".join(f"{item['percentiles'][str(p)]:,.2f}" for p in (25, 50, 75, 90, 95, 99, 100)) + " |"
    for name, item in data["distributions"].items()
)
architecture = "\n".join(
    f"- {name.upper()} used commit `{item['commit']}`, checkpoint format {item['checkpoint_format']}, "
    f"width {item['config']['width']}, {item['config']['heads']} heads and "
    f"{item['total_parameters']:,} parameters."
    for name, item in arch["historical_checkpoints"].items()
)
report = f"""# ASSIST2012 knowledge tracing benchmark

ASSIST2012 was selected as a follow-up to ASSIST2009 and Algebra2005 because genuine response times and multiple sessions per student may support within-session and across-session knowledge tracing. The five models use the same full eligible set and the same validation and test prediction targets.

## Source and preprocessing

The official ASSISTments 2012–2013 school data with affect release contains a single quoted, comma-delimited CSV with {len(prov['columns'])} fields. The parser accepts embedded quotes and line breaks in quoted action logs. The byte stream is decoded with `{prov['encoding_used']}`. The ZIP SHA-256 is `{prov['archive_sha256']}` and the untouched CSV SHA-256 is `{prov['csv_sha256']}`. The archive and extracted member were compared byte for byte. [Publisher release and dictionary]({data['source']}) · [Archive]({data['download']}) · [Full schema and missing counts](datasets/assist2012/summary.json)

The source contains {raw['source_rows']:,} responses, {raw['students']:,} students, {raw['questions']:,} questions, and {raw['atomic_skills']:,} nonempty skill IDs. Every `problem_log_id` is unique in this release. Each student–problem encounter remains distinct even when its question ID repeats. Map `user_id` and `problem_id` from sorted integer catalogs. Each row carries at most one nonempty `skill_id`; the usual composite signature is therefore a singleton or the untagged category. IDs start at one and zero is reserved for padding. The retained cohort has {data['untagged_interactions']:,} untagged interactions and {data['repeated_student_question_events']:,} repeated student–question encounters.

The publisher defines the target as 1 when `correct` is 1 and 0 for scores below 1. The raw file has {raw['fractional_score']:,} fractional scores and {raw['invalid_score']:,} invalid scores. Use the logged first graded action to establish when that binary outcome is observable. A first answer uses its explicit recorded correctness, and a first hint or scaffold represents an unsuccessful first attempt. Retain the publisher's binarized label when it agrees with that observation. The full source has {raw['first_action_outcome_differs_from_binary_label']:,} observed disagreements. This removes ambiguous records according to a prespecified label-consistency criterion. The tutor automatically credits open responses, so scored correctness in those records reflects tutor scoring. Affect, answer contents, response duration, action type, hint counts, attempts, end time and other post-answer fields are excluded from model features.

## Chronology, eligibility and split

The prediction clock is arrival of the first logged answer, hint or scaffold, immediately before observing its result. The clock is the exported problem start plus the recorded relative offset for that action, anchored in UTC using the log's Unix start. Validate the start against the exported local clock in Eastern time to one second, and use the exported end time solely to reject arrival clocks after the completion bound, allowing one second for source clock quantization. This uses the same response-arrival prediction framing as Algebra2005. Prediction before problem presentation is a different task. ASSIST2009 used assignment runs and Phi-only RKT because absolute response times were unavailable.

There are {raw['invalid_start_time']:,} invalid exported starts and {raw['invalid_end_time']:,} invalid exported ends. There are {raw['missing_logged_start']:,} records without a parseable action-log start, {raw['missing_logged_first_action']:,} without a parseable first action, {raw['start_clock_mismatch']:,} start-clock mismatches, and {raw['first_action_after_end']:,} first-action times after the end bound. These diagnostics can overlap. A total of {data['timestamp_eligible_interactions']:,} rows have consistent clocks and targets. Their first-action local times range from {raw['first_action_local_min']} to {raw['first_action_local_max']}.

Sort by student, response arrival and source row ID. Start a session at an inactivity gap of at least ten hours. Retain students with at least five sessions. Assign the last `floor(n/5)` sessions to test, the preceding equal number to validation, and the prefix to training. Score first-session events with an empty history where applicable. In the {data['same_clock_additional_events']:,} extra same-clock rows, history excludes all simultaneous responses until the clock advances. HiTSKT uses the existing separate causal-prefix batching from Algebra2005 for those targets. The mask scores each event exactly once. RKT uses genuine elapsed hours, directed exercise Phi, and five student-level folds. Training uses the complement of each student's fold. Holdouts use the all-training cache.

The {data['source_sessions']:,} sessions before eligibility filtering include students with fewer than five sessions. The benchmark retains **{data['retained_students']:,} students, {data['retained_interactions']:,} interactions, {data['retained_sessions']:,} sessions, {data['questions']:,} questions and {data['skills']:,} skill categories**. This represents {data['retained_fraction']:.4%} of the official rows. The remainder includes {data['excluded_for_session_eligibility']:,} eligible responses from students with fewer than five sessions.

| Partition | Sessions | Targets | Target share | Correct |
| --- | ---: | ---: | ---: | ---: |
{split}

| Distribution | Count | Mean | P25 | Median | P75 | P90 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{dists}

## Model and evaluation protocol

Use seed 42, the inherited Adam learning rates and epoch ceilings, strict validation ROC-AUC selection with patience 5 and minimum delta 0, and best-checkpoint test evaluation. Binary cross-entropy with logits is the loss. MSE uses probabilities. Accuracy, precision, recall, and F1 use threshold 0.5. Ordered histories, target and padding masks, and the train-only RKT cache apply across the five full runs. [Data verification](datasets/assist2012/data_verification.json) · [RKT provenance](datasets/assist2012/rkt_provenance.json)

| Model | Context / history | Learning rate | Batch size | Dropout | Epoch ceiling |
| --- | --- | ---: | ---: | ---: | ---: |
| DKT | 200 / 199 interactions | 0.0002 | 20 | 0.1 configured, no recurrent dropout | 200 |
| DKVMN | 200 / 199 interactions | 0.001 | 32 | 0.0 | 100 |
| SAKT | 100 / 99 interactions | 0.00001 | 10 | 0.2 | 300 |
| RKT | 50 / 49 interactions | 0.001 | 128 | 0.1 | 300 |
| HiTSKT | 16 / 15 sessions | 0.00005 | 64 maximum | 0.1 | 100 |

The historical `reports/final_results.csv` HiTSKT rows map to `scripts/train_hitskt.py`, `ktbench/models/hitskt.py`, and `ktbench/models/performer.py`. Both model source files at the benchmark base revision `{arch['source_main_commit']}` match the historical publication revisions byte for byte. The checkpoint configurations instantiate the same class and load strictly. The attention uses the deterministic ELU+1 feature transform and causal prefix sums, with no FAVOR+ random projection. ASSIST2012 uses fresh seed-42 initialization, width 256, four heads, feed-forward width 2,048, one layer per stage and the inherited 32,768 token budget. It uses the existing mask-aware binary cross-entropy loss, shifted correctness input, embeddings, encoders, decoder and prediction head. [Source and checkpoint verification](datasets/assist2012/historical_architecture.json)

{architecture}

## Results

<!-- RESULTS_START -->
Full GPU training and independent checkpoint replay are pending.
<!-- RESULTS_END -->

## Reproduction

Download the complete publisher ZIP into `data/raw/assist2012`. Keep it and local checkpoints outside Git.

```bash
data=data/processed/assist2012/full
python -m scripts.preprocess_assist2012 data/raw/assist2012/2012-2013-data-with-predictions-4-final.zip "$data"
python -m scripts.prepare_rkt assist2012 "$data/session_store" "$data/rkt"
bash scripts/run_assist2012.sh
```
"""
Path("reports/ASSIST2012_EXPERIMENT_REPORT.md").write_text(report)
