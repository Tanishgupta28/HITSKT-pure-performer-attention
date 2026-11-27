# ASSIST2012 knowledge tracing benchmark

ASSIST2012 was selected as a follow-up to ASSIST2009 and Algebra2005 because genuine response times and multiple sessions per student may support within-session and across-session knowledge tracing. The five models use the same full eligible set and the same validation and test prediction targets.

## Source and preprocessing

The official ASSISTments 2012–2013 school data with affect release contains a single quoted, comma-delimited CSV with 35 fields. The parser accepts embedded quotes and line breaks in quoted action logs. The byte stream is decoded with `utf-8`. The ZIP SHA-256 is `7bb18f932f97e63c2eb34bb869c44cd4f46f4adec4d7fe02298578dbba572445` and the untouched CSV SHA-256 is `1d06aee9e649c5ba9db49052fe9a86e68d0c3ccb4709a7487289d7721f5464db`. The archive and extracted member were compared byte for byte. [Publisher release and dictionary](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect) · [Archive](https://drive.google.com/file/d/1cU6Ft4R3hLqA7G1rIGArVfelSZvc6RxY/view) · [Full schema and missing counts](datasets/assist2012/summary.json)

The source contains 6,123,270 responses, 46,674 students, 179,999 questions, and 265 nonempty skill IDs. Every `problem_log_id` is unique in this release. Each student–problem encounter remains distinct even when its question ID repeats. Map `user_id` and `problem_id` from sorted integer catalogs. Each row carries at most one nonempty `skill_id`; the usual composite signature is therefore a singleton or the untagged category. IDs start at one and zero is reserved for padding. The retained cohort has 2,771,237 untagged interactions and 217,301 repeated student–question encounters.

The publisher defines the target as 1 when `correct` is 1 and 0 for scores below 1. The raw file has 5,323 fractional scores and 0 invalid scores. Use the logged first graded action to establish when that binary outcome is observable. A first answer uses its explicit recorded correctness, and a first hint or scaffold represents an unsuccessful first attempt. Retain the publisher's binarized label when it agrees with that observation. The full source has 9,440 observed disagreements. This removes ambiguous records according to a prespecified label-consistency criterion. The tutor automatically credits open responses, so scored correctness in those records reflects tutor scoring. Affect, answer contents, response duration, action type, hint counts, attempts, end time and other post-answer fields are excluded from model features.

## Chronology, eligibility and split

The prediction clock is arrival of the first logged answer, hint or scaffold, immediately before observing its result. The clock is the exported problem start plus the recorded relative offset for that action, anchored in UTC using the log's Unix start. Validate the start against the exported local clock in Eastern time to one second, and use the exported end time solely to reject arrival clocks after the completion bound, allowing one second for source clock quantization. This uses the same response-arrival prediction framing as Algebra2005. Prediction before problem presentation is a different task. ASSIST2009 used assignment runs and Phi-only RKT because absolute response times were unavailable.

There are 0 invalid exported starts and 0 invalid exported ends. There are 0 records without a parseable action-log start, 0 without a parseable first action, 4,971 start-clock mismatches, and 2,194 first-action times after the end bound. These diagnostics can overlap. A total of 6,108,345 rows have consistent clocks and targets. Their first-action local times range from 2012-09-01 00:00:46.137652 to 2013-09-01 00:03:46.128000.

Sort by student, response arrival and source row ID. Start a session at an inactivity gap of at least ten hours. Retain students with at least five sessions. Assign the last `floor(n/5)` sessions to test, the preceding equal number to validation, and the prefix to training. Score first-session events with an empty history where applicable. In the 0 extra same-clock rows, history excludes all simultaneous responses until the clock advances. HiTSKT uses the existing separate causal-prefix batching from Algebra2005 for those targets. The mask scores each event exactly once. RKT uses genuine elapsed hours, directed exercise Phi, and five student-level folds. Training uses the complement of each student's fold. Holdouts use the all-training cache.

The 402,463 sessions before eligibility filtering include students with fewer than five sessions. The benchmark retains **19,092 students, 5,173,463 interactions, 352,017 sessions, 153,547 questions and 265 skill categories**. This represents 84.4886% of the official rows. The remainder includes 934,882 eligible responses from students with fewer than five sessions.

| Partition | Sessions | Targets | Target share | Correct |
| --- | ---: | ---: | ---: | ---: |
| Test | 63,780 | 901,241 | 17.4205% | 607,218 |
| Train | 224,457 | 3,353,476 | 64.8207% | 2,306,643 |
| Validation | 63,780 | 918,746 | 17.7588% | 631,561 |

| Distribution | Count | Mean | P25 | Median | P75 | P90 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all sessions per student | 46,659 | 8.63 | 1.00 | 3.00 | 9.00 | 22.00 | 36.00 | 78.00 | 130.00 |
| inter session gap hours | 332,925 | 214.08 | 28.56 | 99.51 | 218.02 | 503.48 | 815.75 | 1,792.88 | 8,474.51 |
| questions frequency | 153,547 | 33.69 | 2.00 | 12.00 | 37.00 | 78.00 | 119.00 | 302.00 | 2,091.00 |
| retained interaction history length per student | 19,092 | 270.98 | 90.00 | 160.50 | 337.00 | 601.00 | 864.00 | 1,617.36 | 5,562.00 |
| retained interactions per session | 352,017 | 14.70 | 5.00 | 10.00 | 19.00 | 30.00 | 40.00 | 71.00 | 1,032.00 |
| sessions per retained student | 19,092 | 18.44 | 7.00 | 11.00 | 22.00 | 42.00 | 60.00 | 91.00 | 130.00 |
| skills frequency | 265 | 19522.50 | 223.00 | 1,353.00 | 11,127.00 | 23,189.80 | 40,143.00 | 127,634.64 | 2,771,237.00 |

## Model and evaluation protocol

Use seed 42, the inherited Adam learning rates and epoch ceilings, strict validation ROC-AUC selection with patience 5 and minimum delta 0, and best-checkpoint test evaluation. Binary cross-entropy with logits is the loss. MSE uses probabilities. Accuracy, precision, recall, and F1 use threshold 0.5. Ordered histories, target and padding masks, and the train-only RKT cache apply across the five full runs. [Data verification](datasets/assist2012/data_verification.json) · [RKT provenance](datasets/assist2012/rkt_provenance.json)

| Model | Context / history | Learning rate | Batch size | Dropout | Epoch ceiling |
| --- | --- | ---: | ---: | ---: | ---: |
| DKT | 200 / 199 interactions | 0.0002 | 20 | 0.1 configured, no recurrent dropout | 200 |
| DKVMN | 200 / 199 interactions | 0.001 | 32 | 0.0 | 100 |
| SAKT | 100 / 99 interactions | 0.00001 | 10 | 0.2 | 300 |
| RKT | 50 / 49 interactions | 0.001 | 128 | 0.1 | 300 |
| HiTSKT | 16 / 15 sessions | 0.00005 | 64 maximum | 0.1 | 100 |

The historical `reports/final_results.csv` HiTSKT rows map to `scripts/train_hitskt.py`, `ktbench/models/hitskt.py`, and `ktbench/models/performer.py`. Both model source files at the benchmark base revision `33364c5090d90502f55364575218fd69ac201ca9` match the historical publication revisions byte for byte. The checkpoint configurations instantiate the same class and load strictly. The attention uses the deterministic ELU+1 feature transform and causal prefix sums, with no FAVOR+ random projection. ASSIST2012 uses fresh seed-42 initialization, width 256, four heads, feed-forward width 2,048, one layer per stage and the inherited 32,768 token budget. It uses the existing mask-aware binary cross-entropy loss, shifted correctness input, embeddings, encoders, decoder and prediction head. [Source and checkpoint verification](datasets/assist2012/historical_architecture.json)

- ASSIST2017 used commit `d3ff9ecef406e5f8d4cd1bb04abab42fd112c1d6`, checkpoint format 1, width 256, 4 heads and 6,931,457 parameters.
- EDNET_KT1 used commit `d499585b43743fa64232f938e70c01240d9f6ff9`, checkpoint format 1, width 128, 2 heads and 5,074,433 parameters.
- JUNYI used commit `d6c0ff499c040f6cf2fdec13128b3a6ba9a930d9`, checkpoint format 1, width 128, 2 heads and 8,271,105 parameters.

## Results

<!-- RESULTS_START -->
All five full GPU runs completed and independent checkpoint replay matched the recorded metrics.

| Model | Best / completed epoch | Validation ROC-AUC | Loss ↓ | ROC-AUC ↑ | Accuracy ↑ | Precision ↑ | Recall ↑ | F1 ↑ | MSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HiTSKT | 23 / 28 | 0.756207 | **0.562814** | **0.732993** | **0.719748** | **0.758761** | 0.856294 | 0.804582 | **0.188670** |
| DKT | 5 / 10 | 0.717590 | 0.564471 | 0.718151 | 0.715981 | 0.736387 | **0.900993** | **0.810416** | 0.190243 |
| DKVMN | 7 / 12 | 0.707766 | 0.566936 | 0.711875 | 0.713162 | 0.735501 | 0.896764 | 0.808166 | 0.191683 |
| SAKT | 20 / 25 | 0.726888 | 0.571187 | 0.711071 | 0.709971 | 0.745589 | 0.864531 | 0.800667 | 0.193236 |
| RKT | 2 / 7 | 0.730634 | 0.572393 | 0.698156 | 0.709221 | 0.739913 | 0.876532 | 0.802449 | 0.194430 |

### Metric leaders

- **Loss** — HiTSKT (0.562814)
- **ROC-AUC** — HiTSKT (0.732993)
- **Accuracy** — HiTSKT (0.719748)
- **Precision** — HiTSKT (0.758761)
- **Recall** — DKT (0.900993)
- **F1** — DKT (0.810416)
- **MSE** — HiTSKT (0.188670)

### Parameter counts and runtime

| Model | Total parameters | Trainable parameters | Runtime (seconds) | GPU |
| --- | ---: | ---: | ---: | --- |
| HiTSKT | 84,014,081 | 84,014,081 | 8411.00 | NVIDIA H100 80GB HBM3 MIG 3g.40gb |
| DKT | 101,642 | 101,642 | 4410.64 | NVIDIA H100 80GB HBM3 MIG 3g.40gb |
| DKVMN | 97,301 | 97,301 | 6412.46 | NVIDIA H100 80GB HBM3 MIG 3g.40gb |
| SAKT | 15,583,201 | 15,583,201 | 47343.07 | NVIDIA H100 80GB HBM3 MIG 3g.40gb |
| RKT | 9,881,943 | 9,881,943 | 1598.44 | NVIDIA H100 80GB HBM3 MIG 3g.40gb |

### Comparison

HiTSKT had the highest observed test ROC-AUC (0.732993). RKT had the shortest recorded wall time. DKVMN had the fewest trainable parameters.

- **HiTSKT** ranked 1 by ROC-AUC and recorded 0.804582 F1.
- **DKT** ranked 2 by ROC-AUC, with a signed difference of -0.014842 from HiTSKT. Its F1 was 0.810416.
- **DKVMN** ranked 3 by ROC-AUC, with a signed difference of -0.021118 from HiTSKT. Its F1 was 0.808166.
- **SAKT** ranked 4 by ROC-AUC, with a signed difference of -0.021922 from HiTSKT. Its F1 was 0.800667.
- **RKT** ranked 5 by ROC-AUC, with a signed difference of -0.034837 from HiTSKT. Its F1 was 0.802449.

HiTSKT test ROC-AUC by dataset was ASSIST2012 0.732993, ASSIST2009 0.771071, ALGEBRA2005 0.782119. The comparisons are descriptive because the cohorts, clocks, and splits differ.

These ranks describe a single seeded run on one cohort. The held-out partitions are later sessions from the same students. The benchmark does not estimate uncertainty across seeds or generalization to unseen students.

[Result CSV](assist2012_final_results.csv) · [Result JSON](assist2012_final_results.json) · [Validation loss](../report_graphs/assist2012_loss_vs_epoch.png) · [Validation ROC-AUC](../report_graphs/assist2012_validation_auc_vs_epoch.png) · [Trainable parameters](../report_graphs/assist2012_parameter_comparison.png)
<!-- RESULTS_END -->

## Reproduction

Download the complete publisher ZIP into `data/raw/assist2012`. Keep it and local checkpoints outside Git.

```bash
data=data/processed/assist2012/full
python -m scripts.preprocess_assist2012 data/raw/assist2012/2012-2013-data-with-predictions-4-final.zip "$data"
python -m scripts.prepare_rkt assist2012 "$data/session_store" "$data/rkt"
bash scripts/run_assist2012.sh
```
