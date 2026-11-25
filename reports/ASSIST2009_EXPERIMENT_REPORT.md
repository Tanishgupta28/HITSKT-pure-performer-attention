# ASSIST2009 knowledge tracing benchmark

## Scope and dataset selection

ASSIST2009 was selected after examining the ASSIST2017 results. The historical HiTSKT checkpoint achieved a test ROC-AUC of 0.70962, behind RKT (0.74863) and SAKT (0.73847), and ahead of DKT (0.70700) and DKVMN (0.68225). ASSIST2009 is a follow-up dataset. Comparing the two datasets is descriptive because the cohorts, item catalogs, target definitions, and split units differ.

The source checkout is `main` at `ab08b68e827b995f0ae65ad3d60c230009d2c88d`. The historical HiTSKT is `ktbench.models.hitskt.HiTSKT` trained by `scripts/train_hitskt.py`, with `ktbench.models.performer` implementing causal ELU+1 Performer linear attention. The ASSIST2017 best checkpoint is format 1, epoch 40, with width 256, four heads, feed-forward width 2,048, one layer in each stage, and 6,931,457 parameters. Its validation AUC agrees with the historical CSV at 0.719757256068911. The checkpoint SHA-256 is `085c8d9ae32f4e60892dfb3e4eed93d90a584696d0914456c135dcab7b890164`. The architecture files at the current revision match the revision that published the historical results. [Historical results](https://github.com/Tanishgupta28/HITSKT-pure-performer-attention/blob/main/reports/final_results.csv)

## Dataset and preprocessing

The uploaded Skill Builder CSV contains ASSISTments mathematics interactions from the 2009–2010 school year. Its SHA-256 is `f22e3fb7872c1784ce93b0f9ebabbe0cbcac4f896fd8b4a11667b9715d77dbdc`. The correctness target is the publisher's first-attempt outcome, where requesting help counts as incorrect. [ASSISTments 2009–2010 Skill Builder release](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) · [Publisher data dictionary](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data)

| Dataset property | Count |
| --- | ---: |
| Source rows | 525,534 |
| Extra rows collapsed by `order_id` | 178,674 |
| Rows repeating an existing `order_id` and skill pair | 123,778 |
| Exact duplicate source rows | 0 |
| Conflicting event identity or correctness | 0 |
| Missing or invalid required event identifiers and outcomes | 0 |
| Rows missing `skill_id` | 66,326 |
| Invalid nonmissing skill IDs | 0 |
| Unique interactions | 346,860 |
| Students | 4,217 |
| Questions | 26,688 |
| Original atomic skill IDs | 123 |
| Interactions with multiple distinct skills | 47,037 |
| Interactions with no valid skill annotation | 63,755 |
| Model skill IDs, including the untagged category | 150 |

Aggregate each `order_id` into one target with its sorted set of observed skill tags. Encode each distinct set as a single composite category, following the historical preprocessing convention for single-ID model interfaces. Map the empty set to its own category and reserve zero for padding. Map students and questions by ascending source ID. The catalogs use categorical identities available in the full dataset and contain no aggregate outcomes. There were 2,383 validation targets and 2,905 test targets whose question ID had no training example. One skill category in each holdout had no training example. These categories therefore receive no direct training exposure. Ignore the source's other columns in model training, including opportunities, responses, hints and durations.

Sort each student's targets by ascending `order_id`, which the publisher documents as chronological. Reserve the final `floor(n/5)` interactions for test, the preceding `floor(n/5)` for validation, and the remaining prefix for training. The same arithmetic as the historical session split is applied to interactions here. The 473 students with fewer than five interactions contribute to training only. The dataset retains all 346,860 unique interactions, including first events with empty histories. The split is within-student. During validation and testing, earlier outcomes from the same student's observed sequence can be part of the context for a later prediction. Wall-clock answer availability is unverified. This is a retrospective, incremental prediction protocol.

| Partition | Targets | Share | Correct |
| --- | ---: | ---: | ---: |
| Train | 211,348 | 60.9318% | 136,979 |
| Validation | 67,756 | 19.5341% | 43,847 |
| Test | 67,756 | 19.5341% | 42,992 |

The independent data audit matched each processed event, label and tag set to its original records, verified all chronological partitions and inspected rolling targets and history masks for both supported history lengths. The official corrected release aligns on every `order_id`, with zero mismatches in user, question, and correctness. A clean preprocessing replay reproduced the arrays, metadata and mappings byte-for-byte. Hashes and catalog diagnostics are in [preprocessing statistics](datasets/assist2009/summary.json) and the [independent data audit](datasets/assist2009/validation.json).

## Chronology and session construction

The uploaded file and the publisher's corrected Skill Builder CSV and combined ZIP have no absolute event timestamp or session ID. The corrected file matches all 346,860 event IDs, student IDs, problem IDs, and outcomes. The combined CSV adds no clock field. `ms_first_response` and `overlap_time` describe within-problem durations. The publisher defines `order_id` as chronological, so the benchmark uses its ordering within each `user_id`. It provides no elapsed-time scale. `problem_id` identifies the problem, `assistment_id` groups a main problem with its scaffolds, `position` describes the assignment's placement in the class listing, and `opportunity` counts skill practice. These fields do not locate a wall-clock break. [Publisher data dictionary](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data) · [Publisher combined release](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/combined-dataset-2009-10) · [Release alignment and schema audit](datasets/assist2009/validation.json)

Each candidate was measured on the deduplicated rows. “Five-run retained” and “excluded” show the effect of the historical minimum as a sensitivity check. Under the chosen protocol all 4,217 students and 346,860 interactions are retained.

| Grouping rule | Groups | Mean length | Median length | Groups per student | Students with ≥5 | Five-run retained | Five-run excluded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Assignment runs | 75,829 | 4.57 | 3 | 17.98 | 2,239 | 321,777 | 25,083 |
| Sequence runs | 75,425 | 4.60 | 3 | 17.89 | 2,159 | 320,683 | 26,177 |
| Base-sequence runs | 73,129 | 4.74 | 3 | 17.34 | 2,141 | 320,173 | 26,687 |
| Assignment + sequence runs | 75,829 | 4.57 | 3 | 17.98 | 2,239 | 321,777 | 25,083 |
| Assignment-position runs | 75,531 | 4.59 | 3 | 17.91 | 2,233 | 321,633 | 25,227 |
| Class runs | 8,295 | 41.82 | 15 | 1.97 | 161 | 105,857 | 241,003 |
| Assistment runs | 283,060 | 1.23 | 1 | 67.12 | 3,680 | 345,281 | 1,579 |
| Problem runs | 346,837 | 1.00 | 1 | 82.25 | 3,744 | 345,752 | 1,108 |
| Global student + assignment | 66,239 | 5.24 | 3 | 15.71 | 2,161 | 318,248 | 28,612 |
| Global student + sequence | 59,874 | 5.79 | 4 | 14.20 | 2,060 | 316,580 | 30,280 |
| Global student + base sequence | 55,262 | 6.28 | 4 | 13.10 | 2,037 | 316,001 | 30,859 |

**Selected rule.** A proxy session is a maximal consecutive run of the same assignment within the student's `order_id` sequence. The next event for the same student begins a new run when `assignment_id` changes. Returning to an earlier assignment begins a separate run. All assignment and sequence identifiers were present and consistent across duplicate rows. The publisher defines the assignment as teacher or class specific and the sequence as the reusable problem-set content. Every assignment mapped to one sequence here, while 347 sequences were reused by multiple assignments. The base sequence can also merge copied content. Assignment and sequence together give identical boundaries. The assignment is the closest observed unit of task continuity. [Publisher data dictionary](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data) · [Skill Builder task semantics](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010)

Among 75,829 assignment runs, 9,590 reopen a prior student-assignment pair. Globally grouping by that pair would merge those runs, allowing events after an intervening assignment to enter a supposedly earlier context. Class runs are too coarse, and problem or assistment runs mostly describe individual items. For the selected assignment runs, the length percentiles (25th, 75th, 90th, 95th, 99th) are 1, 6, 10, 14, and 28 interactions. The longest run has 182 interactions. The full histograms, reopened groups, and cross-split counts are recorded for each candidate. A task switch can occur within one login, and an uninterrupted assignment ID can span multiple days. Results therefore measure assignment-based organization under recorded order. [Session audit](datasets/assist2009/session_audit.json)

### Shared split and HiTSKT input

The assignment run is a context boundary, independent of the original train, validation, and test cuts. Of the 75,829 runs, 4,787 intersect multiple partitions. For each intersected partition the target input is the run prefix ending at that partition's last event. Positions from earlier partitions are available to the causal decoder as unscored context. The loss and metric mask select only the requested partition. The 15 preceding assignment runs contribute hierarchical history. First runs use the existing session EOS as an empty-history representation, with zero prior action rows. This scores 1,181 first-run validation events and 861 first-run test events that the historical post-first-session filter would have omitted. The historical five-session filter would also remove 25,083 interactions. Omitting both cohort filters preserves exactly 211,348 training, 67,756 validation, and 67,756 test targets.

The collator handles the loss mask, event indices, and empty history. The model's action encoder, session encoder, correctness encoder, decoder, and causal ELU+1 Performer attention are unchanged. The CUDA smoke test exercised the empty-history forward/backward path and flipped the current scored answer in a split-crossing prefix while verifying that its prediction did not change. Validation and test replay independently check every scored event ID and label. The three original baseline runs are reused with the same split. Identical targets make predictive scores comparable, while each model retains its own history length and representation. HiTSKT additionally uses current assignment membership to structure its input. [HiTSKT assignment smoke](datasets/assist2009/hitskt_assignment_smoke.log)

### RKT timing ablation

A compatible release of absolute times could not be established. Later timestamped ASSISTments releases cover different school years. Recent work reconstructs cumulative durations from ASSIST2009, but accumulating within-question durations leaves inactivity unobserved. Chronological interaction distance is observable, but treating a one-question gap as one unit of forgetting would assume all steps span equivalent time. The original RKT paper explicitly reports a temporal-component ablation (TE, Table 5). We used that design choice on the repository's existing performance-only implementation, with the five-fold student-level cross-fitting preserved. [RKT paper, §3.3 and Table 5](https://arxiv.org/pdf/2008.12736) · [Cui et al., §5.1.1](https://weizhangltt.github.io/paper/TOIS23_KT.pdf)

**RKT (Phi-only)** uses `softmax_masked(Φ)` as the relation distribution. The historical implementation uses `softmax_masked(Φ + exp(-Δhours / S_u))`. The variant removes the learned per-student scale `S_u`, its initialization, and the time tensor. The removed embedding had one coefficient per student plus a reserved index-zero row, for 4,218 fewer trainable parameters. It keeps positional embeddings, the ordinary attention distribution `α`, and the learned mixture `λ·α + (1−λ)·softmax_masked(Φ)`. The training relation cache excludes the target student's entire fold. Validation and test use the cache built from all training folds. Each prediction sees the same 49-or-fewer preceding interactions used by historical RKT. The archived temporal RKT pathway and checkpoints stay compatible. [Phi cache provenance](datasets/assist2009/rkt_phi_cache.json)

## Experimental setup

Train with seed 42 and the ASSIST2017 optimizer settings and epoch ceilings. Use strict best validation ROC-AUC selection with patience 5 and minimum delta 0. Use binary cross-entropy with logits as loss, MSE on predicted probabilities, and a threshold of 0.5 for accuracy, precision, recall and F1. Mask padding. For each held-out target, the context contains only preceding interactions from that student. Test uses a freshly reloaded best checkpoint. Independently replay both held-out target sets in CUDA and verify the checkpoint metrics and parameter counts before export.

| Model | Sequence / history length | Adam learning rate | Batch size | Dropout | Epoch ceiling |
| --- | --- | ---: | ---: | ---: | ---: |
| DKT | 200 / 199 interactions | 0.0002 | 20 | 0.1 configured, no recurrent dropout in the one-layer LSTM | 200 |
| DKVMN | 200 / 199 interactions | 0.001 | 32 | 0.0 | 100 |
| SAKT | 100 / 99 interactions | 0.00001 | 10 | 0.2 | 300 |
| RKT (Phi-only) | 50 / 49 interactions | 0.001 | 128 | 0.1 | 300 |
| HiTSKT (assignment runs) | 16 / 15 sessions | 0.00005 | 64 maximum with token bucketing | 0.1 | 100 |

DKT uses one LSTM layer with 64-dimensional embeddings and hidden states. DKVMN uses 20 memory slots with 50-dimensional keys and 100-dimensional values. SAKT uses one 200-wide attention block with five heads and a relative position range of ten. Gradient clipping is disabled for DKT and set to 50 for DKVMN and 10 for SAKT. All three use zero weight decay. RKT (Phi-only) uses one 64-wide head, a training-only directed Phi cache with five student folds, and weight decay of 0.00001.

Training uses PyTorch CUDA 13.4 on an NVIDIA H100 80GB HBM3 MIG 3g.40gb partition with 40,320 MiB reported by PyTorch. Training used two loader workers and pinned memory. The rolling baselines use vectorized assembly and packed nonblocking transport while preserving minibatch boundaries and order. RKT uses vectorized rolling batches. HiTSKT retains the historical 32,768-token budget, batch maximum of 64, and bucket size of 512. Training uses FP32 with no AMP. For this dataset, DKT runs the same LSTM on right-padded inputs and gathers the state at the last real interaction. The CUDA smoke test compared logits and gradients with the original packed path, including an empty history. It passed within absolute tolerances of `5e-5` for logits and `1e-4` for gradients, with `1e-3` relative tolerance. Different cuDNN kernels cause floating-point differences. cuDNN TF32 was disabled for this DKT path. The historical DKT training path continues to use packed sequences.

The MIG environment reports NVML device-wide utilization as unsupported. Training logs record target throughput and peak CUDA allocation. [ASSIST2009 CUDA smoke log](datasets/assist2009/smoke.log) records actual CUDA kernel counts and durations. Full-run logs report epoch loss, AUC and CUDA memory. The profiler samples report the share of a short process window occupied by this process's kernels. They measure kernel activity over those eight minibatches, independently of GPU-wide utilization or SM occupancy.

## Results

<!-- RESULTS_START -->
Five models completed full GPU training and independent checkpoint replay on the same event targets. The original three ASSIST2009 baseline rows and checkpoints match their prior artifacts. Bold values show the best score for each metric.

| Model | Best epoch / completed | Best validation ROC-AUC | Loss ↓ | ROC-AUC ↑ | Accuracy ↑ | Precision ↑ | Recall ↑ | F1 ↑ | MSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HiTSKT | 24 / 29 | 0.774248 | **0.560811** | **0.771071** | **0.718637** | **0.778673** | 0.777587 | 0.778130 | **0.188051** |
| DKT | 6 / 11 | 0.756206 | 0.563010 | 0.744809 | 0.713929 | 0.730444 | **0.870325** | 0.794273 | 0.190140 |
| SAKT | 22 / 27 | 0.754787 | 0.566369 | 0.741588 | 0.715715 | 0.733471 | 0.867022 | **0.794674** | 0.191220 |
| DKVMN | 8 / 13 | 0.741817 | 0.583211 | 0.731075 | 0.692987 | 0.749124 | 0.776028 | 0.762339 | 0.199062 |
| RKT (Phi-only) | 6 / 11 | 0.740024 | 0.605862 | 0.729489 | 0.691673 | 0.762688 | 0.746278 | 0.754394 | 0.205396 |

### Model comparison

- **HiTSKT** led on test ROC-AUC, accuracy, and precision. It also had the lowest loss and MSE. Its recall and F1 trailed DKT and SAKT at the shared 0.5 threshold.
- **DKT** ranked second by ROC-AUC and had the highest recall.
- **SAKT** ranked third by ROC-AUC and had the highest F1.
- **DKVMN** had lower ROC-AUC than the other two reused baselines and higher precision than either.
- **RKT (Phi-only)** peaked in validation ROC-AUC at epoch 6. Training loss fell from 0.44767 to 0.37370 between epochs 6 and 11, while validation loss rose from 0.58633 to 0.72336.

HiTSKT’s ROC-AUC was 0.026262 higher than DKT’s, the strongest of the three reused baselines by ROC-AUC. These are single-seed point estimates.

### GPU performance

| Model | Trainable parameters | Runtime (s) | Runtime ÷ completed epochs (s) |
| --- | ---: | ---: | ---: |
| HiTSKT | 19,003,393 | 876.56 | 30.23 |
| DKT | 72,087 | 252.65 | 22.97 |
| SAKT | 2,885,801 | 2457.93 | 91.03 |
| DKVMN | 68,551 | 414.94 | 31.92 |
| RKT (Phi-only) | 1,743,874 | 107.32 | 9.76 |

HiTSKT reached 8.49 GiB of peak PyTorch allocation on the H100 MIG device.

RKT (Phi-only) reached 0.11 GiB of peak PyTorch allocation on the H100 MIG device.

Reported runtime includes full training, validation and final test evaluation. The new models trained serially on the NVIDIA H100 3g.40gb partition. CUDA allocator retries in the HiTSKT log recovered without a change in model, token budget or batch limit.

### ASSIST2017 context and limitations

| Model | ASSIST2017 test ROC-AUC | ASSIST2009 test ROC-AUC |
| --- | ---: | ---: |
| HiTSKT | 0.709624 | 0.771071 |
| DKT | 0.707000 | 0.744809 |
| SAKT | 0.738466 | 0.741588 |
| DKVMN | 0.682253 | 0.731075 |
| RKT (historical temporal / current Phi-only) | 0.748626 | 0.729489 |

Historical ASSIST2017 used inactivity-defined sessions and a session-based split. This ASSIST2009 follow-up uses task-based proxy sessions for HiTSKT, keeps the interaction split for all models, and removes RKT’s time-decay component. Differences in cohort, test targets, and available inputs limit cross-dataset conclusions. The original ASSIST2017 results motivated the choice of ASSIST2009. The results measure retrospective predictions within observed students under recorded event order. They leave unseen-student generalization and real-time feedback availability open.

### Artifacts

- ASSIST2009 results [CSV](assist2009_final_results.csv) · [JSON](assist2009_final_results.json)
- Combined benchmark [CSV](combined_benchmark_results.csv) · [JSON](combined_benchmark_results.json)
- Graphs [Validation loss](../report_graphs/assist2009_loss_vs_epoch.png) · [Validation ROC-AUC](../report_graphs/assist2009_validation_auc_vs_epoch.png) · [Parameters](../report_graphs/assist2009_parameter_comparison.png)
- [Delivery verification](datasets/assist2009/delivery_verification.json) · [Environment](datasets/assist2009/environment.json) · [Full test suite](datasets/assist2009/followup_tests.log)
- HiTSKT [epoch metrics](../experiments/hitskt/assist2009/full/metrics.csv) · [CUDA training log](datasets/assist2009/hitskt.log) · [independent replay](../experiments/hitskt/assist2009/full/independent_evaluation.json)
- DKT [epoch metrics](../experiments/dkt/assist2009/full/metrics.csv) · [CUDA training log](datasets/assist2009/dkt.log) · [independent replay](../experiments/dkt/assist2009/full/independent_evaluation.json)
- SAKT [epoch metrics](../experiments/sakt/assist2009/full/metrics.csv) · [CUDA training log](datasets/assist2009/sakt.log) · [independent replay](../experiments/sakt/assist2009/full/independent_evaluation.json)
- DKVMN [epoch metrics](../experiments/dkvmn/assist2009/full/metrics.csv) · [CUDA training log](datasets/assist2009/dkvmn.log) · [independent replay](../experiments/dkvmn/assist2009/full/independent_evaluation.json)
- RKT (Phi-only) [epoch metrics](../experiments/rkt/assist2009/full/metrics.csv) · [CUDA training log](datasets/assist2009/rkt_phi.log) · [independent replay](../experiments/rkt/assist2009/full/independent_evaluation.json)
<!-- RESULTS_END -->

## Reproduction

The published repository includes the metrics, configurations, and independent evaluation records. Raw data, processed caches, and model checkpoints are stored locally. The two plotting commands can also run directly on the published metrics.

For a full rerun, use a separate copy of the repository with PyTorch CUDA, the project dependencies, and the original CSV at the path below. Archive the published `experiments/<model>/assist2009/` directories in that copy first so the trainers have fresh output directories at the paths used by aggregation.

```bash
export PYTHONHASHSEED=42 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m ktbench.data.assist2009 dataset/skill_builder_data.csv data/processed/assist2009
python -m scripts.audit_assist2009_data dataset/skill_builder_data.csv data/processed/assist2009
python -m pytest -q
python -m scripts.smoke_assist2009 data/processed/assist2009 experiments
python -m scripts.smoke_assist2009 data/processed/assist2009 .inspection/assist2009_reference_fixtures --reference-fixtures

for model in dkt dkvmn sakt; do
  run="experiments/$model/assist2009/full"
  python -u -m scripts.train_baseline "$model" assist2009 data/processed/assist2009 "$run" --transport packed64 --workers 2
  python -m scripts.evaluate_checkpoint "$run" data/processed/assist2009 --workers 2 --output "$run/independent_evaluation.json"
done

sessions=data/processed/assist2009_assignment_runs
phi=data/processed/assist2009_phi
python -m ktbench.data.assist2009_sessions dataset/skill_builder_data.csv data/processed/assist2009 "$sessions"
python -m scripts.prepare_rkt assist2009 data/processed/assist2009 "$phi"
python -m scripts.smoke_assist2009_hitskt data/processed/assist2009 "$sessions"
python -m scripts.train_hitskt assist2009 data/processed/assist2009 experiments/hitskt/assist2009/full --assignment-root "$sessions" --workers 2
python -m scripts.evaluate_checkpoint experiments/hitskt/assist2009/full data/processed/assist2009 --assignment-root "$sessions" --workers 2 --output experiments/hitskt/assist2009/full/independent_evaluation.json
python -m scripts.smoke_assist2009_rkt data/processed/assist2009 "$phi"
python -m scripts.train_rkt assist2009 data/processed/assist2009 "$phi" experiments/rkt/assist2009/full --workers 2
python -m scripts.evaluate_checkpoint experiments/rkt/assist2009/full data/processed/assist2009 --prepared-root "$phi" --workers 2 --output experiments/rkt/assist2009/full/independent_evaluation.json

python -m scripts.aggregate_assist2009
python -m report_graphs.generate_comparison_graphs --assist2009 --replace-assist2009
python -m report_graphs.generate_parameter_comparison --assist2009 --replace-assist2009
```

Use a fresh output path for a reproduction. To resume an interrupted experiment at the latest completed epoch, rerun its training command with `--resume` and its original output path. The last checkpoint carries the optimizer, RNG and validation-patience state. If replay was interrupted after training completed, rerun the evaluation command. For the upstream schema and exact record-alignment audit, put the official corrected CSV and combined ZIP in `data/raw/assist2009` with the filenames `official_corrected_collapsed.csv` and `official_combined.zip`, then add `--published data/raw/assist2009` to the data audit command.
