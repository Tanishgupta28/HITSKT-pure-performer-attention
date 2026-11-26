# Algebra I 2005–2006 knowledge tracing benchmark

## Data provenance and conversion

The KDD Cup 2010 Algebra I 2005–2006 development release covers Carnegie Learning Cognitive Tutor activity. The official ZIP has SHA-256 `a38d1fb55b139a38dfb019391417824855aef9104f51291d93fba614dacc9c84`. The ASCII, tab-delimited, 19-column training file contains 809,694 labeled step records. The master file supplies the 3,967 labeled records corresponding to the masked competition test set. Their row identifiers are disjoint from the training file. The two labeled files together contain the published total of 813,661 steps from 575 students. The uploaded two-column file matches the official submission template byte for byte, and all its 3,967 target values are blank. The preprocessing script verifies archive and member identity, schemas, field counts, coverage, and the alignment of the master supplement, test, and submission identifiers. [SIGKDD data and field definitions](https://kdd.org/kdd-cup/view/kdd-cup-2010-student-performance-evaluation/Data) · [Official archive](https://kdd.org/cupfiles/algebra_2005_2006.zip) · [Per-file hashes and audits](datasets/algebra2005/summary.json)

Each labeled source `Row` is one event. A step record aggregates transactions directed at the same step, including repeated attempts and hints. Distinct row IDs always remain distinct events for eligible students. `Correct First Attempt` is the observed binary outcome. All labeled rows have valid identifiers, outcomes and first-transaction times. `Problem Name` and `Step Name` form the documented step key. There were 211,397 distinct pairs and 216,355 distinct curriculum-path, problem, and step tuples in the full release. The benchmark includes the curriculum path in every key to keep instances assigned to different curriculum sections distinct. This assumes that curriculum-specific instances may present a different context even when their step strings match. The processed catalog contains 213,069 questions and the retained histories include 52,666 repeated student-question events.

Map anonymous students and question tuples lexicographically using one-based IDs. Split the default knowledge component field on `~~`, trim whitespace, and sort and deduplicate the tags within an interaction. Encode each distinct set as one category, with a dedicated untagged category and zero reserved for padding. The retained catalog has 112 atomic skills and 287 composite categories, including the untagged category. Of the retained events, 198,145 are untagged and 188,567 carry multiple tags. Opportunities, transaction counts and durations are excluded from model inputs.

## Chronology and split

Use `First Transaction Time` as the prediction clock at arrival of the first transaction, immediately before observing its first-attempt outcome. The publisher records times at second resolution, with an unspecified offset. Treat the timestamps as a common wall-clock axis for within-student intervals. All 813,661 first-transaction times parse, ranging from 2005-08-30 09:50:08 to 2006-06-07 11:12:53. Step start time is missing for 927 events and falls after the first transaction 13 times. Correct-transaction time is missing for 25,986 events and occasionally precedes the recorded first transaction. End time follows or equals first-transaction time in every row. There are 31,145 first-transaction reversals in source row order within students, so sort by student, first transaction, then source row. Neither step-completion field contributes to the models.

Start a session after a gap of at least 10 hours between first transactions. Keep students with at least five sessions. Assign each student's last `floor(sessions / 5)` sessions to test, the preceding equal number to validation, and the prefix to training. All five models score every event for eligible students, including first-session events. HiTSKT uses the existing session EOS for empty hierarchical history, as in the ASSIST2009 benchmark. Earlier validation or test outcomes enter histories only after their corresponding first transactions. The protocol measures prediction at transaction arrival, with the logged step and recorded KC available then. It does not measure pre-step prediction from the opening of an interface or a global calendar holdout.

| Partition | Sessions | Interactions | Interaction share | Correct |
| --- | ---: | ---: | ---: | ---: |
| Train | 8,003 | 507,053 | 63.5894% | 388,011 |
| Validation | 2,384 | 149,361 | 18.7313% | 114,572 |
| Test | 2,384 | 140,972 | 17.6793% | 108,265 |
| Total | 12,771 | 797,386 | 100% | 610,848 |

The source has 13,033 sessions. The five-session eligibility rule excludes 115 students and 16,275 events, leaving 460 students. There are 2,201 additional retained events at timestamps already occupied by another event for the same student. All are scored. DKT, DKVMN, SAKT, and RKT end each history immediately before the timestamp group. HiTSKT scores the first event in its main causal sequence and scores each other simultaneous event using a separate prefix that contains the strictly earlier current-session events followed by that event. Later timestamps can use every response from the tied group. This leaves the encoder, decoder, and Pure Performer attention unchanged. Historical sessions have a strict 10-hour or longer gap from the next session's first event. [Full target coverage and chronology audit](datasets/algebra2005/data_verification.json)

## RKT and comparison with ASSIST2009

RKT uses directed Phi relationships and learned time decay in hours. The five-fold student-level cross-fit excludes the target student's fold during training. Validation and testing use Phi estimated from all training folds. The cache excludes ties from prior-event pairs and uses the latest occurrence of each prior question within the historical 49-event window. Memory-strength initialization uses positive adjacent gaps in each student's training prefix, including gaps between training sessions. The release contains usable first-transaction clocks, allowing observed sessions and elapsed times where ASSIST2009 used assignment runs, interaction splits and Phi-only RKT. [RKT cache provenance](datasets/algebra2005/rkt_provenance.json)

## Model and evaluation configuration

All five runs use seed 42, Adam, full eligible data, binary cross-entropy with logits, and strict validation ROC-AUC improvement with patience five and minimum delta zero. Evaluation uses the best validation checkpoint. Accuracy, precision, recall and F1 threshold probabilities at 0.5. MSE uses probabilities. Epoch ceilings and architecture widths follow the existing ASSIST2009 and historical benchmark configurations.

| Model | Context / history | Learning rate | Batch size | Dropout | Epoch ceiling |
| --- | --- | ---: | ---: | ---: | ---: |
| DKT | 200 / 199 interactions | 0.0002 | 20 | 0.1 configured | 200 |
| DKVMN | 200 / 199 interactions | 0.001 | 32 | 0.0 | 100 |
| SAKT | 100 / 99 interactions | 0.00001 | 10 | 0.2 | 300 |
| RKT | 50 / 49 interactions | 0.001 | 128 | 0.1 | 300 |
| HiTSKT | 16 / 15 sessions | 0.00005 | 64 maximum | 0.1 | 100 |

DKT uses a one-layer 64-wide LSTM and no effective recurrent dropout. The dense CUDA pathway gathers the last actual history state. DKVMN uses 20 memory slots with 50-dimensional keys and 100-dimensional values. SAKT uses one 200-wide attention block with five heads and relative position range ten. Temporal RKT uses one 64-wide head and weight decay 0.00001. HiTSKT uses width 256, four heads, feed-forward width 2,048 and one layer per stage. Dynamic HiTSKT batches cap padded tokens at 32,768, except when one lossless example alone exceeds the token budget. The model uses the original causal ELU+1 Performer. The rolling baselines use vectorized packed batch transport without reordering minibatches or combining optimizer updates.

## Results

<!-- RESULTS_START -->
All five full GPU runs completed and independent checkpoint replay matched the recorded metrics.

| Model | Best / completed epoch | Validation ROC-AUC | Loss ↓ | ROC-AUC ↑ | Accuracy ↑ | Precision ↑ | Recall ↑ | F1 ↑ | MSE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DKT | 9 / 14 | 0.807782 | **0.442531** | **0.784435** | 0.800450 | 0.832197 | 0.927105 | 0.877091 | **0.142055** |
| HiTSKT | 26 / 31 | 0.799176 | 0.450419 | 0.782119 | 0.797910 | **0.839715** | 0.910691 | 0.873764 | 0.144049 |
| SAKT | 27 / 32 | 0.810693 | 0.455869 | 0.780057 | 0.791271 | 0.831757 | 0.912862 | 0.870424 | 0.145925 |
| DKVMN | 10 / 15 | 0.798373 | 0.458062 | 0.771697 | 0.788015 | 0.823165 | 0.922052 | 0.869807 | 0.148223 |
| RKT | 5 / 10 | 0.776141 | 0.494682 | 0.758264 | **0.804330** | 0.822892 | **0.949596** | **0.881715** | 0.148367 |

### Metric leaders

- **Loss** — DKT (0.442531)
- **ROC-AUC** — DKT (0.784435)
- **Accuracy** — RKT (0.804330)
- **Precision** — HiTSKT (0.839715)
- **Recall** — RKT (0.949596)
- **F1** — RKT (0.881715)
- **MSE** — DKT (0.142055)

### Parameter counts and runtime

| Model | Total parameters | Trainable parameters | Runtime (seconds) | GPU |
| --- | ---: | ---: | ---: | --- |
| DKT | 107,296 | 107,296 | 999.88 | NVIDIA H100 80GB HBM3 MIG 3g.40gb |
| HiTSKT | 114,500,609 | 114,500,609 | 1230.29 | NVIDIA H100 80GB HBM3 MIG 3g.40gb |
| SAKT | 21,537,601 | 21,537,601 | 7064.66 | NVIDIA H100 80GB HBM3 MIG 3g.40gb |
| DKVMN | 102,801 | 102,801 | 1386.29 | NVIDIA H100 80GB HBM3 MIG 3g.40gb |
| RKT | 13,672,719 | 13,672,719 | 367.61 | NVIDIA H100 80GB HBM3 MIG 3g.40gb |

### Comparison

DKT had the highest observed test ROC-AUC (0.784435). RKT had the shortest recorded wall time. DKVMN had the fewest trainable parameters.

- **DKT** ranked 1 by ROC-AUC, with a signed difference of +0.002316 from HiTSKT. Its F1 was 0.877091.
- **HiTSKT** ranked 2 by ROC-AUC and recorded 0.873764 F1.
- **SAKT** ranked 3 by ROC-AUC, with a signed difference of -0.002062 from HiTSKT. Its F1 was 0.870424.
- **DKVMN** ranked 4 by ROC-AUC, with a signed difference of -0.010422 from HiTSKT. Its F1 was 0.869807.
- **RKT** ranked 5 by ROC-AUC, with a signed difference of -0.023855 from HiTSKT. Its F1 was 0.881715.

These ranks describe a single seeded run on one cohort. The held-out partitions are later sessions from the same students. The benchmark does not estimate uncertainty across seeds or generalization to unseen students.

[Result CSV](algebra2005_final_results.csv) · [Result JSON](algebra2005_final_results.json) · [Validation loss](../report_graphs/algebra2005_loss_vs_epoch.png) · [Validation ROC-AUC](../report_graphs/algebra2005_validation_auc_vs_epoch.png) · [Trainable parameters](../report_graphs/algebra2005_parameter_comparison.png)
<!-- RESULTS_END -->

## Reproduction

From the repository root, obtain the official archive and run the preprocessing, target audit and RKT preparation. Use a CUDA device for the smoke and full runner.

```bash
mkdir -p data/raw/algebra2005
curl -fL https://kdd.org/cupfiles/algebra_2005_2006.zip \
  -o data/raw/algebra2005/algebra_2005_2006.zip
data=data/processed/algebra2005/full
python -m scripts.preprocess_algebra2005 data/raw/algebra2005/algebra_2005_2006.zip "$data"
python -m scripts.verify_algebra2005_data "$data"
python -m scripts.prepare_rkt algebra2005 "$data/session_store" "$data/rkt"
bash scripts/run_algebra2005.sh
```

Model checkpoints, optimizer state and resume state are local. The run artifacts and figures are produced only from full training and independently replayed checkpoints.
