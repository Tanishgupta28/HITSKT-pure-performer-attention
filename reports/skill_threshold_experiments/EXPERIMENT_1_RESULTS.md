# Experiment 1 · Fixed-history skill score validation

The mean score has empirical value for ranking and grouping Junyi skills. Use it as a noisy estimate of future performance and inspect the held-out class errors before using it to prioritize practice.

## Cohort and checkpoint

The original validation-selected epoch 32 HiTSKT was loaded without training. Its checkpoint, model configuration,
question and skill mappings, original benchmark and session-store hashes are in [experiment provenance](experiment_provenance.json).
The verified full-test export uses shifted real answers within target sessions. Archived predictions were inspected by column name and excluded from skill scoring because that feedback arrives after the fixed cutoff.

For each student, the cutoff was the last interaction before the first original test session. The model received
up to 15 complete sessions ending at that cutoff. Eligibility required at least four complete history sessions,
20 history interactions and five subsequent test answers for a composite skill. Students' histories and answer
timestamps were verified against the original store. The follow-up window ran through each student's last test
interaction. Its median across held-out student–skill pairs was 39.2 days.

Each processed Junyi composite skill has up to five distinct representative questions ranked by interaction frequency
in the original training split, with model question ID breaking frequency ties. The bank was defined without held-out
outcomes or future question frequencies. The archived original skill tags are a set associated with the composite.
A multi-tag question contributes to one composite score and keeps its original full tag set.
See [the question bank](representative_questions.csv).

Each question was scored at a separate first target position with BOS and the same encoded student history. The
original action and session encoders ran once per student. The original correctness encoder, decoder and prediction
head scored the independent probes. The fixed placeholder correctness affects the excluded EOS position. Direct
full-model forwards checked 21 probes with both placeholder values
in full FP32. Maximum probability difference was 1.49e-07, within
the predeclared 1e-05 tolerance. Original weights and source hashes matched before and after.

## Held-out performance

The cohort contains **26,077 students** and **215,487 pairs** across the
two student splits. This table uses only held-out students. Statistics give each pair equal weight except the Brier
score, which weights each future answer equally. The reliability diagram uses 10 fixed-width score bins.

| Minimum future answers | Students | Pairs | MAE | Pearson | Spearman | Response Brier | Calibration ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 13,039 | 107,197 | 0.1456 | 0.6370 | 0.6002 | 0.1944 | 0.0228 |
| 10 | 8,157 | 32,514 | 0.1417 | 0.7031 | 0.7034 | 0.2220 | 0.0690 |
| 20 | 3,202 | 7,244 | 0.1470 | 0.6963 | 0.6906 | 0.2350 | 0.0926 |

At the five-answer minimum, the mean forecast was 73.44% and the mean observed accuracy
was 73.55%. The pair-weighted calibration gap was 2.28%.
The training-wide constant predictor (71.01%) had MAE 0.2011 and
response-weighted Brier 0.2257.
The squared error against the observed pair accuracy was 0.0351.
Held-out score quantiles at 0%, 10%, 25%, 50%, 75%, 90%, and 100% were 0.043, 0.464, 0.628, 0.776, 0.880, 0.933, 0.995.

- [Predicted score versus later performance](figures/predicted_vs_actual.png)
- [Reliability diagram](figures/reliability_diagram.png)
- [Score distributions by observed class](figures/scores_by_actual_category.png)
- [All metrics and reliability-bin counts](experiment_1_metrics.json)
- Pair-level scores and question probabilities are retained on authorized research storage.

## Scope

The observed outcomes reflect the questions students later answered during the original test window, including repeated
questions. The model's representative question bank may differ from those future questions in difficulty and frequency.
At five answers, one response changes observed accuracy by 20 percentage points. The larger-count cohorts measure
sensitivity and select different student–skill pairs. Results apply most directly to learners and skills with enough
history and follow-up. Students were held out from threshold selection. The original HiTSKT checkpoint had seen
earlier training interactions from the same Junyi cohort.
