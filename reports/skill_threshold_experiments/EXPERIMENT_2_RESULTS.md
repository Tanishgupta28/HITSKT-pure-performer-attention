# Experiment 2 · Weak, Intermediate and Strong thresholds

**Weak / Intermediate threshold (T1) = 61%**

**Intermediate / Strong threshold (T2) = 86%**

Scores below T1 are Weak. Scores from T1 inclusive to T2 exclusive are Intermediate.
Scores at or above T2 are Strong.

## Reference labels and selection

Future observed accuracy defines the validation and test reference labels. Accuracy below 60% is Weak, from 60%
inclusive to 85% exclusive is Intermediate, and at or above 85% is Strong. These reference cutoffs define the
actual-performance labels. T1 and T2 are fitted to the HiTSKT score scale.

The seed-42 hash split assigned 13,038 students and
108,290 pairs to threshold optimization. The holdout comprised 13,039
different students and 107,197 pairs. The learner ID sets were verified disjoint.
The grid contains 4,851 pairs in one-percentage-point increments from 1% to 99%.
Selection excludes candidates that produce an empty predicted class in validation.
Maximize validation macro-F1. Within 1e-12, maximize Weak recall, then balanced accuracy, then select the lexicographically lower threshold pair.
Validation macro-F1 was **0.575704**. The selected pair was written before held-out evaluation.

## Held-out evaluation

| Metric | Optimized (61% / 86%) | Baseline (60% / 85%) |
|---|---:|---:|
| Macro-F1 | 0.5715 | 0.5709 |
| Balanced accuracy | 0.5659 | 0.5651 |
| Accuracy | 0.5658 | 0.5648 |
| Weak precision | 0.6240 | 0.6359 |
| Weak recall | 0.5764 | 0.5577 |
| Intermediate F1 | 0.5453 | 0.5375 |
| Strong precision | 0.5933 | 0.5810 |
| Strong recall | 0.5482 | 0.5812 |

The optimized pair changed held-out macro-F1 by +0.05 percentage points and Weak recall by
+1.87 percentage points versus baseline. Predicted Weak, Intermediate, and Strong counts were
24,332, 50,497, 32,368. The application should account for the
observed recall of struggling skills when using the labels to allocate practice.

| Actual \ Predicted | Weak | Intermediate | Strong |
|---|---:|---:|---:|
| Weak | 15,184 | 10,064 | 1,096 |
| Intermediate | 7,496 | 26,261 | 12,069 |
| Strong | 1,652 | 14,172 | 19,203 |

The 95% student-bootstrap interval for held-out macro-F1 was 56.73% to 57.56%.
The paired macro-F1 change interval was -0.05 to +0.17 percentage points.
The 95% interval for held-out Weak recall was 56.50% to 58.84%.
Across 500 validation student resamples, the reoptimized T1 interval was
60.00% to 62.00% and T2 was 85.00% to 86.00%.
The original pair was reselected in 59.6% of resamples.

## Sensitivity to the follow-up minimum

The same frozen thresholds were applied to each holdout subset.

| Minimum future answers | Pairs | Optimized macro-F1 | Baseline macro-F1 | Optimized Weak recall | Baseline Weak recall |
|---:|---:|---:|---:|---:|---:|
| 5 | 107,197 | 0.5715 | 0.5709 | 0.5764 | 0.5577 |
| 10 | 32,514 | 0.6022 | 0.5982 | 0.6216 | 0.6041 |
| 20 | 7,244 | 0.5927 | 0.5861 | 0.6914 | 0.6750 |

- [Validation search results](threshold_search_results.csv) and [heatmap](figures/threshold_optimization.png)
- [Held-out confusion matrix](figures/held_out_confusion_matrix.png)
- [Classification boundaries](figures/classification_boundaries.png)
- [Frozen thresholds](optimized_thresholds.json), [full metrics](final_evaluation_metrics.json), and [classifier](skill_classifier.py)
