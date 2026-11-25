# Predicting weak topics with a Gaussian mixture and a random forest

The study asks whether a learner’s practice history can improve pre-assessment recommendations beyond ordering topics by their HiTSKT scores. The unit of analysis is one student on one Junyi composite skill (the full tag set of a question). A single student can contribute multiple skills. This is a new experiment. The earlier 300-student GMM pilot used a one-feature forest and a separate random split.

## What the recommender learns

1. **Learn a target from later performance.** For every fitting pair, divide later correct answers by later attempts. Standardize these accuracies and fit a three-component Gaussian mixture with five initializations, covariance regularization 0.01 and seed 42. Name the component with the lowest mean accuracy the weak component. Keep each pair’s probability of membership in this component as a continuous training target. The GMM sees only fitting learners. Every pair has equal weight.
2. **Predict that target from earlier history.** Train a random-forest regressor with 200 trees, at least 25 pairs per leaf and seed 42. It estimates the GMM membership score using four features: HiTSKT mean probability of correctness on up to five training-ranked representative questions, number of previous attempts on the skill, fraction of those attempts answered correctly, and days since the last attempt on that skill measured at the final history interaction. Use the last 15 complete history sessions at most. Times tied in the source keep their original order. A skill absent from that window gets zero attempts and a sentinel of −1 for its accuracy and recency. IDs, later answer counts and later accuracy are excluded from forest inputs.
3. **Decide what to flag.** On separate development learners, try risk thresholds from 0.01 to 0.99 in steps of 0.01. Select the threshold maximizing weak-class F1 against the original project definition of weak (later accuracy strictly below 60%). Break ties by higher weak recall and then by lower threshold. Keep the trained forest and selected threshold fixed for testing. The predicted number estimates membership in the GMM component; the study does not establish it as a calibrated probability of later falling below 60%.
4. **Build the list.** Include risks at or above the selected threshold and sort by decreasing risk. Return up to three skills. For the baseline, include HiTSKT scores strictly below 0.61 and sort by increasing score. Both methods break exact ties by increasing composite skill ID. The original 0.86 boundary distinguishes Intermediate from Strong and has no effect on weak-topic lists.

## Which data were used

Both methods are evaluated on exactly the original held-out students and pairs. Freeze history immediately before each learner’s first original test session. Require four or more earlier sessions, 20 or more history interactions, and at least five later answers on each evaluable skill. This last rule permits retrospective scoring of recommendations. At runtime, the later responses are unknown.

The eligible cohort contains 26,077 students and 215,487 pairs. Sort processed student IDs by SHA-256 of `42:<id>`. Use the first 13,038 students (108,290 pairs) for development and the remaining 13,039 students (107,197 pairs) for the original holdout. Within development, sort by SHA-256 of `73:<id>` and use the first 80% of students, rounded down, to fit both models. Use the rest to select the flag threshold. See verification.json for exact split counts, unseen skills within the history window, replay hashes, component means and the selected cutoff. The cohort separation applies to the new recommender; the study reuses the existing HiTSKT checkpoint.

## How success is measured

* **Hit Rate@3** is the fraction of learners whose list includes at least one skill with later accuracy below 60%. **NDCG@3** also rewards placing those skills earlier. Gain is binary, discounted by 1/log2(rank+1), and divided by the gain of an ideal top-three list drawn from all the learner’s eligible skills. Both metrics average over the 7,702 held-out learners with at least one observed weak skill. An empty list scores zero. The 5,337 learners with no observed weak skills still contribute to pair-level classification.
* **Weak precision, recall and F1** compare every flagged pair with its later outcome, before truncating lists at three. Average precision summarizes the ordering of all held-out pairs across possible cutoffs. The counts of flags and learners reached show how much extra study a more permissive rule could assign.
* **ranking_results.csv** gives the primary, flag-filtered top-three results. Each 95% interval uses 2,000 learner bootstrap samples with seed 42. The difference rows resample the same learners for both methods. These intervals describe uncertainty on this holdout with both models fixed.
* **ranking_without_flags.csv** removes the flag cutoffs and ranks all eligible skills. This comparison gives the two methods equal list lengths per learner, separating how well they order topics from how often they issue a recommendation.

In a paper-specific application, the candidate list would be restricted to topics on that paper for which the learner has prior attempts. The retrospective benchmark uses all skills satisfying the original follow-up rule, including skills absent from the 15-session window. It measures weak-topic identification and ordering. The records do not measure the quality of study links or learning after a recommendation.

## Reproducing the result

Use the Junyi source and checkpoint identified in verification.json and the original skill_threshold_experiments.zip. The research checkout used for model code is commit `6094077e0863f5e54dc0839d9e29802c2ffac285` of this repository. Inputs include raw Junyi.csv, the original junyi_hitskt_full_test_predictions.csv, the real epoch-32 checkpoint and the published mappings. The archived question predictions are not the per-skill model input. All intermediate CSVs belong in a local working directory. The commands below use example paths, run from this report folder, with an existing local work directory.

```text
python replay_development.py --repo RESEARCH_CHECKOUT --raw Junyi.csv --export junyi_hitskt_full_test_predictions.csv --archive skill_threshold_experiments.zip --scores work/development_scores.csv
python ../Junyi_original_ranking/replay_scores.py --repo RESEARCH_CHECKOUT --raw Junyi.csv --export junyi_hitskt_full_test_predictions.csv --archive skill_threshold_experiments.zip --scores work/heldout_scores.csv
python history_features.py --repo RESEARCH_CHECKOUT --raw Junyi.csv --export junyi_hitskt_full_test_predictions.csv --output work/history_features.csv
python evaluate.py --development work/development_scores.csv --heldout work/heldout_scores.csv --history work/history_features.csv --output .
```

The evaluation checks input score hashes, development and held-out memberships, pair uniqueness and checkpoint identity, and reproduces the published baseline weak-topic hits and NDCG. Python 3.12 with NumPy, pandas, torch, scikit-learn, matplotlib, threadpoolctl and the research checkout’s model dependencies is required.
