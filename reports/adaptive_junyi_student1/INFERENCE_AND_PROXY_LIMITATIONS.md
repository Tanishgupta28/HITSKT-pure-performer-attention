# Junyi inference and proxy limitations

## What the report measures

The actual paper score is graded from the student's selected options. Observed skill accuracy is the proportion correct among the recorded attempts attributed to that skill. HiTSKT's estimate is the mean predicted correctness on verified representative Junyi questions for an original tag. It uses verified question and skill IDs and legally observed outcomes. The custom Lumen question text is outside the model input.

The report shows original-only estimates separately from estimates that include Lumen outcomes through a representative mapping. A change between those columns reflects a different evidence history. Scores can move in either direction and should be interpreted alongside the observed attempts. This single learner's sequence establishes that the integration works. A study of effectiveness or calibration would require additional evaluation.

## Where the inference is limited

- The original research thresholds were evaluated on fixed-history composite skills. Lumen uses 0.61 and 0.86 as prototype thresholds for original-tag mean estimates and for histories containing proxies. Unattempted is a separate evidence status. An unavailable score stays empty with a reason.
- The proxy transfers the genuine grade to the verified representative's model IDs. The original answer belonging to that representative is never carried over. Where a required context has an unmapped event, the grade remains saved and the estimate is unavailable.
- The original 10-hour inactivity rule determines model sessions. A new paper by itself does not start a session. The forward pass receives at most 15 earlier complete sessions and the observed prefix. The 8,192-token CPU serving budget can leave an oversized single context unavailable rather than truncating it.
- The verified bank limits coverage. For example, there are nine Senior items, so a 20-question Senior-only paper correctly reports shortage. Requested slot quotas can be reallocated under scarcity, with the final counts saved on the paper.
- Gemini is optional. FastAPI can select from verified bank IDs using deterministic rules. Study links come from a limited public English catalogue. Google Search grounding results are excluded from the persistent checklist. A topic without a verified page remains pending for retry or an audited operator action. [Gemini API grounding terms](https://aistudio.google.com/docs/terms#grounding-with-google-search)

## Verification scope

The genuine audit covered three completed papers, 60 graded answers, the authorized baseline and the locked fourth paper. The separate automated fixture exercised submissions, retries, failure handling and progression. Test 4 remains available for the learner's own completion. [Verification record](TEST_RESULTS.md)
