# Modified HiTSKT: supervisor verification brief

19 October 2026 · Evidence baseline: **e1065c9** · Primary training seed: **42**

**Decision:** retain the three historical experiments as provisional evidence.
Further verification is needed before presenting the complete metric sets as
reproduced. No demonstrated training defect currently requires retraining.
**Publication readiness: NOT VERIFIED.**

The model is **hierarchical HiTSKT with deterministic causal ELU+1 linear
attention**. It is a modified model, **not genuine FAVOR+ Performer** and not
an exact replication of every original-paper component.

## Three test results: distinguish history from independent replay

| Dataset | Historical CUDA AUC | Independent full CPU AUC | Masking / accepted data | Fresh CUDA replay | Next step |
| --- | --- | --- | --- | --- | --- |
| ASSIST2017 | 0.7096240159197755 | 0.7096236372142970 | PASS / PASS | NOT VERIFIED | Verify further; no retraining yet |
| Junyi | 0.7978262055191263 | 0.7978264739233538 | PASS / PASS | NOT VERIFIED | Verify further; no retraining yet |
| EdNet-KT1 | 0.7693648364413692 | 0.7693647361509419 | PASS / PASS | NOT VERIFIED | Verify further; no retraining yet |

All three AUC differences pass the unchanged **absolute 1e-6** acceptance
tolerance. **All three complete CPU metric sets FAIL that gate.** ASSIST fails
accuracy, precision, recall, F1 and BCE; Junyi fails accuracy, precision, recall
and F1; EdNet fails accuracy, recall and F1. These failures have not been
relabelled as rounding or hidden behind AUC agreement.

CPU test coverage is **112,252 / 2,456,402 / 13,429,870 targets**, using the
validation-selected checkpoints at epochs **40 / 32 / 30**, respectively.
The historical CUDA prediction vectors and actual precision flags were not
saved. Exact original CUDA-versus-CPU pointwise errors and classification-flip
counts therefore remain unverified. Full float64 ASSIST inference still misses
historical BCE by **2.13e-6**; tested CPU precision changes do not resolve it.

## What the independent checks establish

**Masking PASS:** future-response and own-target perturbations, independent
explicit causal attention outputs and gradients, target shifting, BOS/EOS,
loss masking, student/batch isolation and session boundaries pass within the
production collator's right-padding contract. Arbitrary externally supplied
left/internal padding is outside the full-model guarantee. Sequence causality
does not establish when an earlier response became available in real time.

**Accepted-data reconstruction PASS:** every accepted source and processed/store
event was checked for mappings, sessions, student boundaries, chronological
splits, label alignment and intended target coverage; no intended supervised
target is duplicated or missing. These are exhaustive checks, not samples.

| Dataset | Accepted source rows | Retained learners | Retained events | Sessions |
| --- | --- | --- | --- | --- |
| ASSIST2017 | 942,807 | 1,388 | 885,335 | 12,402 |
| Junyi | 14,660,217 | 29,865 | 14,660,217 | 600,154 |
| Full EdNet-KT1 | 95,293,926 | 116,548 | 81,940,867 | 2,577,988 |

ASSIST/Junyi inputs match the authors' distributed CSV bytes; their original
publisher-to-author transformation remains **NOT VERIFIED**. EdNet checks cover
all **784,309 student files**, ZIP CRCs and official archive byte identity.
HiTSKT deliberately uses the first training session as context only.

<!-- PAGEBREAK -->

## Evaluation meaning and response-timing limits

The existing protocol is **chronological within-student, retrospective and
teacher-forced**: fixed weights predict each target using earlier indexed
responses, including earlier validation/test responses. It is **not
student-disjoint**, a global-calendar holdout or a certified real-time
presentation-time evaluation. An earlier event index does not prove that its
answer or correctness feedback had arrived.

| Dataset | Test-set timing observation | Scientific implication |
| --- | --- | --- |
| ASSIST2017 | 7,059 targets have an exposed earlier recorded end after current presentation. | Problem/step completion does not establish correctness release. Actual availability NOT VERIFIED. |
| Junyi | 2,165,594 targets have a tied-start prior label; all source rows have start=end. | Zero recorded overlaps cannot certify instantaneous feedback. Actual availability NOT VERIFIED. |
| EdNet-KT1 | 4,089,385 targets have a prior completion-proxy overlap; 3,364,799 expose a prior same-episode label. | KT1 lacks final submission/feedback events. Pre-bundle prediction claims are not established. |

These observations are **not counts of confirmed leakage** or estimates of
metric inflation. Response-release timing needs separate evidence. Do not
silently substitute availability-aware V2 or implement it as part of this report.

## Baseline comparison: historical scores, not fresh reproductions

All entries below are **historical stored test AUCs**, rounded to nine decimals.
**None of the 11 completed baseline metric sets has an independent fresh replay
in these audits.** Checkpoint hashes and validation selection pass artifact
checks; this is not metric reproduction.

| Dataset | DKT | DKVMN | SAKT | RKT |
| --- | --- | --- | --- | --- |
| ASSIST2017 | 0.706999742 | 0.682252702 | 0.738465565 | 0.748626490 |
| Junyi | 0.747684150 | 0.741578462 | 0.788796178 | 0.791254849 |
| EdNet-KT1 | MISSING | 0.674105992 | 0.763007857 | 0.747547020 |

Recorded HiTSKT AUC is below the strongest ASSIST baseline and above the best
available Junyi/EdNet baselines by **0.006571357 / 0.006356979**. These are
single-seed descriptive differences, **not demonstrated significant gains**.
Features, history budgets and training target populations differ. Correct saved
history windows are DKT/DKVMN **199**, SAKT **99**, RKT **49**; this corrects a
descriptive error in the earlier report. RKT raw relation counts still need an
independent reconstruction. DKT/EdNet has no final result; its last optimizer
contains **21,749 positive infinities**. Do not resume/reset it.

## Minimum remaining verification, in priority order

1. **Close the HiTSKT metric discrepancy:** on a genuinely idle independent
   GPU, replay all three unchanged best checkpoints with original batches and
   settings. Save runtime flags, hashes, local predictions and all seven metrics;
   apply the original 1e-6 gate and report failures without retuning tolerance.
2. **Verify comparisons:** independently replay completed baseline checkpoints,
   prioritizing RKT/ASSIST, RKT/Junyi and SAKT/EdNet; independently reconstruct
   RKT's training-only relation counts. Keep missing DKT/EdNet missing.
3. **Freeze defensible claims:** retain the retrospective within-student scope
   and disclose timing/provenance limits. Real-time, student-disjoint,
   FAVOR+ or statistically significant-superiority claims need separate evidence.
   V2, FAVOR+ and additional training require an explicit research decision.

The only visible H100 MIG remains occupied by protected training; **no CUDA
replay was attempted**. On 19 October, **59 relevant controls passed again**
and all full cached CPU metrics recalculated identically. Original artifacts
and prior evidence remain preserved. See the
[verification status and replay plan](HITSKT_VERIFICATION_STATUS.md) for hashes,
acceptance rules, detailed scopes and the resource gate.

Evidence: [numerical diagnosis](https://github.com/Tanishgupta28/HITSKT-pure-performer-attention/blob/e1065c9/reports/HITSKT_REPRODUCIBILITY_ROOT_CAUSE.md),
[timing analysis](https://github.com/Tanishgupta28/HITSKT-pure-performer-attention/blob/e1065c9/reports/HITSKT_RESPONSE_TIMING_ANALYSIS.md),
[research decisions](https://github.com/Tanishgupta28/HITSKT-pure-performer-attention/blob/e1065c9/reports/HITSKT_RESEARCH_DECISION_REPORT.md).
