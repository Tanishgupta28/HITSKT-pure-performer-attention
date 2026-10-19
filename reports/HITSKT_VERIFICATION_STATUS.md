# Modified HiTSKT verification status

As of 19 October 2026 UTC. Continued from
`e1065c9c6215f9c676b43e33ee0d5f689bbe3fbb`.
The [supervisor brief](HITSKT_SUPERVISOR_BRIEF.md) is the concise editable
source for the [PDF](HITSKT_SUPERVISOR_BRIEF.pdf). The detailed operational
[replay plan](HITSKT_CHECKPOINT_REPLAY_PLAN.md) is a plan, not a completed run.

**Decision:** all three historical results remain provisional retrospective
results requiring further verification. No established requirement to retrain
any original HiTSKT model follows from the observed numerical discrepancies.
**Publication readiness remains NOT VERIFIED.** All scientific originals and
prior audit evidence are preserved; this document adds evidence and one
explicit correction to an earlier descriptive baseline summary.

## Meaning of each status

- **PASS / VERIFIED:** a named check passed within its specified input and
  numerical scope. It is not a universal proof or a claim that every other
  check passed.
- **FAIL:** the declared criterion was actually tested and not met. In
  particular, a small historical-versus-CPU mismatch exceeding the frozen
  tolerance remains a failure.
- **NOT VERIFIED:** the necessary evidence or execution is absent. GPU
  inference deferred to protect active training is not a failed CUDA replay
  and not a successful reproduction.

## Dataset-level decisions and AUC provenance

Historical values come from each original `final_results.json`. CPU values
come from the original independent full replay manifests, not a rounded
substitute for the CUDA values. No historical score is overwritten.

| Dataset | Historical CUDA test AUC | Independent full CPU test AUC | Absolute difference | AUC gate |
| --- | --- | --- | --- | --- |
| ASSIST2017 | 0.7096240159197755 | 0.7096236372142970 | 3.787054785320265e-7 | PASS |
| Junyi | 0.7978262055191263 | 0.7978264739233538 | 2.6840422751028825e-7 | PASS |
| EdNet-KT1 | 0.7693648364413692 | 0.7693647361509419 | 1.002904272962013e-7 | PASS |

The acceptance criterion remains **absolute error<=1e-6 for every required
metric**, with exact equality recorded separately. AUC agreement alone does
not satisfy the complete-metric gate.

| Check | ASSIST2017 | Junyi | EdNet-KT1 |
| --- | --- | --- | --- |
| Masking / no own or future indexed response, production collator contract | PASS | PASS | PASS |
| Exhaustive accepted-source → processed → store reconstruction | PASS | PASS | PASS |
| Student boundaries, within-student split chronology, rolling history and intended target coverage | PASS | PASS | PASS |
| Untouched publisher source → authors' transformed CSV provenance | NOT VERIFIED | NOT VERIFIED | Not applicable; official archive identity/full reconstruction PASS |
| Original best hash, validation-selected epoch and full CPU target count | PASS | PASS | PASS |
| Historical-versus-CPU AUC at1e-6 | PASS | PASS | PASS |
| Historical-versus-CPU complete metric set at1e-6 | FAIL | FAIL | FAIL |
| Fresh CUDA checkpoint replay | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| Actual correctness-response release before prediction | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| Fresh full completed-baseline metric replays in these audits | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED; DKT also unfinished |
| Final recommendation | Further verification; no retraining yet | Further verification; no retraining yet | Further verification; no retraining yet |

Failed metric names are accuracy/precision/recall/F1/BCE for ASSIST,
accuracy/precision/recall/F1 for Junyi, and accuracy/recall/F1 for EdNet.
MSE passes on all three; EdNet precision and Junyi/EdNet BCE pass. Both
historical FP32 batch-weighted BCE and independent stable float64 logit BCE
are reported in [the numerical report](HITSKT_REPRODUCIBILITY_ROOT_CAUSE.md).
No metric is selected for display merely because it passes.

| Dataset | Test targets | Selected epoch | Original best SHA-256 |
| --- | --- | --- | --- |
| ASSIST2017 | 112,252 | 40 | `085c8d9ae32f4e60892dfb3e4eed93d90a584696d0914456c135dcab7b890164` |
| Junyi | 2,456,402 | 32 | `2308c5bf0c850da1e55daf802a0853a30fac45eeed071c7162c68fc17bfa3bdd` |
| EdNet-KT1 | 13,429,870 | 30 | `58d4b1f393d0fa8ec372ba8bcc00cc8b7168d37817d4badfd26ee42abffd03c8` |

All original configs, result rows, checkpoint bindings, full precision values
and repeat-calculation outcomes are captured in
[supervisor_facts.json](supervisor_verification/supervisor_facts.json).

## Verified model and data scope

The model is **hierarchical HiTSKT with deterministic causal ELU+1 linear
attention**, with the preserved hierarchy, masks, shifts and kernel. No random
feature matrix approximates the softmax kernel. Calling these historical
results genuine FAVOR+ is incorrect. No architecture change, V2 implementation
or FAVOR+ implementation is part of this verification.

Independent explicit-prefix output/gradient references, future-response and
own-label perturbations, EOS/PAD/loss controls, batch/student isolation and
session-boundary tests support the causal-index claim. The full model requires
the production collator's contiguous right padding; attention-level hole tests
do not extend that full-model guarantee to arbitrary external padding.

The completed exhaustive data checks cover all accepted rows and all retained
events, not just a random subset. They establish mappings, session construction,
per-student chronology, supervision alignment, lack of duplicate/missing
intended targets and exact processed/store field equality. ASSIST/Junyi
accepted sources match the authors' distributed LFS objects; untouched
publisher-to-author transforms remain unverified. All784,309 EdNet members and
95,293,926 raw rows were checked, including CRCs and independently downloaded
official archive byte identity. This does not claim a publisher-signed
historical checksum.

| Dataset | Accepted source rows | Retained learners / events / sessions |
| --- | --- | --- |
| ASSIST2017 | 942,807 | 1,388 / 885,335 / 12,402 |
| Junyi | 14,660,217 | 29,865 / 14,660,217 / 600,154 |
| EdNet-KT1 | 95,293,926 | 116,548 / 81,940,867 / 2,577,988 |

Sessions use start-to-start gaps>=10h; retained learners have at least5 sessions.
Validation/test each receive floor(n/5) sessions, with the earlier remainder
used for training. This is an explicitly documented protocol variant, not an
exact end-to-start original-paper reproduction. It is chronological
**within-student**, not student-disjoint or a common global-calendar cutoff.
First-session training events are context-only for HiTSKT, while baselines
supervise them; held-out target populations match. Skill construction uses
label-free full-source metadata, not train-only inductive vocabulary discovery.

Embedding capacities are maximum IDs, not distinct retained counts. True
distinct question/skill counts are3,112/102,25,630/1,521 and12,283/1,490;
the saved capacity indices remain unchanged. See the preserved
[dataset audit](HITSKT_INDEPENDENT_CORRECTNESS_AUDIT.md) for all reconstruction
manifests and [vocabulary evidence](independent_correctness_audit/checkpoint_vocabulary_consistency.json).

## Retrospective evaluation and timing

Weights remain fixed during evaluation. Earlier indexed response labels enter
histories, including earlier labels in the same validation/test split and
strictly earlier positions in a held-out session. This is retrospective
teacher forcing; the current target label does not enter its own prediction.
It does not prove that an earlier label existed at the later question's
presentation. Server-side correctness availability may also precede displayed
learner feedback; neither is established by a row index alone.

ASSIST has7,059 test targets with an exposed earlier recorded problem end after
presentation. That end is not an independently proven first-response release.
Junyi has2,165,594 test targets with tied presentation times; all accepted rows
have start=end, so zero end overlaps are not a timing certificate. EdNet has
4,089,385 test targets with an earlier completion-proxy overlap and3,364,799 with
a prior same-solving-episode label. KT1 does not contain the final action-level
submission/feedback events required to certify a pre-bundle prediction claim.
These are exact counts under their proxy definitions, **not confirmed leakage
counts or estimates of AUC inflation**. The preserved
[timing analysis](HITSKT_RESPONSE_TIMING_ANALYSIS.md) records definitions and
authoritative-source limitations.

For a publishable retrospective benchmark, state these limitations and do not
claim real-time availability. If real-time/pre-bundle prediction is the intended
research endpoint, recover the actual release contract and make a separate
protocol decision before V2 evaluation or training. Do not implement V2 now.

## Numerical limitations that remain open

Original per-target CUDA logits/probabilities and actual CUDA precision flags
were not persisted. Exact original CUDA-versus-CPU pointwise errors and flip
counts are unidentifiable from aggregate metrics alone. Confusion counts imply
at least5/31/40 flips, but these are lower bounds, not exact counts. A future
fresh CUDA-versus-CPU comparison will have its own pointwise scope.

ASSIST's historical-minus-CPU batch-weighted BCE gap is2.1306950720445528e-6;
CPU accumulation precision accounts for only1.4903841494273706e-9. Full ASSIST
float64 network inference changes zero CPU classifications and still misses
historical BCE by2.1312437579190657e-6. This rules out the tested CPU precision
fixes, not every possible historical CUDA operation. Junyi/EdNet float64
inference covered only4,072/2,666 targets; it was not a full test replay.
The underlying CUDA-specific cause remains NOT VERIFIED.

## Baselines and a correction to earlier prose

All11 completed baseline pairs still need independent fresh full test replay.
Their recorded AUCs and all other stored metrics are retained as **historical**
in the supervisor brief and facts manifest. Original finite best weights,
hashes and strict validation selection pass artifact checks. RKT raw training
pair counts remain unverified beyond the earlier cache algebra/domain checks.
Single-seed42 gains are descriptive; there is no demonstrated statistical
significance or controlled attention-only ablation.

**Verified erratum:** the earlier research-decision paragraph says
“DKVMN49, SAKT/RKT99.” All saved configs and applicable best-checkpoint configs,
plus current data/model code, instead show **DKT199, DKVMN199, SAKT99, RKT49**
prior interactions. The correction is in
[baseline_context_correction.json](supervisor_verification/baseline_context_correction.json).
Earlier reports remain unchanged. This is a descriptive comparison error,
not evidence of changed training data, scores or weights, and does not require
retraining. New reports and replay plans use the saved configurations.

DKT/EdNet has no completed final result. Its epoch9 model weights are finite,
but five Adam second-moment states contain21,749 positive infinities. No
optimizer state was restored or reset, and the run was not resumed. Its best
epoch1 validation result is not a final test row. Keep this result missing.

## Prioritized minimum evidence checklist

| Priority | Required evidence | Completion criterion | Current status |
| --- | --- | --- | --- |
| P0 | Preserve/report evidence and correct terminology/protocol | Immutable source bindings, clear CUDA/CPU rows, verified PDF and explicit caveats | Prepared; final checks in this delivery's verification manifest |
| P1 | Full unchanged HiTSKT CUDA replay on independent resources | All3 checkpoints, hashes/configs/runtime/predictions, target-complete metrics, original1e-6 gate and repeatability outcome | NOT VERIFIED; occupied visible MIG |
| P1 | Resolve every historical metric failure | Passing matching-environment reproduction, or a documented unresolved failure with separately identified reproducible environment-specific results; no unsupported universal-reproduction claim | NOT VERIFIED; CPU gates remain FAIL |
| P2 | Independently verify cited baseline comparisons | Claim-critical RKT/ASSIST, RKT/Junyi, SAKT/EdNet first; all11 before asserting the whole baseline table is reproduced | NOT VERIFIED |
| P2 | RKT source relation construction | Complete independent training-only/cross-fitted pair-count reconstruction matches persisted relations | NOT VERIFIED |
| P2 | Lock research scope and provenance | Retrospective within-student claim and timing/input-transform limitations explicit; full publisher lineage if claiming publisher-raw reproduction | Retrospective statement prepared; missing provenance still NOT VERIFIED |
| Conditional | Real-time, student-disjoint, FAVOR+ or significance claims | New evidence specific to that claim and an explicit research decision before any new study | Outside verified claims; no implementation/training approved |

A supervisor can discuss the clearly qualified historical/CPU results now.
Publication as fully reproduced results or robust model superiority requires
the applicable open evidence above. Additional seeds are required for a
multi-seed robustness/significance claim, not an automatic remedy for CPU
rounding. Retraining is not the next action for the three historical models.

The [bounded replay plan](HITSKT_CHECKPOINT_REPLAY_PLAN.md) specifies device
ownership checks, serial execution, RAM/GPU admission, unchanged batch shapes,
all metrics, immutable predictions and failure reporting. The existing
auto-CUDA evaluator is not safe to launch blindly on the occupied GPU.

## Safe verification performed for this delivery

- Read and hash-bind all three completed diagnostic reports from e1065c9;
  preserve375 original research files and156 previous verification files.
- Recalculate independent metrics on every saved full CPU prediction vector
  (all15,998,524 combined targets): exact agreement with the previous CPU
  manifests. This is a fresh metric calculation, **not new model inference**.
- Recheck all12 baseline best checkpoint/config metadata bindings (including
  the incomplete DKT pair as artifact inspection only); correct context prose.
- Rerun59 relevant independent causal/masking/gradient/count-oracle controls:
  **59 PASS** on19 October, with CUDA hidden and one low-priority CPU thread.
- Size every original HiTSKT evaluation batch and all six RKT relation matrices
  per dataset from metadata only. Maximum HiTSKT forward estimates are
  6.799/4.234/18.964GiB; RKT resident array payloads are
  0.321/16.020/25.915GiB. These are not measured peaks or GPU admission.
  EdNet exceeds the16GiB CPU-forward cap; the planned GPU tier permits an
  estimate up to24GiB only with8GiB free reserve and independent ownership.
- Check GPU resources read-only: the only visible H100 MIG remains occupied,
  LightSAKT2 has a current active-job heartbeat and process visibility is
  restricted. **CUDA allocations and CUDA replays by this delivery: zero.**
- Produce and inspect the supervisor PDF, editable source, plans and explicit
  evidence bindings. No full training, new architecture/protocol, checkpoint
  write or LightSAKT2 modification/signal is performed.

Evidence: [facts](supervisor_verification/supervisor_facts.json),
[test log](supervisor_verification/causal_control_recheck.log),
[resource observation](supervisor_verification/resource_observation.json),
[resource preflight](supervisor_verification/replay_resource_preflight.json),
[initial preservation](supervisor_verification/preservation_initial.json),
[final preservation](supervisor_verification/preservation_final.json) and
[delivery verification](supervisor_verification/delivery_verification.json).
Durable continuation memory remains under `project-memory/`.
