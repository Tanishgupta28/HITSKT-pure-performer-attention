# HiTSKT research decision report

Date: 2026-10-18 UTC. Follow-up to audit `5625478`.

**Do not retrain any of the three historical V1 HiTSKT models yet.** Preserve
their checkpoints and original results as provisional historical evidence.
All three need further verification before their complete reported metric set
can be retained as independently reproduced. The safe CPU investigation does
not identify a training defect that a new training run would fix. It also does
not clear the existing results for publication.

If the intended claim is real-time prediction at question/initial-bundle
presentation, V1 does not guarantee that information set. Establish and version
an availability-aware protocol first; train all affected methods under it if
their training contexts change. That is a separate scientific experiment,
not a silent repair to historical metrics.

## Per-dataset decision

| Dataset | Stored AUC | Independent full CPU AUC | CPU metric set | CUDA replay | Actual response availability | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| ASSIST2017 | 0.7096240159197755 | 0.709623637214297 | FAIL at frozen1e-6 | NOT VERIFIED | NOT VERIFIED | 2 — Additional verification; not yet retraining |
| Junyi | 0.7978262055191263 | 0.7978264739233538 | FAIL at frozen1e-6 | NOT VERIFIED | NOT VERIFIED | 2 — Additional verification; not yet retraining |
| EdNet-KT1 | 0.7693648364413692 | 0.7693647361509419 | FAIL at frozen1e-6 | NOT VERIFIED | NOT VERIFIED | 2 — Additional verification; not yet retraining |

**ASSIST2017:** AUC and MSE pass the declared numerical gate, but
accuracy/precision/recall/F1 and BCE fail. All112,252 targets, selected epoch40
and original hash are verified. Full CPU64 inference still fails BCE and leaves
the CPU classifications unchanged. The7,059 full-context recorded-end overlaps
do not establish when first-answer correctness arrived. Keep the V1 artifact
and independently measured CPU result with separate provenance; historical CUDA
verification remains necessary. **No established requirement to retrain V1.**

**Junyi:** AUC, MSE and both BCE definitions pass; all four classification
metrics fail. All2,456,402 test targets, selected epoch32 and hash are verified.
Higher-precision sampled inference covers4,072 targets only; it cannot certify
the full forward precision effect. Every accepted source row has identical
start/end, with2,165,594 tied-start test targets; publisher timing transformation
is unverified. **No established requirement to retrain V1; further CUDA and
timing-provenance verification required.**

**EdNet-KT1:** AUC, precision, MSE and BCE pass; accuracy, recall and F1 fail.
All13,429,870 targets, selected epoch30 and hash are verified. Sigmoid precision
alone changes exactly4 classifications on identical cached logits; a2,666-target
CPU64 sample changes1, which must not be extrapolated. The3,364,799 test targets
with prior same-episode labels prevent certifying a strict initial-bundle
information set from V1. **No established requirement to retrain the historical
retrospective model; corrected availability evaluation, followed by training
under V2 if contexts change, is required for a new pre-bundle/real-time claim.**

These are recommendation **2** for each historical V1 result, not recommendation1
or publication clearance. Recommendation **3** applies to an availability-aware
claim when its required information set differs. No dataset receives an
unqualified retain-and-publish decision.

All original-versus-reproduced accuracy/precision/recall/F1/MSE/BCE values, maximum
measurable numerical errors, exact diagnostic classification counts and missing
historical pointwise evidence are in
[HITSKT_REPRODUCIBILITY_ROOT_CAUSE.md](HITSKT_REPRODUCIBILITY_ROOT_CAUSE.md).
Exact timing counts and defensible claim boundaries are in
[HITSKT_RESPONSE_TIMING_ANALYSIS.md](HITSKT_RESPONSE_TIMING_ANALYSIS.md).
Separate follow-up manifests exist for
[ASSIST](reproducibility_diagnostics/assist2017_diagnostic_manifest.json),
[Junyi](reproducibility_diagnostics/junyi_diagnostic_manifest.json) and
[EdNet](reproducibility_diagnostics/ednet_kt1_diagnostic_manifest.json).

## Architecture identity — PASS as ELU+1; FAIL as FAVOR+

The preserved model is a **hierarchical HiTSKT variant with deterministic causal
ELU+1 linear attention**. In [ktbench/models/performer.py](../ktbench/models/performer.py),
lines68–85, projected queries/keys use ELU(x)+1, and key/key-value prefix sums
compute normalized causal kernel attention. There is no positive orthogonal
random-feature matrix or randomized softmax-kernel approximation. Legacy class
names/config strings containing Performer do not change the mathematics.

For a causal prefix, output is
\(\phi(q_i)^\top\sum_{j\le i}\phi(k_j)v_j^\top /
\max(\phi(q_i)^\top\sum_{j\le i}\phi(k_j),10^{-6})\),
with \(\phi(x)=\operatorname{ELU}(x)+1\), masks and the independently verified
response shifting. This is not exp(q·k/sqrt(d)) attention approximated by FAVOR+.
The model also has pre-normalization/GELU and omits some original-paper details;
historical results describe this variant, not an exact original-paper replica.

The [separate FAVOR+ implementation proposal](HITSKT_FAVOR_PLUS_IMPLEMENTATION_PROPOSAL.md)
specifies positive orthogonal random features, checkpoint identity, causal
stabilization and independent tests. No implementation or full training was
started. A future FAVOR+ claim requires separate training; relabelling or
swapping the historical evaluator is scientifically invalid.

## Baselines: artifacts versus replayed results

All11 completed DKT/DKVMN/SAKT/RKT baseline pairs need a fresh full checkpoint
test replay for independent verification. Rechecked hashes, finite tensors,
strict validation selection and stored-result consistency are useful artifact
evidence, not a substitute for prediction replay. The unfinished twelfth pair
has no final test score. The table below contains **historical stored AUCs**;
none is presented as freshly reproduced here.

| Model | Dataset | Selected epoch | Historical AUC | Hash/selection | Fresh test metrics |
| --- | --- | --- | --- | --- | --- |
| DKT | ASSIST2017 | 12 | 0.7069997420457494 | PASS | NOT VERIFIED — replay needed |
| DKT | Junyi | 3 | 0.7476841496612041 | PASS | NOT VERIFIED — replay needed |
| DKT | EdNet-KT1 | 1 | MISSING | PASS | NOT VERIFIED — unfinished |
| DKVMN | ASSIST2017 | 18 | 0.6822527020970142 | PASS | NOT VERIFIED — replay needed |
| DKVMN | Junyi | 1 | 0.7415784624691203 | PASS | NOT VERIFIED — replay needed |
| DKVMN | EdNet-KT1 | 1 | 0.6741059915424829 | PASS | NOT VERIFIED — replay needed |
| SAKT | ASSIST2017 | 30 | 0.7384655653536991 | PASS | NOT VERIFIED — replay needed |
| SAKT | Junyi | 13 | 0.7887961776898904 | PASS | NOT VERIFIED — replay needed |
| SAKT | EdNet-KT1 | 13 | 0.7630078573122142 | PASS | NOT VERIFIED — replay needed |
| RKT | ASSIST2017 | 14 | 0.7486264898793284 | PASS | NOT VERIFIED — replay needed |
| RKT | Junyi | 8 | 0.7912548486618094 | PASS | NOT VERIFIED — replay needed |
| RKT | EdNet-KT1 | 1 | 0.7475470202501552 | PASS | NOT VERIFIED — replay needed |

Read-only follow-up evidence:
[artifact_and_dkt_forensics.json](reproducibility_diagnostics/artifact_and_dkt_forensics.json).
Full original consistency checks remain in
[benchmark_artifacts.json](independent_correctness_audit/benchmark_artifacts.json).
The raw-label-pair reconstruction for RKT remains NOT VERIFIED; the prior
cache algebra, domains and CRC checks do not prove the underlying co-occurrence
counts were generated correctly. Existing class/mask tests pass their defined
information-index contract, not the physical release-time contract.

Comparability remains qualified: DKT has199 prior interactions and skill inputs,
DKVMN49, SAKT/RKT99, while HiTSKT has15 sessions and question+skill inputs.
Parameter counts and fixed hyperparameters differ. HiTSKT excludes the first
training session from supervision, whereas baselines supervise those training
events with prior context; held-out target populations match. RKT has
training-only cross-fitted relations whose raw construction needs verification.
Shared splits, thresholds and validation selection do not make these controlled
attention-only ablations or exact replicas of every published baseline.

The recorded single-seed42 AUC gain over the best available baseline is
+0.006571356857316912 on Junyi and+0.006356979129155005 on EdNet. On ASSIST it
is−0.03900247395955292. CPU AUC differences are much smaller than those recorded
gaps, but missing baseline replays, differing information budgets, timing
assumptions and one training seed prevent claiming robust superiority or
statistical significance. EdNet's strongest comparison is incomplete without
DKT; no missing result is inferred from its validation checkpoint.

## Unfinished DKT/EdNet: confirmed state defect, origin unresolved

No final_results.json or _SUCCESS exists. Nine complete epochs are retained,
best epoch1 validation AUC0.6845717850759062, last epoch9 validation
AUC0.6108838021742428, non-improvement counter8 and authorized patience10.
Epoch10 progress stopped at36,418,836/53,990,022 training targets on October12;
the stale running status is not an active trainer or a completed result.

Best hash: `56a911479a2c0b53edd457e4179a942be6f652717e7322c35b41e45194e90215`.
Last hash: `ae9f8248669c3800064020cbbb64c4e82ade108cac0cef27e1a23c74f208842b`.
Best model/Adam and last model weights are finite. Last Adam exp_avg_sq contains:

| Parameter | Positive infinities |
| --- | --- |
| interaction_embedding.weight | 4306 |
| rnn.weight_ih_l0 | 11673 |
| rnn.weight_hh_l0 | 5406 |
| rnn.bias_ih_l0 | 182 |
| rnn.bias_hh_l0 | 182 |

Total **21,749**. First moments are finite but reach approximately1.97e13;
finite second moments reach approximately2.56e32. Saved optimizer uses FP32,
Adam lr0.0002, betas(0.9,0.999), eps1e-8, no AMSGrad and no configured gradient
clip. Large historical gradients/second-moment overflow are a plausible cause;
no per-step gradient logs or complete intermediate canonical checkpoints prove
the first offending update. A finite best epoch1 does not locate the failure
within later epochs. The archived epoch1 DKT snapshots are also finite; the
transport-switch snapshots at epochs13/15 are SAKT, not later DKT evidence.

With finite subsequent gradients, beta2*infinity plus a finite squared-gradient
term stays infinite. With finite first moment, the resulting infinite Adam
denominator gives zero update on those coordinates; ordinary decay cannot
repair them. A tiny synthetic algebra diagnostic confirms that behavior without
stepping any historical optimizer. Other coordinates may still update, so
finite epoch losses/model weights do not certify optimizer integrity.

Checkpoint serialization embeds common patience5, whereas active config/events
and authorized DKT/EdNet control use10. This is a preserved metadata discrepancy,
not permission to reset its optimizer or silently restart. The damaged last
state should not be called ready for continuation. Recovery from a validated
earlier finite state or a separately versioned rerun needs an explicit recorded
policy and authorization; no DKT result or recovery was fabricated here. This
baseline defect does not establish a defect in the three finite HiTSKT states.

## Issue register and retraining implications

| Issue | Severity / status | Affected claim | Retraining implication |
| --- | --- | --- | --- |
| Historical-to-CPU metric gate fails | HIGH / FAIL, all3 | Complete historical metric-set reproduction | No demonstrated training defect; obtain matching CUDA replay first |
| Historical vectors/runtime flags absent | HIGH / NOT VERIFIED | Exact CUDA/CPU flips, max errors and operation-level root cause | Additional inference/provenance, not training |
| End/proxy timing and transformed ties | HIGH / availability NOT VERIFIED | Real-time question/bundle presentation claims, all3 | V2 evaluation first; retrain all affected methods if training contexts change |
| Historical architecture called FAVOR+ | HIGH / identity claim FAIL | Performer/FAVOR+ attribution | Correct terminology for V1; genuine FAVOR+ requires separately approved new training |
| Baseline full checkpoint replay absent | MEDIUM / NOT VERIFIED,11 completed pairs | Junyi/EdNet recorded gains | Replay first; no established baseline retraining need |
| RKT raw pair-count reconstruction absent | MEDIUM / NOT VERIFIED | RKT input-relation provenance | Rebuild evidence first; changed relations may require a new RKT run |
| One primary training seed42 | MEDIUM / significance NOT VERIFIED | Robustness and statistical significance | Additional seeds are new experiments, only after core verification |
| DKT/EdNet last optimizer nonfinite, no final test | HIGH / integrity FAIL; score NOT VERIFIED | Missing fifteenth baseline result | No resume now; separately agree finite-state recovery/rerun policy |
| DKT saved patience5 versus active10 | LOW / metadata FAIL | DKT continuation reproducibility | Not by itself a retraining reason |
| Publisher-to-author transform provenance incomplete | MEDIUM / NOT VERIFIED for ASSIST/Junyi | Original raw provenance and timestamp meaning | Recover transformation first; no automatic training change |

## Bounded next steps, after this diagnosis

1. When safely idle independent H100/MIG resources are available, replay the
   three unchanged best checkpoints, capture settings and vectors, and apply
   the original gate. Preserve matching-runtime results separately from CPU
   differences. Do not interrupt LightSAKT2 or tune tolerance to observed metrics.
2. Freshly replay the11 completed baselines and independently check RKT's
   training-only raw pair construction. Do not resume the unfinished DKT run
   as part of an evaluation task.
3. Recover original timing/provenance evidence; adopt
   [availability V2](HITSKT_AVAILABILITY_PROTOCOL_V2_PROPOSAL.md) only as an explicit
   new protocol, with frozen-checkpoint sensitivity before a new training plan.
4. Only after these checks, consider approved matched-information ablations,
   additional training seeds and [genuine FAVOR+](HITSKT_FAVOR_PLUS_IMPLEMENTATION_PROPOSAL.md).
   Those experiments have not been started and do not repair missing evidence
   about historical experiments.

## Preservation and delivery

All375 original captured files, all30 best/last checkpoints and106 previous audit
files remain hash-verified. The original audit report, model, loaders, results,
Git history, unrelated README edit and SAKT console file are preserved. New
evidence is additive and learner-level arrays stay ignored/local.59 relevant
tests pass. No new full training, protocol replacement, GPU allocation or
LightSAKT2 modification/signal occurred. Resource and preservation records are
linked from the root-cause report. Durable continuation memory is
[project-memory/REPRODUCIBILITY_STATE.md](../project-memory/REPRODUCIBILITY_STATE.md).

**Decision: retain all three V1 artifacts provisionally, require further
verification of their reported results, and do not order retraining solely
from these numerical discrepancies. Publication readiness remains NOT VERIFIED.**
