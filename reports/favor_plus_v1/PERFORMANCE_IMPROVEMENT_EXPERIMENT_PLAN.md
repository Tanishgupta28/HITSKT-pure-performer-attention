# HiTSKT performance improvement: architecture review and controlled experiments

25 October 2026. Branch `research/hitskt-favor-plus-v1`; base commit
`50be8f91e43eeb06ad8f4abbe3e189730f8d6f86`.

**Recommendation: first compare fresh FAVOR+128 and FAVOR+512 under matched
conditions. Next test a question-conditioned fusion gate or short attention
over past session representations, one modification at a time.** Keep the
existing residual/pre-normalization structure. More depth or larger embeddings
are lower priorities. Better approximation is a hypothesis for better prediction,
not an established performance improvement.

This task prepared and CPU-validated a reproducible declarative experiment pack.
**No CUDA initialization, GPU probing, optimizer updates or new model evaluation
occurred. No new architecture was installed in the model.** The two proposed
architectural modifications are specifications, not tested implementations.
Historical checkpoints, results and previous frozen experiment contracts remain
unchanged. GPU execution requires a new bounded instruction; presenting this
plan does not approve it.

## What HiTSKT already contains

The actual eight-epoch configuration is width256, four heads, FF2048,
dropout0.1 and one layer in each of Action/Session/CorrectPadding/Decoder.
It has 6,931,457 learned parameters. Config defaults in `hitskt.py` are smaller;
they must not be mistaken for the trained configuration.

| Component | Current mechanism / evidence | Consequence for proposed changes |
|---|---|---|
| ActionEncoder | Question+skill+correctness embeddings, sinusoidal positions, input LayerNorm, causal blocks, EOS summary; [source](../../ktbench/models/hitskt.py#L42) | Already encodes responses and positions; one vector compresses each historical session |
| SessionEncoder | Causal blocks over up to15 historical session vectors plus EOS, returning one EOS vector; [source](../../ktbench/models/hitskt.py#L68) | Historical summary is shared across all target questions; item-conditioned access to individual encoded sessions is absent |
| CorrectPaddingEncoder | Broadcast historical vector **plus** shifted correctness embedding, positions and causal blocks; [source](../../ktbench/models/hitskt.py#L93) | No learned fusion gate. Prior labels enter only through the verified shift |
| Decoder | Current question+skill embeddings, causal cross-attention to aligned CorrectPadding memory; [source](../../ktbench/models/hitskt.py#L109) | Current question information already exists here; proposal must add a specific missing use, not claim the model lacks question embeddings |
| Residuals / normalization | Encoder attention and FF residual additions with pre-LayerNorm; decoder has separate query/memory norms and FF pre-norm; [blocks](../../ktbench/models/performer.py#L92) | Do not add duplicate residuals or advertise pre-LN as new. Existing placement is consistent with the optimization motivation in the [Pre-LN study](https://arxiv.org/abs/2002.04745) |
| Feed-forward / regularization | GELU, two FF dropout modules and attention-output dropout; no learned residual gates | Dropout is already present. A0.1→0.2 ablation is tuning, not a new mechanism |
| Depth / dimensions | Four independent layer-count config fields; widths/heads/FF configurable | Technically possible, but extra depth needs separate verification. The matched four-arm wrapper explicitly rejects more than four attention layers |
| Attention identity | Historical model: deterministic ELU+1. Eight-epoch model: genuine softmax-kernel FAVOR+ with fixed128 ORF features | Preserve names and checkpoint identities; never present ELU+1 scores as FAVOR+ |
| Temporal modeling | Sinusoidal sequence positions, chronological rolling context; no explicit elapsed-time forgetting term in HiTSKT attention | Time-decay/relative-distance attention would be a new mechanism, not an existing feature; defer while timing-availability limitations remain |

The one-vector history compression is a plausible modeling bottleneck, **an
architectural inference rather than a measured cause of the AUC gap**. It
motivates the small, targeted modifications below more directly than simply
adding layers to a model already larger than these baselines.

## What the existing validation evidence actually says

The following table contains **full-validation metrics only**, all on125,800
targets, seed42. Baseline rows are extracted directly from each stored
`metrics.csv` at its validation-AUC-selected epoch and cross-checked against
the saved config. They are historical records, not fresh checkpoint replays.
The FAVOR+ selected checkpoint's complete validation replay did pass in the
eight-epoch study. F1 is from the AUC-selected checkpoint, not independently
maximized across epochs.

| Model | Selected epoch | Full validation AUC | F1 at that epoch | Validation BCE | Completed epochs | Status |
|---|---:|---:|---:|---:|---:|---|
| Historical HiTSKT ELU+1 | 40 | .719757 | .563848 | .595964 | 45 | Stored; fresh validation replay NOT VERIFIED |
| DKT | 12 | .721370 | .511361 | .590488 | 17 | Stored; fresh validation replay NOT VERIFIED |
| DKVMN | 18 | .696252 | .439882 | .604523 | 23 | Stored; fresh validation replay NOT VERIFIED |
| SAKT | 30 | .757706 | .575378 | .564253 | 35 | Stored; fresh validation replay NOT VERIFIED |
| RKT | 14 | .771255 | .586162 | .551800 | 19 | Stored; fresh validation replay NOT VERIFIED |
| HiTSKT FAVOR+128, native FP32 | 7 | .703467 | .508814 | .599038 | 8 | Selected full-validation replay PASS |

[Exact CSV](performance_study_20261025_v1/VALIDATION_CONTEXT.csv),
[source hashes and qualification](performance_study_20261025_v1/VALIDATION_CONTEXT.json).
**No baseline test AUC is used to calculate a validation gain.** These same-split
descriptive rows still do not isolate architecture: historical epoch ceilings,
stopping patience, optimization, features, context and supervised training
populations differ. DKT/DKVMN use199 prior interactions, SAKT99 and RKT49;
HiTSKT uses15 sessions. The corrected context values supersede an earlier report's
description, without modifying that report. Baselines supervise647,283 training
interactions; hierarchical HiTSKT supervises566,107, treating the first training
session as context. Parameter counts are about59.8k/56.6k/528k/239k for
DKT/DKVMN/SAKT/RKT versus6.93m for HiTSKT.

The preparation verifies that **current adapters** select identical full
validation event IDs/labels. Ordered event-ID SHA256:
`7e0da8b0a737122179ade388f7091449c52314b06ab3ec1826bbd33c147f83bb`;
label SHA256 `49b7bd9f25b6d21e2203495a5ed0d0c506ce32f19123a73e713dd5950550c935`.
There are47,584 positives/78,216 negatives. This does not retroactively prove
historical baseline predictions; checkpoint replay remains required.

The eight-epoch curve is not convergence evidence. Validation AUC improved
.649592→.703467 through epoch7, then .700440 at8. The experiment stopped for
its time budget, **not patience-three early stopping**. Further learning may
help, but one worse epoch cannot establish whether it will. Best observed
F1 .532809 at6 is distinct from F1 .508814 at the AUC-selected7. The observed
recall/F1 decline at8 was a downward probability shift at the unchanged0.5
threshold; independent label/metric checks passed. Do not "improve" reported
F1 by silently moving the threshold or changing checkpoint selection.

## Approximation issues remain material

At epoch8, the original six training diagnostics/≤32 prefixes gave median
relative output-L2 errors Action **.262562**, CorrectPadding **.503431** and
Decoder **.562503**, all above **.20 REVIEW**. Production versus independent
feature-kernel computation passed (maximum3.391481e-6), so these were not an
observed causal, normalization or arithmetic defect. The later frozen study
found CPE saved-draw error growing .142→.566 while independent-draw error grew
only .114→.183 (14cases/≤128). Feature concentration, learned K/V geometry
and projection-specific alignment are supported explanations; causal proof of
an optimizer mechanism is absent. Different case/prefix scopes must stay labeled.

For identical **initial** training Q/K/V,14cases/≤128, matched nested projection
pool seed42, the latest CUDA comparison gave:

| Configuration | Action error | CPE error | Action cases>.20 |
|---|---:|---:|---:|
| ORF128 | .243430 | .156680 | 12/14 |
| ORF512 | .152904 | .066212 | 1/14 |
| Antithetic128 | .251865 | .075872 | 11/14 |
| Antithetic512 | .179270 | .049956 | 2/14 |

512 features are a sensible first intervention with unchanged learned parameter
count, but lower feature error does not imply higher AUC. Antithetic512 helps
CPE while ORF512 is better on Action in this draw. All four Action methods
exceeded.20 for all14 cases when diagnostic Q/K norms were scaled1.5×.
Scaling Q/K would change softmax temperature; it is not a silent bug fix.

The strict FP32 ORF512 stress-reference failure remains a FAIL. Separate
FP64-attention configurations passed96 CUDA cases plus masks, gradients and
four exact reloads. The next matched pair therefore uses **FP64 attention for
both arms**, FP32 learned parameters/outputs. Native FP32 antithetic CUDA was
not verified; do not infer it from FP64 success. All antithetic learning and
candidate validation AUC/F1 remain unrun. The common nested512 projection
draw differs from the historical native128 draw for some heads, so fresh paired
training is essential. Historical trained weights must not warm-start a
claimed fair method comparison.

This is consistent with FAVOR+ being a randomized approximation to a softmax
kernel, rather than a guarantee that a particular finite estimator remains
close after adaptive training. See the [Performer paper](https://arxiv.org/abs/2009.14794)
and [complete local comparison](FOUR_ARM_CUDA_COMPARISON_AND_TRAINING_PROPOSAL.md).
FP64 fixes the tested arithmetic residual, not finite-feature approximation.
The.20 flag remains; alongside it report attention-weight TV, absolute RMSE,
exact-output norm, feature ESS and fixed-draw versus audit-draw discrepancies.
An implementation-correct finite-feature model can have useful predictive
metrics while still failing this approximation-review criterion. Report those
two outcomes separately.

## Two additional architectural hypotheses

### A. Question-conditioned history/response fusion gate

Current CPE input is `h + r_t + position_t`, where `h` is historical session
summary and `r_t` is the BOS/previous-response embedding, never current truth.
Proposed separate version:

```text
x_t = existing decoder question_embedding(q_t) + skill_embedding(s_t)
g_t = sigmoid(W_g [h; r_t; x_t] + b_g)
memory_t = 2*g_t*h + 2*(1-g_t)*r_t + position_t
```

Zero-initialize W_g/b_g so `g_t=.5` initially reproduces the original sum,
with the same base learned weights/RNG. Reuse current item embeddings; do not
add target correctness or future features. The inference-time question/skill
is permitted, while response availability retains the original retrospective
qualification. Mask PAD/EOS exactly as before. Added parameters at width256:
**196,864** for a vector gate with three width-sized inputs and bias.

Hypothesis: adapt the influence of long-term state versus the most recent
response to the item being predicted. Risks: gate saturation, overfitting,
amplifying projection alignment and merely adding capacity. Monitor gate
distributions, Q/K norms and approximation, not just AUC. Identity-at-init,
no-own/future-label, padding, gradients and isolated checkpoint tests are required
before a GPU pilot. **Design only: not implemented/tested in this task.**

### B. Question-conditioned exact retrieval over past session states

Expose the already encoded historical SessionEncoder positions in addition to
its global EOS vector. Query them using the current item embedding; apply
scaled dot-product softmax over the **at most15 completed prior sessions**
only, excluding PAD and the artificial EOS. Add the retrieved vector to the
global history before CPE; target positions never query other target positions
in this new retrieval. The existing Action/CPE/Decoder attention stays fixed.

This avoids compressing all past sessions into the same item-independent vector
at this stage. It is a proposed **hybrid FAVOR+/exact-attention** variant, not
an all-FAVOR+ model or an AKT reproduction. Four bias-free width projections
add262,144 parameters. Use local seeded initialization for new Q/K/V and zero
output projection to preserve base behavior initially; Q/K/V learning begins
after the output projection moves. Never zero every new projection. Specify
empty-history behavior explicitly, even though current targets have history.

The knowledge-tracing rationale is question-relevant history selection, as in
[SAKT](https://arxiv.org/abs/1907.06837) and contextual retrieval in
[AKT](https://arxiv.org/abs/2007.12324); **the benefit for this HiTSKT variant is
an untested hypothesis**, not a transferred AUC claim. It can recover between-
session selectivity, but cannot restore information already lost in ActionEOS
summaries. Additional attention scores scale as `B*heads*L_target*15`, not
`B*heads*L_target²`. Risks are overfitting, stale/irrelevant sessions, extra
parameters and accidental mask/split mistakes. Prove prior-session-only masks,
padding, future/own-label invariants and exact reload before GPU use.
**Design only: not implemented/tested in this task.**

Do not combine A and B initially: their individual contributions would be
unidentifiable. The recipe pack fixes their reference to ORF512; if the primary
pair rejects512, create a new prospective version rather than editing this plan.

## Prioritized experiment matrix

| Priority | Comparison | Isolated change / purpose | Readiness / main risk |
|---|---|---|---|
| 1 | Fresh ORF128 versus fresh ORF512 | Feature count only; same FP64/pool/seed/architecture/data/optimizer | Attention CUDA verified; learning comparison unrun. Full-shape memory unknown |
| 2 optional | ORF512 versus antithetic512 | Feature mapping at the same total budget | FP64 CUDA verified; not a guaranteed Action improvement; learning alignment may recur |
| 3A | ORF512 versus ORF512+fusion gate | Adaptive fusion of existing information | Design only; extra196,864 parameters and possible gate saturation |
| 3B | ORF512 versus ORF512+session retrieval | Item-conditioned access to past sessions | Design only; extra262,144 parameters, additional mask contract |
| 4 | Same chosen attention, dropout.1 versus.2 | Controlled regularization ablation | Existing config supports it; no assumed benefit; keep dimensions/depth fixed |
| 5 | Same attention, SessionEncoder depth1 versus2 | More cross-session processing at short length | Config supports it, matched wrapper does not; needs new factory/projection-seed mapping and tests |
| 6 | Width256 versus128, heads4 fixed | Capacity/memory ablation | Changes head dimension/projection geometry too; larger embeddings are not a first recommendation |
| Reference | Fresh ELU+1 HiTSKT under the same new budget | Establish whether FAVOR+ changes performance versus original kernel | Historical40-epoch selected model is descriptive, not an eight-epoch control; separate protected namespace/runner required |

An exact-softmax CPE/Decoder ablation or relative-distance/forgetting attention
is also scientifically possible, but is lower priority than these controls.
The dense target matrix can be large at939 positions. Do not assume an
efficient library backend or arbitrary temporal kernel preserves the existing
linear scan; that requires separate implementation, profiling and causal tests.
Neither new residuals nor switching to pre-LN is proposed: both already exist.

## Reproducible experiment pack and execution boundaries

Generated by [CPU-only preparation script](../../scripts/prepare_hitskt_performance_study.py):

- [Experiment recipes and identities](performance_study_20261025_v1/EXPERIMENT_PLAN.json),
  SHA256 `94ef6169d14aa3fd0d93afee765e349f2d8e019d2f09e2692972b2e75affbbd4`.
- [Exact common eight-epoch training schedules](performance_study_20261025_v1/COMMON_TRAIN_SCHEDULES.json):
  all7,060 target sessions and566,107 labels per epoch,1,454 optimizer batches
  total if original shapes are retained. No truncated histories/actions or
  omitted target sessions. Dataset11-file and source identities are frozen.
- Fixed seed42, fresh learned initialization, fixed stage seeds42–45/common
  pool, no warm start from old epoch7/8/diagnostic50. Identical Adam LR5e-5,
  betas(.9,.999),eps1e-8,weight_decay0,no clipping, dropout.1, width256,
  FF2048,heads4,one layer per stage. Preserve model/dropout RNG alignment.
- First screen:20updates per primary arm,40total; same bounded training order,
  B1/cost2048 and32-session/3157-label validation sample from the earlier sealed
  readiness plan. The [selection expansion](performance_study_20261025_v1/PILOT_SELECTION.json)
  preserves all20 training,32 validation and six diagnostic descriptors with
  source/base-plan hashes, and checks train/validation membership plus1063
  positive labels. It prospectively completes the declarative pilot selection;
  it does not authorize a run. Treat sampled results as stability/exploratory evidence only.
  Do not compare sampled AUC against the table's full-validation AUC.
- Proposed full comparison: **eight complete epochs per arm** with identical
  data-exposure ceiling, full validation every epoch and best-AUC selection;
  no performance early stop within this particular equal-budget comparison.
  Safety/time stops still apply. This differs explicitly from historical
  patience3. A future longer/patience-based study must be registered separately.
- Keep threshold.5; AUC primary, BCE/F1/precision/recall/accuracy secondary.
  Report F1 at the AUC-selected checkpoint; maximum epoch-F1 is a separate
  exploratory statistic. Retain per-example validation predictions/event IDs.
- Reuse fixed training diagnostic cases, prefixes32/128 and preselected long
  rows, original draw plus audit projections17/101. Audit projections are not
  installed in the model or picked by validation score. Record errors, feature
  ESS/concentration and Q/K distributions as learning proceeds.
- Save atomic no-overwrite model/Adam/RNG/projections/sampler/source/data/config
  checkpoints, verify fresh-process next-computation recovery without another
  step. No retry/reset on a failed step. Stop the comparison on severe/resource
  failure, preserving the latest checkpoint and actual update count.

The current20-update execution runner binds **four arms**; the new priorities
do not silently change that old contract into a two-arm run. Two-arm execution
and full-epoch matched-variant wiring still need a separate reviewed execution
scope and targeted tests. The five recipes are declarations, not commands to
the old epoch controller: that controller is fixed to historical native128.
Gate/retrieval implementation is absent. These readiness limits are explicit
in the machine-readable plan.

## GPU/RAM budget: measured facts versus requests

Shared H100 MIG3g.40gb remains the proposed device. No ownership re-audit or
GPU probe was necessary for this CPU planning task. Future admission must
use the existing shared policy; exclusive ownership is not assumed. Preserve
8GiB allocated/reserved ceiling,10GiB advisory total,16GiB admission-free,
6GiB runtime-free,4GiB own hostRSS,2GiB shared-host headroom,.1s pacing and
30s update ceiling. Keep unknown PID attribution/utilization explicitly unknown;
do not stop other jobs or treat global free VRAM as own memory attribution.

| Stage | Proposed training ceiling | Memory evidence / requirement | GPU time budget |
|---|---|---|---|
| Primary pair stability screen | 20fresh updates per arm,40total | Prior moment-reserved live peaks393/987MiB for128/512; allow roughly1–1.5GiB plus unmeasured Adam workspace, hard8GiB cap | Request ≤15min total; planning ceiling, not measured optimizer ETA |
| Representative full-shape profile | Zero optimizer steps | Progressively profile real complete history/batch shapes for both FP64 arms; B64/cost32768 only if it fits; host4GiB | Request ≤10min total; stop on cap/pressure/numerical failure |
| Matched eight-epoch pair | 8full epochs each; no silent partial-epoch claim | Full512/FP64 peak NOT MEASURED; old128/FP32 live3.192GiB, reservation7.990GiB, host1.841GiB are historical only | Runtime NOT ESTIMABLE until full-shape measurements; provisional ≤2GPU-hour reservation for pair is a request/ceiling, not a prediction or approval |
| Gate/retrieval or optional antithetic study | Separate matched budget after earlier results | New-variant peak and long-history behavior unmeasured | NOT ESTIMABLE; no blanket budget/sweep authorized |
| Confirmatory seeds and baseline reruns | Separately approve after freezing method | Profile each distinct architecture/context | NOT ESTIMABLE from the existing singleton benchmark |

CPU component accounting over all original first-eight schedules gives maxima
**1.542GiB for128 /3.604GiB for512** with FP64 attention and FP32 FF/weights.
These are sums of selected retained boundary/log-feature/value/FF/persistent
components, **not GPU peaks, complete upper bounds or fit guarantees**.
They omit other saved tensors, recomputation graph/workspaces and allocator
behavior. Old measured reservation almost reached8GiB despite a lower live
peak. See [component accounting](performance_study_20261025_v1/MEMORY_COMPONENT_ACCOUNTING.json).
Feature-dependent storage scales with M; the whole model need not scale4×.

If either primary arm cannot fit, predeclare one **common** smaller batch/token
schedule for both arms, preserving every target and complete sequence. Seal a
new plan, counts and order before training; disclose the schedule difference
from the historical8-epoch run. Do not silently shrink only512, truncate long
sessions, drop oversize examples, retry steps or increase the shared budget.
After profiling, compute a measured epoch-time projection including validation,
telemetry, diagnostics and recovery; request a bounded time window adequate
for both complete runs. A new reservation never constitutes training approval.
The provisional two-hour reservation also does not increase the existing
45-minute execution ceiling per approved campaign. Longer execution requires
an explicit new time-budget decision; otherwise stop at that ceiling and retain
an incomplete-run status. Never reset the clock to manufacture eight epochs.

## Establishing a defensible improvement over baselines

1. Finish the matched attention learning screen, then the approved full-data
   pair. Inspect both predictive and approximation outcomes; do not select a
   method from initialization error alone. Preserve all negative results.
2. Before claiming a baseline win, freshly replay ASSIST baseline best checkpoints
   on **full validation** with target-ID/label hashes and exact metric definitions.
   RKT raw relation-count provenance remains an additional verification gap.
   Do not modify baselines or resume unfinished DKT/EdNet in this task.
3. For a controlled benchmark, separately register fresh baseline runs with the
   same split/validation population, training target population, seed set,
   data-exposure ceiling and tuning allowance. Preserve their native published
   feature/context choices as documented, or add a separately labeled matched-
   information ablation. They are different questions. Never replace their
   historical artifacts with a rerun or claim existing heterogeneous runs are
   controlled attention ablations.
4. Freeze the selected model, hyperparameters, projection policy and checkpoint
   rule. Propose the same training seeds42/43/44 for selected HiTSKT and baseline
   controls, not a seed search. Report all runs/dispersion; three seeds alone
   do not establish significance. Account for student-clustered target dependence
   in any paired bootstrap/interval and separate this from training-seed variation.
5. Only after selection and separate final-evaluation authorization, evaluate
   the locked models on the same final-test IDs. The current study never uses
   test scores for selection. Legacy baseline test results were already known
   historically, so this is held out from the **new study**, not a claim that
   nobody has ever inspected that split. Stronger generalization evidence may
   need a separately agreed external confirmation study; do not silently resplit.
6. Report absolute AUC/F1/BCE differences on the same split, uncertainty, target
   coverage, parameter/input/context budgets and runtime. AUC improvement with
   poorer recall/F1 is a trade-off, not universal superiority. State significance
   only if actual supporting experiments and the predeclared analysis support it.

Evaluation remains chronological **within-student**, retrospective teacher
forcing, not student-disjoint or guaranteed real-time response availability.
The earlier timing qualification is preserved; it is not declared confirmed
leakage or fixed by any attention/gating change.

## Current outcome

**33 CPU tests PASS (18 new plan-contract tests,15 existing matched-factory
checks); zero failures.** They reject feature-comparison confounds, altered
review thresholds, resource changes, accidental training/test authorization
and mixing test metrics into the validation context. Preparation's source/data
hash checks and common-schedule coverage pass. The tests validate preparation,
not the unimplemented gate/retrieval or a GPU training outcome.

The existing .703467 validation result is a valid recorded starting point;
improvement is unproven. Recommended next execution is the bounded **primary
ORF128/ORF512 learning comparison plus full-shape profiling**, after a new
explicit bounded instruction and matching runner scope. No full training,
projection redraw, architectural replacement or test evaluation was authorized
or performed here. Preserve all original checkpoints, research reports, datasets,
LightSAKT2, unfinished DKT/EdNet and unrelated working-tree edits.
