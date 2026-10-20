# ASSIST2017 FAVOR+: checkpoint20 review and proposed continuation to50

20 October2026 UTC. Research branch `research/hitskt-favor-plus-v1`.
Parent publication `9dcfe5ca439dcca1e84bd27fc2f26a3de16de69b`;
CPU review predeclared in `dd6be27` before the new measurements.

**Checkpoint integrity PASS; approximation REVIEW remains. Conditional GO for a
further bounded diagnostic continuation, not full training. Explicit approval
for30 additional updates is PENDING. No CUDA context, real-data model forward,
new validation evaluation or optimizer update was executed in this preparation.**

## Checkpoint integrity and exact recovery boundary

[CPU verification](assist50_preparation_20261020_v1/CHECKPOINT_VERIFICATION.json)
reloads each original binary with `weights_only=True` and CPU mapping. Whole-file,
seven embedded component and four projection hashes match the published evidence.
All61 Adam states have the correct update counter, finite moments and nonnegative
second moments. Python/NumPy/CPU RNG states restore into isolated CPU generators;
the single saved CUDA RNG buffer is checked and hashed, **not restored on a GPU**.
The original experiment contract,24 source-file hashes,11 processed-file hashes,
configuration hash, ordered identities and sampler cursors validate unchanged.
At20, strict CPU model loading and Adam loading reproduce every saved tensor
exactly; no forward or update is required for this integrity check.

|Update|Original checkpoint SHA256|Integrity now|
|---:|---|---|
|5|`93fca2f31a2441c30781904c89e1b5719761526ea7ca5e5442bab6f1d0d4cb0a`|PASS|
|10|`a47a81892b15240a7423ad73acb4c32141c315cfac3fa0ed0f2c9485aef22e16`|PASS|
|20|`0388e984b8bed21210455be561524a041c892b98b2a87b50fa3ecb1c3ade86f4`|PASS; exact continuation parent|

The previous study already passed three fresh **CUDA** prediction/next-step
gradient recovery tests with zero Adam calls, including checkpoint20. Those are
historical execution results, not newly repeated tests. See the
[20-update report](ASSIST_20_UPDATE_VALIDATION_RESULTS.md) and
[recovery20 receipt](../../experiments/hitskt_favor_plus_v1/assist2017/assist20_recovery_20261020_v1/verify_020/recovery_result.json).
Checkpoint binaries remain local/ignored, each83,802,810bytes; public Git alone
cannot supply them or the dataset. No checkpoint, old contract or result changed.

## Why ActionEncoder exceeds0.20

[Independent CPU review](assist50_preparation_20261020_v1/ACTION_APPROXIMATION_REVIEW.json)
replays all24 saved ActionEncoder prefixes: six preselected training examples at
updates0/5/10/20, each first32 positions/four heads. Each snapshot contains six
distinct Q/K/V/mask tensors; this is still a small correlated training cohort,
not six independent experiments. Raw attention is compared before output
projection/dropout. The metric is complete-prefix relative L2 against exact
causal softmax, not label error, prediction error or AUC.

|Update|Median relative L2|Maximum prefix error|Prefixes above0.20|Median pair-feature ESS|
|---:|---:|---:|---:|---:|
|0|0.213175860|0.235817233|4/6|8.1603|
|5|0.212916830|0.235160217|4/6|8.1782|
|10|0.213326389|0.235059071|4/6|8.1939|
|20|0.213145477|0.234787820|4/6|8.1807|

The error predates learning and does not show sustained growth through20. Saved
GPU outputs agree with the independent float64 feature-kernel reference to
maximum absolute **1.01250e-6**, below the unchanged5e-5 gate. CPU FP32 versus saved
GPU maximum is **4.76837e-7**. Recomputing the scan in FP64 reduces implementation
error to **2.38698e-15**, while leaving the substantial softmax discrepancy.
Float32 rounding therefore does not explain the0.2131 warning in these cases.

For the bounded update20 prefixes, Q norms have p95 across examples4.775–5.432;
K p95 spans **5.127–5.349**. These are unscaled norms, before division by
the head dimension to the one-quarter power (`64**0.25`); neither
the absolute norm nor a single norm-growth ratio alone certifies approximation.
Q representation relative drift is0.60–0.99%, K drift2.11–2.70% from initialization;
the corresponding directional cosines remain≥0.99963. Pair-feature ESS medians
are7.37–8.46 out of128, with individual pair ESS as low as1.016. Mean attention
weight total variation is0.073–0.156. Some per-head/query relative output errors
reach0.935 even when complete-prefix errors are smaller. Concentration and tail
error are substantive warnings, not resolved by finite model/optimizer states.

An additional predeclared **frozen-representation-only** comparison uses the same
six update20 prefixes, m128/256/512, projection seeds0..7,48 cases per count:

|Features|Median error|p90 error|Maximum error|Cases above0.20|
|---:|---:|---:|---:|---:|
|128|0.181615|0.228461|0.251996|17/48|
|256|0.162414|0.199632|0.261762|5/48|
|512|0.134871|0.169778|0.211646|2/48|

This variation, concentration and persistence under FP64 support **finite-feature
sampling/alignment error** as the proximate explanation in these frozen cases.
Gaussian positive features have an exponential dependence on projected Q/K;
128 sampled features need not accurately normalize all causal key distributions.
An unbiased unnormalized kernel estimator also does not imply an unbiased ratio
of weighted sums. More features improve these aggregate diagnostics, but not
every tail: m256 has a worse maximum here. Projection streams are not nested
across counts. These seeds are diagnostic projections, not new training seeds.
The deployed fixed m128/seed42 projections remain unchanged; no feature count
was chosen using validation/test performance. No bounded computation defect was
found; this does **not** prove acceptable approximation for future learned states,
long histories or full training. The original0.20 **REVIEW_REQUIRED remains**.

## Current allocation and preservation

[One synchronous read-only snapshot](assist50_preparation_20261020_v1/RESOURCE_OBSERVATION.json),
14:50:30UTC: only exposed device **MIG-d2fcfd79-720e-5218-82c9-0a132be68920**,
H1003g.40gb, parentGPU-5a568e3c-e96c-169f-b063-f4478340ff51. NVML lists no compute
processes; visible device-handle scan lists no holder/conflict/inaccessible owner.
Free memory42,238,803,968bytes (39.338GiB); total39.375GiB. Container limit90GiB,
usage18.955GiB at observation, inherited OOM counter2/under_oom0.

LightSAKT2 has no visible trainer and its existing hold remains
USER_TRAINING_HOLD_ACTIVE with unchanged SHA65214c2c53852024e970fd0a176f71968f204ec0748635a7e6f46de3ed64c234.
No process was signaled or job/schedule/configuration modified; no LightSAKT2,
DKT/EdNet, original ELU+1 or earlier FAVOR+ artifact was written.

The prior explicit **allocation-owner confirmation/transfer** in this conversation
is the existing allocation authority. No separate new GPU allocation or external
scheduler reservation proof has appeared. Visible idleness is not ownership proof;
this observation cannot exclude invisible/future reservations. Kubernetes403 is
not a denial and is not being used to demand repeated administrator confirmation.
The existing assigned research MIG is the candidate for continuation, subject to
new scoped experiment approval and fresh fail-closed admission immediately before
CUDA. If a separate allocation is actually required or a conflict appears, the
needed resource is exclusive compute use of this UUID (or an explicitly assigned
alternative with updated device-specific admission), ≥10GiB free including8GiB
allocator budget+2GiB overhead margin, ≥4GiB process RAM+2GiB shared headroom,
for at most45min. No resource transfer/change is performed here.

## Concrete proposal awaiting approval

[Frozen proposed handoff](assist50_preparation_20261020_v1/PROPOSED_CONTINUATION_PLAN.json),
prepared by [CPU-only contract helper](../../ktbench/favor_plus_continuation_contract.py):

- Resume checkpoint20 above, preserving all weights, four fixed projections,
  complete Adam state/hyperparameters, saved RNG and cursor20. The first new
  optimization target is original sampler entry21, matching the saved next-step
  probe. Never seed after restoring RNG or reset optimizer state/counters.
- Exactly30 additional Adam calls: total updates21–50, planned stage boundaries
  30/40/50. Atomic no-overwrite checkpoints at25/30/35/40/45/50. Verify restored
  predictions/state and next-step digest without Adam before continuing stages.
  Freeze the first51 original epoch1/seed42 identities: the51st is probe-only
  after50. The old20 identities, six diagnostics and32 validation sessions match.
- Genuine ASSIST training only, original architecture/Adam/data/protocol,
  m128/fixed seed42 projections, B1/padded-cost≤2048. The next30 training sessions
  contain1438 labels, target lengths1–67, histories≤8, padded costs≤515. This still
  does not cover original history15/B64/cost32768 backward shapes.
- Same six training diagnostics and same frozen32 validation sessions at
  20/30/40/50 only, explicitly eval mode. First reproduce the saved checkpoint20
  baseline before applying21. Record metric definitions/sample identity hashes,
  per-update losses/gradient/model/Adam finiteness and deltas, stage Q/K/error/ESS,
  CUDA allocated/reserved peaks, RSS, synchronized and end-to-end timings.
- Keep0.20 REVIEW. Keep all earlier severe stop criteria: feature-kernel abs>5e-5,
  stage median relative error>0.75, any complete-prefix error>1.5, median stage
  pair-ESS<1.25, Q/K p95 growth>2× initial, gradient L2>100, BCE>20; always stop for
  nonfinite values, causal/reload failure, unknown/compute holder, resource/time
  violation.8GiB allocator/4GiB process-RSS/2GiB headroom,120s/update,45min ceiling.
- Stop at50 or first failure; no automatic failed-step retry, test evaluation,
  extra seed, feature-count training, full training or LightSAKT2 interruption.

The current controller deliberately rejects50 and its original contract remains
sealed. A **separately versioned execution controller and explicit migration
receipt must be implemented/tested against this handoff before launch**. The new
helper is preparation, not an executable50-update training runner or GPU permit.
Its approval checker requires exact parent SHA, plan hash,30additional/50total
scope and actual user-approval evidence. A fabricated JSON record is not consent.

|Total update|Status|Frozen sampled validation BCE|Frozen sampled validation AUC|
|---:|---|---:|---:|
|20|Historical, checkpoint verified now|0.741282668|0.508313409|
|30|NOT RUN; approval pending|—|—|
|40|NOT RUN; approval pending|—|—|
|50|NOT RUN; approval pending|—|—|

The20 metrics use a proper held-out **validation split sample** (3157 labels),
original chronological within-student retrospective teacher-forced protocol;
they are not final full-validation/test performance. They do not establish
superiority, convergence or statistical significance. Original timing/feedback
availability qualifications remain. No new predictive metrics were calculated.

## Targeted verification and recommendation

**71 CPU tests PASS, zero failures:**19 new preparation/approval/sampler/cap cases
plus52 existing recovery/atomic checkpoint/resource controls. No229-suite repeat.
[Log](assist50_preparation_20261020_v1/targeted_cpu.log),
[JUnit](assist50_preparation_20261020_v1/targeted_cpu.xml).
The new independent CPU replay took5.219s on one low-priority thread; it performs
no real-data training or GPU work. Original hashes/receipts remain addressable.

**GO conditionally for the requested30-update continuation**, after explicit user
approval, versioned controller validation, exact checkpoint20 baseline/next-step
reproduction and fresh resource admission. **REVIEW for approximation quality;
NO-GO for full training.** At50, assess stability separately from preliminary
sampled predictive changes; flat/worse AUC is a research outcome, not grounds to
hide results or alter thresholds. Subsequent work needs a new decision based on
learned-state tails/concentration and original long-history/batch GPU profiling.
This preparation has no new GPU memory/throughput or30/40/50 results to report.
