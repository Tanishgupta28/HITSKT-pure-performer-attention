# Memory-efficient FAVOR+128/512 training preparation

25 October 2026; branch `research/hitskt-favor-plus-v1`, base
`457322143e64b4366ff47a810cb4573a1e08ecdc`.

**Recommendation: physical batch 2, with supervised-target-weighted gradient
accumulation over the existing logical batches of up to 8.** Profile physical
batch 4 as a common speed candidate. Choose it only if both feature-count arms
meet the prospective memory-headroom criterion. No attention equations,
precision, optimizer settings, dataset splits, seeds, masks or projection policy
are changed. All historical checkpoints/results remain preserved.

**CPU preparation is verified; new GPU memory/Adam runtime is NOT MEASURED.**
The user's latest instruction requires a separately authorized CUDA execution
window. A concrete <=15-minute / <=160-sandbox-update request was presented;
no approval was assumed. No CUDA, GPU probe, actual-data optimizer update,
seven-hour campaign or final-test inference was executed in this task.

## Previous measurements verified

[Raw-receipt verification](microbatch_memory_20261025_v1/PRIOR_MEMORY_REVERIFIED.json)
recomputed the exact previous peak values from both full-shape and timing
receipts, without a GPU query:

| Original physical B8, zero optimizer updates | FAVOR+128 | FAVOR+512 |
|---|---:|---:|
| Peak allocated, GiB | 2.335866 | 7.674421 |
| Peak reserved, GiB | 4.394531 | 7.988281 |
| Real Adam execution measured? | No | No |

The old512 reserved peak leaves only 12 MiB under the 8 GiB cap. Its moment-sized
zero buffers did not exercise real Adam kernels/state initialization. These
numbers are valid historical measurements, not forecasts for the new policy.

The attention implementation already recomputes causal scan chunks in backward.
It retains per-feature prefix boundary states and log-feature/value inputs.
The ActionEncoder flattens up to 15 historical sessions per example: a physical
B8 can therefore process up to 120 history rows, with padding to the longest
history sequence. Splitting physical batches bounds those simultaneous rows and
activations; it does not truncate histories, detach their gradients or alter
causality. The target-side encoders retain full sequences as well.

By comparison, all 6,931,457 float32 parameters occupy 26.44 MiB; two Adam
moments occupy another 52.88 MiB, excluding counters/workspace. Adam's default
CUDA foreach path can use extra intermediate memory of approximately one
parameter set. That is important to measure, but too small to explain a
multi-GiB activation peak. We retain the original `foreach=None`, `fused=None`
behavior instead of changing the optimizer silently. [PyTorch 2.4 Adam documentation](https://docs.pytorch.org/docs/2.4/generated/torch.optim.Adam.html)

## Matched training policy and objective

The new [microbatch helper](../../ktbench/favor_plus_microbatch.py) partitions each
existing logical batch contiguously. It calls `zero_grad` once, performs every
microbatch forward/backward, checks the accumulated gradients and then calls
Adam exactly once. No computation graph, device batch or live logits are kept
between microbatches. Metrics detach before loading the next microbatch.

For logical group G with N supervised targets, optimize:

`L_G = sum_j [n_j / N * masked_mean_BCE(microbatch_j)]`.

This preserves the original masked mean over every supervised target in the
logical group. Simply dividing each microbatch mean by the number of microbatches
would change the objective when session lengths differ. There is no learning-rate
rescaling, clipping, extra Adam call, dropped final group or accumulation across
the original logical-group boundaries.

Both feature-count arms keep seed42, identical fresh learned parameters and
post-factory CPU RNG, stage projection seeds42–45/common512 pool, fixed projections,
FP64 attention accumulation/FP32 neural weights, the original Adam settings and
the same logical schedules. The architecture uses per-example LayerNorm, not
batch-statistics normalization. Each completed logical update is a safe checkpoint
boundary; partial accumulation is never silently resumed/retried.

**This is not bitwise-equivalent to physical B8 training.** Microbatching changes
dropout kernel calls/mask assignment and floating-point reduction/matmul order.
Without dropout, independent CPU full-batch gradients/Adam updates agree within
the declared numerical test tolerances. A fixed microbatch policy is reproducible
within itself, including exact dropout/RNG checkpoint continuation. The comparison
is new128 versus new512 with the same policy, not a claim of an identical trajectory
to the historical native128 run (best validation AUC0.703467).

## Complete-data accounting

The original data identity (11 files), chronological within-student splits,
rolling histories and retrospective teacher-forced prediction protocol are
retained. Each epoch contains 7,060 training target sessions and 566,107 supervised
labels; full validation contains 1,977 sessions /125,800 labels. Initial context-only
interactions retain their original role. No additional interaction is discarded.

| Policy | Physical training forwards over eight epochs | Adam updates per arm | Total padded-token slots | Maximum physical padded cost |
|---|---:|---:|---:|---:|
| Original physical8 | 7,499 | 7,499 | 38,029,141 | 32,751 |
| Physical4, accumulate within original8 | 14,506 | 7,499 | 36,246,672 | 29,942 |
| **Physical2, accumulate within original8** | **28,494** | **7,499** | **33,730,639** | **13,327** |
| Physical1, diagnostic accounting only | 56,480 | 7,499 | 30,114,696 | 7,164 |

Full validation uses the same frozen logical groups/label order, split physically
as needed: 522 forwards at physical4, 1,010 at physical2. The original per-epoch
logical update counts remain935/937/943/939/939/934/938/934. Smaller batches
without accumulation would change the Adam-update budget; that option is not
selected for this comparison.

Independent CPU collation checks compare every real target token, shifted
correctness input, label, prediction/attention mask, historical response token,
EOS, history order and ownership for the representative actual-data extrema.
The complete eight schedules also preserve all example indices, order and target
counts. See [actual-data coverage/mask evidence](microbatch_memory_20261025_v1/ACTUAL_DATA_COVERAGE_AND_MASKS.json).

Length bucketing already exists in the original sampler. A separate CPU-only
within-logical-group length-sort calculation reduces remaining padded slots by
about4.39% at physical2 and3.53% at physical4. It changes per-example stochastic
assignment/order and is **not selected or executed**. Global regrouping would
also change logical Adam membership and is outside this matched protocol.

## Analytical memory comparison: not measured VRAM

The retained-component calculation enumerated all logical groups and their
actual physical subgroups. Its largest sums include retained scan boundaries,
log features/values, selected FF activations and persistent weights/gradients/
moment-sized state. They omit parts of backward workspace, allocator behavior
and driver overhead. **Do not interpret them as GPU peaks or admission proof.**

| Physical batch | 128 largest accounted component sum, GiB | 512 largest accounted component sum, GiB | Actual new allocated/reserved memory |
|---|---:|---:|---|
| 1 | .3441 | .6939 | NOT MEASURED |
| **2** | **.5624** | **1.2215** | **NOT MEASURED** |
| 4 | 1.1447 | 2.6384 | NOT MEASURED |

The largest post-split retained-component group differs from the old B8 extremum;
it is explicitly added to the new frozen GPU cases. Both validation post-split
extrema are also included. No linear extrapolation of the old 7.988 GiB reserved
peak is used to certify a fit. No precision reduction, projection redraw,
attention chunk-size change, parameter freeze or new activation-checkpoint policy
is introduced.

## Prepared real-Adam CUDA profile

Implementation: [profile/policy](../../ktbench/favor_plus_micro_profile.py),
[epoch/checkpoint adapter](../../ktbench/favor_plus_micro_epochs.py),
[CLI](../../scripts/run_favor_plus_microbatch.py).

Final CPU-prepared namespace:
`experiments/hitskt_favor_plus_v1/assist2017/assist2017_orf128_512_logical8_micro_20261025_v3/`.
Plan SHA256: `362a7f8b4db4625d8a6e1d0773d9bab57e7a0f3118217a5e825a61b032a8d632`.
The [metadata inventory](microbatch_memory_20261025_v1/PREPARATION_MANIFEST.json)
and [exact-byte archive](microbatch_memory_20261025_v1/cpu_prepared_plans_v1_v2_v3.tar.gz)
record the plan, prior CPU drafts and false scope templates.

Prospective profile design, requiring new execution approval:

- One absolute **900-second maximum** for the finite sequential campaign.
- Four fresh sandbox workers: physical2/128, physical2/512, physical4/128,
  physical4/512. Each uses the genuine complete training examples, original
  model/Adam settings and fresh seed42; no historical trained weights imported.
- Per worker: nine frozen representative/extreme logical training groups plus
  15 equal-frequency cost strata measured twice = **39 real Adam updates**.
  Total planned **156 updates**, within the requested <=160 ceiling.
- Eight fixed validation timing strata plus the two validation extrema, with no
  optimizer and no sampled AUC-based selection. Initial/final fixed TRAIN
  approximation diagnostics keep the unchanged0.20 REVIEW and severe stops.
- Every worker logs loss, target count, finite-gradient/norm checks, real Adam
  counters/moments, actual Adam-only and complete-body time, allocator peaks,
  shared/host telemetry and runtime. Fixed projection hashes must remain equal;
  learned parameters must actually change.
- Save a separate atomic, hash-verified **PROFILE_ONLY**, format0 checkpoint with
  real optimizer/RNG state. A fresh worker must exactly recover predictions,
  Adam/RNG and the next proof dropout/gradient computation, with zero further
  optimizer steps. These sandbox weights cannot be loaded as training checkpoints
  or used to initialize the actual comparison.

**Prospective configuration acceptance:** both arms must have finite/correct
updates and exact fresh recovery, with peak **allocated and reserved <=6 GiB**.
This leaves at least2 GiB under the unchanged hard8 GiB cap. A below-cap candidate
above6 GiB is recorded `REVIEW_INSUFFICIENT_HEADROOM` and cannot be selected for
training; it is not relabeled as passing. It can still yield valid negative
profiling evidence. A real resource/CUDA/nonfinite/severe-diagnostic failure stops
the whole campaign, preserving evidence; there is no automatic retry/resizing.

Shared policy remains: assigned authorized shared H100 MIG
`MIG-d2fcfd79-720e-5218-82c9-0a132be68920`, hard8 GiB allocator, advisory10 GiB
total,16 GiB free at admission/6 GiB during execution, own host RSS4 GiB/shared
headroom2 GiB, original utilization/fallback/visibility checks and30-second logical
step ceiling. The existing0.1-second pacing now applies after each physical
training microbatch, with pressure checks before and after it. Protection is
retained or more conservative. PID namespace attribution remains unresolved;
no exclusive-use claim, other-user process signals or allocation changes occur.

Only after actual approval and creation of `ACTUAL_PROFILE_SCOPE.json` from the
false `PROFILE_SCOPE_TEMPLATE.json`, with real approval evidence and fresh clocks:

```bash
cd /workspace/capstone
CUDA_VISIBLE_DEVICES=MIG-d2fcfd79-720e-5218-82c9-0a132be68920 \
PYTHONPATH=. PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 223 nice -n 10 python scripts/run_favor_plus_microbatch.py profile-campaign \
  --plan experiments/hitskt_favor_plus_v1/assist2017/assist2017_orf128_512_logical8_micro_20261025_v3/prepared_plan.json \
  --scope reports/favor_plus_v1/microbatch_memory_20261025_v1/ACTUAL_PROFILE_SCOPE.json \
  --execute-cuda
```

Neither the profile nor a seven-hour/eight-epoch run is authorized by this
document or the prepared false records.

## Runtime estimate and exact future training command

**New eight-epoch pair runtime: NOT ESTIMABLE reliably before the real-Adam
profile.** The old4.90h subtotal/6.13h with reserve belongs to physical B8 without
measured Adam arithmetic. It cannot be carried over to the new microbatch policy.
Physical2 performs3.80 times as many forward/backward calls, while processing
less padding and keeping the same Adam count. These counts are not a GPU speed
measurement. Pacing alone adds a floor of1.58h for both physical2 arms, or0.81h
at physical4, before computation, guards, validation and checkpoint I/O.

After successful profiling, `estimate` weights the measured real-Adam logical-body
times by all7,499 groups, adds nine full-validation passes, prior measured I/O/
diagnostic/startup proxies, and the same declared25% planning reserve. It states
which terms remain proxies and is not a learned-state wall-time guarantee.
`training-template` chooses the fastest **common** headroom-qualified candidate
using memory/runtime only. No feature-count/validation/test performance tuning
enters that choice. Runtime approval remains a separate step.

The epoch pipeline is executable with separate format6 checkpoints. It preserves
model/Adam/RNG/projections, completed logical cursor/metric prefixes and full
source/data/config identities. Full validation after every complete epoch reports
AUC, accuracy, precision, recall, F1 and BCE; strict full-validation AUC selects
the best checkpoint, with equal eight-epoch budgets and no performance early
stopping. Fresh exact recovery is mandatory between epochs and selected-checkpoint
validation replay is required at the end. Full-validation comparison/curves are
saved; no final-test evaluation occurs. Profiles/sandbox weights never seed this
training. Both arms start freshly and sequentially under immutable global/arm
clocks. Stop on failure or expiry; do not run incomplete epochs as completed ones.

After the profile, generated false training template, separate actual training/
runtime approval and actual `ACTUAL_TRAINING_SCOPE.json` preparation, the command is:

```bash
cd /workspace/capstone
CUDA_VISIBLE_DEVICES=MIG-d2fcfd79-720e-5218-82c9-0a132be68920 \
PYTHONPATH=. PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 223 nice -n 10 python scripts/run_favor_plus_microbatch.py campaign \
  --plan experiments/hitskt_favor_plus_v1/assist2017/assist2017_orf128_512_logical8_micro_20261025_v3/prepared_plan.json \
  --scope reports/favor_plus_v1/microbatch_memory_20261025_v1/ACTUAL_TRAINING_SCOPE.json \
  --execute-training
```

This command refuses absent/unapproved scopes, missing source-bound real-Adam
profiles, insufficient headroom or missing exact fresh recoveries. Current
`ACTUAL_PROFILE_SCOPE.json` and `ACTUAL_TRAINING_SCOPE.json` do not exist.
Do not use the previous physical-B8 seven-hour launch command for this new policy.
The checkpoint count remains312 per arm; allow approximately60 GiB storage for
the pair plus headroom. New actual VRAM/host peaks and a defensible runtime request
must come from the authorized profile, then the first complete trained epoch.

## CPU verification, failures and readiness decision

[Final new tests](microbatch_memory_20261025_v1/cpu_final49.xml): **49 PASS**.
The earlier combined CPU receipt records110 passes, including62 existing
controller/profile/clock regressions. There are111 distinct passing tests across
those receipts; the full original229 suite was not repeated. Tests include
independent unequal-target-weight full-batch gradient/Adam comparisons for both
feature counts, masked variable padding, no premature Adam on a failed microbatch,
exact dropout checkpoint continuation, fresh-process recovery, format isolation,
hard8 GiB/shared headroom/expiry/approval checks and a simulated finite34-stage
eight-epoch workflow. Synthetic CPU optimizer steps are not real ASSIST results.

Initial failures are preserved: a no-overwrite check ran after validation of an
intentionally stale test payload (moved before payload construction); one resource
mock lacked its required timestamp (fixed mock, guard unchanged); a real-data
CPU reload exposed a persisted string path passed to a Path-only hash helper
(fixed conversions, added tests and mandatory preparation reload). CPU drafts v1
andv2 remain unchanged. Final v3 also augments the post-split worst cases before
any GPU execution. Logs/source snapshots retain these failures and revisions.

| Decision | Status |
|---|---|
| Original B8 memory measurements | VERIFIED unchanged |
| Complete-data accumulation/padding/objective/recovery | CPU PASS |
| Physical2 recommendation | Preferred candidate, GPU fit NOT VERIFIED |
| Physical4 speed alternative | Candidate, GPU fit NOT VERIFIED |
| Real-Adam profile and measured new runtime | PREPARED, NOT RUN; new window required |
| New approximation or epoch AUC/F1 result | NOT MEASURED |
| Start actual matched epoch training now | **NO-GO pending profile and separate runtime approval** |

The shortest next step is the prepared bounded real-Adam profile. Select one
common configuration only after measured headroom/recovery pass, obtain its
measured training budget, then use the first complete epoch as a mandatory
resource/numerical/validation gate. Keep reporting approximation REVIEW findings;
a memory optimization does not resolve learned fixed-projection approximation
error or establish convergence, test performance or baseline superiority.
