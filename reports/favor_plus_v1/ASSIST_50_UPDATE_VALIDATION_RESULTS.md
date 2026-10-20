# ASSIST2017 FAVOR+ continuation:50 updates NOT REACHED

20 October2026 UTC; branch `research/hitskt-favor-plus-v1`.
**RESOURCE IDENTITY BLOCKED before update21. Total completed optimizer updates
remain20; additional updates0. No30/40/50 results or new checkpoints exist.**
Checkpoint20 fresh CUDA recovery PASS; Action approximation REVIEW remains.
NO-GO for further GPU work until the runtime process identity is established;
NO-GO for full training. The model has not failed a new training-stability test:
the authorized continuation could not begin safely.

[Verification summary](assist50_continuation_20261020_v1/VERIFICATION_SUMMARY.json),
[prospective design](assist50_continuation_20261020_v1/PREDECLARED_DESIGN.json),
[actual approval/allocation authority](assist50_continuation_20261020_v1/OWNER_AND_CONTINUATION_AUTHORIZATION.md).

## Authorization and what actually ran

The user explicitly approved exactly30 additional updates from checkpoint20,
max50TOTAL, conditional on authorized independent/exclusive GPU resources.
The existing explicit allocation-owner confirmation of the H100/server and
LightSAKT2→HiTSKT research transfer was retained as permission. No new allocation,
scheduler lease, administrator evidence or reservation override was invented.
Design commit **e795a87** precedes execution. Initial control commit76a0f41 and
corrected executed control commit **b255713** preserve all original scientific
sources and the sealed20-update controller/checkpoint contract.

Preflight at17:15:29UTC passed: exact assigned
**MIG-d2fcfd79-720e-5218-82c9-0a132be68920**, H1003g.40gb,
parentGPU-5a568e3c-e96c-169f-b063-f4478340ff51; no visible GPU holder, inaccessible
owner, NVML compute process or LightSAKT2 trainer. Total39.375GiB/free39.338GiB;
container90GiB RAM. Owner permission plus current visible conflict checks
supported initialization, without claiming knowledge of invisible reservations.
[Admission](assist50_continuation_20261020_v1/resource_admission_v2.json).

Scientific settings stayed fixed: genuine FAVOR+,128 Gaussian positive orthogonal
features, fixed projection base seed42/stage offsets, FP32 accumulation/chunk32,
original width256/heads4/FF2048/dropout0.1; Adam5e-5/betas0.9,0.999/eps1e-8/wd0,
no clipping; original preprocessing and chronological within-student retrospective
teacher-forced protocol. B1/cost2048,8GiB allocator,4GiB process RSS,2GiB shared
headroom,120s/update/45min campaign; same six train diagnostic examples and32
frozen validation sessions. No test evaluation, HPO, extra seed or full training.

## Exact checkpoint20 recovery — PASS

Parent SHA **0388e984b8bed21210455be561524a041c892b98b2a87b50fa3ecb1c3ade86f4**,
83,802,810bytes, remains unchanged. CPU preparation validates source/data/config
hashes, all61 finite Adam states/counters20, model/RNG/projections and cursor20.
First51 original ordered train/probe identities are frozen; the first potential
new update is sample21 and sample51 is recovery-only after50. The old checkpoint
is not converted or rewritten; a new source-bound contract records the handoff.

In a fresh CUDA process, constructors/load operations preceded restoration of
the saved RNG. The original checkpoint's strict recovery proof then passed:

- Exact evaluation logits, maximum absolute difference **0**.
- Identical model weights, Adam tensors, RNG and four fixed projection buffers.
- Exact next train-mode logits/loss/all-parameter-gradient digest under saved RNG.
- **Zero Adam calls** during the proof; total optimizer counter remains20.

[Fresh recovery and migration receipt](../../experiments/hitskt_favor_plus_v1/assist2017/assist50_continuation_20261020_v2/stage_030/incoming_recovery_and_migration.json).
Runtime: Python3.10.12, PyTorch2.4.0a0+f70bd71a48.nv24.06, CUDA12.5, NumPy1.24.4,
capability9.0, one CPU dispatch thread/core223, highest FP32 precision/TF32off,
deterministic algorithms/cuDNN on, benchmark off, CUBLAS_WORKSPACE_CONFIG=:4096:8.
The earlier18 staged CUDA cases remain source-matched historical evidence; they
were not repeated or counted as new cases.

## Runtime PID conflict — why execution stopped

Immediately after recovery, the synchronous boundary before diagnostics20
reported NVML compute **PID67123** on this MIG, using578,813,952bytes. The running
HiTSKT process reported **PID2847964**. Its visible `/proc` scan found no other
device holder, but could not attribute the different NVML PID. The strict guard
raised ResourceConflict at17:15:55UTC, **before the fresh full-sample validation20
replay and before update21**. No optimizer call was interrupted or retried.

[Immutable failure and immediate snapshot](../../experiments/hitskt_favor_plus_v1/assist2017/assist50_continuation_20261020_v2/study_failure.json),
[console](assist50_continuation_20261020_v1/stage030_v2.console.log).
The message says “unverified/competing”; **the evidence establishes unverified
identity, not a confirmed competing job**. Different PID namespaces/backend
identity are possible explanations, but no mapping was obtained. The missing
identity is not certified as a harmless monitor or silently treated as self.

One read-only follow-up at17:17:26UTC found both PIDs absent from visible `/proc`
and NVML compute empty; used MIG memory returned to39,780,352bytes. A new CPU
process's OS PID, `/proc/self` and NSpid matched, providing no cross-namespace
mapping. `/proc/driver/nvidia/clients` was unavailable. The NVML entry disappearing
after the HiTSKT process exited is consistent with a self-PID alias, but cannot
prove it or exclude a transient unrelated workload.
[Read-only investigation/release observation](assist50_continuation_20261020_v1/NVML_PID_CONFLICT_INVESTIGATION.json).
No new CUDA context was launched to infer that mapping; no guard was bypassed,
process signaled, watcher started or automatic training restart performed.

**Specific missing evidence:** an authoritative mapping of the NVML compute PID
to this container's CUDA process, or allocation-level process identity that can
distinguish self from unrelated work while the context is live. A server operator
could provide host-side PID/NSpid/container/job mapping and confirm exclusive
compute use of this UUID; an explicitly assigned alternative with verifiable
process ownership also works. Existing user permission is not the missing item,
and Kubernetes403 was not the reason for stopping. No new allocation is required
if the existing allocation's process identity can be established safely.

## Earlier controller failure and correction — preserved

The first run namespace, assist50_continuation_20261020_v1, was rejected before
CUDA initialization: saved Adam `betas` are a tuple, while the JSON plan records
the same values as a list. The original100 passing tests did not exercise that
JSON boundary. The correction compares the **complete exact JSON metadata
content**, while raw Adam tensor/reload equality remains exact. No optimizer
value, numerical tolerance, model equation or dataset setting changed.

The traceback, admission and explicit0-new-update/0-CUDA failure were preserved;
that directory was never reused. A corrected fresh namespace
assist50_continuation_20261020_v2 starts from the same original checkpoint20 with
the unchanged30-call budget. It then reached the resource stop described above.
This was not a retry of a failed optimizer step. Both failed attempts remain
addressable, and neither is represented as completed training.

## Numerical and predictive findings

The previous independent ActionEncoder review remains applicable to unchanged
checkpoint20: median relative L2 against exact causal softmax
**0.213175860 at0 →0.213145477 at20**, four/six prefixes above the unchanged
**0.20 REVIEW threshold** at each saved milestone. Saved GPU feature-kernel
agreement has maximum absolute error1.01250e-6; FP64 scan/reference error2.38698e-15
does not remove the softmax discrepancy. Pair-feature ESS can be near1 and
individual head/query relative errors can approach0.935. Frozen feature-count
comparisons support finite-feature sampling/alignment error rather than the
tested floating-point defect; higher counts do not uniformly fix tails.
[Detailed investigation](ASSIST_20_TO_50_CONTINUATION_REVIEW.md).

No new training occurred, so **Action error progression beyond20, new training
losses, gradient norms, Adam behavior during further optimization, throughput
and learned predictive changes are NOT VERIFIED**. Fresh recovery's exact
gradient digest supports continuation reliability, not training convergence.

|Total update|Status|Training BCE at that update|Frozen validation BCE|Frozen validation AUC|Accuracy|
|---:|---|---:|---:|---:|---:|
|20|Historical, unchanged checkpoint; fresh CUDA recovery PASS|0.697781444|0.741282668|0.508313409|0.502692430|
|30|NOT RUN|—|—|—|—|
|40|NOT RUN|—|—|—|—|
|50|NOT RUN|—|—|—|—|

Historical sampled20 precision0.331785003, recall0.470366886, F10.389105058,
MSE0.270515942. These validation metrics cover the same3157 labels/32 sessions,
not full dataset/test performance. The latest attempt did not newly recalculate
them. Original20 changes were mixed, including worse accuracy/precision/MSE;
no superiority, statistical significance or convergence claim is supported.
The original response-timing and retrospective-evaluation qualifications remain.

## Resource measurements and their limits

The failed CUDA process's recorded execution interval was **3.767602s**, excluding
import/preflight time. It covers initialization/recovery and failure, not training.
At the conflict snapshot total MIG used memory was **627,769,344bytes**; the
unattributed NVML PID accounted for578,813,952bytes. These are device observations,
**not a verified HiTSKT peak allocation**. The failure occurred before the next
cached GPU memory boundary; actual PyTorch allocated/reserved peaks and final
process peak RSS are unavailable. The last safe pre-initialization RSS was
683,474,944bytes, not a final peak. No training throughput can be calculated from
zero updates. Do not substitute the earlier successful20-run peaks as new values.

LightSAKT2 remained stopped/USER_TRAINING_HOLD_ACTIVE, hold SHA
65214c2c53852024e970fd0a176f71968f204ec0748635a7e6f46de3ed64c234.
No Light/DKT/baseline/scheduler/allocation change or unrelated signal occurred.
All199 original20 campaign files, including5/10/20 checkpoint binaries, rehash
unchanged; unrelated root README and SAKT console remain unchanged/outside commits.
New artifacts stay in the separate FAVOR+ namespaces. No `_SUCCESS` or completed
50-study marker was created.
The final CPU-only host check at17:24:35UTC records inherited OOM counter2,
under_oom0 and the same Light hold hash; it performs no additional GPU probe.
[Final host/preservation observation](assist50_continuation_20261020_v1/FINAL_HOST_AND_PRESERVATION.json).

## Tests and next decision

**103 targeted CPU tests PASS, zero failures:**32 new execution/recovery/resource
cases plus71 prior controls. Corrected JSON plan round-trip/fresh-process
no-step recovery/atomic no-overwrite/corruption/optimizer/RNG tests pass. Two final
explicit NVML tests establish that an exact self PID is accepted and a different
unmapped PID still blocks before CUDA statistics. Earlier100/101 logs are kept;
they are not added together as independent test counts. Full229 scientific suite
was not redundantly repeated. [Final test log](assist50_continuation_20261020_v1/targeted_cpu_final.log).

**Decision: NO-GO for further GPU execution until PID attribution is resolved;
NO-GO for full training.** Preserve the exact20 checkpoint and stop. After actual
resource-identity evidence is supplied, the next experiment remains the approved
bounded20→50 continuation, with no Adam reset, same samples/configuration and all
review/severe/resource gates. Use a fresh versioned output rather than reusing a
failed directory, reverify the checkpoint20 baseline, and stop at50. A successful
50 study would still require learned-state approximation-tail and original
history15/B64/cost32768 backward profiling before considering full ASSIST training;
Junyi/EdNet require their own bounded profile and later explicit training decision.
