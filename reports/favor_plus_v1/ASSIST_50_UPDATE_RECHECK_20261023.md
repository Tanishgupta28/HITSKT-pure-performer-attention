# ASSIST FAVOR+ exact20→50 continuation: 23 October recheck

Branch `research/hitskt-favor-plus-v1`; base
`bd076beeb4b198c7e51363793aa5e6056276ca01`.

**RESOURCE-BLOCKED before update21. Training remains at20 total updates.**
The assigned MIG was idle at admission, and exact checkpoint20 CUDA recovery
passed. A new NVML-versus-local PID mismatch then stopped the unchanged guard.
No additional Adam call, checkpoint30/40/50, new training loss or new validation
metric exists. No retry was attempted. This report supplements, and does not
replace, the previous20-update, failed50-update and attribution reports.

## GPU availability, authorization and Parth's reported job

The user reports Parth acknowledged concurrent work on the assigned GPU and said
he would stop. **No compute workload was observed before this attempt or after
its exit.** This supports that previously observed contention cleared; the
identity and individual shutdown status of Parth's job cannot be independently
established. His statement does not identify every historical/current GPU PID.

[Read-only namespace/resource capture](gpu_recheck_20261023_v1/RESOURCE_AND_NAMESPACE_EVIDENCE.json),
23October00:52:35UTC:

- Exact assigned **MIG-d2fcfd79-720e-5218-82c9-0a132be68920**, H1003g.40gb,
  parent **GPU-5a568e3c-e96c-169f-b063-f4478340ff51**, GI2/CI0. One exposed MIG.
- Native MIG NVML compute count0, used39,780,352bytes (37.94MiB),
  free42,238,803,968bytes (39.34GiB), total42,278,584,320bytes (39.375GiB).
  No visible unknown device holder, inaccessible descriptor owner or LightSAKT2
  trainer. This is broader evidence than an empty local process list alone.
- Host/container RAM limit96,636,764,160bytes (90GiB), observed use
  62,844,764,160bytes; sufficient for unchanged4GiB process and2GiB headroom
  gates. Inherited OOM count2, not a new OOM.
- LightSAKT2 hold active, unchanged SHA256
  `65214c2c53852024e970fd0a176f71968f204ec0748635a7e6f46de3ed64c234`.

[Supplemental native NVML observation](gpu_recheck_20261023_v1/SUPPLEMENTAL_NVML.json),
00:53:44UTC: MIG compute, graphics and MPS APIs all return success/count0.
Parent process queries return status4 (permission denied), consistent with
`nvidia-smi`'s insufficient-permissions display; they are not treated as empty
observations. Utilization rate returns status3 (unsupported); no percentage is
invented. No CUDA/torch was loaded by these diagnostic collectors.

Existing explicit owner-confirmed research-allocation/transfer authorization
was retained. No new administrator lease was claimed, allocation changed,
Kubernetes request repeated or permission re-requested. The fresh operator
window only records that permission and the already approved bounded scope.
[Prospective admission assessment](gpu_recheck_20261023_v1/CURRENT_ADMISSION_ASSESSMENT.md)
distinguishes successful MIG queries/current absence of compute from unavailable
full-host visibility and from the unexplained old PID pair. No known conflicting
reservation or explicit restriction was supplied. At00:55:12UTC the
[unchanged admission controls passed](gpu_recheck_20261023_v1/resource_admission.json).
Availability was rechecked before CUDA; unknown identities remained blocking
throughout execution. That initial pass did not certify future PID identity.

## Exact recovery passed; runtime attribution failed

Fresh, separate namespace:
`experiments/hitskt_favor_plus_v1/assist2017/assist50_continuation_20261023_v1/`.
Prepared plan SHA256
`d68175950f185a7ee44d540ad1091e048cebb412ba27567edb552237f44a929b`.
Both failed20October attempts and the original20 campaign remain unchanged.

Original parent checkpoint20 SHA256:
`0388e984b8bed21210455be561524a041c892b98b2a87b50fa3ecb1c3ade86f4`.
Same m128/seed42/fixed projection matrices/Adam and nonattention settings,
chronological within-student retrospective protocol, B1/cost2048, frozen samples,
source/data/configuration contract,8GiB allocator and45min ceiling. No seed or
optimizer reset after restoration; no feature redraw or attention change.

[Fresh CUDA recovery receipt](../../experiments/hitskt_favor_plus_v1/assist2017/assist50_continuation_20261023_v1/stage_030/incoming_recovery_and_migration.json)
records **PASS**: model/Adam/RNG/projection integrity, exact fixed evaluation
predictions (maximum absolute difference0), exact next train-mode logits/loss/
all-parameter-gradient digest, **zero Adam calls**. This establishes checkpoint
reliability for the tested proof; it is not a new full validation-sample replay.

At00:55:23.733549UTC, before diagnostics20 and update21:

|Observation|Value|
|---|---|
|HiTSKT `os.getpid()`|3843909|
|NVML MIG compute PID|488211|
|NVML process memory|578,813,952bytes (552MiB)|
|MIG memory used at failure|627,769,344bytes (598.69MiB)|
|Other visible descriptor holders / LightSAKT2 trainers|None observed|
|Trusted mapping between these PIDs|Unavailable|
|Guard outcome|ResourceConflict; stopped before any new Adam call|

[Failure record](../../experiments/hitskt_favor_plus_v1/assist2017/assist50_continuation_20261023_v1/study_failure.json)
preserves the conflict and cached last safe resource state; failure handling
made no CUDA calls. No partial optimizer step, new update record, new checkpoint
or new validation prediction file exists. This campaign is quarantined and
must not be rerun. Process execution3.498361s covers initialization/recovery/
failure, **not training throughput**. PyTorch peak VRAM/final RSS were not cached
after initialization before the resource stop, so no new peak is asserted. The
NVML memory above is a conflict observation, not an attributed own-process peak.

CUDA environment: PyTorch2.4.0a0+f70bd71a48.nv24.06, CUDA12.5, driver535.161.08,
Python3.10.12, one CPU thread, deterministic algorithms, highest FP32 matmul
precision, TF32 off, cuDNN deterministic/benchmark off; explicit assigned UUID
and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
[Runtime record](../../experiments/hitskt_favor_plus_v1/assist2017/assist50_continuation_20261023_v1/stage_030/runtime.json).
The historical18 staged CUDA tests remain inherited, source-matched prerequisite
evidence; they were not rerun or represented as new tests here.

[Read-only release](gpu_recheck_20261023_v1/resource_release.json),00:55:46UTC:
no MIG compute worker, memory back to39,780,352bytes, no unknown holder or
LightSAKT2 trainer, hold unchanged/OOM count still2. A verified root-owned
`nvidia-smi` with read-only display arguments was observed and correctly
classified by the existing guard; it was not stopped or exempted as compute.
No process was signaled; no watcher, allocation change or automatic retry ran.

## What this establishes about the PID problem

**Direct cause verified:** the boundary rejects a non-self numeric NVML PID
without a trusted identity bridge. **Underlying mapping NOT VERIFIED:** neither
488211↔3843909 nor historical67123↔2847964 can be proved equal or separate.
The repeat of an empty→unknown-PID→empty pattern around our process, with the
same552MiB process-memory observation as the earlier attempt, supports a
namespace/runtime alias hypothesis. It is correlation, not identity proof;
coincident external work is not ruled out. Parth's acknowledgement cannot
resolve that distinction. No claim that he caused either specific stop is made.

Accessible procfs still exposes a single local NSpid, parent namespace lookup
returns EPERM, and no trusted host proc/runtime mapping endpoint is available.
Prior CPU-compiled NVML ABI-layout verification remains applicable; no native
layout change was found or made. No unknown-PID alias, same-account exemption,
memory/timing heuristic or process-name bypass was installed. All28 bound
scientific/controller/guard sources remain byte-identical.

[Exact remaining technical evidence](gpu_recheck_20261023_v1/REQUIRED_PROCESS_ATTRIBUTION.md):
an authoritative host/runtime mapping of NVML PID domain to the container/job,
with boot/start-time/namespace/cgroup/exact-MIG bindings and visibility of other
workers; include proxy/MPS/client mapping if applicable. If historical logs are
absent, retain historical UNKNOWN and establish a reliable future bridge. A
corrected runtime attribution or tested trusted read-only bridge is needed;
another user approval or free-memory snapshot cannot substitute. Nothing was
sent to an administrator or Parth.

## Update20/30/40/50 comparison and numerical interpretation

The20 metrics below are **historical results on the frozen32-session,
3157-label validation sample**, not new replayed metrics or final dataset
performance. Full frozen validation20 replay was the next planned diagnostic
after recovery and was not reached. No test split was evaluated.

|Check|Update20|Update30|Update40|Update50|
|---|---:|---|---|---|
|Total completed optimizer updates|20|NOT REACHED|NOT REACHED|NOT REACHED|
|Last minibatch training BCE|0.697781444|NOT RUN|NOT RUN|NOT RUN|
|Frozen sampled validation BCE|0.741282668|NOT RUN|NOT RUN|NOT RUN|
|Frozen sampled validation AUC|0.508313409|NOT RUN|NOT RUN|NOT RUN|
|Frozen sampled accuracy|0.502692430|NOT RUN|NOT RUN|NOT RUN|
|Frozen sampled precision|0.331785003|NOT RUN|NOT RUN|NOT RUN|
|Frozen sampled recall|0.470366886|NOT RUN|NOT RUN|NOT RUN|
|Frozen sampled F1|0.389105058|NOT RUN|NOT RUN|NOT RUN|
|Action median prefix relative L2 versus exact softmax|0.213145477, REVIEW|NOT RUN|NOT RUN|NOT RUN|
|Fresh recovery|PASS at20, zero extra Adam calls|NOT RUN|NOT RUN|NOT RUN|

At20, all historical real updates/gradients/Adam states were finite. The new
zero-step recovery's all-gradient digest matched that saved state exactly; no
new optimization trajectory was tested. Action error remains above unchanged
**0.20 REVIEW** on4/6 frozen prefixes, with no new learned representation.
The earlier independent FP64 feature-kernel comparison passed, while FP64 did
not remove the softmax discrepancy. Finite-feature sampling/concentration is
supported by the earlier frozen128/256/512 study; no projection count, threshold,
architecture or optimizer setting was changed here. Post20 norm/ESS/error
progression, throughput and numerical learning stability remain NOT VERIFIED.

## Tests, preservation and decision

**32 targeted existing CPU continuation tests PASS**, zero failures/skips,
6.95s. They cover exact-self/unknown NVML rejection, no CUDA on conflict, scope
gating, atomic/no-overwrite checkpoints, Adam/config/projection/RNG corruption,
fresh-process zero-Adam recovery and the50-total bound.
[Log](gpu_recheck_20261023_v1/targeted_cpu.log),
[JUnit](gpu_recheck_20261023_v1/targeted_cpu.xml).
These use bounded synthetic CPU fixtures, not additional real-data updates;
no new test count or repeated full229-test-suite claim is made.

[Streaming verification summary](gpu_recheck_20261023_v1/VERIFICATION_SUMMARY.json)
rehashes all199 original20 campaign files (including5/10/20 checkpoint binaries),
all9 prior failed-attempt files and28 bound source files unchanged. The new
contract retains all prior science/data/frozen sample fields; the new7-file
attempt inventory is saved. Original11 data identities were checked by actual
pipeline preparation/admission. Light hold and unrelated README/SAKT dirt retain
their hashes; no historical ELU+1, LightSAKT2 or DKT/EdNet artifact was modified.
Checkpoint binaries remain preserved locally/ignored as before; Git publication
alone is not a remote binary backup.

**NO-GO for further GPU execution now; NO-GO for full training.** Recoverability
is PASS, but exclusive process attribution during execution is unresolved.
Do not retry from free-memory evidence or relabel the approximation review.
Next action: establish the trusted attribution bridge, then test it on CPU and
use a new source-bound run to resume the SAME original20 state under the already
approved30-additional/50-total scope. Only after30/40/50 recovery and monitored
diagnostics actually pass can the requested research-readiness decision be made.
Full training additionally needs approximation assessment and original longer
history/batch profiling, validation-based checkpoint selection/full-split metric
replay, dataset-specific bounded checks, and separate explicit authorization.
