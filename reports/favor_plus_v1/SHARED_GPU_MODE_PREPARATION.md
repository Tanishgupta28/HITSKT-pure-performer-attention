# HiTSKT FAVOR+: shared-GPU mode preparation

23October2026; base `efa8a21db5fa2bb9c6414a46ddbb558206b47c32`.

**CPU policy/controller verification PASS:90 tests, zero failures/skips.** A
separate shared-mode continuation is prepared from the exact20 checkpoint.
**CUDA execution/training in this task: NOT RUN. Training remains20 updates.**
Shared use is explicitly authorized by the user; exclusive ownership and trusted
host PID mapping are not prerequisites in this NEW mode. Historical guards and
scientific artifacts remain unchanged. PID attribution itself remains UNKNOWN.

## Which assumptions needed changing

|Existing check|Location|Shared-mode treatment|
|---|---|---|
|Requires `exclusive=true` / independent allocation|`ktbench/favor_plus_gpu_access.py:32`|New explicit shared-use permission, scoped plan/policy/window and acknowledged budget; no implicit fallback|
|Rejects unknown local GPU holders or inaccessible descriptor owners|`ktbench/favor_plus_resource_telemetry.py:159`|Record holders/visibility gaps as shared/unattributed activity; preserve protected LightSAKT2 checks|
|Requires empty NVML compute list before CUDA|`ktbench/favor_plus_assist50.py:120`|Peers allowed; successful device-memory/process telemetry and adequate headroom required|
|Every NVML PID must equal local PID|`ktbench/favor_plus_assist50.py:136`|No self/peer alias inferred; all reported PIDs/memory and arrivals/departures logged; stop on unsafe resources/errors|

Those checks are reasonable for an exclusive reservation but do not represent
this intentionally shared environment. None was deleted or changed in place.
New [shared policy](../../ktbench/favor_plus_shared_gpu.py),
[shared controller](../../ktbench/favor_plus_shared_assist50.py) and
[explicit shared CLI](../../scripts/assist50_favor_plus_shared.py) select the
different execution contract. Legacy/exclusive records cannot select shared mode.

An unequal host/container PID is not proof of another researcher; nor is an
equal integer proof of self across unknown namespaces. Genuine concurrent work
is allowed under the shared-use policy. Unidentified activity remains recorded
as unidentified, without whitelisting it as HiTSKT or calling it unauthorized.

## Predeclared resource policy

[Versioned policy JSON](shared_gpu_v1/PREDECLARED_POLICY.json), SHA256
`4867c950fe93e10720b3b8a967289e2d6be8fa9e7caa151479fc3340584e1f19`.

|Control|Shared v1 limit|
|---|---|
|PyTorch allocator / reserved memory|1GiB, cap applied before model allocation|
|Advisory total VRAM budget|2GiB, including1GiB driver/workspace allowance; not a hard driver-memory guarantee|
|Free VRAM at admission / during execution|At least8GiB /6GiB on the exact MIG|
|Admission observations|Two fresh finite snapshots,0.2s apart; no watcher or wait-until-free loop|
|Utilization admission|Defer if supported utilization>75%; missing/error visibility stops|
|Unsupported MIG utilization|Explicit UNKNOWN; conservative cap/headroom/pacing/latency fallback, never reported as0%|
|Self pacing|0.1s pause between optimizer updates; no other job/process is paused|
|Batch / padded token cost|1 /2048, unchanged|
|Process RSS / shared host headroom|4GiB / at least2GiB|
|Per-update / campaign budget|30s /45min; no automatic retry or run beyond50 total|
|Telemetry age|At most5s; stale/malformed/missing critical memory/process data stops|

The fraction API caps the caching allocator, and does not reserve free memory
against other jobs. Driver/context/workspace allocations require extra headroom.
[PyTorch documentation](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.memory.set_per_process_memory_fraction.html).
The installed2.4 implementation/signature was preserved; no library upgrade.
NVML data are dynamic, and compute-process queries do not cover all graphics
activity; device-wide used/free memory remains the pressure source.
[NVIDIA NVML R535](https://docs.nvidia.com/deploy/archive/R535/nvml-api/group__nvmlDeviceQueries.html).

Monitoring occurs before/after updates and Adam, before checkpoint/recovery and
each diagnostic/validation sample. Journal entries retain raw reported PID lists,
memory, utilization status, visible holders and visibility gaps, plus changes
between snapshots. Permission, exact device exposure, LightSAKT2 hold and the
operator window are rechecked at each boundary. Active protected LightSAKT2
causes HiTSKT to yield by stopping; no Light job is modified.

Own allocator counters and global NVML memory are kept separate. A peer appearing
with healthy resources is logged and allowed. Low headroom, allocator/reserved
cap violations, CUDA/OOM/error, time limits, nonfinite state or scientific gate
failure stop HiTSKT. Last validated checkpoints and cached failure telemetry are
preserved; failure handling makes no new CUDA query or failed-step retry. No
peer termination, GPU reset, MPS/clock/quota/scheduler change or background task.

## Exact continuation and scientific preservation

[Actual prepared plan](../../experiments/hitskt_favor_plus_v1/assist2017/assist50_shared_prepare_20261023_v1/prepared_plan.json),
SHA256 `190dcb1c271035a111bc0b60f6f27b604e73b6141c099d66c3825fdcc5a12d0c`.
Parent20 SHA256
`0388e984b8bed21210455be561524a041c892b98b2a87b50fa3ecb1c3ade86f4`.

All28 previously bound sources remain byte-identical; the shared plan binds
those plus3 new entry/policy/controller files. Model, m128 fixed projections,
seed42, Adam, nonattention architecture, precision,11 verified data identities,
original chronological within-student retrospective protocol and exact sampler
order are preserved. Original0.20 REVIEW and all severe numerical gates remain.
ActionEncoder's historical0.2131 REVIEW is not resolved by resource-policy changes.

Checkpoint format/scientific validator are reused unchanged. Shared execution
mode/policy/source hashes are sealed into the NEW checkpoint contract; the
shared plan has its own kind and legacy CLI rejects it. No old plan is edited.
Mandatory first steps are exact parent20 CUDA prediction/next-gradient recovery
and identical frozen-validation20 probabilities/metrics before update21. Then
only21–50 (30 Adam calls), atomic25/30/35/40/45/50 saves and separate fresh
zero-Adam recovery at30/40/50. Entry51 is a proof-only probe, never optimized.
No test split evaluation, dataset trimming, optimizer/RNG reset or full training.

## CPU verification and current metadata

[Final log](shared_gpu_v1/final_cpu.log), [JUnit](shared_gpu_v1/final_cpu.xml):
**58 new policy/controller tests +32 existing continuation regressions =90PASS**.
Tests include peer churn without aliases; authorization/scope/window/policy
rejection; pressure/stale/denied visibility and allocator stops; no CUDA query
after a pressure failure;1GiB cap before model allocation; synthetic20→30→40→50
atomic saves/fresh zero-step recovery; partial optimizer failure accounting/no
retry; and exact evaluation/RNG behavior with added boundary checks. Synthetic
CPU fixture updates are not real ASSIST training. Old229 scientific CPU and18
CUDA cases remain inherited, not newly repeated or counted here.

[Actual CPU checkpoint verification](shared_gpu_v1/ACTUAL_CHECKPOINT_CPU_VERIFICATION.json):
weights/Adam reload exactly, all61 Adam states step20, component hashes for RNG,
projections and sampler valid. No GPU RNG replay is claimed from this CPU check.

After CPU tests, [two read-only resource snapshots](shared_gpu_v1/RESOURCE_OBSERVATION.json)
at06:00:06UTC observed the exact assigned H100 MIG with42,238,869,504bytes free
(39.34GiB), no reported compute worker, and about30.7GiB host headroom within
the90GiB container limit. MIG utilization returned unsupported/status3:
**load UNKNOWN**, not0%. Both snapshots satisfy shared resource eligibility;
they neither reserve capacity nor authorize execution. Collector loaded neither
torch nor libcuda. No CUDA or real optimizer step was run in this task.

## What is required before GPU execution

Existing shared-GPU permission is accepted. No new exclusive allocation,
administrator PID mapping or Kubernetes access is needed. Before launch:

1. Confirm the declared1GiB allocator/2GiB advisory total budget and bounded
   window with the user/shared-workload coordinator; respect any known incompatible
   reservation or cluster restriction. No confirmation from other users is invented.
2. Record instruction to execute this CPU-verified shared policy, the actual
   plan/policy hashes and a fresh operator window. This follows this task's
   explicit CPU-first/report-before-execution boundary, not another GPU ownership
   request. [Template](shared_gpu_v1/SHARED_EXECUTION_RECORD_TEMPLATE.json) truthfully
   confirms existing shared permission while leaving execution/budget flags false.
3. Fresh admission must pass again at execution. Old idle snapshots cannot be
   reused as a guarantee, and the template correctly refuses execution as saved.

Example first stage, **future command only**, using a separate completed record:

```bash
HITSKT_SHARED_PLAN=experiments/hitskt_favor_plus_v1/assist2017/assist50_shared_prepare_20261023_v1/prepared_plan.json
HITSKT_SHARED_RECORD=/path/to/confirmed-shared-execution-record.json
env CUDA_VISIBLE_DEVICES=MIG-d2fcfd79-720e-5218-82c9-0a132be68920 \
  PYTHONPATH=. PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 223 nice -n 10 python scripts/assist50_favor_plus_shared.py stage \
  --plan "$HITSKT_SHARED_PLAN" --shared-record "$HITSKT_SHARED_RECORD" \
  --update 30 --execute-cuda
```

After successful30, use the same command/environment with action `verify-cuda`
and update30. Review diagnostics/resources, then explicitly stage40/verify40,
stage50/verify50. No50→60 endpoint exists and failed/completed campaigns cannot
restart. Keep all new outputs in this separate FAVOR+ run namespace.

## Decision and residual risks

**GO for a bounded shared-mode GPU pilot after the above coordination/execution
record; NO-GO for full training.** CPU control logic and exact preparation passed;
new-mode CUDA cap/pressure behavior and real continuation remain NOT VERIFIED.
The first GPU stage must pass its zero-step exact recovery before any update.

Shared mode provides resource bounds/observations, not exclusive compute or a
hard reservation. Other users may allocate after a snapshot; pressure checks
cannot preempt an in-flight kernel, and an asynchronous CUDA failure can leave
a partially applied step that must never be silently retried. External driver
memory, PID ownership and unsupported utilization remain uncertain. The allocator
cap is not a guaranteed total-VRAM cap. Pacing reduces demand without enforcing
a precise SM share. Shared-load throughput must not be presented as an isolated
GPU benchmark. Scientific approximation/long-history/original-batch and dataset
training validation remain outstanding. Historical PID relationships stay UNKNOWN;
their resolution is not claimed or required by this separate shared policy.
