# FAVOR+ shared-GPU preparation —23October2026

## Latest scope and outcome

User confirms GPU intentionally shared; exclusive reservation and reliable host
PID mapping are not normal prerequisites. Prepare SEPARATE shared mode, CPU-test
first and report before GPU execution. Existing shared-use permission accepted;
no new administrator/exclusive-allocation request. No CUDA/real training in this
task; total20. Historical PID mappings remainUNKNOWN; this is a changed resource
contract, not proof of identity. Old exclusive guard and28 bound sources frozen.

Report: `reports/favor_plus_v1/SHARED_GPU_MODE_PREPARATION.md`.
New code: ktbench/favor_plus_shared_gpu.py, favor_plus_shared_assist50.py and
scripts/assist50_favor_plus_shared.py. Explicit record/mode and new source-bound
plan, no implicit legacy fallback. Model/optimizer/science unchanged.

## Policy and verification

Versioned policy JSON under reports/favor_plus_v1/shared_gpu_v1, SHA
4867c950fe93e10720b3b8a967289e2d6be8fa9e7caa151479fc3340584e1f19.
1GiB allocator+reserved cap before model,2GiB ADVISORY total incldriver allowance
(not hard driver cap),8GiB free admission/6GiB runtime,4GiBRSS/2GiBhost headroom,
30supdate/45mincampaign, B1/cost2048, .1s self-pacing. Supported admissionutil>75
defers; unsupported MIG utilization explicit UNKNOWN with resource fallback.
Two fresh admission samples, <=5s telemetry, per-step/Adam/checkpoint/eachdiag/
validation sample monitoring. Peer PIDs/holders/gaps/arrivals/departures logged;
no identity alias/ownership claim. Unsafe resources/CUDA/numerical failure stops
HiTSKT only, failure cache/no retry; Light hold retained, no other jobs changed.

90 tests PASS(58new shared policy/controller +32legacy regressions),0failures/
skips. Synthetic CPU20→30→40→50/recovery/atomic25..50 and partial-Adam-error
accounting exercised, not actual ASSIST training. Actual parent20/61 Adam state
CPUreload exact/component hashes valid, no CPU claim about GPU RNG replay.
Prior229/18scientific tests inherited, not repeated. Actual read-only NVML06:00
samples39.34GiBfree/unsupportedutil/no compute/host~30.7GiBheadroom, eligibility
PASS but NOT reservation/permission. Collector no torch/libcuda. No newGPUcall.

## Exact continuation and next action

Plan: `experiments/hitskt_favor_plus_v1/assist2017/assist50_shared_prepare_20261023_v1/prepared_plan.json`.
SHA190dcb1c271035a111bc0b60f6f27b604e73b6141c099d66c3825fdcc5a12d0c.
Parent20SHA0388e984b8bed21210455be561524a041c892b98b2a87b50fa3ecb1c3ade86f4.
Same128/42/fixedprojections/Adam/RNG/sampler51IDs/data11/frozenval3157labels;
config includes old historical8GiB limit but NEW execution_limits resource cap
is1GiB, explicitly separated/sealed. Original.20 REVIEW/severe stops unchanged.
First20 GPUno-step recovery AND exact frozen validation20 before21; planned
30/40/50 and fresh zero-Adam recovery, no51 optimizer call. Checkpoint format2/
legacy continuation validator reused with distinct sealed shared contract;
shared plan kind prevents accidental exclusive-CLI execution.

Await actual budget/window acknowledgment and instruction to execute after this
CPU-first report. Template execution_approved/budget acknowledgment false;
do not fabricate. Existing shared GPU permission true. No exclusive/newadmin
grant needed; coordination can be documented by user/shared workload coordinator.
Then future fresh scoped record/window/admission and shared CLI only, no full
training. New-cap CUDA behavior unverified; GO bounded pilot conditional on
resource coordination, NO-GO full training. Shared memory races/untracked driver
overhead/unknown utilization/in-flight kernels remain risks; throughput not
isolatedbenchmark. Preserve Light/DKT/original results/all failed campaigns.
