# FAVOR+ batch-8 profile: verified handoff

25 October 2026. Branch `research/hitskt-favor-plus-v1`; base commit
`8554b588df84691788d3e151021a823f20934a90`.

## Current verified outcome

Read `reports/favor_plus_v1/B8_GPU_PROFILE_AND_TRAINING_BUDGET.md` and its
`b8_profile_20261025_v1/` evidence. User authorized one <=600-second zero-update
GPU profile, not training. It completed four sequential stages in **215.663 s**:
core128/core512 both 9/9 PASS, then stratified timing128/timing512 PASS. No actual
optimizer steps, new trained checkpoint, epoch AUC/F1, final-test evaluation,
other-job signals or watcher. All original artifacts retained.

- Assigned shared MIG `MIG-d2fcfd79-720e-5218-82c9-0a132be68920`;
  8 GiB hard allocator / 10 GiB advisory total / admission16 GiB / runtime6 GiB
  free / own host RSS4 GiB / shared host headroom2 GiB / .1s pacing unchanged.
- 128 peak allocated/reserved **2.335866/4.394531 GiB**; 512
  **7.674421/7.988281 GiB**. 512 reserved margin only **12 MiB**. Actual Adam
  math/workspace/learned-state memory NOT measured; moment-sized zeros only.
- Own host RSS peaks1.705/1.730 GiB. Container limit90 GiB. PID mapping remains
  unresolved; MIG utilization unsupported/UNKNOWN. Shared authorization does
  not establish exclusivity. One after-exit observation: no compute workers,
  39.337952 GiB free; no availability polling or reservation claim.
- Weighted training-body proxy .941668/1.114164 s per batch, includes existing
  guard calls/pacing/finite checks/F-B/transfers but **excludes Adam.step**.
- Actual B8 plan SHA `cb3bbc9168b712d187a0b2e998c01816b204e2683f74d2e3a01bc5cf0d866edb`
  at `experiments/hitskt_favor_plus_v1/assist2017/assist2017_orf128_512_pair_b8_20261025_v2/prepared_plan.json`.
  Complete7060 training sessions/566107 labels per epoch and1977 validation
  sessions/125800 labels; original order/full lengths/label IDs preserved.
  Same eight schedules/initial learned weights/RNG/Adam/settings for both arms;
  **7499 updates per arm**, maxB8, maxcost32751. B8 is not historical1454-step
  native128 FP32/B64 trajectory. New pair uses common512 pool + FP64 attention.

## Budget and prepared execution

Two-hour paired training is **NO-GO**: measured initial-state proxies total
4.9044 h; predeclared25% reserve gives **6.1305 h**. 128 2.82h and512 3.31h
with reserve. Not a completion guarantee: actualAdam, learned states and future
shared load unmeasured. Proposed **7h pair /4h per arm** requires NEW separate
actual runtime/training approval, not granted by current profile instruction.

New `ktbench/favor_plus_pair_long_window.py` and
`scripts/run_favor_plus_pair_long_window.py` prepare that explicit longer scope.
Parent global and per-arm clocks never reset. Finite34 stages use <=1h leases
bounded by immutable outer deadlines; all original GPU/host guards/numerical
engine/physical plan remain unchanged. First failure STOP/no retry. Current
`SEVEN_HOUR_TRAINING_PROPOSAL_TEMPLATE.json` is all-FALSE proposal; actual
`ACTUAL_APPROVED_TRAINING_SCOPE.json` does NOT exist. Exact future command is in
report. Do not use old2h campaign to claim eight epochs fit. Do not run anything
without actual approval. First authorized complete epoch/recovery is mandatory
before proceeding, with unchanged8GiB cap. Need approximately60GiB storage;
~151GiB was free, not reserved.

## Approximation and reproducibility

Fresh six-TRAIN native<=32 diagnostics:128 Action median .202339/max .250014,
3/6 REVIEW above unchanged.20;512 .132655/.181684,0/6. CPE .154539/.171902
versus .057210/.071775;0/6 each. Independent kernel checks passed also on
extended<=128/audit17/101. Later stage Q/K differ by model; not identical-QKV
all-stage comparison. Initial512 improvements do not prove learned error/AUC.
Historical trained approximation REVIEW remains. Historical best validation AUC
.703466903 at7 / maxF1 .532809339 at6 unchanged; no superiority/convergence.

**18 new CPU tests PASS** (7 profile/IO,11 outer approval/clock/workflow), zero
final failures. Previous66 passes retained, not rerun or added as new. Initial
CPU progress-file reuse bug fixed before training; initial failure/source
preserved. Profiler/model/epoch engine sources used by GPU profile unchanged.

73 protected/source hashes match; original7/8/50 checkpoints, Light hold,
unrelated README/SAKT dirt and both originalfailed/newB8 plans preserved.
Metadata archive SHA `de94182ba292d134b790ad6a99d284b72ddd9141cae60b4936005c8753b2d517`
contains1107 exact-byte entries including16 case0 frozen prefixes; manifest
hash-addresses1191 local files.80 other arrays and4 large untrained/non-resumable
IO proxy binaries remain local; no original artifacts deleted.

## Exact next action

Finish scoped report/memory/code publication to `capstone-gpu` research branch,
verify remote HEAD/main/unrelated dirt, then **WAIT** for the user's separate
runtime/training decision. Do not rerun completed profile, initialize CUDA,
start optimizer steps, alter caps/attention/threshold/data, touch Light/DKT or
other users, or infer training permission from the prepared template. If future
approval arrives, bind actual evidence/fresh outer clock and unchanged
source/plan/profile hashes; use exact report command, first-epoch gate, full
validation selection only. Final test stays untouched.
