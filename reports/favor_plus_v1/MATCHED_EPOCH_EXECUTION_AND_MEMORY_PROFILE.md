# Matched FAVOR+128/512 execution wiring and full-shape profile

25 October 2026. Research branch; base `a372c8432e2296410ad5af325468c8fc2e903c90`.

**Execution wiring is implemented. Original full-shape profile:128 PASS,
512 FAIL/STOP. No real ASSIST optimizer updates or new epoch-validation scores
were produced. Do not launch the paired eight-epoch run yet.** A separately
versioned common batch-size-8 follow-up is CPU-prepared; its GPU profile is
NOT RUN, with a new bounded profiling approval pending. It is not an automatic
retry of the failed campaign, nor evidence that the smaller batch fits.

The historical eight-epoch native128 result remains best full-validation AUC
**0.703466903** at7/F1 **0.508813934**, with maximum observed F1 **0.532809339**
at6. Original checkpoints, datasets, results and other workloads are preserved.

## New executable controller

[Controller](../../ktbench/favor_plus_paired_epochs.py),
[CLI](../../scripts/run_favor_plus_paired_epochs.py), and
[66-case CPU receipt](paired_epochs_execution_20261025_v1/cpu_final66.xml).

- Primary ORF128/ORF512 only; genuine FAVOR+, identical FP64 attention accumulation,
  FP32 neural parameters/output, width256/heads4/FF2048/dropout.1 and one layer
  per stage. All shared learned parameters and post-factory CPU RNG hashes match
  at fresh seed42 initialization. No historical learned weights/optimizer imported.
- Common nested512 projection pool/stage seeds42–45. This is the declared matched
  draw, not a relabeling of the historical native128 projection. Both arms use
  the same policy; only feature count changes within a particular pair.
- Exact complete chronological within-student retrospective protocol:7060 target
  sessions/566107 training labels per epoch,1977/125800 validation labels. No
  test dataset or inference path is constructed. Response-availability qualifications
  remain; this does not become a student-disjoint/online benchmark.
- Eight complete epochs per arm, full validation after each, strict best-AUC
  checkpoint selection; F1 maxima recorded separately, threshold.5 unchanged.
  Performance patience stopping is intentionally disabled for equal eight-epoch
  exposure; safety/time stops remain active. No epoch9 or automatic longer run.
- Atomic/no-overwrite checkpoint **format5**, separate from formats1–4. Save model,
  Adam, RNG, projection identity, source/data/configuration, exact sampler cursor
  and online metric prefix every25 updates and each completed epoch. Verify
  prediction/dropout/loss/gradient/Adam/RNG recovery in a fresh process after
  each epoch, without another optimizer step. A partial-epoch trajectory is
  resumable through the verified restore/epoch helpers; automatic failed-step
  retries and automatic mid-epoch continuation are disabled.
- Validation NPZs carry event IDs, logits, probabilities and labels; frozen ID/
  label hashes are checked before selection. Both arms finish eight verified
  epochs and replay their selected checkpoint before comparison is declared complete.
- Fixed six TRAIN diagnostics retain the historical≤32 independent-kernel/severe
  grid, plus≤128 saved Q/K/V/exact-softmax comparisons and independent audit
  projections17/101. Audit projections never replace model buffers. Extended
  errors retain.20 REVIEW; they do not retroactively change the original grid.
- The finite foreground campaign makes34 staged worker calls:train/recover×8
  and selected validation replay for each arm. It stops at the first failure;
  no watcher, polling daemon or other-process signal is created. Epoch tables,
  curves and a paired validation-only summary are generated after actual execution.

The profile action and training action have separate authorization gates.
Profile permission allows zero optimizer calls for≤600seconds total. Full
training requires an actual new eight-epoch/runtime approval record, including
the proposed≤7200second pair and≤3600second arm ceilings. Those proposed
60-minute arm windows exceed the historical45-minute window and are **not
approved or activated here**. The batch-8 update count means even that proposal
may be inadequate; re-estimate after its profile before requesting training.

## Actual profiling authorization and resource admission

Current user task3 requests bounded representative profiling subject to
authorization, and task4 preserves8GiB. Together with the explicit existing
shared-H100 use permission and referenced10-minute zero-update profile, this
was treated as authority for **that profiling action only**. See
[scope record](paired_epochs_execution_20261025_v1/EXECUTION_SCOPE.md) and
[execution record](paired_epochs_execution_20261025_v1/PROFILE_EXECUTION_RECORD.json).
No training or two-hour runtime approval was inferred.

Actual assigned H10080GB HBM3 MIG3g.40gb UUID:
`MIG-d2fcfd79-720e-5218-82c9-0a132be68920`. Two fresh admission observations
preceded CUDA initialization. Existing shared visibility/pressure/Light-hold
checks remained. No exclusive ownership or guessed PID alias was required.
MIG utilization was unsupported; it remains UNKNOWN. Unknown NVML activity was
logged, not silently identified as self. No unrelated process was changed.

Limits retained:8GiB own allocated/reserved/peak ceiling through allocator
fraction and guard;10GiB advisory total;16GiB admission-free/6GiB runtime-free;
4GiB own host RSS/2GiB shared-host headroom;.1s pacing;30s per profile step.
The advisory total cannot certify own driver/context overhead under unresolved
PID attribution. Whole-device memory is not own-process memory.

Runtime:PyTorch `2.4.0a0+f70bd71a48.nv24.06`, CUDA12.5, H100 capability9.0,
deterministic algorithms, TF32/benchmark disabled, highest matmul precision,
CUBLAS `:4096:8`, seed42, one CPU thread/taskset223/nice10. The global runtime
and allocator implementation were not patched.

## Measured representative full shapes

Source run:
`experiments/hitskt_favor_plus_v1/assist2017/assist2017_orf128_512_pair_epochs_20261025_v1/`.
Plan SHA256 `9192d6636d5792c1727a4dfbfb31cd405432f60cadebb83c7f037dd6e6660fdb`.
[Predeclared cases](paired_epochs_execution_20261025_v1/PREDECLARED_PROFILE_DESIGN.json)
cover all eight schedules, including min/median/max padded cost, maximum chunk
workspace, longest history/target, largest retained-component accounting and
maximum validation cost. Each case uses a full batch with intact examples.
Two Adam-moment-sized buffers were reserved; **Adam.step was never called**.

| Case | Split | Batch | Padded cost |128 F/B seconds |512 F/B seconds |
|---|---|---:|---:|---:|---:|
|0|train|1|363|.846|.867|
|1|train|7|606|.147|.128|
|2|train|4|21299|1.737|2.261|
|3|train|20|31515|1.108|**FAIL during backward**|
|4|train|10|31842|2.336|NOT RUN|
|5|train|14|32218|2.032|NOT RUN|
|6|train|58|32748|.750|NOT RUN|
|7|train|64|32766|.746|NOT RUN|
|8|validation|13|32766|.247 (forward only)|NOT RUN|

These are single synchronized shape measurements, not repeated throughput
benchmarks, optimizer timings or predictive-performance results. Losses and
gradients were finite for every completed case.128 profile weights/projections
were verified unchanged.512 did not finish its final immutability receipt;
the profile has no optimizer operation, but a final successful receipt is not
claimed for the failed arm.

| Measurement |128 PASS |512 completed prefix only |
|---|---:|---:|
|Completed cases|9/9|3/9|
|Validated peak allocation GiB|6.005212|3.841291|
|Validated peak reservation GiB|7.806641|4.539063|
|Minimum observed device free GiB|31.397766|34.438782|
|Peak observed host RSS GiB|1.538620|1.498497|

**512 figures are cached measurements before the failed case, not its
instantaneous failure peak or a full-shape maximum.** Runtime from the scope's
start timestamp to terminal-stop file mtime was31.704seconds; this includes
process/setup/guard overhead and is not CUDA kernel time. All actual values
are retained in [PROFILE_SUMMARY.json](paired_epochs_execution_20261025_v1/PROFILE_SUMMARY.json).

## Allocation failure and secondary NVML assertion

Observed traceback ends in FAVOR+'s recomputed backward `_prefix_chunk` weighted
state operation, then:

`NVML_SUCCESS == r INTERNAL ASSERT FAILED ... CUDACachingAllocator.cpp:841`.

The exact bundled source path is
`/opt/pytorch/pytorch/c10/cuda/CUDACachingAllocator.cpp`.
[Source analysis and hash](paired_epochs_execution_20261025_v1/ALLOCATOR_FAILURE_SOURCE_ANALYSIS.json)
establish:

1. Line841 is the NVML assertion inside `reportProcessMemoryInfo`.
2. The caller at1087 is in the unsuccessful-allocation branch, after verifying
   `params.err == cudaErrorMemoryAllocation` and before emitting the normal OOM
   exception or incrementing the OOM statistic. The NVML reporting assertion
   therefore masks an allocation failure; it is not an attention causality failure.
3. The same source returns `cudaErrorMemoryAllocation` when a request would exceed
   the process memory-fraction ceiling. The stack is consistent with the8GiB
   working-set limit/fragmentation. The last pre-case global free observation was
   34.439GiB and all admission/pressure checks passed; no competing workload
   was identified as the cause. Free global memory does not remove our8GiB cap.
4. The reporting routine queries by physical PCI identity, while the guard uses
   assigned MIG telemetry. Its returned NVML status was not captured, so a
   particular NVML/MIG API failure cannot be confirmed from this assertion alone.

**Confirmed:** allocation-failure path followed by secondary NVML reporting error.
**Not captured:** failed request size, instantaneous own peak, returned NVML error
code or a definitive split between cap pressure and fragmentation. No global
PyTorch patch, larger cap, guessed owner, retry or allocator reset was used.

The failed batch has91 flattened history rows, action length285 and target
length266. At512/FP64, one historical weighted state is.088867GiB; a32-state
component is2.84375GiB before other graph/log-feature/boundary/FF storage.
This is explanatory component accounting, not the failed allocation or a
complete peak estimate. The traceback does not name the attention stage.
See [component analysis](paired_epochs_execution_20261025_v1/MEMORY_FAILURE_COMPONENT_ANALYSIS.json).

After exit, exactly one read-only snapshot showed39.337952GiB free, no reported
NVML compute worker and no visible Light trainer. CUDA remained uninitialized
in that observation process. This is an observation, not an exclusive-allocation
claim or proof about every external job. Other processes were not signalled.

## Explicit common-batch follow-up — CPU prepared, not executed

[Separate preparation helper](../../ktbench/favor_plus_paired_rebatch.py) and
[CPU-only CLI](../../scripts/prepare_favor_plus_paired_rebatch.py) create a fresh
namespace from the preserved failed plan:

`experiments/hitskt_favor_plus_v1/assist2017/assist2017_orf128_512_pair_b8_20261025_v2/`.
Plan SHA256 `cb3bbc9168b712d187a0b2e998c01816b204e2683f74d2e3a01bc5cf0d866edb`.

Both arms use maximum batch8, identical new batch boundaries/order, unchanged
attention/chunk32/precision/weights initialization/seed/Adam settings and all
previous resource ceilings. Each original batch is split into consecutive
whole-example groups; there is **no action truncation, target dropping or
gradient accumulation**. Complete within-example histories are retained. Padded
cost remains at most32751; it is not silently claimed to be2048 or4096.

There are935/937/943/939/939/934/938/934 batches over epochs1–8:
**7499 prospective Adam updates per arm**, versus1454 in the original schedule.
This changes gradient noise and the training trajectory. The paired comparison
remains matched between its two new arms, but it cannot be described as a
feature-only ablation against the old.703467 experiment. A two-hour estimate
cannot be carried over to this larger update count without profiling.

The failed campaign is terminal and unchanged. The new plan is source/data/order
verified on CPU, has false approval templates and no GPU profile. A new bounded
profile approval was requested; pending is not approval. **No new profile or
training was launched automatically.** If its profile also fails, stop again
without further automatic resizing or cap changes.

## Training readiness and actual validation status

| Check | Status |
|---|---|
|Matched two-arm model/data/optimizer/epoch execution wiring|IMPLEMENTED|
|CPU controller/recovery/policy tests|PASS66 distinct (51 new,15 existing factory)|
|Exact fresh-process synthetic model/Adam/RNG/next-computation recovery|PASS for both feature counts|
|Genuine full-data target/order preparation|PASS, both original and explicitB8 schedules|
|Original full-shape128 CUDA profile|PASS9/9|
|Original full-shape512 CUDA profile|FAIL/STOP, allocation failure|
|B8 full-shape CUDA profile|NOT RUN — new scoped approval pending|
|New eight-epoch real-data optimizer updates|NOT RUN,0|
|New epoch-wise validation AUC/F1/BCE|NOT RUN; no values invented|
|Actual CUDA-trained checkpoint recovery for this new controller|NOT VERIFIED|
|Paired two-hour training approval|NOT GRANTED|
|Current full-training recommendation|NO-GO pending successful common-shape profiles and actual runtime approval|

Synthetic CPU unit tests take a few optimizer steps on tiny generated fixtures;
these are not real ASSIST training or AUC evidence. One CPU follow-up preparation
initially failed when converting a relative parent path to a repository-relative
source identity. It was fixed by resolving the path before conversion. The
initial log/source are retained; actual full-data preparation/reload then passed.
This failure occurred before creating the follow-up namespace or any GPU work.

After a successful B8 profile, estimate train+full-validation+diagnostic+
checkpoint/recovery time using its7499-update schedule. Frequent checkpoints
could need roughly50–55GiB for the pair (weights+Adam+metric prefixes, before
logs); confirm storage headroom before training. Do not promise eight completed
epochs in an unmeasured time window. The controller's current proposed2h/1h
limits cannot be silently extended; change a future version only with actual
time-budget authorization if measurements require it.

Once authorized and profiled, the finite training workflow is executable:

```bash
# Future execution only; PROFILE_SCOPE/TRAINING_SCOPE must be actual scoped
# records, never the false templates. Set assigned MIG, seed/CUBLAS/threads.
python scripts/run_favor_plus_paired_epochs.py profile --plan NEW_PLAN --record PROFILE_SCOPE --arm orf128 --execute-cuda
python scripts/run_favor_plus_paired_epochs.py profile --plan NEW_PLAN --record PROFILE_SCOPE --arm orf512 --execute-cuda
python scripts/run_favor_plus_paired_epochs.py campaign --plan NEW_PLAN --record TRAINING_SCOPE --execute-cuda
```

Campaign scope requires `training_approved`, `eight_epoch_pair_runtime_approved`,
`actual_new_time_budget_evidence`, pair7200/arm3600 seconds, eight epochs/arm,
actual approval evidence and a fresh source/plan-bound record. New worker records
start each arm's clock once while preserving the global campaign start. They
do not reset clocks at epochs or recoveries. No test-evaluation action exists.

Compare both arms on identical full-validation targets and equal new data-exposure/
update budgets. Treat native.703467 as historical context. Approximation has to
be assessed after learning, not inferred from initial512 improvements. If512
does not improve the matched128 selected-validation AUC, retain128 as the
reference and test the separately versioned question-conditioned fusion gate
from the performance plan; it is still an unimplemented hypothesis. No baseline
superiority, convergence or statistical significance is established here.
