# FAVOR+ resource status and training plan

Observed **19 October 2026, 15:13 UTC**. Documentation-only continuation on
`research/hitskt-favor-plus-v1`, base
`025a0e932e563c517d59d6b787f43f01a756b2c0`. The supplied full SHA contains a
transcription error; the branch tip and previous handoff identify this base.

**LightSAKT2 is actively using the only exposed MIG. No independent HiTSKT GPU
allocation is established. CUDA verification and real-data optimization remain
NOT RUN; full-training readiness remains NOT ESTABLISHED.** No CUDA calls,
training, allocations, watchers, process signals or protected-workstream edits
were performed. This request does not authorize GPU execution.

The [captured observations](resource_status_20261019_v1/observations.json) contain
timestamps, sanitized process/device observations, source hashes, saved timing
summaries and dataset/resource calculations. Live LightSAKT2 files continue to
change; this snapshot is the evidence for the statements below.

## LightSAKT2 activity, remaining work and ETA

Source: `/workspace/LightSAKT2/reports/progress.json`, updated 15:12:59 UTC;
`reports/execution_health.json`, `reports/ETA.json`, `reports/ETA_MEASURED_TIMINGS.csv`,
`reports/ETA_REMAINING_RUNS.csv`, and the current run's log/history.

| Check | Verified observation |
| --- | --- |
| Schedule entries | **811 completed, 1 running, 145 pending**, 957 total |
| Completed phases | 21 primary, 640 tuning, 75 repeated-seed, 75 ablation entries |
| Independent training versus reuse | ETA timing inventory contains 773 completed independent runs; remaining work is 139 training entries including the running job, plus 7 reuse/evaluation entries. Logical entries are not all independent experiments. |
| Active run | `ASSIST2015-GKT-baseline_tuning-42-d998e3d1a034`, trainer PID2325921 |
| Progress | **Epoch6 completed**; ceiling50, patience5. Earlier handoff had epoch3; this inspection observed advancement from5 to6. No percentage of eventual stopping time is known. |
| Measured speed | Health monitor's recent-epoch mean **608.26 seconds**, about10.14 minutes per complete train/validation epoch |
| Health | Fresh heartbeat; health monitor observed63.01 CPU-seconds of progress over63.02 wall seconds; no stalled checkpoint or failed/interrupted/blocked slot reported |
| Waiting process | PID2404132, run suffix `df10e3998ba8`, is explicitly `pending`, `waiting_for_gpu_admission: true`; blocked in `locks_lock_inode_wait`, no NVIDIA handles/history. This is queued work, not a second actively training experiment or evidence of a stall. |

The five most recent independently trained ASSIST2015/GKT runs took
1.04, 1.16, 1.33, 3.03 and6.03 hours for6,7,8,18 and36 epochs respectively
(`training_seconds`, excluding post-selection test inference). Their complete
epoch means were approximately592–619 seconds. This supports the observed
throughput while illustrating uncertainty in stopping epochs. It does not
justify multiplying a global average by the remaining experiment count.

The existing measured forecast estimates **255.31 occupied MIG hours
(10.64 continuous days)** remaining, conditional training finish
**30 October, approximately06:31 UTC**. Its scenario envelope is **51.09–890.34
hours**, approximately21 October–25 November. This is a provisional early-stop/
maximum-epoch scenario range, **not a confidence interval or guaranteed bound**.
The isolated serial-equivalent estimate is352.54 hours; it is a different quantity.
The occupied-time estimate credits measured1.512× throughput only to admitted
independent GKT-MHA workers; PAM stays exclusive. Future stopping epochs,
configuration costs, sustained concurrency and uninterrupted availability remain
uncertain, especially in strata with only one completed GKT run.

All remaining forecast training time is GKT/GKT-MHA:116 tuning entries and30
repeated-seed entries remain, including the seven reuse/evaluation entries.
The scheduler source `/workspace/LightSAKT2/scripts/schedule.py` invokes
`scripts.finalize` after training. Its saved task list still includes final
tests and checkpoint inference, tables/statistics/figures and manuscript/research
verification. The ETA separately estimates about1.04 GPU hours of final inference,
with configuration/setup and other analysis overhead unmeasured. Completed
efficiency benchmarks are reused. **Training completion is not resource release.**
No administrator schedule or promise of reassignment to HiTSKT is available;
other externally scheduled jobs cannot be ruled out.

No current stall or new resource failure was found. Operational limitations:
GPU utilization is unavailable; the LightSAKT2 monitor reports1864/40320MiB used,
not permission to share; automatic recovery after container replacement is not
configured; NFS retention after deprovisioning is unverified. The shared cgroup
limit is90GiB, usage33.80GiB, historical OOM-kill count2. These observations do not
identify a new OOM or justify changes to LightSAKT2.

## GPU visibility, ownership and required allocation

- One H10080GB parent is enumerated:
  `GPU-5a568e3c-e96c-169f-b063-f4478340ff51`.
- Exactly **one exposed compute MIG**, H100 `3g.40gb`:
  `MIG-d2fcfd79-720e-5218-82c9-0a132be68920`, also the sole entry under
  `/var/run/nvidia-container-devices`. The parent and child are not two available
  compute resources. LightSAKT2 processes182135/2325921 hold NVIDIA handles.
- NVML memory/process queries return insufficient permissions; free memory or
  incomplete PID visibility cannot establish ownership. No Slurm/PBS/LSF
  assignment is present, and their inspected clients and `kubectl` are absent.
  Existing provider-access evidence records denied Kubernetes permissions;
  no accessible allocation interface, independent lease or HiTSKT assignment
  was found. No API credentials were inspected or stored.
- **No genuinely accessible alternative GPU/server is verified.** There is no
  configured SSH route or allocation tool establishing one. GitHub storage and
  other workspace directories are not independent compute allocations.

Request from the administrator:

> Assign HiTSKT an exclusive independent H100 MIG3g.40gb (or an independently
> assigned compatible GPU with at least16GiB as an initial planning reservation),
> without touching LightSAKT2's current instance. Supply the device UUID,
> owner/job, authoritative exclusivity/assignment evidence and validity window;
> expose only the assigned resource to the execution environment and set
> `CUDA_VISIBLE_DEVICES` exactly to its UUID. Reserve16GiB host RAM for the bounded
> verification/pilot as an initial allowance, with usage monitored. Provide an
> extendable window because FAVOR+ GPU runtime has not yet been measured.

The staged harness's8GiB allocated-memory gate plus context/runtime headroom
makes a separate40GB MIG a **reasonable candidate for bounded verification and
B1/cost2048**, subject to measurements. This is not proof of sufficiency for
the longest stages or full history/batch shapes. Gates are checked after each
stage; they are not a hard preemptive VRAM quota or whole-process timeout.
The administrator must explicitly release/reassign the existing MIG after
LightSAKT2's required work if that is chosen instead of a separate resource.

A different GPU/server could preserve methodology by keeping source/data hashes,
splits, seeds/projections, architecture, optimizer, precision policy, selection
and metrics fixed, and recording hardware/software plus CPU/CUDA comparisons.
Hardware changes do not promise bitwise identity. No such alternative is
currently established as accessible, and none was provisioned or requested here.

## FAVOR+ readiness and remaining preparation

Preserved [implementation verification](IMPLEMENTATION_VALIDATION.md),
[numerical study](NUMERICAL_QUALITY_AND_GPU_READINESS.md), and
[previous resource-gate report](GPU_PILOT_AND_TRAINING_READINESS.md) remain the
scientific baseline. Sources inspected include the
[separate pipeline](../../ktbench/favor_plus_pipeline.py),
[staged CUDA harness](../../scripts/validate_favor_plus_gpu.py), and
[synthetic pilot](../../scripts/pilot_hitskt_favor_plus.py).
Small evidence/harness identity checks match the saved
verification manifest; no research source changed and **no tests were rerun**.

| Requirement | Status and limit |
| --- | --- |
| Genuine FAVOR+ CPU implementation | **PASS in tested scope**,229 recorded tests,0 failures: positive orthogonal Gaussian features, softmax-kernel scaling, stable causal prefixes, first-order gradients, masks/shifts/PAD/EOS and checkpoint preservation |
| Known unresolved attention defect | **None observed** in completed CPU checks; this does not certify untested CUDA execution or all learned states |
| Independent mathematical comparison | Double output/gradient maximum errors7.99e-14/3.77e-14 against explicit feature-kernel reference; good implementation agreement does not imply exact-softmax approximation is always good |
| Train/evaluate pipelines, all3 datasets | **Prepared and CPU/synthetic checked**; unchanged verified preprocessing, chronological within-student splits,15-session history, hierarchy/non-attention settings, validation-AUC selection/patience5/best reload, metric definitions; separate FAVOR+ namespace/checkpoints |
| CUDA correctness/runtime/memory | **NOT RUN / NOT MEASURED**, allocation unavailable |
| Real five-update ASSIST pilot | **NOT RUN; runner preparation remains**. Existing `pilot_hitskt_favor_plus.py` is synthetic; full-training CLI has no real-data step-bound mode. |
| Full training / convergence / FAVOR+ AUC | **NOT ESTABLISHED / NOT RUN** |

**128 features remains the first bounded-pilot recommendation**, under the
previously frozen rule, without test-set AUC tuning. Initial-context median/p90
relative error was0.09676/0.15894. Increasing to512 improved initial-sample MSE
44.5% at up to1.75× measured CPU time, but high-norm synthetic MSE improved only
2.99% (1.24159→1.20444). Feature effective sample size near1 indicates severe
concentration. Independent Gaussian/quadrature and explicit-kernel controls
support expected rare-tail sampling error rather than an observed scaling or
gradient defect. Trained FAVOR+ norms and useful learning signals remain unknown;
finite outputs/gradients alone do not establish approximation quality or convergence.
No clipping, architecture change or larger-feature default is justified here.
Norm growth can concentrate the feature mass and degrade attention/learning
while losses and gradients remain finite; the pilot must examine that evolution.

Necessary CPU preparation is limited to an **explicit real-data five-update
runner and instrumentation**, using the original store/collator/history and
whole eligible training sessions, seed42/B1/cost<=2048, no test access. It must
record losses, finite gradients/norms, stage Q/K norms, feature concentration,
bounded frozen-prefix exact-softmax errors, memory/runtime and hashed checkpoint
reload. The stage1 CUDA harness is prepared; the entire stage2 launch is **not
yet turnkey**. This task requests documentation only, so no runner was added.
No further blanket CPU audits or mathematical experiments are currently needed.
The CUDA harness's full-model example uses a reduced vocabulary. Actual-vocabulary
optimization, train-mode behavior and representative history/batch shapes still
need measurement, and dataset-specific profile/checkpoint checks must precede
each dataset's full experiment. The ASSIST harness alone cannot certify all
Junyi/EdNet shapes. The existing pipeline uses no DataLoader worker subprocesses;
host/CPU admission must continue to respect the protected workload.

## Dataset scale and resource evidence

Counts/profile values below come from the preserved
[dataset plan](numerical_quality_v1/dataset_pipeline_plan.json); disk sizes were
obtained from file metadata, without rehashing the large datasets or repeating
their correctness audit. RAM figures are **component lower bounds, not peak estimates**.

| Dataset | Supervised train / validation / test targets | Maximum session actions | Width/heads/FF; epoch ceiling | Store files GiB | FP32 weights+grads+Adam floor GiB | Training metric persistent-array floor GiB |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| ASSIST2017 |566,107 /125,800 /112,252 |938 |256/4/2048;100 |0.0142 |0.1033 |0.0153 |
| Junyi |8,920,358 /2,556,773 /2,456,402 |3924 |128/2/1024;50 |0.2418 |0.1232 |0.2409 |
| EdNet-KT1 |49,458,444 /14,520,975 /13,429,870 |13080 |128/2/1024;40 |1.3390 |0.0756 |1.3358 |

FP32 optimizer floor is16 bytes per unchanged trainable parameter, excluding
projection buffers/activations/allocator/context. Training metrics keep5N bytes
of raw scores/labels plus24N bytes of converted scores/labels/predicted classes;
sorting and other temporary workspaces add RAM. Mapped store disk size is not RSS.
Historical CPU measurements: attention L13081 forward/backward1.078/4.513 seconds,
peak0.786GB decimal; reduced-vocabulary EdNet-profile full model forward/backward
3.506/12.556 seconds, peak1.751GB. Neither is a real-vocabulary full-history GPU
benchmark. No GPU speed or GPU peak is inferred from these CPU measurements.

## Six-stage sequence — planning only

All runtimes below are **NOT ESTIMABLE** from current FAVOR+ CUDA evidence.
No full-training stage is authorized. A16GiB host allowance and16GiB+ exclusive
GPU are bounded-stage planning reservations, not measured minima or full-run quotas.

| Stage | Resource / VRAM / RAM | Dependencies and risks | Completion criteria |
| --- | --- | --- | --- |
|1. CUDA correctness | Exclusive device;128 features, B1; kernel8GiB allocated gate plus overhead. Start with16GiB host allowance; CPU-reference RSS previously0.8–1.75GB in limited cases. GPU/total RAM peaks unknown. | Actual allocation and later execution scope; staged harness with explicit ASSIST profile, lengths17→13081/chunks16/32/64. Risks: CUDA reductions, custom backward, sequential launch cost, post-stage gates. | Output atol5e-5, gradient atol5e-5/rtol1e-3; independent short oracle, causal/future/padding/shift, finite backward, exact checkpoint reload, measured memory/runtime; stop on first failure. |
|2. Five-step ASSIST train pilot | Same exclusive resource; original256-wide model/full vocabulary,128 features, B1/cost<=2048, <=5 optimizer updates. Keep bounded8GiB allocated admission target plus overhead and16GiB host allowance, measure actual use. | Stage1 PASS; implement/review real-data step-bound runner. Risks: Q/K norm growth, concentration, dropout/train-mode numerics and actual-vocabulary memory. | All requested diagnostics, finite losses/gradients/optimizer state, no test access, <=5 updates enforced, hashes and reload integrity; no convergence/AUC claim. |
|3. Larger ASSIST train/validation pilot | Exclusive GPU; admit representative original batches progressively. Peak VRAM/RAM NOT ESTIMABLE; measure before raising bounds.16GiB host is only a starting allowance. | **Separate approval** for predeclared budget/subset/validation use, stages1–2 PASS. Include long/history-heavy and oversize-single-example admission. No HPO or test tuning. | Sustained finite training, norm/softmax diagnostics as learning evolves, reproducible validation/checkpoint choice, representative memory and steps/targets per second. |
|4. Full ASSIST | Exclusive capacity sized fromstage3; total VRAM/RAM NOT ESTIMABLE (0.103GiB optimizer/0.015GiB metric floors above). Original sampler cap64/token32768/history15; up to100 epochs, validation patience5. | Separate full-training approval; stage3 evidence and preset128 decision reviewed. Risk: poor approximation or later norm drift; current resource assumption unproven. | Original protocol, validation-selected best checkpoint/fresh replay/all7 metrics, complete targets, hashes/runtime and losses preserved; no claim of gain from one seed. |
|5. Full Junyi | Independent allocation sized after representative Junyi batch measurements; VRAM/RAM NOT ESTIMABLE (0.123/0.241GiB floors).8.92M train targets, up to50 epochs. | Separate approval, CUDA/pilot gates and dataset-specific scale admission; ASSIST memory/runtime does not certify Junyi. | Same complete training/selection/replay evidence; measured throughput supports revised ETA; retained timing limitations stated. |
|6. Full EdNet-KT1 | Independent allocation after worst-history measurements; VRAM/RAM NOT ESTIMABLE (0.076/1.336GiB floors plus AUC workspace/cache).49.46M train targets, up to40 epochs. | Separate approval and earlier gates; longest sessions/history padding,49M-score metrics, sequential scan throughput and norm drift are major risks. Do not resume DKT. | Complete counts/epochs/selection, stable long batches, finite states, fresh checkpoint replay/predictions/metrics and immutable manifests; no fabricated missing baseline result. |

There are18 kernel length/chunk combinations at128 features. Each has a120-second
timed forward/backward acceptance gate:36 minutes is only the sum of those
timed-part budgets, **not a wall-clock estimate or hard total timeout**. CPU
references, invariance passes, setup, full-model checks and stage2 are additional.
Measure stage1 and representative CUDA batches before quoting hours or days for
FAVOR+ training. Historical ELU+1 or LightSAKT2 throughput cannot supply that ETA.

The original sampler allows an indivisible singleton above token32768; the
budget is not an absolute maximum for every context. Long-history shapes need
actual admission measurements. Do not silently truncate sessions or alter the
protocol to make a memory estimate fit. Existing retrospective teacher-forced
within-student evaluation and its unverified response-availability limitation
remain unchanged; FAVOR+ does not resolve that historical scientific limitation.

## Immediate decision and preservation

**Request an independent exclusive allocation now.** Waiting for LightSAKT2
may mean roughly10.6 more training days plus final work, with a wide uncertainty
range and no automatic reassignment. The shortest safe route is the small
step-bound runner preparation, actual allocation, staged128-feature CUDA checks,
then five train-only ASSIST updates. Those would be first genuine FAVOR+ real-data
optimization observations, not a benchmark result. A larger approved validation
pilot is needed before a defensible full experiment or expected AUC.

Only this new report/evidence and durable memory are changed. Original ELU+1 and
FAVOR+ code, checkpoints, data/results and previous reports remain untouched;
LightSAKT2 was read only and DKT/EdNet unmodified. Prior preservation manifests
remain evidence; no redundant full375-file audit or229-test suite was rerun.
Unrelated README/SAKT-log hashes are unchanged. Commit/push documentation only
to the research branch, never main, then stop without a watcher or GPU job.
