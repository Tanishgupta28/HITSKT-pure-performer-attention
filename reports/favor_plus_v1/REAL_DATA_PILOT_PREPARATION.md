# FAVOR+ real ASSIST2017 pilot: CPU preparation

19 October2026 UTC; branch `research/hitskt-favor-plus-v1`, base `1e6255c`.

**The real-data runner is implemented and its CPU preparation passes. CUDA and
real-data optimization remain NOT RUN. Full-training readiness is NOT ESTABLISHED.**
The latest instruction requires **new user approval before any GPU execution or
resource change**, superseding earlier conditional execution permission. No GPU
probe, CUDA call, training job, watcher, allocation or protected-process change
was performed in this work. The lightweight tests include tiny synthetic
attention forwards and fake optimizer counters; they do not train a model.

## Changes and verified scope

- [Real pilot](../../ktbench/favor_plus_real_pilot.py) and
  [CLI](../../scripts/pilot_hitskt_favor_plus_real.py): CPU-only `prepare`, separately
  gated CUDA-only `execute`; no CPU-training mode, epochs, resume or test evaluation.
- [Metadata-only access guard](../../ktbench/favor_plus_gpu_access.py): actual user
  approval reference/action, exclusive independent allocation, owner/job/evidence,
  valid window and exact UUID visibility required before CUDA/RNG/model/output
  creation. The known protected LightSAKT2 MIG and its whole parent are rejected.
- [Diagnostics](../../ktbench/favor_plus_pilot_diagnostics.py): bounded independent
  feature-kernel/softmax comparisons and fixed-input Q/K tracking; no model changes.
- [Staged CUDA harness](../../scripts/validate_favor_plus_gpu.py): same numerical
  tolerances, now adds the shared approval gate,8GiB allocator quota, device/profile/
  feature/source/oracle provenance and final allocation-window recheck. Its CUDA
  path is still unrun. Historical reports and their original fingerprints remain
  preserved; they do not certify the new GPU path.

The pilot uses original `SessionStore`, `RollingSessionDataset`, production
sampler/collator, ASSIST256-wide/4-head/FF2048 hierarchy, depth/dropout/Adam/LR/loss
and15-session rolling history. FAVOR+ remains128 features, projection seed42,
chunk32, float32 accumulation; model seed42. The explicit pilot exceptions are
B1, padded cost<=2048 and at most5 updates. No Q/K clipping, normalization redesign,
architecture change, session truncation or attention replacement was introduced.
The original ELU+1 implementation and all historical research artifacts are intact.

| Check | Result |
| --- | --- |
| New lightweight CPU cases | **45 PASS** |
| Targeted existing oracle/admission cases | **11 PASS** |
| Final combined run | **56 PASS,0 failures,1 deliberately deselected**;1.03s test-body time; [log](real_pilot_preparation_v1/final_cpu_validation_v3.log), [XML](real_pilot_preparation_v1/final_cpu_validation_v3.xml) |
| CUDA exclusion during final tests | CUDA availability/RNG/init/device/memory APIs replaced with raising traps for the entire process; empty CUDA visibility; one CPU thread/core223 at low priority |
| Actual ASSIST preflight | **PASS**:11 processed-file hashes match the previously audited reference; selection/collation/reload passes with model/Adam constructors and CUDA APIs trapped |
| Actual dataset model forwards / optimizer updates | **0 /0** |
| CUDA checks, GPU peaks/runtime, trained Q/K/softmax quality, AUC/accuracy | **NOT RUN / NOT MEASURED** |
| Prior229 CPU suite | Preserved earlier evidence; not rerun or represented as229 fresh passes |

[Machine-readable verification summary](real_pilot_preparation_v1/VERIFICATION_SUMMARY.json)
records source/evidence identities and the exact limits of these outcomes.

The initial run had39 passes and4 failures because the new test double omitted
the real store's `session_length` method. Correcting that fixture resolved them;
[initial log](real_pilot_preparation_v1/new_cpu_tests_initial.log)/XML and subsequent
passes remain preserved. No failed model experiment was hidden. The existing
full-pipeline synthetic selection test was deliberately excluded to keep this
turn limited to preparation, not another model/optimizer experiment.

## Frozen real-data preparation

Valid [v2 prepared plan](../../experiments/hitskt_favor_plus_v1/assist2017/real_pilot_prepare_cpu_20261019_v2/prepared_plan.json),
SHA256 `1c03b9d1499c738877270e2d9ca46acced8982b006497c675dcc20cafe67f5d1`.
[Actual preflight log](real_pilot_preparation_v1/actual_dataset_preflight_v2.log).
The original v1 plan/log are preserved. V1 now correctly refuses execution after
the final oracle-fingerprint hardening changed source hashes; v2 re-prepares the
same samples against final source. This is intentional stale-plan protection.

| Update | Processed target-session index | Whole target actions | Exact padded cost including EOS/hierarchy |
| ---: | ---: | ---: | ---: |
|1 |1558 |67 |363 |
|2 |6312 |60 |290 |
|3 |3823 |42 |485 |
|4 |5066 |61 |91 |
|5 |6972 |62 |241 |

There are7060 eligible post-first training target sessions; the five selected
targets contain292 actions. Order follows the original seed42/epoch1 shuffled
length-bucket sampler with singleton batches, selecting the first cost-eligible
whole sessions using lengths/indices only. Oversize singletons would be counted
and skipped, never trimmed; none occurred before these five selections. Histories
retain original student/train boundaries. These five histories have1–7 sessions,
so they do **not** certify maximum15-session histories or maximum2048 cost.
Full-store hashing reads bytes for identity, including shared split arrays;
no held-out predictions, label-based selection or validation/test evaluation occur.

The CPU preparation/reload took0.84s after imports and peaked at582,561,792 RSS
bytes (about0.543GiB). These are preparation measurements, not training throughput
or GPU-memory estimates. Nineteen source hashes, processed data hashes, configuration,
indices/order and seed are frozen. A changed source/data/config/order requires
fresh preparation in a new directory. Results remain exclusively under
`experiments/hitskt_favor_plus_v1/`; all writes use fresh directories/files.

## Future diagnostics and failure criteria

All measurements in this section are **prepared, not yet observed on CUDA**.

1. Prerequisite: complete matching ASSIST/128-feature staged CUDA PASS on the
   assigned UUID, unchanged model/pipeline/harness/guard/oracle sources and
   strict FP32/TF32-off deterministic settings. All18 length/chunk combinations,
   mask/future/shift/PAD/EOS checks, gradients and checkpoint evidence must exist;
   partial, stale, different-device/profile or missing-checkpoint evidence refuses
   the real pilot. Existing output atol5e-5 and gradient atol5e-5/rtol1e-3 remain fixed.
2. During each real update: original masked BCE, finite logits/loss/all gradients,
   global gradient L2 norm, finite model **and Adam state**, synchronized training
   wall time, targets/s, padded tokens/s and GPU allocated/reserved peaks. Wall time
   includes dispatch/numerical checks; it is not pure GPU-active kernel time.
3. At initialization and after each update: the same whole training example in
   eval mode, with all valid-token Q/K norm summaries per stage. Fixed IDs/history
   control the input; Q/K tensors evolve with learned weights. Hooks use no-grad
   projections, are removed before optimization, preserve mode/RNG and do not
   alter the training graph or consume dropout draws.
4. For one row and<=32 positions per stage: retain local Q/K/V/projection/masks/
   raw production attention NPZs. Compare production attention with independent
   NumPy/SciPy float64 feature-kernel output (hard max-absolute gate5e-5), then
   separately with exact causal softmax. Record MSE/relative L2/max error, attention
   variation/KL, feature ESS and descriptive kernel/tail statistics.
5. Prospective **review triggers**, not new correctness tolerances: prefix softmax
   relative L2>.20, median feature ESS<2, or Q/K p95>2× initialization. These differ
   from the earlier multi-context/multi-seed feature-count statistic and do not
   replace it. A trigger sets `REVIEW_REQUIRED`; absence means only
   `NO_TRIGGER_IN_TINY_PILOT`. Neither authorizes larger training.
6. Save tagged pilot model/Adam/RNG/config, hash it, verify exact serialization,
   reload model/optimizer and require identical fixed-example CUDA logits. Do not
   take a sixth update to test resume. This is not full validation-selected
   training; the full evaluation pipeline rejects pilot checkpoints as benchmarks.

Stop on the first nonfinite value, correctness/resource/time/lease failure.
Optimizer-call count increments **before** calling Adam, so partial failures are
not reported as no update. Accepted update records survive later diagnostic
failures; a failure manifest retains phase/counts/last finite training observation.
No retry, reset, clipping, resume, overwrite or larger stage follows failure.
Success writes `_PILOT_COMPLETE`, **never benchmark `_SUCCESS`**. Even a numerically
successful pilot always records `full_training_ready: false`.

High-norm approximation is **still unresolved as a research-quality risk**. Prior
independent controls support finite-feature rare-tail sampling error, without a
detected CPU kernel implementation defect. Increasing to512 did not solve the
high-norm regime. Finite losses/gradients can coexist with poor attention; no new
real-data learned-norm or convergence result exists. The retrospective V1 protocol's
response-availability qualification remains unchanged.

## Resource and approval requirements

Request an independent exclusive GPU/MIG, preferably a separate H1003g.40gb,
with administrator/scheduler owner/job, assigned UUID, externally verifiable
exclusivity and start/expiry window; device access must be exposed to the process.
Do not use/reconfigure LightSAKT2's protected MIG or its whole parent. No alternative
GPU allocation is currently confirmed and no resource check was repeated here.
Reusing the presently protected UUID after LightSAKT2 finishes would require
verified release/reassignment, user approval and a separately reviewed guard
change; this runner never unblocks it automatically.

Set `CUDA_VISIBLE_DEVICES` to **exactly the assigned UUID**. The
[example record](real_pilot_preparation_v1/ALLOCATION_RECORD.example.json) is
intentionally invalid/unapproved; do not manufacture authorization from it.
After actual user approval, its action list must explicitly contain
`favor_plus_cuda_validation` and/or `favor_plus_assist2017_five_step_pilot` for
the command being executed. References must be nonsecret. Metadata validation
cannot independently prove ownership or that the user actually approved it;
the operator must establish both from authoritative evidence first.

| Bound | Prepared policy / limitation |
| --- | --- |
| GPU allocation |8GiB PyTorch allocator quota plus external context/runtime overhead; at least10GiB exposed capacity checked before model construction.16GiB+ exclusive GPU or separate40GB MIG is a planning choice, not a measured minimum. |
| Pilot host RAM |4GiB process peak-RSS gate, at least2GiB cgroup headroom at boundaries; start with16GiB administrator host allowance. Current cgroup limit90GiB; preparation did not exhaust RAM. |
| Shapes/workers |B1, cost<=2048,128 features/chunk32, at most5 updates,<=32-position diagnostic references; no DataLoader worker processes, one CPU dispatch thread |
| Time |120s/update acceptance gate, checked before/after Adam;900s cooperative execution deadline. GPU memory/lease/RSS/time are rechecked at boundaries. These cannot preempt an already executing kernel; use the owned-process outer timeout below. |
| Runtime/peak estimates |CUDA and actual training values **NOT ESTIMABLE / NOT MEASURED**. A40GB MIG is a reasonable bounded candidate, not full-batch certification. |

User-approval/allocation gates cover the new real runner and updated CUDA
validation entry point. The pilot's shape/RSS/deadline bounds above apply to the
real runner; the staged harness retains its own length/timing gates and shares
the8GiB allocator quota.
The preserved synthetic/full-training CLIs are not substitutes for this pilot
and are not authorized by this preparation. No external allocator is invoked.

## Commands — future GPU commands are not authorization

CPU-only preparation can be repeated with a **fresh** run ID (no model forward):

```bash
cd /workspace/capstone
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 223 nice -n 10 python -m scripts.pilot_hitskt_favor_plus_real prepare \
  --run-id real_pilot_prepare_cpu_NEW --steps 5
```

**Only after new user approval and an actual independent assignment**, from a
fresh process (replace placeholders; do not execute against the protected MIG):

```bash
export CUDA_VISIBLE_DEVICES='<actually assigned GPU/MIG UUID>'
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

timeout --signal=TERM --kill-after=30s 60m python -m scripts.validate_favor_plus_gpu \
  --allocation-record /approved/allocation.json --run-id cuda_assist_approved_01 \
  --features 128 --profile assist2017
```

Inspect its complete `gpu_validation.json` PASS before the next command:

```bash
timeout --signal=TERM --kill-after=30s 20m python -m scripts.pilot_hitskt_favor_plus_real execute \
  --prepared-plan experiments/hitskt_favor_plus_v1/assist2017/real_pilot_prepare_cpu_20261019_v2/prepared_plan.json \
  --allocation-record /approved/allocation.json \
  --cuda-validation experiments/hitskt_favor_plus_v1/assist2017/cuda_assist_approved_01/gpu_validation.json \
  --run-id real_assist_five_updates_approved_01 --execute-cuda
```

Timeouts apply only to these owned foreground processes; they are prospective
resource ceilings, not GPU runtime estimates or a watcher. Incomplete/terminated
validation produces no acceptable PASS. Preserve partial evidence and investigate
before retrying; never alter numerical tolerances to make a failure pass.

## Beyond five updates and reproducible metrics

Scaling needs separate user approval after complete CUDA correctness, finite
optimization/checkpoint evidence, review of approximation/norm/concentration
trajectories, and real memory/throughput measurements of longer/history-heavy
examples. The selected five small contexts do not supply that evidence alone.
Define the larger training/validation pilot's budget and review criteria before
running it; no feature-count choice or model selection using test AUC.

After a separately approved full experiment, retain original preprocessing,
chronological within-student retrospective protocol, non-attention settings,
validation-AUC checkpoint selection and metric definitions. Reload the selected
checkpoint in a fresh process, export predictions, verify exact counts/data and
checkpoint/source hashes, and independently reproduce AUC, accuracy, precision,
recall, F1, MSE and BCE with unchanged1e-6 metric tolerance. More seeds are needed
for claims of robust gains/significance. Five updates provide no benchmark AUC,
accuracy, convergence or publication-readiness evidence.
