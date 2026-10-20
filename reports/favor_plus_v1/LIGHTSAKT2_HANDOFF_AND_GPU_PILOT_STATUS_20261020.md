# LightSAKT2 handoff and FAVOR+ pilot status

20 October 2026 UTC; HiTSKT branch `research/hitskt-favor-plus-v1`, base
`3bf92ee6488ece91cd92012dd8980016f1cca726`.

**LightSAKT2 is stopped and its known automatic launch paths are held. All
captured research artifacts are preserved. FAVOR+ CUDA verification and the
ASSIST2017 five-step pilot remain NOT RUN because an unrelated GPU workload is
active and exclusive allocation is not verified. Full training is not ready.**

The user now explicitly permits stopping verified LightSAKT2 jobs and, after
actual exclusive GPU assignment, running staged CUDA verification and at most
five ASSIST2017 train-only updates. This supersedes the earlier no-interruption
and repeat-approval requirements only for this scope. It does not authorize
stopping unrelated work, inventing an allocation, full training or DKT/EdNet
recovery. No new user permission is needed for those bounded stages once all
actual resource/correctness gates pass.

## Results of this handoff

| Required check | Status | Evidence / limit |
| --- | --- | --- |
| LightSAKT2 training stopped | **PASS: already stopped before this turn** | No live project trainer/controller/recovery process found; historical PIDs absent/zombie, old heartbeat stale. **Zero signals sent**, so this is not a claim of a new graceful shutdown. |
| Completed/pending schedule identified | **PASS** |813 completed,142 pending,2 interrupted,0 running,957 slots; existing per-run statuses overlaid on manifest. No schedule configurations deleted/changed. |
| Completed artifacts preserved | **PASS** |7,915 protected LightSAKT2 files /1,391,693,255 bytes identical before/after;814 stored result references/776 unique selected-checkpoint paths match hashes; all declared predictions/configs verified. |
| Interrupted state preserved | **PASS** | GKT epoch2 best/latest loaded strictly on CPU; model/Adam finite; optimizer/RNG/loader/config/data/split/history verified; separate content-verified local archive. |
| Automatic LightSAKT2 restart | **Known local paths HELD** | Persistent marker,9 targeted CPU tests PASS; no watcher started. External startup/scheduler configuration cannot be certified from this container. |
| Exclusive GPU ownership | **NOT VERIFIED / BLOCKED** | Unrelated processes hold same GPU/MIG capability handles; no scheduler assignment, Kubernetes403. They were untouched. |
| FAVOR+ CUDA correctness / memory / runtime | **NOT RUN / NOT MEASURED** | No CUDA context or training launched. Read-only NVML resource inspection is not CUDA validation. |
| Five-step ASSIST2017 training pilot | **NOT RUN** |0 real-data forwards,0 optimizer steps, no losses or trained Q/K diagnostics to report. |
| Larger/full training readiness | **NOT ESTABLISHED** | Needs CUDA, bounded optimization, learned-Q/K approximation and measured resource evidence. No full training authorized. |

This is a preservation/control verification, not a numerical replay of completed
LightSAKT2 results. Prior229 FAVOR+ CPU passes and56 pilot-preparation CPU passes
remain earlier evidence; neither suite was repeated or relabeled as CUDA results.
The nine new passes apply only to LightSAKT2 hold behavior.

## LightSAKT2 state and preservation

The old progress files last reported at `2026-10-19T18:11:05Z`; direct process
inspection and interruption logs supersede those stale active-worker references.
`ASSIST2015-GKT-baseline_tuning-42-e01b589e46cc` completed epochs1/2, then its
log ends with `KeyboardInterrupt` during backward. Best/latest epoch2, validation
AUC `0.7327774052048939`, bad-epoch counter0. The initiating actor for that earlier
interruption is not established. No partial later epoch is claimed recoverable.
`ASSIST2015-GKT-baseline_tuning-42-e1c69a92d6b5` interrupted while waiting for
GPU admission; it has no saved epoch. Neither was resumed or marked complete.

Best SHA256 `a36930ca8a0a5d9e0325405de50dc83c82584408080bf2f0f56cd321fb455608`;
latest SHA256 `1a508b505f07f48491e489d06cb837e660cb6dd5378c7ba33ab8ff61a8a0e551`.
All original checkpoints remain in place. The new local archive
`/workspace/LightSAKT2/backups/priority_handoff_20261020/interrupted_gkt_epoch2.tar.gz`
has1,155 content-verified members/3,869,675 bytes, SHA256
`6e3b03248307f609059a4bec7ea0ac203cbbb35054360a871b3b5920bf0a45c0`.
It is ignored/local, not uploaded by this commit. All814 existing completed
backup references are locally present; this does not claim fresh offsite
download verification. Existing checkpoints/results/logs/configs were not deleted.

`/workspace/LightSAKT2/experiments/USER_TRAINING_HOLD.json` blocks normal run,
scheduler, coordinator, recovery start/watch, admission, post-run hooks and
final analyses before launches. Scientific `src/` fingerprint remains
`0a04dc026130226cf58cd1fad8e9f9a696d91088097b2190cdf9e1bcd1fa48f6`.
The final nine CPU hold cases passed; initial import-trap harness failures and
corrected iterations are retained. Interactive terminals/coding sessions and all
unrelated jobs were left alone. External startup hooks/reservations are not visible.

Full LightSAKT2 evidence/report/control changes are committed and pushed on
`research-rebuild` at **`3debdc21ce2380ad4dd3bfaa8b5f14eea208eae4`**:
`/workspace/LightSAKT2/reports/LIGHTSAKT2_HITSKT_PRIORITY_HANDOFF_20261020.md` and
`reports/priority_handoff_20261020/` in that repository. Its substantial
pre-existing dirty scientific state remains untouched and outside the control
commit. The older sibling LightSAKT project is a separate study and was not changed.

## GPU evidence and exact allocation needed

[Resource evidence](priority_handoff_20261020/resource_and_process_evidence.json)
was observed at06:55:23 UTC, read-only, without importing Torch or initializing CUDA.
One parent/one exposed MIG:

- Parent `GPU-5a568e3c-e96c-169f-b063-f4478340ff51`, H10080GB.
- Exposed H1003g.40gb `MIG-d2fcfd79-720e-5218-82c9-0a132be68920`.
- MIG NVML total42,278,584,320 bytes (39.375GiB), used9,922,347,008 bytes
  (about9.24GiB), free32,356,237,312 bytes (about30.14GiB).
- Initial unrelated GPU PID2643837 was outside both project roots. A later
  unrelated worker group rooted at PID2662175 holds the same
  `/dev/nvidia-caps/nvidia-cap21`, cap22, nvidia0/UVM/control devices.
  A common root UID is not proof that these jobs belong to this user. No signals
  were sent to any of them; PIDs are observation-time identities, never kill targets.
- Compute-process/parent-memory query access is restricted. No Slurm/PBS/LSF
  assignment variables or scheduler tools are exposed. Attempted Kubernetes
  pod read using existing serviceaccount returned403; pod name was only inferred
  from HOSTNAME, so no authoritative allocation record was obtained.
- Container RAM limit96,636,764,160 bytes (90GiB); usage65,202,343,936 bytes
  (about60.72GiB) at that snapshot, with2 cumulative OOM events. This is shared
  workload usage, not FAVOR+ peak memory or evidence of a new OOM.

**Required:** administrator/scheduler or allocation-owner evidence assigning an
exclusive GPU/MIG to this user's HiTSKT workload, exact UUID, owner/job, external
allocation reference and valid start/expiry window, correct container device
exposure, and confirmation of no unrelated jobs/reservations. For this same MIG,
its actual owner must clear/reassign unrelated work; the LightSAKT2-only stop
authorization does not cover it. Alternatively an actually assigned separate
16GiB+ GPU or40GB MIG is a planning candidate. None is confirmed accessible.
Free VRAM is not allocation evidence. An allocation clarification was requested;
no affirmative evidence has been received at this report checkpoint.

The H1003g.40gb capacity is a reasonable candidate for bounded checks, subject
to actual admission and measurements. Prepared policy is8GiB PyTorch allocator
quota plus context/runtime overhead, at least10GiB exposed capacity,4GiB process
RSS and2GiB shared-container headroom. A16GiB host allowance is a planning
request, not a measured training minimum. Original batch feasibility and CUDA
training throughput remain **NOT MEASURED / NOT ESTIMABLE**.

## Preserved FAVOR+ state and numerical limits

The model remains hierarchical HiTSKT with genuine softmax-kernel FAVOR+
positive orthogonal random features, separate from the unchanged historical
deterministic ELU+1 model. No model, attention, training, dataset or evaluation
source changed in this handoff. All30 originally captured checkpoint hashes
freshly match (723,673,442 bytes), as do the unrelated README/SAKT log edits and
the prepared pilot source hashes. [Preservation evidence](priority_handoff_20261020/hitskt_preservation.json).
The full375-file data/audit digest inventory was not redundantly rerun; prior
evidence is retained and no original research file was edited.

128 features remains the predeclared first diagnostic configuration, not an
established full-training optimum. Prior CPU studies matched independent feature
kernel outputs/gradients but found substantial error versus exact softmax for
high-norm Q/K.512 features was not a general cure. This is consistent with
finite-feature rare-tail sampling, not an observed implementation defect in the
tested scope. CUDA and evolving learned FAVOR+ Q/K are still unverified; finite
loss/gradients alone would not establish good softmax approximation.
See [numerical-quality report](NUMERICAL_QUALITY_AND_GPU_READINESS.md) and
[pilot preparation](REAL_DATA_PILOT_PREPARATION.md), retained unchanged.

## Next authorized stages; no commands executed

1. Obtain actual exclusive allocation. Recheck live identity/reservations once
   before admission; no watcher or repeated background polling. The existing
   safety guard intentionally still rejects this MIG/parent. If they are genuinely
   released/reassigned, make a narrowly reviewed/tested guard update reflecting
   that evidence under the current authorization. Do not bypass it or manufacture
   an `independent: true` record. Such a source change invalidates the old plan;
   prepare a fresh source-bound plan/directory.
2. Run the existing staged CUDA harness with128 features/ASSIST profile, strict
   FP32/TF32-off settings and unchanged tolerances (outputs atol5e-5, gradients
   atol5e-5/rtol1e-3). Short-to-long causal attention lengths17–13081, chunks16/32/64;
   CPU/independent references, masks/future responses, shifted correctness/PAD/EOS,
   finite gradients and checkpoint reload. Stop on first failure,8GiB quota or
   timing admission breach. Record UUID/settings/hashes/memory/runtime.
3. Only after complete matching CUDA PASS, execute the source/data-bound ASSIST
   train-only pilot:128 features, seed42, B1, padded cost<=2048, at most5 Adam
   updates, original hierarchy/preprocessing/within-student split. No validation
   or test evaluation. Capture loss/global gradients/all-stage QK norms/feature
   concentration/exact-softmax prefix errors, finite parameters/Adam, time/throughput,
   allocator peaks and checkpoint/RNG identity. Existing frozen samples total292
   actions, with costs363/290/485/91/241 and1–7 history sessions; they do not
   certify maximum histories or original full batches.

Future command templates remain in the unchanged
[preparation report](REAL_DATA_PILOT_PREPARATION.md). Use fresh run IDs, the actual
lease and a freshly prepared plan if source changes; never run placeholders:

```bash
export CUDA_VISIBLE_DEVICES='<verified exclusively assigned UUID>'
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
timeout --signal=TERM --kill-after=30s 60m python -m scripts.validate_favor_plus_gpu \
  --allocation-record /approved/allocation.json --run-id cuda_assist_NEW \
  --features 128 --profile assist2017
# Inspect complete matching gpu_validation.json PASS before this next command.
timeout --signal=TERM --kill-after=30s 20m python -m scripts.pilot_hitskt_favor_plus_real execute \
  --prepared-plan /path/to/fresh/prepared_plan.json \
  --allocation-record /approved/allocation.json \
  --cuda-validation experiments/hitskt_favor_plus_v1/assist2017/cuda_assist_NEW/gpu_validation.json \
  --run-id real_assist_five_updates_NEW --execute-cuda
```

Those timeouts apply only to newly owned foreground pilot processes. They are
ceilings, not measured runtime predictions. No acceptance tolerance was changed.

## Scaling decision

**Do not start larger training yet.** After successful CUDA/five-step evidence,
review high-norm approximation and learned trajectories; seek separate approval
for an ASSIST training/validation pilot with a predeclared budget, maximum-history
shape coverage and graduated batch-memory measurements. Do not change architecture
or choose features with test AUC. Five updates establish neither convergence nor
expected AUC. Estimate epoch runtime only from measured training throughput over
representative shapes, not these tiny contexts or CPU-to-GPU extrapolation.

Full ASSIST2017 requires that review and separate explicit approval. Junyi then
needs its own profile/shape/memory and bounded training checks before a separately
approved full run; EdNet-KT1 follows with full-history/large-dataset resource checks
and measured throughput. Each retains original verified preprocessing, chronological
within-student retrospective teacher-forced evaluation, unchanged nonattention
settings and validation-AUC checkpoint selection. Fresh selected-checkpoint replay,
prediction/hash/count evidence and all seven independent metrics are needed for
reproducible AUC/accuracy claims. Response-availability limitations and single-seed
limitations remain; no publication-readiness claim follows from this handoff.

HiTSKT main/history/ELU+1 artifacts and unfinished DKT/EdNet remain untouched.
Detailed outcome summaries: [handoff evidence](priority_handoff_20261020/handoff_evidence.json).
