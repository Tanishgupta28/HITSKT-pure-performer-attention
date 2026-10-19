# FAVOR+ GPU pilot and training readiness

19 October2026 UTC. Base: `d7399a0baf288eb035628273355ae3a005088acb`.
Branch: `research/hitskt-favor-plus-v1`.

**Resource gate BLOCKED. CUDA verification and the real ASSIST2017 pilot were
NOT RUN. Full-training readiness is NOT ESTABLISHED for all three datasets.**
LightSAKT2 remains active. No exclusive independent GPU assignment to HiTSKT
could be established, so GPU work stopped before CUDA initialization. There
was no failed CUDA test or failed optimizer step; the prerequisite was missing.

## Current resource and process evidence

[Read-only observation](gpu_pilot_readiness_20261019_v1/resource_and_process_evidence.json):

- At14:49:54 UTC, `nvidia-smi -L` exposed one H10080GB parent and one
  `3g.40gb` MIG instance:
  `MIG-d2fcfd79-720e-5218-82c9-0a132be68920`.
- GPU memory/utilization and compute-process PID queries returned insufficient
  permissions. Their output cannot establish vacancy, ownership or isolation.
- The recorded14:51:35 UTC process observation read a LightSAKT2 heartbeat
  from14:51:05 UTC, about30 seconds old:811 completed,145 pending, **1 running**.
  The active job was `ASSIST2015-GKT-baseline_tuning-42-d998e3d1a034`, with
  epoch3 last reported complete.
- Live processes182135 and2325921 had working directories inside
  `/workspace/LightSAKT2`, module `scripts.run`, and open NVIDIA device handles,
  including `/dev/nvidia0`, UVM and capability devices21/22. Both were sleeping
  at the instant of inspection; that state does not establish GPU idleness.
- No Slurm/PBS/LSF job assignment appeared in the inspected environment;
  `scontrol`, `squeue` and `kubectl` were unavailable. No external allocator or
  provider evidence of an exclusive HiTSKT assignment was supplied or found
  through the available resource interfaces.

The decision uses ownership evidence and protected-workstream activity, rather
than interpreting free memory or incomplete process visibility as permission.
No PyTorch import was needed for resource inspection. No CUDA initialization,
allocation, CUDA availability probe, GPU test, or GPU training was performed by
this continuation. No processes were stopped, paused, signaled or modified.
No watcher or repeated background polling task was created.

## Verification outcomes

| Requirement | Status | Evidence or missing prerequisite |
| --- | --- | --- |
|Read memory, latest scientific report and GPU plan |PASS |Reviewed preserved quality report and staged harness |
|Check LightSAKT2 activity |PASS: ACTIVE |Fresh heartbeat, live processes and NVIDIA handles |
|Independent exclusive GPU ownership |BLOCKED / NOT ESTABLISHED |Only protected existing MIG visible; no independent assignment |
|CUDA forward/backward and CPU/CUDA comparisons |NOT RUN |Resource gate blocked |
|CUDA causal masks, future responses, shifts, PAD/EOS |NOT RUN |Resource gate blocked |
|CUDA numerical stability, checkpoint reload |NOT RUN |Resource gate blocked |
|GPU memory/runtime and maximum-length behavior |NOT MEASURED |Resource gate blocked |
|Real ASSIST2017 five-step training pilot |NOT RUN:0 optimizer steps |Required CUDA verification has not passed |
|ASSIST2017/Junyi/EdNet full-training readiness |NOT ESTABLISHED |CUDA, pilot and representative batch-memory evidence missing |

Prior CPU evidence remains separate:229 tests passed with zero failures at
d7399a0;288 synthetic and576 frozen training-sample comparisons,24 double
gradient comparisons and54 independent precision controls passed. These tests
were **not rerun or relabeled as CUDA verification** in this resource-blocked
turn. Their source identities were checked against the preserved evidence.
See [the numerical-quality report](NUMERICAL_QUALITY_AND_GPU_READINESS.md) and
[verification summary](numerical_quality_v1/VERIFICATION_SUMMARY.json).

No new training-loss values, gradient/QK evolution, feature-concentration
measurements, softmax-error measurements during optimization, GPU peaks,
runtimes, pilot checkpoint or AUC exist for this continuation. Prior synthetic
optimization and no-update training-split diagnostics are not the requested
real-data CUDA training pilot.

## Exact allocation required to continue

An administrator, scheduler or resource provider must assign an **exclusive
GPU or independent MIG instance to HiTSKT**, isolated from LightSAKT2 and other
workers for the entire verification/pilot window. Another resource is required
while the current protected MIG remains in use; do not repartition or repurpose
it to obtain capacity.

Required operational evidence and access:

1. The assigned GPU/MIG UUID, named owner/job and externally verifiable
   scheduler/provider assignment proving exclusivity. An empty process table,
   memory reading or self-authored `exclusive: true` flag is insufficient.
2. A valid allocation window through all admitted stages and pilot steps. The
   harness allocation record requires `resource_uuid`, `owner`, `exclusive`,
   `expires_utc` and `external_allocation_evidence`. The record documents actual
   external allocation; it does not create or independently prove ownership.
3. The allocated device exposed to this container, with
   `CUDA_VISIBLE_DEVICES` set exactly to that UUID and the required device-node
   permissions. Existing protected-device visibility is not an assignment.
4. VRAM headroom for the harness's8GiB allocation ceiling **plus** CUDA/context
   overhead. An exclusive16GiB-or-larger resource, or a separate40GiB H100 MIG,
   is a planning choice with headroom, not a measured minimum model requirement.
   Actual GPU usage remains unknown; admission must stop on resource failure.

No cloud purchase, resource reconfiguration or external allocation was attempted.
This is the missing operational input, not another request to approve the
already-authorized conditional verification and five-step pilot.

## Authorized continuation after allocation

Use the preserved [staged harness](../../scripts/validate_favor_plus_gpu.py)
with **128 features** and explicit `--profile assist2017`. It prepares
17/65/257/939/3925/13081-position attention checks with chunks16/32/64,
strict float32/TF32-off comparisons, causality/padding/shift checks and a full
hierarchy checkpoint/gradient check. Its8GiB allocated-memory and120-second
timed-stage limits remain unchanged. Execution must stop at the first resource,
nonfinite-value or correctness failure; a longer stage is conditional on earlier
success and resource admission. The CUDA path remains unverified today.

Only after the required CUDA checks pass, the latest user instruction authorizes
the following real ASSIST2017 training-split pilot:

- `hitskt_favor_plus_v1`,128 features, seed42, original ASSIST architecture and
  original preprocessing, splits,15-session rolling history and non-attention
  training settings; B1, exact padded-token cost<=2048, at most5 optimizer steps.
- Use whole eligible sessions and preserve student/history boundaries. Do not
  truncate sessions, change the split/evaluation protocol or evaluate test data.
- Record each loss, gradient norms/finiteness, Q/K norm distributions, feature
  concentration and independent approximate-versus-exact causal-softmax error
  on small frozen prefixes as optimization proceeds. Record GPU peak/runtime,
  runtime settings, dataset/config/checkpoint hashes and reload consistency.
- Save all new outputs in a fresh run under
  `experiments/hitskt_favor_plus_v1/assist2017/`; preserve every previous run.
  No hyperparameter sweep, silent architecture change or full training.

Five steps would establish only bounded execution behavior. They cannot
establish convergence, expected AUC, statistical gains or publication readiness.
The current user authorization persists; once genuine allocation and CUDA
prerequisites are satisfied, it need not be requested again.
The existing `pilot_hitskt_favor_plus.py` runner is synthetic. A real-data
pilot must explicitly enforce the five-optimizer-step bound with the preserved
SessionStore/collator; a full-training command or five-epoch override is not a
substitute. No new real-data runner was created in this resource-blocked turn.

## Remaining scientific and scaling risks

The frozen feature-count study recommends128 for the first diagnostic pilot,
not for unconditional full training. High-norm softmax approximation remained
poor even at512 features; trained FAVOR+ Q/K evolution and feature concentration
have not been observed. Previously tested finite gradients do not guarantee a
useful learning signal. CPU numerical agreement does not certify CUDA precision
or sequential-scan throughput.

Original flattened histories and full-run batch shapes still need GPU memory
measurements: the B1/cost2048 pilot cannot certify batch cap64/token budget32768
or simultaneous longest sessions. Even a successful five-step pilot would need
that scaling assessment and an explicit later full-training decision.

The hierarchy and non-attention components remain unchanged. FAVOR+ uses
positive orthogonal Gaussian random features for the softmax kernel; historical
ELU+1 attention remains a separate deterministic model. Existing within-student,
retrospective teacher-forced evaluation and its response-availability limitation
also remain unchanged. Historical results are neither overwritten nor newly
attributed to FAVOR+.

## Preservation and handoff

[Fresh preservation check](gpu_pilot_readiness_20261019_v1/preservation.json)
passes for375 original files and407 additional prior source/evidence/output
files, including earlier FAVOR+ reports, local smoke checkpoints/logits and
frozen diagnostic arrays. The unrelated README edit and SAKT console log are
unchanged. No model, training/evaluation source, dataset, checkpoint or result
was modified in this continuation; no new runtime/architecture tests were
needed for this documentation-only checkpoint.

LightSAKT2 remains protected, DKT/EdNet remains unresumed, and no full training
was started. Publish this report/evidence and durable memory to the research
branch only, then wait for independently assigned resources. Do not merge main
or create an automatic retry schedule.
