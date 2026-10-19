# HiTSKT and baseline checkpoint replay plan

Frozen for verification on 19 October 2026; evidence baseline `e1065c9`.
**PLANNED, NOT EXECUTED.** No training, original result replacement or new
scientific protocol is part of this plan. Gate outcomes and current resource
observations belong in [HITSKT_VERIFICATION_STATUS.md](HITSKT_VERIFICATION_STATUS.md).

## Admission: independent GPU, not merely available memory

The visible H10080GB MIG3g.40gb currently has allocated memory, a current
LightSAKT2 active-job heartbeat and restricted compute-process visibility.
Its empty process table is not evidence of idleness. **Do not run there.**

Before CUDA initialization, require an explicitly assigned independent GPU/MIG
UUID, an operational reservation/lease excluding other work and positive
evidence of idle ownership. Check inventory, process visibility, memory use
and allocation/reservation state read-only. Recheck immediately before launch;
obtain an exclusive lease so a momentary idle observation cannot race another
worker. If ownership or visibility is inconclusive, record NOT VERIFIED and
stop before CUDA initialization. Do not kill/pause another process, change MIG
partitioning or poll indefinitely for its job to end. No automatic watcher is
being installed.

Use the original H100/MIG architecture and best-supported NVIDIA runtime for
historical reproduction. A different device is a separately labelled portability
replay, not exact historical environment reproduction. An independent device
can be used while LightSAKT2 continues elsewhere only if host-memory/CPU
admission below also passes.

## Common fixed controls

- One model process and one checkpoint at a time; fresh process between pairs.
  No model parallelism or concurrent evaluation pool. CPU helpers use at most
  two threads, `workers=0`, low CPU/I/O priority and a small allowed CPU affinity.
- Read the actual cgroup limit and current RSS/usage; do not rely on host RAM.
  Current limit is90GiB. Require at least24GiB unused host headroom after the
  estimated job allocation. Independently check cgroup usage and resident pages;
  cache reclaimability is not a guarantee. CPU forward estimates must remain
  <=16GiB; preflight all original batch shapes before allocating.
- On an assigned GPU, require estimated working storage <=24GiB and at least
  8GiB genuinely free GPU reserve after that estimate. Use a disposable child
  process to validate peak allocated/reserved memory on selected original
  batches before full replay. Estimates are admission heuristics, not proofs
  against OOM. Stop on headroom loss or an estimate exceeding the cap; do not
  shrink, truncate, reorder or split historical batches to force a reproduction.
- Protect originals with streaming SHA-256 before/after. Require existing
  `_SUCCESS`, final result, configuration, best checkpoint, strict validation
  selection, expected dataset hashes and target counts. Refuse incomplete pairs.
  Do not instantiate or restore an optimizer for inference; discard its saved
  payload after extracting model weights/configuration.
- Seed42; `.eval()`, inference/no-grad, FP32, no autocast, no optimizer, no
  dropout, no optimizer resume or test-time adaptation. Record actual PyTorch
  commit/build, Python/NumPy, CUDA runtime/driver/cuDNN, GPU/MIG UUID and type,
  matmul precision/TF32, cuDNN flags, deterministic/warn-only settings,
  cuBLAS workspace configuration and environment overrides.
- Preserve source version, original evaluation sampler and padded batch shapes.
  Record batch membership, event IDs, labels, original order, all targets once,
  finite logits/probabilities and zero cross-student histories. Use memory-mapped
  score arrays and disk-backed/single-process metric reduction, not giant
  in-memory Python lists or one GPU tensor spanning an entire dataset.
- Separate immutable run namespaces, for example
  `reports/checkpoint_replays/<run_id>/` for aggregate manifests and ignored
  `.inspection/checkpoint_replays/<run_id>/` for event-indexed predictions.
  Never write into `experiments/`, old audit directories or LightSAKT2.
  Capture terminal COMPLETE/FAILED/NOT VERIFIED and partial-target counts. A
  partial replay never supplies a full test result or success marker.

## Metadata-only resource preflight completed

Every original evaluation batch was sized without a model forward or CUDA
initialization. The regenerated batch-order hashes match the previous full CPU
capture, and the first three estimated batch shapes per dataset agree with the
actual unchanged collator. Phi sizes come from NPZ/NPY headers, without loading
the arrays or repository.

| Dataset | HiTSKT maximum FP32 forward estimate | Six RKT Phi matrices: array payload |
| --- | --- | --- |
| ASSIST2017 | 6.799GiB | 0.321GiB |
| Junyi | 4.234GiB | 16.020GiB |
| EdNet-KT1 | 18.964GiB | 25.915GiB |

The conservative forward estimate is three copies of the largest original
causal key/value outer-product prefix plus1GiB overhead. It is not a measured
peak or a promise against OOM. EdNet exceeds the16GiB CPU-forward cap; do not
run that full original-batch CPU replay under this cap. The separate24GiB GPU
estimate cap and8GiB GPU reserve are predeclared from shapes, before any new
CUDA metrics. This resource rule does not alter the numerical tolerance,
architecture, batch shapes or scientific protocol. All GPU work remains gated
on an independent lease and a measured disposable-process memory check.
RKT payload totals exclude runtime overhead, temporary decompression and
forward/metric arrays; fresh host-memory admission is mandatory.

Evidence: [replay_resource_preflight.json](supervisor_verification/replay_resource_preflight.json).

## HiTSKT: first execution priority

| Dataset | Original selected epoch | Test targets | Best checkpoint SHA-256 |
| --- | --- | --- | --- |
| ASSIST2017 | 40 | 112,252 | `085c8d9ae32f4e60892dfb3e4eed93d90a584696d0914456c135dcab7b890164` |
| Junyi | 32 | 2,456,402 | `2308c5bf0c850da1e55daf802a0853a30fac45eeed071c7162c68fc17bfa3bdd` |
| EdNet-KT1 | 30 | 13,429,870 | `58d4b1f393d0fa8ec372ba8bcc00cc8b7168d37817d4badfd26ee42abffd03c8` |

Use each saved configuration:15 historical sessions, token budget32768, batch
cap64, bucket512, shuffleFalse, evaluation sampler epoch0. Replay ASSIST first,
then Junyi, then full EdNet. Score every test target; do not mistake the earlier
Junyi/EdNet float64 samples for full test reproduction.

Retain the established evaluation contract: chronological within-student,
retrospective teacher forcing, right padding, original shifts/masks/EOS and
FP32 sigmoid probabilities classified at `p>=0.5`. Keep original batch-weighted
FP32 BCE and compute an independent stable float64 logit BCE separately.
Compute AUC, accuracy, precision, recall, F1, MSE and BCE using independent
metric code/golden cases, with an auxiliary production-helper crosscheck.

Do two identical fresh-process full replays serially if admission remains valid,
or mark repeatability only sampled if only selected repeats were completed.
Store local float32 logits/probabilities, uint8 labels and int64 event IDs,
hash the files, and compare fresh CUDA against the saved CPU vectors for exact
flip counts and max absolute errors. Bind both source prediction sets by hash.

For every metric report exact equality and **absolute error<=1e-6**, the unchanged
historical-reproduction gate. The overall gate passes only when every required
metric passes and coverage/hashes/selection checks pass. Runtime settings were
not fully saved historically: record any reconstruction uncertainty even if
the metrics match. A TF32-disabled diagnostic is a separate run; no test-score
search over flags, tolerance changes or selective metric reporting.

Original per-target CUDA vectors were not saved. Fresh CUDA-versus-CPU flips
can be counted exactly; they cannot establish the unsaved original pointwise
vectors. Aggregate historical agreement must be labelled separately.

## Completed baseline queue and resource controls

All11 completed baseline metric sets need full replay. Start with the three
claim-critical comparisons; continue serially through the remaining eight:

1. RKT/ASSIST2017 (strongest stored ASSIST baseline), RKT/Junyi and SAKT/EdNet.
2. DKT/ASSIST, DKT/Junyi; DKVMN/ASSIST, DKVMN/Junyi, DKVMN/EdNet;
   SAKT/ASSIST, SAKT/Junyi; RKT/EdNet.

The priority order is an operational schedule, not selection by replay outcome.
Publish every completed scheduled replay, including failures. It does not
create new training runs or complete the missing DKT/EdNet result.

| Model | Saved history | Saved batch size | Additional control |
| --- | --- | --- | --- |
| DKT | 199 prior interactions | 20 | Read finite best weights only; preserve recurrent backend/packing and effective dropout settings. |
| DKVMN | 199 prior interactions | 32 | Preserve saved memory dimensions and affine-scan evaluation path. |
| SAKT | 99 prior interactions | 10 | Preserve question/skill features, masks and relative positions. |
| RKT | 49 prior interactions | 128 | Verify training-only Phi hashes/metadata and relation-parameter evaluation behavior; use no worker copies. |

These history lengths correct the earlier descriptive DKVMN/RKT error; see
[baseline_context_correction.json](supervisor_verification/baseline_context_correction.json).
Preserve each run's transport and historical evaluation grouping. Using a
smaller batch or new packing path may be useful for a separate portability
test, but cannot silently replace a historical reproduction.

RKT's repository constructor loads the all-training Phi matrix plus five
fold-excluded matrices. Read NPZ member uncompressed sizes and shapes before
loading; compressed file size understates RAM. Admit the entire retained
repository plus one forward batch and metrics under the same host budget.
Do not rely on compressed-NPZ mmap or spawn copies across loader workers.
If it does not fit, mark replay NOT VERIFIED pending appropriate resources;
do not alter production Phi selection semantics to force execution. Raw
train-only pair-count reconstruction is a separate streaming/external-sort
verification that must match every persisted count/hash; it is not satisfied
by replaying cached Phi or by sampling pairs. No rebuild is started here.

## Evaluator readiness and acceptance

The existing [evaluate_checkpoint.py](../scripts/evaluate_checkpoint.py) is
**not a safe automatic launch command here**: it chooses CUDA automatically,
calls production `_run_epoch`/metric helpers, and only writes successful exact
matches. It lacks this independent resource gate, persistent failed-run
evidence and per-target output capture. Do not launch it on the occupied GPU.

Before full baseline replay, prepare an isolated evaluator with explicit device
selection and this admission contract, preserving the unchanged model/data
interfaces. Validate its metrics against small independent goldens, and check
its first original batches against production outputs and raw-store target IDs.
CPU-only interface smoke tests are possible with CUDA hidden; a smoke test or
partial replay must remain separately labelled.

Report both exact identity and the predeclared1e-6 numerical comparison. Never
claim the existing exact-equality verifier passed if only the numerical screen
passed. Preserve all failures and all runtime settings. Baseline conclusions
remain NOT VERIFIED until these full target-complete records exist.

## Explicit exclusions and stopping rules

DKT/EdNet lacks `_SUCCESS`/final results and is excluded. Its best validation
checkpoint is not a completed benchmark score. Do not resume epoch10, restore
the nonfinite last optimizer, reset moments, change patience or generate an
invented final result. No V2, FAVOR+, hyperparameter sweep, seed sweep or full
training is authorized by this inference plan. If any immutable artifact hash
differs, a resource lease fails, a target is missing/duplicated, or logits are
nonfinite, stop and retain the failure evidence before deciding the next step.
