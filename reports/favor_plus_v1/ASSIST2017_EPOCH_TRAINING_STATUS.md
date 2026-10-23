# ASSIST2017 FAVOR+: epoch-training preparation and resource-admission status

23October2026, branch `research/hitskt-favor-plus-v1`; baseline
`df3dcb43ba264a668364f36890862c6e433d7fce`.

**Actual full epochs completed:0. New real-data optimizer updates:0.
Full-validation AUC/F1/BCE: NOT RUN. CUDA model initialization:NOT RUN.**
The user has authorized a fresh complete-data run of up to10 epochs with
patience3. That training permission is accepted; it is not being requested again.
Execution is blocked by the existing1GiB allocator /batch1/cost2048 resource
limits, which cannot admit the original full-data batches. A separate budget
question is pending; no response or approval is inferred from its default option.

## What is ready

New controller:[favor_plus_assist_epochs.py](../../ktbench/favor_plus_assist_epochs.py).
CLI:[train_assist_favor_plus_epochs.py](../../scripts/train_assist_favor_plus_epochs.py).
It preserves the original model/attention/data/shared-safety source files.
The older generic FAVOR+ entry point uses patience5 and automatically evaluates
the test set, so it was preserved and a separate scoped entry point was prepared.

- Fresh seed42 model and Adam, with128 fixed genuine FAVOR+ positive ORFs,
  chunk32,FP32 accumulation, highest matmul/TF32off and original nonattention
  architecture (width256/heads4/FF2048/dropout0.1,one layer per stage).
- AdamLR5e-5,betas0.9/0.999,eps1e-8,weight decay0; no clipping or feature tuning.
- Original full-data bucket sampler:history15,token32768,maxbatch64,bucket512,
  seed42 and explicit epoch1..10 ordering. No oversize-target omission.
- **User-requested max10 epochs and patience3**, local to this run. Original
  registry ceiling100/common patience5 and all historical runs remain unchanged.
  These stopping differences must be disclosed in future benchmark comparisons.
- Strict full-validation-AUC checkpoint selection,min_delta0; best observed F1
  is recorded separately and does not replace the AUC selection criterion.
- Full train/validation target accounting each epoch; validation includes AUC,
  accuracy,precision,recall,F1,BCE and MSE. No test data loader or test inference
  action is exposed by this new CLI.
- Immutable atomic/fsync/roundtrip checkpoints every25 batches and each epoch:
  model,Adam,RNG,actual ORF buffers,exact epoch/batch target identities,cursor,
  metric prefixes and source/data/configuration hashes. The partial-resume CPU
  test reproduces subsequent updates/metrics exactly; no failed optimizer step
  is automatically restarted. Explicit recovery of a stopped study is an
  operator action, not a background retry.
- Fresh-process CUDA checkpoint proof required after each epoch before the
  next epoch; fixed predictions and restored-RNG training loss/gradient digest,
  zero optimizer calls. Final selected checkpoint has a separate fresh full-
  validation replay command with exact score/label/logit comparisons and the
  unchanged1e-6 metric criterion. These CUDA checks are prepared, **not run**.
- Training-only fixed diagnostics on the original six cases/all four stages at
  initialization and each epoch: independent exact-softmax/feature-kernel
  comparisons,Q/K norms and feature ESS.0.20 REVIEW and original severe stops
  remain unchanged. No attention architecture change was made to force a pass.
- Curves are generated only from actual completed epoch metrics. No production
  curve was fabricated from diagnostic50 or synthetic test data.

Diagnostic50 remains a valid preserved historical checkpoint. A fresh run was
chosen to keep the earlier singleton/subset optimization separate from the
full-data,original-batch protocol. No trained diagnostic weights or Adam state
were imported into the new full-run configuration.

## Dataset and implementation verification

All35 prior bound model/controller source hashes and11 verified processed
ASSIST2017 identities match. Existing independent masking/gradient/stability
findings remain applicable; the historical229 tests and RFF sweeps were not
repeated. The previous Action error0.212305 is a retained approximation REVIEW,
not a discovered masking or arithmetic defect. The post50 investigation found
production/reference agreement within1.771e-6, concentrated finite random
features and learned Q/K alignment drift, especially in CPE. No model correction
was justified; no threshold,feature count or projection was changed.

The new CPU preparation verifies student boundaries,contiguous session numbering,
monotone within-student train/validation/test split metadata,strictly prior
history bounds,same-student history and no future-split visibility. Hash checks
read opaque processed bytes; test examples/labels were not used for selection.

|Split used by this run|Complete target sessions|Supervised labels|
|---|---:|---:|
|Training|7060|566107|
|Validation|1977|125800|
|Test evaluation|NOT RUN|NOT RUN|

The original885335 retained interactions remain in the unchanged store. First
sessions are context rather than supervised targets under the original HiTSKT
protocol; they were not deleted. All valid target sequence lengths are retained
(up to938 actions plusEOS), with no action truncation/chunking/filtering. Internal
attention checkpoint chunks are computation, not data truncation.

Evaluation remains **chronological within-student retrospective teacher forcing**,
not student-disjoint evaluation or a guarantee of real-time response availability.
The previously documented response-timing qualification remains; no new claim
of confirmed leakage or guaranteed online availability is made.

## Genuine execution blocker

The [existing approved policy](shared_gpu_v1/PREDECLARED_POLICY.json) enforces
1GiB own allocated/reserved,2GiB advisory own total,batch1/cost2048,8GiB admission
free/6GiB runtime free,4GiB host RSS/2GiB host headroom,0.1s pacing,30s per update
and45min execution ceiling. Training authorization does not silently raise these
resource limits.

The first actual planned batch has **64 examples /18,642 padded-token cost**.
Across ten planned training epochs and one full validation schedule,1898 batches
violate the old batch/cost limits. The [admission attempt](assist_epochs_v1_20261023/CURRENT_LIMIT_ATTEMPT.json)
was refused before CUDA initialization or any model/optimizer work. No target
was dropped to make it fit. The proposal's larger integer shape limits admit
all schedules, but this does **not** establish GPU-memory fit.

Previous qualified CPU accounting found70/181 epoch1 batches exceed1GiB in partial
forward components even after excluding gradient buffers, before omitted tensors
and recomputation/workspace overhead. Thus merely changing the token limit would
not establish a safe original-shape run. No new massive audit was required to
rediscover this. Resource/shape gates are kept intact rather than bypassed.

Initial read-only observation this task: assigned
`MIG-d2fcfd79-720e-5218-82c9-0a132be68920`,H1003g.40gb,39.337952GiB free,
no reported compute worker/no visible LightSAKT2 trainer,29.295815GiB shared-host
headroom. MIG utilization is unsupported and remainsUNKNOWN. This is a dated
observation,not a reservation or future admission receipt. Shared use is already
authorized; exclusive ownership,administrator metadata and host PID mapping are
not being demanded. Other jobs/LightSAKT2 were not signaled or modified.

## Concrete budget proposal awaiting response

Requested **8GiB hard PyTorch allocated/reserved cap,10GiB advisory total,
16GiB admission-free floor**, originalmaxbatch64/cost32768, keeping runtime free
floor6GiB,host limits,pacing,visibility/protected-Light checks and30s/45min stops.
This is a resource change requiring the actual pending response; no approval is
invented and old policy files are not edited. The larger total remains advisory,
not a certified process-total limit without PID attribution.

First run a staged **zero-optimizer-update** CUDA memory profile: one actual
singleton,small/median/maximum-cost original batches,worst history-row workspace
shape and full-validation worst-cost shape. Preallocate Adam-sized moment
buffers without a fake Adam update. Record gradients,timing,own allocator peaks
and sampled device-wide pressure. Stop at the first memory/30s/numerical failure;
8GiB is a proposed ceiling, **not a promise that every shape fits**. No automatic
budget increase or profile retry follows a failure.

Only after matching profile PASS and fresh admission run epoch1/full validation,
then fresh checkpoint recovery and review its resource/numerical outcomes. The
existing approval then permits epochs2..10 automatically until patience3 or a
safety/time stop. No repeated per-epoch training approval is needed. The45min
execution ceiling remains; reaching it stops rather than extending runtime.

Prepared run:
`experiments/hitskt_favor_plus_v1/assist2017/assist2017_full_epochs_pat3_20261023_v1/`.
Plan SHA256:
`8a0958f4572e11d9ae0d5722f595ef070c84129d9460f4b382b70c70ac2427cb`.
The current-limit record intentionally retains1GiB. A separately recorded
8GiB response-bound execution record does not exist until approval is received.

Archive/preparation commands (do not recreate the existing prepared directory):

```text
scripts/train_assist_favor_plus_epochs.py prepare --run-id assist2017_full_epochs_pat3_20261023_v1
```

After an actual budget response, use fresh approved record/UUID/environment,
with one CPU thread,taskset223,nice10,PYTHONHASHSEED42,CUBLAS_WORKSPACE_CONFIG=:4096:8:

```text
scripts/train_assist_favor_plus_epochs.py profile --plan PLAN --record APPROVED_RECORD --execute-cuda
scripts/train_assist_favor_plus_epochs.py train-epoch --plan PLAN --record APPROVED_RECORD --epoch 1 --execute-cuda
scripts/train_assist_favor_plus_epochs.py verify-epoch --plan PLAN --record APPROVED_RECORD --epoch 1 --execute-cuda
```

Then sequential train/verify for permitted epochs2..10 only while selection.stop
is false and resource/time gates pass. Finally `validate-best` performs full
validation replay only. No command above changes the GPU allocation,stops a peer,
starts a background watcher or evaluates the untouched test split.

## Current evidence and limitations

25 main CPU controller/serialization/data/metric tests PASS,one curve test PASS,
two exact operator-window tests PASS:28 distinct cases. Tested interrupted
checkpoint restoration reproduces next optimizer updates,complete epoch metrics,
weights,Adam and RNG exactly on tiny synthetic CPU data. This is **not a real
ASSIST epoch or CUDA certification**. Historical full FAVOR+/causal tests were
reviewed rather than unnecessarily repeated.

An initial development test assumed a two-batch checkpoint interval while the
production default is25; the fixture now explicitly uses a two-batch interval.
The initial failure is preserved. An initial operator record used two separate
clock reads and exceeded the2h bound by microseconds; it was correctly refused
before CUDA. The corrected record uses one timestamp plus2h; the strict bound
was not relaxed and two edge-case tests pass. These were preparation issues,
not failures of FAVOR+ equations or actual training. No optimizer step was retried.

|Requested output|Actual current status|
|---|---|
|Completed full epochs|0|
|Full-epoch validation AUC/F1/BCE|NOT RUN|
|Best full-validation checkpoint|NOT CREATED|
|Epoch curves|NOT GENERATED; no measured epoch data yet|
|Fresh CUDA epoch checkpoint recovery|NOT RUN; CPU synthetic proof PASS|
|Untouched test evaluation|NOT RUN|
|Diagnostic50 preservation|PASS|

A final professor-facing performance report will require actual completed epoch
metrics and checkpoint verification. The earlier sampled AUC0.5192/F1 values
must not be presented as full-validation results. No convergence,final-test or
model-superiority claim is justified here.

[Prepared manifest](../../experiments/hitskt_favor_plus_v1/assist2017/assist2017_full_epochs_pat3_20261023_v1/prepared_plan.json),
[current-budget refusal](assist_epochs_v1_20261023/CURRENT_LIMIT_ATTEMPT.json),
[hypothetical shape fit—not approval](assist_epochs_v1_20261023/PROPOSED_BUDGET_SHAPE_FIT.json),
[verification](assist_epochs_v1_20261023/VERIFICATION_SUMMARY.json),
[professor update](assist_epochs_v1_20261023/PROFESSOR_UPDATE.md).
