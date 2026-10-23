# HiTSKT FAVOR+: shared-GPU20→25 pilot results

23October2026. Branch `research/hitskt-favor-plus-v1`; implementation/design
committed before CUDA as `19da63c5f542ec803df69659d739cbfa8ed64bd1`.

**PASS for shared-mode execution, finite optimization and checkpoint recovery.
APPROXIMATION REVIEW remains. Exactly FIVE new optimizer calls completed;
training stopped at25TOTAL. No update26 optimizer call or full training.**

|Check|Result|
|---|---|
|Fresh shared admission / approved limits|PASS|
|Exact checkpoint20 model/Adam/RNG/projection recovery|PASS, fixed logits difference0 and exact next loss/all-gradient digest|
|Original frozen validation20 replay before update21|PASS, probabilities/labels/metrics exactly equal|
|Updates21–25 / parameter changes|PASS, five returned calls and all nonzero parameter-delta norms|
|Gradients/model/Adam / CUDA errors|PASS, finite; no observed CUDA/OOM/error or retry|
|Atomic25 checkpoint / separate fresh CUDA recovery|PASS, zero optimizer calls during recovery|
|Current/future-response perturbation invariants|PASS on the unchanged checked cases at20 and25|
|Independent explicit feature-kernel agreement|PASS; max absolute error at25 6.826e-7, limit5e-5 unchanged|
|ActionEncoder softmax approximation|REVIEW, median0.212836>0.20;4/6 prefixes exceed review|
|Host/container PID mapping / attributed process-total VRAM|NOT VERIFIED / NOT MEASURABLE; shared mode does not require exclusive attribution|
|Full training readiness|NO-GO; further bounded research only with approval|

## Configuration, admission and preservation

Run: `experiments/hitskt_favor_plus_v1/assist2017/assist25_shared_cuda_20261023_v1/`.
[Plan](../../experiments/hitskt_favor_plus_v1/assist2017/assist25_shared_cuda_20261023_v1/prepared_plan.json)
SHA256 `8b677f602950f438812e0b9d1cd7597a84a00484a58e71e95f5e8291964c2f4a`.
[Predeclared design](shared25_pilot_20261023_v1/PREDECLARED_DESIGN.json) and
[actual user authorization](shared25_pilot_20261023_v1/USER_PILOT_AUTHORIZATION.md).

Genuine `hitskt_favor_plus_v1`,128 fixed random features, original seed42/Adam/
nonattention architecture/FP32 settings/preprocessing/splits/sampler. Same
chronological within-student retrospective teacher-forced protocol and original
diagnostic/frozen validation samples. No test-set evaluation or scientific
configuration/attention/threshold change. New pilot CLI/contract hard-cap total25
and five additional calls, independently of the historical20→50 permission.

Approved1GiB allocator+reserved cap,2GiB advisory total,8GiB admission free,
6GiB runtime free, B1/cost2048,4GiB process RSS/2GiB host headroom,30s update/
45min campaign budget and0.1s inter-update self-pacing. No limit was increased.
The older general policy's50-update capability is not this pilot's permission.

Exact H1003g.40gb **MIG-d2fcfd79-720e-5218-82c9-0a132be68920** was visible;
fresh two-snapshot admission passed. Initial free memory39.34GiB, no reported
compute worker, adequate host headroom, protected LightSAKT2 hold active. MIG
utilization remained unsupported/status3, recorded as **load UNKNOWN**, with
the approved shared-resource fallback. [Admission](shared25_pilot_20261023_v1/resource_admission.json).

Runtime: PyTorch2.4.0a0+f70bd71a48.nv24.06, CUDA12.5, H100 capability9.0,
deterministic algorithms, highest FP32 matmul, TF32 off, cuDNN deterministic /
benchmark off, one CPU thread and CUBLAS_WORKSPACE_CONFIG=:4096:8.
[Runtime](../../experiments/hitskt_favor_plus_v1/assist2017/assist25_shared_cuda_20261023_v1/stage_025/runtime.json).
All33 bound scientific/controller sources remained hash-matching throughout.

## Five training updates and numerical stability

|Total update|Training BCE|Gradient L2|Parameter-delta L2|Targets|Padded cost|Synchronized interval seconds|
|---:|---:|---:|---:|---:|---:|---:|
|21|0.631208897|1.902359|0.023794|58|127|0.351730|
|22|0.690848172|1.939177|0.023334|66|110|0.302506|
|23|0.779484808|2.588187|0.022215|49|376|0.304601|
|24|0.686952651|2.274305|0.022505|66|158|0.295892|
|25|0.639421523|2.225239|0.023279|62|160|0.537598|

301 genuine training targets, target-weighted BCE **0.682338406**. The historical
update20 minibatch BCE was0.697781444; different batches/dropout mean these are
not a monotonic learning curve. Gradient range1.902359–2.588187 is finite and
below unchanged severe limit100. All gradients were present; all model/Adam
states stayed finite. All61 Adam states now have step25; second moments remain
nonnegative (min0, max0.004183977). Overall saved state delta20→25 L2=0.109592.
Fixed projection buffers remain identical to20; optimizer settings were not reset.

## Approximation, concentration and causal checks

Same six training-only examples, same bounded frozen prefixes and all four
attention stages, independent NumPy/SciPy float64 feature-kernel/exact-softmax
references. Errors are median complete-prefix relative L2 across six cases;
the0.20 review trigger and separate severe-stop criteria are unchanged.

|Stage|Error20|Error25|Maximum case25|Prefixes>0.20 at20→25|Median pair ESS20→25|
|---|---:|---:|---:|---|---|
|ActionEncoder|0.213145477|0.212835732|0.235301773|4/6→4/6|8.181→8.194|
|CorrectPaddingEncoder|0.142361690|0.151291355|0.182097336|0/6→0/6|9.732→9.506|
|Decoder|0.072271783|0.079839188|0.126186932|0/6→0/6|9.443→9.335|
|SessionEncoder|0.095313878|0.085124980|0.160381411|0/6→0/6|5.721→5.541|

**Action REVIEW persists**, with nearly unchanged median and slightly increased
maximum case. CPE error rises modestly but remains below0.20 on this cohort.
The largest matched Q/K p95 growth from20 to25 is1.004743× (0.4743%); largest
ratio to the original untrained baseline is1.018511×, below unchanged2× stop.
No norm explosion or numerical nonfiniteness was observed. Individual pair ESS
still reaches about1.016 in ActionEncoder, so concentration/tail approximation
risk remains despite healthy stage medians. No feature-count tuning occurred.

At25, max production-versus-independent-feature-kernel error6.825843e-7 passes
the5e-5 limit. This establishes tested implementation agreement, not accurate
softmax approximation. All unchanged severe containment gates passed. Raw-response
re-collations flipping positions0/144/287 preserved own/earlier predictions exactly
at20 and25. These sequence-causal checks do not resolve the original feedback-
availability/timing limitation of retrospective teacher-forced evaluation.

## Memory, overhead and time actually observed

Fresh process allocator peaks include initialization: **no peak reset** was used.
Training and fresh recovery ran sequentially; their peaks must not be added.

|Measurement|Training/diagnostics process|Fresh recovery process|
|---|---:|---:|
|Peak PyTorch allocated|249.984MiB|199.776MiB|
|Peak PyTorch reserved|338.000MiB|216.000MiB|
|Maximum sampled device-wide NVML used|744.688MiB|620.688MiB|
|Minimum sampled free VRAM|38.648GiB|38.769GiB|
|Peak process RSS|1.864GiB|1.738GiB|
|Minimum shared-host headroom|29.457GiB|29.623GiB|
|Resource observations|112|3|
|Process execution interval|29.968084s|3.497301s|

The maximum observed device-wide used memory outside own PyTorch reserved pool
was **406.688MiB** in training; subtracting that phase's pre-context baseline
gives a maximum **368.812MiB residual**. These measured residuals include driver/
context/workspaces, system memory and possible peer changes. **They are not
certified own-process overhead or a continuous VRAM peak.** Host PID mapping is
still unresolved. Device-wide NVML maxima are synchronous sampled maxima; the
own PyTorch peaks are allocator peaks. The2GiB total is advisory, and all observed
device-wide usage was below it; unobserved driver transients cannot be excluded.

Five synchronized update intervals total1.792327s (mean0.358465s/update), or
2.790 updates/s and167.938 targets/s for these intervals. They include update
work/checks and exclude later diagnostics/checkpointing/pacing. The complete
29.968s stage also includes initial exact recovery, pre/post diagnostics and
sampled validation, resource journaling and checkpoint I/O. These are bounded
shared-load observations, not isolated kernel throughput or full-training forecasts.
Execution intervals exclude module imports and plan/admission preflight.
No CUDA/OOM/resource-limit error, step retry or budget increase was observed.

## Checkpoint and metric verification

Original20 SHA256
`0388e984b8bed21210455be561524a041c892b98b2a87b50fa3ecb1c3ade86f4` unchanged.
New25 SHA256
`e67724a4ade70fe04de9d0c1a1f0fcb8bd1acc984c3506cda5cce2685ab2b5fc`,
83,819,514bytes, atomic/no-overwrite CPU-validated serialization.
[Receipt](../../experiments/hitskt_favor_plus_v1/assist2017/assist25_shared_cuda_20261023_v1/stage_025/checkpoint_receipt_025.json),
[fresh CUDA recovery](../../experiments/hitskt_favor_plus_v1/assist2017/assist25_shared_cuda_20261023_v1/verify_025/recovery_result.json).

Checkpoint25 preserves weights, all61 Adam states, RNG states, actual projection
buffers, sampler cursor25, exact sample identities and source/data/config/scope
hashes. A separate fresh CUDA process reproduced fixed predictions with maximum
difference0 and the next train-mode logits/loss/all-gradient digest exactly,
without changing model/Adam/RNG/projections or taking an Adam step. Sample26 was
only a forward/backward recovery proof, never an optimizer update. Checkpoint
binaries remain locally preserved/ignored as before; Git is not a binary backup.

The original frozen32-session/3157-label validation sample was evaluated using
the unchanged protocol. **Preliminary sampled results only, not full-dataset
performance or validation-selected best-checkpoint results:**

|Metric|Update20, exact replay|Update25|
|---|---:|---:|
|Validation BCE|0.741282668|0.727024655|
|ROC-AUC|0.508313409|0.510618521|
|Accuracy|0.502692430|0.522331327|
|Precision|0.331785003|0.337709701|
|Recall|0.470366886|0.435559737|
|F1|0.389105058|0.380443714|
|MSE|0.270515942|0.263983875|

Recall and F1 decreased. The small AUC increase is not significance, convergence
or superiority. Saved probabilities independently reproduce rank-AUC and score
metrics within1.12e-16. BCE reconstructed from rounded probabilities differs
from logit-derived BCE by2.48e-8 at20 /7.18e-9 at25; no tolerance was changed.
This is cached-score consistency, not an additional fresh full-validation25 replay.

75 targeted CPU tests PASS (20 new pilot cases +55 existing shared-policy cases).
The earlier90/229 suites and18 CUDA scientific tests are inherited, not newly
summed. [CPU evidence](shared25_pilot_20261023_v1/cpu_tests.log),
[independent saved-evidence verification](shared25_pilot_20261023_v1/VERIFICATION_SUMMARY.json).

## Decision

**GO, retaining approximation REVIEW, for a separately approved bounded25→50
research continuation from the NEW25 checkpoint. NO-GO for full training.**
Shared mode worked in the observed resource conditions and preserved deterministic
recovery while avoiding false ownership claims. Actual concurrent high-pressure
GPU contention was not observed; its stop behavior is CPU-tested. High-norm /
long-history approximation adequacy and original large batch shapes remain open.

The next study needs a NEW25→50 scope/contract and fresh shared admission, keeping
the same resource limits, science and milestone recovery. Do not use the old
20→50 controller, which starts from20, or reset Adam/RNG. Current approval is
fully consumed: exactly25TOTAL, no more training now. Full training still needs
separate approval and broader numerical/resource/performance verification.

[Final read-only release](shared25_pilot_20261023_v1/resource_release.json) observed
no compute worker and near-baseline37.94MiB device use, with LightSAKT2 inactive
and hold hash unchanged. No other job/process, allocation, ELU+1 checkpoint/result,
LightSAKT2 artifact, DKT/EdNet state or previous report was modified. No watcher,
process signal, test evaluation or full-training launch occurred.

[Preservation/inventory evidence](shared25_pilot_20261023_v1/ARTIFACT_INVENTORY.json)
rehashes199 original20-campaign files including checkpoints5/10/20,9 earlier
failed-continuation files and8 latest failed-attempt files unchanged. The new
run has316 captured files:315 noncheckpoint artifacts plus the locally preserved
checkpoint binary. Existing unrelated README/SAKT working-tree changes were
left untouched and excluded from this work.
