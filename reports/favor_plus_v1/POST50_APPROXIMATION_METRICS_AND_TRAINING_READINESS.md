# HiTSKT FAVOR+: post50 approximation, metrics and training-readiness investigation

23October2026. Research branch `research/hitskt-favor-plus-v1`; research baseline
`3a9617ab08b854016c5424dcf3357f623b2e4265`.

**No attention/masking/metric correctness defect found in the tested cases.
Approximation REVIEW remains. Conditional GO for another explicitly approved,
short monitored research block; NO-GO for full training or treating50 steps as
predictive validation. No new optimizer update, CUDA model execution, validation
inference or test evaluation was performed. A CPU-autograd internal driver-query
warning is preserved and explicitly qualified below; strict no-probe isolation
was not achieved by that library build. Training remains at50.**

|Check|Status|
|---|---|
|Checkpoint50 and all prior checkpoint/artifact hashes|PASS, unchanged|
|Action/CPE masks, shifting, normalization and actual projection identity|PASS for inspected source and tested cases|
|Production versus independent explicit feature kernel|PASS, max1.771e-6<5e-5|
|Action approximation versus exact causal softmax|REVIEW, persists above0.20 on many inputs|
|CorrectPadding approximation|REVIEW, learned-state drift and longer-target errors|
|Independent metric implementation/label/order/threshold consistency|PASS, no discovered scoring defect|
|Recall/F1 decline|Measured probability/decision shift; not proof of convergence or an attention defect|
|Original-shape1GiB feasibility|NO-GO planning evidence; actual GPU peak NOT MEASURED|
|CPU profiler arithmetic / gradients|PASS on2 small real shapes|
|Strict CPU-only runtime isolation|WARNING: internal device-count attempts; future guard corrected|
|Longer bounded research / full training|CONDITIONAL GO with REVIEW / NO-GO|

## Scope, selection and preservation

[Design](post50_cpu_investigation_20261023_v1/PREDECLARED_DESIGN.json) was committed
before execution as `8b0b683823d2ab589d805fbe27644a27ed62d948`.
[Prepared diagnostic plan](../../experiments/hitskt_favor_plus_v1/assist2017/post50_cpu_diagnostics_20261023_v1/prepared_plan.json)
freezes source/data identities and14 training cases: original six plus one
median-cost example per original length/history stratum, then maximum target
length and maximum singleton cost. Selection uses metadata, not labels,
validation scores or approximation outcomes. Cohort costs117–7164, histories1–11;
the entire training distribution reaches15 histories. These are descriptive,
correlated training examples, not a representative random validation study.

A separate prospective [maximum-history annex](post50_cpu_investigation_20261023_v1/MAX_HISTORY_ANNEX_DESIGN.json)
adds one metadata-selected training example after static inspection exposed that
the main14 cases omitted the longest Action history. It preserves the primary
cohort/feature grid. That example has7 histories including938 actions and cost6643.
No favorable-result replacement or new projection/threshold selection occurred.

All computations use the exact frozen50 checkpoint SHA256
`93ebe9d2a29be9a6e4deb278916513339de61c9ac8f72a454a31111829440a73`.
The hierarchy,128 saved ORFs,seed42,chunk32,FP32 accumulation,Adam state,original
nonattention architecture,data/splits and0.20 threshold remain unchanged.
No optimizer was constructed or called. New raw Q/K/V evidence is under
`experiments/hitskt_favor_plus_v1/assist2017/post50_cpu_diagnostics_20261023_v1/`.
Only training tensors were newly forwarded. Validation analysis reads the same
saved32-session/3157-label predictions at20/25/30/40/50. No test label/example was
used; processed-file integrity checks hash opaque bytes, not test performance.

## Source and mathematical correctness

[ActionEncoder](../../ktbench/models/hitskt.py#L42) embeds complete earlier sessions,
adds positions/input normalization, then applies pre-normalized causal blocks.
Its EOS summary represents the completed history session. [CorrectPaddingEncoder](../../ktbench/models/hitskt.py#L93)
combines the earlier-session vector with the shifted correctness input and
positions, then uses the same pre-normalized blocks. CPE receives previous
correctness at a current prediction position; BOS/EOS/PAD semantics are inherited.
[PerformerBlock](../../ktbench/models/performer.py#L94) supplies LayerNorm before
learned Q/K/V. Large norm drift is not necessary for projection-alignment drift.

The separate FAVOR+ module masks inputs before projection, excludes PAD keys/
values, includes the causal diagonal and keeps numerical prefix gauges causally
compensated. Query-wise constants cancel in normalization; key norms and feature
normalization are retained. Empty prefixes return zero. No new model code was
changed. Generic own/future-label/PAD/EOS/gradient/long-sequence tests remain green;
new oracle tests include holes, empty keys and future perturbations. This does
not newly establish real-world response-feedback availability: evaluation is
still chronological within-student retrospective teacher forcing, not student-
disjoint or guaranteed online response-available evaluation.

For head width d, x=q/d^(1/4), y=k/d^(1/4), the positive feature is
`exp(omega·x-||x||²/2)/sqrt(m)`. Gaussian-marginal orthogonal blocks with chi radii
estimate the softmax kernel; a finite normalized attention ratio remains an
approximation. The IID positive-feature relative kernel variance is
`(exp(||x+y||²)-1)/m`; our logged IID proxy is **not an ORF variance bound**.
Orthogonality reduces variance, without certifying every finite draw. These are
the paper's mechanisms, not an invented guarantee that128 features meet0.20.
[Performer paper,§§2.3–2.4,Eq7–8,AppendixB](https://arxiv.org/html/2009.14794v4).

All four saved projection buffers regenerate exactly from seeds42–45; within-
block orthogonality and radius summaries are recorded. For m>d, independent
blocks are orthogonal internally, not mutually across all rows. The new FP64
oracle constructs explicit feature/exact causal weight matrices and masks before
row normalization. Its compensated gauges are independently crosschecked against
a direct pairwise logsumexp oracle on16 positions and in unit tests. Production
continues to use causal prefix gauges. Max CPU production/oracle difference over
new main cases is1.770896e-6; the additional Action939 case is≤8.676e-7.
This supports numerical implementation correctness on tested representations,
not softmax approximation adequacy or predictive performance.

## Approximation results on identical frozen representations

Reported output error is `||Ahat V-A V||F / ||A V||F`, raw attention before output
projection/dropout. The actual trained128-feature matrices are separate from
fresh diagnostic draws. For the14 main cases, cap128 means first128 positions
or the whole shorter row; this is a mixed-length descriptive cohort.

|Stage|Deployed128 median,cap128|Maximum|Cases>0.20|Median pair-feature ESS|Median exact effective keys|
|---|---:|---:|---:|---:|---:|
|ActionEncoder|0.211255033|0.256385757|10/14|8.132/128|43.848|
|CorrectPaddingEncoder|0.197297617|0.224453059|7/14|9.025/128|49.879|

Effective keys are exp(entropy) of attention over eligible **keys**; pair-feature
ESS measures concentration over **random features**, a different axis. At cap128,
median exact maximum weights are≈0.037(Action)/0.027(CPE), with43.8/49.9 effective
keys. Thus the dominant warning is not simply extremely peaked exact attention.
The random-feature contributions remain concentrated (ESS≈8–9 out of128), and
individual head/query errors can be much larger than full-output medians.
Unscaled Q/K median norms are≈4.5–4.6 before division by64^(1/4); neither norm
explosion nor nonfinite arithmetic is observed.

Production projection, fixed prefix lengths (counts differ because only rows
that reach a length are included; do not interpret as a longitudinal same-cohort trend):

|Stage|Length|Cases|Median error|Maximum error|Cases>0.20|
|---|---:|---:|---:|---:|---:|
|ActionEncoder|8|14|0.141423703|0.223862968|3|
|ActionEncoder|32|13|0.200565053|0.249949709|7|
|ActionEncoder|128|5|0.213903883|0.256385757|5|
|ActionEncoder|256|1|0.194768004|0.194768004|0|
|ActionEncoder|512|1|0.181596923|0.181596923|0|
|CorrectPaddingEncoder|8|14|0.148296697|0.215730364|1|
|CorrectPaddingEncoder|32|10|0.198757599|0.221018326|4|
|CorrectPaddingEncoder|128|6|0.200941490|0.224453059|4|
|CorrectPaddingEncoder|256|3|0.202518950|0.242824692|3|
|CorrectPaddingEncoder|512|1|0.275855336|0.275855336|1|
|CorrectPaddingEncoder|939|1|0.262987898|0.262987898|1|

The annex Action history has939 positions including EOS. Its errors at
128/256/512/939 are0.196072/0.186577/0.170216/0.158714. Longer is not uniformly
worse: causal averaging, values and feature alignment matter. CPE's longest
939-position target case reaches0.262988 (512-prefix0.275855). One long example
per extreme does not establish a distribution-wide guarantee.

Frozen-representation comparison:14 cases×4 projection seeds(0,1,2,3), first
min(128,length), same Q/K/V for all counts. These are56 correlated diagnostic
cases per stage/count, **not training runs or training seeds**. Diagnostic matrices
are FP64; the actual production rounded-FP32 matrices are tested separately.
Projection streams across counts are not fully nested across heads.

|Stage|Features|Median error|p95|Maximum|Cases>0.20|
|---|---:|---:|---:|---:|---:|
|ActionEncoder|128|0.196284281|0.242076826|0.271035235|25/56|
|ActionEncoder|256|0.176692565|0.211449746|0.271912050|7/56|
|ActionEncoder|512|0.141857157|0.177312379|0.215086855|1/56|
|CorrectPaddingEncoder|128|0.117416529|0.201250235|0.210176785|4/56|
|CorrectPaddingEncoder|256|0.084301499|0.123115432|0.167462309|0/56|
|CorrectPaddingEncoder|512|0.076080336|0.181237598|0.221316553|1/56|

Aggregate improvement supports finite-feature sampling/alignment error. CPE's
512-feature p95 is worse than256's and one512 case exceeds0.20: more features do
not guarantee every tail improves. The deployed CPE seed can be less favorable
than other draws on this small cohort; selecting a new seed after seeing it
would be a new research choice, not a correctness fix. No deployed projection,
feature count or checkpoint was changed or chosen using test/validation AUC.

## Why Action persists and CPE worsened

Original six paired25/50 frozen arrays isolate representational sensitivity.
Mean complete-prefix output errors below use the same masks/projection; hybrid
arrays are explicit diagnostic counterfactuals, not model updates.

|Stage|All25|Only Q50|Only K50|Only V50|Q/K50,V25|All50|
|---|---:|---:|---:|---:|---:|---:|
|ActionEncoder|0.194003866|0.194106830|0.194052086|0.194430671|0.194152603|0.194579888|
|CorrectPaddingEncoder|0.153049167|0.171827926|0.169791754|0.154624777|0.189875072|0.191824032|

Action's median remains0.212305: slight median movement is not resolution;
4/6 cases remain>0.20 and its worst head/query error is≈0.948. Frozen swaps have
small effects: this is mostly persistent fixed-feature approximation, not a new
arithmetic problem. CPE's Q/K swap explains most of its mean error increase
0.153049→0.191824; V-only reaches0.154625. Mean attention-weight TV changes
0.108259→0.134008 with Q/K changes, while V-only cannot change the weights.
This local algebraic sensitivity supports learned Q/K alignment drift as the
proximate mechanism; it does not causally attribute training dynamics or identify
one parameter update as responsible.

Scaling frozen Q/K together by0.5/1/1.5 produces original-six median errors
0.037827/0.212305/0.454815(Action) and0.032989/0.192699/0.363861(CPE).
Norm sensitivity is real, even though learned p95 norm growth25→50 was only
0.13%(Action)/2.28%(CPE). Directions/alignment can move without exploding norms.
These probes changed arrays only; **no clipping, normalization redesign or Q/K
rescaling was applied to the actual model**. FP64 feature-oracle agreement rules
out rounding as the main explanation in these cases. Feature concentration and
finite draw variance remain risks; their downstream predictive effect is not
isolated by this study.

## Predictive-metric diagnosis

[Metric implementation](../../ktbench/metrics.py#L44), masks and alignment pass
independent rank-AUC/confusion calculations. All saved milestone label arrays
are identical and preserve3157 targets/1063 positives/2094 negatives. Official
threshold stays0.5 (`>=`); PAD/EOS are excluded. No new validation forward was run.

|Update|TP|FP|TN|FN|Predicted-positive fraction|Mean probability|Recall|F1|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|20|500|1007|1087|563|0.477352|0.488826|0.470367|0.389105|
|25|463|908|1186|600|0.434273|0.470823|0.435560|0.380444|
|30|475|942|1152|588|0.448844|0.477730|0.446849|0.383065|
|40|374|681|1413|689|0.334178|0.433044|0.351834|0.353163|
|50|326|586|1508|737|0.288882|0.415659|0.306679|0.330127|

Between25 and50,137 true positives become false negatives;323 false positives
become true negatives, with one reverse negative-class crossing. Total461
classification changes are explained by the saved probabilities, not evaluation
order or precision mismatch. No saved probability is within1e-6 of0.5 at any
milestone (only9 within0.001 at50). Both classes shift downward: positive mean
0.475344→0.423334, negative mean0.468528→0.411762. Average shift−0.055164 in
probability/−0.244684 in derived logit. Scalar head bias changes only−0.00003844,
so bias alone cannot explain it; head/representation contributions are not
separately identified from saved validation outputs.

Positive-class BCE worsens0.813698→0.937382, negative-class BCE improves
0.683026→0.568226. With66.329% negatives, weighted changes+0.041646 and−0.076145
sum to the overall improvement−0.034500. BCE and accuracy can improve while
recall/F1 fall at a fixed threshold. AUC0.510619→0.519220 is a small descriptive
ranking change, not evidence of convergence or superiority. No experiment here
isolates approximate attention as the cause of F1 decline.

The50 singleton updates saw2528 labels (1055 positive,41.733%), only0.447% of
566107 full training targets. Full training prevalence is37.253%; frozen
validation prevalence33.671%. Session-wise mean pilot prevalence43.874% and
short sessions also affect noisy per-update gradients; the gradient14.98 spike
at44 is a one-target minibatch. This is neither an epoch under the original
batching regimen nor a representative performance study. The mechanism is a
measured short-run decision shift; its future trajectory is unknown.

Predeclared sensitivity thresholds are descriptive, never selected:

|Update|Fixed diagnostic threshold|Recall|F1|
|---:|---:|---:|---:|
|25|0.4|0.658514|0.446856|
|25|0.5|0.435560|0.380444|
|25|0.6|0.250235|0.297373|
|50|0.4|0.529633|0.417501|
|50|0.5|0.306679|0.330127|
|50|0.6|0.149577|0.216917|

The published0.5 contract remains. A lower threshold trading precision for recall
is not an implementation correction, and threshold sensitivity cannot repair
weak ranking. A constant probability equal to this sample's prevalence gives
BCE0.638830/Brier0.223337, versus model50 BCE0.692525/Brier0.247541. This fitted-
to-sample descriptive baseline is not an independently trained benchmark, but
shows that the pilot has not demonstrated useful calibration/generalization.
Changing class weighting, thresholds or checkpoint selection is not warranted
by these results without a separate prospective study.

## Realistic training shapes and CPU memory evidence

Exact original train-only sampler,seed42/epoch1,history15,token32768,maxbatch64:
7060 examples/566107 supervised labels,181 batches; batch size median37/range4–64,
padded-cost median31612/range7606–32766. Target lengths median63/p95=206/max938;
history count median3/p95=7/max15. History sessions reach938 actions. Median
padding fraction45.34%, p95=71.83%, max79.54%; padded positions are still computed.

Singleton cost median354/max7164. The current2048 pilot ceiling excludes158
training sessions(2.238%) containing22467 targets(3.969%). That is valid bounded
pilot admission, **not** a valid way to drop targets during a future full run.
The original32768 sampler admits all training targets without truncation.

Weights occupy27,725,828bytes and projection buffers524,288bytes. Scan retained
states scale with `rows*heads*ceil(L/chunk)*m*(dv+2)` plus log features/V; FF
activations also matter. Recomputed chunks retain a local backward graph:
linear attention in L does not mean negligible batch/history/chunk memory.
Static accounting includes two FF hidden activations and selected scan state,
not all temporaries/workspaces/allocator/driver overhead.

The initial four-parameter-copy planning sum (weights,gradients,Adam m/v) has
median1.004862GiB/max1.094987GiB;93/181 exceed1GiB. Because gradients are set to
None before forward, these copies are not necessarily co-live. The qualified
**gradient-free forward** component sum is median0.979040GiB/max1.069166GiB;
**70/181 still exceed1GiB**, before omitted tensors. The largest one-weighted-
state-per-recomputed-token trajectory component is1.027344GiB. Same-batch
forward+that component reaches2.071178GiB, but this is not a complete or measured
GPU peak. Do not treat2.1GiB as a safe allocation request.

Feature-count planning sensitivity, including a full gradient copy (not all
necessarily concurrent), while leaving FF/nonattention dimensions unchanged:

|Features|Median component sum GiB|Maximum component sum GiB|Max single recomputed-state trajectory GiB|
|---:|---:|---:|---:|
|128|1.004862|1.094987|1.027344|
|256|1.299696|1.434157|2.054688|
|512|1.889427|2.134306|4.109375|

These are CPU shape calculations, not GPU allocation measurements or feature-
count recommendations. The1GiB own allocator cap cannot be presumed adequate
for original batches; physical MIG capacity and free VRAM do not override the
approved cap or shared-use coordination. GPU runtime/full-training ETA remains
NOT ESTIMABLE from these CPU measurements.

Measured CPU profiles, original model dimensions, actual training examples,
forward/backward only and no optimizer:

|Shape|Padded cost|Forward saved bytes MiB|Peak unique saved bytes MiB|Peak process RSS GiB|CPU forward/backward seconds|
|---|---:|---:|---:|---:|---:|
|median_singleton|344|18.956|29.008|0.775349|0.664404|
|largest_singleton_under2048|2047|95.284|99.497|0.996235|1.577374|

Saved-storage tracking excludes parameter/buffer storages and unsaved temporaries;
it preserves gradients exactly in unit tests. The partial retained-component
accounting is only57.3–57.6% of actually saved forward storage in these two cases,
so it is not a complete memory budget. No original full batch met the predeclared
≤4 examples/cost2048 CPU profile admission: the optional third profile was not
run. No original64-example CUDA shape was measured. Hook semantics follow the
[PyTorch saved-tensor API](https://docs.pytorch.org/docs/2.14/autograd.html#torch.autograd.graph.saved_tensors_hooks);
actual runtime is the preserved installed2.4.0a0 build, not the documentation's
version. CPU RSS/saved storage must not be reported as VRAM.

### CPU runtime isolation warning and correction

Both CPU backward children emitted an internal `cudaGetDeviceCount` warning,
error2/out-of-memory, despite CUDA_VISIBLE_DEVICES='' and CPU tensors. PyTorch's
CUDA-linked autograd engine attempted device enumeration. The planned strict
no-GPU-probe condition was therefore **not fully met by library internals**.
No CUDA tensor/model operation or optimizer call was requested; PyTorch's CUDA-
initialized flag stayed false. That flag is not proof of no driver-level call.
The child virtual-address ceiling may interact with driver initialization, but
that cause was not isolated and the warning is not evidence of actual VRAM
exhaustion. No further CUDA/NVML probe was made to investigate it.

Raw logs/results are retained. [Executed source snapshot](post50_cpu_investigation_20261023_v1/EXECUTED_SOURCE_SNAPSHOT.json)
preserves the exact source hash from the frozen plan. The current runner now
[fails closed before CPU backward](../../scripts/favor_plus_post50_cpu.py#L55)
with a CUDA-linked build; future strict CPU backward profiling requires a CPU-
only PyTorch build. No package was replaced. Two new guard cases plus the
existing visibility guard pass without creating an autograd graph. This is a
diagnostic-tool correction, not an attention or scientific-protocol change.

## Meaning of0.20 and required corrections

0.20 is the unchanged project review trigger for aggregate relative output L2.
It means20% of the comparator output's Frobenius norm; it does not mean20% label
error,20% wrong attention decisions or a bound on AUC. It is invariant to common
V scaling but sensitive to value direction/cancellation and small exact-output
norms. Constant V can hide attention-weight errors entirely; this is explicitly
unit tested. Stage aggregation and prefix selection can hide head/query tails.

It remains meaningful as a **consistent warning on the identical metric/input
protocol**. It is not a calibrated universal downstream pass/fail criterion.
Preserve every historical REVIEW result and supplement it with weight TV,
feature ESS, absolute/value-normalized errors, norm/direction diagnostics and
subsequent properly selected held-out performance. No threshold was relaxed.
No evidence currently requires rewriting the attention, rescaling Q/K, resetting
Adam, discarding checkpoint50 or retraining the original ELU+1 models.

Required before confident scaling: an approved resource plan for original
shapes; continued learned-state approximation/tail monitoring; full validation-
based selection and independent held-out replay after meaningful training.
The strict CPU-profiler build guard is the only implemented tool correction.
No architecture, feature policy, class weights,threshold or dataset was changed.

## Decision and next bounded proposal

**CONDITIONAL GO for another short monitored research continuation, retaining
REVIEW. NO-GO for full ASSIST2017/Junyi/EdNet training or professor-facing claims
of successful predictive performance. No training/GPU execution is authorized by
this report.** The existing variant is mathematically/causally testable and
checkpoint-reproducible; approximation limitations alone do not invalidate its
identity as genuine FAVOR+ or require changing architecture.

Recommended next instruction, if the user chooses continuation: exact50→75,
maximum25 new calls, original128 projections/Adam/RNG/sampler, unchanged shared
1GiB/2GiB advisory/8–6GiB free/B1/cost2048 controls, fresh admission and atomic
checkpoints55/60/65/70/75 with fresh recovery60/75. Freeze the next actual sampler
identities and the expanded training-only diagnostic cohort before launch;
monitor at50/55/60/65/70/75. Keep0.20 REVIEW and all existing severe/resource
stops. Use a finite time budget and no retry/reset. Evaluate the existing frozen
validation sample only at predefined milestones, with classwise BCE/confusion
counts alongside AUC. This would study stability and learning trajectory, not
provide final results. Do not launch automatically.

Before original-scale/full training, obtain a separately bounded **zero-optimizer
GPU shape profile** with shared permission and explicit caps. Start below the
current cap using representative actual training shapes, stop on predicted or
observed cap/headroom/time/numerical failure; do not attempt known-over-budget
original shapes or raise limits automatically. Present measured resource needs
for a separately approved larger envelope. Preserve all histories/targets and
original nonattention training settings; reducing batch/history for a full run
would be a separately documented methodological decision.

A feature-policy change requires an explicit separately versioned comparison,
not editing this trained lineage to improve diagnostics. Frozen256/512 evidence
is motivation for that decision, not approval or test-tuned selection.

After numerical/resource review and explicit full-training approval, use original
preprocessing/splits/full validation AUC/strict checkpoint improvement/patience5.
Reload the validation-selected best checkpoint in a fresh process, then evaluate
the untouched full test split for all seven metrics with saved predictions,
checkpoint/data hashes and independent metric verification. Keep response-timing,
within-student/retrospective and single-seed qualifications. Baseline comparisons
need comparable protocols and independently replayed results; no significance
claim follows from one seed. Junyi/EdNet need their own train-only shape/numerical
profiles and bounded real-data pilots before any full schedule.

## Verification and evidence index

82 main CPU cases PASS(18 new diagnostics+58 FAVOR++6 original-model regressions).
Three CPU-isolation guard checks PASS(2 new,1 overlapping).84 distinct cases
across those suites, not85; the earlier18-case development run overlaps the82.
Historical229 CPU/18 CUDA evidence is not newly counted. No full GPU test suite,
new validation/test model evaluation or optimizer update occurred.

[Independent verification](post50_cpu_investigation_20261023_v1/VERIFICATION_SUMMARY.json),
[metric shift supplement](post50_cpu_investigation_20261023_v1/METRIC_SHIFT_SUPPLEMENT.json),
[memory accounting qualification](post50_cpu_investigation_20261023_v1/MEMORY_ACCOUNTING_QUALIFICATIONS.json),
[CPU environment warning](post50_cpu_investigation_20261023_v1/CPU_PROFILING_ENVIRONMENT_WARNING.json),
[artifact inventory](post50_cpu_investigation_20261023_v1/ARTIFACT_INVENTORY.json).

Rehashed774 previous50-run files,316 previous25 files,199 original20 files and
9+8 failed-attempt files unchanged, including all original checkpoints. All35
prior bound sources and11 processed hashes unchanged. LightSAKT2 hold and
unrelated README/SAKT files unchanged. No unrelated job was signaled or modified;
no allocation, background watcher, full training, main merge or history rewrite.
Checkpoint binaries remain local/ignored; Git is not their off-machine backup.
