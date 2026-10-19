# FAVOR+ numerical quality and GPU readiness

19 October2026 UTC. Base `31886f6ceab1dbcb0243b40468855256ba8f1f5c`, branch
`research/hitskt-favor-plus-v1`. No production architecture or defaults changed.

**Decision: retain128 features for the first bounded diagnostic pilot.
CPU numerical checks PASS; CUDA verification is PENDING. A dataset GPU pilot
cannot begin on the currently visible resource because independent exclusive
ownership has not been established. Full training remains on hold.**

The feature-count recommendation meets the engineering targets frozen before
measurements; it does not establish adequate approximation throughout training
or publication readiness. No test AUC was read or used for feature selection.

## Scope and reproducible design

[Design](numerical_quality_v1/PREDECLARED_DESIGN.json) was committed as
`49ed057` before measurements. Fixed counts128/256/512, projection seeds0–7,
input seeds91/193, head width64, heads2, lengths16/64 and scales0.35/1/3.
These are numerical diagnostics, not a training hyperparameter search.
The prior report used head width8; the present width64 matches the real models.
Errors from the two grids must not be treated as a regression comparison.

The decision uses12 initialized-FAVOR+ frozen attention contexts: four stages
for each dataset. It selects the smallest count whose median context error is
<=0.10 and context p90<=0.20, where each context error is the mean relative L2
output error across eight projection seeds. Other requirements are numerical
gates, peak CPU RSS<=2GiB and CPU time<=4×128 features. These are explicitly
provisional engineering targets. The full rule, fallback and limits are in the
frozen JSON; no criterion was changed after seeing results.

All feature counts use the actual production generator. Reusing a seed does
not make every head's projections nested across counts, because generator
consumption changes with m. Results are averaged across the declared seeds;
individual seeds need not improve monotonically. These are projection seeds,
not independent training runs.

## Root cause: sampling error rather than an observed implementation defect

New independent oracles:
[NumPy/SciPy explicit kernel and softmax](../../scripts/favor_plus_quality_common.py),
[synthetic/distribution study](../../scripts/diagnose_favor_plus_quality.py),
[precision controls](../../scripts/verify_favor_plus_precision.py).
The oracle constructs the feature-kernel matrix explicitly in float64, then
normalizes over causal keys. It does not call production feature transforms,
prefix states, stabilization or backward code. Hand-calculated prefix averages,
a cosh kernel identity and finite differences independently test the oracles.

| Check | Outcome |
| --- | --- |
|288 synthetic feature-kernel comparisons | PASS |
|576 frozen training-sample/context comparisons | PASS |
|Maximum double output difference from independent feature kernel |7.993605777301127e-14 |
|24 double output/gradient comparison cases | PASS; maximum gradient difference3.774758283725532e-14 |
|Gaussian marginal/distribution audit | PASS,32 seeds/32768 rows/d64 |
|Zero/antiparallel kernel identities | PASS independently of Monte Carlo convergence |
|54 independent precision output/gradient controls | PASS |

The generator audit found mean0.000126, variance0.998971, fourth moment2.999171
and squared-radius mean63.934/variance125.542, versus64/128 for chi-square64.
Maximum within-block orthogonality error was9.76e-16. Maximum covariance
off-diagonal magnitude was0.003445; pooled normal empirical CDF distance0.000554.
These are descriptive checks, not iid significance tests: orthogonal rows in a
block are dependent. Gaussian row radii are retained; fixed-radius features
would represent a different kernel.

The positive-feature identity and norm-dependent variance are described in
sections2.3/2.4, Lemma2 and AppendixB of the primary
[Performer paper](https://arxiv.org/html/2009.14794v4). For x=q/d^(1/4),
y=k/d^(1/4), a single Gaussian feature-product estimate Y has
`E[Y]=exp(x·y)` and `Var(Y)/E[Y]^2=exp(||x+y||²)-1`.
For iid averages this relative variance divides by m. ORF correlations can
reduce variance, but this iid expression is explanatory rather than an exact
ORF variance formula. Normalizing random kernel estimates also creates a ratio;
kernel unbiasedness does not imply an unbiased attention output.

Our independent one-dimensional Gaussian quadrature recovered the exact
parallel-input kernel to relative error<=2.23e-16. With scaled vectors
x=y of norm3, even32768 sampled rows recovered only7.22% of the kernel mean
for ORF and4.55% for an independent NumPy iid control. The largest1% of ORF
terms supplied99.84% of the sampled mass. Small-norm controls were near their
analytical means. This is consistent with rare, dominant lognormal tails;
finite-sample underestimation is not proof that the estimator is biased.

The evidence supports **expected finite-feature approximation error**, not a
detected scaling, normalization, projection-law or gradient defect, within the
tested scope. It does not prove the absence of every possible defect.

Mean output MSE versus exact causal softmax on the fixed d64 synthetic grid:

| Input component scale |128 features |256 |512 |
| ---: | ---: | ---: | ---: |
|0.35 |0.002134 |0.001293 |0.000591 |
|1 |0.207327 |0.182987 |0.154108 |
|3 |1.241590 |1.233412 |1.204444 |

At scale3, quadrupling the feature count reduced MSE by only2.99%. Float64
computation still showed this error while matching the independent feature
kernel. Greater precision therefore does not solve this sampling problem.
Increasing m to512 is not a general remedy for high-norm Q/K.
The mean pairwise median feature effective-sample-size was only1.12/1.18/1.23
at128/256/512: the contribution was typically concentrated in about one feature.

## Real training-split norms and feature-count decision

[Training-sample index](numerical_quality_v1/training_sample_summary.json),
[decision and per-dataset summaries](numerical_quality_v1/quality_decision.json).
Four whole target sessions per dataset were selected deterministically by
length bins from1024 equally spaced training-target candidates, with distinct
students, original15-session rolling history, session lengths<=256 and exact
padded-token cost<=2048. No session was truncated. All visible history sessions
were training sessions from the same student. There were12 distinct target
sessions,936 target actions, and **zero optimizer steps**.
This small length-filtered sample is not representative of the full training
population, longest sessions or all possible learned states; only model seed42
was used. It cannot establish a universal feature-count choice.

| Dataset | Target lengths | Largest one-example padded cost | Q-norm p95 range across stages, initialized FAVOR+ | Q-norm p95 range, trained ELU+1 control |
| --- | --- | ---: | ---: | ---: |
|ASSIST2017 |25/41/81/167 |1907 |5.21–5.86 |5.90–16.37 |
|Junyi |13/53/74/138 |1690 |5.05–5.28 |13.44–40.39 |
|EdNet-KT1 |20/55/99/170 |1652 |4.74–5.27 |22.95–57.48 |

Norms are raw projected Q/K per head before d^(-1/4) scaling, including valid
EOS positions. Full K-norm distributions are retained in each manifest.
The initialized model used actual dataset vocabularies and original registered
width/heads/FF/depths; only its existing128-feature attention was used to obtain
frozen contexts. Comparisons then varied the projection count on those same
Q/K/V tensors, with bounded prefixes<=32 and all eight projection seeds.
Changing m in a complete hierarchy could change downstream Q/K; this study is
conditional attention analysis, not a full-model trained feature-count result.

Historical best checkpoints were loaded into **unchanged ELU+1 models** as
separately labeled learned-norm stress controls. Their bytes were preserved.
Their weights were never loaded into FAVOR+, and these control norms must not
be represented as learned FAVOR+ norms or new historical evaluations.

Frozen initialized-context decision population, all datasets/stages weighted
equally:

| m | Mean output MSE | Median context error | Context p90 | MSE reduction vs128 | Long CPU time ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
|128 |0.00267619 |0.096761 |0.158938 |— |1.00× |
|256 |0.00188761 |0.076703 |0.135837 |29.47% |1.25× |
|512 |0.00148414 |0.072388 |0.117974 |44.54% |1.75× |

All three counts passed the frozen initial-context targets. **128 is the
smallest qualifying count**, so the predeclared rule recommends retaining it
for the first diagnostic pilot. Counts256/512 provide meaningful improvement
on these initial contexts at the measured CPU cost, but neither is justified
as a mandatory full-training default. Defaults have not changed.
Output error also depends on the current value vectors and can be small despite
larger kernel/attention-weight errors; these initial-context targets are not a
uniform kernel-approximation guarantee.

On trained-ELU stress inputs, mean relative L2 errors at128/512 were
0.555/0.497 (ASSIST),0.956/0.961 (Junyi) and0.944/0.919 (EdNet).
These are failures of close softmax approximation on those stress inputs,
not failures to compute the specified feature kernel. No claim about future
trained FAVOR+ behavior or invalidity of historical ELU results follows.

## Numerical stability, memory and runtime

[Precision evidence](numerical_quality_v1/precision_controls.json).
Float32 inputs/projection coordinates were held identical for independent
float64 references. Synthetic scales1/30/1000 and both weight-source decoder
contexts for all datasets/counts were checked.
The frozen float32 attention gate is absolute5e-5, with gradient atol5e-5 and
rtol1e-3. This is a new numerical diagnostic criterion, distinct from the
unchanged historical metric acceptance threshold1e-6. Double checks retain
output atol2e-12/rtol2e-11 and gradient atol3e-11/rtol3e-10.

- Default float32: max output error3.822082656435022e-5; max gradient error
  5.377737579692621e-6. All predeclared output/gradient gates passed.
- Existing float64 accumulation with float32 output: maxima1.191000280620358e-7
  and1.643510424975681e-9 respectively. No default precision change was made.
- All outputs/gradients were finite. At the extreme synthetic scale, naive
  float32 exponentiation would zero100% of key features; the existing causal
  scaled/log representation continued to produce finite normalized outputs.
- No Q/K clipping, additional normalization, attention redesign or relaxed
  acceptance tolerance was introduced. Stability is not approximation quality.
  Finite or saturated gradients also do not establish a useful learning signal.

Fresh child processes, one low-priority CPU thread, CUDA hidden, B1/H2/d64,
chunk32; wall time is one forward plus backward, including cold framework
overhead. RSS includes libraries. These are measurements, not GPU forecasts.

| Length including EOS | m | Forward s | Backward s | Peak RSS GB, decimal |
| ---: | ---: | ---: | ---: | ---: |
|257 |128 |0.0250 |0.4314 |0.703 |
|257 |256 |0.0289 |0.3403 |0.703 |
|257 |512 |0.0386 |0.4416 |0.718 |
|3925 |128 |0.3240 |1.6249 |0.703 |
|3925 |256 |0.3840 |2.0603 |0.743 |
|3925 |512 |0.5162 |2.8904 |0.845 |

All resource gates passed. Earlier128-feature13081-position and full-model
13080-action checks remain preserved in [the implementation report](IMPLEMENTATION_VALIDATION.md).
This stage did not certify512 features at maximum length, maximum flattened
history batch sizes, CUDA, mixed precision or long-run learned-norm behavior.
Actual container limit was90GiB; OOM-kill count remained2.

## GPU and pipeline status

[Resource evidence](numerical_quality_v1/gpu_resource_status.json): one visible
H100 MIG; usage-query permissions are restricted, no task-specific exclusive
allocation exists, and protected LightSAKT2 reports an active job. Neither an
empty process table nor a metadata assertion is sufficient proof of ownership.
**No CUDA initialization, tensor allocation, GPU inference or GPU training was
performed.** The CUDA checks below remain PENDING, not passed by CPU inference.

[Staged harness](../../scripts/validate_favor_plus_gpu.py) prepares17/65/257/
939/3925/13081-position checks, chunks16/32/64, explicit feature count, TF32 off,
CPU/NumPy/CUDA output and gradient consistency, causal/PAD/response-shift tests,
full-model backward/checkpoint reload and peak-memory/runtime recording.
It stops on the first failure, with8GiB allocation and120s timed-stage limits.
The external allocation record and exact `CUDA_VISIBLE_DEVICES` UUID must be
validated before any CUDA API use. Such a record documents externally verified
ownership; the script does not obtain an allocation or prove it automatically.
Admission rejection/CPU reference helpers pass tests; the CUDA execution path
remains unverified.

[Dataset compatibility plan](numerical_quality_v1/dataset_pipeline_plan.json)
confirms all three original session stores, historical non-attention profiles
and supervised counts. Actual processed checksums were rehashed by preservation.

| Check | ASSIST2017 | Junyi | EdNet-KT1 |
| --- | --- | --- | --- |
|Dataset/store/profile/count consistency |PASS |PASS |PASS |
|Train-only sampled forward, frozen attention, numerical controls |PASS |PASS |PASS |
|Original non-attention architecture, shifts, loss/metric definitions |PASS |PASS |PASS |
|Validation-AUC checkpoint selection and best reload |PASS |PASS |PASS |
|CUDA correctness/memory/performance |PENDING |PENDING |PENDING |
|New FAVOR+ dataset training/AUC |NOT RUN |NOT RUN |NOT RUN |
|Full-training readiness |NOT ESTABLISHED |NOT ESTABLISHED |NOT ESTABLISHED |

Manual source review and a new synthetic three-epoch control verify that the
runner selects the validation peak at epoch2, keeps last epoch3 separately and
reloads epoch2 before test, even when reported training AUC is higher. This is
a test fixture, not a research result. Original threshold>=0.5, ROC AUC,
accuracy/precision/recall/F1, MSE and masked BCE remain unchanged.

V1 is still chronological **within-student**, retrospective teacher-forced
evaluation, with original rolling visibility. It is not student-disjoint and
does not guarantee actual response availability at prediction wall-clock time.
No availability-aware V2 or new claim of confirmed timing leakage is introduced.

## Failures, preservation and next bounded action

One **new diagnostic attempt failed**: a hand-written cost estimate omitted
session-encoder EOS slots. An assertion stopped before sampled model forward.
The selector now calls the original `padded_token_cost` helper, with the same
frozen2048-token bound. The failed log/directory and
[failure record](numerical_quality_v1/training_sample_first_failure.json) are
preserved; the corrected attempt used fresh `quality_cpu_20261019_v2` paths.
This affected no production preprocessing, sampler, model or result.

The full suite passes229 cases:217 prior cases plus12 new numerical-oracle,
sampling, validation-selection and GPU-admission controls. See
[test log](numerical_quality_v1/full_suite.log) and
[XML](numerical_quality_v1/full_suite.xml). The12 warnings are from the existing
transport-resume tests requesting2 workers under1-CPU affinity.

[Preservation](numerical_quality_v1/preservation.json) verifies375 original
research files and341 additional prior code/report/manifest/local-smoke files,
including all earlier FAVOR+ implementation/evidence and old checkpoints.
Historical ELU results and unrelated README/SAKT edits remain untouched.
LightSAKT2 and DKT/EdNet were not modified, signaled, resumed or reset.

**A small dataset pilot is software/data-ready, conditional on the missing GPU
verification. It must not start on this resource now.** Once an independent
exclusive allocation exists, run the staged synthetic CUDA checks first. Only
after those pass should a separately bounded training-split pilot begin:
128 features, B1, whole-session cost<=2048, at most5 optimizer steps initially,
with per-stage Q/K norms, feature concentration, exact-softmax error on bounded
frozen prefixes, gradients, loss, memory/runtime and checkpoint hashes recorded.
No test-set evaluation is needed for that readiness pilot.

Full training still needs actual CUDA evidence, a measured batch-memory budget,
trained-FAVOR+ norm/approximation monitoring and an explicit decision after
reviewing the pilot. Higher-order gradients remain unsupported. Counts256/512
are diagnostic alternatives, not an automatic search schedule. No architecture
change, dataset optimization, full training, historical retraining or main
merge was performed, and publication readiness is not claimed.
