# HiTSKT FAVOR+: controlled four-arm verification and training proposal

24 October 2026. Branch `research/hitskt-favor-plus-v1`; starting commit
`33a17cf9e744342b2436d818554d4c619135e66b`.

**Decision: GO for a separately approved, bounded matched learning comparison;
REVIEW / NO-GO for selecting a full-epoch winner yet. Zero optimizer updates
were performed in this study.** Antithetic features help some stages but are
not a universal correction. The original eight-epoch experiment and its
best epoch-7 checkpoint remain unchanged.

The new CUDA study found a float32 stress-case precision failure. It was
preserved and stopped. A separately declared configuration using the existing
float64 attention-accumulation option passed all four arms without changing
attention equations, numerical tolerances or the 0.20 approximation-review
threshold. This report does not relabel the float32 failure as a pass.

## Evidence and scientific scope

- [Initial predeclared design](four_arm_cuda_comparison_20261024_v1/PREDECLARED_DESIGN.json)
  and [explicit FP64 follow-up](four_arm_cuda_comparison_20261024_v1/fp64_v2/PREDECLARED_DESIGN.json).
- [Machine-readable comparison summary](four_arm_cuda_comparison_20261024_v1/fp64_v2/COMPARISON_SUMMARY.json),
  [unchanged-threshold table](four_arm_cuda_comparison_20261024_v1/fp64_v2/APPROXIMATION_TABLE.md),
  [final test receipt](four_arm_cuda_comparison_20261024_v1/fp64_v2/regression_tests.xml).
- Raw run roots:
  `experiments/hitskt_favor_plus_v1/assist2017/four_arm_cuda_readiness_20261024_v1/`
  and `four_arm_cuda_readiness_fp64_20261024_v2/`.
- [Previous learned-alignment investigation](POSTEPOCH8_APPROXIMATION_INVESTIGATION.md)
  remains the evidence about the historical trained model, rather than a fair
  training comparison involving the new antithetic method.

Four controlled arms: ORF128, ORF512, antithetic128 and antithetic512. All use
positive softmax-kernel features, original HiTSKT non-attention architecture,
fixed projections and causal prefix attention. Antithetic M features use M/2
Gaussian orthogonal base directions plus their negatives. Both exponential
signs are positive; the expected kernel is unchanged. Normalized attention
is a ratio estimator and is not thereby exactly unbiased.

All arms use the same nested 512-row projection pool per head/stage, seed
42 plus stage offset. ORF takes the first M rows; antithetic takes the first
M/2 rows and their negatives. **This paired-draw policy differs from the
historical native 128-row sampler for some heads.** The historical sampler
and checkpoints were not modified. The new wrapper uses checkpoint format 4,
separate from original ELU+1 format 1, original FAVOR+ format 2 and native
antithetic-prototype format 3. Learned parameter hashes and post-factory CPU
RNG states match across all four fresh arms.

The attention comparison uses 14 preselected genuine ASSIST2017 TRAIN examples,
four stages, and the same 56 frozen **initial** Q/K/V/mask arrays. No epoch-7/8
trained representations were used as evidence of a fair new-method comparison.
These arrays were reconstructed from the original seed-42 initialization in
the preceding investigation; an original initialization checkpoint was absent.
They are common stage-local estimator inputs, not four independent full-model
training trajectories. Action-stage rows select the longest history row per
example. Prefixes 8/32/128/256/512 and complete selected rows, where available,
reach 939 positions. Norm multipliers 0.5/1/1.5 yield **483 correlated
comparisons per arm**, not 483 independent experiments.

The independent NumPy FP64 oracle explicitly forms causal feature-kernel and
exact-softmax matrices. It does not call the production feature transform or
prefix scan. Synthetic CUDA gradients also use a separate PyTorch explicit
feature reference. Approximation error is raw attention output
`||Y_feature - Y_softmax||₂ / ||Y_softmax||₂` over valid positions, before output
projection/dropout. Its value depends on V geometry and the denominator;
0.20 remains the existing REVIEW trigger, not an AUC guarantee or a correctness
identity. Total variation, RMSE, Q/K norms and feature ESS supply context.

## Correctness and the precision failure

| Check | Outcome |
|---|---|
| Native prototype and original attention CPU regressions | PASS: final combined run 110 tests, zero failures; includes 22 new matched-factory/pilot-preparation cases and 88 existing cases |
| Original FP32 ORF128 CUDA verification | PASS: 24 kernel cases, full model and exact fresh reload |
| Original FP32 ORF512 CUDA verification | **FAIL**: independent-reference rule at length 65, chunk 16, Q/K/V multiplier 3 |
| Original FP32 antithetic CUDA arms | NOT VERIFIED: sequence stopped at preceding failure |
| Separate FP64 attention configuration, all four arms | PASS: 24 cases per arm, 96 total; no nonfinite outputs/gradients |
| Independent causal feature outputs/gradients | PASS in FP64 configuration; max output absolute difference 4.740996e-7, max gradient difference 9.353855e-9 |
| Full-model CPU/CUDA logits and parameter gradients | PASS: max absolute differences 3.576279e-7 and 1.788139e-7 |
| Future response/current-label and batch isolation | PASS: exact output invariants and zero forbidden gradient paths in tested cases |
| BOS/EOS/PAD and poisoned loss labels | PASS: exact valid-position invariants; six supervised labels retained in variable-length synthetic batch |
| Four fresh-process CUDA checkpoint reloads | PASS: exact logits, learned weights and restored CPU/CUDA RNG |
| Softmax approximation at 0.20 | **REVIEW**: residual stage/case failures remain |
| Antithetic optimization, Adam continuation and new AUC/F1 | **NOT RUN / NOT VERIFIED**; new user approval required |

Kernel lengths were 17/65/257/939, chunks 16/32 and norm multipliers
0.35/1/3. Output absolute tolerance remained 5e-5; gradient atol 5e-5,
rtol 1e-3. Causal and reload checks require exact equality. Independent
gradient reference checks cover lengths up to 65; longer kernels compare
CPU/CUDA outputs and gradients and exact future-prefix invariants.

The first failure was a genuine strict numerical acceptance failure, not an
OOM, causal-mask failure or softmax-approximation threshold failure. Its original
combined assertion did not serialize the individual GPU oracle deltas before
raising. We therefore **do not claim a measured GPU delta for that failed
assertion**. The original executed sources, logs and terminal failure remain.
CPU reconstruction of the deterministic inputs in the original runtime gave
FP32 output error **6.582114e-5**, exceeding 5e-5; the supported FP64 attention
option gave **4.629215e-7**. Both gradient comparisons passed. Query norms in
that stress case had median 23.7585 and maximum 29.0513, substantially above
the fresh real-input norms. Reconstruction provenance explicitly distinguishes
recreated inputs from an exception-time GPU tensor dump.

See [CPU numerical diagnosis](four_arm_cuda_comparison_20261024_v1/CPU_NUMERICAL_FAILURE_DIAGNOSIS.json)
and [precision configuration](four_arm_cuda_comparison_20261024_v1/fp64_v2/PRECISION_CONFIGURATION.md).
The follow-up used FP64 for feature generation/scan accumulation, returning
FP32 attention outputs; embeddings, learned weights and feed-forward components
remain FP32. All four arms used the same precision. Individual reference deltas
are now persisted before their assertion. The original 45-minute verification
campaign clock was inherited, not restarted. No historical attention default
or original experiment was changed. FP64 addresses this arithmetic residual;
it does not remove random-feature approximation error.

## Approximation on identical training inputs

Median relative-L2 errors across 14 cases per stage, original Q/K norms and
prefix `min(128, selected-row length)`. Parentheses count cases exceeding 0.20.

| Stage | ORF128 | ORF512 | Antithetic128 | Antithetic512 |
|---|---:|---:|---:|---:|
| ActionEncoder | 0.243430 (12/14) | 0.152904 (1/14) | 0.251865 (11/14) | 0.179270 (2/14) |
| SessionEncoder | 0.038142 (0/14) | 0.024416 (0/14) | 0.049660 (0/14) | 0.015897 (0/14) |
| CorrectPaddingEncoder | 0.156680 (1/14) | 0.066212 (0/14) | 0.075872 (0/14) | 0.049956 (0/14) |
| Decoder | 0.082800 (0/14) | 0.058666 (0/14) | 0.089765 (0/14) | 0.058187 (0/14) |

Increasing features improves medians here. Antithetic128 substantially improves
CPE against ORF128 but does not improve Action median. Antithetic512 improves
CPE/Session against ORF512; ORF512 improves Action more. Decoder is similar at
512. This single projection-pool seed does not establish statistical significance,
universal variance reduction or a full-training winner.

Norm sensitivity, same cases and ≤128 prefixes (Action / CPE medians):

| Q/K multiplier | ORF128 | ORF512 | Antithetic128 | Antithetic512 |
|---|---:|---:|---:|---:|
| 0.5 | .03907 / .02889 | .01960 / .01371 | .02754 / .00929 | .01403 / .00478 |
| 1.0 | .24343 / .15668 | .15290 / .06621 | .25186 / .07587 | .17927 / .04996 |
| 1.5 | .49987 / .34442 | .41434 / .16069 | .54202 / .18553 | .45331 / .12793 |

All four Action configurations exceed 0.20 in all 14 scaled-up cases. Norm
growth remains a real approximation risk even after arithmetic correctness
passes. Scaling is a diagnostic intervention, **not a correction applied to
the model**: scaling Q and K changes exact-softmax temperature as well.

On complete selected rows (up to 939), original-norm Action medians are
.23802/.15069/.23700/.17927 and CPE medians
.16835/.06613/.07443/.04995 in arm order. One antithetic512 Action case still
exceeds .20; the same draw is not a universal long-session fix. Full distributions,
lengths, Q/K p95, attention-weight total variation and feature ESS are in the
summary and raw comparison receipts. All 1,932 production-versus-independent
feature comparisons passed; maximum absolute difference 1.145447e-7. Thus
these softmax discrepancies are finite-estimator behavior on these inputs,
not an observed mask, scan or normalization defect.

The previous learned-alignment hypothesis remains plausible, not resolved by
fresh-input success. No antithetic model has learned in this study. Matched
learning with fixed audit projections is needed to see whether original-draw
error/concentration again separates from independent-draw error.

## Actual GPU and host measurements

Assigned shared device: `MIG-d2fcfd79-720e-5218-82c9-0a132be68920`, H100 80GB
HBM3 MIG 3g.40gb; CUDA capability 9.0. PyTorch
`2.4.0a0+f70bd71a48.nv24.06`, CUDA 12.5, Python 3.10.12, NumPy 1.24.4.
Deterministic algorithms/cuDNN enabled, benchmark/TF32 disabled,
matmul precision highest, seed 42, one CPU thread/taskset223/nice10.

| Arm | Peak live PyTorch MiB | Peak reserved MiB | Peak host RSS MiB | Longest measured real forward/backward s |
|---|---:|---:|---:|---:|
| ORF128 | 393.143 | 486 | 1607.801 | .51054 |
| ORF512 | 986.627 | 1220 | 1656.801 | .52145 |
| Antithetic128 | 393.143 | 486 | 1612.039 | .54271 |
| Antithetic512 | 986.627 | 1220 | 1656.496 | .56303 |

Profiles used four complete real TRAIN examples at padded costs
88/246/485/**2047**, batch 1, with two moment-sized parameter buffers allocated
to reserve Adam-state storage. No training interaction was truncated. The
20 proposed training examples cost 88–485; all methods use the same examples.
Oversize singletons are explicitly outside this bounded pilot only, not removed
from the original full-training dataset. The 2047-cost example yielded
250–276 supervised labels/s in forward/backward only. These are single shape
measurements, not optimizer throughput, repeated benchmarks or full-epoch
performance estimates. Adam workspace/step time and learned-state recovery
remain unmeasured for these new configurations.

The approved shared limits remained: 8 GiB PyTorch allocation/reservation/peaks,
10 GiB advisory total, 16 GiB free at admission, 6 GiB during execution,
4 GiB own host peak RSS, 2 GiB shared-host headroom, 30s step ceiling and
0.1s pacing. Cgroup RAM limit was 90 GiB; actual shared usage/headroom are
recorded at every boundary. Minimum sampled GPU free memory across the four
verification workers was 37.786 GiB. No CUDA/OOM/nonfinite/resource-gate failure
occurred in the FP64 sequence. The FP32 oracle failure remains separate.

PID namespace attribution remains **UNRESOLVED**, and MIG utilization queries
returned unsupported (NVML status 3). This is the explicitly authorized shared
policy, not an exclusive-ownership claim. Unknown activity is logged; no PID
is silently treated as self. Process-local protections, fresh resource
observations, memory caps and pacing remained active. Driver/context/other-job
memory cannot be attributed reliably from the sampled total-device counters;
do not subtract global VRAM and call it measured own overhead. LightSAKT2
remained stopped in observed local snapshots; its hold file was preserved.
No process was signalled and no other workload/configuration was changed.

## Prepared next experiment — approval pending

Frozen proposal:
`experiments/hitskt_favor_plus_v1/assist2017/matched_four_arm20_fp64_20261024_v1/prepared_plan.json`.
SHA256 `d1ff9b3d858d9a0ce49905aedcd66e8bc672a7254acc630cdb5c239774004c15`.
The [approval template](four_arm_cuda_comparison_20261024_v1/MATCHED_TRAINING_RECORD_TEMPLATE.json)
is deliberately false/unapproved. Verification permission cannot authorize
optimizer execution. A new actual user response is required.

- Four sequential fresh arms, **20 updates each / 80 maximum in total**;
  seed-42 learned weights/post-factory RNG, identical data order, fixed pool
  policy and feature precision. Do not resume historical trained weights or
  warm-start from verification/profile RNG.
- Original Adam: LR5e-5, betas(.9,.999), eps1e-8, weight decay0, no clipping;
  unchanged non-attention architecture, preprocessing and retrospective
  chronological within-student protocol. No online-response-availability or
  student-disjoint claim is added.
- Batch1, padded cost≤2048, fixed 20 whole TRAIN examples; frozen 32 validation
  sessions / 3157 labels (1063 positive,2094 negative), evaluated only at
  initialization/5/20. Report sampled BCE/AUC/accuracy/precision/recall/F1 as
  exploratory, not final validation/test performance. No test access.
- Six fixed TRAIN diagnostics at 0/5/20; saved Q/K/V for native and predeclared
  independent audit projections17/101, never installed in the model. Track
  original-draw versus audit error, feature ESS, Q/K p95 growth and per-stage
  output error. Audit draws are not extra training seeds or tuning candidates.
- Atomic model/Adam/RNG/projection/data-order/source/config checkpoints every
  five updates, component hashes and exact fresh-process update-20 recovery
  including the next dropout/loss/gradient computation **without another step**.
- Expected working live allocation approximately **1–1.5 GiB**, conditional on
  measured moment-reserved profiles; actual optimizer workspace is unmeasured.
  Retain the existing 8 GiB hard cap and all shared pressure guards.
- Planning allowance **10–15 minutes total**, not a measured training ETA;
  enforce a stricter **15-minute total campaign ceiling**. The measured maximum
  .563s forward/backward does not include resource telemetry, Adam, validation,
  diagnostics, serialization or fresh-process recovery. Stop rather than
  increase any resource/time budget or silently retry an incomplete step.

The original .20 REVIEW flag remains. Pre-existing severe stops are separately
retained: nonfinite states, causal/reload/resource failure, BCE>20,
gradient L2>100, independent kernel absolute error>5e-5, any diagnostic
relative-L2>1.5, stage median>0.75, stage median feature ESS<1.25 or Q/K p95
growth>2×initial. These are containment criteria, not a replacement pass
threshold. Any candidate above .20 must remain REVIEW even if training is finite.

Future commands after explicit approval and a fresh source/plan-bound record
are below. **These training commands have not been run.** Use the existing
MIG environment plus deterministic CUBLAS/Python settings from the verification
sequence; use a new actual approval record, not the false template.

```bash
python scripts/pilot_favor_plus_matched.py train --arm orf128 --record APPROVED_RECORD.json --execute-optimizer
python scripts/pilot_favor_plus_matched.py recover --arm orf128 --record APPROVED_RECORD.json --execute-optimizer
# Repeat train then fresh-process recover for orf512, anti128, anti512 in order.
```

CPU tests establish runner approval refusal before CUDA/optimizer construction
and exact zero-step model/projection/RNG/probe serialization for all arms.
They do **not** establish the unrun optimizer path or trained-checkpoint CUDA
recovery. Record any future failure, stop the sequence and preserve its evidence.

## Recommendation and preservation

**No full-epoch configuration is selected yet.** ORF512 and antithetic512 are
both credible bounded-study candidates; their stage trade-off makes the matched
learning comparison more useful than assuming antithetic is uniformly better.
Keep 128-feature arms as controls. Require retained approximation behavior
after learning, finite Adam/model states, exact checkpoint continuation and
resource safety before a separately approved longer study. Twenty updates
cannot establish convergence; a later full-data validation-selected comparison
and untouched final test evaluation remain necessary for reportable research
performance. No significance or model-superiority claim is supported here.

Original full-validation AUC7 **0.703466903**, F1 at that checkpoint
**0.508813934**, and best observed F1 epoch6 **0.532809339** remain historical
FAVOR+ results. They are not antithetic results and must not be compared as
equivalent to the proposed sampled validation metrics.

Historical epoch7 SHA256:
`07b944e71683c415cc5c00fad42648d22cb5b8fa790bc39c3c5d24b50c781781`;
epoch8 `e0fe6216e05abb2a2bd1de906d6a3f6e0319e2538ad994ad1363c446311f344b`;
old diagnostic50 `93ebe9d2a29be9a6e4deb278916513339de61c9ac8f72a454a31111829440a73`.
Original ELU+1/FAVOR+ sources, all 11 processed identity files, historical
checkpoints/results/audits, LightSAKT2 and unfinished DKT/EdNet are preserved.
Checkpoint binaries remain local, with hashes; publishing metadata is not a
claim of an external durable binary backup. New numerical/process receipts are
archived with entry hashes; frozen input arrays remain linked to their prior
manifest. See the final preservation receipt and archive manifest in this
report's evidence directory.
