# HiTSKT FAVOR+ — post-epoch8 approximation investigation

24 October 2026. Branch `research/hitskt-favor-plus-v1`; starting/source-evidence
commit `24206be171bbf3d973ad388e484d1828ecd32492`.

**Decision: retain the original eight-epoch results; REVIEW before additional
full training. No original implementation bug was found in the tested regimes.**
The strongest evidence points to projection-specific learned alignment, finite
random-feature error and value-vector geometry. A separate antithetic positive-ORF
prototype is implemented and CPU-tested; it is a partial variance-reduction
candidate, **not a validated complete correction**. The original model,128-feature
configuration,0.20 threshold,checkpoints,dataset and published metrics are unchanged.
No optimizer update,CUDA execution,new validation inference or test evaluation
was performed during this investigation.

## Scope and reproducibility

The [predeclared design](postepoch8_approximation_20261024_v1/PREDECLARED_DESIGN.json)
reuses14 TRAIN examples selected before this investigation for history/session-
length coverage. It uses reconstructed seed42 initialization plus protected
checkpoints7 and8, all four attention stages, and identical frozen projected
Q/K/V,masks and values within each comparison. Stage-local attention is compared
before output projection/dropout, not against a differently trained model.
The original epoch8 six-case≤32 warnings remain Action0.262562 and CPE0.503431.
New fourteen-case/longer-prefix statistics have a different stated scope and do
not replace those historical values or redefine the review threshold.

There are168 captured stage/example/state records and4,032 feature-grid
comparisons. Original-draw prefix checks cover1,8,32,128,256,512 and each complete
selected row when available, up to939 positions includingEOS. Action-stage
capture selects the longest historical row per example; it is representative
sampling, not an assertion about every historical row or the whole dataset. The projection
sweep uses total feature counts128/256/512, four fixed independent seeds
17/101/202/303 and common nested512-row pools across counts. Sampling seeds are
**diagnostic projection draws**, not additional training seeds. Each stage/arm/
count has112 correlated case/draw contrasts (14 examples×2 checkpoints×4draws),
not112 independent research runs or a basis for significance claims.

The main feature sweep uses prefixes≤128. A saved128 draw is reported separately
from fresh independent128 draws. Additional prospective annexes test the initial
versus trained draw gap, the prototype's actual default stage seeds42/43/44/45,
long extreme rows, and full-model exact-operator sensitivity. Each annex was
recorded before execution and is distinguished from the original design.
No seed/count was selected using validation or test AUC.

Workers used an isolated **PyTorch2.4.1+cpu** environment,CUDA hidden,one CPU thread,
taskset223,nice10,4GiB peak-RSS/2GiB shared-host-headroom gates and1200s per-action
ceiling. The CPU guard checks both build metadata and absence of libtorch_cuda.
The shared GPU training runtime was not modified. Initial environment setup
encountered missing ensurepip and inherited shared-library resolution; process-
local pip/LD_LIBRARY_PATH fixes were applied before diagnostic execution. The
setup failures/logs remain preserved. CPU capture and reference comparisons are
not claimed to be a rerun of historical CUDA validation metrics.

Initialization was reconstructed from the original seed/configuration; an
original initialization checkpoint was not available. On the original six
examples and≤32 prefixes, its and the trained states' diagnostic errors agree
with stored GPU diagnostics to a maximum absolute relative-error difference
**1.019350e-7**. Protected7/8 hashes and all saved ORF matrices were verified.
This is a continuity check, not a new tolerance or a full prediction replay.

## Correctness checks and original implementation review

|Check|Status|Evidence|
|---|---|---|
|Production feature-kernel computation|PASS|Maximum absolute difference3.113340e-6 over original-draw prefixes, below unchanged5e-5|
|FP64 production versus independent feature reference|PASS|Maximum4.44e-15 on frozen trained prefixes; large softmax error remains|
|Projection identity and sampling|PASS|Saved matrices regenerate bit-exactly for stage seeds42–45; Gaussian/chi-radii and block-orthogonality regression tests pass|
|Scaling and normalization|PASS in tested regimes|d^(-1/4) on each Q/K; positive features; query constants cancel; per-feature prefix gauges are compensated|
|Causal masks/response shifting/PAD/EOS|PASS tested invariants|Original FAVOR+ and new prototype future/own-label,batch,padding and gradient regressions pass|
|Original model/optimizer/data/results preservation|PASS|37 original source and11 processed checksums,7/8 checkpoints,180 previously published nonmemory files unchanged|
|Softmax approximation versus0.20|REVIEW|Action/CPE/Decoder remain above threshold; not relabeled PASS|
|New candidate CUDA/training/AUC|NOT VERIFIED|No such execution or metric is claimed|

The independent reference explicitly builds causal FP64 feature/exact-softmax
matrices; it does not call the production feature transform or prefix scan.
A separate direct logsumexp-pair oracle checks numerical gauges. FP64 removes
production arithmetic residuals without removing the approximation discrepancy.
This rules out float32 rounding as its principal cause on these inputs.

All128 projection rows are not mutually orthogonal: head dimension64 requires
independent orthogonal blocks. Independent chi64 radii retain Gaussian row
marginals; they are not replaced by fixed radii. Saved projections are buffers,
not learned parameters, and did not redraw or drift during training.
The prefix maximum/sum/weighted-value gauges cancel algebraically and use no
future maximum in production. New tests verify derivatives of the feature-kernel
operator, not equality of its derivatives to exact softmax derivatives.
No new arithmetic,projection,mask or normalization fix to the original model
was justified.

## Main finding: alignment specific to the fixed projection

CorrectPaddingEncoder on14 cases,≤128 positions; errors are median raw attention
relativeL2. Pair ESS uses the fixed sparse diagnostic pair grid, so its numerical
values should not be interchanged with the earlier all-pair≤32 summaries.

|State|Saved128 error|Independent128 error|Saved pair ESS|Independent pair ESS|Mean max saved feature mass|
|---|---:|---:|---:|---:|---:|
| initial | 0.142040 | 0.114252 | 9.401197 | 8.267063 | 0.295932 |
| 7 | 0.545165 | 0.192859 | 1.974317 | 6.646078 | 0.643301 |
| 8 | 0.566154 | 0.183067 | 1.844643 | 6.186328 | 0.676758 |

At initialization, saved128 error0.142040 versus fresh0.114252 is a modest gap;
saved feature ESS is actually higher. At epoch8, the gap is0.566154 versus0.183067,
and saved pair ESS1.844643 versus fresh6.186328. Mean maximum mass of one feature
rises0.295932→0.676758. Mean maximum pair/projection cosine rises0.309501→0.428070.
The projection matrices are identical, so the representations have moved toward
particular directions of the frozen draw.

**Interpretation:** learned alignment/co-adaptation to the fixed features is the
best-supported explanation for much of CPE's excess error; it is not simply a
projection that was already catastrophically unlucky before learning. This is
controlled frozen-input evidence, not a randomized causal training intervention.
Four fresh draws and two related trained checkpoints cannot prove a universal
optimizer mechanism or distinguish every source of adaptive bias.

For fixed inputs independent of the draw, positive Gaussian features estimate
the softmax kernel in expectation. Training makes Q/K functions of the fixed
projection, so that independent-input argument cannot simply be applied to the
observed learned Q/K and their original draw. Changing the projection after
training also changes the actual finite-feature model; fresh-draw diagnostics
are not evidence of improved AUC or permission to replace saved buffers.

## Norms, concentration and value geometry

CPE median query norm rises4.5633→5.2492 and key norm4.4818→4.6599 from reconstructed
initialization to8. The median scaled pair-sum squared norm rises5.2017→6.1827;
pair cosine rises0.00842→0.06897. This is moderate norm/alignment growth, not
observed numerical norm explosion. The positive-feature variance is sensitive
to the **sum** of scaled Q/K, not just their separate lengths. For independent
Gaussian features and fixed inputs, relative kernel variance is
`(exp(||q/d^.25 + k/d^.25||²)-1)/m`; this is explanatory, **not an ORF/adaptively-
trained error bound**. Positive hyperbolic features and their variance rationale
come from [Performer §2.3/Lemmas1–2](https://arxiv.org/html/2009.14794).

Low random-feature ESS is not the same as sharply concentrated exact attention.
Across≤32 case prefixes, mean exact effective-key counts remain15.27 initially
and14.30 at8. Meanwhile mean attention total variation rises0.0953→0.2917.
Thus the discrepancy is present in attention weights and is not solely a small
output denominator. Exact attention is still relatively diffuse on average;
this does not exclude concentrated individual heads/queries or other inputs.

The following algebraic swaps hold masks/projection fixed and compare the same
frozen arrays (≤128,14cases,pooled7/8). They identify sensitivity, not additive
causal contributions of individual learned parameters/upstream modules.

|Stage|Initial|TrainedQ only|TrainedK only|TrainedV only|TrainedQK|TrainedQKV|
|---|---:|---:|---:|---:|---:|---:|
| Action | 0.207775 | 0.211562 | 0.265283 | 0.264246 | 0.280754 | 0.319873 |
| Session | 0.065603 | 0.041216 | 0.059528 | 0.078264 | 0.065489 | 0.111407 |
| CorrectPadding | 0.142040 | 0.166231 | 0.345507 | 0.263529 | 0.440228 | 0.561886 |
| Decoder | 0.089584 | 0.093925 | 0.285731 | 0.260060 | 0.302928 | 0.545084 |

Key changes explain more of CPE's increase than query changes alone; learned
values further amplify its output-relative error. Decoder shows a particularly
large value contribution. A common value-offset control leaves absolute error
and attention-weight error unchanged to about3.2e-15 while changing output-relative
error. The0.20 threshold therefore measures a particular diagnostic quantity,
not a universal AUC bound or implementation-correctness theorem. It remains the
original REVIEW trigger and has not been relaxed or replaced by another metric.

Diagnostic Q/K scaling by0.5 can reduce error, but changes the exact softmax
logit temperature by0.25. It is **not** a target-preserving bug fix and was not
applied to the model. First-valid-key translation preserves exact softmax and
causality, but was not robust empirically: saved-draw CPE improved22/28cases,
while Action improved only12/28 and Decoder4/28; fresh-draw comparisons often
worsened. It is rejected as a general correction rather than selected because
it helps a particular saved projection.

## Feature-count and candidate comparisons

Median errors on frozen trained representations,≤128; independent diagnostic
pools, total feature count held explicit. All columns use identical Q/K/V and
exact targets (key translation leaves exact normalized weights unchanged).
Shading in the linked plot is descriptive10–90% across case/draw contrasts,
not confidence intervals. More features reduce aggregate error, not necessarily
every individual error monotonically.

|Stage|Total features|ORF|Antithetic ORF|First-key-centered ORF|
|---|---:|---:|---:|---:|
| Action | 128 | 0.264874 | 0.249346 | 0.329027 |
| Action | 256 | 0.218442 | 0.208797 | 0.273117 |
| Action | 512 | 0.186169 | 0.175246 | 0.236719 |
| Session | 128 | 0.074183 | 0.058387 | 0.054666 |
| Session | 256 | 0.056724 | 0.051958 | 0.040939 |
| Session | 512 | 0.047049 | 0.047622 | 0.034054 |
| CorrectPadding | 128 | 0.186981 | 0.166288 | 0.324824 |
| CorrectPadding | 256 | 0.168436 | 0.159554 | 0.242379 |
| CorrectPadding | 512 | 0.129161 | 0.131909 | 0.170137 |
| Decoder | 128 | 0.428573 | 0.383202 | 0.653501 |
| Decoder | 256 | 0.389748 | 0.351649 | 0.562063 |
| Decoder | 512 | 0.336250 | 0.316268 | 0.488962 |

[Feature comparison figure](postepoch8_approximation_20261024_v1/FROZEN_FEATURE_COMPARISON.png).
[Projection alignment figure](postepoch8_approximation_20261024_v1/PROJECTION_ALIGNMENT_EVIDENCE.png).

At512 independent ORF features, Action/CPE medians fall to0.186169/0.129161, but
Decoder remains0.336250. Antithetic512 yields0.175246/0.131909/0.316268 for
Action/CPE/Decoder. These are aggregate diagnostic results, not evidence that
all cases pass or the actual default training projection will behave identically.
The high saved CPE128 versus fresh128 gap also shows feature count alone is not
a complete explanation or a guaranteed cure for learned alignment.

## Separate antithetic experimental configuration

Implemented in [favor_plus_antithetic_experimental.py](../../ktbench/models/favor_plus_antithetic_experimental.py):
`hitskt_favor_plus_antithetic_experimental_v1`, explicit checkpoint **format3**,
original nonattention architecture and inherited validated causal scan.
512 total positive features mean256 Gaussian ORF base directions and their
negatives, **not512 independent directions**. The declared nested512-row base
pool exactly matches the controlled diagnostic draw construction. Both signs
retain positive exponential features; the expected softmax-kernel target is
unchanged. The finite estimator,correlations and gradients differ as intended. In scaled
coordinates, its feature map is `exp(-||x||²/2)/sqrt(M) × [exp(Wx),exp(-Wx)]`.
The Gaussian moment identity gives expected kernel `exp(x·y)`; it does not make
normalized attention, a ratio estimator, exactly unbiased.

Configuration:[ANTITHETIC_EXPERIMENTAL_CONFIG.json](postepoch8_approximation_20261024_v1/ANTITHETIC_EXPERIMENTAL_CONFIG.json).
Projection RNG,dropout/learned initialization,causality,first-order gradients,
PAD/EOS/full-model propagation and checkpoint isolation have CPU tests. No old
model or default registry/pipeline was switched. The original training CLI will
not silently train this variant. A complete training/evaluation runner with
optimizer/RNG recovery and GPU admission for this new variant is **not yet wired**.
No candidate validation AUC,accuracy or F1 exists.

The actual default stage seeds42–45 were tested separately rather than inferred
from average fresh diagnostic draws:

|Stage|Default antithetic128|256|512|
|---|---:|---:|---:|
| Action | 0.340397 | 0.298661 | 0.255859 |
| Session | 0.063250 | 0.056429 | 0.038731 |
| CorrectPadding | 0.269297 | 0.224023 | 0.161238 |
| Decoder | 0.425075 | 0.467255 | 0.354527 |

**Even default512 does not solve all0.20 reviews.** On the preselected longest
939-position target, CPE error is0.243086 at7 and0.223764 at8; Decoder0.485985/
0.487329. The high-history extreme's640-position Action error is0.292957/0.302919.
All16 extreme-row production/reference comparisons passed. These long cases
prevent presenting a short-prefix median as a universal fix.

On one frozen128-position CPE input, no-grad single-core CPU median forward time
was13.224/17.210/31.043ms for128/256/512 total features (one warmup,three measured
calls). One weighted state grows0.125/0.250/0.500MiB. This does not measure complete
model peak memory,backward cost or GPU speed. Feature-state/log-feature storage
scales with feature count; unchanged feed-forward/embedding costs do not.
The earlier128-feature8GiB GPU admission is not evidence that original full batch
shapes of this512-feature prototype fit safely. No GPU performance is invented.

## Final-output sensitivity and existing predictive results

A separate frozen-weight CPU annex temporarily substitutes exact causal attention
in memory on the original full TRAIN examples, restores original states exactly,
and never serializes a counterfactual model as a checkpoint. This differs from
stage-local frozen-QKV comparisons because downstream representations can change.
The exact operator uses masked FP64 scores with original learned Q/K/V/output
projections. This is **not a separately trained exact-softmax baseline**.

|Counterfactual|Median case mean absolute probability change|Maximum probability change|Changed0.5 decisions /presentations|
|---|---:|---:|---:|
|CPE exact only|0.007178|0.065989|109/5170|
|All four stages exact|0.041082|0.302781|385/5170|

5,170 is repeated target presentations across the same14cases/two checkpoints,
not independent validation examples. Changes can be material, but no claim is
made about which predictions are better. No candidate or exact-counterfactual
validation/test scoring was used to choose features.

The reviewed eight-epoch full-validation measurements remain unchanged:

|Epoch|Train BCE|Validation AUC|Accuracy|Precision|Recall|F1|Validation BCE|
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.638942 | 0.649592 | 0.658553 | 0.605563 | 0.279085 | 0.382081 | 0.627900 |
| 2 | 0.605684 | 0.674090 | 0.663633 | 0.570184 | 0.449794 | 0.502884 | 0.621430 |
| 3 | 0.596876 | 0.686732 | 0.677941 | 0.637717 | 0.343960 | 0.446887 | 0.607752 |
| 4 | 0.590313 | 0.692310 | 0.679253 | 0.614180 | 0.408877 | 0.490929 | 0.606542 |
| 5 | 0.586927 | 0.698317 | 0.683156 | 0.621306 | 0.415749 | 0.498155 | 0.601908 |
| 6 | 0.582839 | 0.700271 | 0.678704 | 0.592022 | 0.484364 | 0.532809 | 0.604904 |
| 7 | 0.580976 | 0.703467 | 0.685254 | 0.620946 | 0.430985 | 0.508814 | 0.599038 |
| 8 | 0.579697 | 0.700440 | 0.681828 | 0.668435 | 0.315169 | 0.428363 | 0.600564 |

Best AUC remains0.703467 at7, with F10.508814; highest observed F1 remains0.532809
at6. Stored predictions previously passed bit-identical fresh CUDA replay and
independent metrics. Their original evidence is preserved, not recomputed or
replaced by this investigation. See [complete epoch report](ASSIST2017_EPOCH_TRAINING_RESULTS.md)
and [original curves](assist_epochs_execution_20261024_v1/PERFORMANCE_CURVES.png).

AUC improved through7 then declined0.003027 at8, while training BCE continued
down and validation BCE worsened. The previous score-shift analysis explains
recall/F1 movement at the unchanged0.5 threshold; neither label/order nor metric
mismatch was found. One nonimproving epoch is not patience3 or proof of convergence.
Further training **might** improve validation, but eight correlated epochs cannot
establish that it is likely or that it will reduce approximation error. More
optimization is not supported as a correction for the observed fidelity problem.

## Tests, resource use, failures and preservation

**125 distinct final CPU cases PASS:47 newly added,78 existing regressions.**
These include independent feature-output/gradient references,finite-difference
checks,Gaussian marginals/antithetic scaling,seed/projection/RNG behavior,
future/own-label invariants,variable sessions,empty/holed masks,PAD/EOS,
full-model backward/checkpoint and variant rejection. The full historical229
suite was not rerun; counts are deduplicated across the three final XML files.
The phase-one35 cases overlap later frozen-control checks and are not added twice.

One initial experimental-checkpoint test failed: prototype format1 collided
with the legacy loader's format. It was corrected to format3; initial source,
failed log/XML and correction rationale are preserved. Final prototype/regression
suite88 passes. This was a prototype metadata defect, not an original model bug;
no historical checkpoint was rewritten. Environment bootstrap/library-resolution
issues were also retained rather than hidden. No final failure is unresolved
in the tested regimes; broader behavior and GPU execution remain unverified.

Peak measured diagnostic RSS **1.235382GiB**, within4GiB; largest diagnostic action was the
feature grid at53.063s, within1200s. Native CPU-only guards stayed active. No GPU
availability polling,CUDA context,optimizer or other workload modification was
needed. Original7/8 checkpoint hashes,37 source and11 processed identities and
180 prior published nonmemory artifacts match. Protected Light hold and unrelated
working-tree README/SAKT log changes are unchanged.

Protected best7:
`07b944e71683c415cc5c00fad42648d22cb5b8fa790bc39c3c5d24b50c781781`.
Protected latest8:
`e0fe6216e05abb2a2bd1de906d6a3f6e0319e2538ad994ad1363c446311f344b`.

Full168 frozen-array files remain local with hashes. A compact72-record≤32prefix
bundle and all diagnostic JSON results are published with manifests; it does not
pretend to contain every full-length Q/K/V array. Reproduction of full captures
needs the protected local checkpoints/dataset; no public checkpoint backup is
claimed. All current raw outputs remain untouched in the separate namespace.

## Recommended next experiment

**GO** for the separately configured CPU prototype and qualified reporting of
existing validation results. **REVIEW / NO-GO** for automatically extending full
training, declaring an all-stage0.20 pass, or replacing old projections/checkpoints.
No evidence requires discarding the correctly measured original results or
retraining them merely to erase a warning.

The next useful learning experiment is a **separately approved, small matched
pilot**, not another full epoch schedule: original128 versus antithetic512,
seed42,same predeclared ASSIST TRAIN examples/optimizer/nonattention architecture,
checks at initialization/5/20updates, independently drawn audit projections,
fixed short and long training prefixes,unchanged0.20 review/severe gates. First
perform bounded GPU memory/gradient admission for512; preserve shared caps and
headroom and stop on failure. This would test whether frozen gains survive
learning, rather than presume it from a static comparison. It is a proposal,
not execution authority or a promise of better AUC.

If fixed-draw concentration recurs, a future separately versioned,checkpointed
redraw-policy study could test the co-adaptation hypothesis; no redraw policy
was implemented into the original experiment. Do not choose an attractive
post-training projection using validation/test performance. Norm/temperature
changes require a distinct architecture decision; first-key centering is not
recommended from these negative controls. Keep best7 for reporting and latest8
for any separately decided exact continuation; never silently change feature
maps while resuming either optimizer.

Evaluation remains chronological within-student retrospective teacher forcing
with the previous response-availability qualification, not student-disjoint or
unconditionally online evaluation. Final test performance,multi-seed uncertainty
and matched-baseline superiority remain unverified. Freeze the research decision
and validation-selection protocol before an eventual untouched-test evaluation.

[Structured summary](postepoch8_approximation_20261024_v1/INVESTIGATION_SUMMARY.json),
[main design](postepoch8_approximation_20261024_v1/PREDECLARED_DESIGN.json),
[raw result archive](postepoch8_approximation_20261024_v1/DIAGNOSTIC_RESULTS.tar.gz),
[compact numerical audit](postepoch8_approximation_20261024_v1/AUDIT_PREFIXES_32.npz),
[CPU-only environment](postepoch8_approximation_20261024_v1/CPU_ENVIRONMENT.json).
