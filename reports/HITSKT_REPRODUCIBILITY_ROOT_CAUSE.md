# HiTSKT reproducibility discrepancy investigation

Date: 2026-10-18 UTC. Base audit: `5625478ab0be344b145d2813939a71a0f6e49924`.

**Outcome: all three historical-to-CPU metric gates remain FAIL at the original
absolute tolerance of 1e-6.** AUC passes that tolerance for every dataset.
ASSIST2017's BCE discrepancy survives both high-precision metric accumulation
and a full float64 model-forward diagnostic. The exact originating CUDA
operation/settings remain **NOT VERIFIED**. No tolerance was changed, result
overwritten, architecture changed, training started or external GPU job interrupted.

This follows [the immutable first audit](HITSKT_INDEPENDENT_CORRECTNESS_AUDIT.md).
New evidence lives in [reproducibility_diagnostics](reproducibility_diagnostics/).
The original all-target CPU predictions are hash-verified inputs, not newly
invented CUDA predictions. Fresh inference in this follow-up consists of the
full ASSIST precision diagnostic and bounded Junyi/EdNet samples.

## Historical results versus independent full CPU replay

All numbers below are proportions/losses, not percentages. BCE has two CPU
forms: independent stable float64 logit BCE, and the original FP32 batch-mean
loss weighted by valid-target count. The latter matches the production
definition in [scripts/train_hitskt.py](../scripts/train_hitskt.py), lines43–95.
PASS means absolute difference <=1e-6; it does not mean bitwise equality.
None of the stored-versus-CPU metrics is exactly equal.

### ASSIST2017 — CPU FAIL; CUDA NOT VERIFIED

All **112,252** test targets; selected epoch **40**. Checkpoint `experiments/hitskt/assist2017/full/best_model.pt`, SHA-256
`085c8d9ae32f4e60892dfb3e4eed93d90a584696d0914456c135dcab7b890164`.

| Metric | Stored CUDA | Full CPU | Absolute difference | Gate |
| --- | --- | --- | --- | --- |
| AUC | 0.7096240159197755 | 0.709623637214297 | 3.787054785320265e-07 | PASS |
| Accuracy | 0.6744200548765278 | 0.6743755122403164 | 4.454263621134213e-05 | FAIL |
| Precision | 0.5866082262338113 | 0.5865484621861226 | 5.976404768870403e-05 | FAIL |
| Recall | 0.5440494235450829 | 0.5439805245510082 | 6.889899407469535e-05 | FAIL |
| F1 | 0.5645278522490319 | 0.5644630856489204 | 6.476660011145441e-05 | FAIL |
| MSE | 0.2098452017338206 | 0.2098443178262471 | 8.839075735511681e-07 | PASS |
| BCE (stable float64 logits) | 0.6087257417328864 | 0.6087236095474302 | 2.13218545619398e-06 | FAIL |
| BCE (historical FP32 batch definition) | 0.6087257417328864 | 0.6087236110378144 | 2.130695072044553e-06 | FAIL |

Evidence: [assist2017_diagnostic_manifest.json](reproducibility_diagnostics/assist2017_diagnostic_manifest.json); [original complete CPU manifest](independent_correctness_audit/assist2017_evaluation_manifest.json).

### Junyi — CPU FAIL; CUDA NOT VERIFIED

All **2,456,402** test targets; selected epoch **32**. Checkpoint `experiments/hitskt/junyi/full/best_model.pt`, SHA-256
`2308c5bf0c850da1e55daf802a0853a30fac45eeed071c7162c68fc17bfa3bdd`.

| Metric | Stored CUDA | Full CPU | Absolute difference | Gate |
| --- | --- | --- | --- | --- |
| AUC | 0.7978262055191263 | 0.7978264739233538 | 2.684042275102883e-07 | PASS |
| Accuracy | 0.75555059798844 | 0.7555583328787389 | 7.734890298816133e-06 | FAIL |
| Precision | 0.785045267643441 | 0.7850550531960312 | 9.785552590235724e-06 | FAIL |
| Recall | 0.8815436621676446 | 0.8815400666253979 | 3.595542246714878e-06 | FAIL |
| F1 | 0.83050075255717 | 0.8305046326706952 | 3.88011352514539e-06 | FAIL |
| MSE | 0.1647862988059791 | 0.1647862100342857 | 8.877169338505908e-08 | PASS |
| BCE (stable float64 logits) | 0.4975778811287057 | 0.4975776789342685 | 2.021944371377593e-07 | PASS |
| BCE (historical FP32 batch definition) | 0.4975778811287057 | 0.4975776799704835 | 2.011582221905783e-07 | PASS |

Evidence: [junyi_diagnostic_manifest.json](reproducibility_diagnostics/junyi_diagnostic_manifest.json); [original complete CPU manifest](independent_correctness_audit/junyi_evaluation_guarded_manifest.json).

### EdNet-KT1 — CPU FAIL; CUDA NOT VERIFIED

All **13,429,870** test targets; selected epoch **30**. Checkpoint `experiments/hitskt/ednet_kt1/full/best_model.pt`, SHA-256
`58d4b1f393d0fa8ec372ba8bcc00cc8b7168d37817d4badfd26ee42abffd03c8`.

| Metric | Stored CUDA | Full CPU | Absolute difference | Gate |
| --- | --- | --- | --- | --- |
| AUC | 0.7693648364413692 | 0.7693647361509419 | 1.002904272962013e-07 | PASS |
| Accuracy | 0.7388606144363273 | 0.7388628482628647 | 2.23382653741222e-06 | FAIL |
| Precision | 0.7639387403579302 | 0.7639391533725004 | 4.130145702108123e-07 | PASS |
| Recall | 0.8946143710041425 | 0.8946181815727218 | 3.810568579232232e-06 | FAIL |
| F1 | 0.8241286589004618 | 0.8241305161089085 | 1.857208446698344e-06 | FAIL |
| MSE | 0.173712574181304 | 0.1737125946897854 | 2.050848138535066e-08 | PASS |
| BCE (stable float64 logits) | 0.5185181172959707 | 0.5185181658640189 | 4.856804813879734e-08 | PASS |
| BCE (historical FP32 batch definition) | 0.5185181172959707 | 0.5185181668280653 | 4.953209453706364e-08 | PASS |

Evidence: [ednet_kt1_diagnostic_manifest.json](reproducibility_diagnostics/ednet_kt1_diagnostic_manifest.json); [original complete CPU manifest](independent_correctness_audit/ednet_kt1_evaluation_guarded_manifest.json).

## Exactly what can be recovered about classification changes

The original CUDA runs retained aggregate metrics, not per-target logits,
probabilities or classifications. Searching the experiment/inspection evidence
located CPU audit vectors only. Therefore the **exact original CUDA-versus-CPU
number of changed classifications, and maximum logit/probability difference,
are NOT VERIFIED for all three datasets**. Aggregate accuracy differences are
net changes in correct predictions, not total flips. Matching aggregates in a
future CUDA replay would still not prove its vectors equal the unsaved original
vectors; report fresh-CUDA-versus-CPU differences with that scope.

Integer confusion matrices can be recovered consistently from the stored
accuracy/precision/recall/F1 and verified label population. Equivalent formulas
are checked within eight representation ULPs solely to reconstruct integer
counts; this does not alter the 1e-6 acceptance criterion. Counts are TP/FP/TN/FN.

| Dataset | Stored confusion (inferred) | CPU confusion | Net correct delta | Confusion-only possible flips | Parity |
| --- | --- | --- | --- | --- | --- |
| ASSIST2017 | 23689/16694/52016/19853 | 23686/16696/52014/19856 | -5 | 5–73,099 | odd |
| Junyi | 1471061/402794/384875/197672 | 1471055/402769/384900/197678 | 19 | 31–1,165,125 | odd |
| EdNet-KT1 | 8217016/2539103/1705786/967965 | 8217051/2539108/1705781/967930 | 30 | 40–5,347,462 | even |

These are sharp bounds from confusion counts alone, not an assertion that
the maximum occurred. At least **5 / 31 / 40** classifications differ, respectively.
The independent exhaustive small-population oracle verifies the bounds and
parity. See [cached_metrics_diagnostics.json](reproducibility_diagnostics/cached_metrics_diagnostics.json)
and [the diagnostic tests](../tests/test_reproducibility_diagnostics.py).

## Probability rounding on identical CPU logits

Recomputing sigmoid in float64 for every cached float32 logit isolates sigmoid
and metric arithmetic from model-forward differences. The historical threshold
remains float32 probability >=0.5. It was not replaced with a logit threshold.

| Dataset | Targets | Max absolute probability difference | Exact classification changes | Cached p=0.5 | Within 1e-5 of 0.5 |
| --- | --- | --- | --- | --- | --- |
| ASSIST2017 | 112,252 | 8.510043714160531e-08 | 0 | 0 | 3 |
| Junyi | 2,456,402 | 8.828051345499688e-08 | 0 | 0 | 49 |
| EdNet-KT1 | 13,429,870 | 8.867596146266976e-08 | 4 | 4 | 286 |

EdNet's four cached p=0.5 cases have negative, nonzero logits. Float64
sigmoid moves all four below0.5. This proves a quantization effect in this
specific counterfactual; it does not explain the exact original CUDA/CPU flips.
ASSIST and Junyi have no classification changes from sigmoid precision alone.

## Model arithmetic and determinism

Fresh processes reloaded the unchanged saved float32 weights. A separate model
cast those weights to float64; evaluation used original batches, .eval(), no
gradients and no autocast. The source-generated sinusoidal position tables are
still computed in float32 before casting, as prescribed by the unchanged
implementation, so this is a higher-precision network-forward diagnostic with
the same positional templates, not a new full-precision model specification.

ASSIST covers every test target. Junyi/EdNet use six preselected original batches
each (including the batch containing the globally smallest cached absolute
logit). They are diagnostic samples, not random samples or full reproduced test
results; none was skipped. Do not extrapolate sampled maxima or flip counts.

| Dataset | Evaluated targets / scope | Max absolute logit difference | Max absolute probability difference | Exact CPU32→CPU64 classification changes |
| --- | --- | --- | --- | --- |
| ASSIST2017 | 112,252 / ALL | 1.487006064060381e-06 | 3.105441224171557e-07 | 0 |
| Junyi | 4,072 / 6 sampled batches | 1.18927824299675e-06 | 1.866159566876746e-07 | 0 |
| EdNet-KT1 | 2,666 / 6 sampled batches | 1.350615904449626e-06 | 2.307068923679267e-07 | 1 |

EdNet's sampled change is positive-to-negative; ASSIST/Junyi have zero.
All repeated float32 forwards at unchanged settings are bitwise identical in
the probed batches. Changing CPU thread count from2 to1 or disabling MKLDNN
does change logits slightly, demonstrating deterministic arithmetic depends on
the backend/settings. Junyi/EdNet's earlier full CPU capture used one thread;
these fresh probes used two. This is not random evaluation or evidence of
target misalignment. No claim of global bitwise CPU determinism is inferred
from sampled repeats.

| Dataset | Probed batches | Max 1-vs-2-thread logit difference | Max MKLDNN-off logit difference | Max fresh FP32 vs cached logit difference |
| --- | --- | --- | --- | --- |
| ASSIST2017 | 3 | 1.072883605957031e-06 | 1.311302185058594e-06 | 0 |
| Junyi | 6 | 1.430511474609375e-06 | 1.192092895507812e-06 | 1.430511474609375e-06 |
| EdNet-KT1 | 6 | 1.430511474609375e-06 | 1.668930053710938e-06 | 1.430511474609375e-06 |

Stage hooks in ASSIST's first three batches locate ordinary CPU32/CPU64
differences throughout ActionEncoder, SessionEncoder, CorrectPaddingEncoder,
Decoder and prediction; their maxima are recorded in the full precision
manifest. These are within-CPU diagnostics, not a localization of the missing
historical CUDA discrepancy. Detailed settings and scopes:
[ASSIST full](reproducibility_diagnostics/assist2017_fp64_full.json),
[Junyi sample](reproducibility_diagnostics/junyi_fp64_sample.json),
[EdNet sample](reproducibility_diagnostics/ednet_kt1_fp64_sample.json).

## ASSIST2017 BCE: ruled-out explanations and unresolved cause

| Computation | BCE |
| --- | --- |
| Historical CUDA | 0.6087257417328864 |
| CPU historical FP32 batch weighting | 0.6087236110378144 |
| Same CPU logits, stable float64 accumulation | 0.6087236095474302 |
| Same CPU logits, single global FP32 reduction (counterfactual) | 0.6087236404418945 |
| Fresh float64 network forward, all targets | 0.6087236104891285 |

The CPU batch-FP32 minus stable-FP64 difference is **1.4903841494273706e-9**,
while the historical-minus-batch-CPU gap is **2.1306950720445528e-6**.
Changing only accumulation precision leaves essentially the whole discrepancy.
Full double network inference changes stable CPU BCE by approximately9.42e-10,
and still misses historical BCE by **2.1312437579190657e-6**. It changes zero
classifications. Thus neither of these CPU precision changes resolves the strict
failure. A single global FP32 reduction also fails to close the gap.

An upstream CUDA numerical path/settings difference is a plausible hypothesis,
not a proven root cause. Missing original pointwise scores and runtime flags
prevent identifying a specific historical operation. The issue cannot honestly
be marked resolved as harmless rounding or relabelled PASS.

## Runtime, batch ordering and CUDA verification

The current runtime is PyTorch `2.4.0a0+f70bd71a48.nv24.06`, CUDA build12.5,
cuDNN90100, NumPy1.24.4. Fresh CPU processes report float32 matmul precision
`high`, `cuda.matmul.allow_tf32=True`, `cudnn.allow_tf32=True`, MKLDNN enabled.
These flags were read without allocating CUDA. The project does not explicitly
set TF32/matmul precision; historical artifacts do not capture those actual
values. Upstream release defaults must not be substituted for this NVIDIA
build's observed defaults or assumed to prove historical settings.

[ktbench/config.py](../ktbench/config.py), lines11–31, sets workspace config
`:4096:8` only if unset, seeds42, requests deterministic algorithms with
`warn_only=True`, disables cuDNN benchmarking and enables cuDNN determinism.
Deterministic algorithms do not imply cross-device bit identity. HiTSKT uses
linear projections, normalization, activations and prefix reductions; it has no
convolution/LSTM, so changing a cuDNN flag alone is not an established explanation.
[PyTorch's numerical-accuracy note](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.0/docs/source/notes/numerical_accuracy.rst)
and [CUDA precision documentation](https://docs.pytorch.org/docs/2.4/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)
support investigating hardware and reduction paths, not excusing a failed gate.

| Dataset | Regenerated batches | Exact IDs/labels/cache order | Validation-selected epoch / completed |
| --- | --- | --- | --- |
| ASSIST2017 | 98 | PASS | 40 / 45 |
| Junyi | 3658 | PASS | 32 / 37 |
| EdNet-KT1 | 24248 | PASS | 30 / 35 |

All test event IDs, labels, uniqueness and regenerated batch memberships
match the prior full CPU capture, using historical token budget32768, batch cap64,
bucket512, shuffleFalse, sampler epoch0, seed42. See [evaluation_order.json](reproducibility_diagnostics/evaluation_order.json).
Original CUDA batch vectors were not retained. The unchanged source constructs
the same evaluation sampler; a historical backend-specific forward path remains
unverified. All three best hashes, validation selection, original dataset/store
hashes and target populations are preserved; [artifact_and_dkt_forensics.json](reproducibility_diagnostics/artifact_and_dkt_forensics.json)
and the per-dataset first-audit manifests give the exact bindings.

Historical configs identify **H10080GB, MIG3g.40gb**. The only visible partition
is that same architecture and is occupied; LightSAKT2 has an active job, and
compute-process visibility is restricted. No CUDA allocation was attempted.
**CUDA verification: NOT VERIFIED for all datasets**, not a failed CUDA replay
and not a successful matching-architecture reproduction. Evidence:
[resource_final.json](reproducibility_diagnostics/resource_final.json).

Once a genuinely idle independent partition is established, the next bounded
verification is a fresh process with the original batch shapes and source, H100
MIG architecture, recorded NVIDIA runtime, seed42 and original evaluation
definition. Capture actual TF32/cuBLAS/cuDNN settings rather than assume them;
first replay the best-supported historical settings without tuning on test
outcomes. Save local event-indexed logits/probabilities and two identical repeats,
hash all inputs, compare every stored metric using the unchanged1e-6 gate, and
count fresh-CUDA-versus-CPU pointwise differences exactly. A separately labelled
TF32-disabled diagnostic may isolate numerical sensitivity but cannot silently
replace historical settings. If historical flags cannot be established, report
the reconstruction limitation even if aggregate metrics match. Never stop
LightSAKT2 to obtain this evidence.

## Verification and preservation

**59 relevant tests PASS** in this follow-up: the existing49 independent
correctness/audit-oracle tests plus10 new exhaustive diagnostic cases. This
includes explicit causal output/gradient, masking and future-response checks;
passing them does not certify response availability or historical metrics.
[verification_tests.log](reproducibility_diagnostics/verification_tests.log)
records the executed scope. The first audit's broader133-test result remains
preserved, not presented as a new133-test run.

Initial and final preservation checks cover all375 original captured files and
106 prior audit report/evidence/script/test files. See
[preservation_initial.json](reproducibility_diagnostics/preservation_initial.json)
and [preservation_final.json](reproducibility_diagnostics/preservation_final.json).
No original checkpoint/result or production scientific file was changed.
Ignored local arrays contain learner-level scores/IDs; only aggregate evidence
and hashes enter Git. The actual cgroup limit is90GiB; model inference was serial
with CUDA hidden, two CPU threads, low priority and a16GiB forward-allocation
estimate guard. The preexisting OOM-kill counter remained2, with no new OOM kill
observed. This estimate is not a claim of a measured16GiB total RSS peak.

The defensible decision is **additional verification, not automatic retraining**
for historical V1 on each dataset. Retain original evidence provisionally,
accompany CPU metrics with their environment, and do not describe the historical
metric set as independently reproduced. Availability and publication decisions
are addressed in [the timing analysis](HITSKT_RESPONSE_TIMING_ANALYSIS.md) and
[the research decision report](HITSKT_RESEARCH_DECISION_REPORT.md).
