# ASSIST2017 HiTSKT FAVOR+ — complete epoch-training results

24 October 2026. Branch `research/hitskt-favor-plus-v1`; execution source commit
`7b0328693fbaf33304c6e6748f49805a17f58a02`. Frozen plan SHA256:
`8a0958f4572e11d9ae0d5722f595ef070c84129d9460f4b382b70c70ac2427cb`.

**Eight full training epochs and full validation after each epoch completed.
1,454 real Adam updates. Best validation AUC: 0.703467 at epoch 7; F1 at that
checkpoint: 0.508814. Highest observed validation F1: 0.532809 at epoch 6.**
No test evaluation. No convergence, superiority or statistical-significance claim.

## Actual stopping decision

The user approved up to ten epochs/patience three and explicitly accepted the
separate profile/full-run resource proposal. The unchanged campaign limit was
45 minutes. After eight verified epochs, the foreground controller declined
another full epoch: 2,451.736 seconds had elapsed; the measured previous epoch
plus final-evaluation allowance required 354.566 seconds, exceeding the remaining
248.264 seconds. Epoch 9 was never started. Early stopping had **not** triggered
(one nonimproving epoch; patience remains three). The final selected-checkpoint
replay finished at 2,498.133 seconds, **41 minutes 38 seconds**, within the ceiling.
The timer was not reset and no optimizer step was retried.

The CPU-only time-closure procedure validates all completed epoch/recovery receipts,
retains the actual AUC selection and records `TIME_BOUNDED_EPOCH_TRAINING_COMPLETE`;
it does not pretend ten epochs or patience-three stopping occurred.
See [time-stop receipt](assist_epochs_execution_20261024_v1/CONTROLLER_TIME_STOP.json),
[closure record](assist_epochs_execution_20261024_v1/time_closure.console.log) and
[approved budget](assist_epochs_execution_20261024_v1/BUDGET_APPROVAL.md).

## Model, data and training protocol

This is hierarchical HiTSKT with **genuine softmax-kernel FAVOR+ attention**,
positive orthogonal random features, causal prefix computation and fixed
128-feature projections. It is distinct from the preserved historical deterministic
ELU+1 model. All nonattention components remain the original width256/heads4/
FF2048/dropout0.1/one-layer architecture. Adam: LR5e-5, betas0.9/0.999, eps1e-8,
weight decay0, no clipping. Seed42, FP32, highest matmul precision, TF32 off,
deterministic algorithms; fixed original per-stage projection policy/chunk32.

Fresh seed42 initialization was used, preserving diagnostic checkpoint50
unchanged. Its subset/singleton optimization was not imported into the full-data
experiment. Checkpoint50 SHA256:
`93ebe9d2a29be9a6e4deb278916513339de61c9ac8f72a454a31111829440a73`.

Complete training: **7,060 target sessions /566,107 labels per epoch**.
Complete validation: **1,977 target sessions /125,800 labels per epoch**;
47,584 positive labels (37.8251%). Batch maximum64, token cost32768,
bucket512/history15, exact seeded epoch schedules. All target/historical action
lengths are preserved (maximum938 actions plusEOS). All valid target sessions
are included exactly once per epoch: no filtering, truncation, duplicate targets
or dropped oversize examples. First context sessions follow the unchanged
original supervision protocol. Eight epochs present 4,528,856 training labels;
these are repeated presentations of 566,107 distinct supervised interactions.

All37 source identities and11 processed-file checksums remain unchanged. The
previous preparation checked student boundaries, monotone chronological splits,
past-only history and target alignment. The historical229 CPU/18 staged CUDA
checks remain prior evidence; they were not rerun or counted as new tests here.
New full-size profiling and actual full-data gradients/serialization/replays
provide execution evidence for this run. The controller's28 targeted CPU tests
passed during preparation; the CPU time-closure self-test passed on the first
four actual completed receipts, including rejection of out-of-order epochs.

Evaluation remains **chronological within-student retrospective teacher forcing**,
not student-disjoint or guaranteed real-time response availability. The documented
response-timing limitation remains; it is not relabeled confirmed leakage. Test
examples were never constructed/evaluated by this controller; no test AUC was used
for selection, feature choice or thresholds. Localpatience3/max10 differ from
historical commonpatience5/max100 and must be disclosed in comparisons.

## Full-validation results

All entries below are complete-validation measurements at fixed threshold0.5.
Training BCE is the label-weighted online training loss across the epoch, with
training dropout active; it is not a final-weight evaluation of the train split.
Selection uses strict full-validation-AUC improvement only. Maximum F1 is recorded
separately and does not override AUC selection.

|Epoch|Training BCE|Validation AUC|Accuracy|Precision|Recall|F1|Validation BCE|
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.638942 | 0.649592 | 0.658553 | 0.605563 | 0.279085 | 0.382081 | 0.627900 |
| 2 | 0.605684 | 0.674090 | 0.663633 | 0.570184 | 0.449794 | 0.502884 | 0.621430 |
| 3 | 0.596876 | 0.686732 | 0.677941 | 0.637717 | 0.343960 | 0.446887 | 0.607752 |
| 4 | 0.590313 | 0.692310 | 0.679253 | 0.614180 | 0.408877 | 0.490929 | 0.606542 |
| 5 | 0.586927 | 0.698317 | 0.683156 | 0.621306 | 0.415749 | 0.498155 | 0.601908 |
| 6 | 0.582839 | 0.700271 | 0.678704 | 0.592022 | 0.484364 | 0.532809 | 0.604904 |
| 7 | 0.580976 | 0.703467 | 0.685254 | 0.620946 | 0.430985 | 0.508814 | 0.599038 |
| 8 | 0.579697 | 0.700440 | 0.681828 | 0.668435 | 0.315169 | 0.428363 | 0.600564 |

[CSV](assist_epochs_execution_20261024_v1/EPOCH_METRICS.csv).
[Measured curves](assist_epochs_execution_20261024_v1/PERFORMANCE_CURVES.png).

Epoch8's F1 decline is retained. Against epoch7, mean predicted probability fell
from0.405008 to0.361890 and positive predictions fell from33,027 to22,436.
Exactly12,845 decisions changed at the unchanged0.5 threshold:6,087 TP→FN,
576 FN→TP,5,631 FP→TN and551 TN→FP. Recall fell from0.430985 to0.315169;
precision rose from0.620946 to0.668435. Independent metrics and identical labels/
ordering exclude a metric or alignment discrepancy. These observations describe
a short-run score/decision shift; they do not establish convergence or its causal
optimizer dynamics. No threshold was tuned.
[Threshold analysis](assist_epochs_execution_20261024_v1/VALIDATION_7_TO_8_THRESHOLD_ANALYSIS.json).

## FAVOR+ approximation and numerical findings

Same six fixed training-only examples/all four stages, independent float64
NumPy/SciPy exact-softmax and explicit feature-kernel references on prefixes up
to32 positions. Q/K norms also cover valid full example tokens. Relative-L2 values
below are medians over those six fixed cases; ESS is median over case pair-ESS
medians. These are diagnostic-prefix statistics, not whole-dataset error bounds.

|Checkpoint|Action error|CorrectPadding error|Decoder error|Session error|CorrectPadding pair ESS|
|---|---:|---:|---:|---:|---:|
| Initial | 0.213176 | 0.136497 | 0.101452 | 0.070492 | 9.637485 |
| 1 | 0.224943 | 0.507049 | 0.333437 | 0.064196 | 2.028782 |
| 2 | 0.235979 | 0.516964 | 0.467088 | 0.065842 | 1.338817 |
| 3 | 0.241381 | 0.506866 | 0.553815 | 0.070150 | 1.271827 |
| 4 | 0.247596 | 0.464314 | 0.534829 | 0.085888 | 1.540043 |
| 5 | 0.253887 | 0.537922 | 0.595761 | 0.071178 | 1.376363 |
| 6 | 0.257516 | 0.445471 | 0.650414 | 0.104066 | 1.637757 |
| 7 | 0.263931 | 0.475592 | 0.564818 | 0.167728 | 1.658390 |
| 8 | 0.262562 | 0.503431 | 0.562503 | 0.111974 | 1.601716 |

**0.20 REVIEW is unchanged. Action, CorrectPadding and Decoder do not pass it.**
At epoch8, Action error0.262562, CPE0.503431 and Decoder0.562503 remain substantial.
CPE ESS fell from9.637485 initially to1.271827 at epoch3, close to the severe1.25
floor, then recovered to1.601716 at epoch8. Decoder error peaked at0.650414 in
epoch6. No declared severe stage failed: median error0.75, any-prefix error1.5,
stage median pair-ESS1.25 minimum, Q/Kp95 growth2.0 maximum, or kernel-reference
max-absolute error5e-5. The largest recorded Q/Kp95 growth ratio was1.399396.

Production versus independent feature-kernel maximum absolute error across
these epoch diagnostics was **3.391481e-6**, below5e-5. Meanwhile softmax error
and concentration worsened with learning. This supports a finite-feature/
learned-concentration limitation rather than a discovered arithmetic/masking
bug or norm explosion. It does not identify the exact cause of each stage's
learning trajectory; controlled frozen-representation work would be needed.
The prior post50 investigation documents distribution/alignment sensitivity.
No architecture, projection, feature count, normalization or acceptance threshold
was changed during training. No new attention bug was established or silently
patched. Review findings remain scientifically material despite useful validation
metrics; the legacy diagnostic decision's `full_training_ready:false`/bounded-only
label is retained. The user's explicitly bounded epoch study is not publication
readiness or a fidelity PASS.

All training losses, gradients, model parameters and Adam states were finite;
all61 parameter tensors had finite gradients. Actual training gradientL2 range
0.155776–1.781248; maximum batch BCE0.756431. Zero-update profile gradients are
separate (its first singleton norm5.633857). No CUDA/OOM error, optimizer reset,
automatic retry, missing target or violated resource gate was observed.

## GPU admission, resource use and runtime

Shared authorization is recorded separately from the old pending proposal.
Exact assigned resource: H10080GB HBM3 MIG3g.40gb,
`MIG-d2fcfd79-720e-5218-82c9-0a132be68920`. Two fresh admission samples,
protected-Light hold/process checks, NVML memory/process visibility, host and
runtime checks remained active. No exclusive ownership or guessed host/container
PID alias was required. Other users' processes were not signaled or modified.

Approved8GiB allocated/reserved cap,10GiB advisory total,16GiB admission-free,
6GiB runtime-free, hostRSS4GiB/headroom2GiB, pacing0.1s,30s step/45min campaign.
Six progressive original-shape CUDA profile cases passed with **zero Adam calls**;
profile peak3.147229GiB allocated/5.980469GiB reserved. Actual full epochs include
all shapes, not just those profile cases.

|Actual measurement|Value|
|---|---:|
|Peak PyTorch allocation|3.192142 GiB|
|Peak PyTorch reservation|7.990234 GiB|
|Lowest sampled device free memory|30.987610 GiB|
|Highest sampled device-wide memory use|8.387390 GiB|
|Peak process host RSS|1.841232 GiB|
|Minimum shared-host headroom|22.022743 GiB|
|Resource observations, all admitted|5293|
|Allocator observations|5257|
|Largest measured optimizer-update duration|2.561888 s|
|Training-pass throughput (includes intermediate checkpoint overhead)|2278–2326 labels/s|
|Measured profile/training/recovery/final-replay campaign|2498.133 s|

Cached reservation reached7.990234GiB: little reservation-cap margin remained,
although peak live allocation was3.192142GiB. Future memory budgets must be
measured; no automatic increase is justified. Per-epoch update medians are
0.760–0.801s; this is measured computation/check overhead, not isolated kernel
utilization. Full train+validation+recovery epochs after the first took about
five minutes. No performance estimate was inferred from CPU benchmarks.

MIG utilization queries are unsupported: utilization remains **UNKNOWN**, not0%.
Host/container process attribution remains unresolved. Maximum reported NVML
process memory8.341797GiB and device-wide use are observations, not certified
HiTSKT process-total/nonallocator-overhead measurements. The10GiB total remains
advisory; it cannot be independently enforced from incomplete attribution.
Trusted read-only monitoring was logged and not treated as a compute owner.
After exit, one read-only snapshot reports39.337952GiB free, no NVML compute
worker/no visible Light trainer. No watcher/background polling task was created.
[After-exit snapshot](assist_epochs_execution_20261024_v1/AFTER_EXIT_RESOURCE_SNAPSHOT.json).

Environment: Python3.10.12, Torch2.4.0a0+f70bd71a48.nv24.06, CUDA runtime12.5,
NumPy1.24.4, one CPU thread/taskset223/nice10; cuDNN benchmarkoff/deterministic,
TF32off, CUBLAS_WORKSPACE_CONFIG=:4096:8, PYTHONHASHSEED42. Exact runtime/settings
and all source/data/configuration identities are retained in run receipts.

## Checkpoint and metric reproducibility

- **PASS:**72 atomic/no-overwrite checkpoints locally, including every25batches,
  final training batches and completed epochs. Model, Adam, RNG, ORF buffers,
  exact sampler cursor/target identities and metric prefixes are preserved.
- **PASS:**all eight complete checkpoints fresh-process recover exactly;
  predictions and restored-RNG next-loss/gradient probe match, zero Adam calls.
- **PASS:**AUC-selected epoch7 reloaded in another fresh CUDA process and replayed
  all125800 validation labels. Logits, probabilities and labels are bit-identical;
  AUC/accuracy/precision/recall/F1/MSE/BCE differences are all0. No test inference.
- **PASS:**independent sklearn metrics and float64 NumPy logaddexp BCE agree for
  all eight saved validation arrays, maximum absolute difference6.888404e-9,
  below unchanged1e-6. The label array/order is identical across epochs.
- **PASS:**original checkpoint50,37 bound sources,11 processed checksums,
  Light hold and pre-existing unrelated README/SAKT log changes preserved.
- **REVIEW:**finite-feature softmax fidelity. **NOT VERIFIED:**final-test
  performance, convergence, multi-seed variation, baseline superiority.

Selected checkpoint:
`experiments/hitskt_favor_plus_v1/assist2017/assist2017_full_epochs_pat3_20261023_v1/checkpoints/epoch_007.pt`.
SHA256:`07b944e71683c415cc5c00fad42648d22cb5b8fa790bc39c3c5d24b50c781781`.
Latest resumable epoch8 SHA256:
`e0fe6216e05abb2a2bd1de906d6a3f6e0319e2538ad994ad1363c446311f344b`.
Checkpoint binaries remain preserved locally; Git records evidence/receipts and
hashes, not those large binaries. Durable external checkpoint storage is not
claimed. The [verified evidence archive](assist_epochs_execution_20261024_v1/NEW_RUN_EVIDENCE.tar.gz) and
[entry manifest](assist_epochs_execution_20261024_v1/ARCHIVE_MANIFEST.json) preserve new noncheckpoint run files,
predictions and telemetry for inspection; no original result was overwritten.

A CPU postprocessing filename mismatch caused one summary attempt to fail;
the eight-epoch metric proof already existed under the earlier audit filename.
It was retained at that path, linked under the intended all-attempted filename,
and the report script destination corrected. The original script/failure log/
correction record are preserved. No metric value, source-bound training file,
checkpoint, GPU inference or optimizer update changed because of this correction.
[Correction record](assist_epochs_execution_20261024_v1/POSTPROCESSING_CORRECTION.json).

## Research decision and next action

**GO:**present these qualified, complete-validation results to the professor as
an actual single-seed eight-epoch FAVOR+ research run with exact replay evidence.
**REVIEW / NO-GO for unqualified longer/all-dataset training or publication-ready
claims:**the attention approximation warning persists and expanded stage errors
are substantial. Training stability and prediction quality are distinct from
softmax-approximation fidelity. No requirement to discard/retrain this unchanged
model follows merely from these diagnostics; retain its honest measured results.

Prioritize frozen training-only Q/K/concentration analysis at the saved selected
and latest checkpoints, including longer prefixes and predeclared feature-count
comparisons without test tuning. A different feature/normalization policy would
be a separately versioned study, never a silent in-run fix. If a further bounded
epoch continuation is chosen, preserve exact epoch8 weights/Adam/RNG/projections/
sampler and approve a new finite time window; no automatic timer reset or training
is occurring. Any further budget/profile must retain shared protections.

Before final research evaluation: settle the approximation interpretation and
configuration, complete the approved training/selection protocol, archive the
selected checkpoint/data identities, perform matched baseline and additional-seed
experiments if authorized, and only then evaluate the untouched test split once
under a frozen protocol. Current results do not establish final test accuracy,
statistical significance, convergence or model superiority.

[Structured verified summary](assist_epochs_execution_20261024_v1/VERIFIED_EXECUTION_SUMMARY.json),
[independent metric proof](assist_epochs_execution_20261024_v1/INDEPENDENT_ALL_ATTEMPTED_METRIC_VERIFICATION.json),
[professor update](assist_epochs_execution_20261024_v1/PROFESSOR_UPDATE.md).
