# HiTSKT FAVOR+ v1 — implementation and CPU verification

19 October 2026 UTC. Branch: `research/hitskt-favor-plus-v1`; continuation base:
`bc262e66a8bb8ae25dc4dc38d7ca47d74c516375`.

**CPU correctness: PASS. Candidate for a bounded pilot on an independently
available GPU. CUDA correctness/performance and full dataset results: NOT
VERIFIED / NOT RUN.** No FAVOR+ dataset AUC is reported here.

The user explicitly corrected the nonexistent attachment dependency and
authorized implementation from scratch. This report supersedes the dependency
and implementation status in the preserved, dated `INTEGRATION_STATUS.md` and
earlier proposal; neither historical document has been overwritten.

## Implementation and mathematical identity

New code: [favor_plus.py](../../ktbench/models/favor_plus.py),
[hitskt_favor_plus.py](../../ktbench/models/hitskt_favor_plus.py).
This is positive orthogonal random-feature softmax attention, following
sections 2.3–2.4 and Appendix B of the primary
[Performer paper](https://arxiv.org/html/2009.14794v4). It is a separate
hierarchical HiTSKT variant, `hitskt_favor_plus_v1`.

For head width d, x=q/d^(1/4), y=k/d^(1/4), the feature map is
`phi(x)[r] = exp(omega[r]·x - ||x||²/2) / sqrt(m)`.
Its inner product estimates `exp(q·k/sqrt(d))`. Gaussian QR blocks with sign
correction supply orthogonal directions; independent chi_d radii preserve
Gaussian marginals. Blocks are independent when m>d. Heads and layers receive
separate projections. The kernel estimator's unbiasedness does not imply an
unbiased normalized attention ratio. In contrast, the unchanged historical
model uses deterministic `ELU(x)+1`, without random softmax features.

The following stabilization and backward implementation are independently
derived engineering choices, tested against an explicit feature-kernel oracle:

- Per feature r and causal prefix i, retain maximum M, sum Z of
  `exp(log(phi(k_j))[r]-M)`, and weighted sum S of that quantity times v_j.
  When M increases, rescale both Z and S. All eligible j satisfy j<=i.
- Compute feature weights with `softmax_r(log(phi(q_i))[r]+M[r]+log(Z[r]))`,
  then sum `weight[r] * S[r]/Z[r]`. This equals the normalized feature-kernel
  prefix in exact arithmetic. Query-wise constants cancel and are omitted.
  Per-feature scales are explicitly compensated; keys are not independently
  normalized without compensation.
- No future/global sequence maximum, additive feature epsilon, denominator
  floor or Q/K clipping changes the kernel. Before the first valid key, output
  is zero. PAD keys/values are inert, including nonfinite masked values.
- Random-feature arithmetic accumulates in float32 by default; float64 is
  supported and double inputs remain double. CPU bfloat16/autocast controls
  verify the precision island. This is not CUDA mixed-precision certification.
- A custom first-order VJP saves chunk-boundary states and recomputes one
  chunk's graph backward, carrying state gradients across every boundary.
  There is no truncation of gradients. Higher-order derivatives are unsupported.

The default is m=128, chunk size32, projection seed42, independent local CPU
generators, and fixed projections throughout training/evaluation. Buffers save
the actual matrices. Format2 checkpoints include the variant, model and FAVOR+
configuration; legacy format1 and FAVOR+ loaders reject one another's files.
There is no implicit conversion of historical ELU+1 weights into FAVOR+ results.

The hierarchy, embeddings, positions, normalization, feed-forward layers,
residuals, output head, learned Q/K/V/output projections, dropout placement,
shift/EOS handling, loss and metrics reuse the original implementation.
An explicit test proves identical initial learned tensors, parameter counts
and global RNG state under equal seeds; only the attention algorithm/buffers
change. Original `hitskt.py`, `performer.py`, registry and data code are unchanged.

## Correctness evidence

**74 new FAVOR+-specific test cases PASS, zero failures. 143 existing regression
cases PASS. Total:217 distinct cases.** The final combined run has214 cases
(71 new +143 existing), followed by3 new pilot-admission/CPU cases. See
[final suite](implementation_v1/final_verified_suite.log),
[final suite XML](implementation_v1/final_verified_suite.xml), and
[pilot tests](implementation_v1/pilot_tests.log).
The12 warnings in the full suite come from the existing transport-resume tests
requesting2 DataLoader workers while the test process is pinned to1 CPU.

| Check | Status and scope |
| --- | --- |
| Projection generation | PASS: within-block orthogonality, chi/Gaussian marginal statistics, seed changes/repeatability, unchanged global RNG |
| Feature formula/scaling | PASS: independent positive-feature formula including d^(-1/4) and 1/sqrt(m) |
| Explicit causal feature-kernel oracle | PASS: independent L² log-kernel construction; lengths1–11, masks/holes/empty keys, batches/heads, uneven chunks |
| Gradients | PASS: Q/K/V analytic comparisons, finite differences, cross-boundary propagation, finite full-model gradients |
| Causality | PASS: bitwise future invariance in tested float32 cases; zero future/masked-key/cross-batch gradients |
| Own-label and future-response exclusion | PASS: independently recollated labels, lengths1/2/4/7, histories1/3, mixed batches; full-model correctness-input Jacobian; fixed-RNG dropout training |
| PAD/EOS | PASS: right-padded full-model batches, EOS included in attention/excluded from loss, masked-input contamination and shorter companions |
| Checkpoints | PASS: exact frozen-buffer/logit reload, cross-variant rejection, fresh-process CPU replay for all3 synthetic profiles |
| Pipeline | PASS on tiny synthetic store: original sampler/counts, one test epoch, validation-selected checkpoint, best reload, all metrics and fresh replay, hash mismatch rejection, namespace protection |
| Long sequences | PASS: attention forward/backward at939/3925/13081 positions; full hierarchy at13080 target actions plus EOS and a shorter companion |
| Original model regression | PASS:143 existing tests; no old scientific source modifications |
| CUDA / real training / real FAVOR+ AUC | NOT VERIFIED / NOT RUN |

Oracle tolerances were specified in tests: double outputs atol2e-12/rtol2e-11,
gradients atol3e-11/rtol3e-10; finite differences eps1e-6, atol2e-6/rtol2e-4.
Softmax approximation is assessed separately from equivalence to a fixed
feature kernel. Historical metric acceptance remains absolute1e-6; no gate
was loosened. New checkpoint replay records all seven differences against that
gate if matching stored FAVOR+ results exist; execution alone is not PASS.

Initial49/62-case runs had no correctness failures. Their logs, initial kernel
snapshot and subsequent69/70-case runs remain in `implementation_v1/`.
The original implementation's graph-retention memory problem and measurements
were preserved before refinement; passing small tests did not establish memory
safety.

## Long-session runtime, memory and approximation limitations

Serial CPU thread222, nice19, CUDA hidden, one BLAS/OpenMP thread; actual
container limit90GiB. Measurements use fresh child processes, B=1/H=2/d=64/
m=128. Length includes EOS; these are synthetic attention checks, not epochs.
RSS includes Python/PyTorch libraries. See
[initial diagnostics](implementation_v1/cpu_diagnostics.json) and
[optimized diagnostics](implementation_v1/optimized_cpu_diagnostics.json).

| Positions | Chunk | Optimized forward s | Backward s | Peak RSS GB (decimal) |
| ---: | ---: | ---: | ---: | ---: |
| 939 | 32 | 0.080 | 0.581 | 0.656 |
| 3925 | 32 | 0.325 | 1.543 | 0.686 |
| 13081 | 16 | 1.100 | 4.448 | 0.832 |
| 13081 | 32 | 1.078 | 4.513 | 0.786 |
| 13081 | 64 | 1.079 | 4.626 | 0.799 |

At13081/chunk32, the earlier framework-checkpoint version used5.330GB and
5.112s forward/12.157s backward. The optimized kernel used0.786GB and
1.078s/4.513s. This is one measured comparison, not a universal speedup claim.

Forward state is O(BHmdv), with linear feature/input/output storage. Backward
retains O(BH ceil(L/C) mdv) boundary states and O(BHCmdv) temporary recomputation
workspace; memory is not constant in L. There is no retained full L-by-m-by-dv
prefix tensor or L-by-L attention matrix. A saved-tensor structural test checks
chunk-boundary retention. Larger flattened history batches still cost memory.

[Full-model evidence](implementation_v1/long_full_model.json): unchanged
EdNet width128/heads2/FF1024, m128, target lengths13080/17, histories[21,4]/[3],
13,097 supervised targets, finite forward/backward, 3.506s/12.556s, peak1.751GB.
Companion-padding max logit difference was4.76837158203125e-7. No optimizer step
was taken in this long-model check. This does not cover maximum history lengths
and maximum batch size simultaneously.

The prespecified approximation grid uses input scales0.35/1/3, m8/32/128/256,
eight projection seeds and length24/d8. Mean output MSE against exact causal
softmax is:

| Input scale | m8 | m32 | m128 | m256 |
| ---: | ---: | ---: | ---: | ---: |
| 0.35 | 0.003781 | 0.000865 | 0.000288 | 0.000141 |
| 1 | 0.095649 | 0.060560 | 0.034040 | 0.026206 |
| 3 | 1.054732 | 0.883735 | 0.758011 | 0.732035 |

The high-norm results are a material limitation: this implementation is a
random-feature approximation, not exact softmax or a guarantee of adequate
approximation for trained Q/K. Default m128 has not been selected using dataset
outcomes and is not validated as an optimal scientific setting. No clipping,
extra Q/K normalization or data-protocol change was added to hide these results.

## Separate training/evaluation and dataset compatibility

[Pipeline](../../ktbench/favor_plus_pipeline.py),
[training CLI](../../scripts/train_hitskt_favor_plus.py),
[evaluation CLI](../../scripts/evaluate_hitskt_favor_plus.py),
[bounded pilot CLI](../../scripts/pilot_hitskt_favor_plus.py).
All run paths are fresh directories under
`experiments/hitskt_favor_plus_v1/<dataset>/<run_id>/`; path traversal, existing
runs and symlink escapes are rejected. No implicit resume or CUDA selection.
Smoke/pilot markers cannot be mistaken for completed benchmark `_SUCCESS`.

[Read-only plan](implementation_v1/dataset_pipeline_plan.json) verifies all
non-attention settings against the actual historical saved configurations and
the supervised counts against their dataset statistics:

| Dataset | Width/heads/FF | LR | Train / validation / test targets |
| --- | --- | ---: | --- |
| ASSIST2017 | 256/4/2048 | 5e-5 | 566107 /125800 /112252 |
| Junyi | 128/2/1024 | 5e-5 | 8920358 /2556773 /2456402 |
| EdNet-KT1 | 128/2/1024 | 8e-5 | 49458444 /14520975 /13429870 |

Original SessionStore, 15-session rolling history, chronological within-student
splits, seed42, Adam/zero decay, dropout0.1, stage depths1, batch cap64,
token budget32768, bucket512, strict validation-AUC selection/patience5 and
epoch ceilings100/50/40 are retained. Original processed checksums are indexed
in the plan and independently rehashed by the preservation check.

V1 remains **retrospective teacher-forced evaluation**, with preceding recorded
responses visible, including earlier validation/test responses under fixed
weights. It is not student-disjoint and does not guarantee wall-clock response
availability. Existing response-timing limitations remain; no new assertion of
confirmed timing leakage or deployment validity is made. Availability-aware V2
is not implemented.

All three registered profiles passed two CPU synthetic optimization steps and
independent fresh-process checkpoint/logit replay, with max difference0:
`experiments/hitskt_favor_plus_v1/{assist2017,junyi,ednet_kt1}/cpu_smoke_20261019_v1/`.
Each has a JSON manifest, locally retained checkpoint/logits and `_SMOKE_SUCCESS`;
replay evidence is in `implementation_v1/*_fresh_smoke_replay.json`.
These use full registered widths/FF/LR but synthetic vocabularies20/5 and ten
targets per step. **They are not training results for those datasets.** Binary
checkpoint/prediction arrays are excluded from Git in the new namespace.

CPU reproduction command (choose a new run ID):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=. python scripts/train_hitskt_favor_plus.py junyi --run-id my_cpu_smoke
```

Full training is prepared behind `--mode full --allow-full-training --store`
and an explicit device; **it has not been authorized or executed here**.
Fresh evaluation accepts only the new variant and matching dataset/profile/
protocol hashes, exports logits/probabilities/labels, all seven metrics,
settings and hashes, and reports PASS/FAIL/NOT VERIFIED for metric reproduction.

## GPU pilot decision and remaining work

**Ready for a bounded correctness/performance pilot, conditional on an
independently available GPU and an explicit execution decision.** CUDA has not
been allocated in this work. The currently occupied H100 MIG is not available
to this task. The new pilot runner rejects unconfirmed CUDA execution before
initialization, caps B=1, steps<=5 and target length<=13080, uses a small synthetic
vocabulary, and writes only in the new namespace. The confirmation flag is an
operator assertion, not an automatic availability detector. Its CPU path and
admission checks pass; CUDA execution is NOT VERIFIED.

Minimum next evidence, in order:

1. Establish an independent device allocation without interrupting LightSAKT2.
   Run the bounded pilot at length257, then939/3925/13081 positions including
   EOS. Record CUDA runtime/TF32/determinism, wall time, peak VRAM, finite
   gradients and checkpoint replay. Sequential token scans may incur substantial
   GPU launch overhead; no GPU throughput claim follows from CPU timings.
2. On that device, compare output/first-order gradients with the explicit
   feature-kernel oracle and CPU for frozen projections; test masks, chunks,
   dropout-controlled causality and long sessions. Mixed precision and wider
   batches require their own evidence. Keep numerical gates fixed in advance.
3. Use only an approved bounded training-split sample to inspect learned Q/K
   norms, feature concentration and approximation variance across projection
   seeds. Decide feature counts on training/validation evidence, not test AUC.
   Measure the actual flattened-history batch memory before scaling.
4. Only after reviewing the pilot decide separately whether to authorize full
   new FAVOR+ training. Historical ELU+1 results are not FAVOR+ controls trained
   under these features, and publication evidence will require actual new runs,
   fresh evaluation, multiple training seeds and valid baseline comparisons.

No full experiment, availability-aware V2, historical retraining, DKT/EdNet
resume/optimizer reset or LightSAKT2 modification/signal occurred.
[Preservation](implementation_v1/preservation.json) verifies375 originals
(all30 captured checkpoints),277 prior source/evidence files and the unrelated
README/SAKT log. All earlier audit/preparation evidence and Git history remain.
Publication readiness is not claimed.
