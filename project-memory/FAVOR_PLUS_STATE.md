# FAVOR+ v1 verified CPU implementation

19 October2026 UTC. Branch research/hitskt-favor-plus-v1; continuation base
bc262e6. Latest user explicitly authorizes genuine FAVOR+ from scratch;
no starter/attachment dependency exists. Historical preparation is retained in
[FAVOR_PLUS_PREPARATION_HISTORY.md](FAVOR_PLUS_PREPARATION_HISTORY.md).

CPU correctness PASS:74 new FAVOR+-specific tests +143 legacy regression cases
=217 distinct passes,0 failures/skips. Combined214-case suite plus3 later pilot
admission/CPU tests. Evidence: reports/favor_plus_v1/implementation_v1/
final_verified_suite.log, pilot_tests.log, verification_summary.json.
Main report: reports/favor_plus_v1/IMPLEMENTATION_VALIDATION.md.

New isolated ktbench/models/favor_plus.py and hitskt_favor_plus.py implement
positive Gaussian-marginal orthogonal softmax features, d^(-1/4) scaling,
compensated per-feature causal maxima/sums, fixed local-seed projection buffers,
and a custom first-order chunk-recomputed backward. Format2/variant tag rejects
legacy format1. All learned initialization/hierarchy/components remain identical
outside the attention algorithm. Original scientific source unchanged.

Independent explicit feature-kernel output/gradient and finite-difference tests,
mask/own-label/future/cross-batch/dropout checks, PAD/EOS, mixed precision CPU
controls, full-model gradients and checkpoints PASS. Long attention13081
positions/H2/d64/m128/chunk32 peaks0.786GB vs initial5.330GB, fwd/bwd1.078/4.513s.
Full EdNet-profile synthetic hierarchy, targets13080/17, finite fwd/bwd, peak
1.751GB; padding delta4.7684e-7. Initial source/measurements remain preserved.
Higher-order derivatives unsupported. Large flattened history batches untested.

Separate pipeline/CLIs reuse original data/splits/sampler/profile/loss/selection.
Read-only all3 profile/target-count check PASS. All3 registered profiles passed
2-step CPU synthetic optimization and fresh-process checkpoint/logit replay
(max difference0), under experiments/hitskt_favor_plus_v1/<dataset>/
cpu_smoke_20261019_v1/. Reduced vocabularies; no real dataset AUC. New binaries
stay local/ignored. Explicit device, fresh guarded paths, no implicit resume.
Bounded synthetic pilot CLI has a CUDA admission guard and CPU-tested path;
no CUDA executed. Future full training is behind an explicit execution flag.

Readiness: candidate for bounded pilot on an independently available GPU;
CUDA correctness/performance and dataset training NOT VERIFIED/NOT RUN.
High-norm softmax approximation is poor (scale3/m128 mean MSE0.758 in declared
synthetic grid); no universal approximation or publication claim. Feature count
not selected on test outcomes. V1 response-availability limitation remains.

Preservation PASS:375 originals/all30 checkpoints +277 prior source/evidence
files, including all old audits/preparation, and unrelated README/SAKT log.
No LightSAKT2 change/signal, DKT resume/reset, full training, CUDA, or V2.
Cgroup90GiB, OOM kills2 unchanged. Implementation/report commit 67f09d2291e632398826af21e5f731f8762a05f9
is pushed and remote-verified on the research branch; main unchanged. Handoff
publication receipt is in implementation_v1/publication_receipt.json. Wait for
independent GPU availability and explicit execution direction. See
[NEXT_CHECKPOINT.md](NEXT_CHECKPOINT.md).
