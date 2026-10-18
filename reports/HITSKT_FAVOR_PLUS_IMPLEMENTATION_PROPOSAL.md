# Genuine FAVOR+ HiTSKT — implementation proposal only

Status: **NOT IMPLEMENTED / NOT TRAINED**. Preserve the historical hierarchical
HiTSKT variant with deterministic causal ELU+1 linear attention. Existing class
and artifact names containing `Performer` are legacy labels, not evidence of
FAVOR+. No historical result should be attributed to randomized softmax
approximation. See [the research decision report](HITSKT_RESEARCH_DECISION_REPORT.md).

## Mathematical contract

The current kernel is

\[
 k_{ELU}(q,k)=(\operatorname{ELU}(q)+1)^\top(\operatorname{ELU}(k)+1).
\]

For each head and position it evaluates the normalized prefix sums of that
kernel, with a denominator floor of `1e-6`. Its feature dimension is the head
width. No random feature matrix exists.

A separately named FAVOR+ variant would approximate
\(\exp(q^\top k/\sqrt d)\). Set \(x=q/d^{1/4}\),
\(y=k/d^{1/4}\), and use positive features

\[
 \phi_\Omega(x)_r=m^{-1/2}\exp(\omega_r^\top x-\|x\|^2/2),
 \qquad \phi_\Omega(x)^\top\phi_\Omega(y)
 \approx \exp(x^\top y).
\]

Generate orthogonal Gaussian feature blocks with correct Gaussian marginal
scales, QR sign handling and independent blocks as needed. Orthogonal unit
vectors without the appropriate radial distribution are insufficient. The
normalized attention ratio is approximate; an unbiased kernel estimator does
not imply an unbiased normalized attention output. This follows the primary
[Performer paper](https://arxiv.org/abs/2009.14794), distinct from general
[deterministic linear attention](https://proceedings.mlr.press/v119/katharopoulos20a.html).

## Isolated implementation

Use a new variant ID such as `hitskt_favor_plus_v1`, new checkpoint format field
and separate experiment directories. Keep hierarchy, shifts, masks, target
cohort, feature inputs and registered training settings fixed for an attention
ablation. Store feature count, feature seed, projection matrices, sampling
algorithm, scaling, stabilization, redraw policy and numeric precision in every
checkpoint. Freeze random features during evaluation and persist any training
redraw RNG/state. Do not load old ELU checkpoints as if they were trained FAVOR+.

Compute causal key and key/value prefix states using the random features.
Avoid an L-by-L score matrix. Chunk or stream the prefix implementation to
bound temporary storage; simply materializing `[B,H,L,m,dv]` can exhaust memory
despite linear asymptotic sequence scaling.

Stabilization must preserve the causal kernel ratio. A query-dependent common
positive factor cancels in its normalized output. Key stabilization must use
a shared scale for all accumulated past keys and rescale the accumulated states
consistently when an online scale changes. Independent uncorrected per-key
max subtraction changes attention weights. A maximum over future keys also
creates a future dependency; do not introduce one silently. Specify and test
the denominator floor and feature under/overflow handling.

## Required independent acceptance evidence

- Compare outputs and gradients with an explicit causal feature-kernel matrix
  using identical frozen projections, including masks, EOS, padding, chunk
  boundaries and nonuniform batches.
- Separately compare approximation error to explicit causal softmax across
  feature counts and independent projection seeds. Exact feature-kernel
  equivalence is not proof of good softmax approximation.
- Test projection distribution/scales, save/reload identity, reproducible redraws,
  long-prefix numerical stability, and future-response invariance at every stage.
- Record memory/runtime and approximation error without selecting a feature
  count on test outcomes. Predeclare seeds, comparison protocol and numerical
  tolerances. Keep random-feature variance distinct from training-seed variance.

Changing the feature map changes the model. A scientifically attributable
FAVOR+ result requires new, separately approved training; substituting FAVOR+
only while replaying an ELU checkpoint does not verify historical experiments.
Do not begin full training or change the present architecture under this proposal.
