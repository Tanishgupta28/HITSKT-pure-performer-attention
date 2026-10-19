# HiTSKT FAVOR+ v1 integration preparation

19 October 2026 UTC. Research branch: `research/hitskt-favor-plus-v1`.
Base: `f465d84838e5f8b1c637736e362a945c20cfde10`.

**Implementation and GPU-pilot readiness: NOT VERIFIED.** The user-referenced
starter implementation and its standalone/integration tests are not visible in
the conversation, repository, common attachment mounts or workspace archive.
The existing FAVOR+ proposal is a design document, not the missing starter.
A local-path/download-link clarification has been requested. No replacement
implementation was invented or represented as the user's starter.

## Verified preparation

- Created the separate research branch; canonical `main` remains unchanged.
- Reconciled durable memory and the new authorization. The new implementation
  is authorized on this branch; full dataset training remains prohibited.
- Verified375 original research files, including all30 original best/last
  checkpoints, and268 additional prior implementation/report/test files against
  the base commit and captured hashes. See
  [preservation_baseline.json](preservation_baseline.json) and
  [preservation_handoff.json](preservation_handoff.json).
- Re-ran59 existing independent ELU+1 masking/causality/gradient/audit controls
  on one low-priority CPU thread with CUDA hidden: **59 PASS**. These are
  legacy controls, not FAVOR+ starter or integration test results. See
  [legacy_cpu_controls.log](legacy_cpu_controls.log).
- Preserved the unrelated root README edit and SAKT console log. LightSAKT2
  and the unfinished DKT/EdNet run were not modified or signalled. No CUDA
  allocation, training-pipeline execution or new experiment was started.

## Work that depends on the starter

| Requirement | Actual status |
| --- | --- |
| Inspect/hash supplied starter, dependencies and license/provenance | NOT RUN: source unavailable |
| Run unmodified standalone starter tests first | NOT RUN: tests unavailable |
| Run full-model integration tests before fixing failures | NOT RUN: source/tests unavailable |
| Implement isolated `hitskt_favor_plus_v1` model and training pipeline | NOT IMPLEMENTED |
| Optimize long-session forward/backward memory/runtime | NOT RUN |
| Validate numerics, gradients, causal prefixes, save/reload and approximation | NOT RUN |
| Bounded FAVOR+ CPU smoke training/evaluation | NOT RUN |
| Ready for an independent GPU pilot | NOT VERIFIED |

No failed FAVOR+ result is inferred from a missing file, and no legacy test is
counted as proof that FAVOR+ works.

## Frozen integration boundary and next bounded action

Once the starter is available, preserve an original source hash/snapshot and
run its standalone and full-model integration tests before editing. Retain
initial failures. The integration should add isolated modules/configuration
and a variant-specific checkpoint loader; it must not alter or relabel any
historical ELU+1 checkpoint. Changing attention only must leave embeddings,
normalization, feed-forward networks, hierarchy, positions, EOS/shift/loss masks
and non-attention initialization consistent with the historical architecture.

Use the same `SessionStore`, rolling dataset, collator, original dataset-specific
HiTSKT profiles, splits and retrospective teacher-forced V1 evaluation. Use
`experiments/hitskt_favor_plus_v1/<dataset>/<run_id>/` for new outputs only.
Record feature count/seed, projection matrices/sampling law, stabilization,
redraw policy, numeric precision and source hashes in new checkpoints. Reject
cross-variant loading by default. Do not implement availability-aware V2.

Required CPU validation includes an independent explicit feature-kernel oracle
for outputs and gradients; a separate softmax-approximation assessment; own/
future-response perturbations across masks/batches/sessions; projection save/
reload identity; uneven/chunk-boundary/long-session forward and backward tests;
causality-safe stabilization; and measured memory/runtime. Current real session
maxima are938/3,924/13,080 actions. Avoid a full `[B,H,L,m,dv]` prefix tensor;
validate chunking/backward memory rather than assuming linear-time attention
is memory-safe. Use bounded CPU examples first and report actual tested scopes.

The host cgroup is90GiB with protected training active; preserve ample headroom,
serial low-priority execution and CUDA-hidden CPU tests. No full training or
GPU pilot is to be launched by this preparation checkpoint. The missing starter
is the immediate required input before implementation can continue.
