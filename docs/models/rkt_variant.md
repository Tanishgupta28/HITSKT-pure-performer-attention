# RKT paper-faithful performance-only Phi-relation variant

The benchmark implementation is labeled exactly:

> RKT paper-faithful performance-only Phi-relation variant with 5-fold
> student-level cross-fitting.

It is a reproduction of RKT adapted only where the three accepted datasets do
not provide comparable authoritative exercise text. It is not a generic
Transformer substitute.

## Primary references and implementation source

- Paper: Shalini Pandey and Jaideep Srivastava, *RKT: Relation-Aware
  Self-Attention for Knowledge Tracing*, CIKM 2020,
  DOI [10.1145/3340531.3411994](https://doi.org/10.1145/3340531.3411994).
  Preprint: <https://arxiv.org/abs/2008.12736>.
- Authors' repository: <https://github.com/shalini1194/RKT>.
- Inspected source: commit
  [`cac60f512f839721be0bd2757e76a0dc4466837b`](https://github.com/shalini1194/RKT/commit/cac60f512f839721be0bd2757e76a0dc4466837b),
  specifically `RKT/model_rkt.py` and `RKT/train_rkt.py`.
- Capstone implementation: `ktbench/models/rkt.py`, `ktbench/rkt/data.py`,
  and `ktbench/rkt/phi.py`.

The retained RKT components are correctness-gated question embeddings,
query/key/value scaled dot-product attention, performance-derived directed Phi
relations, temporal forgetting, convex fusion of content attention and the
relation/time distribution, dropout, and binary next-response prediction. The
rolling context has at most 49 strictly prior interactions, giving a maximum
length of 50 when the current target context is counted. Every supervised
interaction is used as one target exactly once.

The exercise-text/cosine relation is omitted. ASSIST2017, Junyi, and EdNet do
not all expose comparable authoritative question text, and the project does
not fabricate question text or textual embeddings. This is therefore an
**RKT performance-only Phi-relation reproduction/variant necessitated by
unavailable comparable authoritative question-text metadata across all three
datasets.**

## Exact tensors and masking

For a target interaction, the model receives a left-padded tensor of its most
recent 49 prior interactions only. The target label is stored separately and
never enters the history tensor. A Boolean history mask excludes every PAD
position. This target-versus-prior representation is causally masked by
construction: there is no current or future key/value position to attend to.

The question embedding has width 64. As in the reference implementation, an
interaction is the concatenation of two copies of that embedding, gated by
incorrect/correct response state. A learned positional embedding of width 128
is added before projection to width 64. A single scaled dot-product attention
head produces

```text
alpha = masked_softmax(Q K^T / sqrt(64)).
```

For prior position `j` and target `i`, with elapsed time in hours,

```text
S_u      = softplus(rho_u) + 1e-6
T_ij     = exp(-DeltaHours_ij / S_u)
R_ij     = masked_softmax(Phi_ij + T_ij)
lambda   = sigmoid(eta)
beta_ij  = lambda * alpha_ij + (1 - lambda) * R_ij
```

The attended value is passed through residual/layer-normalization and a
width-64 ReLU feed-forward block, then to one logit. Masked softmax returns all
zeros for an empty history. PAD positions therefore contribute neither to
`alpha`, `R`, nor `beta`; targets, loss, and metrics contain no PAD values.

`eta` starts at zero, so `lambda=0.5`. Both `eta` and the student-specific
`rho_u` values are optimized only during training and are explicitly frozen
for validation and test.

## Timestamp normalization and memory initialization

The native timestamp unit was checked against the actual accepted source for
each dataset before building model inputs:

| Dataset | Verified native unit | Preprocessing conversion |
|---|---|---|
| ASSIST2017 | Unix seconds (`1096470301` observed) | `timestamp_ms = startTime * 1000` |
| Junyi | Unix seconds (`1536849900` observed) | `timestamp_ms = startTime * 1000` |
| EdNet-KT1 | Unix milliseconds, shifted for privacy (`1565096190868` observed) | `timestamp_ms = timestamp` |

All session stores persist `timestamp_ms`; RKT uses
`DeltaHours = (target_timestamp_ms - history_timestamp_ms) / 3,600,000`.
Negative deltas fail validation.

Each student's initial `S_u` is the median positive consecutive gap within
that student's training partition only. A student with no positive training
gap receives the global median of all positive training gaps; validation and
test never influence either value. On the three full stores, the global
training-only fallbacks and actual fallback counts are:

| Dataset | Global fallback (hours) | Students using fallback |
|---|---:|---:|
| ASSIST2017 | 0.0030555555555555557 | 0 / 1,388 |
| Junyi | 0.25 | 0 / 29,865 |
| EdNet-KT1 | 0.009909722222222223 | 0 / 116,548 |

The paper does not explicitly specify timestamp units, `S_u` initialization,
the details of trainable-parameter initialization for `S_u`, or the
initialization/learning rule for `lambda`. The choices above are the approved,
fully recorded definitions used by this benchmark.

## Directed Phi and leakage prevention

Phi is computed from 2-by-2 correctness contingency counts. With rows equal
to prior-question correctness and columns equal to target-question
correctness, counts are `n00,n01,n10,n11`, and

```text
Phi = (n11*n00 - n01*n10) /
      sqrt((n10+n11)(n00+n01)(n01+n11)(n00+n10)).
```

The ordered key is `(target_question, prior_question)`, so Phi is directed.
Only the latest occurrence of a prior question inside the target's rolling
49-interaction context contributes to its pair contingency. An absent pair or
zero denominator maps to `0.0`. No 0.8 threshold is applied: that threshold
belongs to the omitted Phi-plus-text relation.

Training uses deterministic five-fold student-level cross-fitting. Sorted
student IDs are permuted with NumPy's seeded generator (`seed=42`) and assigned
round-robin to folds 0--4. For a target in fold `f`, Phi comes from the cached
sum of training histories in the other four folds; consequently the target
student's entire fold, its own label, and its future interactions are absent.
Validation/test targets use the all-fold cache derived exclusively from the
training partition. The saved `student_folds.csv`, per-fold contingency files,
complement caches, and metadata make the optimization auditable without
changing these semantics.

The full deterministic fold sizes are:

| Dataset | Fold sizes | Balance |
|---|---|---|
| ASSIST2017 | 278, 278, 278, 277, 277 | differ by one |
| Junyi | 5,973, 5,973, 5,973, 5,973, 5,973 | exact |
| EdNet-KT1 | 23,310, 23,310, 23,310, 23,309, 23,309 | differ by one |

## Paper versus released-code deviations

These choices follow the paper and approved data contract where the released
code conflicts with them:

1. **Temporal decay:** paper form `exp(-DeltaHours/S_u)` replaces released
   `exp(-abs(raw_delta_t))`; units are explicitly hours.
2. **Student memory:** positive trainable `S_u=softplus(rho_u)+epsilon` is
   implemented and initialized only from training gaps; the released code has
   no per-student `S_u`.
3. **Width:** 64 from the paper, not the released default 200.
4. **Dropout:** 0.1 from the paper, not the released default 0.2.
5. **Position:** learned positional embeddings are enabled through length 50;
   the released executable path disables its positional option.
6. **Heads:** one head is used because the paper does not explicitly specify a
   count; the released five-head default is incompatible with width 64.
7. **Phi:** raw directed Phi is constructed from training-only histories with
   five-fold student cross-fitting rather than loading an opaque external
   relation matrix.
8. **Text relation:** cosine similarity over question text is omitted because
   comparable authoritative text is unavailable for all three datasets.
9. **Threshold:** the 0.8 Phi-plus-text threshold is not used in this
   performance-only variant.

The residual/layer-normalization and width-64 feed-forward path described in
the paper are retained even though the released single-layer executable does
not fully expose that paper path.

## Pre-training smoke gate

The bounded ASSIST2017 smoke used 20 students (four per fold), 10,409 training
targets, 1,931 validation targets, and 2,114 test targets. One seeded training
step, `[128,49]` input/Phi tensors, backward propagation, strict checkpoint
round-trip, frozen validation/test relation parameters, unique targets, and all
required metrics passed. These values are diagnostics, not final benchmark
results. Reproduce them with:

```bash
PYTHONPATH=. python scripts/prepare_rkt.py assist2017 \
  data/processed/assist2017/full/session_store \
  data/processed/assist2017/full/rkt_phi_smoke \
  --smoke-students-per-fold 4

PYTHONPATH=. python scripts/smoke_rkt.py \
  data/processed/assist2017/full/session_store \
  data/processed/assist2017/full/rkt_phi_smoke \
  experiments/rkt/assist2017/smoke
```

Automated tests cover timestamp conversion, fold determinism and exclusion,
causal/latest-prior Phi construction, undefined Phi, target uniqueness,
rolling length, tensor shapes, positive and train-only `S_u`, lambda
initialization/freeze, metrics, forward/backward, and checkpoint loading.

Full RKT training uses the project-wide validation ROC-AUC controller:
patience 5, `min_delta=0`, strict improvement, and best-checkpoint reload before
test. The controller may stop RKT early but cannot extend the authors'
repository ceiling of 300 epochs.
