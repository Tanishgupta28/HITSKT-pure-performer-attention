# Variable-length HiTSKT batching and memory benchmark

Status: implemented and validated on 2026-09-10 against the complete processed
ASSIST2017, Junyi, and EdNet-KT1 stores.

## Scientific contract

- The accepted 10-hour sessions are immutable units. Actions are neither
  truncated nor split/chunked.
- Every retained session after a student's first becomes one target example,
  and every real action in that target contributes one supervised label.
- Each example uses the most recent 15 strictly earlier sessions, preserving
  the legacy `session_size=16` hierarchy without the legacy action cap. A
  validation target therefore sees only chronological train history plus any
  earlier validation history; test follows the same past-only rule through
  train and validation.
- Historical sessions are flattened for the Action Encoder, target sequences
  are padded independently, and encoded histories are repacked for the Session
  Encoder. All padding is local to the current batch.
- A stable length sort forms buckets of 512 examples. Training can shuffle
  buckets and the contents of each bucket deterministically by epoch. Greedy
  packing uses a configured 32,768 padded-token budget and batch-size cap 64.
  A single indivisible example may exceed the budget; it remains complete in a
  singleton batch.
- Question and skill PAD are `0`; correctness PAD is `2`, EOS is `3`, and BOS
  is `4`. Attention masks exclude PAD. EOS is encoded but the metric mask
  contains real targets only. Correctness inputs are shifted `[BOS, y0, ...]`,
  preventing the label at time `t` from leaking into its own prediction.

The model path remains hierarchical: Action Encoder, Session Encoder,
Correct/Padding Encoder, Decoder, and prediction head. All four sequence stages
use causal multi-head `ELU(x)+1` linear attention computed with prefix sums of
`K` and `outer(K,V)`. No stage computes a standard softmax or an `L x L`
attention matrix. Positional encodings are generated at runtime, so they impose
no fixed action length.

## Lossless store validation

| Dataset | Interactions | Students | Sessions | Maximum session | Truncated | Chunked |
|---|---:|---:|---:|---:|---:|---:|
| ASSIST2017 | 885,335 | 1,388 | 12,402 | 938 | 0 | 0 |
| Junyi | 14,660,217 | 29,865 | 600,154 | 3,924 | 0 | 0 |
| EdNet-KT1 | 81,940,867 | 116,548 | 2,577,988 | 13,080 | 0 | 0 |

The store builder reconciles these counts with the independently validated
Parquet summaries, checks session numbering and student boundaries, and records
`action_truncation=false` and `action_chunking=false` in `metadata.json`.

## Full batch plans

The following are actual plans over every usable post-first target, not a
sample. The difference between training session counts and training example
counts is exactly one non-targetable first session per retained student.

| Dataset / split | Examples | Batches | Maximum batch used | Maximum padded tokens | Singleton batches over budget |
|---|---:|---:|---:|---:|---:|
| ASSIST / train | 7,060 | 164 | 64 | 32,768 | 0 |
| ASSIST / validation | 1,977 | 82 | 64 | 32,766 | 0 |
| ASSIST / test | 1,977 | 98 | 64 | 32,768 | 0 |
| Junyi / train | 351,277 | 8,828 | 64 | 32,768 | 0 |
| Junyi / validation | 109,506 | 3,645 | 64 | 32,768 | 0 |
| Junyi / test | 109,506 | 3,658 | 64 | 35,341 | 1 |
| EdNet / train | 1,511,786 | 54,750 | 64 | 82,499 | 208 |
| EdNet / validation | 474,827 | 23,188 | 64 | 121,468 | 154 |
| EdNet / test | 474,827 | 24,248 | 64 | 196,280 | 321 |

Thus the actual maximum configured batch size used is 64, the normal token
budget is 32,768, and the actual maximum is the indivisible 196,280-token EdNet
singleton. These exceptions are explicit rather than silently truncated.

## H100 MIG memory benchmark

Hardware: NVIDIA H100 80GB HBM3 exposed as one 3g.40gb MIG device; PyTorch
2.4.0a0+f70bd71a48.nv24.06; float32; one Action, Session, Correct/Padding, and
Decoder layer; forward, masked BCE loss, and backward. ASSIST uses width 256,
four heads, and feed-forward width 2,048. Junyi/EdNet use width 128, two heads,
and feed-forward width 1,024. Dropout is 0.1.

| Dataset worst planned batch | Batch | Padded tokens | Real targets | Host tensors | CUDA allocated peak | CUDA reserved peak | Forward+backward |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASSIST2017 | 15 | 32,768 | 2,414 | 863,694 B | 7.066 GiB | 8.686 GiB | 0.850 s |
| Junyi | 1 | 35,341 | 5 | 883,482 B | 4.593 GiB | 5.621 GiB | 0.368 s |
| EdNet-KT1 | 1 | 196,280 | 48 | 4,907,290 B | 24.989 GiB | 30.861 GiB | 0.532 s |

The old fixed loader is not a correctness-equivalent memory baseline: its
small allocation is achieved by dropping 34.33% of retained ASSIST actions,
29.40% of Junyi actions, and 42.96% of EdNet actions. For context, padding the
same measured batches to each dataset's global complete-session maximum would
consume 225,600, 62,816, and 209,312 token slots respectively, versus 32,768,
35,341, and 196,280 dynamically padded slots. We therefore report real dynamic
CUDA peaks and token-slot comparisons rather than presenting the scientifically
invalid truncating loader as an equivalent before/after GPU benchmark.

## Verification coverage

Automated tests cover heterogeneous lengths, exact dynamic shapes, history and
target padding masks, direct causal-prefix invariance, BOS/EOS target alignment,
long-session singleton behavior, no PAD/EOS contribution to loss or metrics,
forward/backward gradients through every hierarchy stage, chronological rolling
history, complete-session persistence, output shapes, and strict checkpoint
round-trip loading. `python -m pytest -q` passes all 28 tests.
