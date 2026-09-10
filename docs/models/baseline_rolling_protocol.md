# DKT, DKVMN, and SAKT rolling-target protocol

The three baseline adapters preserve the architectures and hyperparameters in
the continuation repository while replacing its non-overlapping sequence
chunks with the user-approved rolling target-once construction. This
standardizes evaluation without claiming identical model contexts or model
hyperparameters.

## Implementation provenance

- DKT continues from `Other_models/dkt.py`, whose repository source note points
  to <https://github.com/THUwangcy/HawkesKT>. The clean adapter is
  `ktbench/models/baselines.py::DKT`.
- DKVMN continues from `Other_models/dkvmn.py`, whose source note points to
  <https://github.com/jennyzhang0215/DKVMN>. The adapter retains dynamic
  key/value memory reads and writes while keeping writes differentiable.
- SAKT continues from `Other_models/sakt.py`, whose source note points to
  <https://github.com/theophilee/learner-performance-prediction>. The adapter
  retains question/skill interaction embeddings, five-head self-attention, and
  learned clipped relative key/value positions.

The legacy files remain for provenance. Final experiments must use the clean
adapters and shared rolling data contract, not their old fixed arrays or
non-overlapping chunk loaders.

The adapters also repair execution defects without changing the named model
components: DKT derives real sequence lengths from the PAD mask; DKVMN does not
wrap each updated value memory in `Parameter(memory.data)` (which detached the
write path); and SAKT gates its duplicated interaction embedding with binary
correctness rather than the legacy loader's incremented padding encoding.
SAKT's full causal matrix is expressed equivalently as one target query over
strictly prior keys because the rolling dataset constructs one target at a
time. These fixes are covered by backward, PAD-invariance, and target-label
exclusion tests.

## Preserved configurations

| Setting | DKT | DKVMN | SAKT |
|---|---:|---:|---:|
| Context including target | 200 | 200 | 100 |
| Strictly prior history | 199 | 199 | 99 |
| Repository epoch ceiling | 200 | 100 | 300 |
| Batch size | 20 | 32 | 10 |
| Learning rate | 0.0002 | 0.001 | 0.00001 |
| Optimizer | Adam | Adam | Adam |
| Weight decay | 0 | 0 | 0 |
| Dropout | 0.1 configured | 0 | 0.2 |
| Gradient clipping | none | 50 | 10 |

DKT uses embedding width 64, hidden width 64, and one LSTM layer. The legacy
code records dropout 0.1 but does not pass it into its one-layer LSTM, so its
effective recurrent dropout remains zero; both facts are recorded in every
configuration.

DKVMN uses question/KC embedding width 50, interaction embedding width 100, 20
memory slots, key width 50, value width 100, and final width 50. As in the
repository loader, the supplied interaction-level/composite skill is the
memory-addressing item.

SAKT uses width 200, one layer, five heads, clipped relative-position range 10,
and dropout 0.2. It uses both question and supplied skill identities, matching
the repository architecture.

These values apply unchanged to ASSIST2017, Junyi, and EdNet. Vocabulary-sized
embeddings and outputs cause parameter counts to differ by dataset; that is a
data-interface consequence, not dataset-specific tuning.

## Rolling target and masking contract

For every retained student interaction in a split, `RollingTargetDataset`
creates exactly one target. Its key/value history consists only of the latest
199 or 99 interactions strictly before that target and may roll across earlier
session boundaries for the same student. Validation can see earlier training
and validation history; test can see earlier training, validation, and test
history. Neither the target response nor any future event enters its input.

The first interaction remains a target with empty history. Batches pad only to
the longest history present in that batch. The Boolean history mask prevents
PAD from changing LSTM summaries, DKVMN memory writes, or SAKT attention. Loss
and metrics operate on the one genuine target label per example, so PAD is not
a candidate target. Histories retain the deterministic chronological ordering
already validated in each session store.

`RollingLengthBatchSampler` groups similar history lengths within bounded
4,096-target buckets. It never constructs an EdNet-sized random permutation;
bucket and within-length ordering use the centralized seed 42. Each epoch still
visits every target exactly once.

## Real-data smoke results

All nine model/dataset interfaces passed one real-data forward/backward update,
dynamic-PAD invariance, required history-cap exercise, strict checkpoint
round-trip, and bounded validation/test metric computation. The smoke metrics
are diagnostics and are not final scientific results.

| Dataset | Full train targets | Full validation targets | Full test targets | DKT history | DKVMN history | SAKT history |
|---|---:|---:|---:|---:|---:|---:|
| ASSIST2017 | 647,283 | 125,800 | 112,252 | 199 | 199 | 99 |
| Junyi | 9,647,042 | 2,556,773 | 2,456,402 | 199 | 199 | 99 |
| EdNet-KT1 | 53,990,022 | 14,520,975 | 13,429,870 | 199 | 199 | 99 |

The counts sum exactly to the full retained interaction totals because every
valid split event is one target. Smoke configurations, data counts, logs, and
results are under `experiments/{dkt,dkvmn,sakt}/{dataset}/smoke/`; every file
records seed 42 and the exact vocabulary-dependent parameter count.

Reproduce all three models for one dataset with:

```bash
PYTHONPATH=. python scripts/smoke_baselines.py assist2017 \
  data/processed/assist2017/full/session_store experiments
```

Replace the dataset/store pair with `junyi` or `ednet_kt1` for the other
validated commands. Automated tests additionally prove target uniqueness,
rolling caps, chronology, future exclusion, deterministic complete batching,
PAD invariance, target-label exclusion, backward propagation, metric target
counts, and checkpoint loading.

Full experiments select checkpoints and stop on validation ROC-AUC using the
common approved protocol: patience 5, `min_delta=0`, and strict improvement.
The best checkpoint is reloaded before final test evaluation. Early stopping
never extends the DKT/DKVMN/SAKT ceilings of 200/100/300 epochs. Smoke configs
record the same protocol even though a one-step smoke cannot trigger it.

## Full-scale throughput gate

A real H100 MIG 3g.40gb benchmark repeated full-history forward, loss,
backward, clipping where configured, and Adam updates at each repository batch
size. The measurements are compute-only lower bounds: data loading, validation,
and checkpoint I/O are excluded.

| Dataset | DKT h/epoch | DKVMN h/epoch | SAKT h/epoch |
|---|---:|---:|---:|
| ASSIST2017 | 0.099 | 0.982 | 0.063 |
| Junyi | 1.457 | 14.817 | 0.879 |
| EdNet-KT1 | 8.169 | 81.614 | 4.953 |

The exact device, seed, batch counts, timings, exclusions, and peak allocations
are versioned in
[`reports/batching/baseline_rolling_throughput.json`](../../reports/batching/baseline_rolling_throughput.json).
Memory is not the limiting resource. DKVMN's 199 sequential memory writes at
batch size 32 dominate runtime. A bounded `torch.compile` trial was interrupted
after more than three minutes without completing its first graph; it did not
produce a validated speedup and is not enabled in the accepted path.

These measurements do not authorize sampling, larger batches, shorter
histories, non-overlapping chunks, or architecture changes. They establish the
cost that must be considered when fixing a common validation-AUC patience for
the full runs.
