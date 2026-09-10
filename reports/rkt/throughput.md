# Full-cache RKT throughput gate

Status: all three complete production Phi repositories loaded and executed on
the H100 MIG device with seed 42, batch size 128, rolling history 49, and
maximum gradient norm 10.

The first measurement performed one sparse-array binary search per example.
The accepted implementation groups each batch by the exact applicable cache:
training examples by excluded student fold, and validation/test examples by the
all-training cache. It then performs one vectorized search per group. Tests
prove every grouped value equals the former scalar lookup, including a batch
covering all five folds. The optimization changes neither pair direction nor
fold, target, history, missing-relation, or leakage semantics.

| Dataset | Cache load | Peak host RSS | Train targets/s | Eval targets/s | Estimated train | Estimated validation | Estimated epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASSIST2017 | 0.70 s | 1.75 GiB | 9,190 | 14,503 | 70.4 s | 8.7 s | 79.1 s |
| Junyi | 34.25 s | 17.60 GiB | 7,967 | 12,148 | 20.2 min | 3.5 min | 23.7 min |
| EdNet-KT1 | 60.38 s | 27.73 GiB | 7,561 | 11,411 | 1.98 h | 21.2 min | 2.34 h |

Each mode measured 20 deterministic random batches (2,560 targets) after one
training warmup batch. Timing includes full-cache lookup, dataset access,
collation, host-to-device transfer, and the appropriate model step. Training
includes forward, BCE, backward, norm-10 clipping, and Adam update; evaluation
includes forward and BCE without gradients. Projections exclude full-epoch
loader setup and checkpoint I/O and remain planning estimates, not results.

Before grouping, projected epochs were 96.3 seconds, 98.0 minutes, and 9.89
hours for ASSIST2017, Junyi, and EdNet respectively. The final per-dataset JSON
files contain raw seconds, target counts, cache-load time, host/GPU peaks, and
scope notes. No smoke or throughput metric is a scientific benchmark result.
