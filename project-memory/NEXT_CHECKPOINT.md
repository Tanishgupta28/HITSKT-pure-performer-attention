# Next checkpoint

Updated: 2026-09-10 UTC

## Last valid terminal checkpoint

Full ASSIST2017, Junyi, and EdNet preprocessing is validated. Lossless session
stores, variable-length token-budgeted batching, the hierarchical pure
Performer HiTSKT path, and its CUDA smoke gate are complete. The approved RKT
variant, train-only/cross-fitted Phi cache, timestamp normalization, `S_u` and
lambda behavior, all metrics, and ASSIST smoke gate are implemented. All 35
tests pass. Seed 42 is centralized. No full scientific training run is active
or complete.

## Exact next bounded action

Build the common seeded training/evaluation artifact runner and adapt DKT,
DKVMN, and SAKT to the accepted lossless target/split contract. Complete their
forward/backward/checkpoint/metric smoke gates, then implement full RKT cache
preparation with an efficiency benchmark before starting any of the 15 full
scientific experiments.

## Stop conditions

- Do not start the 15 experiments before the smoke-test gates pass.
- Do not substitute the supplied Riiid file or reduced `ednetnew.csv`.
- Do not silently sample EdNet or alter HiTSKT's Performer architecture.
- Do not report the RKT smoke diagnostics as scientific results.
- Stop if full Phi preparation exposes a material scalability issue that would
  require changing cross-fitting, history semantics, or retained targets.
