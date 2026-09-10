# Next checkpoint

Updated: 2026-09-10 UTC

## Last valid terminal checkpoint

Full ASSIST2017, Junyi, and EdNet preprocessing is validated. Lossless session
stores, variable-length token-budgeted batching, the hierarchical pure
Performer HiTSKT path, and its CUDA smoke gate are complete. The approved RKT
variant, train-only/cross-fitted Phi cache, timestamp normalization, `S_u` and
lambda behavior, all metrics, and ASSIST smoke gate are implemented. All 40
tests pass. Seed 42 is centralized. No full scientific training run is active
or complete.

## Exact next bounded action

Obtain one common validation-AUC early-stopping patience for the full benchmark.
Then build the common seeded training/evaluation artifact runner around the
now-validated DKT/DKVMN/SAKT rolling adapters. Before launching full runs,
account for the measured compute-only lower bounds, particularly DKVMN's
81.614-hour EdNet epoch, without silently changing batch size, context,
targets, or dataset scope.

## Stop conditions

- Do not start the 15 experiments before the smoke-test gates pass.
- Do not substitute the supplied Riiid file or reduced `ednetnew.csv`.
- Do not silently sample EdNet or alter HiTSKT's Performer architecture.
- Do not report the RKT smoke diagnostics as scientific results.
- Stop if full Phi preparation exposes a material scalability issue that would
  require changing cross-fitting, history semantics, or retained targets.
- Do not invent an early-stopping patience: it materially controls total
  runtime and is absent from the legacy repository.
