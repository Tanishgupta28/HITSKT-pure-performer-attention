# Next checkpoint

Updated: 2026-09-10 UTC

## Last valid terminal checkpoint

Full ASSIST2017, Junyi, and EdNet preprocessing is validated. Lossless session
stores, variable-length token-budgeted batching, the hierarchical pure
Performer HiTSKT path, mask-safe loss/counts, and the full real-data batching
and CUDA benchmark are implemented locally. All 28 tests pass. No scientific
training run is active or complete.

## Exact next bounded action

Checkpoint and push the validated variable-length implementation. Then add the
common training/evaluation runner and complete DKT, DKVMN, SAKT, RKT, and
HiTSKT forward/backward/checkpoint/metric smoke gates before starting the 15
full scientific experiments.

## Stop conditions

- Do not start the 15 experiments before the smoke-test gates pass.
- Do not substitute the supplied Riiid file or reduced `ednetnew.csv`.
- Do not silently sample EdNet or alter HiTSKT's Performer architecture.
