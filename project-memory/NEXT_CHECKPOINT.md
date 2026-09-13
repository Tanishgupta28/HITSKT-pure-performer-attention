# Next checkpoint

Updated: 2026-09-13 UTC

## Last valid terminal checkpoint

Full ASSIST2017, Junyi, and EdNet preprocessing is validated. Lossless session
stores, variable-length token-budgeted batching, the hierarchical pure
Performer HiTSKT path, and its CUDA smoke gate are complete. The approved RKT
variant, train-only/cross-fitted Phi cache, timestamp normalization, `S_u` and
lambda behavior, all metrics, and ASSIST smoke gate are implemented. Production
runners for all five models share the common validation-AUC controller and pass
end-to-end fixture tests. RKT clipping at maximum norm 10 is explicitly
approved and recorded. All 51 tests pass. Seed 42 is centralized. All five
ASSIST2017 and all five Junyi model runs plus full RKT/EdNet are complete and
independently reproduced from their best checkpoints; the generated master
result has 11/15 rows.

## Exact next bounded action

Checkpoint the verified full RKT/EdNet result, then reverify the active
HiTSKT-to-Performer call path and launch full HiTSKT/EdNet. Use the complete
EdNet store, lossless variable-length token-budgeted batching, seed 42, and the
common validation-AUC early-stopping protocol. Monitor durable resumable
artifacts and independently verify its best checkpoint. Do not silently alter
the Performer architecture, batching semantics, targets, or dataset scope.

## Stop conditions

- Do not start the 15 experiments before the smoke-test gates pass.
- Do not substitute the supplied Riiid file or reduced `ednetnew.csv`.
- Do not silently sample EdNet or alter HiTSKT's Performer architecture.
- Do not report the RKT smoke diagnostics as scientific results.
- Stop if full Phi preparation exposes a material scalability issue that would
  require changing cross-fitting, history semantics, or retained targets.
- Do not change the approved common patience 5 or `min_delta=0`.
