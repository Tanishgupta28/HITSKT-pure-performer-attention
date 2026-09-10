# Next checkpoint

Updated: 2026-09-10 UTC

## Last valid terminal checkpoint

Full ASSIST2017, Junyi, and EdNet preprocessing is validated. Lossless session
stores, variable-length token-budgeted batching, the hierarchical pure
Performer HiTSKT path, and its CUDA smoke gate are complete. The approved RKT
variant, train-only/cross-fitted Phi cache, timestamp normalization, `S_u` and
lambda behavior, all metrics, and ASSIST smoke gate are implemented. Production
runners for all five models share the common validation-AUC controller and pass
end-to-end fixture tests. RKT clipping at maximum norm 10 is explicitly
approved and recorded. All 49 tests pass. Seed 42 is centralized. All five
ASSIST2017 model runs are complete and
independently reproduced from their best checkpoints; the generated master
result has 5/15 rows.

## Exact next bounded action

Checkpoint the verified full DKVMN/ASSIST2017 result and its updated throughput
measurements, then launch full RKT/Junyi as the lowest projected complete
Junyi run under its approved model-specific configuration and common
early-stopping protocol. Monitor durable epoch artifacts and independently
verify the best checkpoint before acceptance.
Account for the measured compute-only lower bounds, particularly DKVMN's
81.614-hour EdNet epoch, without silently changing batch size, context,
targets, or dataset scope.

## Stop conditions

- Do not start the 15 experiments before the smoke-test gates pass.
- Do not substitute the supplied Riiid file or reduced `ednetnew.csv`.
- Do not silently sample EdNet or alter HiTSKT's Performer architecture.
- Do not report the RKT smoke diagnostics as scientific results.
- Stop if full Phi preparation exposes a material scalability issue that would
  require changing cross-fitting, history semantics, or retained targets.
- Do not change the approved common patience 5 or `min_delta=0`.
