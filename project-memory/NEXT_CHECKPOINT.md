# Next checkpoint

Updated: 2026-09-15 UTC

## Last valid terminal checkpoint

Full ASSIST2017, Junyi, and EdNet preprocessing is validated. Lossless session
stores, variable-length token-budgeted batching, the hierarchical pure
Performer HiTSKT path, and its CUDA smoke gate are complete. The approved RKT
variant, train-only/cross-fitted Phi cache, timestamp normalization, `S_u` and
lambda behavior, all metrics, and ASSIST smoke gate are implemented. Production
runners for all five models share the common validation-AUC controller and pass
end-to-end fixture tests. RKT clipping at maximum norm 10 is explicitly
approved and recorded. All 52 tests pass. Seed 42 is centralized. All five
ASSIST2017 and all five Junyi model runs plus full RKT/EdNet are complete and
independently reproduced from their best checkpoints; the generated master
result has 11/15 rows.

## Exact next bounded action

Monitor the clean full HiTSKT/EdNet rerun every 20 minutes. It was launched
2026-09-15 UTC in a separate session with `setsid nohup`, PID 2267845; read
`experiments/hitskt/ednet_kt1/full/runner.pid` and verify the actual process.
The first plain-nohup launch exited before initialization; the separate-session
launch is live and has emitted the correct CUDA/seed-42 config. The earlier
epoch-20 run is archived under
`experiments/hitskt/ednet_kt1/interrupted_no_rng_resume_epoch20_20260914/`;
it has no final test or success marker and is not a scientific result.

The active Performer call path was reverified with four CausalLinearAttention
modules, no quadratic-attention modules, and nine focused tests before the
original launch. The restart changes only runner durability: model, optimizer,
RNG, and patience state are now saved. An exact interruption/resume equality
test passes; the full suite has 52 passing tests. Use `--resume` after an
unexpected process exit; never silently overwrite an incomplete directory.

When HiTSKT completes, independently replay the best-checkpoint test metrics,
validate artifacts, aggregate to 12/15, document and commit. Then run the
remaining full EdNet DKT, DKVMN, and SAKT benchmarks with unchanged established
profiles. Keep all downstream final-report tasks in prompt.txt in scope.

Original commit identity is Tanish Gupta / GitHub noreply ID 148684341.
The current gh login is Fyxod and cannot access the original private remote;
the user explicitly instructed retaining Tanish identity and remote, continuing
local commits, and deferring pushes until they log in. Do not change remotes,
create another repository, or repeatedly retry pushes before that change.

## Stop conditions

- Do not start the 15 experiments before the smoke-test gates pass.
- Do not substitute the supplied Riiid file or reduced `ednetnew.csv`.
- Do not silently sample EdNet or alter HiTSKT's Performer architecture.
- Do not report the RKT smoke diagnostics as scientific results.
- Stop if full Phi preparation exposes a material scalability issue that would
  require changing cross-fitting, history semantics, or retained targets.
- Do not change the approved common patience 5 or `min_delta=0`.
