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
approved and recorded. All 55 tests pass. Seed 42 is centralized. All five
ASSIST2017 and all five Junyi model runs plus full RKT/EdNet are complete and
independently reproduced from their best checkpoints; the generated master
result has 11/15 rows.

## Exact next bounded action

Monitor the clean full HiTSKT/EdNet rerun every 20 minutes. It was launched
2026-09-15 UTC in a separate session with `setsid nohup`, PID 2267845; read
`experiments/hitskt/ednet_kt1/full/runner.pid` and verify the actual process.
The first plain-nohup launch exited before initialization; the separate-session
launch is live. At 07:01 UTC, epoch 1 completed with validation AUC
0.7527808881272077 (identical to the old first epoch), checkpoint saved,
patience zero. The actual checkpoint was loaded and verified to include all
four RNG families, Adam state, and exact early-stopping state. At 07:41 UTC,
epoch 2 completed with validation AUC 0.7573268796700343 (also identical to
the old run), checkpoint saved, patience zero. At 08:21 UTC, epoch 3 completed
at validation AUC 0.7588930018050819 (also identical), checkpoint saved,
patience zero. Latest 09:41 UTC snapshot: epochs 4 and 5 also matched the
archived run exactly, latest best epoch 5 at AUC 0.7612169898244615,
patience zero. Epoch 6 is running.
Monitor terminal session 66396 emits process/epoch snapshots every
20 minutes; if that monitor exits, the detached trainer remains independent.
The earlier
epoch-20 run is archived under
`experiments/hitskt/ednet_kt1/interrupted_no_rng_resume_epoch20_20260914/`;
it has no final test or success marker and is not a scientific result.

The active Performer call path was reverified with four CausalLinearAttention
modules, no quadratic-attention modules, and nine focused tests before the
original launch. The restart changes only runner durability: model, optimizer,
RNG, and patience state are now saved. An exact interruption/resume equality
  test passes; the full suite now has 55 passing tests. Use `--resume` after an
unexpected process exit; never silently overwrite an incomplete directory.

When HiTSKT completes, independently replay the best-checkpoint test metrics,
validate artifacts, aggregate to 12/15, document and commit. Then run the
remaining full EdNet DKT, DKVMN, and SAKT benchmarks with unchanged established
profiles. Keep all downstream final-report tasks in prompt.txt in scope.

Reusable replay command is now implemented and tested across all three model
families: `PYTHONPATH=. python scripts/evaluate_checkpoint.py
experiments/hitskt/ednet_kt1/full data/processed/ednet_kt1/full/session_store
--workers 8 --output experiments/hitskt/ednet_kt1/full/independent_evaluation.json`.
The production replay must wait for `_SUCCESS`; it reloads the best checkpoint
and fails on any exact metric mismatch. RKT additionally needs its prepared
Phi cache via `--prepared-root`.

`scripts/generate_report.py` now validates all seven table metrics, epochs,
best-checkpoint selection, and CSV/JSON equality before generating tables and
curves. All 11 current production artifacts pass the read-only validator.
The generator requires 15 completed runs by default; invoke it only after the
matrix is complete. Its synthetic full-matrix tests verify seven CSV tables,
15 curve figures plus three comparison figures (PNG/SVG), and source hashes.

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
