# Next checkpoint

Updated: 2026-09-17 UTC

## Last valid terminal checkpoint

Full ASSIST2017, Junyi, and EdNet preprocessing is validated. Lossless session
stores, variable-length token-budgeted batching, the hierarchical pure
Performer HiTSKT path, and its CUDA smoke gate are complete. The approved RKT
variant, train-only/cross-fitted Phi cache, timestamp normalization, `S_u` and
lambda behavior, all metrics, and ASSIST smoke gate are implemented. Production
runners for all five models share the common validation-AUC controller and pass
end-to-end fixture tests. RKT clipping at maximum norm 10 is explicitly
approved and recorded. All 60 tests pass. Seed 42 is centralized. All five
ASSIST2017 and all five Junyi model runs plus full RKT/EdNet are complete and
independently reproduced from their best checkpoints; the generated master
result has 12/15 rows. Full HiTSKT/EdNet completed at epoch 35, selected epoch
30 (validation AUC 0.7681769525193056), and produced test AUC
0.7693648364413692 over 13,429,870 targets. Fresh-process checkpoint replay
matched all seven metrics exactly and strict report validation passed.

## Exact next bounded action

DKVMN/EdNet has now completed and passed independent replay. The master
results contain 13/15 runs. Best epoch 1, validation AUC 0.6719177243615195,
test AUC 0.6741059915424829; stopped after epoch 6 with exactly five misses.
Run full SAKT/EdNet next, followed by DKT/EdNet, then generate the complete
15-run report. Keep 30-minute checks and defer pushes. SAKT must retain
context/history 100/99, batch 10, width 200, heads 5, dropout 0.2, Adam
LR 1e-5, gradient clip 10, ceiling 300, seed 42, patience 5/min_delta 0.
SAKT is now running as PID/SID 2535997 (parent 1); the saved configuration
passed assertions and is committed locally. Read its runner.pid at
`experiments/sakt/ednet_kt1/full/runner.pid` and continue 30-minute monitoring.

## Completed DKVMN/EdNet audit trail

Run full DKVMN/EdNet-KT1 using the unchanged repository profile, detached from
the controlling terminal, and monitor it every 30 minutes. Command:

`PYTHONPATH=. python scripts/train_baseline.py dkvmn ednet_kt1 data/processed/ednet_kt1/full/session_store experiments/dkvmn/ednet_kt1/full --workers 8`

It must retain context/history 200/199, batch size 32, Adam LR 0.001,
memory/key/value sizes 20/50/100, seed 42, ceiling 100, and common strict
validation-AUC patience 5. On completion, run `scripts/evaluate_checkpoint.py`
against the same store, validate curves, aggregate to 13/15, document, and
commit. Then run SAKT and DKT EdNet with their already-approved unchanged
profiles. Pushes remain deferred until the user restores the Tanish login.
The run is now active as PID/SID 2443018 with parent 1. Its initialized config
passed every assertion above. Read `experiments/dkvmn/ednet_kt1/full/runner.pid`
rather than trusting the historical PID, and preserve ignored `last_model.pt`
for exact resume once epoch checkpoints begin.
At 2026-09-16 09:40 UTC, epoch 1 completed with validation AUC
0.6719177243615195, checkpoint saved, patience zero. Its exact-resume payload
was verified and committed locally; epoch 2 is running. Continue 30-minute
checks through strict early stopping/ceiling, then independently replay test.
At 2026-09-16 14:00 UTC, epoch 2 validation AUC 0.6656104984379707 did not
improve. Best remains epoch 1, patience 1/5; epoch 3 is running. The epoch-2
logs/config are committed and the ignored exact-resume checkpoint is intact.
Latest 2026-09-16 18:01 UTC check: epoch 3 validation AUC
0.6511616713824253 did not improve; best remains epoch 1, patience 2/5,
and epoch 4 is running. Monitor session 19124 now checks every 30 minutes.
Latest 2026-09-16 22:01 UTC check: epoch 4 validation AUC
0.6500716087367917 did not improve; best remains epoch 1, patience 3/5,
and epoch 5 is running. Keep monitoring session 19124 every 30 minutes.
Latest 2026-09-17 02:01 UTC check: epoch 5 validation AUC
0.648117614809817 did not improve; best remains epoch 1, patience 4/5,
and epoch 6 is running. Another miss must trigger final best-checkpoint test.

## Completed HiTSKT/EdNet audit trail

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
archived run exactly. Latest 11:41 UTC snapshot: epochs 6–8 also matched;
best epoch 7 at AUC 0.7622619624598154. Epoch 8 AUC 0.7599432662688099
was the first non-improvement. Latest 13:01 UTC snapshot: best epoch 10
at AUC 0.764228728194286, patience zero; epoch 11 is running. A read-only
comparison proves all ten complete epoch JSON records (train/validation
metrics, target counts, seed, and checkpoint/patience flags) exactly equal
the archived run's first ten records. Latest 16:21 UTC snapshot: epoch 15
improved to best AUC 0.7668270365637668. Latest 17:41 UTC snapshot: best
epoch 17 at AUC 0.7672589703242605, patience zero; epoch 18 is running.
Latest 19:01 UTC snapshot: epoch 19 AUC 0.7672127018589197 did not improve;
best remains epoch 17, patience 2/5, and epoch 20 is running. The actual
best_model.pt was included in local commit 2d8ce3e. Latest 20:21 UTC snapshot:
epoch 21 improved to AUC 0.7675438123942788, resetting patience to zero;
epoch 22 is running. The new actual best checkpoint is committed locally;
last_model.pt
remains an ignored local exact-resume artifact and must not be removed.
Latest 21:21 UTC snapshot: epoch 22 is best at AUC 0.7677329409251583,
patience zero; epoch 23 is running. New tests and guards ensure an already
stopped checkpoint resumes directly into final best-checkpoint test evaluation,
without any extra train/validation epoch, across all five models.
Latest 23:01 UTC snapshot: epoch 25 AUC 0.7676484582020142 did not improve.
Best remains epoch 22, patience 3/5, and epoch 26 is running. Actual best
checkpoint is in local commit 1fee049; updated epoch-25 logs/memory are local.
Latest 23:41 UTC snapshot: epoch 26 improved to best AUC 0.7677920121661311,
patience zero; epoch 27 is running. New actual best checkpoint is committed
locally under Tanish identity. Continue the same ceiling-40/patience-5 protocol.
Latest 2026-09-16 01:41 UTC snapshot: epoch 29 improved to best AUC
0.7678508108666452, patience zero; epoch 30 is running. The actual new best
checkpoint and updated artifacts/memory are committed locally. Monitor
session 66396 remains active; continue its 20-minute process/epoch snapshots.
Latest 2026-09-16 03:01 UTC snapshot: best epoch 30 at AUC
0.7681769525193056; epoch 31 AUC 0.7679458417050529 did not improve.
Patience 1/5, epoch 32 is running. Actual epoch-30 best checkpoint and all
epoch-31 artifacts/memory are committed locally under the original identity.
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
  test passes; the full suite now has 60 passing tests. Use `--resume` after an
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
