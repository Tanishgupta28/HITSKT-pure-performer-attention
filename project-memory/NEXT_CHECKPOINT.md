# Next checkpoint

Updated: 2026-09-23 UTC

## Last valid terminal checkpoint

Full ASSIST2017, Junyi, and EdNet preprocessing is validated. Lossless session
stores, variable-length token-budgeted batching, the hierarchical pure
Performer HiTSKT path, and its CUDA smoke gate are complete. The approved RKT
variant, train-only/cross-fitted Phi cache, timestamp normalization, `S_u` and
lambda behavior, all metrics, and ASSIST smoke gate are implemented. Production
runners for all five models share the common validation-AUC controller and pass
end-to-end fixture tests. RKT clipping at maximum norm 10 is explicitly
approved and recorded. All 60 tests pass. Seed 42 is centralized. All five
ASSIST2017 and all five Junyi model runs plus full DKVMN, RKT, and HiTSKT/EdNet are complete and
independently reproduced from their best checkpoints; the generated master
result has 13/15 rows. Full HiTSKT/EdNet completed at epoch 35, selected epoch
30 (validation AUC 0.7681769525193056), and produced test AUC
0.7693648364413692 over 13,429,870 targets. Fresh-process checkpoint replay
matched all seven metrics exactly and strict report validation passed.

## Exact next bounded action

Latest user decision: **Keep it public and push**. Public/private hold lifted;
push queued and future relevant commits to capstone-gpu (redirects to canonical
Tanishgupta28/HITSKT-pure-performer-attention). Do not change visibility or
stage unrelated README edits. Continue DKT66426 and SAKT replay81202 monitors.

Latest Sep23 12:11UTC: user explicitly overrode sequential scheduling; DKT full
EdNet now runs CUDA PID3412383 with validated original profile and packed64.
Monitor66426 checks every2h (next14:11UTC). SAKT replay3368228 still active;
monitor81202 checks next13:43UTC, active functions cell39 waits on it. Continue
both without ending the turn for informational questions. On successful replay,
aggregate14/15 and commit verification; keep DKT running through standard stop.
Prior instructions to wait for replay before DKT are superseded.

Latest Sep23 07:42UTC: SAKT_SUCCESS after18epochs, best13/testAUC0.7630078573122142;
all epoch/stop/checkpoint/target records verified. Detached independent replay
PID3368228 is running with reference transport; output independent_evaluation.json.
Wait on two-hour replay monitor81202, first check~09:43UTC.
Wait on replay before14/15 aggregation and DKT launch. SAKT monitor87358 finished
normally. Preserve two-hour check cadence, all scientific settings and push hold.

Latest Sep23 01:41UTC: epoch17 complete/verified, val0.7614665404487208,
best13/patience4. Epoch18 runs unchanged PID3100394. Epoch17 took13.2189h
(shared-workload caveat). Wait on monitor87358, next03:41UTC. If epoch18 fails
to improve, ensure stop at5misses and best-checkpoint test, then independent
replay and14/15 aggregation before DKT. No extra training after stop criterion.

Latest Sep23 00:38 UTC: still epoch17, now >12h, process3100394 active/no
console error. Epoch16 checkpoint remains intact. User reports additionalGPU
workloads; contention plausible, not measurable with current telemetry access.
Preserve current progress and other jobs. Continue monitor87358 (next01:41UTC),
verify new checkpoints when available. No training/settings changes authorized.

Latest 2026-09-22 15:55 UTC: epoch16 verified complete, val AUC
0.760941857184205, best13/patience3. Epoch17 running PID3100394. Actual first
optimized full-epoch duration6.1031h vs original average7.8188h; user informed
that ten-hour uptime includes two epochs. Report documents observed1.2811x
speedup caveats. Continue monitor87358, next scheduled ~17:41 UTC; verify and
commit subsequent epochs, independent final replay on success, DKT afterward.

**Latest 2026-09-22 05:40 UTC:** user authorized discarding the unfinished epoch.
Optimized packed64 transport is now running on CUDA, PID/SID 3100394, resumed
from completed epoch 15; epoch 16 restarted, best13/patience2 preserved. All75
tests passed including exact production transport and checkpoint continuation.
Backups and audit in `reports/performance/README.md`. Replacement two-hour
monitor terminal87358 started ~05:42 UTC, first output ~07:42. Observe first completed
optimized epoch, verify target counts/checkpoint/protocol and actual elapsed
time, then continue remaining SAKT and independent final evaluation. DKT next;
packed64 is tested for DKT too. Keep push hold and preserve user README edits.
The following pre-deployment instructions are historical and superseded.

Report the completed conservative speedup experiment to the user before any
production switch. `reports/performance/README.md` records actual repeated
1.84x training / 4.38x validation short-window gains with bitwise-identical
tested model/Adam/RNG/output/metric states. All 70 tests pass. No production
trainer/model/loader was changed, and the live CUDA job/progress is intact.
The candidate only batches transport for 64 unchanged size-10 minibatches and
defers metric copies; it does not enlarge optimizer batches. A production
handoff without losing the active epoch has not yet been established. Do not
kill/restart the live job to adopt this candidate. Continue two-hour monitoring
and local commits; repository visibility decision is still pending.

Latest 2026-09-22 04:11 UTC: SAKT has completed epochs 14 and 15 without
improvement, best epoch 13 at validation AUC 0.7623242165859992, patience 2/5;
epoch 16 is running. All fifteen target counts and actual checkpoint/RNG/
optimizer/patience state passed verification. Continue two-hour monitor 39203
and local checkpoint commits. The user's latest "continue" does not resolve
public/private visibility; retain the push hold below. Preserve the unrelated
README working-tree edit. DKT and final report remain pending as before.

**New visibility decision required:** the existing GitHub redirect resolves to
`Tanishgupta28/HITSKT-pure-performer-attention`, now verified public. Earlier
authorization specified private. Hold pushes and ask whether to use the public
repository or restore private visibility; do not change visibility unilaterally.
Keep local commits and the existing CUDA trainer intact.

Latest 2026-09-21 13:47 UTC: SAKT epoch 13 improved validation AUC to
0.7623242165859992; best epoch 13, patience 0/5, epoch 14 running.
Thirteen epoch target counts and actual best/last checkpoint, optimizer, RNG,
and patience passed verification. Continue two-hour monitor terminal 39203.

Latest 2026-09-21 05:47 UTC snapshot: SAKT epoch 12 validation AUC
0.7620451155493354 did not improve; best remains epoch 10 at
0.7621860047840096, patience 2/5, epoch 13 is running. Twelve epoch target
counts, actual last checkpoint/RNG/optimizer/patience, and unchanged best
checkpoint SHA passed verification. Continue two-hour monitor terminal 39203,
checkpoint pushes, and the unchanged training/stopping protocol.

Latest 2026-09-20 21:47 UTC snapshot: SAKT epoch 11 validation AUC
0.761805075443002 did not improve; best remains epoch 10 at
0.7621860047840096, patience 1/5, epoch 12 is running. Eleven epoch target
counts, actual last checkpoint/RNG/optimizer/patience, and unchanged best
checkpoint SHA passed verification. Continue two-hour monitor terminal 39203;
commit/push updated logs and memory. Do not start DKT before SAKT completes
and its final best-checkpoint metrics are independently reproduced.

**Latest instruction, 2026-09-20 19:47 UTC:** check once every two hours,
superseding all earlier cadences below. SAKT PID/SID 2535997 is still active
in epoch 11, best epoch 10, validation AUC 0.7621860047840096, patience 0/5.
The old hourly monitor has exited; use two-hour monitor terminal 39203. Preserve the
unchanged CUDA run, verify/push new epoch artifacts, independently replay on
completion, then proceed to DKT and the complete 15-run report.

**Historical instruction, earlier 2026-09-20:** check once per hour, superseding earlier
cadences in the historical notes below. Use monitor terminal 70233; old monitor
10273 was stopped independently of the unchanged trainer PID/SID 2535997.
Latest 15:57 UTC snapshot: SAKT epoch 10 improved validation AUC to
0.7621860047840096, best epoch 10, patience 0/5; epoch 11 is running.
All ten target counts and checkpoint/RNG/optimizer/early-stopping state passed
verification. Commit/push the new checkpoint, artifacts, and memory, then
continue hourly checks. DKT still waits for SAKT completion and independent
best-checkpoint replay; no model, batch size, or stopping-rule change is allowed.

DKVMN/EdNet has now completed and passed independent replay. The master
results contain 13/15 runs. Best epoch 1, validation AUC 0.6719177243615195,
test AUC 0.6741059915424829; stopped after epoch 6 with exactly five misses.
Run full SAKT/EdNet next, followed by DKT/EdNet, then generate the complete
15-run report. Keep 30-minute checks and checkpoint pushes. SAKT must retain
context/history 100/99, batch 10, width 200, heads 5, dropout 0.2, Adam
LR 1e-5, gradient clip 10, ceiling 300, seed 42, patience 5/min_delta 0.
SAKT is now running as PID/SID 2535997 (parent 1); the saved configuration
passed assertions and is committed locally. Read its runner.pid at
`experiments/sakt/ednet_kt1/full/runner.pid` and continue 30-minute monitoring.
At 2026-09-17 14:52 UTC SAKT epoch 1 was complete, validation AUC
0.7470684949420671, checkpoint saved, patience zero; epoch 2 is running.
The actual best/last checkpoint state, optimizer, all four RNG families, seed,
target counts, and early-stopping state passed read-only verification.
Monitor terminal session 65227 emits snapshots every 30 minutes; it is separate
from the detached training process. DKT must wait for SAKT completion and
independent best-checkpoint replay.
Latest 2026-09-17 22:22 UTC snapshot: SAKT epoch 2 improved to validation
AUC 0.7515455799702323, best epoch 2, patience zero; epoch 3 is running.
The actual new best checkpoint and all epoch-2 artifacts/memory are committed
locally under the original Tanish identity. Keep the unchanged profile.
Latest 2026-09-18 05:52 UTC snapshot: SAKT epoch 3 improved validation AUC
to 0.7542712165233839, best epoch 3, patience zero; epoch 4 is running.
The new best/last checkpoint state and all three target counts passed
verification. Artifacts and memory are committed locally; continue the same
30-minute monitoring and unchanged experiment profile.
Latest 2026-09-18 13:22 UTC snapshot: SAKT epoch 4 improved validation AUC
to 0.7562697781063842, best epoch 4, patience zero; epoch 5 is running.
Actual checkpoint/RNG/optimizer state and all four epoch target counts passed
verification. The new best checkpoint, logs, and memory are committed locally.
Latest 2026-09-18 22:22 UTC snapshot: SAKT epoch 5 improved validation AUC
to 0.7575770754914868, best epoch 5, patience zero; epoch 6 is running.
The new actual checkpoint, RNG/optimizer state, and all five target counts
passed verification. Checkpoint/logs/memory are committed locally; keep the
same 30-minute checks and unchanged profile.
Latest 2026-09-19 07:09 UTC snapshot: SAKT epoch 6 improved validation AUC
to 0.7592312015241357, best epoch 6, patience zero; epoch 7 is running.
All six target counts and actual checkpoint/RNG/optimizer state passed
verification. Trainer PID 2535997 was not interrupted. Monitoring session
65227 expired; replacement session 10273 checks every 30 minutes.
Latest 2026-09-19 12:10 UTC snapshot: SAKT epoch 7 improved validation AUC
to 0.7602704091829113, best epoch 7, patience zero; epoch 8 is running.
All seven target counts and checkpoint/RNG/optimizer state passed verification.
GitHub authentication is restored and the previous backlog was pushed through
0999329. Continue checkpoint commits and pushes, then independent replay and
DKT only when SAKT finishes under the unchanged stopping protocol.
Latest 2026-09-19 19:40 UTC snapshot: SAKT epoch 8 improved validation AUC
to 0.7612905096381523, best epoch 8, patience zero; epoch 9 is running.
All eight target counts and checkpoint/RNG/optimizer state passed verification.
The preceding epoch-7 commit b0a2b1c is verified on the private GitHub remote.
Continue the same monitor session 10273 and unchanged experiment configuration;
commit and push the epoch-8 artifacts and memory, then continue waiting.
Latest 2026-09-20 03:40 UTC snapshot: SAKT epoch 9 improved validation AUC
to 0.7620343712659119, best epoch 9, patience zero; epoch 10 is running.
All nine target counts and actual checkpoint/RNG/optimizer/early-stopping state
passed verification. The epoch-8 commit 30364cc is verified on the private
GitHub remote. Commit/push the new artifacts and memory, then continue the
same 30-minute monitor session 10273 and unchanged experiment configuration.

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
On 2026-09-19 the user restored GitHub authentication. Active account
Tanishgupta28 and access to the existing private remote are verified; resume
checkpoint pushes. Retain the original identity and remote, and do not create
a replacement repository. Earlier push-deferral notes are historical.

## Stop conditions

- Do not start the 15 experiments before the smoke-test gates pass.
- Do not substitute the supplied Riiid file or reduced `ednetnew.csv`.
- Do not silently sample EdNet or alter HiTSKT's Performer architecture.
- Do not report the RKT smoke diagnostics as scientific results.
- Stop if full Phi preparation exposes a material scalability issue that would
  require changing cross-fitting, history semantics, or retained targets.
- Do not change the approved common patience 5 or `min_delta=0`.
