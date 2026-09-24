# Current state

Updated: 2026-09-24 UTC

## DKT epoch1 verified — 2026-09-24 05:39 UTC

- Epoch1 completed with val AUC0.6845717850759062, best1, patience0/5;
  full train53,990,022 and validation14,520,975 targets verified. Runtime
  60443.006547887344s (16.789724h). Actual last/best model equality,
  Adam/RNG/seed/early-stopping state verified. Best SHA256
  56a911479a2c0b53edd457e4179a942be6f652717e7322c35b41e45194e90215.
- Epoch2 running unchanged CUDA PID3412383. Old monitor66426 expired after
  interrupted turn; read-only process check found no remaining monitor.
  Replacement two-hour monitor **90580**, first output~07:40UTC. Trainer was
  not interrupted or restarted. Commit/push checkpoint and continue monitoring.

## Latest monitor — 2026-09-24 04:44 UTC

- User asked about16h epoch time. Read-only diagnosis:2,702,138 train batches
  +726,758 val batches; rough Junyi target-scaled comparison9.08h, not ETA.
  DKT.forward retains model-internal CUDA synchronization/per-student preparation
  overhead unaffected by packed transport. Other GPU contention plausible but
  unverified. Reported honestly as unusually long vs history, not a proven hang
  or isolated optimization regression. No training changes; details in report.

- Final DKT/EdNet PID/SID3412383 remains active at16h33m uptime, epoch1
  not yet logged complete. Console has no errors; saved CUDA packed64 profile
  and all scientific settings unchanged. No restart/signals/config changes.
- Monitor66426 delivered Sep23 20:11/22:11 and Sep24 00:11/02:11/04:11
  snapshots. Continue its two-hour waits (next06:11UTC), verify/push first
  completed checkpoint when available. Last pushed commit e0c3c95 contains
  SAKT exact replay and14/15 generated results. Other user GPU jobs untouched.

## SAKT replay verified; 14/15 complete — 2026-09-23 17:43 UTC

- Independent SAKT/EdNet reference-transport replay succeeded: all seven test
  metrics and13,429,870 targets exactly match packed64 final evaluation. Best13
  SHA1de01970ef3ebf8a3b136347f4db21f7a746ec610eaa95a86051ddc84a5497ea.
  Replay runtime35364.1847898094s under concurrent workloads; monitor81202
  ended normally. Generated reports/final_results.csv,json now contain14/15.
- Final DKT/EdNet PID3412383 remains active in epoch1. Continue its two-hour
  monitor66426 (next18:11UTC), verify new checkpoints, commit/push artifacts.
- GitHub push through2b141cc was verified against remote main; public visibility
  explicitly approved. User README edit remains untouched.

## Public push authorized — 2026-09-23

- User explicitly chose **Keep it public and push**. Reverified GitHub's
  canonical repository `Tanishgupta28/HITSKT-pure-performer-attention` is public;
  old capstone-gpu remote redirects there. Push hold is lifted; synchronize
  queued and subsequent relevant commits without changing visibility.
- Preserve original Tanish identity and unrelated root README working-tree
  changes. DKT3412383 and SAKT replay3368228 continue unchanged; monitors66426
  and81202. Previous hold instructions below are historical and superseded.

## Parallel DKT launched — 2026-09-23 12:11 UTC

- User explicitly approved starting DKT before the SAKT replay finishes.
  Full DKT/EdNet is now detached CUDA PID/SID **3412383**, parent1, directory
  `experiments/dkt/ednet_kt1/full`; command `PYTHONPATH=. python
  scripts/train_baseline.py dkt ednet_kt1
  data/processed/ednet_kt1/full/session_store experiments/dkt/ednet_kt1/full
  --workers 8 --transport packed64` (setsid/nohup, console.log).
- Actual saved config matches every repository DKT profile field: batch20,
  context/history200/199, embedding/hidden64, one layer, stated dropout0.1
  (effective recurrent0), AdamLR0.0002/wd0, no clipping, ceiling200,
  seed42, patience5/min_delta0, parameters417752. All full target counts
  verified. No completed epoch yet.
- DKT two-hour monitor terminal **66426**, next~14:11UTC. SAKT independent
  reference replay PID3368228 is still active; its monitor **81202** remains,
  next13:43UTC; functions exec cell39 currently waits on that monitor.
  No processes were stopped. Preserve user workloads. Parallel timing is
  confounded, not an isolated throughput benchmark. Push hold/README edit intact.

## SAKT complete; independent replay running — 2026-09-23 07:42 UTC

- SAKT/EdNet stopped correctly after epoch18 (five consecutive misses), best13
  val AUC0.7623242165859992. Final best-checkpoint test AUC0.7630078573122142,
  targets13,429,870, loss0.5245477522701514, accuracy0.7392807972080147,
  precision0.7565031146532126, recall0.9124938853983476,
  F1=0.8272087563782752, MSE=0.1752942014962786. _SUCCESS at07:09UTC.
- All18 epoch counts/stop decisions, one final test, last Adam/RNG/patience5,
  and unchanged best13 SHA verified. Runtime513909.28537087236s excludes
  authorized discarded attempt (documented separately). Epoch18+test consumed
  22137.231776336674s; epoch17's longer timing did not persist unchanged.
- Fresh independent replay started detached PID/SID3368228 at~07:42UTC,
  `scripts/evaluate_checkpoint.py experiments/sakt/ednet_kt1/full
  data/processed/ednet_kt1/full/session_store --workers 8 --output
  experiments/sakt/ednet_kt1/full/independent_evaluation.json`.
  Uses original reference transport, testing exact agreement with packed64
  final metrics. Wait for verification, aggregate14/15, then launch DKT.
  SAKT monitor87358 exited normally on SUCCESS. Replay two-hour monitor terminal
  **81202** started~07:43UTC; next check~09:43UTC. Push hold remains.

## Latest checkpoint — 2026-09-23 01:41 UTC

- Epoch17 completed at01:00 UTC: val AUC0.7614665404487208, best13 remains
  0.7623242165859992, patience4/5. Epoch18 running CUDA PID3100394.
  All17 target counts, actual last Adam/RNG/patience state and unchanged best
  SHA verified. Runtime491772.0535945357s, epoch17 duration47588.16266192822s
  (13.2189h). User-reported additionalGPU workloads are a possible, unconfirmed
  timing confound. No methodological/settings changes. Continue monitor87358
  every2h; another miss must stop training and reload best for final test.

## Latest monitoring — 2026-09-23 00:38 UTC

- CUDA trainer3100394 remains active; no console error, last complete epoch16
  unchanged, best13/patience3. Epoch17 has exceeded12h since epoch16 completed
  at Sep22 11:47 UTC. This is slower than the first optimized epoch and prior
  average; do not extrapolate epoch16's1.281x observation to current workload.
- User reported launching additional GPU processes. Contention is plausible
  if resources overlap, not proven; nvidia-smi per-process telemetry returned
  insufficient permissions. Do not stop/modify other workloads. CPU process
  time is not GPU-utilization evidence. Running model/optimizer/settings intact.
- Monitor87358 delivered Sep22 19:41,21:41,23:41 snapshots. Continue waiting
  for its two-hour checks (next Sep23 01:41 UTC); no new epoch to commit yet.

## Active checkpoint — 2026-09-22

- Latest 15:55 UTC: optimized epoch16 completed at 11:47 UTC, validation AUC
  0.760941857184205; best13 remains 0.7623242165859992, patience3/5. Epoch17
  active on CUDA PID3100394. All sixteen epoch target counts, last checkpoint
  Adam/RNG/patience and unchanged best SHA verified. First packed64 full epoch
  took 21,971.072426555213 seconds (6.1031h), versus original fifteen-epoch
  average 7.8188h: observed1.2811x / ~22% less time, not the short-window1.84x.
  Ten-hour uptime covers completed16 plus in-progress17, not a slower epoch.
  User asked about slowdown; explained measured timing and monitoring delay.
  Continue terminal87358 two-hour checks (next ~17:41 UTC), local commits;
  push hold and unrelated README edit remain.

- **Latest, 05:40 UTC: packed64 deployed.** User explicitly approved discarding
  the unfinished epoch. Full suite 75 passed, including bit-exact production
  DKT/SAKT transport and stochastic SAKT reference-to-packed resume tests.
  Old group 2535997 terminated only after backing up/validating epoch-15 model,
  Adam/RNG/config/logs and best epoch-13 SHA. New CUDA PID/SID **3100394**
  runs `--workers 8 --resume --transport packed64`; epoch 16 restarted,
  best 13 / val AUC 0.7623242165859992 / patience 2 unchanged.
  Backup: `.inspection/transport_switch_20260922/`. About 92 minutes of
  unfinished-epoch work discarded by authorization, excluded from accumulated
  runner runtime and documented in performance report. No methodology changed.
  Old monitor 39203/2741319 is no longer alive; replacement two-hour monitor
  terminal **87358** started approximately 05:42 UTC (first check ~07:42).
  Push hold and unrelated README edit remain. Earlier no-deployment notes below
  describe the pre-approval state, superseded by this entry.

- User authorized an isolated exact-training transport optimization benchmark,
  with no compromise or loss of current progress. Implemented experimental
  `ktbench/data/fast_rolling.py`, `scripts/benchmark_exact_transport.py`, and
  ten new fixture tests; production trainer/model/loader remain untouched.
  Actual EdNet epoch-15 checkpoint copy: SHA-256
  `d2bb25fd81908d70a5dc610a74e6877bc96576b069a7949c2b446471df5e7d8b`.
  Three repeated 2,048-step comparisons plus 64 warmup steps measured 1.843059x
  training and 4.377113x validation throughput with packed transfers and
  64-step deferred metric copies. All 18 executions matched reference model,
  Adam, RNG, per-step loss/probabilities/labels, and all seven metrics exactly;
  all planned input tensors/IDs/shapes/masks matched too. CPU vectorization
  alone offered no measurable gain. See `reports/performance/README.md` and
  generated `transport_repeated.json` for measured times and limitations.
  Shared-device short-window result, not a full-epoch guarantee. Full suite:
  **70 passed**, 35 existing deterministic-CUDA warnings, 15.76 seconds.
  At approximately 05:21 UTC trainer 2535997 was still active in epoch 16.
  Do not deploy mid-epoch or restart it; a lossless handoff and optimized-runner
  integration/resume tests are not yet implemented. Report results to user first.
- Latest SAKT/EdNet snapshot, 2026-09-22 04:11 UTC: epochs 14 and 15
  completed without improvement (validation AUC 0.758903934020328 and
  0.761176496772544). Best remains epoch 13 at 0.7623242165859992;
  patience 2/5, epoch 16 is running. All fifteen target counts, actual last
  checkpoint/RNG/optimizer/patience, and unchanged best checkpoint SHA passed
  verification. Accumulated runtime: 422,212.81850605225 seconds.
  Continue two-hour monitor terminal 39203 and local commits; visibility
  decision is still pending. Preserve the unrelated working-tree README edit.
- **Push hold:** GitHub now reports the renamed repository
  `Tanishgupta28/HITSKT-pure-performer-attention` as public (`isPrivate=false`,
  verified with `gh repo view` after the 13:47 UTC snapshot). Earlier user
  authorization specified a private repository. Do not change visibility or
  push further artifacts until the user confirms the intended visibility.
  Prior epoch-12 push 9b081ee succeeded through the old redirect; visibility
  at the time of that push was not checked. Local commits and training continue.
- Latest SAKT/EdNet snapshot, 2026-09-21 13:47 UTC: epoch 13 improved
  validation AUC to 0.7623242165859992, best epoch 13, patience 0/5;
  epoch 14 is running. All thirteen target counts and actual best/last
  checkpoint/RNG/optimizer/patience and model equality passed verification.
  Accumulated runtime: 368,394.98422291316 seconds. Best SHA-256:
  `1de01970ef3ebf8a3b136347f4db21f7a746ec610eaa95a86051ddc84a5497ea`.
  Continue two-hour monitor terminal 39203; preserve the unchanged CUDA run.
- Latest SAKT/EdNet snapshot, 2026-09-21 05:47 UTC: epoch 12 validation
  AUC 0.7620451155493354 did not improve. Best remains epoch 10 at
  0.7621860047840096; patience 2/5, epoch 13 is running. All twelve epoch
  target counts, last checkpoint/RNG/optimizer/patience, and unchanged best
  checkpoint SHA passed verification. Accumulated runtime is
  336,003.5465371683 seconds. Continue two-hour monitor terminal 39203.
- Latest SAKT/EdNet snapshot, 2026-09-20 21:47 UTC: epoch 11 validation
  AUC 0.761805075443002 did not improve. Best remains epoch 10 at
  0.7621860047840096; patience 1/5, epoch 12 is running. All eleven epoch
  target counts, last checkpoint/RNG/optimizer/patience, and unchanged best
  checkpoint SHA passed verification. Accumulated runtime is
  307,812.1677355771 seconds. Continue two-hour monitor terminal 39203.
- Monitoring cadence is now **two hours**, last requested on 2026-09-20;
  this supersedes the hourly and all earlier cadences below. At 19:47 UTC,
  trainer PID/SID 2535997 remained active in epoch 11; best epoch 10 validation
  AUC 0.7621860047840096, patience 0/5. Previous checkpoint af54f1a is pushed.
  The old hourly monitor no longer exists; two-hour monitor terminal 39203
  replaces it (first scheduled check around 21:48 UTC).
- Historical monitoring change: **one hour**, requested earlier on 2026-09-20;
  this supersedes all historical 30-minute/20-minute instructions below.
  The old monitor process group 2573137 was stopped without touching trainer
  PID/SID 2535997. New monitor terminal 70233 checks every 60 minutes.
- Latest SAKT/EdNet snapshot, 2026-09-20 15:57 UTC: epoch 10 improved
  validation AUC to 0.7621860047840096, best epoch 10, patience 0/5;
  epoch 11 is running. All ten target counts and actual best/last checkpoint,
  optimizer, RNG, seed, and early-stopping state passed verification.
  Accumulated runtime: 275,302.1220032424 seconds. Best SHA-256:
  `983bc5b818786bdd60e27f97cb958e5fc8ada378265891d0bccb1d7d143c7747`.
  The preceding epoch-9 commit 29af803 is verified on the private GitHub remote.
- Latest SAKT/EdNet snapshot, 2026-09-20 03:40 UTC: epoch 9 improved
  validation AUC to 0.7620343712659119, best epoch 9, patience 0/5;
  epoch 10 is running. All nine target counts and actual best/last checkpoint,
  optimizer, RNG, seed, and early-stopping state passed verification.
  Accumulated runtime: 245,997.1484295642 seconds. Best SHA-256:
  `1b530feef87d5c3b94bf6cc7ae7fb75a61d4afd404caca2f49e136932a2ef0d1`.
  The preceding epoch-8 commit 30364cc is verified on the private GitHub remote.
  Continue monitoring session 10273 every 30 minutes without changing the run.
- Latest SAKT/EdNet snapshot, 2026-09-19 19:40 UTC: epoch 8 improved
  validation AUC to 0.7612905096381523, best epoch 8, patience 0/5;
  epoch 9 is running. All eight target counts and actual best/last checkpoint,
  optimizer, RNG, seed, and early-stopping state passed verification.
  Accumulated runtime: 218,852.2276108954 seconds. Best SHA-256:
  `15d3a53377b91d50422db7fc563b94962a16e8a34e70ff57f10d6875c1fce9ea`.
  The previous epoch-7 commit b0a2b1c was pushed and remote SHA verified.
  Continue 30-minute checks, commits and pushes under the original identity.
- Latest SAKT/EdNet snapshot, 2026-09-19 12:10 UTC: epoch 7 improved
  validation AUC to 0.7602704091829113, best epoch 7, patience 0/5;
  epoch 8 is running. All seven target counts and actual best/last checkpoint,
  optimizer, RNG, seed, and early-stopping state passed verification.
  Accumulated runtime: 191,236.4744380503 seconds. Best SHA-256:
  `f1dc31f195d6a774079d0e1265edf8eece7cd0a16c7cbe0503877a5f250f8eba`.
  Monitor session 10273 checks the unchanged detached trainer every 30 minutes.
- All previously queued commits were successfully pushed to private
  Tanishgupta28/capstone-gpu/main through 0999329; remote SHA matched local.
- GitHub authentication restored: active account Tanishgupta28 and access to
  the existing private capstone-gpu repository verified. Push deferral is
  lifted; resume checkpoint pushes under the original Tanish identity.
- Latest SAKT/EdNet snapshot, 2026-09-19 07:09 UTC: epoch 6 improved
  validation AUC to 0.7592312015241357, best epoch 6, patience 0/5;
  epoch 7 is running. Actual best/last checkpoint state, optimizer, all RNG
  families, seed 42, and all six epochs' target counts passed verification.
  Accumulated runtime: 163,773.49147867132 seconds. Best SHA-256:
  `4c8074533dea335719cb2da7d2c55072a0dc2cf4d8281dea95169aa93a7fbdc7`.
  Detached trainer PID 2535997 remained alive. The old monitor connection
  expired; replacement terminal session 10273 checks every 30 minutes.

- Latest SAKT/EdNet snapshot, 2026-09-18 22:22 UTC: epoch 5 improved
  validation AUC to 0.7575770754914868, best epoch 5, patience 0/5;
  epoch 6 is running. Actual best/last checkpoint state, optimizer, all RNG
  families, seed 42, and all five epochs' target counts passed verification.
  Accumulated runtime: 136,659.67931209318 seconds. Best SHA-256:
  `567e6fc245e7c2a776fa49ca0eb67dbdd19b98a30fb1d0c4c4039418ded06be3`.
  Continue 30-minute monitoring of the same detached trainer, PID 2535997.
- Latest SAKT/EdNet snapshot, 2026-09-18 13:22 UTC: epoch 4 improved
  validation AUC to 0.7562697781063842, best epoch 4, patience 0/5;
  epoch 5 is running. Actual best/last checkpoint state, optimizer, all RNG
  families, seed 42, and all four epochs' target counts passed verification.
  Accumulated runtime: 109,496.36242577527 seconds. Best SHA-256:
  `44e376fc909871a9a26f036027b02bd52c868f3d007f37ebe26084bfe43bc6d3`.
  Continue the same 30-minute monitor (session 65227) and trainer PID 2535997.
- Latest SAKT/EdNet snapshot, 2026-09-18 05:52 UTC: epoch 3 improved
  validation AUC to 0.7542712165233839, best epoch 3, patience 0/5;
  epoch 4 is running. Both actual best/last checkpoints, optimizer, seed,
  RNG, early-stopping state, and all three epochs' target counts passed
  read-only verification. Accumulated runtime: 82,062.76307904115 seconds.
  Best checkpoint SHA-256:
  `306dffde3b2286a3b8a6b8aafdf0077ebaa2018e69fb43c093b173b1842ddae0`.
  Continue 30-minute monitoring of the same detached PID/SID 2535997.

- Full DKVMN/EdNet completed after six epochs, best epoch 1 validation AUC
  0.6719177243615195. Test AUC 0.6741059915424829, accuracy
  0.6988751938775283, precision 0.7053461512590434, recall
  0.9612753689964084, F1 0.8136602984915351, MSE 0.19920125153954787,
  loss 0.5833078272692482, targets 13,429,870. Runtime 88,075.51393550122
  seconds; 404,801 parameters. Fresh-process replay matched all metrics
  exactly; SHA-256 `23e5ea19fd63c23647869012197d93159b87cc1c76399e6d7136bedcd26bd4e2`.
  All six checkpoint/patience decisions and 13 CSV rows passed validation.
  Aggregate now has 13/15 verified results. Remaining runs: SAKT and DKT EdNet.
- Full SAKT/EdNet launched 2026-09-17 around 06:52 UTC, detached PID/SID
  2535997, parent 1. Its saved config passed assertions: context/history
  100/99, batch 10, width 200, heads 5, dropout 0.2, LR 1e-5, clip 10,
  ceiling 300, seed 42, strict validation-AUC patience 5/min_delta 0.
  Parameter count 1,668,401. Monitor every 30 minutes; commit locally and
  continue to independent replay/aggregation, then DKT, when complete.
  At the 14:52 UTC check epoch 1 was complete, validation AUC
  0.7470684949420671, best epoch 1, patience zero; epoch 2 is running.
  The actual best/last checkpoints agree, with seed 42, Adam state, all four
  RNG families, and the exact early-stopping state verified. Epoch 1 runtime
  was 27,432.569405004382 seconds. Best checkpoint SHA-256:
  `b4933dd40f888ed9cd1df5a4ec8c67d12eb3c19c20ee260a52ce49d6dcf9223e`.
  Monitor terminal session 65227 is independent of training.
- At 2026-09-17 22:22 UTC SAKT epoch 2 improved validation AUC to
  0.7515455799702323, best epoch 2, patience 0/5; epoch 3 is running.
  Both epoch target counts and the actual best/last checkpoint optimizer,
  seed, RNG, and patience state passed read-only verification. Accumulated
  runtime: 54,548.55061540101 seconds. New best checkpoint SHA-256:
  `26d8394bca51329e1e3521570e83fb73b9d1130f821961e3e559335fba80bb76`.
- Check training status every 30 minutes, superseding the previous 20-minute
  cadence. Keep Tanish's commit identity; authentication is now restored.
## Completed-run monitoring history (historical snapshots)

The counts, process IDs, and active-run descriptions below record their stated
times. The active checkpoint above supersedes them.

- Full HiTSKT/EdNet-KT1 completed successfully after 35 epochs under the
  common patience-5 rule. Best epoch 30 validation AUC is
  0.7681769525193056. Its best-checkpoint final test evaluated exactly
  13,429,870 targets: AUC 0.7693648364413692, accuracy 0.7388606144363273,
  precision 0.7639387403579302, recall 0.8946143710041425,
  F1 0.8241286589004618, MSE 0.17371257418130398, and loss
  0.5185181172959707. Runtime was 83,399.66895867884 seconds on the H100 MIG.
- Fresh-process independent replay loaded the actual epoch-30 checkpoint and
  reproduced every test metric exactly; checkpoint SHA-256 is
  `58d4b1f393d0fa8ec372ba8bcc00cc8b7168d37817d4badfd26ee42abffd03c8`.
  Strict curve/artifact validation passed, and generated master results now
  contain 12/15 verified scientific rows. Remaining: EdNet DKT, DKVMN, SAKT.
- Full DKVMN/EdNet-KT1 launched in a separate detached process session as PID
  and SID 2443018 (parent PID 1). Its initialized config was asserted before
  continuing: context/history 200/199, batch 32, Adam LR 0.001, memory/key/value
  20/50/100, dropout 0, seed 42, ceiling 100, strict patience 5/min_delta 0,
  and best-checkpoint reload. No profile or methodology change was made.
- 2026-09-16 09:40 UTC: DKVMN/EdNet epoch 1 completed after roughly four
  wall-clock hours with train AUC 0.6816267576040387 and validation AUC
  0.6719177243615195. It is the current best, patience 0/5; epoch 2 is
  running. The actual last checkpoint was loaded read-only and verified to
  contain model/optimizer state, seed 42, all four RNG families, and exact
  best/patience state. The actual best checkpoint and epoch logs are committed.
- 2026-09-16 14:00 UTC: DKVMN/EdNet epoch 2 completed with validation AUC
  0.6656104984379707, below epoch 1. Epoch 1 remains best at
  0.6719177243615195; patience is 1/5 and epoch 3 is running. Updated
  config/logs are committed locally; ignored last_model.pt remains available
  for exact continuation from epoch 2.
- 2026-09-16 18:01 UTC: DKVMN/EdNet epoch 3 completed with validation AUC
  0.6511616713824253. Best remains epoch 1 at 0.6719177243615195;
  patience is 2/5 and epoch 4 is running. Monitoring session 19124 checks
  every 30 minutes. Updated logs/config and memory are committed locally.
- 2026-09-16 22:01 UTC: DKVMN/EdNet epoch 4 validation AUC is
  0.6500716087367917. Best remains epoch 1 at 0.6719177243615195;
  patience is 3/5 and epoch 5 is running. Logs/config and memory are
  checkpointed locally under the original Tanish identity.
- 2026-09-17 02:01 UTC: DKVMN/EdNet epoch 5 validation AUC is
  0.648117614809817. Best remains epoch 1 at 0.6719177243615195;
  patience is 4/5 and epoch 6 is running. If epoch 6 does not improve,
  the common controller must stop and test the epoch-1 checkpoint.
- The prior interrupted epoch-20 directory remains preserved only as audit
  provenance. It is not included in the master results. The clean completed
  run is the accepted full EdNet HiTSKT result.

- Historical run log follows. The original full
  HiTSKT/EdNet process stopped after epoch 20 during a terminal interruption,
  before the patience-5 stopping point or final test. Its best epoch was 17,
  validation AUC 0.7672589703242605, with three subsequent misses. Its legacy
  model/optimizer checkpoints omitted RNG state, so it was archived without
  claiming a final result. See the archive's audit.json.
- HiTSKT runner exact resume support is committed as `2b7c904`; a new test
  proves interrupted/resumed and uninterrupted two-epoch model tensors and
  test metrics are identical. All 52 tests pass. The active model/attention
  architecture is unchanged, with four pure ELU+1 CausalLinearAttention stages.
- The user approved the clean rerun after the interruption. A plain-nohup
  launch was cleaned up before initialization; the replacement separate-session
  launch is live as PID 2267845, parent PID 1, SID 2267845. Correct config:
  width 128, heads 2, FF 1024, LR 8e-5, seed 42, token budget 32768,
  maximum batch 64, patience 5, epoch ceiling 40, no truncation/chunking.
- Monitor every 20 minutes and automatically verify/aggregate/commit after
  completion, then advance to remaining EdNet baselines. Read current runner
  PID/artifacts rather than trusting this historical PID blindly.
- Reporting and replay tooling is now implemented while the replacement
  trainer runs. `scripts/evaluate_checkpoint.py` validates and independently
  replays completed best checkpoints with exact metrics and hashes; its three
  model-family fixture paths pass. `scripts/generate_report.py` requires the
  complete 15-run matrix, validates curves/results before writing seven tables,
  15 training-curve plots and three comparisons, and records source hashes.
  All 11 existing production runs pass read-only artifact validation; no
  incomplete scientific report or invented result has been generated.
  The full suite now has 55 passing tests.
- 2026-09-15 07:01 UTC monitor: clean rerun epoch 1 completed, validation AUC
  0.7527808881272077, matching the original first epoch. Actual epoch-1
  last checkpoint was loaded read-only and verified to contain Python,
  NumPy, torch CPU, one CUDA RNG state, Adam state, model tensors, seed 42,
  and best-epoch/AUC/patience state. The run is healthy and training epoch 2.
- 2026-09-15 07:41 UTC monitor: epoch 2 completed, validation AUC
  0.7573268796700343, again identical to the original run. Best checkpoint
  advanced to epoch 2, patience zero; detached trainer is now in epoch 3.
- 2026-09-15 08:21 UTC monitor: epoch 3 completed, validation AUC
  0.7588930018050819, identical to the old run. Best epoch 3, patience zero;
  detached trainer is healthy and training epoch 4.
- 2026-09-15 09:41 UTC monitor: epoch 5 completed, validation AUC
  0.7612169898244615, best epoch 5, patience zero. All five completed epochs
  match the original archived run exactly; detached trainer is in epoch 6.
- 2026-09-15 11:41 UTC monitor: epochs 6–8 also match the archived run.
  Epoch 7 is best at AUC 0.7622619624598154. Epoch 8 AUC
  0.7599432662688099 did not improve, so patience is 1/5; detached trainer
  is healthy and in epoch 9.
- 2026-09-15 13:01 UTC monitor: epoch 10 completed, best AUC
  0.764228728194286, patience zero; detached trainer is in epoch 11.
  A full read-only comparison of all ten epoch JSON records against the
  archived run's first ten records passed exact equality, including every
  train/validation metric and checkpoint/patience decision.
- 2026-09-15 16:21 UTC monitor: epoch 15 completed, best AUC
  0.7668270365637668, patience zero after recovery from the epoch-14 miss.
  Detached trainer remains healthy and is training epoch 16. Metrics and
  exact resume artifacts are intact; no final test has run yet.
- 2026-09-15 17:41 UTC monitor: epoch 17 completed, best AUC
  0.7672589703242605, patience zero; detached trainer is in epoch 18.
  A fuller local checkpoint includes the actual best_model.pt, epoch logs,
  config, and memory. Ignored last_model.pt remains intact for exact resume.
- GitHub authentication changed to Fyxod, and push to the original private
  remote returned repository-not-found. The user instructed preserving Tanish
  commit identity and remote, continuing local commits, and pushing later
  after they restore the original login. No remote/repository change is allowed.
- 2026-09-15 19:01 UTC monitor: epoch 19 completed with validation AUC
  0.7672127018589197. Epoch 17 remains best at 0.7672589703242605;
  epochs 18 and 19 are two consecutive non-improvements (patience 2/5).
  Detached trainer PID 2267845 is healthy and training epoch 20. No final
  test or success marker exists yet. Updated logs/config are committed locally
  under the original Tanish identity; pushes remain explicitly deferred.
- 2026-09-15 20:21 UTC monitor: epoch 21 improved validation AUC to
  0.7675438123942788, becoming the new best checkpoint and resetting patience
  to zero. Epoch 20 had been the third miss, but the approved controller
  correctly continues after this improvement. Epoch 22 is running. The
  actual new best checkpoint, logs/config, and memory are checkpointed locally.
- 2026-09-15 21:21 UTC monitor: epoch 22 remains the new best with validation
  AUC 0.7677329409251583, patience zero; epoch 23 is running. Local checkpoint
  now includes the actual epoch-22 best_model.pt.
- A restart edge case is fixed in all five model runners: if interruption
  occurs after the fifth miss but before final test evaluation, restored
  patience now skips further training/validation and directly reloads the
  best checkpoint for test. Five explicitly mocked protocol tests exercise
  the real save/restore path and verify no extra epoch/checkpoint mutation.
  All 60 tests pass, including real stochastic-resume/forward-backward tests.
  The already-running trainer is unaffected; no model/attention changes were made.
- 2026-09-15 23:01 UTC monitor: epoch 25 completed at validation AUC
  0.7676484582020142. Best remains epoch 22 at 0.7677329409251583;
  epochs 23–25 are three consecutive misses, patience 3/5. Epoch 26 is
  running on the healthy detached trainer. No final test/result is available.
- 2026-09-15 23:41 UTC monitor: epoch 26 improved to validation AUC
  0.7677920121661311, becoming best and resetting patience to zero.
  Epoch 27 is running; the actual new best checkpoint is committed locally.
- 2026-09-16 01:41 UTC monitor: epoch 29 improved to validation AUC
  0.7678508108666452, becoming best and resetting patience to zero after
  two non-improving epochs. Epoch 30 is running. The actual new best
  checkpoint, epoch-29 artifacts, and project memory are committed locally
  under the original Tanish identity. Pushes remain deferred by instruction.
- 2026-09-16 03:01 UTC monitor: epoch 30 improved to best validation AUC
  0.7681769525193056. Epoch 31 AUC 0.7679458417050529 did not improve,
  giving patience 1/5; epoch 32 is running. A local checkpoint now includes
  the actual epoch-30 best_model.pt and complete logs/config through epoch 31.

## Established

- `/workspace/capstone` initially contained only `prompt.txt` and a notebook
  checkpoint copy; it was not a Git repository.
- GitHub repository `pokerme7777/HiTSKT` exists with default branch `main`.
- The upstream continuation point is now checked out locally on branch `work`,
  with `origin/main` retained as the source baseline.
- Private checkpoint repository `Tanishgupta28/capstone-gpu` was created at the
  user's direction. Local branch `work` tracks its `main` branch; checkpoint
  commit `88aef9f` contains the prompt, assessment, directives, and memory.
- Upstream contains HiTSKT and baseline files for DKT, DKVMN, and SAKT. It does
  not contain an RKT implementation in the inspected tree.
- Upstream dataset CSV paths are Git LFS pointer files locally, not usable data.
- Available compute observed during inspection: an NVIDIA H100 MIG 3g.40gb
  device with about 40 GiB visible to PyTorch, approximately 2 TiB system RAM,
  and 224 logical CPUs. PyTorch 2.4.0a0 reports CUDA 12.5 and successfully sees
  the GPU.

Evidence: repository paths `../README.md`, `../model_layers.py`,
`../Other_models/`, and the 2026-09-09 Git/host inspection recorded in the
active task transcript.

## Assessment conclusion

The inspection is complete. See
[ASSESSMENT_2026-09-09.md](ASSESSMENT_2026-09-09.md). The user has approved the
dataset identity/selection, tag mapping, session/split protocol, rolling-history
evaluation, Performer consolidation, and RKT sourcing decisions. Implementation
is authorized and is beginning with authoritative EdNet-KT1 acquisition.

## Not yet validated or completed

- The genuine EdNet-KT1 interaction archive and official contents archive were
  downloaded from the official EdNet repository's published Google Drive
  objects. Both passed full ZIP integrity validation. The KT1 archive is
  1,201,163,816 bytes with SHA-256
  `0d13933f90201c5101c7fe8659e44474fa049e3fb93181a8ba6fb3e63267b535`;
  it contains 784,309 student CSV members totaling 3,072,366,053 uncompressed
  bytes. The contents archive is 173,976 bytes with SHA-256
  `aa910a0436d9dbac0ba232f55e27b37ad8d39da285fcf15f1c1eb5d064be98e7`.
- Official EdNet `questions.csv` contains 13,169 questions. All have tags;
  6,049 questions have multiple semicolon-delimited tags (maximum seven).
- The user explicitly extended the complete sorted composite-tag rule to EdNet
  and designated `tags = -1` as missing metadata mapped to one `<UNTAGGED>`
  composite skill ID. The provisional vocabulary is compatible with the
  existing embedding approach.
- A deterministic streaming preprocessor, README protocol, and focused tests
  have been implemented. All 11 current synthetic preprocessing test cases pass.
- The first real 1,000-student smoke invocation stopped before processing any
  interactions, as designed, on a previously unknown duplicate tag within a
  question. The user approved set deduplication, and the exact example now has a
  regression test.
- A full metadata audit found 197 questions with duplicated tag IDs: duplicate
  `176` in 85 questions, `178` in 75, and `177` in 37. There are 83 distinct raw
  affected tag strings and no other malformed format besides approved `-1`.
  The approved implementation deduplicates these repeated IDs before sorting.
- The second real smoke invocation passed tag mapping and stopped at interaction
  row 674 of `KT1/u4.csv` because `user_answer` is empty. A bounded audit of the
  first 1,000 numeric student files (1,427,687 interactions) found 2,898 empty
  answers across 349 students and no empty values in the other four columns.
  No rows were dropped and no correctness label was inferred.
- The user directed exclusion of empty answers from supervision with separate
  audit retention. The implementation now filters before rebuilding sessions,
  splits, and attempt counters. All 11 synthetic cases pass, including audit
  row preservation and no synthetic label.
- The third real first-1,000-student smoke run completed and was independently
  validated: 2,898/1,427,687 rows (0.2029856684%) across 349 students were
  excluded to audit; 1,409,948 supervised interactions from 704 eligible
  students formed 32,795 sessions. Session counts changed for 19 students (net
  -18); answered-event grouping changed for one. Parquet row counts, labels,
  ordering, session continuity, per-session positions, split chronology, and
  completion markers all passed.
- The full EdNet pass completed across 784,309 files and 95,293,926 source
  interactions. It excluded/audited 27,646 unanswered rows across 4,477
  students (0.0290112929%), leaving 95,266,280 supervised interactions.
- Post-filter session rebuilding changed session counts for 148 students (net
  -150) and answered-event grouping for six. The five-session filter retained
  116,548 students, 81,940,867 interactions, and 2,577,988 sessions, split into
  53,990,022 train, 14,520,975 validation, and 13,429,870 test interactions.
- Independent streaming validation passed all 82 event shards and 27,646 audit
  rows. Aggregate reports, distributions, and deterministic question/tag/skill
  mappings are staged under `../reports/datasets/ednet_kt1/`; interaction-level
  Parquet remains ignored and local.
- Accepted ASSIST2017 and full Junyi source files are staged under `data/raw/`
  with verified hashes. A vectorized shared preprocessor and three synthetic
  fail-closed tests have been added.
- The real ASSIST2017 run stopped before output completion because 697/3,162
  questions have multiple observed row-level skills (682 have two, 15 have
  three). They cover 440,761/942,807 interactions (46.749865%) and 3,874 unique
  question/skill pairs. The user directed preservation of supplied row-level
  skills. The completed build retains 885,335 interactions from 1,388 eligible
  students in 12,402 sessions; independent validation passes.
- Full Junyi composite mapping found 1,326 genuine original tags and 1,521
  complete-set composite skills. There are 776 multi-tag questions covering
  961,084 interactions (6.555728%). The validated build retains all 14,660,217
  interactions, 29,865 students, and 600,154 rebuilt sessions across 15 shards.
- Aggregate mappings/reports for ASSIST2017 and Junyi are staged under
  `../reports/datasets/`; interaction Parquet remains local and ignored.
- A full session-length audit found that the legacy fixed action capacity would
  discard 303,971 ASSIST interactions (34.334009%), 4,310,593 Junyi
  interactions (29.403337%), and 35,198,097 EdNet interactions (42.955485%).
  Maximum session lengths are 938, 3,924, and 13,080.
- The user selected variable-length length-bucketed batches, per-batch dynamic
  padding, and token-budgeted batch sizes, with no action truncation or session
  chunking. Lossless memory-mapped session stores now reconcile all 97,486,419
  retained interactions and all 3,190,544 sessions across the three datasets.
- The consolidated HiTSKT path explicitly retains Action Encoder, Session
  Encoder, Correct/Padding Encoder, Decoder, and prediction stages. All
  attention uses causal prefix-sum ELU+1 linear attention with PAD masks; there
  is no softmax or quadratic sequence attention in the new implementation.
- Full batch planning uses the legacy rolling context of at most 15 earlier
  complete sessions, a normal 32,768 padded-token budget, and batch-size cap
  64. Every post-first session remains one complete target. The worst EdNet
  example is indivisible and uses a 196,280-token singleton batch.
- Real forward/loss/backward benchmarks passed on the H100 MIG device. Peak
  allocated/reserved CUDA memory was 7.066/8.686 GiB for ASSIST, 4.593/5.621
  GiB for Junyi, and 24.989/30.861 GiB for the worst EdNet singleton.
- All HiTSKT batching/model tests pass, covering dynamic shapes/padding, causal and PAD
  masks, target shifting, EOS metric exclusion, long singleton batching,
  past-only rolling history, all-stage backward gradients, and strict
  checkpoint loading. Exact evidence is versioned in
  `docs/data/variable_length_batching.md` and
  `reports/batching/dynamic_batching_benchmark.json`.
- The centralized project seed is 42. It controls Python, NumPy, PyTorch
  CPU/CUDA, batch shuffling, data sampling, and deterministic five-fold RKT
  assignment. Full fold sizes are ASSIST 278/278/278/277/277, Junyi five equal
  folds of 5,973, and EdNet 23,310/23,310/23,310/23,309/23,309.
- The approved RKT paper-faithful performance-only variant is implemented from
  the authors' repository reference at commit `cac60f512f`. It uses width 64,
  dropout 0.1, one head, learned positions, rolling 49-event histories,
  directed raw Phi, positive trainable `S_u`, and learned lambda initialized
  at 0.5. Exact deviations are recorded in `../docs/models/rkt_variant.md`.
- RKT timestamps are verified per source and normalized to hours. Full-store
  training-only `S_u` initialization reports zero fallback students in all
  three datasets; global fallback values are recorded in the RKT report.
- Sparse Phi caches encode deterministic five-fold student cross-fitting for
  training and all-training-only lookup for validation/test. Tests prove fold,
  own-label, future-interaction, and validation/test exclusion.
- Full-population Phi preparation is complete for all three datasets. ASSIST
  processed 647,283 training targets and 13,530,869 contributions in 14.17 s;
  Junyi processed 9,647,042 targets and 410,926,464 contributions in 734.99 s.
  EdNet processed 53,990,022 targets and 2,430,771,986 contributions in
  1,984.27 s. Exact unique-pair, scratch, RSS, cache-size, sample, and historical
  estimate evidence is versioned in `../reports/rkt/phi_scaling_benchmark.*`.
- The bounded 20-student ASSIST RKT smoke passed a forward/backward update,
  `[128,49]` shapes, unique-target accounting, checkpoint round-trip, frozen
  evaluation relation parameters, and all required metrics. Its results are
  diagnostic only.
- No full model/dataset experiment has run; there are no valid final benchmark
  results.
- The user fixed baseline contexts/hyperparameters across datasets. Clean DKT,
  DKVMN, and SAKT adapters now use rolling target-once histories of 199, 199,
  and 99 prior interactions respectively, with dynamic PAD masking and the
  repository's unchanged model settings.
- Nine real-data/schema smoke gates pass across all model/dataset pairs. Full
  train/validation/test target counts reconcile to every retained interaction;
  each required history cap, forward/backward, all metrics, and strict
  checkpoint round-trip passed.
- Real H100 compute-only timing makes large rolling runs expensive, especially
  DKVMN: EdNet lower bounds are 8.169 h/epoch DKT, 81.614 h/epoch DKVMN, and
  4.953 h/epoch SAKT. Exact results and exclusions are in
  `../reports/batching/baseline_rolling_throughput.json`.
- The user fixed a common early-stopping protocol for all five models:
  validation ROC-AUC, patience 5, `min_delta=0`, strict improvement, and best
  checkpoint reload before test. `ktbench/training.py` enforces it and exact
  stop/reset behavior is tested. Existing smoke configs now record it.
- A production DKT/DKVMN/SAKT runner now writes per-epoch metrics/logs,
  best/last checkpoints, updated config state, dataset statistics, and final
  results while stopping within the preserved epoch ceiling. It has not been
  launched on a full dataset.
- The baseline production runner passed an end-to-end bounded-fixture test,
  including artifact writing and best-checkpoint reload.
- Matching production RKT and HiTSKT runners now use the common validation-AUC
  controller, save best/last checkpoints and complete configs/logs/metrics,
  reload the best checkpoint before test, and pass bounded end-to-end tests.
- RKT uses maximum gradient norm 10. The paper is silent on clipping; this is
  the authors' released-trainer default and was retained with explicit user
  approval. The setting and its provenance are persisted in RKT configs.
- A CUDA-only implicit indexed-broadcast failure for the learned HiTSKT
  session-EOS vector was fixed by explicitly expanding the same vector across
  batch rows. EOS placement and the research architecture are unchanged.
- All 51 automated tests pass. PyTorch warns that the CUDA cumulative-sum
  kernel used by Performer attention has no deterministic implementation in
  this installed build; seeding and deterministic warn-only mode remain on so
  this limitation is visible.
- The first full scientific experiment, RKT on ASSIST2017, completed in
  1,474.90 s. Strict early stopping selected epoch 14 at validation AUC
  0.7712554524 and stopped at epoch 19 after exactly five misses. Reloading the
  best checkpoint produced test AUC 0.7486264899, accuracy 0.6965399280,
  precision 0.6399917287, recall 0.4975655689, F1 0.5598625216, MSE
  0.1963360075, and loss 0.5771361224 across all 112,252 test targets.
- An independent full test pass from `best_model.pt` reproduced every stored
  test metric exactly.
- Full HiTSKT on ASSIST2017 completed in 968.04 s. Strict early stopping
  selected epoch 40 at validation AUC 0.7197572561 and stopped at epoch 45.
  Best-checkpoint test metrics over all 112,252 held-out actions are: AUC
  0.7096240159, accuracy 0.6744200549, precision 0.5866082262, recall
  0.5440494235, F1 0.5645278522, MSE 0.2098452017, loss 0.6087257417.
  Independent best-checkpoint evaluation reproduced every value exactly.
- HiTSKT used all 566,107 actions in post-first training sessions; the 81,176
  actions in each student's first session are context-only because the
  hierarchical model requires an earlier session. Validation/test contain no
  first student session and retain all 125,800/112,252 targets. No session was
  truncated or chunked.
- The aggregate `reports/final_results.{csv,json}` is generated from completed
  experiment artifacts and currently contains 11/15 rows; no smoke or
  throughput result enters it.
- Full DKT on ASSIST2017 completed in 6,233.07 s with the unchanged batch-20,
  context-200 configuration. Strict early stopping selected epoch 12 at
  validation AUC 0.7213696056 and stopped at epoch 17. Best-checkpoint test
  metrics over all 112,252 targets are: AUC 0.7069997420, accuracy
  0.6750258347, precision 0.6113932435, recall 0.4451564007, F1 0.5151970231,
  MSE 0.2083804281, and loss 0.6057200542. Independent full test evaluation
  reproduced every value exactly.
- Full SAKT on ASSIST2017 completed in 10,551.13 s with the unchanged batch-10,
  context-100 configuration. Strict early stopping selected epoch 30 at
  validation AUC 0.7577061439 and stopped at epoch 35. Best-checkpoint test
  metrics over all 112,252 targets are: AUC 0.7384655654, accuracy
  0.6902950504, precision 0.6240144686, recall 0.5071425291, F1 0.5595408532,
  MSE 0.1998719304, and loss 0.5853560606. Independent full test evaluation
  reproduced every value exactly.
- DKVMN's identical affine memory writes now use a chronological balanced
  composition rather than 199 Python-driven GPU launches. Reference output and
  gradient equivalence is tested. Actual full-history ASSIST step time improved
  24.0x from 0.174721 s to 0.007266 s; projected compute-only epoch time is now
  0.0408 h, with no model or protocol change.
- Post-optimization full-history measurements project compute-only DKVMN
  epochs at 0.0408 h (ASSIST), 0.6164 h (Junyi), and 3.4523 h (EdNet), down
  from 0.9817/14.8168/81.6145 h. The raw measurements are persisted in
  `reports/batching/dkvmn_affine_throughput.json` and exclude loading,
  validation, and checkpoint I/O.
- Baseline production runs now have a validated explicit resume path that
  restores model, Adam, and early-stopping state and appends durable artifacts.
  DKVMN/ASSIST was externally interrupted after epoch 19 and resumed at epoch
  20 without losing scientific state.
- Full DKVMN on ASSIST2017 completed across one validated resume in an
  estimated combined 3,888.21 s. Strict early stopping selected epoch 18 at
  validation AUC 0.6962520470 and stopped at epoch 23. Best-checkpoint test
  metrics over all 112,252 targets are: AUC 0.6822527021, accuracy
  0.6635961943, precision 0.6169661648, recall 0.3500987552, F1 0.4467106227,
  MSE 0.2141332602, and loss 0.6182015214. Independent full test evaluation
  reproduced every value exactly. The pre-resume wall-time component is
  explicitly marked as estimated from artifact mtimes; metrics are exact.
- RKT and baseline checkpoints now persist Python, NumPy, PyTorch CPU, and all
  CUDA RNG states alongside model, optimizer, and strict early-stopping state.
  An exact two-epoch RKT test proves resumed dropout training matches an
  uninterrupted run tensor-for-tensor and metric-for-metric. A seven-epoch
  Junyi attempt whose older checkpoint lacked RNG state is preserved as an
  excluded audit artifact and will be restarted from seed 42.
- Full RKT on Junyi completed across one exact RNG-safe resume in 9,365.18 s.
  Strict early stopping selected epoch 8 at validation AUC 0.7954429584 and
  stopped at epoch 13. Best-checkpoint test metrics over all 2,456,402 targets
  are: AUC 0.7912548487, accuracy 0.7523398043, precision 0.7789090882, recall
  0.8872965298, F1 0.8295774825, MSE 0.1670380883, and loss 0.5036462177.
  Independent full test evaluation reproduced every value exactly.
- Immediately before the full Junyi HiTSKT run, a static import/call audit and
  runtime module enumeration reconfirmed that the production runner reaches
  four `CausalLinearAttention` modules through the Action, Session,
  Correct/Padding, and Decoder stages. All use ELU+1 prefix-sum linear
  attention; no PyTorch Transformer/MultiheadAttention or legacy softmax module
  is present in the instantiated model. Nine focused causal, PAD, dynamic
  padding, checkpoint, gradient, and production-runner tests passed.
- Full HiTSKT on Junyi completed in 14,088.52 s. Strict early stopping selected
  epoch 32 at validation AUC 0.8008507200 and stopped at epoch 37 after exactly
  five misses. Reloading `best_model.pt` produced test AUC 0.7978262055,
  accuracy 0.7555505980, precision 0.7850452676, recall 0.8815436622, F1
  0.8305007526, MSE 0.1647862988, and loss 0.4975778811 across all 2,456,402
  test targets. A fresh-process independent evaluation reproduced every stored
  metric exactly. The run retained all complete target sessions with dynamic
  token-budgeted padding and no truncation or chunking.
- Full DKVMN on Junyi completed in 15,310.30 s with the unchanged batch-32,
  context-200/history-199 repository configuration and optimized equivalent
  affine-write execution. Strict early stopping selected epoch 1 at validation
  AUC 0.7450961224 and stopped at epoch 6 after exactly five misses. Reloading
  `best_model.pt` produced test AUC 0.7415784625, accuracy 0.7234760434,
  precision 0.7394883095, recall 0.9154544196, F1 0.8181163363, MSE
  0.1836369781, and loss 0.5455991907 across all 2,456,402 test targets. A
  fresh-process independent evaluation reproduced every stored metric exactly.
- Full DKT on Junyi completed in 46,586.80 s across one exact RNG-safe resume,
  with the unchanged batch-20, context-200/history-199 repository
  configuration. Strict early stopping selected epoch 3 at validation AUC
  0.7499506386 and stopped at epoch 8 after exactly five misses. Reloading
  `best_model.pt` produced test AUC 0.7476841497, accuracy 0.7257260823,
  precision 0.7664082388, recall 0.8576722579, F1 0.8094759909, MSE
  0.1820032698, and loss 0.5415314254 across all 2,456,402 test targets. A
  fresh-process independent evaluation reproduced every stored metric exactly.
- Full SAKT on Junyi completed in 88,701.94 s with the unchanged batch-10,
  context-100/history-99 repository configuration. Strict early stopping
  selected epoch 13 at validation AUC 0.7928608706 and stopped at epoch 18
  after exactly five misses. Reloading `best_model.pt` produced test AUC
  0.7887961777, accuracy 0.7494701600, precision 0.7709668341, recall
  0.8979818821, F1 0.8296411292, MSE 0.1684005803, and loss 0.5071997748
  across all 2,456,402 test targets. A fresh-process independent evaluation
  reproduced every stored metric exactly.
- Full RKT on EdNet-KT1 completed in 24,318.55 s with the approved
  paper-faithful performance-only Phi-relation variant and five-fold
  student-level cross-fitting. Strict early stopping selected epoch 1 at
  validation AUC 0.7479626872 and stopped at epoch 6 after exactly five
  misses. Reloading `best_model.pt` produced test AUC 0.7475470203, accuracy
  0.7257950375, precision 0.7316373002, recall 0.9460949348, F1 0.8251594973,
  MSE 0.1819252324, and loss 0.5400390939 across all 13,429,870 test targets.
  A fresh-process independent evaluation reproduced every stored metric
  exactly. Phi remained training-only/cross-fitted for all 53,990,022 training
  targets and training-only for validation/test.

## Known implementation risks requiring evidence

- The supplied Drive “EdNet” data is Riiid and cannot be used; genuine full
  EdNet-KT1 must be acquired and provenance-verified.
- Loading the complete production Phi repositories and measuring RKT epoch
  throughput is validated. Grouping identical lookups by applicable fold cache
  is exactly equal to scalar lookup in tests and improves projected epochs from
  96.3 s/98.0 min/9.89 h to 79.1 s/23.7 min/2.34 h for ASSIST/Junyi/EdNet.
  These are planning estimates, not scientific results; raw evidence is under
  `../reports/rkt/`.

## RKT protocol resolution

- The paper-author repository is available at `shalini1194/RKT`; author Shalini
  Pandey's current reference was inspected at commit `cac60f512f`. The paper is
  arXiv `2008.12736`.
- Full RKT defines exercise relations as the thresholded sum of a train-data
  performance Phi coefficient and cosine similarity of textual exercise
  embeddings. The paper also reports performance-only Phi and same-KC relation
  ablations. The accepted capstone datasets do not expose authoritative,
  comparable full exercise text for all three datasets.
- The authors specify maximum interaction length 50. The user approved a
  rolling 49-prior-interaction construction that retains every supervised
  target exactly once without forcing full sessions into one RKT matrix.
- The user approved performance-only Phi relations from training information
  and paper-faithful handling of these paper/code incompatibilities:
  - paper time weight is `exp(-delta_t / S_u)` with trainable student strength;
    released code uses `exp(-abs(raw_delta_t))`, has no `S_u`, and does not
    normalize seconds versus EdNet milliseconds, causing practical underflow;
  - paper settings state width 64, dropout 0.1, batch 128, and positional
    embeddings; executable defaults are width 200, five heads, dropout 0.2,
    batch 200, and positional encoding disabled. Width 64 is not divisible by
    the released five-head default;
  - paper performance-only ablation says to use Phi (Equation 2), while the
    full relation applies a 0.8 threshold only after adding text similarity;
    released code consumes an external precomputed relation file and does not
    implement Phi construction, so it does not resolve thresholding for the
    no-text variant.
- The complete approved formulation, five-fold leakage protocol, timestamp
  audit, initialization, source citation, and smoke evidence are implemented
  and documented in `../docs/models/rkt_variant.md`. No current material RKT
  ambiguity remains.

Local-only inspection material is under `../.inspection/`; it is untracked and
contains downloaded dataset copies and source variants. It is evidence staging,
not a deliverable or accepted data layout.
