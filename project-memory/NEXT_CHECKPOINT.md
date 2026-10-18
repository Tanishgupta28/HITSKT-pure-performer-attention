# Next checkpoint

Updated: 2026-10-18 UTC

## Immediate action

**Wait for further user instructions. Do not start the 15th training.**
The GPU-reset request authorizes reconciliation and synchronization only.
The older three-hour sleep/check schedule and automatic >2h stall recovery
are superseded for this session by [USER_DIRECTIVES.md](USER_DIRECTIVES.md).
Do not launch training, independent evaluation, monitoring or final reporting
as a side effect of reading this handoff.

This milestone includes epochs6–9 config/CSV/JSONL, reset evidence and refreshed
memory. Confirm the scoped commit reaches `capstone-gpu/main` without force,
then end the task and wait. Preserve unrelated README and SAKT console edits.

## Last valid evidence and outstanding issue

- Benchmark14/15; only DKT/full EdNet is unfinished. Canonical output is
  `experiments/dkt/ednet_kt1/full/` inside this repository.
- Nine complete epochs; epoch10 interrupted at67.454753% of train, last
  progress2026-10-12T04:40:41Z. No active trainer or final-test artifact.
- Best1/AUC0.6845717850759062; saved counter8 and active patience10. Exact
  checkpoint hashes/config/target counts are in [CURRENT_STATE.md](CURRENT_STATE.md)
  and `reports/performance/dkt_reset_reconciliation_2026-10-18.json`.
- Last9 loads and model weights are finite, but **five Adam second-moment
  tensors contain21,749 positive infinities**. Best1 model/Adam are finite.
  Epoch9 is not certified ready to resume. Cause/recovery remain unresolved.
- Preserve `.inspection/dkt_reset_20261018/`, current best/last and all logs.
  `last_model.pt` and snapshots are local/ignored, not in the public commit.
- Embedded checkpoint patience5 is serializer metadata; active saved config10
  and counter8 govern resume. Do not silently revert to5 or reset state.

## Conditional continuation after new user direction

1. Re-read latest directives and reconcile processes, artifacts and source hash.
2. Review the nonfinite Adam finding with the user before choosing any scientific
   recovery change. Do not silently clear moments, restart from another epoch,
   change precision/clipping/hyperparameters, or overwrite the interrupted run.
   A validated recovery contract must precede any future training launch.
3. If continuation is later authorized, preserve full data, seed42, width64,
   context200, batch20, Adam0.0002, packed64/workers8, RNG, strict AUC/min_delta0,
   ceiling200 and the DKT/EdNet-only patience10 exception. A last9 continuation
   would start epoch10, not claim within-epoch continuation; readiness is unresolved.
4. Only after actual training completion and `_SUCCESS`, independently replay
   the selected best checkpoint with all13,429,870 test targets and require exact
   metric equality. Then generate the strict15/15 report from actual results,
   complete `prompt.txt` documentation and publish authorized final artifacts.
   None of these conditional stages was executed during reset reconciliation.

The old active-run handoff is preserved as
[October 10 checkpoint](NEXT_CHECKPOINT_history_before_2026-10-18_reset.md) and
[earlier checkpoint](NEXT_CHECKPOINT_history_before_2026-10-09_resume.md).
