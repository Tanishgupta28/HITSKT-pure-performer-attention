# Next checkpoint

Updated: 2026-10-10 UTC

## Latest actual assistant check — 2026-10-10 08:35 UTC

- User status question interrupted sleep; primary assistant directly checked the run. Epoch5/train: 2,700,724/53,990,022 targets (5.00% of this phase), 135,168 minibatches, as of2026-10-10T08:35:30Z.
- Trainer787181 is active with unchanged start identity. Profile/source match. Console errors found: 0. _SUCCESS: False. Recovery decision: 2026-10-09 user: you have full authorization to restart it if it gets stuck for more than 2 hours.
- Verified 4 completed epoch(s), all train/validation target counts, strict validation-AUC selection and patience. Last4/best1, best validation AUC0.6845717850759062, patience3/5. Actual best/last checkpoints, saved Adam state and RNG were checked; hashes are in the ledger.
- Benchmark remains14/15 until DKT is complete and independently verified. No scientific settings changed.
- Remain in the active turn, sleep another three hours, and personally check around2026-10-10 10:39:33 UTC. The detached supervisor remains stopped. If batch progress is stalled for more than two hours, use the standing user authorization to preserve evidence and resume the latest verified completed epoch.
- At completed epochs verify and push actual artifacts. On _SUCCESS independently replay best, generate strict15/15 reports, and finish prompt documentation/publication.
- Evidence: `reports/performance/dkt_assistant_monitoring_2026-10-08.jsonl` and `.inspection/dkt_restart_20261008/assistant_monitoring.json`.

## Approved recovery and standing authorization — 2026-10-09

- User answered “Resume from epoch 2 (Recommended).” Stopped trainer506822 and its
  eight verified workers with SIGTERM at16:28UTC; no unrelated process was targeted.
- Repeated unfinished epoch3 from epoch2. Completed epochs, best1 and patience1
  preserved. The discarded prefix was47,794,594 training targets (88.52%), with
  approximately6.23h useful unfinished work followed by approximately9.97h stalled.
- Historical run snapshot: `.inspection/dkt_epoch3_stalled_before_resume_20261009_162819/`.
  Recovery evidence: `../reports/performance/dkt_resume_epoch2_2026-10-09.json`.
- Exact cause remains unconfirmed. Fresh CUDA/checkpoint execution succeeded.
  New local wrapper `.inspection/resume_dkt_with_stacks.py` registers SIGUSR1
  to `.inspection/dkt_restart_20261008/python_stacks.log`; handler exercised
  successfully on the new trainer. Never send this signal to a trainer without
  verifying registration. Trainer scientific source was not modified.
- Latest user directive: **“you have full authorization to restart it if it gets
  stuck for more than 2 hours.”** At each personal three-hour check, recover
  without further permission if verified batch progress is stalled for >7200s.
  Preserve stack evidence and incomplete-run snapshot, verify the latest completed
  checkpoint/control/RNG/optimizer, stop only owned processes, and resume that
  completed epoch. A recovery repeats only the unfinished epoch.
- Runtime reporting excludes the discarded unfinished/stalled interval from
  trainer accumulated runtime; excluded overhead is recorded separately in
  recovery provenance. Do not present accumulated runtime as total wall time.

## Exact continuation

1. Keep the primary assistant's turn open; call `clock.sleep` for three hours.
2. Personally run `python .inspection/check_dkt_training.py` from repository root.
   Compare actual batch progress and its age, process identity, config/source,
   complete-epoch target counts, best/last checkpoint hashes and saved Adam/RNG.
3. If progress is stalled >7200s, use the standing authorization above. Capture
   at least two SIGUSR1 stacks from the new registered trainer and preserve them,
   snapshot the incomplete run, verify completed checkpoint and selection control,
   terminate only verified own trainer/workers, then run the unchanged CLI via
   `.inspection/resume_dkt_with_stacks.py ... --workers 8 --transport packed64 --resume`.
   Update runner.pid/control identity and provenance; verify resumed batch advancement.
   Do not repeat from scratch or change data/model/precision/optimizer/protocol.
4. At completed epochs verify and publish checkpoint/log/config/memory milestones.
   Use `python .inspection/update_dkt_check_memory.py` for actual scheduled checks,
   validate memory, inspect scoped Git diff, commit and push `HEAD:main` to capstone-gpu.
5. On _SUCCESS verify the frozen stopping/selection rules and complete test counts.
   Independently replay in a fresh CUDA process:
   `PYTHONPATH=. python scripts/evaluate_checkpoint.py experiments/dkt/ednet_kt1/full data/processed/ednet_kt1/full/session_store --workers 8 --output experiments/dkt/ednet_kt1/full/independent_evaluation.json`.
6. Require exact replay/saved-test equality, then run
   `PYTHONPATH=. python scripts/generate_report.py --experiments-root experiments --output-root reports`.
   Review all15 actual complete pairs and figures. Finish `../prompt.txt` documentation,
   update memory, commit/push verified outputs and provide the final handoff.

## Boundaries and historical evidence

- No fabricated results, silent data sampling, architecture changes or changes to
  seed/context/batch/FP32 arithmetic/optimizer/strict patience5/min_delta0.
- Diagnostics and resumed batches are not final benchmark outcomes.
- Keep other14 experiments and Achin's platform work intact.
- The old exact handoff is preserved in
  [historical next checkpoints](NEXT_CHECKPOINT_history_before_2026-10-09_resume.md).
  Its pending-approval and older monitoring instructions have been superseded.
