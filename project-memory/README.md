# HiTSKT capstone project memory

This directory is a compact, evidence-backed continuation index for the HiTSKT
capstone. The Git working tree and experiment artifacts remain the source of
truth; this memory never substitutes for code, logs, checkpoints, or results.

## Resume order

1. Read [USER_DIRECTIVES.md](USER_DIRECTIVES.md).
2. Read [CURRENT_STATE.md](CURRENT_STATE.md).
3. Read the newest entries in [DECISION_LOG.md](DECISION_LOG.md).
4. Follow [NEXT_CHECKPOINT.md](NEXT_CHECKPOINT.md).
5. Reconcile every claim with the linked workspace evidence before acting.

## Authoritative roots

- Project repository: `/workspace/capstone`
- Project specification: `../prompt.txt`
- Upstream continuation point: `origin/main` from `pokerme7777/HiTSKT`
- Experiment outputs: `../experiments/`; only directories carrying `_SUCCESS`
  are eligible for the generated final benchmark tables

## Current short status

2026-10-08: **14/15 completed pairs**. The user explicitly requested a fresh
restart of the fifteenth experiment, DKT/full EdNet, without restoring its old
checkpoint. Fresh CUDA trainer `506822` is advancing in epoch1; detached
supervisor `506821` records six-hour checks and automatically performs
independent replay and strict15/15 report generation after training succeeds.
All77 tests pass, including exact equivalence of progress logging.

Read [CURRENT_STATE.md](CURRENT_STATE.md) and
[NEXT_CHECKPOINT.md](NEXT_CHECKPOINT.md) for authoritative current actions.
Historical PID/resume instructions in older entries are superseded.
`reports/performance/dkt_restart_2026-10-08.json` records this restart's evidence.
Live runtime files and the old abandoned artifacts remain local and ignored.
The unrelated root README edit is preserved. Public pushes remain authorized
using the configured original Tanish identity.
