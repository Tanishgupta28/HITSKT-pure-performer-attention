# HiTSKT capstone project memory

This directory is a compact, evidence-backed continuation index for the HiTSKT
capstone. The Git working tree and experiment artifacts remain the source of
truth; this memory never substitutes for code, logs, checkpoints, or results.

## Parallel platform workstream — 2026-10-08

Achin is building FastAPI/Next.js/MongoDB under `D:\CS\HITSKT-platform` in
this same repo. Read [PLATFORM_STATE.md](PLATFORM_STATE.md) and
[PLATFORM_HANDOFF.md](PLATFORM_HANDOFF.md) for its scope, validation and model
integration request. Existing GPU training ownership and scientific rules
remain intact; platform changes do not alter research source or live runs.

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
supervisor `509095` records three-hour checks and automatically performs
independent replay and strict15/15 report generation after training succeeds.
All77 tests pass, including exact equivalence of progress logging.

Read [CURRENT_STATE.md](CURRENT_STATE.md) and
[NEXT_CHECKPOINT.md](NEXT_CHECKPOINT.md) for authoritative current actions.
Historical PID/resume instructions in older entries are superseded.
`reports/performance/dkt_restart_2026-10-08.json` records this restart's evidence.
Live runtime files and the old abandoned artifacts remain local and ignored.
The latest three-hour cadence and supervisor handoff are recorded in
`reports/performance/dkt_monitoring_3h_2026-10-08.json`.
The unrelated root README edit is preserved. Public pushes remain authorized
using the configured original Tanish identity.
