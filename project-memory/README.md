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

2026-10-10:14/15 pairs complete and independently verified. DKT/full EdNet
has three verified completed epochs. The user-approved epoch2 recovery
completed epoch3 (validation AUC0.6785532792), and epoch4 is advancing.
Best1/AUC0.6845717851 remains selected; patience2/5. TrainerPID787181.

The primary assistant personally checks after three-hour sleeps in the active
turn. The user authorizes recovery whenever verified progress is stalled for
more than two hours; no further confirmation is needed. No detached monitor
or subagent. Scientific settings remain frozen; exact old stall cause unknown.
Read [CURRENT_STATE.md](CURRENT_STATE.md), [USER_DIRECTIVES.md](USER_DIRECTIVES.md),
and [NEXT_CHECKPOINT.md](NEXT_CHECKPOINT.md). Recovery provenance is
`../reports/performance/dkt_resume_epoch2_2026-10-09.json`; scheduled checks are
in the monitoring ledger. Historical state/handoffs are retained in
[previous state](CURRENT_STATE_history_before_2026-10-09_resume.md) and
[previous handoff](NEXT_CHECKPOINT_history_before_2026-10-09_resume.md).
Public pushes/Tanish identity remain authorized; unrelated README edit preserved.
