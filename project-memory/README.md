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

2026-10-09:14/15 completed pairs. DKT/full EdNet has two verified completed
epochs. Epoch3 is stalled at88.52% of its training pass, with no batch update
since06:30UTC despite a live process. Fresh CUDA and saved epoch2 execution pass;
the exact live-process cause is unknown because stack inspection is denied.

The primary assistant personally checks after three-hour sleeps in the active
turn. No subagent or detached monitoring supervisor. A user decision is pending
before discarding unfinished epoch3 and resuming from verified epoch2; no restart
has occurred. Read [CURRENT_STATE.md](CURRENT_STATE.md),
[USER_DIRECTIVES.md](USER_DIRECTIVES.md), and [NEXT_CHECKPOINT.md](NEXT_CHECKPOINT.md).
Evidence: `reports/performance/dkt_stall_2026-10-09.json` and the monitoring ledger.
Public pushes/Tanish identity remain authorized; unrelated README edit is preserved.
