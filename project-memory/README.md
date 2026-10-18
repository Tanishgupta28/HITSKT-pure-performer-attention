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
- Project specification: `/workspace/capstone/prompt.txt`
- Upstream continuation point: `origin/main` from `pokerme7777/HiTSKT`
- Experiment outputs: `/workspace/capstone/experiments/`; only directories carrying `_SUCCESS`
  are eligible for the generated final benchmark tables

## Current short status

The independent correctness audit is complete; see [AUDIT_STATE.md](AUDIT_STATE.md)
and `reports/HITSKT_INDEPENDENT_CORRECTNESS_AUDIT.md`. 133 tests and
full data reconstruction pass; all three CPU metric reproduction gates fail.
Exact CUDA/timing/significance evidence remains incomplete. Publish the scoped
audit commit, then wait; production training remains on hold and LightSAKT2 stays
outside scope.

2026-10-18 reset reconciliation: **14/15** pairs complete. DKT/full EdNet
has **nine** complete epochs; epoch10 stopped at67.45% of training on October12.
Best1/AUC0.6845717851, saved counter8/10. No DKT trainer is active.
**User says do not start the 15th training; sync, then wait.** This supersedes
older automatic monitoring/restart directions for this session.

Epoch9 model weights are finite, but five saved Adam second-moment tensors
contain21,749 positive infinities. Recovery/readiness is unresolved; no repair
or training was attempted. Detailed evidence:
`reports/performance/dkt_reset_reconciliation_2026-10-18.json`.

GitHub authentication/API/fetch work as Tanishgupta28. Restored missing Git LFS
and the requested durable memory skill after reset. The scoped sync includes
uncommitted epoch6–9 config/log/metric artifacts and current memory. Preserve
the unrelated root README edit and SAKT console log; last checkpoint/data and
inspection snapshots remain local/ignored.

Read [CURRENT_STATE.md](CURRENT_STATE.md), [USER_DIRECTIVES.md](USER_DIRECTIVES.md)
and [NEXT_CHECKPOINT.md](NEXT_CHECKPOINT.md). Prior live-run handoffs are retained
in [October 10 state](CURRENT_STATE_history_before_2026-10-18_reset.md) and
[October 10 checkpoint](NEXT_CHECKPOINT_history_before_2026-10-18_reset.md).
