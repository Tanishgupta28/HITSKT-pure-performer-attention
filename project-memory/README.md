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
- Experiment outputs: `../experiments/` (smoke diagnostics currently; no final
  benchmark result yet)

## Current short status

Full preprocessing and lossless HiTSKT batching are validated for all three
datasets. The pure Performer HiTSKT gate and the approved paper-faithful,
cross-fitted performance-only RKT smoke gate pass. Project seed 42 is
centralized and verified. Production runners for all five models pass their
bounded end-to-end gates under the common validation-AUC controller. Full
DKT/ASSIST2017, RKT/ASSIST2017, and HiTSKT/ASSIST2017 are complete,
independently reproduced from their best checkpoints, and occupy three of 15
generated master-result rows.
Full Phi caches and production-cache throughput gates pass for all datasets.
Full SAKT/ASSIST2017 is the next experiment.
