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

Full preprocessing and lossless HiTSKT batching are validated for all three
datasets. The pure Performer HiTSKT gate and the approved paper-faithful,
cross-fitted performance-only RKT smoke gate pass. Project seed 42 is
centralized and verified. Production runners for all five models pass their
bounded end-to-end gates under the common validation-AUC controller. All five
ASSIST2017 and all five Junyi model runs plus full DKVMN, RKT, and HiTSKT EdNet
are complete, independently reproduced from their best checkpoints, and occupy
thirteen of 15 generated master-result rows. Full Phi caches and production-cache
throughput gates pass for all datasets. Full HiTSKT/EdNet stopped at epoch 35,
selected epoch 30, and achieved test AUC 0.7693648364413692 over 13,429,870
targets; exact fresh-process metric replay passed. Exact resume support passes a tensor/metric equality
test; all 60 tests pass. Already stopped runs resume directly into final test
without an extra epoch, verified across all five models. Final table/plot
generation and independent checkpoint
replay commands are implemented/tested, and all 13 real completed artifacts
pass report validation. Continue with the remaining EdNet SAKT and DKT
runs using two-hour monitoring (latest approval 2026-09-20), original Tanish commit
identity, and the existing remote. GitHub authentication was restored and
verified on 2026-09-19; resume checkpoint pushes to the existing private
Tanishgupta28/capstone-gpu repository. **2026-09-21 update:** that remote now
redirects to public `Tanishgupta28/HITSKT-pure-performer-attention`; further
pushes are held pending the user's visibility decision. Keep local commits and
the active training run intact.
