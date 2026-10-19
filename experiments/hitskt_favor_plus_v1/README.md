# Isolated HiTSKT FAVOR+ v1 experiments

Use only `experiments/hitskt_favor_plus_v1/<dataset>/<fresh_run_id>/`.
Do not place FAVOR+ files inside the historical `experiments/hitskt/` tree.

`cpu_smoke_20261019_v1` is a two-step synthetic CPU check for each registered
dataset profile, with reduced vocabulary. It has no dataset AUC or full-training
result. `_SMOKE_SUCCESS` and `_PILOT_COMPLETE` do not mean a benchmark completed.
New checkpoint/prediction binaries stay local and are ignored here.

The default training CLI is CPU synthetic smoke. Full training is held pending
a separate decision. The pilot CLI requires an explicit device; a CUDA pilot
also requires an operator assertion that an independent GPU is available.
Do not allocate the occupied H100 MIG or disturb LightSAKT2/DKT experiments.

Implementation, commands, limits and evidence:
[validation report](../../reports/favor_plus_v1/IMPLEMENTATION_VALIDATION.md).
