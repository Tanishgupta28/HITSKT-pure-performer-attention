# FAVOR+ real-data pilot CPU preparation

19 October2026 UTC; research/hitskt-favor-plus-v1, base1e6255c.
Latest user authorizes CPU preparation only and explicitly requires new approval
before ANY GPU execution/resource changes. This supersedes earlier conditional
auto-execution permission. No GPU probes, CUDA calls, actual model training,
watchers, allocation changes, LightSAKT2 interference or DKT changes this turn.

New ktbench/favor_plus_real_pilot.py, favor_plus_gpu_access.py,
favor_plus_pilot_diagnostics.py and scripts/pilot_hitskt_favor_plus_real.py;
updated staged CUDA harness safety/provenance/quota only. Genuine FAVOR+,
ELU+1 models and existing full/synthetic pipelines unchanged. Guard requires
actual approval action/reference plus externally assigned independent exclusive
UUID/owner/job/window/evidence/exactCUDA_VISIBLE_DEVICES, rejects protected MIG
and whole parent. Metadata cannot manufacture approval or allocation.

Real ASSIST pilot: original hierarchy/store/splits/history/collator/loss/Adam/LR,
seed42/features128/chunk32/FP32, B1/cost2048/max5 updates, no test/val access or
resume. Updated CUDA PASS for same UUID/profile/features/source/oracle/checkpoint
required before execution. Prepared diagnostics cover finite losses/all grads/
Adam, QK norms, feature concentration, independent kernel/exactsoftmax error on
<=32-position fixed-input eval prefixes, GPU peaks/throughput and model/Adam/RNG
reload. Hard kernel5e-5 unchanged; approximation/norm review flags separate.
8GiB allocator quota,4GiB process RSS,2GiB cgroup headroom,120s/update/900s
cooperative deadline plus future owned-process outer timeout. No full-training
approval from five steps, even if finite. High-norm approximation remains a risk.

Final lightweight CPU validation:45 new+11 targeted existing=56 PASS,0failures,
1 intentionally deselected legacy full-pipeline test. CUDA APIs globally trapped
in final process, empty visibility/one low-priority CPU core. Prior229 suite not
rerun. Initial fixture lacked session_length:4 failures/39 passes preserved,
fixed to match real store, then45 new PASS and final56 PASS. No CPU optimizer
training; fake update counters and tiny synthetic attention forwards only.

Actual ASSIST data preflight with model/Adam constructors/CUDA APIs trapped:
all11 processed hashes match verified reference;7060 post-first train targets,
five selected whole targets292 actions, costs363/290/485/91/241. Zero real-data
forwards/updates. Valid plan under experiments/hitskt_favor_plus_v1/assist2017/
real_pilot_prepare_cpu_20261019_v2/prepared_plan.json,
sha2561c03b9d1499c738877270e2d9ca46acced8982b006497c675dcc20cafe67f5d1.
V1 plan preserved, now correctly refuses after final oracle fingerprint change;
v2 re-prepared same samples. Preparation/reload0.84s after imports, RSS0.543GiB;
not training/GPU measurements. Samples cover1–7 histories/cost<=485, not worst
15-history/2048 shape. All new outputs remain variant namespace.

Report reports/favor_plus_v1/REAL_DATA_PILOT_PREPARATION.md; logs/XML/template/
verification summary in real_pilot_preparation_v1/ under same reports directory.
CUDA and real optimizer pilot NOT RUN, full training NOT ESTABLISHED; no AUC.
Core/previous report identities and unrelated edits preserved; no redundant
375-file re-audit. Current independent allocation still not established; last
resource inspection remains [FAVOR_RESOURCE_PLAN_STATE.md](FAVOR_RESOURCE_PLAN_STATE.md),
not a new GPU observation. Request separate exclusive allocation (16GiB+ planning
or separate40GB MIG/16GiB host); verify actual authority and get user approval.
Next: scoped validation/commit/push then stop. No automatic CUDA retry/watcher;
larger train/validation pilot/full training require separate decisions.
