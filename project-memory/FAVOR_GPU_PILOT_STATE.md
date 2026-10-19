# FAVOR+ GPU pilot resource checkpoint

19 October2026 UTC, from d7399a0, research/hitskt-favor-plus-v1.
User authorizes128-feature CUDA verification only after independent exclusive
assignment, then <=5 ASSIST2017 train updates (seed42/B1/cost2048). No full
training/test evaluation/HPO. This authorization persists; no re-approval is
needed once resource and required CUDA gates pass. See
[USER_DIRECTIVES.md](USER_DIRECTIVES.md).

Current result: RESOURCE BLOCKED, CUDA NOT RUN, real ASSIST pilot NOT RUN
(0 updates), full-training readiness NOT ESTABLISHED for all3 datasets.
At14:49 UTC one existing H100 MIG visible, usage/PID fields restricted.
No task allocation, scheduler assignment or external independent lease.
LightSAKT2 heartbeat14:51:05 observed14:51:35 reports811 complete/145 pending/
1 running, active ASSIST2015-GKT baseline_tuning seed42 job (last epoch3).
Processes182135/2325921 in LightSAKT2 have NVIDIA handles and module scripts.run.
Their instantaneous sleeping state does not imply GPU idle/available.

Evidence reports/favor_plus_v1/gpu_pilot_readiness_20261019_v1/
resource_and_process_evidence.json; required report
reports/favor_plus_v1/GPU_PILOT_AND_TRAINING_READINESS.md.
Prior229 CPU test result/source identities verified, not rerun or relabeled
CUDA. Preservation PASS375 originals +407 prior source/report/output files,
including previous smoke checkpoints and frozen arrays. Unrelated README/SAKT
log unchanged. No production code/dataset/checkpoint/results changes.
Resource/preservation checks used stdlib only, no PyTorch import, CUDA probe,
CUDA context or tensor. No protected signals/edits, DKT resume/reset, watcher
or repeated background polling. OOM kill count2 unchanged.

Missing input: administrator/provider must independently assign an exclusive
GPU/MIG UUID with owner/job/externally verifiable assignment and validity
window, expose it to container, CUDA_VISIBLE_DEVICES exactly that UUID. Budget
for8GiB test allocation cap plus context/headroom;16GiB+ or separate40GiB MIG
is a planning choice, not measured minimum. Current protected MIG must not be
repurposed/reconfigured while LightSAKT2 runs. Free memory/process absence or
self-authored exclusive flag cannot prove assignment.

After allocation: staged scripts/validate_favor_plus_gpu.py with128 and explicit
assist2017 profile, short-to-long admitted checks, stop on failure. Only after
required CUDA PASS, implement/review explicit5-update real train-only loop with
original SessionStore/collator/architecture/history and record full requested
pilot diagnostics/checkpoint. Existing pilot CLI is synthetic; never use full
training or epoch override as a step-limited pilot. CUDA batch-memory/norm
learning/approximation/throughput and checkpoint behavior remain unverified.
High-norm approximation poor even512; five steps cannot prove convergence/AUC.
See [FAVOR_QUALITY_STATE.md](FAVOR_QUALITY_STATE.md) for preserved scientific evidence.

Publish report/evidence/memory research branch only, verify remote, then wait
for allocation. No automatic GPU rechecks or full training.
