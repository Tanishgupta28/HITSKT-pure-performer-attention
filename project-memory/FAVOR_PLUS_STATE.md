# FAVOR+ v1 research branch

19 October2026 UTC. PREPARATION VERIFIED; integration awaits missing input.
Branch:`research/hitskt-favor-plus-v1`, base:f465d84. Publish only this branch,
not canonical main. Latest [USER_DIRECTIVES.md](USER_DIRECTIVES.md) authorizes
new FAVOR+ code/pipeline and bounded CPU verification, not full training or
GPU pilot execution. V2 remains out of scope. Preserve all historical artifacts.

Required input: user-referenced attached starter implementation and standalone/
full-model tests. None is visible in the conversation, repo, common attachment
mounts, workspace archive or mounted resources. Asked for a local path/download
link asynchronously; no answer received at this checkpoint. Do not substitute
newly invented code or call the existing proposal an implementation.

Completed:
- New research branch created fromf465d84838e5f8b1c637736e362a945c20cfde10.
-375 originals/all30 checkpoints and268 additional prior source/verification
  files PASS baseline/handoff hashes; unrelated root README/Sakt log preserved.
-59 existing independent legacy causal/mask/gradient/audit controls PASS again,
  one low-priority CPU thread, CUDA hidden. These are not FAVOR+ tests.
- Integration boundary and explicit NOT RUN/NOT IMPLEMENTED statuses recorded
  in `reports/favor_plus_v1/INTEGRATION_STATUS.md`. Preservation JSON/logs and
  `legacy_cpu_controls.log` sit beside it. New preservation helper:
  `scripts/check_favor_plus_preservation.py`.

Not done: starter test run, full-model FAVOR+ integration, kernel optimization,
new training pipeline, FAVOR+ CPU smoke or GPU-readiness verification. Readiness
is NOT VERIFIED. Do not claim legacy tests establish genuine FAVOR+ correctness.
No CUDA, training, new experiment, LightSAKT2 modification/signal or DKT resume.
Actual cgroup90GiB; protected training active; previous OOM counter2 unchanged.

Next exact step: obtain/hash the actual starter and tests; inspect them and run
their unmodified standalone/full-model tests first, preserving failures. Then
implement isolated hitskt_favor_plus_v1, leaving non-attention components and V1
data/splits/protocol unchanged. Outputs under separate variant/run directories.
The integration-status document records the test/architecture/checkpoint contract.

Commit/push this verified preparation checkpoint normally to the research branch
and verify remote equality. No implementation-completion claim or automatic
training schedule. The missing source requires user input before dependent work.
