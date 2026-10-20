# LightSAKT2 priority handoff — 20 October 2026

HiTSKT branch `research/hitskt-favor-plus-v1`, base
`3bf92ee6488ece91cd92012dd8980016f1cca726`.

User now permits stopping only verified LightSAKT2 jobs, preserving all work and
preventing restarts; then actual exclusively allocated GPU CUDA checks and <=5
ASSIST2017 train-only updates after correctness PASS. This supersedes older
no-interruption/repeat-approval rules for this scope only. No full training,
unrelated termination, DKT/EdNet recovery or invented allocation.

Established:

- LightSAKT2 already interrupted before this turn:813 complete/142 pending/
  2 interrupted/0 live training. No signals sent. Known launch paths held by
  `experiments/USER_TRAINING_HOLD.json`;9 targeted CPU checks pass.
- 7915 scientific files/1,391,693,255 bytes match before/after;814 recorded result
  checkpoint references and declared predictions/configs verified. Interrupted
  GKT epoch2 CPU model/Adam/RNG/data/config checks pass; separate local archive.
  No new offsite download verification. Waiting GKT has no saved epoch.
- LightSAKT2 controls/report/memory pushed at
  `3debdc21ce2380ad4dd3bfaa8b5f14eea208eae4` on `research-rebuild`.
  Substantial older dirty research state remains untouched/uncommitted.
- Only exposed H100 MIG still has unrelated worker handles outside both roots.
  No authoritative HiTSKT allocation; Kubernetes403/no scheduler. Never signal
  those jobs. NVML free memory is not permission. CUDA/pilot NOT RUN.
- HiTSKT model/data/evaluation source unchanged.30 originally captured checkpoint
  hashes and19 prepared-source hashes freshly match. Prior229 core/56 preparation
  tests inherited, not rerun; no new FAVOR+ numerical results.

Evidence: `reports/favor_plus_v1/LIGHTSAKT2_HANDOFF_AND_GPU_PILOT_STATUS_20261020.md`,
`reports/favor_plus_v1/priority_handoff_20261020/{handoff_evidence,resource_and_process_evidence,hitskt_preservation}.json`.
Detailed Light evidence remains in `/workspace/LightSAKT2/reports/priority_handoff_20261020/`;
its new durable memory is separate from the older LightSAKT_Project study.

Next: stop for actual allocation-owner/admin evidence (UUID/owner/job/time-window,
external authority, no unrelated reservations). Question requested; no evidence
received at this checkpoint. Same MIG remains hard-blocked in existing guard;
verified reassignment needs a narrow tested guard update and new source-bound
preparation. Do not fabricate a lease, poll automatically or launch GPU work
before ownership is established.128 features remains diagnostic only; high-norm/
learned-QK/GPU memory/performance/full-training readiness unresolved.
