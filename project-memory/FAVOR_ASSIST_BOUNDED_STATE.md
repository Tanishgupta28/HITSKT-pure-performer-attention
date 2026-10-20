# FAVOR+ bounded ASSIST study — 20 October2026

Latest task continues eac2a375, design committed b45906a before measurements,
executed source1966fa1. User authorizes conditional max100 TOTAL updates/45min,
B1/cost2048/m128/seed42/original Adam/architecture/data/protocol, with review at20
then50 then100; no test/full training. Existing owner-confirmed H100 allocation
permission persists; do not ask administrator again solely for Kubernetes403.

## Established evidence

- CPU phase1 v2: six training-only length/history strata, old saved5 checkpoint
  vs fresh seed42;1152 frozen128/256/512 × projection-seeds0..7 comparisons.
  CPE deployed median relative-softmax error0.136497→0.198126/max0.222793;
  original single-example0.181169→0.236181 REVIEW_REQUIRED/.20 unchanged.
  Independent production-feature reference max8.315e-7 PASS. Q/K cross-combinations
  explain rising approximation; V-only little effect, ESS decreases.512 improves
  median but not all tails. No tested numerical implementation defect; high-norm
  scientific adequacy unresolved. Only bounded diagnostic continuation supported.
- New frozen validation sample:32 validation targets/3157 labels (1063positive,
  2094negative),29 students; original retrospective within-student rolling visibility.
  Six diagnostic training sessions/734 labels. No test targets evaluated.
- Resource preflight11:48:08UTC exact owner MIG-d2fcfd79-720e-5218-82c9-0a132be68920
  idle/no visible conflicting processes, LightSAKT2 stopped and hold active, RAM90GiB.
  Existing18-case CUDA scientific source/oracle/harness/guard hashes unchanged/PASS.
- One new CUDA run assist_bounded100_20261020_v1_s020 completed14 successful Adam
  calls/752 targets, all gradients/model/Adam finite, all parameter deltas nonzero.
  After step14 resource gate rejected GPU-handle PID2751270; gone before identity
  inspection, ownership/purpose NOT VERIFIED. No override/retry/reset/signals.
  Failure before20; NO20/50/100 checkpoint/validation/continuation, NO new checkpoint
  at14, in-memory state lost on safe exit. Do NOT silently start a replacement run.
- Initialization sampled validation BCE0.742138526/AUC0.505231540/accuracy0.525815648;
  train-diagnostic eval BCE0.781498117. These are untrained exploratory scores only.
  Steps1→14 minibatch BCE1.100964904→0.715913832; target-weighted0.794004521.
  Gradient L2 range1.98417–7.85442. No fixed-sample improvement/convergence evidence.
- Recorded allocator max270.119MiB, training intervals2.666099s/stage10.035558s.
  Full final RSS/reserved high-water NOT RECORDED; baseline RSS1.471GiB/reserved154MiB
  are lower bounds. Future failure telemetry needs improvement.
- Discovered old five-pilot auxiliary fixed_training_example_eval_bce was actually
  training-mode/dropout: helper restored train mode before extra no_grad forward.
  Original report's 'without dropout' curve interpretation superseded in new report.
  Previous training losses and actual attention diagnostics remain valid, old files
  unchanged, no retraining implied. New validation explicitly eval/no RNG consumption;
  diagnostic RNG difference explains different first-five optimization trajectory.
-102 targeted CPU cases PASS (32new+70existing), zero final failures. Full229 inherited.
  Earlier fixture type mismatch/preflight helper field/classifier mistakes preserved.
  Prior70 pilot files freshly rehashed unchanged incl2 local checkpoints/24 arrays;
 11 processed ASSIST hashes checked; old source/Light hold/unrelated edits unchanged.
- Final11:52:42UTC release: no visible GPU holders/Light trainer, hold same hash,
  MIG used39,714,816 baseline, cumulative OOM2 unchanged. No background jobs/watchers.

## Decision and exact next action

REVIEW another bounded attempt; NO-GO full training. Need conflict-free interval
and immediate safe resource-conflict identity/final RSS-reserved telemetry, fresh
source-bound plan and separately explicit new budget/version after resource abort.
Do not weaken unknown-process gate or require a new allocation unnecessarily.
Next GPU study must establish20/50/100 learned-state approximation/ESS/causality,
frozen sampled validation and fresh-process model/Adam/RNG/cursor continuation
before discussing full training. Original15-history/B64/cost32768 backward shapes,
Junyi/EdNet real-data checks and full validation-selected benchmark evidence remain.
No Light restart, DKT changes, original model/data/results edits or main merge.

Report: `reports/favor_plus_v1/ASSIST_BOUNDED_TRAINING_VALIDATION_PILOT.md`.
Evidence: reports/favor_plus_v1/assist_bounded_v1/; experiments/hitskt_favor_plus_v1/
assist2017/assist_bounded_review_20261020_v2 and assist_bounded100_20261020_v1_s020.
Prior CPU v1 preparation retained, final v2 binds stricter hold/containment controls.
Scoped publication only to capstone-gpu research/hitskt-favor-plus-v1; preserve
unrelated README/SAKT log. Next work requires user decision after reviewing abort.
