# Current state

Updated: 2026-10-19 UTC

## FAVOR+ CPU implementation verified — latest

Genuine FAVOR+ from scratch on research/hitskt-favor-plus-v1 from bc262e6:
74 new FAVOR+ tests and143 legacy controls PASS (217 distinct,0 failures).
Separate model/pipelines, original attention-only boundary, all3 CPU synthetic
profiles and fresh checkpoint replays verified. Long attention13081 peak0.786GB;
full hierarchy13080-action target peak1.751GB.375 originals/all30 checkpoints
and277 prior evidence/source files unchanged. No CUDA/full dataset training.
Candidate for independently available GPU pilot; GPU performance/learned-norm
approximation remain unverified, high-norm synthetic error documented.
See [FAVOR_PLUS_STATE.md](FAVOR_PLUS_STATE.md). Implementation 67f09d2291e632398826af21e5f731f8762a05f9
is pushed/remote-verified on the research branch; main unchanged. Wait for execution direction. No attachment dependency, LightSAKT2/DKT
changes, availability-aware V2 or original retraining.

## Supervisor briefing — historical checkpoint

Two-page supervisor PDF plus editable Markdown/HTML, verification status and
bounded replay plan are verified. See [SUPERVISOR_STATE.md](SUPERVISOR_STATE.md).
Fresh saved-score recalculation matches all15,998,524 CPU targets;59 relevant
controls PASS again;375 originals and156 previous verification files unchanged.
Corrected a baseline-context description in new documents:DKVMN199/RKT49.
Older reports and all scientific originals remain intact. No training or CUDA
allocation. Only visible MIG occupied; LightSAKT2 active and protected.
All3 full CPU metric gates still FAIL; CUDA/timing/publication NOT VERIFIED.
Complete scoped publication/remote check, then await independent resources or
instructions. Do not start DKT, V2, FAVOR+ or another experiment schedule.

## Completed follow-up diagnosis — historical

The user authorized the reproducibility/timing follow-up after audit5625478.
See [REPRODUCIBILITY_STATE.md](REPRODUCIBILITY_STATE.md). Three requested reports
and separate availability-V2/FAVOR+ proposals are verified. All59 relevant tests
pass;375 originals and106 previous audit files remain unchanged. No training,
GPU allocation or LightSAKT2 edits/signals. No active diagnostic jobs remain.
All3 historical CPU metric gates still FAIL; full ASSIST FP64 does not resolve
BCE. CUDA replay and actual response availability remain NOT VERIFIED. Keep
historical V1 evidence provisional; no established need to retrain yet.
Finish scoped commit/push, verify remote equality, then wait. Do not automatically
start DKT, V2, FAVOR+, multi-seed runs or CUDA work on the occupied MIG.

## Completed independent audit — historical

The audit completed all three full CPU replays and exhaustive data reconstruction.
**133 tests PASS; all three metric reproduction gates FAIL** at declared 1e-6,
while all AUC differences pass that tolerance. Response-arrival timing, exact CUDA
reproduction and significance remain NOT VERIFIED. The active kernel is ELU+1,
not FAVOR+. 375 original file hashes and all 30 checkpoints are preserved.
See [AUDIT_STATE.md](AUDIT_STATE.md) and the
`reports/HITSKT_INDEPENDENT_CORRECTNESS_AUDIT.md`.

No production training was resumed. LightSAKT2 was not edited or signalled.
After scoped commit/push, wait for user instructions. Reset-only notes below
are historical; their evaluation hold was superseded for this completed audit.

## Historical reset reconciliation and established outcome

The GPU was reset. The user requested memory reconciliation, GitHub verification
and pending-commit synchronization only, then waiting. **Do not start the 15th
training.** Historical three-hour checks and >2h automatic recovery are on hold
under the latest [user directive](USER_DIRECTIVES.md). No trainer, evaluation
runner or monitor was launched during that reset-only stage; no capstone training
process was found.

The benchmark remains **14/15**. All 14 canonical completed directories have
`_SUCCESS` and match `reports/final_results.json`. Prior independent verification
is retained from memory, not rerun here. DKT/full EdNet has neither `_SUCCESS`
nor `final_results.json`; no DKT EdNet test outcome is available.

## Reconciled DKT/full EdNet interruption

The October 10 handoff was stale: it described five complete epochs and live
epoch6. Actual `experiments/dkt/ednet_kt1/full/` artifacts contain:

- Nine contiguous complete epochs, with exact CSV/JSON metric equality and all
  53,990,022 training / 14,520,975 validation targets in every epoch.
- Best checkpoint: epoch1, validation AUC **0.6845717850759062**, byte-identical
  to the already committed best. SHA256:
  `56a911479a2c0b53edd457e4179a942be6f652717e7322c35b41e45194e90215`.
- Last completed checkpoint: epoch9, validation AUC0.6108838021742428,
  best1 and non-improvement counter **8/10**. SHA256:
  `ae9f8248669c3800064020cbbb64c4e82ade108cac0cef27e1a23c74f208842b`.
- Stale `progress.json` / console tail: epoch10 training, 36,418,836 of
  53,990,022 targets (**67.454753%**), 1,822,720 minibatches, last update
  **2026-10-12T04:40:41Z**. This is unfinished work, not a complete epoch.
  Old `runner.pid`999437 is historical; there is no matching trainer now.
- Saved accumulated completed-epoch runtime255,195.88471189688 seconds.
  It excludes unfinished epoch10 and previously documented discarded/stalled
  intervals; do not report it as total elapsed wall time.

Primary evidence is `reports/performance/dkt_reset_reconciliation_2026-10-18.json`.
The frozen config remains seed42, width64, context200/history199, batch20,
Adam0.0002, FP32, workers8, packed64 transport, ceiling200, strict validation-AUC
improvement/min_delta0 and best reload before test. Patience10 is authorized
only for DKT/full EdNet; every other run/default remains5.

## Checkpoint integrity finding — unresolved

CPU inspection loaded both checkpoints and verified finite model weights,
saved Adam state and Python/NumPy/CPU/CUDA RNG entries. The **epoch9 Adam state
contains 21,749 positive infinities in five `exp_avg_sq` tensors** (embedding,
both LSTM weights and both LSTM biases). The best epoch1 checkpoint has finite
model and optimizer tensors. The initial all-finite integrity check therefore
failed on the last checkpoint. Detailed counts are in the reconciliation JSON.

Do not describe epoch9 as fully validated for resumption. Cause and appropriate
recovery remain unresolved; no training, state reset, clipping, checkpoint
rewrite or scientific change was attempted. A future continuation must address
this finding with the user before choosing a consequential recovery policy.

Both checkpoint payloads also embed common patience5 metadata because
`scripts/train_baseline.py:_checkpoint_payload` serializes
`COMMON_EARLY_STOPPING`. Actual config and epochs6–9 retain approved10. Source
inspection confirms resume creates the controller from saved config10, then
restores best/counter8. Do not reduce patience or reset the counter based on
embedded metadata. This was recorded without changing source/checkpoints.

## GitHub, skill and retained local state

- `gh auth status`, authenticated repository API and fetch succeeded as
  **Tanishgupta28** with push permission on canonical `capstone-gpu`:
  `https://github.com/Tanishgupta28/HITSKT-pure-performer-attention.git`.
  Before this sync, local `work` and fetched `main` both pointed to`0964a33`;
  no existing commits were ahead or behind. Epoch6–9 config/CSV/JSONL were
  uncommitted and are included in this reconciliation milestone.
- Git LFS was absent after reset and required by the existing pre-push hook.
  Restored git-lfs3.0.2; hook retained. Never force push.
- Durable memory skill was absent in searched skill roots; restored from the
  user-supplied repository to `/root/.codex/skills/durable-agent-memory` and
  applied in resume/update mode. It will be discoverable on the next turn.
- Scientific source remains unchanged. Trainer SHA256:
  `f1a3f89364728a598a67bba9684970ff64709b61a4ecebd15a381ae562192beb`.
- Preserved unrelated local `README.md` modification and untracked
  `experiments/sakt/ednet_kt1/full/independent_evaluation.console.log`;
  exclude both from this scoped commit.
- `last_model.pt`, raw/processed data, console/progress/PID and inspection
  snapshots remain local/ignored. Interruption snapshot:
  `.inspection/dkt_reset_20261018/` (best/last, config/stats, CSV/JSONL, progress).
  The last checkpoint is not recoverable from the public commit alone.
- H100/MIG3g.40gb inventory is visible. GPU memory queries lack permissions;
  CUDA training/readiness was not exercised. The full EdNet session-store
  metadata is present; its hash is recorded in the reconciliation JSON.

## Platform and continuation

Achin's FastAPI/Next.js/MongoDB platform work remains intact. The matching
content/checkpoint evidence request is still pending; see
[PLATFORM_STATE.md](PLATFORM_STATE.md) and [PLATFORM_HANDOFF.md](PLATFORM_HANDOFF.md).
No platform implementation or serving work is part of this reset reconciliation.

After publishing this scoped milestone, wait for the user. Exact future gates
are in [NEXT_CHECKPOINT.md](NEXT_CHECKPOINT.md). Prior beliefs and instructions
are retained in [October 10 state](CURRENT_STATE_history_before_2026-10-18_reset.md),
[earlier state](CURRENT_STATE_history_before_2026-10-09_resume.md) and
[DECISION_LOG.md](DECISION_LOG.md); their live PIDs/schedules are historical.
