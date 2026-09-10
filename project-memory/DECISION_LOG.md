# Decision log

## 2026-09-09 — Restore the continuation repository in place

- Decision: initialize `/workspace/capstone` as a Git worktree and check out a
  local `work` branch tracking `origin/main`.
- Rationale: the directory had no repository, while the specification requires
  continuing from `pokerme7777/HiTSKT` and doing all work in this directory.
- Consequence: `prompt.txt` remains untracked and tracked upstream code remains
  unchanged during the mandatory assessment phase.

## 2026-09-09 — Bootstrap durable project memory

- Decision: use `project-memory/` following `Fyxod/durable-agent-memory`.
- Rationale: explicitly requested by the user for this multi-stage project.
- Consequence: material conclusions, decisions, failures, and exact handoffs
  will be recorded here with workspace evidence and validation.

## 2026-09-09 — Stop after assessment for consequential choices

- Decision: do not modify tracked model/training code or start experiments
  until the user resolves the ambiguities documented in the assessment.
- Evidence: [ASSESSMENT_2026-09-09.md](ASSESSMENT_2026-09-09.md).
- Rationale: `prompt.txt` explicitly requires a question instead of a silent
  assumption for dataset identity/version, column mapping, sessions, splits,
  conflicting implementations, and model protocol.
- Consequence: prior Drive logs remain reference evidence only and no reported
  value is accepted as a final benchmark result.

## 2026-09-09 — Freeze dataset, split, session, and model protocol

- Decision: the user approved all eight recommendations/questions recorded in
  the initial assessment. The explicit multi-tag directive names Junyi.
- Evidence: [USER_DIRECTIVES.md](USER_DIRECTIVES.md), section “Approved
  scientific protocol,” and the active user message.
- Consequence: genuine EdNet-KT1 acquisition and schema verification may begin;
  legacy Riiid/`ednetnew.csv` outputs remain excluded from final results. Skill
  mapping implementation must wait for clarification because official EdNet is
  multi-tag while the supplied Junyi schema is single-skill.

## 2026-09-09 — Publish private checkpoints to capstone-gpu

- Decision: create private repository `Tanishgupta28/capstone-gpu` and push
  cohesive code and project-memory checkpoints to its `main` branch.
- Evidence: explicit user authorization and successful GitHub push of commit
  `88aef9f`.
- Consequence: `origin` remains the read-only conceptual upstream and remote
  `capstone-gpu` is the project publication/checkpoint destination.

## 2026-09-09 — Extend composite-tag mapping to EdNet

- Decision: canonicalize every EdNet question's complete numerically sorted tag
  set and map it to one persisted composite skill ID. Do not expand rows or
  choose a primary tag.
- Evidence: explicit user clarification following the EdNet metadata audit.
- Consequence: the provisional canonical vocabulary has 1,495 values and is
  compatible with the existing embedding approach.

## 2026-09-09 — Map EdNet -1 to one untagged skill

- Decision: `tags = -1` means missing question tag metadata, is excluded from
  the genuine original-tag count, and maps to the single reserved composite
  skill ID `1` named `<UNTAGGED>`. Every interaction remains exactly once.
- Evidence: explicit user clarification following the sentinel audit.
- Consequence: preprocessing rejects every other malformed tag representation.
  The real-data smoke test consequently stopped on duplicated IDs in 197
  question tag strings; their treatment requires another explicit decision.

## 2026-09-09 — Deduplicate repeated EdNet question tags

- Decision: treat each positive tag list as a mathematical set, remove repeated
  IDs within the question, numerically sort, and generate one composite skill.
- Evidence: explicit user decision and the required canonicalization example.
- Consequence: the 197 affected questions are retained without interaction
  expansion. A subsequent real smoke run reached an empty `user_answer`, whose
  correctness treatment is not yet authorized.

## 2026-09-09 — Exclude unanswered EdNet rows before rebuilding sequences

- Decision: empty/missing `user_answer` rows are excluded from supervised model
  history and targets, never relabeled, and written exactly once to separate
  audit data. Sessions, splits, and sequence counters are rebuilt afterward.
- Evidence: explicit user decision and successful real first-1,000-student
  smoke validation under `data/processed/ednet_kt1/smoke-1000-v3/`.
- Consequence: the smoke supervised sequence is shorter by 2,898 interactions;
  session counts changed for 19 students and grouping changed for one. Any
  other response encoding remains a fail-closed condition.

## 2026-09-09 — Accept the validated full EdNet preprocessing build

- Decision: accept `data/processed/ednet_kt1/full/` as the local full EdNet
  event/audit build and publish only its mappings and aggregate reports.
- Evidence: `_SUCCESS`, `manifest.json`, full validator output, and versioned
  artifacts under `../reports/datasets/ednet_kt1/`.
- Consequence: EdNet event preprocessing is complete. Interaction Parquet is
  not committed; hashes and all deterministic metadata mappings are committed.

## 2026-09-09 — Stop on ASSIST2017 many-to-many question skills

- Decision: do not silently force one skill per ASSIST2017 question.
- Evidence: the accepted full file has 697 multi-skill questions covering
  440,761 interactions; see repository path
  `docs/data/assist2017_provenance.md`.
- Consequence: shared flat preprocessing is fail-closed until the user chooses
  row-level supplied skills or one globally constructed question-level
  composite set. Junyi preprocessing is also paused at this checkpoint.

## 2026-09-09 — Preserve ASSIST2017 row-level skills

- Decision: retain each interaction's supplied `skill`; do not infer a global
  question-level composite or expand rows.
- Evidence: explicit user decision and validated full ASSIST2017 artifacts.
- Consequence: ASSIST uses 102 row-level skills and records the 697 multi-skill
  questions as provenance, not as a transformation rule.

## 2026-09-09 — Accept validated full Junyi composite preprocessing

- Decision: use the full source to form each question's complete deduplicated,
  sorted tag set and assign its one deterministic composite ID to every row.
- Evidence: the approved Junyi multi-tag rule and full mapping/validator output.
- Consequence: 1,326 original tags map into 1,521 composite skills; all
  14,660,217 rows remain exactly once.

## 2026-09-09 — Stop before legacy HiTSKT action truncation

- Decision: do not generate fixed tensors using the legacy last-`action_size-1`
  truncation until the user reviews its measured data loss.
- Evidence: repository path `docs/data/session_window_audit.md`.
- Consequence: the validated event stores remain authoritative. No downstream
  session chunks or variable-length batches have been chosen yet.

## 2026-09-10 — Adopt lossless variable-length token-budgeted batches

- Decision: preserve every complete 10-hour session, use length buckets and
  per-batch dynamic padding, greedily cap normal batches at 32,768 padded token
  slots and 64 examples, and place an indivisible over-budget example alone.
  Do not truncate actions or chunk sessions.
- Evidence: explicit user directive, 28 passing tests, complete store counts,
  full split plans, and real H100 measurements in
  `../docs/data/variable_length_batching.md`.
- Consequence: the legacy fixed `action_size` loader is excluded from final
  HiTSKT experiments. The existing `session_size=16` semantics are retained as
  up to 15 chronological earlier sessions plus one complete target. The full
  EdNet worst case fits the available 40 GiB device at 24.989 GiB allocated and
  30.861 GiB reserved, so no methodological exception is needed.

## 2026-09-10 — Implement the paper-faithful performance-only RKT variant

- Decision: reproduce the authors' RKT architecture where supported, using raw
  directed performance Phi, paper width/dropout/positions, positive trainable
  student memory strength, learned fusion initialized at 0.5, and rolling 49
  prior interactions. Omit only the unavailable comparable text relation and
  its 0.8 combined-relation threshold.
- Evidence: authors' repository commit `cac60f512f`, arXiv `2008.12736`, the
  formulas and deviation table in `../docs/models/rkt_variant.md`, and passing
  component/smoke tests.
- Consequence: this experiment must carry the exact approved variant label. It
  is not interchangeable with the released executable defaults or a generic
  Transformer.

## 2026-09-10 — Cross-fit Phi at student level

- Decision: deterministically assign students to five seed-42 folds. For a
  training target, use cached contingency counts from the other four folds;
  for validation/test, use only the all-training cache. Keep each target once,
  cap history at 49, use latest prior occurrences, and map undefined Phi to
  zero.
- Evidence: `ktbench/rkt/phi.py`, `tests/test_rkt_variant.py`, and persisted
  ASSIST smoke artifacts. Full fold sizes are documented in the RKT report.
- Consequence: a target student's own fold and all validation/test interactions
  are structurally absent from the relevant Phi cache. Cached sparse counts
  preserve the exact leakage rule without recomputing a dense question matrix.

## 2026-09-10 — Centralize the benchmark seed at 42

- Decision: `ktbench.config.PROJECT_SEED` is the only primary benchmark seed.
  Seed Python, NumPy, PyTorch CPU/CUDA, fold assignment, sampling, and batching
  from it, and record it in configurations, logs, and result summaries.
- Evidence: seed tests, `reports/benchmark_config.json`, experiment config/log,
  and deterministic full-population fold audit.
- Consequence: the batching default is no longer zero and retained legacy
  training entry points call the centralized seeding function.

## 2026-09-10 — Pass the bounded RKT smoke gate

- Decision: accept the 20-student ASSIST2017 run only as a pre-training smoke
  validation, not as a benchmark result.
- Evidence: `../experiments/rkt/assist2017/smoke/` and 35 passing tests.
- Consequence: tensor shapes, one forward/backward update, target uniqueness,
  relation freezing, checkpoint round-trip, and all required metrics are
  verified. Full RKT Phi preparation/training has not begun.

## 2026-09-10 — Preserve baseline-specific rolling contexts

- Decision: use rolling 199-prior histories for DKT/DKVMN and 99-prior
  histories for SAKT, retaining their repository hyperparameters unchanged
  across all datasets. Every split event is one target; legacy non-overlapping
  chunks are excluded.
- Evidence: explicit user resolution, `../docs/models/baseline_rolling_protocol.md`,
  40 passing tests, and nine real-schema smoke artifact sets.
- Consequence: the three baselines intentionally have different contexts and
  hyperparameters, while seed, splits, chronology, masking, metrics, and
  validation-AUC checkpoint policy are standardized.

## 2026-09-10 — Record rolling-baseline full-scale cost before training

- Decision: do not start a large run until the missing common early-stopping
  patience is explicitly fixed and the measured rolling cost is reviewed.
- Evidence: `../reports/batching/baseline_rolling_throughput.json`; EdNet
  compute-only estimates are 8.169 h/epoch DKT, 81.614 h/epoch DKVMN, and
  4.953 h/epoch SAKT. A `torch.compile` DKVMN trial failed to complete initial
  compilation in over three minutes and was interrupted.
- Consequence: memory is not a blocker and no methodology was changed. The
  estimates exclude loading, validation, and I/O, so they are lower bounds.

## 2026-09-10 — Fix one validation-AUC early-stopping protocol

- Decision: all five models use patience 5, `min_delta=0`, strict validation
  ROC-AUC improvement, immediate stop at five consecutive misses, and best
  checkpoint reload before test. Existing epoch ceilings remain upper bounds.
- Evidence: explicit user decision, `ktbench/training.py`, controller tests,
  `reports/benchmark_config.json`, README, and updated experiment configs.
- Consequence: no runner may add a sixth non-improving epoch or use a
  model-specific patience. Every final artifact records best epoch/AUC and
  actual completed epochs.

## 2026-09-10 — Retain the authors' RKT gradient clipping

- Decision: use maximum gradient norm 10 for the RKT paper-faithful
  performance-only Phi variant.
- Evidence: the RKT paper is silent on this detail, the authors' released
  trainer uses 10, and the user explicitly approved retaining it.
- Consequence: smoke and production RKT paths share this setting and record its
  provenance in each configuration; it is not presented as paper-specified.

## 2026-09-10 — Complete all five production runner contracts

- Decision: use the common validation-AUC early-stopping/checkpoint controller
  in dedicated production paths for DKT, DKVMN, SAKT, RKT, and HiTSKT.
- Evidence: 46 passing tests, including bounded end-to-end RKT and HiTSKT runs,
  strict checkpoint reload, artifact fields, and the prior baseline fixture.
- Consequence: implementation gates are complete. The remaining pre-launch
  scalability gate is full RKT Phi preparation; smoke outputs remain
  diagnostic and no scientific run is yet claimed.

## 2026-09-10 — Accept the full RKT/ASSIST2017 result

- Decision: accept the completed seed-42 run as the first scientific benchmark
  result and generate the master row from its per-experiment artifacts.
- Evidence: 19 epoch records; strict best validation AUC 0.7712554524 at epoch
  14; five subsequent misses; best-checkpoint reload; 112,252 test targets;
  independent full test recomputation exactly matching every stored metric.
- Consequence: `reports/final_results.{csv,json}` contains 1/15 actual result
  rows. The diagnostic smoke and throughput artifacts remain excluded.

## 2026-09-10 — Accept the full HiTSKT/ASSIST2017 result

- Decision: accept the completed seed-42 lossless variable-session run as the
  second scientific benchmark result.
- Evidence: 45 epoch records; best validation AUC 0.7197572561 at epoch 40;
  exactly five subsequent misses; best-checkpoint reload; all 112,252 test
  actions; independent full test recomputation exactly matching every metric.
- Consequence: the generated master files contain 2/15 rows. The training
  target count is 566,107 because each student's first session supplies
  context but cannot be a hierarchical target without an earlier session; no
  action in any target session was truncated or chunked.

## 2026-09-10 — Accept the full DKT/ASSIST2017 result

- Decision: accept the completed seed-42 rolling target-once DKT run as the
  third scientific benchmark result.
- Evidence: unchanged batch 20/context 200; 17 epoch records; best validation
  AUC 0.7213696056 at epoch 12; exactly five subsequent misses; best-checkpoint
  reload; all 112,252 test targets; independent full test recomputation exactly
  matching every metric.
- Consequence: the generated master files contain 3/15 actual rows. DKT remains
  directly comparable under the standardized split/target/metric protocol
  while retaining its established model-specific context and hyperparameters.

## 2026-09-10 — Accept the full SAKT/ASSIST2017 result

- Decision: accept the completed seed-42 rolling target-once SAKT run as the
  fourth scientific benchmark result.
- Evidence: unchanged batch 10/context 100; 35 epoch records; best validation
  AUC 0.7577061439 at epoch 30; exactly five subsequent misses;
  best-checkpoint reload; all 112,252 test targets; independent full test
  recomputation exactly matching every metric.
- Consequence: the generated master files contain 4/15 actual rows. SAKT
  retains its established repository hyperparameters under the standardized
  split, target-once, masking, checkpoint, and metric protocol.

## 2026-09-10 — Execute DKVMN writes with balanced affine composition

- Decision: compute the unchanged DKVMN erase/add recurrence by composing its
  per-position affine writes in a chronological balanced tree instead of
  issuing 199 Python-driven GPU recurrence steps.
- Evidence: each write is exactly `M <- A*M + B`; composition uses
  `(A2,B2) o (A1,B1) = (A2*A1, A2*B1+B2)`. An explicit test compares the
  production reduction against the retained sequential reference across
  variable-length/PAD histories and matches outputs and every parameter
  gradient within tight floating-point tolerances. On full-history ASSIST
  batches, measured training-step time fell from 0.174721 s to 0.007266 s
  (24.0x), while peak allocation remained only 383,422,976 bytes.
- Consequence: model parameters, memory equations, chronological order,
  context, targets, optimizer, and hyperparameters are unchanged. Only the
  associative execution schedule is optimized, making the full benchmark
  tractable without changing scientific methodology.

## 2026-09-10 — Resume interrupted baseline runs from durable epoch state

- Decision: baseline production runs may use explicit `--resume` only when no
  `_SUCCESS` marker exists and all config/log/model/optimizer artifacts agree.
- Evidence: an end-to-end interruption fixture proves that resume restores the
  last model and Adam state, strict validation-AUC best and patience state,
  continues with the next epoch's deterministic seed-42 bucket order, appends
  prior metrics, and performs one final best-checkpoint test evaluation.
- Consequence: the externally interrupted DKVMN/ASSIST run can continue from
  completed epoch 19 without repeating targets from prior epochs or resetting
  its one-epoch non-improvement count. All 49 tests pass.
