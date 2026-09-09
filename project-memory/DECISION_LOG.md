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
