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
