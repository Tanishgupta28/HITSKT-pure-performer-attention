# Durable user directives

## 2026-09-09 — Capstone specification

The complete authoritative specification is `../prompt.txt`.
The durable constraints most likely to affect resumption are:

- Continue from `pokerme7777/HiTSKT`; do not create a disconnected rewrite.
- HiTSKT remains the primary research model and must use pure Performer-based
  hierarchical attention, not standard quadratic softmax attention.
- Inspect the GitHub repository and all three supplied Google Drive folders
  before changing code or starting long experiments.
- Ask the user before resolving any consequential ambiguity listed in the
  specification, especially data schema/version, preprocessing, splitting,
  session definition, model protocol, and evaluation choices.
- The final benchmark is five models (DKT, DKVMN, SAKT, RKT, HiTSKT) on three
  datasets (ASSISTment 2017, Junyi, full EdNet), using actual trained results.
- Do not substitute sampled or compressed EdNet for the final full-EdNet runs.
- Exclude PAD/EOS/special tokens consistently from all loss and metric targets.
- Use leakage-resistant sequential evaluation, validation-AUC checkpointing,
  reproducible configs, per-run artifacts, dataset reports, plots, and generated
  aggregate result files as specified.
- Do all work in `/workspace/capstone`.

## 2026-09-09 — Durable memory

The user explicitly requested use of `Fyxod/durable-agent-memory`. This
`project-memory/` directory follows that protocol. It must be updated at
material boundaries, preserve failed/superseded work, and keep unsupported
plans distinct from validated outcomes.

The user authorized creation of private repository
`Tanishgupta28/capstone-gpu` and periodic pushes of relevant code and project
memory at sensible checkpoints. Raw datasets remain local and ignored.

## 2026-09-09 — Approved scientific protocol

The user approved all of the following as binding implementation decisions:

- Use genuine full EdNet-KT1 with original interaction files and question/tag
  metadata. Keep `ednetnew.csv` only as a legacy reduced/debugging experiment.
- Use the full 14.66M-row Junyi file for final runs; use the 1M subset only for
  smoke tests and pipeline debugging.
- For Junyi multi-tag questions, represent the complete sorted tag set as one
  deterministic composite skill ID; never expand an interaction into multiple
  rows.
- Define sessions using a 10-hour inactivity threshold.
- Exclude students with fewer than five sessions, then use chronological
  per-student 60/20/20 session splits with rolling past-only history.
- Break timestamp ties with an authoritative event index when available;
  otherwise preserve original CSV row order.
- Consolidate HiTSKT while preserving Action Encoder, Session Encoder,
  Correct/Padding Encoder, Decoder, ELU+1 linear Performer attention, causal
  masking, and PAD masking. Add causal, padding, shape, and checkpoint tests.
- Prefer the paper authors' official/reference RKT implementation. If none is
  available, make and document a reproduction and every deviation.
- Stop and ask before resolving any new ambiguity that could materially affect
  data, preprocessing, architecture, evaluation, or scientific validity.
