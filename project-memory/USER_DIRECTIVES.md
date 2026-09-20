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
- Apply the same rule to EdNet: collect every question tag, sort tag IDs
  deterministically, map the complete set to one persisted composite skill ID,
  and assign that one ID to each interaction without row expansion or
  primary-tag selection. Report original/composite vocabulary sizes, multi-tag
  questions, the composite distribution, and multi-tag interaction count.
- If the EdNet composite vocabulary is unexpectedly large or incompatible with
  HiTSKT's embedding/configuration, stop before changing the modeling approach.
- Treat EdNet `tags = -1` as missing metadata, not as a genuine original tag.
  Retain every such interaction exactly once using the one deterministic
  `<UNTAGGED>` composite skill ID, distinguish missing/genuine counts in reports,
  and document the decision in the dataset report and README.
- Do not infer a policy for any other missing or malformed EdNet tag value; stop
  and ask before proceeding if one is encountered.
- Treat positive EdNet tags as a mathematical set. Remove repeated IDs within a
  question before numeric sorting and composite-ID generation. Record that 197
  questions were affected; never expand or duplicate interactions because of
  tags.
- Exclude empty/missing `user_answer` rows from the supervised KT sequence. Do
  not label them incorrect or add a third response state. Preserve their source
  fields separately for audit, rebuild sessions/sequences after filtering, and
  report total/rate/affected-student and session/length effects. Stop on any
  other ambiguous response encoding.
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
- For ASSIST2017 specifically, preserve the supplied row-level skill on each
  interaction, even when a question appears with different skills elsewhere.
  Do not infer a complete mapping, merge skills, or expand rows. Record 3,162
  questions, 697 multi-row-skill questions, and 440,761 affected interactions
  (46.75%). Revisit only if authoritative complete metadata is found.

## 2026-09-10 — Variable-length HiTSKT and RKT protocol

- Preserve complete 10-hour sessions. Use length-bucketed, dynamically padded,
  token-budgeted batches; never truncate or chunk long sessions. Rare long
  examples use smaller or singleton batches. PAD must be masked in attention,
  loss, and metrics, and EOS/BOS target alignment must remain valid.
- Implement the authors' RKT repository as the reference, but use the
  performance-only Phi variant because comparable authoritative question text
  is unavailable across all three datasets. Never fabricate text features.
- Label it “RKT paper-faithful performance-only Phi-relation variant with
  5-fold student-level cross-fitting.” Use width 64, dropout 0.1, learned
  positions through context 50, one head, and rolling 49 prior interactions;
  retain every target once.
- Use directed raw Phi from training histories only. Assign deterministic
  seed-42 student folds. Training targets use Phi from the other four complete
  student folds; validation/test use all training folds. No target label,
  same-student future, or validation/test interaction may enter Phi. Undefined
  relations map to zero and no 0.8 threshold/text relation is used.
- Normalize verified source timestamps to hours. Use
  `S_u=softplus(rho_u)+epsilon`, initialized from the student's median positive
  training gap or a global training-only fallback. Use
  `lambda=sigmoid(eta)`, initialized at 0.5. Learn both during training only and
  freeze them for validation/test.
- Stop on any further material RKT ambiguity instead of copying an arbitrary
  released-code default.

## 2026-09-10 — Project-wide seed

- Use 42 as the one centralized seed for Python, NumPy, PyTorch CPU/CUDA,
  student folds, data sampling, randomized split generation, length-bucket
  shuffling, and every primary DKT, DKVMN, SAKT, RKT, and HiTSKT run across all
  three datasets. Do not retain zero as the batching default.
- Persist seed 42 in every experiment config and training log and in every
  final-results row/equivalent benchmark summary. The primary benchmark is
  single-seed unless a separately documented multi-seed run is approved.

## 2026-09-10 — Baseline contexts and hyperparameters

- Preserve repository configurations across every dataset rather than tuning
  per dataset. DKT and DKVMN use context 200 as 199 strictly prior interactions
  plus the target. SAKT uses context 100 as 99 prior plus target.
- Replace non-overlapping chunks with rolling target-once histories. Preserve
  chronological ordering, exclude future information, and keep PAD out of
  model state, loss, and metrics.
- Use validation ROC-AUC for early stopping and best-checkpoint selection.
  Persist all model/dataset configuration fields, seed 42, patience, best
  epoch, and parameter counts. State explicitly that model contexts and
  hyperparameters differ while evaluation methodology is standardized.
- Smoke-test target counts, caps, chronology, future exclusion, loss/metric
  masking, and checkpoint loading before large runs. Stop before changing any
  incompatible context or hyperparameter on Junyi/EdNet.

## 2026-09-10 — Common early stopping

- For DKT, DKVMN, SAKT, RKT, and HiTSKT, select on validation ROC-AUC with
  patience 5 and `min_delta=0`. Save only on strict improvement, stop
  immediately after five consecutive non-improving epochs, and reload that
  best checkpoint before final test evaluation.
- Preserve every existing model-specific epoch ceiling; early stopping may
  shorten but never extend it. Do not add model-specific patience without
  stopping for approval.
- Every experiment must persist patience, min delta, best epoch, best
  validation AUC, and total completed epochs. Avoid any unnecessary epoch after
  the stopping condition.

## 2026-09-16 — Monitoring cadence

- Check active training every 30 minutes, superseding the earlier 20-minute
  cadence. Continue automatically with verification and subsequent runs.
- Keep committing under the original Tanish identity and defer pushes until
  the user logs in to the original GitHub account.

## 2026-09-19 — GitHub authentication restored

- User confirmed `gh authorized`. Verified the active account is Tanishgupta28
  and the existing Tanishgupta28/capstone-gpu repository is private and accessible.
- Resume checkpoint pushes to the existing remote, retaining Tanish commit
  identity. This supersedes the temporary push deferral above.

## 2026-09-20 — Hourly monitoring

- Check active training once per hour, superseding the earlier 30-minute
  cadence. Continue automatically with checkpoint verification, commits/pushes,
  independent final evaluation, and subsequent approved runs.
- This changes monitoring frequency only, not the running experiment or any
  scientific configuration.
