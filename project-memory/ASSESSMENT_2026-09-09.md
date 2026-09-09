# Initial repository and data assessment — 2026-09-09

## Continuation point and runnable state

- GitHub `origin/main` is commit `6c930c5` (2024-09-25). Its two tracked CSVs
  are LFS pointer text locally, so the checked-out training command cannot load
  data until datasets are hydrated.
- The documented HiTSKT command is not runnable as written: `main.py` declares
  `--learning_rate` and `--dropout` as `int`, and argparse rejects values such
  as `5e-5` and `0.1`.
- Upstream HiTSKT uses quadratic scaled-dot-product/softmax attention in
  `../attention_modules.py`. That is not the requested pure Performer variant.
- All inspected Python files parse successfully in the installed interpreter.

## Drive inventories and dataset evidence

All three supplied folders were recursively enumerated. They contain complete
working-directory mirrors (including `.git`, caches, source, logs, and shared
checkpoints), not just canonical dataset packages. Relevant data observed:

| Label | File | Bytes | Rows | Students | Questions | Skills | Sessions |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASSIST2017 | `2017.csv` | 82,778,007 | 942,807 | 1,709 | 3,162 | 102 | 13,506 |
| Junyi full | `Junyi.csv` / `junyi_processed.csv` | 671,494,411 | 14,660,217 | 29,865 | 25,630 | 1,326 | 598,872 |
| Junyi debug | `junyi_1M.csv` | 45,816,531 | 1,000,243 | not fully audited | not fully audited | not fully audited | not fully audited |
| legacy derivative | `ednet_processed.csv` | 7,850,241 | 181,171 | 240 | 12,635 | 5,631 | 6,031 |
| legacy derivative | `ednetnew.csv` | 13,251,192 | 312,815 | 1,000 | 12,247 | 8,544 | 10,178 |

The two full Junyi names are byte-identical (SHA-256
`8a18000806279ae4feae90a94a5af75b7688f2e4ba666c9a0662c75e1a404b6a`).
ASSIST2017 SHA-256 is
`7577287d4ead073fdb9393b586a260043151f27338246bd076ca740e28be8bbb`,
matching the upstream LFS object ID. These supplied processed files contain no
missing cells or exact duplicate rows and are nondecreasing by timestamp within
student. Junyi has 12,933,808 within-student timestamp ties, so timestamp alone
does not define a unique interaction order.

The Drive file called `ednet.csv` is 5,846,760,913 bytes, but its header is:

`row_id,timestamp,user_id,content_id,content_type_id,task_container_id,user_answer,answered_correctly,prior_question_elapsed_time,prior_question_had_explanation`

This is the Riiid Answer Correctness Prediction schema, not the native EdNet
KT1/KT4 schema. No `questions.csv`, tag/skill metadata, or other content
metadata is present in the supplied EdNet folder.

The existing preprocessor maps Riiid `task_container_id` to `skill`. A task
container is not question skill/tag metadata. It also discards
`content_type_id`; consequently lecture rows with `answered_correctly == -1`
are converted to correct answers because Python treats `-1` as truthy. The
Drive ASSIST preprocessor additionally limits this input to `nrows=200000`, and
the other Drive EdNet preprocessor body is commented out. None is a valid full
pipeline.

## Models and evaluation

- DKT, DKVMN, and SAKT exist upstream and their Drive copies are logically
  identical (only line endings differ). They are reusable architecture
  references, not benchmark runners: each hardcodes `dataset/2017.csv`, owns a
  separate split/loader/loop, logs primarily AUC, and does not emit the required
  standardized artifacts. DKVMN also passes probabilities directly to
  `accuracy_score`, which is an execution error for continuous predictions.
- RKT is absent from GitHub and every inspected Drive tree.
- HiTSKT's hierarchy (action encoder, session encoder, correctness/padding
  encoder, decoder, prediction head) is present and reusable.
- Three different Drive attention/model/training variants exist. They use an
  ELU+1 kernelized linear-attention implementation described there as
  Performer. A direct synthetic causality test perturbed only a future value:
  the ASSIST variant changed the earliest output by 3.53 and the EdNet variant
  by 3.10 (failure); the Junyi variant changed it by 0.0 (pass). Thus the
  ASSIST/EdNet versions ignore triangular causality and can leak future target
  correctness through the shifted-correctness encoder.
- PAD/EOS label masking in the newer Drive loops is present, but it is not yet
  covered by regression tests and the different models do not share a single
  target/masking implementation.
- Existing splitting is per-student, chronological by session, nominally
  60/20/20 using integer fifths. For supplied ASSIST it yields 7,849/1,974/1,974
  examples (66.53/16.73/16.73); 321 students have fewer than five sessions and
  contribute no validation/test examples. Full Junyi yields
  350,497/109,255/109,255 (61.60/19.20/19.20).

## Prior run classification

- ASSIST pure-Performer log: training reached 100 epochs but has no final test
  result and used the causality-failing implementation. Invalid as final
  evidence.
- Junyi log dated 2026-09-09: 50 epochs on `junyi_1M.csv`, no final test result.
  It is a debug/sample run, not the full Junyi benchmark.
- Legacy `ednetnew.csv` log dated 2026-09-09: completed 40 epochs and a test,
  but uses the sampled derivative and causality-failing implementation. Invalid
  as the requested full-dataset result.
- No complete, comparable 5-by-3 result matrix exists.

## Reuse and required work

Reuse the hierarchy, compatible model internals, supplied full ASSIST/Junyi
data, chronological intent, and validation-AUC checkpoint concept. Consolidate
the runners around one validated data/split/masking/metrics/artifact contract;
implement streaming/indexed full-data sequence generation; add RKT; add unit
and integration gates for causality, target alignment, PAD/EOS exclusion,
checkpoint round trips, and split leakage; then run the prescribed experiment
order. Existing logs/checkpoints must remain legacy references only.

## User decisions required before implementation

1. Dataset identity: use genuine EdNet KT1/KT4 (requiring the actual EdNet raw
   interaction and question/tag metadata), or continue with the supplied Riiid
   file? Recommendation: use genuine EdNet KT1 and keep Riiid artifacts clearly
   labeled as legacy, because calling Riiid an EdNet result is scientifically
   inaccurate.
2. Junyi version: use the byte-identical full `Junyi.csv`/`junyi_processed.csv`
   (14.66M rows) or the 1M subset? Recommendation: full for final experiments;
   retain 1M only for smoke/debug runs.
3. Skill mapping: for the chosen EdNet-family data, what authoritative
   question-to-skill/tag metadata should be used, and how should multi-tag
   questions be encoded? Recommendation: provide/join the official question
   metadata, preserve the tag set, and choose a documented deterministic
   multi-skill policy; do not use `task_container_id` as skill.
4. Sessions: retain the project's 10-hour inactivity boundary for every
   dataset, or use another definition? Recommendation: retain 10 hours for
   continuity unless the capstone methodology specifies a different threshold.
5. Split eligibility: retain the current per-student 60/20/20 session split
   with short-history students train-only, or filter all datasets to students
   with at least five sessions? Recommendation: apply the same >=5-session
   eligibility rule to all three before a per-student chronological 60/20/20
   split, and report the exclusions.
6. Tied timestamps: should original CSV/row order be the tie-breaker, or can the
   original raw Junyi ordering key be supplied? Recommendation: preserve stable
   input row order and document it unless an authoritative event index exists.
7. Performer source: is the active Junyi causal Performer variant the intended
   primary implementation? Recommendation: consolidate from it, preserve the
   HiTSKT hierarchy and ELU+1 linear attention, and add explicit causal/PAD
   invariance tests. The ASSIST and EdNet variants must not be used unchanged.
8. RKT origin: should the missing baseline be adapted from a specific prior
   implementation? Recommendation: use the paper authors' official reference
   implementation if you have one; otherwise authorize a documented clean
   reproduction against the paper.
