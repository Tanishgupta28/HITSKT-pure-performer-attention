# Current state

Updated: 2026-09-09 UTC

## Established

- `/workspace/capstone` initially contained only `prompt.txt` and a notebook
  checkpoint copy; it was not a Git repository.
- GitHub repository `pokerme7777/HiTSKT` exists with default branch `main`.
- The upstream continuation point is now checked out locally on branch `work`,
  with `origin/main` retained as the source baseline.
- Private checkpoint repository `Tanishgupta28/capstone-gpu` was created at the
  user's direction. Local branch `work` tracks its `main` branch; checkpoint
  commit `88aef9f` contains the prompt, assessment, directives, and memory.
- Upstream contains HiTSKT and baseline files for DKT, DKVMN, and SAKT. It does
  not contain an RKT implementation in the inspected tree.
- Upstream dataset CSV paths are Git LFS pointer files locally, not usable data.
- Available compute observed during inspection: an NVIDIA H100 MIG 3g.40gb
  device with about 40 GiB visible to PyTorch, approximately 2 TiB system RAM,
  and 224 logical CPUs. PyTorch 2.4.0a0 reports CUDA 12.5 and successfully sees
  the GPU.

Evidence: repository paths `../README.md`, `../model_layers.py`,
`../Other_models/`, and the 2026-09-09 Git/host inspection recorded in the
active task transcript.

## Assessment conclusion

The inspection is complete. See
[ASSESSMENT_2026-09-09.md](ASSESSMENT_2026-09-09.md). The user has approved the
dataset identity/selection, tag mapping, session/split protocol, rolling-history
evaluation, Performer consolidation, and RKT sourcing decisions. Implementation
is authorized and is beginning with authoritative EdNet-KT1 acquisition.

## Not yet validated or completed

- The genuine EdNet-KT1 interaction archive and official contents archive were
  downloaded from the official EdNet repository's published Google Drive
  objects. Both passed full ZIP integrity validation. The KT1 archive is
  1,201,163,816 bytes with SHA-256
  `0d13933f90201c5101c7fe8659e44474fa049e3fb93181a8ba6fb3e63267b535`;
  it contains 784,309 student CSV members totaling 3,072,366,053 uncompressed
  bytes. The contents archive is 173,976 bytes with SHA-256
  `aa910a0436d9dbac0ba232f55e27b37ad8d39da285fcf15f1c1eb5d064be98e7`.
- Official EdNet `questions.csv` contains 13,169 questions. All have tags;
  6,049 questions have multiple semicolon-delimited tags (maximum seven).
- The user explicitly extended the complete sorted composite-tag rule to EdNet
  and designated `tags = -1` as missing metadata mapped to one `<UNTAGGED>`
  composite skill ID. The provisional vocabulary is compatible with the
  existing embedding approach.
- A deterministic streaming preprocessor, README protocol, and focused tests
  have been implemented. All 11 current synthetic preprocessing test cases pass.
- The first real 1,000-student smoke invocation stopped before processing any
  interactions, as designed, on a previously unknown duplicate tag within a
  question. The user approved set deduplication, and the exact example now has a
  regression test.
- A full metadata audit found 197 questions with duplicated tag IDs: duplicate
  `176` in 85 questions, `178` in 75, and `177` in 37. There are 83 distinct raw
  affected tag strings and no other malformed format besides approved `-1`.
  The approved implementation deduplicates these repeated IDs before sorting.
- The second real smoke invocation passed tag mapping and stopped at interaction
  row 674 of `KT1/u4.csv` because `user_answer` is empty. A bounded audit of the
  first 1,000 numeric student files (1,427,687 interactions) found 2,898 empty
  answers across 349 students and no empty values in the other four columns.
  No rows were dropped and no correctness label was inferred.
- The user directed exclusion of empty answers from supervision with separate
  audit retention. The implementation now filters before rebuilding sessions,
  splits, and attempt counters. All 11 synthetic cases pass, including audit
  row preservation and no synthetic label.
- The third real first-1,000-student smoke run completed and was independently
  validated: 2,898/1,427,687 rows (0.2029856684%) across 349 students were
  excluded to audit; 1,409,948 supervised interactions from 704 eligible
  students formed 32,795 sessions. Session counts changed for 19 students (net
  -18); answered-event grouping changed for one. Parquet row counts, labels,
  ordering, session continuity, per-session positions, split chronology, and
  completion markers all passed.
- No model/dataset experiment ran; there are no valid benchmark results.
- No full-dataset preprocessing, tensor/model/checkpoint/metrics smoke
  test, or scientific experiment ran.

## Known implementation risks requiring evidence

- The supplied Drive “EdNet” data is Riiid and cannot be used; genuine full
  EdNet-KT1 must be acquired and provenance-verified.
- Existing baseline scripts are dataset-hardcoded and do not provide the full
  common evaluation contract.
- RKT is missing locally; an authors' reference implementation must be located
  or its absence documented before a reproduction is written.

Local-only inspection material is under `../.inspection/`; it is untracked and
contains downloaded dataset copies and source variants. It is evidence staging,
not a deliverable or accepted data layout.
