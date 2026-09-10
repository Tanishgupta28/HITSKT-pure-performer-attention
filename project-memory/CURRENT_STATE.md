# Current state

Updated: 2026-09-10 UTC

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
- The full EdNet pass completed across 784,309 files and 95,293,926 source
  interactions. It excluded/audited 27,646 unanswered rows across 4,477
  students (0.0290112929%), leaving 95,266,280 supervised interactions.
- Post-filter session rebuilding changed session counts for 148 students (net
  -150) and answered-event grouping for six. The five-session filter retained
  116,548 students, 81,940,867 interactions, and 2,577,988 sessions, split into
  53,990,022 train, 14,520,975 validation, and 13,429,870 test interactions.
- Independent streaming validation passed all 82 event shards and 27,646 audit
  rows. Aggregate reports, distributions, and deterministic question/tag/skill
  mappings are staged under `../reports/datasets/ednet_kt1/`; interaction-level
  Parquet remains ignored and local.
- Accepted ASSIST2017 and full Junyi source files are staged under `data/raw/`
  with verified hashes. A vectorized shared preprocessor and three synthetic
  fail-closed tests have been added.
- The real ASSIST2017 run stopped before output completion because 697/3,162
  questions have multiple observed row-level skills (682 have two, 15 have
  three). They cover 440,761/942,807 interactions (46.749865%) and 3,874 unique
  question/skill pairs. The user directed preservation of supplied row-level
  skills. The completed build retains 885,335 interactions from 1,388 eligible
  students in 12,402 sessions; independent validation passes.
- Full Junyi composite mapping found 1,326 genuine original tags and 1,521
  complete-set composite skills. There are 776 multi-tag questions covering
  961,084 interactions (6.555728%). The validated build retains all 14,660,217
  interactions, 29,865 students, and 600,154 rebuilt sessions across 15 shards.
- Aggregate mappings/reports for ASSIST2017 and Junyi are staged under
  `../reports/datasets/`; interaction Parquet remains local and ignored.
- A full session-length audit found that the legacy fixed action capacity would
  discard 303,971 ASSIST interactions (34.334009%), 4,310,593 Junyi
  interactions (29.403337%), and 35,198,097 EdNet interactions (42.955485%).
  Maximum session lengths are 938, 3,924, and 13,080.
- The user selected variable-length length-bucketed batches, per-batch dynamic
  padding, and token-budgeted batch sizes, with no action truncation or session
  chunking. Lossless memory-mapped session stores now reconcile all 97,486,419
  retained interactions and all 3,190,544 sessions across the three datasets.
- The consolidated HiTSKT path explicitly retains Action Encoder, Session
  Encoder, Correct/Padding Encoder, Decoder, and prediction stages. All
  attention uses causal prefix-sum ELU+1 linear attention with PAD masks; there
  is no softmax or quadratic sequence attention in the new implementation.
- Full batch planning uses the legacy rolling context of at most 15 earlier
  complete sessions, a normal 32,768 padded-token budget, and batch-size cap
  64. Every post-first session remains one complete target. The worst EdNet
  example is indivisible and uses a 196,280-token singleton batch.
- Real forward/loss/backward benchmarks passed on the H100 MIG device. Peak
  allocated/reserved CUDA memory was 7.066/8.686 GiB for ASSIST, 4.593/5.621
  GiB for Junyi, and 24.989/30.861 GiB for the worst EdNet singleton.
- All 28 automated tests pass, covering dynamic shapes/padding, causal and PAD
  masks, target shifting, EOS metric exclusion, long singleton batching,
  past-only rolling history, all-stage backward gradients, and strict
  checkpoint loading. Exact evidence is versioned in
  `docs/data/variable_length_batching.md` and
  `reports/batching/dynamic_batching_benchmark.json`.
- No model/dataset experiment ran; there are no valid benchmark results.
- The HiTSKT tensor/forward/backward/checkpoint/masking smoke gate has passed;
  baseline smoke gates and all scientific training experiments remain pending.

## Known implementation risks requiring evidence

- The supplied Drive “EdNet” data is Riiid and cannot be used; genuine full
  EdNet-KT1 must be acquired and provenance-verified.
- Existing baseline scripts are dataset-hardcoded and do not provide the full
  common evaluation contract.
- RKT is missing locally; an authors' reference implementation must be located
  or its absence documented before a reproduction is written.

## Newly identified RKT protocol ambiguity

- The paper-author repository is available at `shalini1194/RKT`; author Shalini
  Pandey's current reference was inspected at commit `cac60f512f`. The paper is
  arXiv `2008.12736`.
- Full RKT defines exercise relations as the thresholded sum of a train-data
  performance Phi coefficient and cosine similarity of textual exercise
  embeddings. The paper also reports performance-only Phi and same-KC relation
  ablations. The accepted capstone datasets do not expose authoritative,
  comparable full exercise text for all three datasets.
- The authors specify maximum interaction length 50 and partition longer
  sequences. Their released model materializes quadratic attention, relation,
  and time matrices. Requiring complete 13,080-action EdNet sessions in a
  single RKT pass would materially depart from the author protocol and carries
  a substantial GPU-memory risk.
- No RKT implementation has been changed. User direction is required on the
  no-text relation variant and whether author-standard bounded rolling context
  is permitted for RKT while retaining every supervised target exactly once.
- The user approved performance-only Phi relations from training information
  and rolling 49-interaction history. A second fail-closed audit found further
  material paper/code incompatibilities before implementation:
  - paper time weight is `exp(-delta_t / S_u)` with trainable student strength;
    released code uses `exp(-abs(raw_delta_t))`, has no `S_u`, and does not
    normalize seconds versus EdNet milliseconds, causing practical underflow;
  - paper settings state width 64, dropout 0.1, batch 128, and positional
    embeddings; executable defaults are width 200, five heads, dropout 0.2,
    batch 200, and positional encoding disabled. Width 64 is not divisible by
    the released five-head default;
  - paper performance-only ablation says to use Phi (Equation 2), while the
    full relation applies a 0.8 threshold only after adding text similarity;
    released code consumes an external precomputed relation file and does not
    implement Phi construction, so it does not resolve thresholding for the
    no-text variant.
- RKT remains unchanged pending a user choice between paper-faithful repairs
  and literal released-code behavior for these conflicts.

Local-only inspection material is under `../.inspection/`; it is untracked and
contains downloaded dataset copies and source variants. It is evidence staging,
not a deliverable or accepted data layout.
