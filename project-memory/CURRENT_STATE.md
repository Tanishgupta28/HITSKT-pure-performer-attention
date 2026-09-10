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
- All HiTSKT batching/model tests pass, covering dynamic shapes/padding, causal and PAD
  masks, target shifting, EOS metric exclusion, long singleton batching,
  past-only rolling history, all-stage backward gradients, and strict
  checkpoint loading. Exact evidence is versioned in
  `docs/data/variable_length_batching.md` and
  `reports/batching/dynamic_batching_benchmark.json`.
- The centralized project seed is 42. It controls Python, NumPy, PyTorch
  CPU/CUDA, batch shuffling, data sampling, and deterministic five-fold RKT
  assignment. Full fold sizes are ASSIST 278/278/278/277/277, Junyi five equal
  folds of 5,973, and EdNet 23,310/23,310/23,310/23,309/23,309.
- The approved RKT paper-faithful performance-only variant is implemented from
  the authors' repository reference at commit `cac60f512f`. It uses width 64,
  dropout 0.1, one head, learned positions, rolling 49-event histories,
  directed raw Phi, positive trainable `S_u`, and learned lambda initialized
  at 0.5. Exact deviations are recorded in `../docs/models/rkt_variant.md`.
- RKT timestamps are verified per source and normalized to hours. Full-store
  training-only `S_u` initialization reports zero fallback students in all
  three datasets; global fallback values are recorded in the RKT report.
- Sparse Phi caches encode deterministic five-fold student cross-fitting for
  training and all-training-only lookup for validation/test. Tests prove fold,
  own-label, future-interaction, and validation/test exclusion.
- Full-population Phi preparation is complete for all three datasets. ASSIST
  processed 647,283 training targets and 13,530,869 contributions in 14.17 s;
  Junyi processed 9,647,042 targets and 410,926,464 contributions in 734.99 s.
  EdNet processed 53,990,022 targets and 2,430,771,986 contributions in
  1,984.27 s. Exact unique-pair, scratch, RSS, cache-size, sample, and historical
  estimate evidence is versioned in `../reports/rkt/phi_scaling_benchmark.*`.
- The bounded 20-student ASSIST RKT smoke passed a forward/backward update,
  `[128,49]` shapes, unique-target accounting, checkpoint round-trip, frozen
  evaluation relation parameters, and all required metrics. Its results are
  diagnostic only.
- No full model/dataset experiment has run; there are no valid final benchmark
  results.
- The user fixed baseline contexts/hyperparameters across datasets. Clean DKT,
  DKVMN, and SAKT adapters now use rolling target-once histories of 199, 199,
  and 99 prior interactions respectively, with dynamic PAD masking and the
  repository's unchanged model settings.
- Nine real-data/schema smoke gates pass across all model/dataset pairs. Full
  train/validation/test target counts reconcile to every retained interaction;
  each required history cap, forward/backward, all metrics, and strict
  checkpoint round-trip passed.
- Real H100 compute-only timing makes large rolling runs expensive, especially
  DKVMN: EdNet lower bounds are 8.169 h/epoch DKT, 81.614 h/epoch DKVMN, and
  4.953 h/epoch SAKT. Exact results and exclusions are in
  `../reports/batching/baseline_rolling_throughput.json`.
- The user fixed a common early-stopping protocol for all five models:
  validation ROC-AUC, patience 5, `min_delta=0`, strict improvement, and best
  checkpoint reload before test. `ktbench/training.py` enforces it and exact
  stop/reset behavior is tested. Existing smoke configs now record it.
- A production DKT/DKVMN/SAKT runner now writes per-epoch metrics/logs,
  best/last checkpoints, updated config state, dataset statistics, and final
  results while stopping within the preserved epoch ceiling. It has not been
  launched on a full dataset.
- The baseline production runner passed an end-to-end bounded-fixture test,
  including artifact writing and best-checkpoint reload.
- Matching production RKT and HiTSKT runners now use the common validation-AUC
  controller, save best/last checkpoints and complete configs/logs/metrics,
  reload the best checkpoint before test, and pass bounded end-to-end tests.
- RKT uses maximum gradient norm 10. The paper is silent on clipping; this is
  the authors' released-trainer default and was retained with explicit user
  approval. The setting and its provenance are persisted in RKT configs.
- A CUDA-only implicit indexed-broadcast failure for the learned HiTSKT
  session-EOS vector was fixed by explicitly expanding the same vector across
  batch rows. EOS placement and the research architecture are unchanged.
- All 47 automated tests pass. PyTorch warns that the CUDA cumulative-sum
  kernel used by Performer attention has no deterministic implementation in
  this installed build; seeding and deterministic warn-only mode remain on so
  this limitation is visible.
- The first full scientific experiment, RKT on ASSIST2017, completed in
  1,474.90 s. Strict early stopping selected epoch 14 at validation AUC
  0.7712554524 and stopped at epoch 19 after exactly five misses. Reloading the
  best checkpoint produced test AUC 0.7486264899, accuracy 0.6965399280,
  precision 0.6399917287, recall 0.4975655689, F1 0.5598625216, MSE
  0.1963360075, and loss 0.5771361224 across all 112,252 test targets.
- An independent full test pass from `best_model.pt` reproduced every stored
  test metric exactly.
- Full HiTSKT on ASSIST2017 completed in 968.04 s. Strict early stopping
  selected epoch 40 at validation AUC 0.7197572561 and stopped at epoch 45.
  Best-checkpoint test metrics over all 112,252 held-out actions are: AUC
  0.7096240159, accuracy 0.6744200549, precision 0.5866082262, recall
  0.5440494235, F1 0.5645278522, MSE 0.2098452017, loss 0.6087257417.
  Independent best-checkpoint evaluation reproduced every value exactly.
- HiTSKT used all 566,107 actions in post-first training sessions; the 81,176
  actions in each student's first session are context-only because the
  hierarchical model requires an earlier session. Validation/test contain no
  first student session and retain all 125,800/112,252 targets. No session was
  truncated or chunked.
- The aggregate `reports/final_results.{csv,json}` is generated from completed
  experiment artifacts and currently contains 4/15 rows; no smoke or
  throughput result enters it.
- Full DKT on ASSIST2017 completed in 6,233.07 s with the unchanged batch-20,
  context-200 configuration. Strict early stopping selected epoch 12 at
  validation AUC 0.7213696056 and stopped at epoch 17. Best-checkpoint test
  metrics over all 112,252 targets are: AUC 0.7069997420, accuracy
  0.6750258347, precision 0.6113932435, recall 0.4451564007, F1 0.5151970231,
  MSE 0.2083804281, and loss 0.6057200542. Independent full test evaluation
  reproduced every value exactly.
- Full SAKT on ASSIST2017 completed in 10,551.13 s with the unchanged batch-10,
  context-100 configuration. Strict early stopping selected epoch 30 at
  validation AUC 0.7577061439 and stopped at epoch 35. Best-checkpoint test
  metrics over all 112,252 targets are: AUC 0.7384655654, accuracy
  0.6902950504, precision 0.6240144686, recall 0.5071425291, F1 0.5595408532,
  MSE 0.1998719304, and loss 0.5853560606. Independent full test evaluation
  reproduced every value exactly.
- DKVMN's identical affine memory writes now use a chronological balanced
  composition rather than 199 Python-driven GPU launches. Reference output and
  gradient equivalence is tested. Actual full-history ASSIST step time improved
  24.0x from 0.174721 s to 0.007266 s; projected compute-only epoch time is now
  0.0408 h, with no model or protocol change.
- Baseline production runs now have a validated explicit resume path that
  restores model, Adam, and early-stopping state and appends durable artifacts.
  DKVMN/ASSIST was externally interrupted after epoch 19 (best epoch 18,
  validation AUC 0.6962520470; one patience miss) and is ready to resume at
  epoch 20 without losing scientific state.

## Known implementation risks requiring evidence

- The supplied Drive “EdNet” data is Riiid and cannot be used; genuine full
  EdNet-KT1 must be acquired and provenance-verified.
- Loading the complete production Phi repositories and measuring RKT epoch
  throughput is validated. Grouping identical lookups by applicable fold cache
  is exactly equal to scalar lookup in tests and improves projected epochs from
  96.3 s/98.0 min/9.89 h to 79.1 s/23.7 min/2.34 h for ASSIST/Junyi/EdNet.
  These are planning estimates, not scientific results; raw evidence is under
  `../reports/rkt/`.

## RKT protocol resolution

- The paper-author repository is available at `shalini1194/RKT`; author Shalini
  Pandey's current reference was inspected at commit `cac60f512f`. The paper is
  arXiv `2008.12736`.
- Full RKT defines exercise relations as the thresholded sum of a train-data
  performance Phi coefficient and cosine similarity of textual exercise
  embeddings. The paper also reports performance-only Phi and same-KC relation
  ablations. The accepted capstone datasets do not expose authoritative,
  comparable full exercise text for all three datasets.
- The authors specify maximum interaction length 50. The user approved a
  rolling 49-prior-interaction construction that retains every supervised
  target exactly once without forcing full sessions into one RKT matrix.
- The user approved performance-only Phi relations from training information
  and paper-faithful handling of these paper/code incompatibilities:
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
- The complete approved formulation, five-fold leakage protocol, timestamp
  audit, initialization, source citation, and smoke evidence are implemented
  and documented in `../docs/models/rkt_variant.md`. No current material RKT
  ambiguity remains.

Local-only inspection material is under `../.inspection/`; it is untracked and
contains downloaded dataset copies and source variants. It is evidence staging,
not a deliverable or accepted data layout.
