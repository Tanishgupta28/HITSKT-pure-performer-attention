# HiTSKT: A Hierarchical Transformer Model for Session-awared Knowledge Tracing
## Keywords
-  Hierarchical Transformer
-  Session-awared Knowledge Tracing
-  Knowledge Tracing
  
🔥🔥🔥This is the code for the paper HiTSKT: A Hierarchical Transformer Model for Session-awared Knowledge Tracing, accepted by Knowledge-Based Systems [[paper](https://www.sciencedirect.com/science/article/pii/S0950705123010481)].

## Setup

The requiring environments is as bellow:

- Python 3.6+
- PyTorch 1.9.0
- Scikit-learn 0.24.2
- Numpy 1.19.5
- Pandas 1.1.5
- Dask 2021.3.0

## Data and Data Preprocessing

### Reproducible benchmark pipeline

The legacy instructions below describe the original research repository and
are retained for provenance. They do **not** define the capstone's final EdNet
benchmark: the prior Kaggle/Riiid file and `ednetnew.csv` are reduced or
different datasets and are not accepted as full EdNet results.

The final EdNet source is genuine official EdNet-KT1 plus its official
`contents/questions.csv`. Acquisition hashes and schema evidence are recorded
in [the EdNet provenance report](docs/data/ednet_kt1_provenance.md). Raw and
processed data are ignored by Git.

Run a bounded smoke test first:

```bash
python -m ktbench.data.ednet \
  --kt1-zip data/raw/ednet/archives/EdNet-KT1.zip \
  --contents-zip data/raw/ednet/archives/EdNet-Contents.zip \
  --output-dir data/processed/ednet_kt1/smoke \
  --max-students 1000
```

Run the full deterministic preprocessing by omitting `--max-students`:

```bash
python -m ktbench.data.ednet \
  --kt1-zip data/raw/ednet/archives/EdNet-KT1.zip \
  --contents-zip data/raw/ednet/archives/EdNet-Contents.zip \
  --output-dir data/processed/ednet_kt1/full
```

EdNet tags are treated as mathematical sets: repeated IDs within a question are
deduplicated, then the remaining IDs are numerically sorted and mapped to one
stable composite skill ID. Interactions are never expanded and no primary tag
is selected. The literal metadata value `-1` is not an original tag. All such
questions use the single reserved composite ID `1`, named `<UNTAGGED>`; `0` is
reserved for padding. The generated `mappings/` tables make question, original
tag, composite skill, and student IDs reproducible. Any other missing or
malformed tag representation stops preprocessing for explicit review.

Rows with an empty `user_answer` are excluded from the supervised sequence and
are never relabeled as incorrect or converted to a third model state. Their
original fields are preserved under `audit/unanswered_events/`. Sessions,
sequence lengths, split assignments, and question/skill attempt counters are
rebuilt after filtering. The generated report quantifies the full and bounded
sample exclusion rates and any resulting session changes.

The completed full-dataset mappings, aggregate distribution, validation
manifest, and dataset report are versioned under
[`reports/datasets/ednet_kt1/`](reports/datasets/ednet_kt1/). Event and audit
Parquet shards remain local because they contain interaction-level data.

Sessions use the approved 10-hour inactivity boundary. Students with fewer
than five sessions are excluded, and each retained student's sessions are split
chronologically 60/20/20 using earlier remainder sessions for training. The
event table retains chronological rows and split membership so validation and
test examples can use rolling, strictly past-only histories.

Run all preprocessing, batching, masking, model, and checkpoint tests with:

```bash
python -m pytest -q
```

### Reproducibility seed

The single project-wide seed is **42**, defined once as `PROJECT_SEED` in
[`ktbench/config.py`](ktbench/config.py). `seed_everything()` applies it to
Python `random`, NumPy, PyTorch CPU, and every available PyTorch CUDA generator;
it also enables deterministic PyTorch/cuDNN behavior. Data sampling,
student-fold assignment, length-bucket shuffling, split generation whenever it
uses randomness, and all primary DKT, DKVMN, SAKT, RKT, and HiTSKT experiments
must use this value. The former batching default of zero and legacy inactive
seed 123 snippets are no longer part of the accepted benchmark path.

Every experiment `config.json` and training-log event records `"seed": 42`.
The benchmark-wide summary contract is persisted in
[`reports/benchmark_config.json`](reports/benchmark_config.json); each future
row of the actual final-results table must also carry `seed=42`. No multiple
seed aggregate is part of the primary benchmark unless separately approved and
documented.

### Common early stopping

All five models use the same checkpoint-selection rule. After every completed
training epoch, validation ROC-AUC is compared with the best prior value. A
checkpoint is saved only when AUC strictly improves (`min_delta=0`). Five
consecutive epochs without improvement stop training immediately; the counter
resets after an improvement. The best validation-AUC checkpoint is reloaded
before the one final test evaluation.

Patience is 5 for DKT, DKVMN, SAKT, RKT, and HiTSKT. Early stopping operates
inside, and never extends, the established epoch ceilings: DKT 200, DKVMN 100,
SAKT 300, RKT 300, and HiTSKT 100/50/40 for ASSIST2017/Junyi/EdNet. Every
experiment config/result records patience, `min_delta`, best epoch, best
validation AUC, and total epochs completed. The reusable strict controller is
implemented in [`ktbench/training.py`](ktbench/training.py).

Machine-readable benchmark rows are generated only from completed experiment
directories by `scripts/aggregate_results.py`. The current
[`reports/final_results.csv`](reports/final_results.csv) and JSON companion
contain seven of 15 expected rows: all five ASSIST2017 results plus full RKT
and HiTSKT on Junyi.
Smoke and throughput diagnostics are excluded automatically.

Long baseline runs can be continued with `scripts/train_baseline.py --resume`.
Resume validates the stored model/dataset/seed/batching contract, restores the
last model and optimizer checkpoint plus exact validation-AUC patience state,
continues at the next epoch's deterministic seed-42 bucket order, and appends
to existing metrics/logs. Completed runs cannot be resumed or overwritten.

RKT provides the same explicit `--resume` contract and additionally persists
Python, NumPy, PyTorch CPU, and every CUDA RNG state. This preserves the exact
dropout trajectory across interruption; resume is refused if either RNG or
early-stopping state is absent. An automated test proves an interrupted/resumed
two-epoch RKT run is tensor-for-tensor and metric-for-metric identical to its
uninterrupted counterpart.

The accepted ASSIST2017 and full Junyi sources use the same rebuilt session and
split contract:

```bash
python -m ktbench.data.flat assist2017 \
  data/raw/assist2017/2017.csv data/processed/assist2017/full
python -m ktbench.data.flat junyi \
  data/raw/junyi/Junyi.csv data/processed/junyi/full
```

ASSIST2017 preserves the supplied row-level `skill` on each interaction because
no authoritative complete question-tag mapping is available. Junyi constructs
one deterministic composite skill from the complete deduplicated, numerically
sorted set of tags observed for each question in the accepted full source. In
both cases one source interaction remains one row. Versioned mappings and full
aggregate reports are under [`reports/datasets/assist2017/`](reports/datasets/assist2017/)
and [`reports/datasets/junyi/`](reports/datasets/junyi/).

### RKT reproduction

The implemented baseline is explicitly labeled **RKT paper-faithful
performance-only Phi-relation variant with 5-fold student-level
cross-fitting.** It follows the paper and the authors' reference repository at
<https://github.com/shalini1194/RKT> while omitting the exercise-text relation,
because comparable authoritative question text is unavailable across all
three accepted datasets.

RKT uses width 64, dropout 0.1, one attention head, learned positions, and the
latest 49 strictly prior interactions for each target. Every interaction is a
target exactly once. The directed raw Phi relation is computed from training
histories only: each training student's complete deterministic fold is
excluded from that target's Phi cache, while validation/test use an all-train
cache and never contribute to it. `S_u=softplus(rho_u)+epsilon` and
`lambda=sigmoid(eta)` are learned only during training, then explicitly frozen.
All timestamp deltas are verified and converted to hours. Training clips the
gradient norm at 10, the authors' released-trainer setting; the paper does not
specify this detail, so that provenance is explicit in every RKT configuration.

The exact formulas, timestamp evidence, seed-42 fold counts, paper-versus-code
deviations, cache semantics, and successful ASSIST2017 smoke gate are recorded
in [the RKT methodology and provenance report](docs/models/rkt_variant.md).
Smoke outputs live under
[`experiments/rkt/assist2017/smoke/`](experiments/rkt/assist2017/smoke/); they
are diagnostic and must not be reported as final benchmark results. The
production entry point is `scripts/train_rkt.py`; its artifact and
best-checkpoint-reload contract has passed a bounded end-to-end test. The full
ASSIST2017 RKT result is included in the generated master result table; smoke
artifacts remain excluded.

### DKT, DKVMN, and SAKT baselines

The clean baseline path preserves the continuation repository's established
model settings across all datasets: DKT and DKVMN use 199 prior interactions
plus the current target (context 200), while SAKT uses 99 prior interactions
plus the target (context 100). DKT remains a width-64 one-layer LSTM, DKVMN
retains its 20-slot dynamic key/value memory, and SAKT retains width 200, one
layer, five heads, clipped relative positions, and dropout 0.2.

DKVMN evaluates the unchanged erase/add memory recurrence using a
chronological balanced composition of its affine writes. This is
mathematically the same recurrence as the retained sequential reference but
avoids 199 separate Python-driven GPU steps. Automated tests compare outputs
and every parameter gradient across variable-length/PAD histories; this
changes only the floating-point execution order, not architecture, parameters,
context, targets, or training hyperparameters.

Legacy non-overlapping chunks are excluded. The shared loader creates one
rolling example per valid target, preserves chronological past-only history,
dynamically pads each batch, and masks PAD from model state, loss, and metrics.
The models intentionally do not have identical contexts or hyperparameters;
only target construction, splits, metrics, seed 42, validation-AUC checkpoint
selection, and leakage controls are standardized. Exact provenance and all
nine real-data smoke gates are in
[the baseline rolling protocol](docs/models/baseline_rolling_protocol.md).

Run the tested gates with:

```bash
PYTHONPATH=. python scripts/smoke_baselines.py assist2017 \
  data/processed/assist2017/full/session_store experiments
PYTHONPATH=. python scripts/smoke_baselines.py junyi \
  data/processed/junyi/full/session_store experiments
PYTHONPATH=. python scripts/smoke_baselines.py ednet_kt1 \
  data/processed/ednet_kt1/full/session_store experiments
```

### Variable-length HiTSKT batching

The accepted HiTSKT path no longer calls the legacy fixed-action array loader.
Every 10-hour session is stored and encoded at its complete observed length;
there is no action truncation and no artificial session chunking. Each
post-first session is used once as a target session. Its input is a rolling
window of at most 15 strictly earlier sessions, retaining the original
`session_size=16` context convention while making every action inside those
sessions available. Validation histories may contain only earlier training
sessions, and test histories may contain only earlier training/validation
sessions.

Build the lossless memory-mapped session indices with:

```bash
PYTHONPATH=. python scripts/build_session_store.py \
  data/processed/assist2017/full data/processed/assist2017/full/session_store
PYTHONPATH=. python scripts/build_session_store.py \
  data/processed/junyi/full data/processed/junyi/full/session_store
PYTHONPATH=. python scripts/build_session_store.py \
  data/processed/ednet_kt1/full data/processed/ednet_kt1/full/session_store
```

`LengthBucketTokenBatchSampler` stably orders examples by their largest action
sequence, shuffles only within bounded buckets for training, and greedily packs
up to 64 examples under a 32,768 dynamically padded-token budget. Historical
sessions are flattened for the Action Encoder and padded only to the longest
history in that batch; target sessions and the Session Encoder hierarchy are
padded independently. A session that is itself larger than the budget remains
whole in a singleton batch. PAD positions are excluded by explicit attention
and metric masks; EOS participates in sequence encoding but never in loss or
metrics; target correctness is shifted with BOS so the target response is not
visible to its own prediction. The production entry point is
`scripts/train_hitskt.py`; it uses the common validation-AUC early-stopping
controller and the dataset-specific HiTSKT epoch ceilings above. Its artifact
and best-checkpoint-reload path has passed a bounded end-to-end test. Full
scientific ASSIST2017 and Junyi HiTSKT results are included in the generated
benchmark tables; EdNet remains pending.

The consolidated implementation remains
`ActionEncoder -> SessionEncoder -> CorrectPaddingEncoder -> Decoder ->
prediction`. Every attention layer is causal ELU+1 prefix-sum linear attention;
it does not materialize a quadratic softmax attention matrix. Runtime
sinusoidal positions remove the old fixed positional cap. Exact full-data plan
counts and real H100 forward/backward memory measurements are documented in
[the variable-length batching report](docs/data/variable_length_batching.md).
Reproduce a plan and its worst-batch CUDA benchmark with, for example:

```bash
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/benchmark_dynamic_batching.py ednet_kt1 \
  data/processed/ednet_kt1/full/session_store \
  --token-budget 32768 --max-batch-size 64
```

### Legacy repository instructions

We list the command to run the HiTSKT on different datasets. Listed hyperparameters are the optimal parameters for the respective datasets. The preprocessed data of ASSISTment 2017 and Junyi datasets are provided in the ``dataset`` directory. Due to the file size limitation of GitHub, we are not able to provide the preprocessed data of the EdNet dataset at this stage. Please download the ``train.csv`` from [this kaggle page](https://www.kaggle.com/c/riiid-test-answer-prediction/data), rename it to "ednet.csv".

Then, create a new directory ``Dataset`` and put ``ednet.csv`` into this directory.

```
mkdir Dataset
```

To preprocess the ``ednet.csv`` file, run the preprocessing script.

```
python preprocessing.py --dataset=ednet 
```

## Training and Testing HiTSKT


Running the HiTSKT model on the ASSISTment 2017 dataset:

```
python main.py --dataset='2017' --epoch_num=100 --batch_size=64 --session_size=16 --action_size=64 --embedding_size=256 --learning_rate=5e-5 --d_inner=2048 --n_layers=1 --n_head=4 --d_k=64 --d_v=64 --dropout=0.1 
```

Running the HiTSKT model on the Junyi dataset:

```
python main.py --dataset='Junyi' --epoch_num=50 --batch_size=64 --session_size=16 --action_size=32 --embedding_size=128 --learning_rate=5e-5 --d_inner=1024 --n_layers=1 --n_head=2 --d_k=64 --d_v=64 --dropout=0.1 
```

Running the HiTSKT model on the EdNet dataset:

```
python main.py --dataset='ednet' --epoch_num=40 --batch_size=64 --session_size=16 --action_size=32 --embedding_size=128 --learning_rate=8e-5 --d_inner=1024 --n_layers=1 --n_head=2 --d_k=64 --d_v=64 --dropout=0.1 
```
