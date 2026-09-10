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
visible to its own prediction.

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
