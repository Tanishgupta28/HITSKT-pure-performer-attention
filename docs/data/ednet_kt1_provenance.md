# EdNet-KT1 provenance and schema

Status: acquired and integrity-validated on 2026-09-09 UTC. Raw archives are
local, ignored, and are not committed to Git.

## Authoritative source

- Project: [riiid/ednet](https://github.com/riiid/ednet)
- Inspected source revision:
  [`27db572eeaf4455a1f6c029ba27e34f78bb49d73`](https://github.com/riiid/ednet/tree/27db572eeaf4455a1f6c029ba27e34f78bb49d73)
- KT1 link published by the project: `http://bit.ly/ednet_kt1`
- Resolved Google Drive object ID: `1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw`
- Contents link published by the project: `http://bit.ly/ednet-content`
- Resolved Google Drive object ID: `117aYJAWG3GU48suS66NPaB82HwFj6xWS`

## Verified artifacts

| Artifact | Bytes | SHA-256 | Integrity result |
| --- | ---: | --- | --- |
| `EdNet-KT1.zip` | 1,201,163,816 | `0d13933f90201c5101c7fe8659e44474fa049e3fb93181a8ba6fb3e63267b535` | Full `unzip -tqq` passed |
| `EdNet-Contents.zip` | 173,976 | `aa910a0436d9dbac0ba232f55e27b37ad8d39da285fcf15f1c1eb5d064be98e7` | Full `unzip -t` passed |

`EdNet-KT1.zip` contains exactly 784,309 non-directory student CSV members,
matching the official repository. Their summed uncompressed member size is
3,072,366,053 bytes. Member paths have the form `KT1/u{integer}.csv`.

## Interaction schema

Every inspected student member uses this header:

```text
timestamp,solving_id,question_id,user_answer,elapsed_time
```

- `timestamp`: Unix timestamp in milliseconds, shifted for privacy by the
  dataset publisher.
- `solving_id`: one-based identifier for a delivered question bundle. The
  official documentation calls this a learning session, but it is not the
  project's inactivity-derived session ID and is not a unique event index.
- `question_id`: `q{integer}` identifier joined to `questions.csv`.
- `user_answer`: submitted choice `a` through `d`.
- `elapsed_time`: time spent on the question in milliseconds.
- Student identity: derived from the member filename `u{integer}.csv`.

Correctness is not stored in KT1. It will be derived deterministically as
`user_answer == correct_answer` after a validated many-to-one join on
`question_id` to official `questions.csv`. Any missing or duplicate metadata
key is a preprocessing failure, not a row to infer or silently discard.

KT1 has no unique interaction/event index. `solving_id` identifies a bundle and
can repeat across questions. Therefore, consistent with the approved tie rule,
the original row ordinal within each student CSV is retained and used as the
deterministic tie-breaker for equal timestamps. The 10-hour inactivity rule,
not `solving_id`, defines the benchmark sessions.

## Question metadata schema

Official `contents/questions.csv` uses:

```text
question_id,bundle_id,explanation_id,correct_answer,part,tags,deployed_at
```

Observed metadata facts:

- 13,169 question rows and 13,169 unique question IDs.
- No missing `tags` values.
- Tags are semicolon-delimited integer identifiers.
- 7,120 questions have one tag; 6,049 have multiple tags.
- The largest tag set contains seven tags.
- There are 1,792 distinct tag-set strings before canonicalization.

## Composite-skill protocol

The user confirmed that EdNet uses the same complete sorted composite-tag rule:
all tags associated with a question form one canonical set and one persisted
skill ID. Interactions are never expanded and no primary tag is selected.

Numeric canonicalization of the metadata produces a provisional 1,495 distinct
tag sets when repeated IDs are collapsed, so the vocabulary is not unexpectedly
large for HiTSKT's embedding approach. The user designated the 797 questions
whose literal value is `tags = -1` as missing metadata. They retain every
interaction and share composite skill ID `1`, named `<UNTAGGED>`; `-1` is
excluded from the 188 genuine original tags.

The fail-closed real-data smoke test identified 197 questions that repeat one
tag ID within the semicolon list. Tag `176` is repeated in 85 questions, `178`
in 75, and `177` in 37; these comprise 83 distinct raw strings. The user
confirmed that tags form a mathematical set, so repeated IDs are removed before
numeric sorting. For example,
`152;147;163;179;178;149;178` canonicalizes to
`147;149;152;163;178;179`. No other format outside positive
semicolon-delimited integers and the exact approved `-1` sentinel was found.

## Unanswered-interaction policy

After tag canonicalization passed, a real-data smoke run found an empty
`user_answer` in `KT1/u4.csv`. The user directed that empty/missing responses be
excluded from supervised sequences, never labeled incorrect, and preserved
separately for audit. Sessions, sequence lengths, splits, and attempt counters
are rebuilt after this filtering.

The completed first-1,000-student validation covered 1,427,687 original
interactions and found 2,898 unanswered interactions across 349 students, an
exclusion rate of 0.2029856684%. The other four KT1 columns had no empty cells
in this sample. All 2,898 source rows were written exactly once to audit
Parquet. Filtering changed the session count for 19 students (net -18 sessions)
and changed the grouping of answered events for one student. The rebuilt
supervised output contains 1,409,948 interactions from 704 students meeting the
five-session threshold, organized into 32,795 sessions.

The completed full pass covered all 784,309 files and 95,293,926 source
interactions. It found 27,646 unanswered interactions across 4,477 students
(0.0290112929%); all are preserved in audit data and excluded from supervision.
After filtering, 95,266,280 supervised interactions remained. Session counts
changed for 148 students (net -150 sessions), and the grouping of answered
events changed for six students, so downstream sessions and counters were
rebuilt rather than reused.

Applying the minimum-five-session rule retained 116,548 students, 81,940,867
interactions, and 2,577,988 sessions. The chronological split contains
53,990,022 training, 14,520,975 validation, and 13,429,870 test interactions.
The final tag report contains 188 genuine original tags, 1,495 composite skill
IDs, 797 `<UNTAGGED>` questions, 19,430 source interactions on those questions,
19,318 supervised interactions after unanswered filtering, and 18,307 retained
benchmark interactions.

Independent validation streamed all 82 supervised Parquet shards and reconciled
every row, student, session, split, mapping, distribution, and all 27,646 audit
rows. Versioned aggregate outputs and deterministic mappings are under
[`reports/datasets/ednet_kt1/`](../../reports/datasets/ednet_kt1/).

## Downstream variable-length representation

All 81,940,867 retained supervised interactions were rebuilt into a lossless
memory-mapped store of 2,577,988 sessions. The maximum complete session has
13,080 actions. No action is truncated and no session is chunked. Every
post-first session is a target using up to 15 strictly earlier complete
sessions. Length bucketing, dynamic per-batch padding, a 32,768-token budget,
and a maximum batch size of 64 make long sessions automatically use smaller
batches. An indivisible over-budget example remains whole in a singleton batch.
The actual full split plans and worst-case CUDA result are in
[the shared batching report](variable_length_batching.md).
