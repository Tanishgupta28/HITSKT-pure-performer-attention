# ASSISTments 2017 provenance and schema audit

Status: accepted source acquired locally. The user explicitly resolved the
many-to-many question/skill relationship in favor of preserving supplied
interaction-level skills.

- File: `2017.csv`
- Bytes: 82,778,007
- SHA-256: `7577287d4ead073fdb9393b586a260043151f27338246bd076ca740e28be8bbb`
- Rows: 942,807
- Students: 1,709
- Questions (`problemId`): 3,162
- Row-level skills: 102
- Binary targets: 351,366 correct and 591,441 incorrect
- Missing required values: none
- Authoritative within-student event order: `action_num`; original CSV row is
  retained as the final deterministic fallback.
- Native `startTime` unit: Unix seconds, verified from the accepted file (for
  example `1096470301`). Preprocessing stores
  `timestamp_ms = startTime * 1000`; model elapsed times use
  `delta_hours = delta_timestamp_ms / 3,600,000`.

## Interaction-level skill decision

The real-data fail-closed run found that 697 of 3,162 questions occur with more
than one `skill` value: 682 questions have two observed skills and 15 have
three. These questions account for 440,761 interactions (46.749865% of the
source) and produce 3,874 unique question/skill pairs.

No authoritative complete question-to-skill/tag metadata is available in the
accepted dataset. The supplied row-level `skill` is therefore retained for its
specific interaction. Skills observed on other rows are not inferred, merged,
or used to create a question-level composite. No interaction is expanded and
no primary skill is selected. If authoritative complete metadata is later
found, this policy must not change without explicit user review.

## Validated preprocessing result

The completed build preserves all 942,807 source interaction rows through skill
mapping. After rebuilding sessions and excluding students with fewer than five,
1,388 students, 885,335 interactions, and 12,402 sessions remain. The split has
647,283 training, 125,800 validation, and 112,252 test interactions. Independent
streaming validation passed the event shard, chronological `action_num` tie
ordering, 10-hour boundaries, session positions, split ordering, mappings, and
all aggregate totals. Versioned outputs are under
[`reports/datasets/assist2017/`](../../reports/datasets/assist2017/).

## Downstream variable-length representation

The validated events were rebuilt into a lossless memory-mapped session store
that persists normalized `timestamp_ms` for every event:
885,335 interactions, 12,402 sessions, and a maximum complete session length of
938. No action is truncated and no session is chunked. Every post-first session
is a target using up to 15 strictly earlier complete sessions. Length-bucketed,
dynamically padded batches use a 32,768-token budget and maximum batch size 64.
The full split plan and real CUDA benchmark are in
[the shared batching report](variable_length_batching.md).
