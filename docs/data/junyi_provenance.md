# Junyi provenance and composite-skill report

Status: the accepted full 14.66M-row source has been processed and independently
validated.

- File: `Junyi.csv`
- Bytes: 671,494,411
- SHA-256: `8a18000806279ae4feae90a94a5af75b7688f2e4ba666c9a0662c75e1a404b6a`
- Rows/interactions: 14,660,217
- Students: 29,865
- Questions: 25,630
- Genuine original tags: 1,326
- Missing required values or target labels: none
- Equal-timestamp ordering: original CSV row order, because the accepted source
  has no authoritative unique interaction index.

## Complete tag-set mapping

Each question's complete set of tags observed in the accepted full source is
deduplicated and numerically sorted. The complete set receives one stable
composite skill ID and every interaction for that question receives that single
ID. Interactions are never expanded and no primary tag is selected.

There are 1,521 distinct composite skills. Of 25,630 questions, 776 have
multiple observed tags: 774 have two and two have three. Multi-tag questions
account for 961,084 interactions (6.555728%). Deterministic original-tag,
question, question-to-composite, and composite mappings are versioned under
[`reports/datasets/junyi/`](../../reports/datasets/junyi/).

## Validated preprocessing result

All 29,865 students have at least five rebuilt sessions, so the final build
retains all 14,660,217 interactions in 600,154 sessions. The chronological split
has 9,647,042 training, 2,556,773 validation, and 2,456,402 test interactions.
Independent streaming validation passed all 15 event shards, original-row tie
ordering, binary targets, exact 10-hour boundaries, rebuilt positions, split
chronology, composite mappings, and aggregate distributions.

## Downstream variable-length representation

The validated 14,660,217 interactions were rebuilt into a lossless
memory-mapped store of 600,154 sessions. The maximum complete session has 3,924
actions. No action is truncated and no session is chunked. Every post-first
session is a target using up to 15 strictly earlier complete sessions.
Length-bucketed, dynamically padded batches use a 32,768-token budget and
maximum batch size 64; indivisible longer examples use singleton batches. See
[the shared batching report](variable_length_batching.md).
