# junyi preprocessing report

- Source rows: 14,660,217
- Source SHA-256: `8a18000806279ae4feae90a94a5af75b7688f2e4ba666c9a0662c75e1a404b6a`
- Source students: 29,865
- Questions: 25,630
- Genuine original tags/skills: 1,326
- Deterministic composite skill IDs: 1,521
- Questions with multiple observed row-level skills: 776
- Interactions on those questions: 961,084 (6.555728%)
- Missing target labels: 0
- Retained students (at least five rebuilt sessions): 29,865
- Excluded students: 0
- Retained interactions: 14,660,217
- Retained sessions: 600,154
- Train interactions: 9,647,042
- Validation interactions: 2,556,773
- Test interactions: 2,456,402

IDs are determined by numeric sorting of the complete source vocabularies, with
`0` reserved for padding. Each question's complete set of tags observed in the accepted full source is deduplicated and numerically sorted into one composite skill. Every occurrence receives that one ID; rows are never expanded. Rows are ordered per student by
timestamp and original CSV row order for equal timestamps. Sessions are rebuilt at
gaps of at least 10 hours. Per-student sessions use chronological 60/20/20
splits with `floor(n/5)` validation and test sessions and earlier remainder
sessions in training. All positions and attempt counters are rebuilt afterward.
