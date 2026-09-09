# assist2017 preprocessing report

- Source rows: 942,807
- Source SHA-256: `7577287d4ead073fdb9393b586a260043151f27338246bd076ca740e28be8bbb`
- Source students: 1,709
- Questions: 3,162
- Genuine original tags/skills: 102
- Deterministic composite skill IDs: 102
- Questions with multiple observed row-level skills: 697
- Interactions on those questions: 440,761 (46.749865%)
- Missing target labels: 0
- Retained students (at least five rebuilt sessions): 1,388
- Excluded students: 321
- Retained interactions: 885,335
- Retained sessions: 12,402
- Train interactions: 647,283
- Validation interactions: 125,800
- Test interactions: 112,252

IDs are determined by numeric sorting of the complete source vocabularies, with
`0` reserved for padding. The supplied skill on each interaction is preserved. No authoritative complete question-tag mapping is available, so skills are not merged across rows and interactions are never expanded. Rows are ordered per student by
timestamp and authoritative `action_num` with source row as fallback. Sessions are rebuilt at
gaps of at least 10 hours. Per-student sessions use chronological 60/20/20
splits with `floor(n/5)` validation and test sessions and earlier remainder
sessions in training. All positions and attempt counters are rebuilt afterward.
