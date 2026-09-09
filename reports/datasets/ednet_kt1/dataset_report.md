# EdNet-KT1 preprocessing report

Generated from the genuine full official EdNet-KT1 archive.

## Tag and composite-skill mapping

- Genuine original tags (excluding `-1`): 188
- Questions with missing tag metadata (`-1`): 797
- Questions whose repeated tag IDs were deduplicated: 197
- Multi-tag questions: 6,049
- Unique composite skill IDs (including `<UNTAGGED>`): 1,495
- `<UNTAGGED>` composite skill ID: 1

Every genuine tag set is numerically sorted and serialized with semicolons. One
stable contiguous composite ID represents the entire set. No interaction is
expanded and no primary tag is selected. The literal `-1` is missing metadata,
not a genuine tag; all such questions use the one `<UNTAGGED>` skill ID.

## Full source interactions

- Students/files: 784,309
- Interactions: 95,293,926
- Unanswered interactions excluded from supervision: 27,646
- Students with at least one unanswered interaction: 4,477
- Full interaction exclusion percentage: 0.029011%
- First-1,000-student sample interactions: 1,427,687
- First-1,000-student sample unanswered interactions: 2,898
- Sample exclusion percentage: 0.202986%
- Interactions on multi-tag questions: 48,039,039
- Interactions on `<UNTAGGED>` questions: 19,430
- Equal-timestamp adjacent interactions after stable ordering: 1,314,673
- Nonmonotonic timestamp transitions in original file order: 0

Unanswered rows are not labeled incorrect, are not supervised targets, and are
not included in model history. Their exact source fields are preserved under
`audit/unanswered_events/`. Sessions and sequence counters are rebuilt only
after this filtering step.

## Filtering effect on sessions and sequences

- Supervised interactions after unanswered filtering: 95,266,280
- Sequence length reduction: 27,646 interactions
- Raw sessions before filtering: 3,466,942
- Supervised sessions after filtering: 3,466,792
- Students whose session count changed: 148
- Students whose answered-event grouping changed: 6
- Net session-count change: -150

## Benchmark population

- Retained students (at least five sessions): 116,548
- Excluded students: 667,761
- Retained interactions: 81,940,867
- Retained sessions: 2,577,988
- Retained multi-tag interactions: 41,804,909
- Retained `<UNTAGGED>` interactions: 18,307
- Training interactions: 53,990,022
- Validation interactions: 14,520,975
- Test interactions: 13,429,870

Sessions start when the preceding interaction gap is at least 10 hours. Each
student's sessions are split chronologically with `floor(n/5)` validation and
test sessions and the remaining earlier sessions assigned to training. Event
rows retain full chronology so downstream rolling-history examples can use only
observations strictly before each target.

Detailed question, composite, user, and distribution tables are saved under
`mappings/` and `reports/`. Supervised event shards are under `events/`; removed
source rows and per-student session effects are under `audit/`.
