# ASSISTments 2017 provenance and schema audit

Status: accepted source acquired locally; preprocessing intentionally stopped
before skill mapping because the source contains a consequential many-to-many
question/skill relationship.

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

## Open question/skill mapping decision

The real-data fail-closed run found that 697 of 3,162 questions occur with more
than one `skill` value: 682 questions have two observed skills and 15 have
three. These questions account for 440,761 interactions (46.749865% of the
source) and produce 3,874 unique question/skill pairs. The preprocessor has not
selected a primary skill, built a global composite, expanded a row, or silently
accepted varying row-level skills.

The scientifically material alternatives are to preserve each interaction's
supplied row-level skill, or to treat the complete set of skills observed for a
question as one question-level composite skill. Explicit user direction is
required before proceeding.
