# Lumen synthetic mathematics interactions v2

This fictional fixture contains **561 submitted answers, 54 sessions, 18 learners
with recorded answers, and 108 question versions**. The sample class has 24
learners. Six have no diagnostic yet. There are 12, 3 and 3 learners in the
train, validation and test splits. Each split includes all four session modes.
Learners occur in only one split.

The cohort includes a common diagnostic, a released paper with skipped items,
short focused practice, longer mixed practice, and submitted work on an open
paper. The interaction CSV includes submitted answers. A skipped question has
no response row. The live demo models it explicitly in the paper navigator.

| Sample learner | Diagnostic | Released unit check | Practice history |
| --- | ---: | ---: | --- |
| Asha (`synthetic-001`) | 6/12, fractions 0/3, equations 3/3 | 8/12 | Fractions, 3/6 |
| Rohan (`synthetic-002`) | 9/12, fractions 3/3, equations 1/3 | 10/12 | Equations, 3/6. Mixed, 9/12 |

- `interactions.csv` records one submitted answer per row. `response` is a
  zero-based option index. `correct` is binary. Timestamps are UTC and response
  times are milliseconds. `adaptive` is the persisted mode for Mixed practice.
- `questions.csv` contains the versioned prompts, options, answer indexes and
  explanations so records can be joined on question ID and content version.
- `manifest.json` contains counts, the split unit and SHA-256 file hashes.

Run `python backend/tools/demo_dataset.py` for byte-reproducible output. Use
`--output` to generate into another directory. The local demo anchors fixture
dates to the sample class and retains additional attempts made during use.
Names, emails and credentials do not appear in the CSVs. The data supports a
product demonstration and pipeline checks. Learning gain, calibration and
benchmark conclusions require separate evidence.
