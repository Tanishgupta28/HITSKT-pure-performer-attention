# Lumen synthetic mathematics interactions

This reproducible, fictional dataset contains **864 attempts, 72 learning
sessions, 18 learners, and 48 distinct versioned mathematics questions**.
There are 12 train, 3 validation and 3 test learners. No learner spans splits.
There are no real learners, names, emails, credentials or benchmark ID mappings.

- `interactions.csv`: one record per submitted answer, including learner/session
  aliases, question and content version, topic, skill, zero-based response index,
  binary correctness, response duration in milliseconds, UTC attempt timestamp,
  hint count, mode, source and split.
- `questions.csv`: the original Lumen question text, four choices, answer index,
  explanation and content version. Keep answer keys out of an assessment client.
- `manifest.json`: deterministic seed, split unit, counts, schema and SHA-256s.

Generate: `python backend/tools/demo_dataset.py`. `--output` can select a
different output directory. Stable IDs and timestamp order are for engineering
checks. The simulation is not a deployment evaluation or evidence of learning
gains, model superiority, calibrated mastery or benchmark reproducibility.

The local sample accounts use a demo-only cohort. They start with scenarios
based on this generator; the application creates new `synthetic`-realm records
as a panelist uses it. Those records are independent of the frozen CSVs.
