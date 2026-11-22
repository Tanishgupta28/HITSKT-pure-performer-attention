# Junyi pilot verification

## Verified production inputs

- The approved 806,537-byte bank matches SHA-256 `579c11905685790bb3e3da086493dceee1c36b0f552e9a9cfcfca3682991f7a0`. The real runtime importer accepted 3,978 questions from 1,326 original tags, three questions per tag. Stage totals are 2,352 elementary, 1,617 junior and 9 senior. All 1,326 tags have matched model representatives.
- Allocation using the real bank and synthetic skill categories produced 20 unique questions with the 8/6/4/2 mix for Elementary, Junior and all stages. Senior produced the explicit nine-question shortage. This check wrote no learner records or papers.
- The 99,330,610-byte Junyi checkpoint matches SHA-256 `2308c5bf0c850da1e55daf802a0853a30fac45eeed071c7162c68fc17bfa3bdd`. The original-model smoke checks exercised 64 of 6,307 legal mapped probes with synthetic earlier history, including a 9-session / 179-interaction test context. BOS, shifted-prefix and future-placeholder checks passed. Direct forward parity differed by at most `8.94e-8` in the two runs. This is checkpoint-backed smoke evidence using test history.
- Gemini structured selection accepted an allowed synthetic question ID with `gemini-3.1-flash-lite` and `gemini-3-flash-preview`. Google Search grounding returned HTTP 429 on the live checks. A live verification of the curated fraction-addition link succeeded. The Gemini provider tests also covered failure, missing grounding evidence, unreachable and off-topic links, duplicates and the four-link limit.

## Application checks

- `python -m pytest backend/tests -q` passed 28 tests. Next.js type checking and the production build both passed.
- The MongoDB fixture covered a single active paper across parallel creation requests, required study only for weak skills on that frozen paper, locked question reads and answer writes, an audited operator override, real grading of fixture selections, exact retry behavior, exclusive progression, persisted proxy outcomes and the next paper using the updated snapshot.
- The Microsoft Edge browser fixture completed all 20 displayed questions after study and operator unlock, reloaded and resumed the same paper and draft, reached results and opened the next assessment. The student question player was also visually examined at 390 pixels wide.

## Pending Student 1 run

The private history could not be decrypted on this device because the authorized passphrase file was absent. The original 9-session / 179-interaction production baseline, actual first paper, private learner probabilities, live grounded resources, and actual submission outcomes have no live verification here. Fixture outcomes were stored only in disposable test databases. No learner histories, answer keys, provider credentials or model artifacts belong in Git.
