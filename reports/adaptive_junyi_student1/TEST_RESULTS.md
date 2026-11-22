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

## Follow-up to the original fixture verification

The original encrypted history handoff could not be decrypted on this device. The later offline recovery used the accepted raw Junyi source and matched all 11 published SessionStore member digests and the original student crosswalk digest. The real 9-session / 179-interaction baseline now passes the runtime loader with the approved chronological cutoff. Fixture outcomes were stored only in disposable test databases. No learner histories, answer keys, provider credentials or model artifacts belong in Git.

The original checkpoint ran all 6,307 distinct representatives and produced 1,326 finite tag estimates from the real approved history. The optimized serving path used shared encoder outputs and matched direct full-model calls to within `1.2e-7` in both BOS and an observed prefix check. The full optimized comparison run took 25.47 s on this CPU. All 28 backend tests passed after the serving change.

The authenticated pilot resumed and froze one real 20-question paper. The eight attempted-Weak slots were reallocated to unattempted skills under the normal selector policy, resulting in 14 unattempted, 4 review and 2 broad questions. Gemini contributed 11 verified bank IDs and deterministic selection supplied the other nine. The paper opened without a study gate because no attempted skill had a Weak estimate. The website served four options while keeping the grading key private. The paper has zero submitted answers, so live grading and the next real adaptive cycle remain dependent on learner input.

The production frontend build and Microsoft Edge smoke check passed for the real pilot. The test checked POST-only sign-in before hydration, signed in with the current local credential, and resumed the existing paper at question 1 of 20 without submitting an answer. The pre-fix temporary student password was rotated and existing local student sessions were revoked.

## Genuine learning cycle and follow-up study gate

- Subsequently, all 20 learner-submitted answers were saved once and graded against their frozen options. All answers mapped to valid app proxies. The post-paper snapshot used the 20 new app responses and scored all 1,326 tags. Private ignored exports contain 20 genuine result rows and a replayable history of 179 original plus 20 app records.
- The next paper used the post-paper evidence, respected the 8/6/4/2 allocation, and contained 20 distinct unseen questions. It required eight study topics. Verified curated links replaced unavailable search results on the same paper. The authenticated question endpoint continued to deny reads while the eight study requirements were pending. No study completion or follow-up answers were generated during validation.
- The backend suite passed 29 tests after the curated-resource regression check was added. The evidence above concerns the actual pilot; fixture checks remain separate.
