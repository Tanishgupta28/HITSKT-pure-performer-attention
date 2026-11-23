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

## Follow-up completion and Phase 2 replay

All eight study requirements were marked complete by the learner, and the follow-up paper recorded 20 genuine answers. The model snapshot used all 40 app answers from both papers plus the 179 approved original interactions. It returned estimates for 1,326 skills. A direct run of the original checkpoint on 12 representative inputs matched the stored estimates within `2.99e-8`. The authenticated results endpoint confirmed the second paper and its analysis are complete.

## Phase 3 assessment backend closeout

- Reconciled all 40 genuine grades across the first two papers against the versioned question bank and the saved option order. Confirmed the required study completions preceded grading and that an exact live answer retry preserved the original data. A changed retry, an unauthenticated read and a read of an already completed question were rejected.
- Added regression coverage for queued quota and repeat settings surviving a restart and for detection of a stale snapshot with a matching answer count. All 30 backend tests passed.
- Both completed results survived a backend restart. Simultaneous live starts returned the same third paper. Its verified answer history contained exactly the 40 completed learner submissions. The paper held 20 questions with no repeats from the two earlier papers. A shortage of two unused review questions was recorded and those slots went to targeted skills under the existing selector rules.
- The third paper has ten verified study topics. Three additional curated lessons were verified through the live resource verifier. The frozen items remained unchanged during retries. The browser showed the resource checklist, and the authenticated question endpoint returned HTTP 423 while study was pending. There were zero submissions on the third paper during verification. The private local second-paper result file has 20 rows, and its separate history export has 219 original and genuine application records.

## Phase 4 question selection and study recommendations

| Check | Evidence |
| --- | --- |
| Existing third paper | Verified all 17 Gemini picks against the exact eligible group lists for the saved 40-answer snapshot. All 20 questions are unique and unseen across the first two papers. |
| Live question-selection replay | One request to `gemini-3.1-flash-lite` produced 17 valid picks, with rules providing three. Final allocation was 10 weak, 6 unattempted, 2 review and 2 broad. The metadata sent went from 8,911 bytes in the previous shape to 3,589 bytes in the revised shape. The measured live call used 1,222 input tokens and 152 output tokens and took 3.88 s on the pilot connection. |
| Question fallback | A controlled HTTP 429 and a separate replay with Gemini disabled produced the same 20-question rules paper, with the same allocation as live Gemini. One incidental live HTTP 503 also triggered deterministic selection successfully. |
| Verified study links | All ten existing links were fetched successfully over HTTPS and met the corresponding topic checks. In a separate read-only lookup for equation solving, Gemini returned a valid order for two independently verified lessons. The live request used 137 input tokens and 24 output tokens. |
| Study fallback and expiry | Controlled HTTP 429 preserved both verified equation-solving lessons in catalogue order. Fixtures also covered cooldown, unrelated pages, sign-in pages, unreachable pages, duplicate links, the four-link limit, HTTPS, and expiration of cached verification. |
| Lock and data preservation | After restarting the isolated backend, both completed paper documents, the third paper's frozen items and its study records matched their original fingerprints. Ten required items remained pending, the student question endpoint returned HTTP 423, and the third paper still contained zero answers. |

All 34 backend tests passed. The real-history tests reused the saved snapshot read-only, without saving a replay paper or learner activity. The source model, serving adapter, preprocessing and verified checkpoint were unchanged. Google Search grounding links are excluded from the persistent checklist. The approved English catalogue supplied all ten current topics and requires additional verified coverage for other topics. [Gemini API grounding terms](https://aistudio.google.com/docs/terms#grounding-with-google-search)

## Phase 5 genuine response history and repeatable exports

| Check | Evidence |
| --- | --- |
| Real responses and grade integrity | Reconciled all 40 unique saved responses and both completed results with the frozen question options and the verified bank. The response fingerprints matched the saved analysis and its use in the third paper. |
| Original checkpoint replay | All 40 responses mapped to legal Junyi question and composite skill IDs. At the second paper's completion all 40 were in the observed session prefix after the nine original sessions. The unchanged checkpoint reproduced all 1,326 saved original-only and all 1,326 saved proxy-inclusive estimates exactly. The local CPU replay took 56.12 s. |
| Delayed analysis and missing mappings | Fixture tests confirmed that a retry at the exact 10-hour boundary produces the original paper snapshot and that a subsequent paper uses the new session window. An unrelated or missing representative preserved the saved grade and produced the explicit unavailable analysis state when the context could not be scored. |
| Local exports | The verified export included 40 assessment rows and 219 history rows. All 40 joins by paper number and request ID matched grade, timestamp, bank item and tag. A repeat was byte-identical, and an export through the incomplete third paper was rejected. Both outputs are ignored under `reports/adaptive_junyi_student1/private_exports/through-paper-02/`. |
| Preservation | Both completed paper documents, the existing third paper and its study records matched their originals. Ten study items remained pending, question access returned HTTP 423, and there were zero new submissions. The research source, preprocessing, model adapter, baseline and pinned checkpoint remained unchanged. |

All 37 backend tests passed. After restarting the local backend, Edge showed both completed results, the original-versus-app estimates and the third paper's unchanged checklist. Tests that simulate additional submissions used isolated fixture data. A real run with 60 answers and the subsequent paper is pending learner completion of the third test.

## Phase 6 live third result and new paper

| Check | Evidence |
| --- | --- |
| Real result and history | Verified three completed papers, 60 genuine grades and ten required study completions on the third paper. The third paper's report matches the exact 60-answer fingerprint. The authorized local export has 60 results plus a combined history of 179 original and 60 app events. |
| Unchanged full checkpoint replay | Exact agreement for the 1,326 original-only and 1,326 proxy-inclusive estimates in the saved snapshot. The protected baseline, checkpoint, model and preprocessing were preserved. |
| Test 4 selection and study | One saved paper with 20 unique, unseen questions. Allocation was 8 weak, 6 unattempted, 4 review and 2 broad. There were 18 valid Gemini selections and 2 rules selections. Verified eight lesson links after retrying three follow-up study lookups. Resume returned the same paper. It remained locked with zero answers. |
| Frontend changes | Added test numbers, saved evidence counts and a practised-skills filter. Required and optional study use the saved paper's skill tags. Older responses cannot overwrite a newer checklist and verified study pages no longer show a retry action. |

Phase 6 used read-only checks for completed papers and exact checkpoint replay. Creating the next paper and retrying three study lookups used the normal authenticated student endpoints. The verification did not complete any lessons or record any answers on Test 4.

All 37 backend tests, all three isolated Edge tests, frontend type checking and the production build passed. The isolated browser suite covered sign-in, mobile submission through results, and an out-of-order checklist response. The restored local production site passed a separate authenticated Edge run against the real account, including the numbered results, the 60-answer report and the eight available lessons on a 390-pixel viewport. No question options were visible while the study lock was active.

## Phase 7 end-to-end verification and handoff

| Check | Result |
| --- | --- |
| Genuine results and lock | Read all three completed results through the authenticated API and verified the exact 60-response fingerprint in both the report and the saved next-paper snapshot. Test 4's eight study links were independently fetched and matched to their topics. Its question endpoint continued to return HTTP 423. All four paper documents matched their earlier audit exactly. |
| Original HiTSKT | The original checkpoint reproduced every field of the saved 1,326-skill snapshot at the Test 3 cutoff. The read-only CPU replay took 67.74 seconds. The original baseline, checkpoint, source, serving adapter and preprocessing matched their earlier hashes or Git contents. |
| Selection and fallback | All 18 saved Gemini suggestions matched offered candidates in the assigned slots. A rules-only replay of the same 60-answer history produced 20 legal, distinct unseen items in memory with the same 8/6/4/2 allocation. Controlled provider failures are covered by the backend suite. |
| Separate end-to-end fixtures | All 38 backend tests passed. A new progression fixture exercised 60 synthetic grades and a rules-only fourth paper. All three isolated Edge tests passed. The browser submitted an answer, lost the response after server grading, retried the identical request and confirmed that still only one answer was saved. |
| Live browser | Authenticated Edge smoke verified the 60-answer report, original-versus-app comparison, distinct optional recommendations and the locked eight-item checklist on a 390-pixel viewport. Test 4 still had zero answers. |
| Production build | Next.js type checking and a clean production build passed. The production build ran in an isolated directory while the learner's site remained online. |
| Private review | The export repeated byte-for-byte. Its manifest verified nine files, including an inventory of 3,978 questions, 239 history events, 1,326 skill estimates and 6,626 ranked tag/probe rows. The redacted Test 4 summary contains no question IDs, options or grading key. |

See the [architecture](ARCHITECTURE.md), the [dataset and private artifact inventory](DATASET_AND_SKILL_MAPPING.md), and the [inference and proxy limitations](INFERENCE_AND_PROXY_LIMITATIONS.md). The actual next cycle with Test 4 submissions remains dependent on the learner's own study and answers.

## Live Test 4 follow-up

- Reconciled all 80 genuine grades with the verified bank and saved option order. All eight required study completions preceded the first Test 4 submission. The authenticated Test 4 result and learning dashboard used the same 80-answer history as the next paper. A read-only CPU replay taking 81.03 seconds reproduced the entire 1,326-skill snapshot exactly with the original checkpoint.
- Test 5 contains 20 unseen questions and no answers. The deterministic selector produced 12 weak, 6 unattempted and 2 broad questions, reallocating four review slots. Eight missing study topics received independently verified lessons on that same paper. All 12 links were fetched over HTTPS and matched their respective topics. All 12 requirements remained pending, and question access returned HTTP 423. The four completed paper documents and Test 5's frozen questions were preserved.
- All 39 backend tests passed. The authenticated Edge check showed all four reports, the same 80-answer skill comparison and the locked Test 5 checklist on a 390-pixel viewport. The repeatable private review through paper four contains 80 result records and 259 history events, with a redacted Test 5 summary. The protected research code, preprocessing, serving adapter, checkpoint and original baseline were verified unchanged.
