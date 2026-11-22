# Junyi adaptive assessment workstream

Started 2026-11-23 from `5492344e320a5fbbfec9a7f180f09bc452d5a910` on `codex/junyi-assessment`, in the isolated `D:/CS/HITSKT-platform-junyi` worktree. The source checkout contains an unrelated local environment-example edit. Scope is the platform and its tests, deployment instructions and memory. The research paths and GPU workstream are read-only.

## Phase 1 — bank and mappings

Verified the provided Drive CSV against manifest SHA-256 `579c11905685790bb3e3da086493dceee1c36b0f552e9a9cfcfca3682991f7a0`. The real importer accepted 3,978 questions, three per each of the 1,326 official tags. Stage coverage is elementary 2,352, junior 1,617 and senior 9. The ranking from handoff `9ff17e674c37c3a9ad59201133f6fac9acc99381` supplies five valid probes for 1,323 tags, four for two and three for one. Mapping digest `041f2980dcc3d233223b38fcdd8bb5a3e997218b3b279971451e7e0e0b2da974`. The imported bank and ranking are in ignored local runtime storage. The bank never ships in the repository.

## Runtime dependency

The original handoff's private passphrase file was unavailable on this machine. The encrypted archive is verified and stored in `.platform-runtime/junyi/junyi-adaptive-runtime-v1.tar.gz.gpg`. The full checkpoint was verified against SHA-256 `2308c5bf0c850da1e55daf802a0853a30fac45eeed071c7162c68fc17bfa3bdd` and stored in `.platform-runtime/junyi/best_model.pt`. Both paths are ignored. The verified raw-source recovery below supersedes the passphrase dependency for this workstation.

## Phases 2–7 — delivered and verified to fixture scope

2. The read-only adapter calls the original model and collator. The pinned checkpoint passed 64 short synthetic-history probes with causal placeholder invariance and a direct-forward delta below `1e-6`. The real Student 1 forward is pending the private approved input.
3. The verified importer, 20-question selection and frozen bank/option contract are connected to MongoDB. Locks, concurrent creation, concurrent answer writes and idempotent retry were exercised with isolated fixture accounts.
4. Gemini selection and grounded resource jobs have independent prompts and verification. Structured selection passed a live metadata-only check with 3.1 Flash-Lite and 3 Flash Preview. Live grounded requests returned HTTP 429. The curated fractions link verified live and provider failures passed controlled tests. The operator override is explicit and audited.
5. Genuine app responses are isolated from historical Junyi data. Fixtures covered a timestamp-based session boundary, legal prefix replay, reversible proxy mapping, actual results and the next paper with the new snapshot.
6. Next.js connects Today, Learn, Progress, the player, and the authorized admin controls. Standard focused practice remains available. Mobile UI was visually examined and a full Edge browser fixture went from sign-in through the next paper.
7. The verification record is `reports/adaptive_junyi_student1/TEST_RESULTS.md` and local setup is `docs/platform/junyi.md`. That fixture milestone left real history and account bindings pending. See the follow-up below for the recovery distinction.

## Follow-up — independently verified raw-source recovery

Achin authorized an alternative because the research host and passphrase are unavailable. The local `Junyi.csv` matched the accepted source SHA-256 `8a18000806279ae4feae90a94a5af75b7688f2e4ba666c9a0662c75e1a404b6a`. The offline tool `backend/tools/rebuild_junyi_student1.py` reconstructed and matched all 11 member hashes in `reports/independent_correctness_audit/junyi_evaluation_manifest.json` and the complete processed student crosswalk hash in `junyi_data_manifest.json`. It verified source ID 1 → processed ID 1 and wrote only the 9 approved sessions / 179 interactions before the first test event, under ignored `data/processed/junyi/adaptive_runtime_v1/`. Phase 1 now has the verified real input. The same original checkpoint and framework are used for the follow-up run. The isolated loopback pilot runs on `http://localhost:3108`, with private account configuration under `.platform-runtime/junyi/`.

The real initial run scored all 6,307 distinct probes and 1,326 tags. Serving reuses shared encoder outputs and calls the checkpoint's original decoder, with no research-source changes. Direct comparisons differed by at most `1.2e-7` for BOS and an observed prefix. Full CPU scoring completed in 25.47 s in the local comparison run. The authenticated loopback app resumed the original paper ID and persisted 20 unique questions. It used 14 unattempted, 4 review and 2 broad questions because the approved history has no attempted Weak skill under the frozen thresholds. No study gate was required for this paper. Gemini supplied 11 validated IDs and rules filled nine. No answers were recorded; the next live checkpoint needs the student's own submitted responses.

The production Next.js build on loopback passed an Edge sign-in and resume check with the real assigned account. A cold-form check revealed the native GET default before hydration, so the auth form now submits by POST and holds the input controls until hydration. The previous temporary password was rotated and its local sessions revoked. A browser check confirmed the fallback form was safe with JavaScript disabled, the fresh sign-in used the API, and the real player opened question 1 of 20 with zero submissions.
