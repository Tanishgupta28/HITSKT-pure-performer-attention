# Junyi adaptive assessment workstream

Started 2026-11-23 from `5492344e320a5fbbfec9a7f180f09bc452d5a910` on `codex/junyi-assessment`, in the isolated `D:/CS/HITSKT-platform-junyi` worktree. The source checkout contains an unrelated local environment-example edit. Scope is the platform and its tests, deployment instructions and memory. The research paths and GPU workstream are read-only.

## Phase 1 — bank and mappings

Verified the provided Drive CSV against manifest SHA-256 `579c11905685790bb3e3da086493dceee1c36b0f552e9a9cfcfca3682991f7a0`. The real importer accepted 3,978 questions, three per each of the 1,326 official tags. Stage coverage is elementary 2,352, junior 1,617 and senior 9. The ranking from handoff `9ff17e674c37c3a9ad59201133f6fac9acc99381` supplies five valid probes for 1,323 tags, four for two and three for one. Mapping digest `041f2980dcc3d233223b38fcdd8bb5a3e997218b3b279971451e7e0e0b2da974`. The imported bank and ranking are in ignored local runtime storage. The bank never ships in the repository.

## Runtime dependency

The expected private passphrase file is absent on this machine. The authorized reduced Student 1 history remains pending. The encrypted archive is verified and stored in `.platform-runtime/junyi/junyi-adaptive-runtime-v1.tar.gz.gpg`. The full checkpoint was verified against SHA-256 `2308c5bf0c850da1e55daf802a0853a30fac45eeed071c7162c68fc17bfa3bdd` and stored in `.platform-runtime/junyi/best_model.pt`. Both paths are ignored. Fixture checks are independent of the missing history.

## Phases 2–7 — delivered and verified to fixture scope

2. The read-only adapter calls the original model and collator. The pinned checkpoint passed 64 short synthetic-history probes with causal placeholder invariance and a direct-forward delta below `1e-6`. The real Student 1 forward is pending the private approved input.
3. The verified importer, 20-question selection and frozen bank/option contract are connected to MongoDB. Locks, concurrent creation, concurrent answer writes and idempotent retry were exercised with isolated fixture accounts.
4. Gemini selection and grounded resource jobs have independent prompts and verification. Structured selection passed a live metadata-only check with 3.1 Flash-Lite and 3 Flash Preview. Live grounded requests returned HTTP 429. The curated fractions link verified live and provider failures passed controlled tests. The operator override is explicit and audited.
5. Genuine app responses are isolated from historical Junyi data. Fixtures covered a timestamp-based session boundary, legal prefix replay, reversible proxy mapping, actual results and the next paper with the new snapshot.
6. Next.js connects Today, Learn, Progress, the player, and the authorized admin controls. Standard focused practice remains available. Mobile UI was visually examined and a full Edge browser fixture went from sign-in through the next paper.
7. The verification record is `reports/adaptive_junyi_student1/TEST_RESULTS.md` and local setup is `docs/platform/junyi.md`. Next checkpoint is to provision the authorized history/passphrase and account bindings, verify the real input contract, then run the live Student 1 assessment and retest search grounding with available quota. Do not substitute fixture results for that run.
