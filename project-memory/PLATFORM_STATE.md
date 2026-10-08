# Platform workstream — Achin

Updated: 2026-10-08 (Asia/Kolkata)

## Scope and ownership

Achin asked for a full-stack platform in the same canonical repository using
FastAPI, Next.js and MongoDB, with development under `D:\CS`, reuse of this
existing memory and collaboration with the separate GPU training agent.
Achin confirmed Git is configured and selected school mathematics as the first
usable subject. The local platform checkout is `D:\CS\HITSKT-platform`.
Work began from `08838ec` on `codex/adaptive-learning-platform`.

The GPU workstream keeps ownership of all scientific code, preprocessing,
training, experiment artifacts, monitoring and reporting. Existing training
memory remains authoritative for that workstream. Do not stop or restart its
processes to deploy the platform. Achin specifically asked whether existing
code was being altered; the platform preserves all research/training source.
Platform commit identity is the locally configured Achin-Agarwal identity;
historic Tanish identity instructions apply to the GPU workstream.

## Latest platform enrichment — 2026-10-08

Achin requested more data and fields after account creation. `backend/app/profile.py`
and `frontend/src/components/learner-profile.tsx` add validated, editable grade,
curriculum, goal, rolling seven-day question target, adaptive session length,
study days and focus concepts. MongoDB stores preferences separately from
assessment evidence. The overview derives a study suggestion and real answer
counts; new adaptive sessions use the saved6/12/18-question length, while active
sessions retain their original length. Grade/curriculum describe preferences,
not full syllabus coverage.

The bank now contains384 original questions, with expanded lessons, practical
word problems and geometry. A UTF-8 source comparison confirmed all original192
question records remain exactly equal, including IDs, grading and versions.
Added items use `math-original-v2`; the original items stay `math-original-v1`.

One requested local account received example preferences marked `sample` via
`backend/tools/seed_sample_profile.py`. The tool preserves saved profiles and
creates a profile-only backup under ignored `.platform-runtime/`. Credential and
learning-document fingerprints were equal before/after; repeat seeding made no
change. No account identifier, credential or private profile data is published.
No synthetic response, mastery or completed assessment was inserted. Saving
preferences through the UI marks them as user-provided.

All14 backend tests and TypeScript/production build passed. The updated local
production runtime is healthy. The expanded Edge production-browser flow passed
in29.5s (34.9s total), including profile save/reload, real target counts, the full
diagnostic cycle, expanded lesson and focused practice. Desktop1440x1080 and
mobile390x844 screenshots were reviewed; no page errors or horizontal overflow.
The first browser run had an exact-label test-selector mismatch on the
curriculum dropdown. Its trace/context are preserved locally under
`.platform-runtime/profile-selector-failure/`; the corrected selector and bounded
action timeout passed. This failure was in the test locator, not profile storage.
Latest evidence: `reports/platform/profile_validation.json` and
`backend/tests/test_profile.py`. Initial validation below remains historical.

Next bounded action is still the matching-content/checkpoint review in
[PLATFORM_HANDOFF.md](PLATFORM_HANDOFF.md). No GPU-run change is requested.

## Implemented platform

- `backend/app/`: FastAPI accounts, Argon2 passwords, hashed opaque session
  cookies, MongoDB TTL sessions/rate limits, owner-scoped assessments, diagnostic
  coverage, answer grading, Bayesian concept estimates, adaptive difficulty,
  prerequisite recommendations, focused practice and progress snapshots.
- Responses, concept updates and next questions persist in one atomic
  version-checked learning aggregate. UUID request IDs provide idempotent replay.
- `frontend/src/`: responsive Lumen account entry, workspace, subject entry,
  concept map, lessons, adaptive assessment/feedback, completion and progress UI.
- `backend/app/inference.py`: optional import-only adapter for unchanged HiTSKT,
  exact vocabulary mappings, checkpoint hash validation and genuine ten-hour
  session semantics. Provider provenance remains explicit.
- `docs/platform/`: reproducible setup, practical applications, API/storage
  documentation and model/content integration contract.

## Initial validation milestone — 2026-10-08

- All **10 platform backend tests passed** against real local MongoDB, including
  diagnostic-to-practice, adaptation, persistence across app restart, concurrency,
  idempotency, account isolation, origin checks and the PyTorch serving adapter.
- TypeScript (`next typegen` plus `tsc --noEmit`) and the final Next.js production
  build passed, including the icon and standalone production startup helper.
- The complete browser flow passed on installed Edge with desktop 1440x1080 and
  mobile 390x844 viewports: register, diagnose, save/resume after reload, finish,
  view progress, open a lesson and begin targeted practice. No page errors or
  mobile horizontal overflow. Screenshots were visually reviewed locally under
  `.platform-runtime/`; they are intentionally not committed as source assets.
- `reports/platform/inference_smoke.json` records **exact adapter/direct-model
  parity** for the actual EdNet checkpoint hash
  `58d4b1f393d0fa8ec372ba8bcc00cc8b7168d37817d4badfd26ee42abffd03c8`.
  This explicitly synthetic, mapped history tests the exact ten-hour session
  boundary and exclusion of the candidate label; it is not an outcome evaluation.
- The initial direct package import failed on benchmark-only dependencies and
  Linux `resource`. `backend/app/research.py` now executes the exact original
  Performer/HiTSKT source without the unrelated package initializer. No research
  file was modified. The adapter tests and real checkpoint smoke then passed.
- Downloaded Chromium failed to launch on this Windows host (`spawn UNKNOWN`).
  Browser validation succeeded with `PLAYWRIGHT_CHANNEL=msedge`. The initial
  five-second assertion also expired during first dev-route compilation; the
  final test uses an explicit 20-second assertion timeout and passed completely.
- The full Edge browser flow also passed against the **standalone production
  build** (37.3s test, 51.0s total). Final screenshots were reviewed.
- Both PowerShell helpers were runtime-validated: production start, stop of the
  recorded owned process trees, verification that app ports closed while MongoDB
  remained running, then production restart. Proxied API health is `ok`; the
  icon/static production route returns HTTP200. Services are running locally.
- Docker packaging is provided but not runtime-validated on this host, which
  has no Docker CLI. `reports/platform/validation.json` is the validation index.
- Memory strict validation passed without warnings after correcting a link
  escaping the memory root. Research source diff checks are empty.

## Established limitation

Benchmark question embeddings cannot be assigned to newly authored math
questions. The 384 original platform items use an explicitly labeled Bayesian
cold-start estimator. Trained HiTSKT integration requires licensed matching
question content, exact model IDs, checkpoint provenance and a real history
fixture. No user-facing claim of trained HiTSKT inference on original content.
The adapter's tiny test checkpoint is untrained and tests the interface only.

The first release contains one subject and no content-admin/teacher dashboard.
The response aggregate caps a subject at 5,000 answers pending an archival
storage migration.

## Collaboration

Read [PLATFORM_HANDOFF.md](PLATFORM_HANDOFF.md) before integrating checkpoint
assets. Keep platform edits under its new folders/docs; coordinate through
this memory, fetch current remote changes before pushing, and never force-push.

## Published implementation checkpoint

- Implementation commit **`a2c27b8`** was atomically pushed to canonical remote
  `main` and `codex/adaptive-learning-platform` on 2026-10-08. Both advanced
  from the inspected scientific base `08838ec`; no force push was used.
- All existing research/training code, benchmark tests, root `pyproject.toml`,
  root README and experiment artifacts were preserved. Existing-file edits
  are `.gitignore` and additive project-memory entries; other files are new.
- The Windows checkout is clean after the implementation commit. Local-only
  dependencies, model cache, screenshots, process records and logs stay ignored.
- The production app runs at `http://localhost:3000` on this Windows host,
  with API at `http://127.0.0.1:8000`. It can be stopped with
  `scripts/stop-platform.ps1` and restarted using
  `scripts/run-platform.ps1 -Mode production`.
- Latest valid evidence is `reports/platform/validation.json` and
  `reports/platform/inference_smoke.json` at implementation commit `a2c27b8`.
  The next bounded action is the reviewed matching-content/checkpoint handoff;
  no GPU run interruption or scientific protocol change is requested.
