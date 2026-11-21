# Platform workstream — Achin

Updated: 2026-11-21 (Asia/Kolkata)

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

## Practice-only product scope - 2026-11-21

Achin removed the Teacher workspace and Assessments after identifying overlap
with mixed practice. The two supported roles are Student and Administrator.
Student navigation is Today, Learn and Progress. Focused and mixed practice,
recommendations, answer review and profile preferences remain available.
Administrators retain content, taxonomy, interactions export and Model Lab.

Teacher sign-in and existing teacher sessions are blocked. Class, paper and
diagnostic routes are removed. New practice uses /api/practice and /practice.
Previously saved practice links redirect. The student history and completed
set counts show only practice. MongoDB records and knowledge evidence already
incorporated into learner profiles are preserved. Fresh demos seed only the
practice subset of the immutable v2 fixture. See docs/platform/product.md.

The production build, 22 backend tests and 7 Edge browser tests passed. Browser
coverage used an isolated demo database and included mobile registration, mixed
and focused practice, retry/resume, history, admin export and Model Lab.

## Distinct demo learner and class histories — 2026-11-20

Achin requested an inconsistency audit and greater variety across the sample
learners. The new v2 fixture uses a common 12-question baseline with grades
of 6/12 for Asha and 9/12 for Rohan. Their released unit checks are 8/12 and
10/12, and practice differs by topic, length and mode. The demo class includes
24 learners, six without a diagnostic, and an open paper with three submissions
awaiting release. The frozen `backend/data/lumen-synthetic-v2/` fixture has 561
submitted answers, 54 sessions, 18 responding learners and 108 question versions.
Every split contains all four modes. The historical v1 CSVs remain available.

Local seeding recognizes old fixture IDs, keeps credentials and user-generated
answers and recomputes estimates using the changed sample evidence. Seeded
paper dates follow assignment, submission, closing and release. The student
list displays actual attempt state, and practice review renders older records
with missing feedback. The class distribution uses correctly labeled, neutral
frequency bars. The research checkpoint and original maths bank are unchanged.
Evidence is in `reports/platform/demo_variety_validation.json`.

## Usability consolidation — 2026-11-20

Achin requested a usefulness audit, fewer repeated features and an easier
question-solving experience. Student navigation is now Today, Learn,
Assessments and Progress. Learn consolidates practice, revision and lessons;
Progress owns both the learning report and session history. Profile/classes
are accessed through the account footer. Legacy revision/history links remain
valid. Mixed practice is clearly separate from formal class papers. Teacher
Overview owns actions/activity; Analytics owns detailed topic comparisons.
Administrator content/data/research features remain distinct and available.

The practice player adds explicit response states, focused feedback, optional
ungraded working space, tab-local drafts, offline messaging and safe retry of
an uncertain submission. The original selection/request ID stay locked until
confirmation. The fixed player saves selected answers before navigation,
leaving or submission review. Grading, roles, timing and release remain server
enforced. No backend, dataset or research source change is part of this pass.

Review decisions and verification are in `docs/platform/usability.md` and
`reports/platform/usability_validation.json`. The local research replay's cold
load exceeded the old 30-second proxy timeout; the frontend now allows a bounded
120 seconds and explains preparation. The unchanged checkpoint's warm direct
CPU replay still returned 0.38643354177474976. The model/content handoff remains
unchanged. No training or GPU interruption was requested.

## Approved three-role product implementation — 2026-11-17

Achin approved the 83-board Lumen design and requested implementation, easy
demo access, public student registration and a dataset. Implemented the
responsive student workspace, teacher classroom and assessment workflows,
administrator question bank, taxonomy, interaction export and Model Lab. The
learning flow uses recorded attempts and BKT. Formal class papers use locked
question versions, server deadlines and explicit teacher release. Scores and
answers stay out of the student's dashboard and estimates until release.

The local `-Demo` option seeds separate synthetic accounts and a demo class
without overwriting existing learner records. `backend/data/lumen-synthetic-v1/`
contains 864 attempts from 18 learners across 72 sessions, with a student-level
split and hashed manifest. Live interaction exports are pseudonymous and are
kept separate from synthetic exports. Live password email requires SMTP.

Next.js 16.4.6 production build, npm production audit, 21 backend tests and four
Edge browser checks passed. Browser writes used an isolated QA database. The
normal local database was restored; proxied health, registration, all four demo
logins and the same-score/different-topic comparison were confirmed. Model Lab
computed the original synthetic replay at 0.386434 on CPU from the verified
checkpoint. This does not establish inference on the live question bank. See
`docs/platform/product.md` and `reports/platform/product_validation.json`.

The latest inspected research base for this work is `f465d84`. The lab reads
the independent CPU audit directly: the historical CUDA AUCs agree within
tolerance, while full metric vectors and response availability remain open.
No training processes, checkpoint assets or research source were changed.

## Latest design refresh — 2026-10-08

Achin requested larger typography/icons, a more polished design, animations,
possible Lottie artwork and a3D-style logo. The frontend now uses locally bundled
DM Sans and Plus Jakarta Sans through `next/font/local`; font licenses and exact
hashes are documented in `docs/platform/design.md`. Core copy/inputs are16px,
desktop nav16px with23px icons, and primary controls52px high. Cards, navigation,
profiles, lessons, assessment choices and feedback share the refreshed styling.

`BrandMark` is an SVG sprout on a layered CSS tile with perspective, highlights,
extrusion and gentle floating motion in the hero. `GrowthScene` lazy-loads
`lottie-web@5.13.0` with original local orbital animation data. Players pause off
screen/in hidden tabs and are destroyed on unmount. Header motion controls persist
the pause preference and stop CSS motion and Lottie; system reduced motion always
takes precedence. Decorative art is hidden from assistive technology.

Development checks passed: TypeScript and two design browser tests, including
local fonts, real icon/type sizes, playback/pause/reload, system reduced motion,
and widths1440/1024/768/390/360 without page overflow. Desktop/mobile screenshots
were reviewed. The motion control was moved into headers after visual review to
avoid overlaying page controls. The production build and all three production
browser tests passed (reported1.7m), including the complete learning cycle.
Final screenshot review prompted two small mobile text-spacing fixes. The final
production rebuild passed, and both focused final design checks passed in15.8s.
Corrected screenshots were reviewed. Local production health is`ok`, and strict
memory validation passed without errors or warnings. No external font requests,
page errors or horizontal page overflow were observed in the design checks.

Fetched/merged the GPU agent's latest memory commit`81ee41a` before the visual
work. Backend/research logic and student data are unchanged. The matching-content
checkpoint request remains the next model-integration action after this design
delivery. Assets and UI instructions are in `docs/platform/design.md`; the
validation index is `reports/platform/design_validation.json`.

Design implementation **`0aa74cf`** was atomically pushed to canonical `main`
and `codex/adaptive-learning-platform`, preserving the GPU agent's merged memory
updates. The final standalone production app remains running locally. Fonts,
licenses and original animation data are committed; screenshots and runtime
records stay local and ignored. Next bounded model action remains the reviewed
matching-content/checkpoint handoff, with no training interruption requested.

## Profile enrichment — 2026-10-08

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

Enrichment implementation **`4ed99e7`** was atomically pushed to canonical
`main` and `codex/adaptive-learning-platform`. Strict memory validation passed
without errors or warnings. Profile backups, screenshots, failed-test trace and
runtime process records stay local and ignored. No personal database data was
pushed. The updated production app remains running at`http://localhost:3000`.

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

Mathematics is the only subject. The response aggregate caps a subject at 5,000
practice answers pending an archival storage migration. Classroom and content
administration are available as of the November product implementation.

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
