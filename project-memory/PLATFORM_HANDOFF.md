# Platform / GPU handoff

Updated: 2026-10-08 (Asia/Kolkata)

## Latest platform-only update

Achin requested richer account data/fields. Added persisted learner preferences,
study suggestions, actual seven-day question targets, expanded lessons and384
original math questions. All original192 question records and all research
source remain unchanged.14 backend tests and the expanded production Edge flow
passed; evidence is `reports/platform/profile_validation.json` and
[PLATFORM_STATE.md](PLATFORM_STATE.md). Example preferences were added to one
requested local account with an explicit sample label; no learning history was
fabricated. Implementation`4ed99e7` is published on canonical`main` and the
platform branch. The checkpoint/content request below is unchanged.

## To the GPU training agent

Achin has authorized a parallel platform workstream in this same repo. It adds
`backend/` and `frontend/` with FastAPI, Next.js and MongoDB. Your existing
benchmark/training work remains yours and all research source stays unchanged.
Continue your established run and scientific protocol. No GPU interruption or
new model training is requested by this handoff.

When convenient, inspect [PLATFORM_STATE.md](PLATFORM_STATE.md) and
`docs/platform/model-integration.md` in the repository. Please record your
reply in this file or a linked project-memory note on a synchronized branch.
Useful integration evidence:

1. Deployable HiTSKT checkpoint path/commit and immutable SHA256.
2. Exact question/composite-skill mapping-report identities for that checkpoint.
3. Intended content domain and availability/publication rights for original
   question text, answer choices and grading keys. New math content has no
   valid benchmark embedding assignment without separate training.
4. A small genuine mapped multi-session history and candidate, plus your CPU
   inference output, for independent serving parity and latency validation.

Do not invent mappings or treat model probabilities as calibrated mastery.
The platform already labels its Bayesian cold-start estimates separately.

## Next bounded platform action

Backend, adapter, trained-checkpoint compatibility, final production build and
desktop/mobile Edge checks on the production bundle have passed. Windows
production start/stop/restart also passed. Details are in `PLATFORM_STATE.md`
and `reports/platform/validation.json`. The next bounded action is to review the
matching-content/checkpoint provenance requested above, then integrate that
content and compare serving output against a genuine GPU-workstream fixture.
After model/content evidence is available, integrate reviewed matching content
as a platform-only change and validate trained inference parity.

## Current publication state

**Published implementation: `a2c27b8`**, atomically pushed to canonical `main`
and `codex/adaptive-learning-platform` on 2026-10-08, using Achin's configured
identity. Training/research source and scientific artifacts remain unchanged.
The production site is running on the local Windows host, with ignored runtime
logs/process identities under `.platform-runtime/`. Checkpoint compatibility and
production-browser/runtime checks passed. No new benchmark result is claimed.
