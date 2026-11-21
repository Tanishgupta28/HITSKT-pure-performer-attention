# Platform / GPU handoff

Updated: 2026-11-21 (Asia/Kolkata)

## Practice-only platform handoff - 2026-11-21

Achin retired Teacher and Assessments. Student and Administrator remain
available. BKT still drives live mathematics practice. HiTSKT remains in the
separate Model Lab. Existing learner evidence and both published synthetic
datasets are retained. Fresh demo databases load practice only. The GPU
workstream, training data, experiment artifacts and model code are unaffected.
Use docs/platform/product.md for the current demo and product navigation.

## Demo fixture v2 handoff — 2026-11-20

The localhost showcase now illustrates distinct learners and genuinely varied
session histories. The v2 synthetic dataset is for the product demo and pipeline
tests. The old v1 CSVs remain frozen, while the sample accounts and report IDs
remain usable. Updated demo steps are in `docs/platform/product.md` and the
verification receipt is `reports/platform/demo_variety_validation.json`.
The demo data still uses application question IDs. The benchmark checkpoint,
training work and live BKT / separate HiTSKT boundary are unchanged.

## Platform usability handoff — 2026-11-20

Achin requested the removal of repeated features and a clearer practice player.
The frontend now consolidates student learning/revision and progress/history,
with four primary destinations and fewer repeated teacher summaries. The
practice player has local drafts/working, explicit feedback, offline recovery
and same-request retry. Fixed assessment navigation/leave/review saves the
current changed selection first. Existing records and backend contracts remain
intact. See [PLATFORM_STATE.md](PLATFORM_STATE.md), `docs/platform/usability.md`
and `reports/platform/usability_validation.json` for scope and validation.
The research replay remains separate; no scientific code or datasets changed.

## Product and research boundary — 2026-11-17

The approved Lumen platform now includes student, teacher and administrator
workspaces and an isolated Model Lab. Launch locally from the repository with
`scripts/run-platform.ps1 -Mode production -Demo`. See [PLATFORM_STATE.md](PLATFORM_STATE.md)
and `reports/platform/product_validation.json` for checked behavior. The demo
CSV is synthetic, generated from original Lumen math content; its IDs cannot
be used as EdNet embeddings. No new checkpoint training is requested.

Model Lab uses the existing trusted EdNet checkpoint for a synthetic sequence
replay, or labels the saved verified parity output if that checkpoint or CPU
runtime is absent. It reads the research agent's independent AUC audit and
keeps the unresolved full metric and response-availability evidence visible.
Live practice remains on BKT. The next model integration step remains a
licensed matched-content package and a verified causal replay fixture with the
exact question/composite-skill IDs and checkpoint hash. Do not derive learner
mastery from a HiTSKT correctness probability or reinterpret synthetic results
as student outcomes.

## Latest visual update — 2026-10-08

Achin requested larger fonts/icons, a polished interface, motion and a3D-style
logo. The platform now has local licensed variable fonts, larger controls/icons,
dimensional CSS/SVG branding and original Lottie artwork with persistent pause
and reduced-motion support. Details and final verification/publication status
are in [PLATFORM_STATE.md](PLATFORM_STATE.md), `docs/platform/design.md` and
`reports/platform/design_validation.json`. Backend/student/research data and the
existing checkpoint/content request remain unchanged. GPU memory update`81ee41a`
was merged before the work.
Design implementation`0aa74cf` is published on canonical`main` and the platform
branch. The final production build and focused design checks passed; the complete
production learning flow also passed before the final text-spacing polish.

## Profile update

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
