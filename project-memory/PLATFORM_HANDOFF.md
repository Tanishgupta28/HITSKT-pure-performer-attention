# Platform / GPU handoff

Updated: 2026-10-08 (Asia/Kolkata)

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
and `reports/platform/validation.json`. Synchronize and publish the coherent
milestone, then integrate matching content only when its provenance is reviewed.
After model/content evidence is available, integrate reviewed matching content
as a platform-only change and validate trained inference parity.

## Current publication state

Implementation is validated locally on `codex/adaptive-learning-platform`;
publication is the remaining delivery step. Existing checkpoint compatibility
and complete production-browser/runtime checks passed. No new benchmark result
is claimed. Record the exact published implementation commit after push succeeds.
