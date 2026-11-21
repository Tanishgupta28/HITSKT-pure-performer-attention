# Lumen: adaptive learning with HiTSKT

## Current product build

Use [the Lumen product guide](product.md) for the Student and Administrator
workspaces, sign-in, review journeys and synthetic data. The platform lives
in the FastAPI backend and the Next.js frontend. MongoDB persists accounts,
sessions, responses, concept estimates and progress. The research models,
training and experiment artifacts have a separate workflow.

## Run on Windows

Prerequisites: Python 3.13+, Node 20.9+ and MongoDB running at localhost:27017.
Run these commands from the repository root in PowerShell:

```powershell
python -m venv backend/.venv
./backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
cd frontend
npm.cmd ci
npm.cmd run dev
```

In a second PowerShell terminal, from the repository root:

```powershell
cd backend
./.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:3000 and create an account. The API documentation is at
http://127.0.0.1:8000/docs. Copy the two `.env.example` files only if overriding
defaults; never commit `.env` files. Set `COOKIE_SECURE=true` when serving HTTPS.
Use the configured `FRONTEND_ORIGIN` consistently; cookies are host-specific.
The Next.js server proxies `/api` requests to FastAPI so the browser uses one origin.

After installing dependencies, `./scripts/run-platform.ps1` starts both development
services in hidden windows and records their identities/logs in `.platform-runtime`.
`./scripts/stop-platform.ps1` stops only those recorded process identities. It
does not stop MongoDB, training or unrelated processes. The helper refuses to
launch duplicate services on occupied ports.
After `npm.cmd run build` in `frontend`, use
`./scripts/run-platform.ps1 -Mode production` to start the tested standalone
production bundle with its static assets copied automatically.

Alternatively, with Docker installed, run `docker compose -f compose.platform.yml up --build`.
MongoDB data is stored in a named persistent volume. The compose stack binds
web/API ports to loopback. Stop the locally running services first if using the
same ports. Docker is an optional portable path, not required for local development.

## The working cycle

1. Register or sign in with an HttpOnly opaque session cookie. Passwords use Argon2.
2. Choose a mathematics topic to focus on or start a mixed practice set.
3. Submit each answer. The server validates and grades the assigned question.
4. Save the answer, concept update and next question in one MongoDB
   compare-and-swap. Replayed request IDs never count twice.
5. Select the next question using estimated success, weak topics, difficulty,
   and previous exposure.
6. Review the worked explanation, then explore topic estimates, response
   counts and prerequisite-aware recommendations in Learn and Progress.

The original bank contains 384 multiple-choice items covering fractions,
percentages, linear equations and geometry. Administrators can publish
reviewed questions. Focused practice uses six questions, and mixed practice
uses the selected 6, 12 or 18. Profile preferences include learning goal,
weekly question target, practice length, study days and focus concepts. Grade
and curriculum describe the learner's preferences. The weekly goal counts
submitted answers in the rolling past seven days.

For an explicitly selected local account with no saved profile,
`backend/tools/seed_sample_profile.py --user-id <uuid>` previews example
preferences. Add `--apply` to populate that account once. The script backs up
only its previous profile state under ignored `.platform-runtime/`, labels the
new preferences as sample data, preserves existing profiles and never creates
responses or learning outcomes. Saving preferences in the UI changes the label
to user-provided data. No personal account identifiers belong in Git or memory.

## What the estimates mean

HiTSKT predicts the probability of a correct answer conditioned on earlier
sessions and shifted, strictly prior responses. A probability is not a direct,
validated measurement of mastery. The UI's concept estimate uses an explicit
Bayesian knowledge tracing update with slip, guess and learning defaults.
These product parameters are not fitted scientific results.

For genuinely mapped content, a loaded HiTSKT checkpoint supplies the probability
used in next-question selection. Otherwise the Bayesian observation model supplies
it and the response explicitly names that provider. No random prediction or
untrained checkpoint is presented as a trained HiTSKT result.

See [model integration](model-integration.md) for the exact serving contract.

## Practical applications, grounded in the repository

| Application | Why it fits | Remaining domain work |
| --- | --- | --- |
| Adaptive math practice | Repeated answers, concepts and difficulty suit sequential KT | Pilot with reviewed content and measure actual learning gains |
| TOEIC revision | EdNet-KT1 checkpoint already models that question/skill domain | Licensed original question text and exact trained ID mappings |
| Prerequisite remediation | Concept estimates can guide what to revisit before harder material | Expert-reviewed prerequisite graph and thresholds |
| Longitudinal revision planning | Session history can support repeated checks over time | Validate forgetting/scheduling policy and notification preferences |

Evidence: `reports/final_results.csv` at the inspected base commit `08838ec`
reports HiTSKT test ROC-AUC 0.709624 on ASSIST2017, 0.797826 on Junyi and
0.769365 on EdNet-KT1. These are dataset-specific next-answer benchmarks;
they do not establish improvement in learning outcomes or performance on the
original platform question bank. Memory reports 14/15 completed model/dataset
pairs and a separate GPU agent training the remaining DKT/EdNet run. The
platform does not alter that experiment or claim it is finished.

## API and storage

- `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`,
  `POST /api/auth/logout`.
- `GET /api/subjects`, `GET /api/learning/{subject_id}`.
- `POST /api/profile` updates only the authenticated learner's validated preferences.
  Auth responses include the current profile; existing accounts receive defaults
  until they save preferences. Unknown fields and invalid concept IDs are rejected.
- `POST /api/practice`, `GET /api/practice/{id}`,
  `POST /api/practice/{id}/answers`.
- `GET /api/health` reports MongoDB and model-provider readiness.

Collections: `users` (unique normalized email), `sessions` (hashed token, TTL),
`rate_limits` (TTL) and `learning` (one atomic student/subject aggregate).
Only the current question is sent to the browser. Its answer and explanation
remain server-side until that question is answered. Authorization filters all
practice reads/writes through the student's own learning document.

The aggregate approach runs on standalone MongoDB without a transaction replica
set. This first release caps a subject at 5,000 recorded answers; migrate to
archived response collections and transactional aggregates before larger usage.
Password recovery requires configured SMTP. For a public deployment, use a
protected MongoDB instance, HTTPS and a production operational review.

## Verification

The refreshed visual system, locally bundled font licenses and motion behavior
are documented in [design and motion](design.md). Header controls pause animation;
system reduced-motion preferences render still artwork automatically.

```powershell
./backend/.venv/Scripts/python.exe -m pytest backend/tests -q
cd frontend
npm.cmd run typecheck
npm.cmd run build
npx.cmd playwright test
```

The API tests require a real MongoDB server. They create unique
`hitskt_platform_test_<uuid>` databases and drop only those generated databases.
Set `MONGODB_TEST_URI` to use a dedicated server. The optional model test uses
an explicitly untrained tiny checkpoint solely to verify the serving interface.
Install `backend/requirements-model.txt` to execute it. Browser tests require the
two services running and a Playwright Chromium installation.

On this Windows host, the downloaded Chromium executable could not launch;
the complete browser flow passed with installed Edge instead:
`$env:PLAYWRIGHT_CHANNEL='msedge'; npx.cmd playwright test`. Mobile checks use
a 390-pixel browser viewport, not a physical-device certification.

## Collaborating with the GPU agent

Read `project-memory/PLATFORM_STATE.md` and `project-memory/PLATFORM_HANDOFF.md`.
The GPU workstream retains its existing training memory and ownership.
Fetch and merge current remote changes before a platform push. Stage explicit
platform/documentation/memory paths, preserve all research changes, and never
force-push. Platform commits use Achin's configured Git identity; historical
Tanish identity instructions continue to describe the GPU workstream.
