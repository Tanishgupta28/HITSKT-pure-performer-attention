# Lumen product guide

Lumen is a mathematics practice app with Student and Administrator workspaces.
Students practise, get feedback, and see how their topic evidence changes.
Administrators maintain questions and inspect learning data. The Teacher
workspace, diagnostics and class assessments were retired on 2026-11-21.

## Launch and sign in

Install the dependencies in the [Windows setup guide](README.md), then build
the frontend with npm.cmd run build. From the repository root, run:

    ./scripts/run-platform.ps1 -Mode production -Demo

Open http://localhost:3000. In demo mode the sign-in screen offers three sample
accounts:

| Account | Email |
| --- | --- |
| Asha | student@demo.lumen.example.com |
| Rohan | rohan@demo.lumen.example.com |
| Administrator | admin@demo.lumen.example.com |

All three use Lumen12345. Demo mode checks for a loopback frontend and MongoDB.
Omit the Demo flag to disable demo access. New students can register with a
name, email and password of at least 10 characters. Grade level is required
during setup. Curriculum and learning preferences are optional. The interactive
backend/tools/manage_accounts.py utility provisions administrators.

Password recovery uses expiring one-use tokens. Set SMTP_HOST, SMTP_FROM and
any provider credentials in the local backend environment for email delivery.

## Student workspace

- **Today** shows the most useful next step, progress at a glance and the
  latest completed practice. New students can start practising immediately.
- **Learn** shows the recommended topic and its reason, topic lessons with
  worked examples, and six-question focused sets. Expand **Mix topics &
  plan your week** to start mixed practice or view the weekly question target.
  Mixed sets cover multiple topics and use the selected 6, 12 or 18 questions.
- **Progress** separates answer accuracy, estimated topic knowledge, completed
  practice sets and session history. A completed set shows correct answers
  over questions in the set. Overall accuracy uses all recorded, incorporated
  responses. BKT topic estimates use prior answers and should be read alongside
  the number of observations.

In practice the student can keep a working draft, ask for hints, submit an
answer and read a worked explanation before going on. Confirmed answers are
saved on the server. The player can resume after refresh or leaving and lets
students retry an uncertain save without counting it twice. Each completed
set can be reviewed from Session history.

Existing practice bookmarks under /assessment/ redirect to /practice/.
Historical records remain in storage, along with evidence already counted in
overall progress. The student session list and completed-set count cover
practice only.

## Administrator workspace

- **Question bank** creates, validates, publishes, versions and archives
  multiple-choice and numeric items. Published items need a skill and an
  explanation. Previously served questions keep their original snapshots.
- **Topics & skills** maintains the mathematics hierarchy and maps new topics
  to a supported foundation for live knowledge estimates.
- **Interactions** shows pseudonymous learning records and exports a CSV.
  The live export can include retained historical interaction records.
- **Model Lab** replays a sequence, predicts the next answer, and reveals the
  held-out outcome only after prediction. Its evaluation view separates public
  benchmarks, synthetic replay and Lumen evidence.

Today, Lumen mathematics practice uses Bayesian Knowledge Tracing. HiTSKT
runs only on separately mapped research sequences in Model Lab, where a
verified checkpoint can compute the prediction or the app can show the clearly
labelled saved result. The model comparison contains no Lumen classroom
evaluation. See [model integration](model-integration.md) for the input
mapping and checkpoint requirements.

The [v2 synthetic fixture](../../backend/data/lumen-synthetic-v2/README.md)
is a reproducible archive. Fresh demo databases seed its practice subset.
Browser verification uses an isolated database with demo accounts.
