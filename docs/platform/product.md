# Lumen product implementation

The 2026-11-20 [usability review](usability.md) supersedes the navigation below:
Practice and Revision are now Learn, History is inside Progress, and the account
footer opens Profile & classes. Mixed practice is in Learn; formal papers stay
in Assessments. Existing links continue to work.

The November product build implements the approved visual review artifact, with
the student, teacher and administrator workspaces in the existing FastAPI,
Next.js and MongoDB app. Begin at `http://localhost:3000/`. The local production
build serves the same routes used by the browser verification.

## Launch and sign in

On Windows, with MongoDB running and the dependencies installed from the
existing setup guide, build `frontend` with `npm.cmd run build`, then run from
the repository root:

```powershell
./scripts/run-platform.ps1 -Mode production -Demo
```

The `-Demo` flag provisions sample accounts only on a loopback frontend and
MongoDB. It preserves existing users, passwords and attempts on re-run. Four
public test accounts are `student@demo.lumen.example.com`,
`rohan@demo.lumen.example.com`, `teacher@demo.lumen.example.com` and
`admin@demo.lumen.example.com`. Each uses `Lumen12345`. In the sign-in screen,
the local sample buttons fill the credentials. Omit `-Demo` to disable them.

New students register at `/signup` with a name, email and password of at least
10 characters and choose their grade. The subject is Mathematics; curriculum
and class membership are optional. Old learner accounts remain students.
Public registration cannot create an educator or administrator. A school
operator can provision those accounts with the interactive `backend/tools/manage_accounts.py`
command and a password entered locally, never placed in a command-line argument.

The recover/reset screens, one-use expiring tokens, and session invalidation
are implemented. Email delivery requires `SMTP_HOST`, `SMTP_FROM` and the
provider’s optional authentication values in local backend `.env`. Until those
are configured, the form directs the user to the administrator. There is no
email claim for localhost without an SMTP provider.

## Review the journeys

1. **New student:** sign up → set up learning → start a 12-question diagnostic →
   save, skip or resume → submit → see marks, answer accuracy, topic evidence,
   and a recommendation → open revision → answer a practice question → see a
   hint and explanation if appropriate → continue or review past sessions.
2. **Returning student:** open Today. Resume an active session first or follow
   a recommended topic. Practice is adaptive and immediately graded; an
   adaptive mock is not used as a formal grade. Both have their own history.
3. **Teacher:** open 8B Mathematics → Analytics → Compare topic evidence. Asha
   scored 6/12 with 0/3 in fractions and 3/3 in equations. Rohan scored 9/12
   with 3/3 in fractions and 1/3 in equations. Check each learner’s recommendation
   in Learn, then open one of their reviewed practice sets under Progress →
   Session history. Asha starts with Fractions on a clean fixture and can move
   to Percentages after additional practice. Select a class and inspect a learner’s
   observations during the enrollment period. Create a class, share the join
   code, or enroll a registered student. Build a paper with published item
   versions, set duration and feedback rules, optionally grant extra time,
   assign to class, review submissions, close and release results.
4. **Administrator:** search or preview the question bank; save a draft,
   map a skill, publish a version, archive it or delete an unused draft.
   Add topics/skills, open interaction details, and export valid CSV with the
   schema. Model Lab can replay a synthetic BKT history or the checked HiTSKT
   interface fixture and compare them with the separate benchmark evidence.

### Fixed papers and versioning

Diagnostics cover three questions from each of the four maths foundations.
Class exams lock selected questions and scoring rules at assignment. A student
gets one attempt, saved responses, optional skips/backtracking and an
authoritative deadline. Expired attempts submit confirmed answers on the next
check. Before release, scores and answers do not enter the student's dashboard
or BKT history; the teacher can review results and the administrator can inspect
de-identified submitted interaction rows. One versioned learner document protects
saves and prevents duplicate submissions. Question edits never change historical
snapshots. Teachers can use published items marked as exam-only by an admin;
practice and diagnostics exclude them.

### Data and evidence

`backend/data/lumen-synthetic-v2/` holds the reproducible seed-42 demonstration
dataset (561 attempts, 54 sessions and 108 question versions), a question file
and a hashed manifest. The original v1 CSVs remain available. New demo activity
is labeled synthetic. The administrator export produces a distinct CSV from
MongoDB with new pseudonyms per export, validates question/skill identifiers,
correctness, timing and timezone, omits direct identifiers, and records an
export audit entry. Real and synthetic realms cannot be joined or exported as
one dataset through the app. Production data access, retention and any research
use require agreement from the school and appropriate consent.

BKT remains the live platform estimator. HiTSKT Model Lab is independent of
the live question bank. If the EdNet checkpoint and optional CPU dependencies
exist, Lab verifies its exact hash and computes on the synthetic mapped fixture;
otherwise it labels the saved parity output. BKT can run on the built-in fixture
or the displayed prior responses from a selected pseudonymous demo learner.
The held-out target is not returned until a prediction receipt is presented.
No verified real-student HiTSKT replay is packaged. The historical benchmark
table and independent CPU audit are distinct: AUC agrees within the recorded
tolerance, but full metric vectors and response availability remain unresolved.
No comparison in the app proves improved classroom learning.

## Scope and validation

Mathematics is the current subject. Built-in foundations are stable BKT keys;
new display topics and skills can be mapped to those keys. Browser recovery
shows confirmed saves and retries; it does not claim offline drafts reached the
server. Staff provisioning and SMTP remain operator-managed. No third-party
email was sent by the tests, which simulate the recovery transport.

Backend tests use uniquely named MongoDB databases. Browser workspace tests run
against a production server started with `-Demo` and an isolated QA database;
do not run them against personal or panel data. See `reports/platform/product_validation.json`
for the verified build, backend tests and browser journeys.
