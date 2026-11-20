# Lumen usability review — 2026-11-20

The review follows the student learning cycle and the separate responsibilities
of teachers and administrators. The product retains its original mathematics
content, records and BKT/HiTSKT research boundary.

## Feature decisions

| Before | Decision | Purpose |
| --- | --- | --- |
| Seven student destinations | Four primary destinations: Today, Learn, Assessments, Progress | Make the next destination predictable |
| Practice and Revision as separate pages | Merge into Learn | Recommendation, explanation and practice belong together |
| History and repeated earlier-check tables in Progress | Put Session history inside Progress, beside Learning report | One place to inspect past work |
| Profile in navigation and account footer | Account footer owns Profile & classes | Reduce duplicate navigation |
| Full recommendations repeated on Today, Revision and Progress | Today has one next action; Learn owns the complete plan | Keep the dashboard short |
| Adaptive mock beside teacher exams | Label Mixed practice and place under optional Learn controls | Distinguish learning activity from graded fixed papers |
| Weekly targets and study-day lists as prominent planner panels | Optional Learn details and editable profile preferences | Retain useful preferences without blocking practice |
| Free-text learning goal with no effect on adaptation | Remove from the form; preserve existing stored values | Ask for preferences that actually affect the experience |
| Class charts and activity repeated on teacher Overview and Analytics | Overview owns teaching actions and activity; Analytics owns detailed topic charts and comparisons | Distinguish action from investigation |
| Question bank and Topics & skills | Keep | Reviewed, versioned content and stable concept mapping are essential |
| Interactions, validation and export | Keep | Collect inspectable historical data without direct identifiers in exports |
| Separate Model Lab | Keep | Demonstrate research inference without implying HiTSKT powers live mathematics |
| Formal diagnostic and teacher papers | Keep and correct selected-answer save behavior | Preserve comparable coverage, rules and teacher-controlled result release |

Legacy `view=revision`, `view=history` and report links remain usable. They lead
to the consolidated views. Data and API routes have not been deleted.

## Solving questions

The practice player emphasizes the question and answer choices, with larger
type, explicit selected/correct/incorrect states and a single primary action.
After an answer is confirmed, the explanation and response comparison appear
in the same card. Focus moves to the feedback heading and then to the next
question. Answer checking and question advancement are deliberate actions.

Hints remain optional. Working space is an ungraded text area; it is not sent
to the backend or used as model evidence. A current unchecked selection,
working and idempotency request ID can survive refresh in session storage for
that tab and question. These are local drafts, not server-confirmed answers.
Storage being unavailable does not prevent practice. Drafts are removed after
a confirmed answer. The offline state disables answer submission and explains
what is saved. An uncertain submission locks the original selection until it
can be retried with the same request ID; it cannot silently grade a different
choice. The working area collapses after submission to foreground feedback.

The formal-test player now saves a changed selected answer before leaving,
opening another question or reviewing submission. It stays on the assessment
when a save fails. The server still controls timing, navigation, answer keys,
grading and release. No hints or correctness were added to formal tests.

Visual review also identified a mobile sticky action footer covering part of
the explanation. The footer now follows the explanation in normal document
flow. A browser geometry assertion checks that the feedback ends before the
action footer begins.

The initial cross-role browser check caught a cold HiTSKT load exceeding Next's
default 30-second proxy timeout. The direct, warm CPU replay returned the same
verified probability. The proxy now allows a bounded 120 seconds, with a visible
preparation message. The research model itself is unchanged.

## Evidence and limits

Validation results are recorded in `reports/platform/usability_validation.json`.
Browser checks use an isolated QA database, not the user's learning records.
This is an expert review and functional verification; it is not a measured
student usability study. Local draft storage cannot replace a confirmed server
save and is unavailable across browser tabs or devices.
