"use client";

import { useEffect, useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  BookOpen,
  ClipboardList,
  Clock3,
  Target,
} from "lucide-react";
import {
  Assessment,
  Dashboard,
  LearnerProfile,
  User,
  percent,
  send,
} from "@/lib/api";
import {
  ActionStatus,
  Badge,
  Boundary,
  Button,
  Confirm,
  Empty,
  EvidenceBars,
  Field,
  Heading,
  Notice,
  Panel,
  Stat,
  Table,
  duration,
  label,
  topicName,
  useAction,
  useLoad,
  when,
} from "./product";

export type HistoryRow = {
  id: string;
  mode: string;
  status: string;
  title: string;
  answered: number;
  score: number | null;
  total: number;
  date: string;
  elapsed_seconds: number | null;
  fixed: boolean;
  concept_id?: string;
};
type Paper = {
  _id: string;
  title: string;
  class_name: string;
  status: string;
  duration_minutes: number;
  extra_minutes: number;
  question_count: number;
  opens_at: string | null;
  closes_at: string | null;
  attempt_id: string | null;
  review_allowed: boolean;
  skips_allowed: boolean;
};
type Membership = { _id: string; name: string; grade_level: string };
const grades = [
  "Grade 6",
  "Grade 7",
  "Grade 8",
  "Grade 9",
  "Grade 10",
  "Grade 11",
  "Grade 12",
  "Independent learner",
];
const boards = ["Not set", "CBSE", "ICSE", "State board", "Other"];
const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const basic = ["fractions", "percentages", "equations", "geometry"];
const modeName = (m: string) =>
  m === "adaptive" ? "Mixed practice" : m === "exam" ? "Fixed paper" : label(m);
const path = (row: HistoryRow) =>
  `/${row.fixed ? "paper" : "assessment"}/${row.id}`;

function Profile({
  user,
  onSaved,
  onWelcome,
  compact = false,
}: {
  user: User;
  onSaved: (u: User) => void;
  onWelcome?: () => void;
  compact?: boolean;
}) {
  const action = useAction();
  const classes = useLoad<Membership[]>("/student/classes");
  const [confirm, setConfirm] = useState<Membership | null>(null);
  const [profile, setProfile] = useState<LearnerProfile>(user.profile);
  const [code, setCode] = useState("");
  const [saved, setSaved] = useState(false);
  const change = <K extends keyof LearnerProfile>(k: K, v: LearnerProfile[K]) =>
    setProfile((p) => ({ ...p, [k]: v }));
  async function submit(e: FormEvent) {
    e.preventDefault();
    const { source, ...body } = profile;
    const u = await action.run(
      () => send<User>("/profile", body),
      "Profile saved",
    );
    if (u) {
      setSaved(true);
      onSaved(u);
      if (compact && onWelcome) onWelcome();
    }
  }
  async function join(e: FormEvent) {
    e.preventDefault();
    const data = await action.run(
      () => send("/student/classes/join", { code: code.trim() }),
      "Joined your class",
    );
    if (data) {
      setCode("");
      classes.reload();
    }
  }
  return (
    <>
      <Heading
        title={compact ? "Set up your learning" : "Learning profile"}
        subtitle={
          compact
            ? "Choose your level and subject to get started. You can change these later."
            : "Preferences guide your revision plan. Your assessment history stays separate."
        }
      />
      <ActionStatus a={action} />
      <Panel className="p-account-profile">
        <form className="p-stack" onSubmit={submit}>
          <div className="p-fields">
            <Field label="Grade / class">
              <select
                required
                value={
                  profile.grade_level === "Not set" ? "" : profile.grade_level
                }
                onChange={(e) => change("grade_level", e.target.value)}
              >
                <option value="">Choose a grade</option>
                {grades.map((x) => (
                  <option key={x}>{x}</option>
                ))}
              </select>
            </Field>
            <Field label="Subject">
              <input readOnly value="Mathematics" />
            </Field>
            <Field label="Curriculum · optional">
              <select
                value={profile.curriculum}
                onChange={(e) => change("curriculum", e.target.value)}
              >
                {boards.map((x) => (
                  <option key={x}>{x}</option>
                ))}
              </select>
            </Field>
          </div>
          {!compact && (
            <details className="p-extra-learning">
              <summary>Learning preferences · optional</summary>
              <div className="p-fields">
                <Field label="Weekly question target">
                  <input
                    type="number"
                    min={5}
                    max={200}
                    value={profile.weekly_question_target || ""}
                    onChange={(e) =>
                      change("weekly_question_target", +e.target.value)
                    }
                    required
                  />
                </Field>
                <Field label="Mixed practice length">
                  <select
                    value={profile.adaptive_session_questions}
                    onChange={(e) =>
                      change("adaptive_session_questions", +e.target.value)
                    }
                  >
                    {[6, 12, 18].map((x) => (
                      <option key={x} value={x}>
                        {x} questions
                      </option>
                    ))}
                  </select>
                </Field>
              </div>
              <fieldset className="p-field-group">
                <legend>Study days</legend>
                <div className="p-inline">
                  {days.map((x) => (
                    <label className="p-check" key={x}>
                      <input
                        type="checkbox"
                        checked={profile.study_days.includes(x)}
                        onChange={(e) =>
                          change(
                            "study_days",
                            e.target.checked
                              ? days.filter(
                                  (d) =>
                                    d === x || profile.study_days.includes(d),
                                )
                              : profile.study_days.filter((d) => d !== x),
                          )
                        }
                      />
                      {x}
                    </label>
                  ))}
                </div>
              </fieldset>
              <fieldset className="p-field-group">
                <legend>Focus topics</legend>
                <div className="p-inline">
                  {basic.map((x) => (
                    <label className="p-check" key={x}>
                      <input
                        type="checkbox"
                        checked={profile.focus_concepts.includes(x)}
                        onChange={(e) =>
                          change(
                            "focus_concepts",
                            e.target.checked
                              ? [...profile.focus_concepts, x]
                              : profile.focus_concepts.filter((y) => y !== x),
                          )
                        }
                      />
                      {topicName(x)}
                    </label>
                  ))}
                </div>
              </fieldset>
            </details>
          )}
          <div className="p-actions">
            <Button disabled={action.busy} type="submit">
              {compact ? "Continue" : "Save preferences"}
              <ArrowRight size={17} />
            </Button>
            {saved && (
              <span className="p-faint">Your work has been saved.</span>
            )}
          </div>
        </form>
      </Panel>
      {!compact && (
        <Panel
          className="p-account-profile"
          title="Your classes"
          subtitle="Join with a code from your teacher. You can also learn independently."
        >
          <Boundary value={classes}>
            {!classes.data?.length ? (
              <p className="p-muted">No assigned classes yet.</p>
            ) : (
              <div className="p-list">
                {classes.data.map((c) => (
                  <div key={c._id}>
                    <span>
                      <strong>{c.name}</strong>
                      <small>{c.grade_level}</small>
                    </span>
                    <Button variant="secondary" onClick={() => setConfirm(c)}>
                      Leave class
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </Boundary>
          <form className="p-inline" onSubmit={join}>
            <Field label="Class code">
              <input
                minLength={8}
                maxLength={8}
                value={code}
                onChange={(e) => setCode(e.target.value.toUpperCase())}
                placeholder="8-character code"
                required
              />
            </Field>
            <Button type="submit" disabled={action.busy || code.length !== 8}>
              Join class
            </Button>
          </form>
        </Panel>
      )}
      {confirm && (
        <Confirm
          title={`Leave ${confirm.name}?`}
          action="Leave class"
          danger
          onClose={() => setConfirm(null)}
          busy={action.busy}
          onConfirm={async () => {
            const result = await action.run(
              () => send(`/student/classes/${confirm._id}/leave`),
              "Class left",
            );
            if (result) {
              setConfirm(null);
              classes.reload();
            }
          }}
        >
          You’ll keep submitted work. Active class assignments may become
          unavailable.
        </Confirm>
      )}
    </>
  );
}

export function StudentWorkspace({
  user,
  tab,
  navigate,
  onSaved,
}: {
  user: User;
  tab: string;
  navigate: (s: string) => void;
  onSaved: (u: User) => void;
}) {
  const route = useRouter();
  const dashboard = useLoad<Dashboard>("/learning/mathematics");
  const history = useLoad<HistoryRow[]>("/student/history");
  const papers = useLoad<Paper[]>(
    tab === "assessments" ? "/student/papers" : null,
  );
  const [welcome, setWelcome] = useState(false),
    [intro, setIntro] = useState<"diagnostic" | Paper | null>(null),
    [modeFilter, setModeFilter] = useState("all"),
    [progressView, setProgressView] = useState("report"),
    [lesson, setLesson] = useState<string | null>(null);
  const action = useAction();
  useEffect(() => {
    const read = () =>
      setProgressView(
        new URLSearchParams(location.search).get("view") === "history" ||
          new URLSearchParams(location.search).get("section") === "history"
          ? "history"
          : "report",
      );
    read();
    addEventListener("popstate", read);
    return () => removeEventListener("popstate", read);
  }, [tab]);
  function showProgress(section: string) {
    setProgressView(section);
    window.history.replaceState(
      {},
      "",
      `/?view=progress${section === "history" ? "&section=history" : ""}`,
    );
  }
  const d = dashboard.data;
  const h = history.data ?? [];
  useEffect(() => {
    setIntro(null);
    setLesson(null);
  }, [tab]);
  const prefer = d?.recommendations[0];
  const recent = h.filter(
    (x) => x.status === "completed" || x.status === "submitted",
  );
  const lastDiag = recent.find((x) => x.mode === "diagnostic");
  const onset = d?.active_assessment;
  const activeDiag = d?.active_diagnostic;
  function diagnostic() {
    if (activeDiag) route.push(`/paper/${activeDiag.id}`);
    else setIntro("diagnostic");
  }
  async function start(mode: "adaptive" | "practice", concept?: string) {
    const value = await action.run(() =>
      send<Assessment>("/assessments", {
        subject_id: "mathematics",
        mode,
        concept_id: concept || null,
      }),
    );
    if (value) route.push(`/assessment/${value.id}`);
  }
  async function startPaper() {
    if (!intro) return;
    const result = await action.run(() =>
      send<{ id: string }>(
        intro === "diagnostic"
          ? "/student/fixed/diagnostic"
          : `/student/papers/${intro._id}/start`,
      ),
    );
    if (result) route.push(`/paper/${result.id}`);
  }
  if (!user.onboarding_complete)
    return (
      <Profile
        user={user}
        onSaved={onSaved}
        compact
        onWelcome={() => {
          setWelcome(true);
          dashboard.reload();
          navigate("today");
        }}
      />
    );
  if (tab === "profile")
    return (
      <Profile
        user={user}
        onSaved={(u) => {
          onSaved(u);
          dashboard.reload();
        }}
      />
    );
  if (intro) {
    const isDiag = intro === "diagnostic";
    return (
      <>
        <Heading
          title={isDiag ? "Mathematics diagnostic" : intro.title}
          subtitle={
            isDiag
              ? "Establish a starting point for your revision plan."
              : `Assigned to ${intro.class_name}`
          }
        />
        <ActionStatus a={action} />
        <div className="p-grid wide-left">
          <Panel
            className="p-intro"
            title="Before you start"
            subtitle={
              isDiag
                ? "12 questions · About 12 minutes · Untimed"
                : `${intro.question_count} questions · ${intro.duration_minutes + intro.extra_minutes} minutes · One attempt`
            }
          >
            <ol>
              <li>Choose or enter one answer per question.</li>
              <li>
                {isDiag || intro.skips_allowed
                  ? "You may skip questions and save and leave."
                  : "Save an answer before moving to the next question."}
              </li>
              <li>
                {isDiag || intro.review_allowed
                  ? "You can review saved answers before submitting."
                  : "This paper does not allow going back."}
              </li>
              <li>
                {isDiag
                  ? "Explanations appear after submission."
                  : "Hints are disabled and your teacher controls result release."}
              </li>
            </ol>
            {!isDiag && (
              <Notice title="The timer keeps running if you leave">
                {intro.closes_at
                  ? `Closes ${when(intro.closes_at)}.`
                  : "The timer begins when you start."}
              </Notice>
            )}
            <div className="p-actions">
              <Button onClick={startPaper} disabled={action.busy}>
                {isDiag ? "Begin diagnostic" : "Begin timed assessment"}
                <ArrowRight size={17} />
              </Button>
              <Button variant="secondary" onClick={() => setIntro(null)}>
                Back
              </Button>
            </div>
          </Panel>
          {isDiag && (
            <Panel title="Topic coverage" subtitle="Three questions per topic">
              <EvidenceBars
                items={basic.map((x) => ({
                  name: topicName(x),
                  count: 3,
                  total: 3,
                }))}
              />
            </Panel>
          )}
        </div>
      </>
    );
  }
  const historyTable = (list: HistoryRow[]) =>
    list.length ? (
      <Table
        headers={["Session", "Topic", "Status", "Result", "Time", "Date"]}
        rows={list.map((x) => [
          <Link className="p-text-link" href={path(x)} key="r">
            {x.title} →
          </Link>,
          topicName(x.concept_id),
          label(x.status),
          x.score == null
            ? x.status === "submitted"
              ? "Awaiting release"
              : `${x.answered} saved`
            : `${x.score} / ${x.total}`,
          duration(x.elapsed_seconds),
          when(x.date),
        ])}
        caption="Practice results describe that set; compare formal scores on similar papers."
      />
    ) : (
      <Empty
        title="No sessions to show"
        action={<Button onClick={diagnostic}>Start diagnostic</Button>}
      >
        Finish a diagnostic or practice set to build your learning history.
      </Empty>
    );
  const activityHistory = (
    <>
      <div className="p-toolbar">
        <Field label="Session type">
          <select
            value={modeFilter}
            onChange={(e) => setModeFilter(e.target.value)}
          >
            <option value="all">All sessions</option>
            {["diagnostic", "exam", "adaptive", "practice"].map((x) => (
              <option key={x} value={x}>
                {modeName(x)}
              </option>
            ))}
          </select>
        </Field>
      </div>
      <Boundary value={history}>
        {historyTable(
          h.filter((x) => modeFilter === "all" || x.mode === modeFilter),
        )}
      </Boundary>
    </>
  );
  if (tab === "assessments")
    return (
      <>
        <Heading
          title="Assessments"
          subtitle="Find your starting point or complete a paper assigned by your teacher."
        />
        <ActionStatus a={action} />
        <div className="p-grid">
          <Panel
            title="Mathematics diagnostic"
            subtitle="Fixed coverage across four topics. Feedback after submission."
          >
            <Button onClick={diagnostic}>
              {activeDiag
                ? "Resume diagnostic"
                : lastDiag
                  ? "Take a fresh diagnostic"
                  : "View instructions"}
              <ArrowRight size={17} />
            </Button>
          </Panel>
        </div>
        <h2>Class assignments</h2>
        <Boundary value={papers}>
          {papers.data?.length ? (
            <Table
              headers={[
                "Assessment",
                "Class",
                "Duration",
                "Availability",
                "Next action",
              ]}
              rows={papers.data.map((x) => [
                x.title,
                x.class_name,
                `${x.duration_minutes + x.extra_minutes} min`,
                x.status === "released"
                  ? "Results released"
                  : x.closes_at && Date.parse(x.closes_at) < Date.now()
                    ? "Closed"
                    : x.opens_at && Date.parse(x.opens_at) > Date.now()
                      ? `Opens ${when(x.opens_at)}`
                      : label(x.status),
                x.attempt_id ? (
                  <Button
                    key="t"
                    href={`/paper/${x.attempt_id}`}
                    variant="secondary"
                  >
                    Open attempt
                  </Button>
                ) : (
                  <Button
                    key="t"
                    onClick={() => setIntro(x)}
                    variant="secondary"
                    disabled={
                      x.status !== "assigned" ||
                      Boolean(
                        x.closes_at && Date.parse(x.closes_at) < Date.now(),
                      ) ||
                      Boolean(x.opens_at && Date.parse(x.opens_at) > Date.now())
                    }
                  >
                    View instructions
                  </Button>
                ),
              ])}
            />
          ) : (
            <Empty
              title="No assigned papers"
              action={
                <Button variant="secondary" onClick={() => navigate("profile")}>
                  Join a class
                </Button>
              }
            >
              Practice independently, or use a code from your teacher.
            </Empty>
          )}
        </Boundary>
      </>
    );
  if (tab === "practice" || tab === "revision")
    return (
      <>
        <Heading
          title="Your learning plan"
          subtitle="Review a concept, practise it, and check what changes."
        />
        <ActionStatus a={action} />
        <Boundary value={dashboard}>
          {d && (
            <>
              {onset ? (
                <section className="p-hero-action">
                  <div>
                    <div className="p-eyebrow">SESSION IN PROGRESS</div>
                    <h2>Continue your practice</h2>
                    <p>
                      {onset.answered} of {onset.total} answers saved. Finish
                      this set before starting another.
                    </p>
                  </div>
                  <Button href={`/assessment/${onset.id}`} variant="light">
                    Resume practice
                  </Button>
                </section>
              ) : !d.diagnostic_complete && !lastDiag ? (
                <section className="p-hero-action">
                  <div>
                    <div className="p-eyebrow">START WITH EVIDENCE</div>
                    <h2>Find your starting point</h2>
                    <p>
                      A short diagnostic helps us recommend a useful topic. You
                      can also explore below.
                    </p>
                  </div>
                  <Button variant="light" onClick={diagnostic}>
                    {activeDiag ? "Resume diagnostic" : "Start diagnostic"}
                  </Button>
                </section>
              ) : (
                <section className="p-hero-action">
                  <div>
                    <div className="p-eyebrow">
                      RECOMMENDED NEXT · SIX QUESTIONS
                    </div>
                    <h2>{prefer?.name ?? "Mathematics"}</h2>
                    <p>{prefer?.reason}</p>
                  </div>
                  <Button
                    variant="light"
                    onClick={() => start("practice", prefer?.concept_id)}
                    disabled={action.busy}
                  >
                    Start recommended practice
                  </Button>
                </section>
              )}
              <div className="p-learning-grid" aria-label="Topics to learn">
                {d.concepts.map((c) => (
                  <Panel key={c.id} title={c.name} subtitle={c.description}>
                    <p className="p-muted">
                      {c.attempts
                        ? `${c.correct} correct out of ${c.attempts} answers`
                        : "Not practised yet"}
                    </p>
                    <div className="p-actions">
                      <Button
                        variant="secondary"
                        onClick={() => setLesson(c.id)}
                      >
                        Review lesson
                      </Button>
                      <Button
                        onClick={() => start("practice", c.id)}
                        disabled={action.busy || Boolean(onset)}
                      >
                        Practise {c.name.toLowerCase()}
                      </Button>
                    </div>
                  </Panel>
                ))}
              </div>
              <details className="p-extra-learning">
                <summary>Mix topics & plan your week</summary>
                <div className="p-grid">
                  <Panel
                    title="Mixed practice"
                    subtitle={`${user.profile.adaptive_session_questions} adaptive questions across topics. This is practice, not a graded exam.`}
                  >
                    <Button
                      onClick={() => start("adaptive")}
                      disabled={action.busy || Boolean(onset)}
                    >
                      Start mixed practice
                    </Button>
                  </Panel>
                  <Panel
                    title="Weekly target"
                    subtitle={`${d.weekly_goal.answered} / ${d.weekly_goal.target} answers in the past seven days`}
                  >
                    <p>
                      {d.weekly_goal.remaining
                        ? `${d.weekly_goal.remaining} more to reach your goal.`
                        : "Your target is complete."}
                    </p>
                    <Button variant="link" onClick={() => navigate("profile")}>
                      Change learning preferences
                    </Button>
                  </Panel>
                </div>
                {d.study_plan.length > 0 && (
                  <p className="p-muted">
                    Suggested study days:{" "}
                    {d.study_plan
                      .map((x) => `${x.day} · ${x.concept_name}`)
                      .join("; ")}
                    . Start whenever you’re ready.
                  </p>
                )}
              </details>
            </>
          )}
        </Boundary>
        {lesson &&
          d &&
          (() => {
            const c = d.concepts.find((c) => c.id === lesson);
            if (!c) return null;
            return (
              <Confirm
                title={c.name}
                action={onset ? "Resume practice" : "Practise this topic"}
                onClose={() => setLesson(null)}
                busy={action.busy}
                onConfirm={() =>
                  onset
                    ? route.push(`/assessment/${onset.id}`)
                    : start("practice", c.id)
                }
              >
                <p>{c.lesson}</p>
                {c.steps && (
                  <ol>
                    {c.steps.map((x) => (
                      <li key={x}>{x}</li>
                    ))}
                  </ol>
                )}
                {c.worked_example && (
                  <div className="p-example">
                    <strong>Worked example</strong>
                    <p>{c.worked_example}</p>
                  </div>
                )}
                {c.common_mistake && <p>{c.common_mistake}</p>}
              </Confirm>
            );
          })()}
      </>
    );
  if (tab === "progress")
    return (
      <>
        <Heading
          title="Your progress"
          subtitle="See your topic evidence or review a previous session."
        />
        <div className="p-view-switch" role="group" aria-label="Progress view">
          <button
            aria-pressed={progressView === "report"}
            onClick={() => showProgress("report")}
          >
            Learning report
          </button>
          <button
            aria-pressed={progressView === "history"}
            onClick={() => showProgress("history")}
          >
            Session history
          </button>
        </div>
        {progressView === "history" ? (
          activityHistory
        ) : (
          <Boundary value={dashboard}>
            {d && !d.stats.answers ? (
              <Empty
                title="No learning evidence yet"
                action={
                  <Button onClick={diagnostic}>Take the diagnostic</Button>
                }
              >
                Your report will be based on the answers you actually submit.
              </Empty>
            ) : (
              d && (
                <>
                  <div className="p-stats">
                    <Stat
                      title="Latest diagnostic score"
                      value={
                        lastDiag?.score == null
                          ? "—"
                          : `${lastDiag.score} / ${lastDiag.total}`
                      }
                      caption="Marks on one fixed blueprint"
                    />
                    <Stat
                      title="Answer accuracy"
                      value={
                        d.stats.accuracy == null
                          ? "—"
                          : percent(d.stats.accuracy)
                      }
                      caption={`${d.stats.answers} submitted responses`}
                    />
                    <Stat
                      title="Estimated mastery"
                      value={
                        d.concepts.filter((c) => c.attempts >= 5).length
                          ? "Building evidence"
                          : "Early evidence"
                      }
                      caption="BKT learning estimate · not a grade"
                    />
                  </div>
                  <Table
                    headers={[
                      "Topic",
                      "Correct / answered",
                      "Estimated understanding",
                      "Evidence",
                    ]}
                    rows={d.concepts.map((c) => [
                      c.name,
                      `${c.correct} / ${c.attempts}`,
                      c.attempts >= 5
                        ? percent(c.mastery)
                        : "Not enough evidence",
                      c.attempts
                        ? `${c.evidence_count} observed responses`
                        : "Unexplored",
                    ])}
                    caption="Estimates guide practice; they are not exam scores."
                  />
                  <p className="p-muted">
                    Need the question-level detail? Open Session history and
                    choose an attempt.
                  </p>
                </>
              )
            )}
          </Boundary>
        )}
      </>
    );
  return (
    <>
      <Heading
        title={`Good to see you, ${user.name.split(" ")[0]}`}
        subtitle={`Mathematics · ${user.profile.grade_level} · Your next useful step comes first.`}
      />
      <ActionStatus a={action} />
      <Boundary value={dashboard}>
        {d && (
          <>
            <section className="p-hero-action">
              <div>
                <div className="p-eyebrow">
                  {onset || activeDiag
                    ? "CONTINUE LEARNING"
                    : !lastDiag && !d.diagnostic_complete
                      ? "YOUR FIRST STEP"
                      : "RECOMMENDED NEXT"}
                </div>
                <h2>
                  {activeDiag
                    ? "Resume your diagnostic"
                    : onset
                      ? "Pick up where you left off"
                      : !lastDiag && !d.diagnostic_complete
                        ? "Find your starting point"
                        : `Practise ${prefer?.name.toLowerCase() ?? "mathematics"}`}
                </h2>
                <p>
                  {activeDiag
                    ? "Your saved answers are ready."
                    : onset
                      ? `${onset.answered} of ${onset.total} answers saved.`
                      : !lastDiag && !d.diagnostic_complete
                        ? "A short diagnostic will help you see what to work on first."
                        : prefer?.reason}
                </p>
              </div>
              <div className="p-actions">
                <Button
                  variant="light"
                  disabled={action.busy}
                  onClick={() =>
                    activeDiag
                      ? diagnostic()
                      : onset
                        ? route.push(`/assessment/${onset.id}`)
                        : !lastDiag && !d.diagnostic_complete
                          ? diagnostic()
                          : start("practice", prefer?.concept_id)
                  }
                >
                  {activeDiag || onset
                    ? "Resume"
                    : !lastDiag && !d.diagnostic_complete
                      ? "Start diagnostic"
                      : "Start practice"}
                  <ArrowRight size={17} />
                </Button>
              </div>
            </section>
            {welcome && (
              <Notice title="Your learning space is ready" tone="ok">
                Start with the diagnostic when you’re ready.
              </Notice>
            )}
            <div className="p-grid">
              <Panel
                title="Your week"
                subtitle="Confirmed answers in the past seven days"
              >
                <div className="p-week-count">
                  {d.weekly_goal.answered}
                  <span> / {d.weekly_goal.target}</span>
                </div>
                <p className="p-muted">
                  {d.weekly_goal.remaining
                    ? `${d.weekly_goal.remaining} answers to reach your target.`
                    : "Target complete. Keep a pace that works for you."}
                </p>
              </Panel>
              <Panel
                title="Latest result"
                subtitle="Open the full attempt to review the evidence."
              >
                {recent[0] ? (
                  <div className="p-list">
                    <Link href={path(recent[0])}>
                      <span>
                        <strong>{recent[0].title}</strong>
                        <small>{when(recent[0].date)}</small>
                      </span>
                      <Badge>
                        {recent[0].score == null
                          ? "Awaiting release"
                          : `${recent[0].score} / ${recent[0].total}`}
                      </Badge>
                    </Link>
                  </div>
                ) : (
                  <p className="p-muted">
                    Your first completed session will appear here.
                  </p>
                )}
                <Button variant="link" onClick={() => navigate("progress")}>
                  View progress
                </Button>
              </Panel>
            </div>
          </>
        )}
      </Boundary>
    </>
  );
}
