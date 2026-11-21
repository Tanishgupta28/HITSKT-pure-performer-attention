"use client";

import { useEffect, useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowRight } from "lucide-react";
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
  Field,
  Heading,
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
  concept_id?: string;
};
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
const modeName = (mode: string) =>
  mode === "adaptive" ? "Mixed practice" : "Focused practice";
const path = (row: HistoryRow) => "/practice/" + row.id;

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
  const [profile, setProfile] = useState<LearnerProfile>(user.profile);
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
  return (
    <>
      <Heading
        title={compact ? "Set up your learning" : "Learning profile"}
        subtitle={
          compact
            ? "Choose your level and subject to get started. You can change these later."
            : "Choose how you want to practise and review."
        }
      />
      <ActionStatus a={action} />
      <Panel className="p-account-profile">
        <form className="p-stack" onSubmit={submit}>
          <div className="p-fields">
            <Field label="Grade level">
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
  const [modeFilter, setModeFilter] = useState("all"),
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
    setLesson(null);
  }, [tab]);
  const prefer = d?.recommendations[0];
  const recent = h.filter((row) => row.status === "completed");
  const onset = d?.active_assessment;
  async function start(mode: "adaptive" | "practice", concept?: string) {
    const value = await action.run(() =>
      send<Assessment>("/practice", {
        subject_id: "mathematics",
        mode,
        concept_id: concept || null,
      }),
    );
    if (value) route.push(`/practice/${value.id}`);
  }
  if (!user.onboarding_complete)
    return (
      <Profile
        user={user}
        onSaved={onSaved}
        compact
        onWelcome={() => {
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
          x.status === "completed"
            ? x.score + " / " + x.total
            : x.answered + " saved",
          duration(x.elapsed_seconds),
          when(x.date),
        ])}
        caption="Each result reflects the questions in that practice set."
      />
    ) : (
      <Empty
        title="No sessions to show"
        action={
          <Button onClick={() => navigate("practice")}>Explore practice</Button>
        }
      >
        Start a practice set to build your learning history.
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
            {["adaptive", "practice"].map((x) => (
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
                  <Button href={`/practice/${onset.id}`} variant="light">
                    Resume practice
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
                    subtitle={`${user.profile.adaptive_session_questions} adaptive questions across topics. Questions span several maths topics.`}
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
                    ? route.push(`/practice/${onset.id}`)
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
                  <Button onClick={() => navigate("practice")}>
                    Explore practice
                  </Button>
                }
              >
                Your report will be based on the answers you actually submit.
              </Empty>
            ) : (
              d && (
                <>
                  <div className="p-stats">
                    <Stat
                      title="Completed practice sets"
                      value={d.stats.completed_practice}
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
                      caption="Bayesian knowledge tracing · based on recorded answers"
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
                    caption="The estimate helps choose what to practise next."
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
                  {onset ? "CONTINUE LEARNING" : "RECOMMENDED NEXT"}
                </div>
                <h2>
                  {onset
                    ? "Pick up where you left off"
                    : "Practise " +
                      (prefer?.name.toLowerCase() ?? "mathematics")}
                </h2>
                <p>
                  {onset
                    ? onset.answered + " of " + onset.total + " answers saved."
                    : prefer?.reason}
                </p>
              </div>
              <div className="p-actions">
                <Button
                  variant="light"
                  disabled={action.busy}
                  onClick={() =>
                    onset
                      ? route.push("/practice/" + onset.id)
                      : start("practice", prefer?.concept_id)
                  }
                >
                  {onset ? "Resume practice" : "Start practice"}
                  <ArrowRight size={17} />
                </Button>
              </div>
            </section>
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
                        {recent[0].score} / {recent[0].total}
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
