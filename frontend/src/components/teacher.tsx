"use client";
import { useEffect, useState, type FormEvent } from "react";
import { ArrowRight, Plus, Search, Users } from "lucide-react";
import { send, type Question } from "@/lib/api";
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

type ClassInfo = {
  _id: string;
  name: string;
  grade_level: string;
  curriculum: string;
  student_count: number;
  join_code: string;
};
type TopicEvidence = { concept_id: string; correct: number; total: number };
type Learner = {
  id: string;
  name: string;
  membership: string;
  diagnostic: {
    score: number;
    total: number;
    topics: {
      concept_id: string;
      correct: number;
      answered: number;
      shown: number;
    }[];
  } | null;
  attempts: number;
  last_activity: string | null;
};
type Roster = {
  class: ClassInfo;
  students: Learner[];
  diagnostic_count: number;
  topics: TopicEvidence[];
  distribution: { low: number; high: number; count: number }[];
};
type LearnerDetail = Learner & {
  history: {
    id: string;
    attempts: number;
    correct: number;
    hints: number;
    mode: string;
    date: string;
  }[];
};
type PaperInput = {
  title: string;
  class_id: string;
  question_ids: string[];
  duration_minutes: number;
  opens_at: string | null;
  closes_at: string | null;
  review_allowed: boolean;
  skips_allowed: boolean;
  show_explanations: boolean;
  extra_minutes: Record<string, number>;
};
type Paper = PaperInput & { _id: string; class_name: string; status: string };
type PaperResults = {
  paper: Paper;
  topics: TopicEvidence[];
  submissions: {
    student_id: string;
    name: string;
    status: string;
    score: number | null;
    total: number;
    submitted_at: string | null;
    elapsed_seconds: number | null;
  }[];
};
type Taxon = {
  id: string;
  name: string;
  kind: "topic" | "skill";
  concept_id: string;
  topic_id: string;
};
const toDate = (x: FormDataEntryValue | null) =>
  x ? new Date(String(x)).toISOString() : null;
const fromDate = (value: string | null) =>
  value
    ? new Date(Date.parse(value) - new Date(value).getTimezoneOffset() * 60000)
        .toISOString()
        .slice(0, 16)
    : "";

function CompareLearners({
  students,
  demo,
}: {
  students: Learner[];
  demo: boolean;
}) {
  const eligible = students.filter(
    (s) => s.membership === "active" && s.diagnostic,
  );
  const [one, setOne] = useState(""),
    [two, setTwo] = useState("");
  const left =
    eligible.find((s) => s.id === one) ??
    (demo ? eligible.find((s) => s.name === "Asha Shah") : undefined) ??
    eligible[0];
  const right =
    eligible.find((s) => s.id === two) ??
    (demo
      ? eligible.find((s) => s.name === "Rohan Mehta" && s.id !== left?.id)
      : undefined) ??
    eligible.find(
      (s) =>
        s.id !== left?.id && s.diagnostic!.score === left?.diagnostic!.score,
    ) ??
    eligible[1];
  if (!left || !right)
    return (
      <Empty title="Two diagnostic results are needed">
        Compare topic evidence once two students have completed a diagnostic.
      </Empty>
    );
  const topics = Array.from(
    new Set(
      [...left.diagnostic!.topics, ...right.diagnostic!.topics].map(
        (t) => t.concept_id,
      ),
    ),
  );
  const value = (s: Learner, id: string) => {
    const t = s.diagnostic?.topics.find((t) => t.concept_id === id);
    return t ? `${t.correct} / ${t.shown}` : "—";
  };
  const focus = (s: Learner) => {
    const match = s
      .diagnostic!.topics.filter((t) => t.answered)
      .sort((a, b) => a.correct / a.answered - b.correct / b.answered)[0];
    return match
      ? `${topicName(match.concept_id)} · ${match.correct}/${match.answered} correct on answered items`
      : "Not enough evidence";
  };
  return (
    <Panel
      title={
        left.diagnostic!.score === right.diagnostic!.score
          ? "Same score, different topic evidence"
          : "Compare topic evidence"
      }
      subtitle="Class diagnostics use the same coverage. A few responses are an early signal, not a grade prediction."
    >
      <div className="p-fields">
        <Field label="First learner">
          <select value={left.id} onChange={(e) => setOne(e.target.value)}>
            {eligible.map((s) => (
              <option key={s.id} value={s.id} disabled={s.id === right.id}>
                {s.name}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Second learner">
          <select value={right.id} onChange={(e) => setTwo(e.target.value)}>
            {eligible.map((s) => (
              <option key={s.id} value={s.id} disabled={s.id === left.id}>
                {s.name}
              </option>
            ))}
          </select>
        </Field>
      </div>
      <div className="p-section-label">Diagnostic evidence</div>
      <Table
        headers={[
          "Topic",
          `${left.name} · ${left.diagnostic!.score}/${left.diagnostic!.total}`,
          `${right.name} · ${right.diagnostic!.score}/${right.diagnostic!.total}`,
        ]}
        rows={[
          ...topics.map((t) => [topicName(t), value(left, t), value(right, t)]),
          ["Consider reviewing", focus(left), focus(right)],
        ]}
      />
    </Panel>
  );
}

function Builder({
  paper,
  classes,
  initial,
  onBack,
  onDone,
}: {
  paper: Paper | null;
  classes: ClassInfo[];
  initial: string;
  onBack: () => void;
  onDone: (x: Paper) => void;
}) {
  const [step, setStep] = useState(1),
    [search, setSearch] = useState(""),
    [topic, setTopic] = useState(""),
    [confirm, setConfirm] = useState(false);
  const stored = paper
    ? {
        title: paper.title,
        class_id: paper.class_id,
        question_ids: paper.question_ids,
        duration_minutes: paper.duration_minutes,
        opens_at: paper.opens_at,
        closes_at: paper.closes_at,
        review_allowed: paper.review_allowed,
        skips_allowed: paper.skips_allowed,
        show_explanations: paper.show_explanations,
        extra_minutes: paper.extra_minutes,
      }
    : {
        title: "",
        class_id: initial || classes[0]?._id || "",
        question_ids: [],
        duration_minutes: 20,
        opens_at: null,
        closes_at: null,
        review_allowed: true,
        skips_allowed: true,
        show_explanations: true,
        extra_minutes: {},
      };
  const [id, setId] = useState(paper?._id ?? ""),
    [d, setD] = useState<PaperInput>(stored);
  const bank = useLoad<(Question & { topic_id: string })[]>(
      "/teacher/question-options",
    ),
    tax = useLoad<Taxon[]>("/content/taxonomy"),
    roster = useLoad<Roster>(
      d.class_id ? `/teacher/classes/${d.class_id}` : null,
    ),
    a = useAction();
  const change = <K extends keyof PaperInput>(k: K, v: PaperInput[K]) =>
    setD((s) => ({ ...s, [k]: v }));
  const selected =
    bank.data?.filter((q) => d.question_ids.includes(q.id)) ?? [];
  const available =
    bank.data?.filter(
      (q) =>
        (!topic || q.topic_id === topic) &&
        `${q.id} ${q.prompt}`.toLowerCase().includes(search.toLowerCase()),
    ) ?? [];
  async function persist(assign = false) {
    await a.run(
      async () => {
        const saved = await send<Paper>(
          id ? `/teacher/papers/${id}/edit` : "/teacher/papers",
          d,
        );
        setId(saved._id);
        const p = assign
          ? await send<Paper>(`/teacher/papers/${saved._id}/assign`)
          : saved;
        onDone(p);
      },
      assign ? "Assigned to class" : "Draft saved",
    );
  }
  if (!classes.length)
    return (
      <Empty
        title="Create a class before assigning a paper"
        action={<Button onClick={onBack}>Back to assessments</Button>}
      >
        An assessment needs an assigned cohort.
      </Empty>
    );
  return (
    <>
      <Heading
        title={paper ? "Edit assessment draft" : "Build an assessment"}
        subtitle="Fixed paper · same questions and grading for everyone in the class."
        action={
          <Button variant="secondary" onClick={onBack}>
            Back to assessments
          </Button>
        }
      />
      <div className="p-steps-indicator">
        {["Questions", "Rules", "Review"].map((x, i) => (
          <Badge tone={step === i + 1 ? "sage" : "neutral"} key={x}>
            {i + 1} · {x}
          </Badge>
        ))}
      </div>
      <ActionStatus a={a} />
      {step === 1 && (
        <Panel
          title="Choose coverage"
          subtitle="Select published questions. The selected versions are locked when assigned."
        >
          <form
            className="p-stack"
            onSubmit={(e) => {
              e.preventDefault();
              setStep(2);
            }}
          >
            <div className="p-fields">
              <Field label="Assessment title">
                <input
                  required
                  minLength={3}
                  maxLength={120}
                  value={d.title}
                  onChange={(e) => change("title", e.target.value)}
                  placeholder="Grade 8 unit check"
                />
              </Field>
              <Field label="Class">
                <select
                  value={d.class_id}
                  onChange={(e) =>
                    setD((x) => ({
                      ...x,
                      class_id: e.target.value,
                      extra_minutes: {},
                    }))
                  }
                >
                  {classes.map((c) => (
                    <option key={c._id} value={c._id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            <div className="p-fields">
              <Field label="Search questions">
                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search ID or question text"
                />
              </Field>
              <Field label="Filter by topic">
                <select
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                >
                  <option value="">All topics</option>
                  {tax.data
                    ?.filter((t) => t.kind === "topic")
                    .map((x) => (
                      <option key={x.id} value={x.id}>
                        {x.name}
                      </option>
                    ))}
                </select>
              </Field>
            </div>
            <Boundary value={bank}>
              <div
                className="p-question-picker"
                aria-label="Published questions"
              >
                {available.length ? (
                  available.slice(0, 100).map((q) => (
                    <label className="p-check" key={q.id}>
                      <input
                        type="checkbox"
                        checked={d.question_ids.includes(q.id)}
                        disabled={
                          d.question_ids.length >= 60 &&
                          !d.question_ids.includes(q.id)
                        }
                        onChange={(e) =>
                          change(
                            "question_ids",
                            e.target.checked
                              ? [...d.question_ids, q.id]
                              : d.question_ids.filter((i) => i !== q.id),
                          )
                        }
                      />
                      <span>
                        <strong>{q.id}</strong> · {q.prompt}{" "}
                        <small className="p-faint">
                          · {topicName(q.concept_id)}
                        </small>
                      </span>
                    </label>
                  ))
                ) : (
                  <p>No matching published questions.</p>
                )}
              </div>
              {available.length > 100 && (
                <p className="p-faint">
                  Showing the first 100 matches. Use search or a topic filter to
                  narrow the list.
                </p>
              )}
            </Boundary>
            <div className="p-actions">
              <Badge>{d.question_ids.length} selected</Badge>
              <span className="p-faint">
                {Array.from(
                  new Set(selected.map((x) => topicName(x.concept_id))),
                ).join(" · ")}
              </span>
            </div>
            <div className="p-actions">
              <Button type="submit" disabled={!d.question_ids.length}>
                Next: set rules <ArrowRight size={17} />
              </Button>
              <Button
                variant="secondary"
                onClick={() => persist()}
                disabled={a.busy || d.title.trim().length < 3}
              >
                Save draft
              </Button>
            </div>
          </form>
        </Panel>
      )}
      {step === 2 && (
        <Panel
          title="Assessment rules"
          subtitle="Hints are disabled. Grades stay hidden until you release results."
        >
          <form
            className="p-stack"
            onSubmit={(e) => {
              e.preventDefault();
              if (
                d.opens_at &&
                d.closes_at &&
                Date.parse(d.opens_at) >= Date.parse(d.closes_at)
              ) {
                a.setError("The closing time must follow the opening time.");
                return;
              }
              setStep(3);
            }}
          >
            <div className="p-fields">
              <Field label="Duration (minutes)">
                <input
                  type="number"
                  required
                  min={2}
                  max={180}
                  value={d.duration_minutes || ""}
                  onChange={(e) => change("duration_minutes", +e.target.value)}
                />
              </Field>
              <Field label="Number of attempts">
                <input readOnly value="One per learner" />
              </Field>
              <Field label="Opens · optional">
                <input
                  type="datetime-local"
                  value={fromDate(d.opens_at)}
                  onChange={(e) => change("opens_at", toDate(e.target.value))}
                />
              </Field>
              <Field label="Closes · optional">
                <input
                  type="datetime-local"
                  value={fromDate(d.closes_at)}
                  onChange={(e) => change("closes_at", toDate(e.target.value))}
                />
              </Field>
            </div>
            <label className="p-check">
              <input
                type="checkbox"
                checked={d.review_allowed}
                onChange={(e) => change("review_allowed", e.target.checked)}
              />
              Allow learners to return to earlier questions
            </label>
            <label className="p-check">
              <input
                type="checkbox"
                checked={d.skips_allowed}
                onChange={(e) => change("skips_allowed", e.target.checked)}
              />
              Allow skipped questions
            </label>
            <label className="p-check">
              <input
                type="checkbox"
                checked={d.show_explanations}
                onChange={(e) => change("show_explanations", e.target.checked)}
              />
              Show answer keys and explanations after release
            </label>
            <details>
              <summary>Extra time accommodations</summary>
              <p className="p-faint">
                Extra time cannot extend past the assignment’s closing time.
              </p>
              <div className="p-fields">
                {roster.data?.students
                  .filter((s) => s.membership === "active")
                  .map((s) => (
                    <Field label={`${s.name} · extra minutes`} key={s.id}>
                      <input
                        type="number"
                        min={0}
                        max={120}
                        value={d.extra_minutes[s.id] ?? 0}
                        onChange={(e) =>
                          change("extra_minutes", {
                            ...d.extra_minutes,
                            [s.id]: +e.target.value,
                          })
                        }
                      />
                    </Field>
                  ))}
              </div>
            </details>
            <div className="p-actions">
              <Button variant="secondary" onClick={() => setStep(1)}>
                Back to questions
              </Button>
              <Button type="submit">
                Review assignment <ArrowRight size={17} />
              </Button>
            </div>
          </form>
        </Panel>
      )}
      {step === 3 && (
        <Panel
          title={d.title}
          subtitle={`${d.question_ids.length} questions · ${d.duration_minutes} minutes · ${classes.find((c) => c._id === d.class_id)?.name}`}
        >
          <Table
            headers={["Question", "Topic", "Difficulty"]}
            rows={selected.map((q) => [
              q.prompt,
              topicName(q.concept_id),
              ["", "Foundation", "Building", "Stretch"][q.difficulty],
            ])}
          />
          <Notice title="Versions are locked after assignment" tone="ok">
            Future edits won’t change an assigned paper or earlier attempts.
          </Notice>
          <div className="p-actions">
            <Button variant="secondary" onClick={() => setStep(2)}>
              Back to rules
            </Button>
            <Button
              variant="secondary"
              onClick={() => persist()}
              disabled={a.busy}
            >
              Save draft
            </Button>
            <Button
              onClick={() => setConfirm(true)}
              disabled={a.busy || selected.length !== d.question_ids.length}
            >
              Assign to class
            </Button>
          </div>
        </Panel>
      )}
      {confirm && (
        <Confirm
          title="Assign this fixed paper?"
          action="Assign to class"
          onClose={() => setConfirm(false)}
          busy={a.busy}
          onConfirm={() => persist(true)}
        >
          <p>
            Students in {classes.find((c) => c._id === d.class_id)?.name} will
            see the paper when it opens. Content and rules are locked.
          </p>
          <ActionStatus a={a} />
        </Confirm>
      )}
    </>
  );
}

export function TeacherWorkspace({
  name,
  tab,
  navigate,
  demo,
}: {
  name: string;
  demo: boolean;
  tab: string;
  navigate: (s: string) => void;
}) {
  const classes = useLoad<ClassInfo[]>("/teacher/classes"),
    papers = useLoad<Paper[]>(tab === "assessments" ? "/teacher/papers" : null),
    a = useAction();
  const [selected, setSelected] = useState(""),
    [showClass, setShowClass] = useState(false),
    [create, setCreate] = useState(false),
    [enroll, setEnroll] = useState(false),
    [learner, setLearner] = useState(""),
    [search, setSearch] = useState(""),
    [remove, setRemove] = useState<Learner | null>(null),
    [paperId, setPaperId] = useState(""),
    [building, setBuilding] = useState<Paper | null | undefined>(undefined),
    [confirm, setConfirm] = useState<"close" | "release" | null>(null);
  const cid =
    selected ||
    classes.data?.slice().sort((x, y) => y.student_count - x.student_count)[0]
      ?._id ||
    "";
  const roster = useLoad<Roster>(cid ? `/teacher/classes/${cid}` : null);
  const last = roster.data;
  const detail = useLoad<LearnerDetail>(
    learner && cid ? `/teacher/classes/${cid}/students/${learner}` : null,
  );
  const outcomes = useLoad<PaperResults>(
    paperId && tab === "assessments"
      ? `/teacher/papers/${paperId}/results`
      : null,
  );
  useEffect(() => {
    setLearner("");
    setCreate(false);
    setEnroll(false);
    setPaperId("");
    setBuilding(undefined);
  }, [tab]);
  const viewClass = () => {
    navigate("classes");
    setShowClass(true);
  };
  async function addClass(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const body = Object.fromEntries(new FormData(e.currentTarget));
    const c = await a.run(
      () => send<ClassInfo>("/teacher/classes", body),
      "Class created",
    );
    if (c) {
      setSelected(c._id);
      setShowClass(true);
      setCreate(false);
      classes.reload();
    }
  }
  async function addStudent(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const body = Object.fromEntries(new FormData(e.currentTarget));
    const result = await a.run(
      () => send(`/teacher/classes/${cid}/enroll`, body),
      "Student enrolled",
    );
    if (result) {
      setEnroll(false);
      roster.reload();
      classes.reload();
    }
  }
  const switcher = (
    <Field label="Class">
      <select
        value={cid}
        onChange={(e) => {
          setSelected(e.target.value);
          setLearner("");
        }}
      >
        {classes.data?.map((c) => (
          <option key={c._id} value={c._id}>
            {c.name}
          </option>
        ))}
      </select>
    </Field>
  );
  const chart = (topics: TopicEvidence[]) =>
    topics.length ? (
      <EvidenceBars
        items={topics.map((x) => ({
          name: topicName(x.concept_id),
          count: x.correct,
          total: x.total,
        }))}
      />
    ) : (
      <p className="p-analytics-empty">
        Topic evidence appears when students finish the diagnostic.
      </p>
    );
  if (tab === "assessments" && building !== undefined)
    return (
      <Builder
        paper={building}
        classes={classes.data ?? []}
        initial={cid}
        onBack={() => setBuilding(undefined)}
        onDone={(p) => {
          setBuilding(undefined);
          papers.reload();
          if (p.status !== "draft") setPaperId(p._id);
        }}
      />
    );
  if (tab === "assessments" && paperId)
    return (
      <>
        <Heading
          title={outcomes.data?.paper.title ?? "Assessment results"}
          subtitle={outcomes.data?.paper.class_name}
          action={
            <Button
              variant="secondary"
              onClick={() => {
                setPaperId("");
                papers.reload();
              }}
            >
              Back to assessments
            </Button>
          }
        />
        <ActionStatus a={a} />
        <Boundary value={outcomes}>
          {outcomes.data && (
            <>
              <div className="p-stats">
                <Stat
                  title="Submitted"
                  value={`${outcomes.data.submissions.filter((s) => s.status === "submitted").length} / ${outcomes.data.submissions.length}`}
                />
                <Stat
                  title="Paper status"
                  value={label(outcomes.data.paper.status)}
                  caption="Students only see results after release"
                />
              </div>
              <Table
                headers={["Student", "Status", "Score", "Elapsed", "Submitted"]}
                rows={outcomes.data.submissions.map((x) => [
                  x.name,
                  label(x.status),
                  x.score == null ? "—" : `${x.score} / ${x.total}`,
                  duration(x.elapsed_seconds),
                  when(x.submitted_at),
                ])}
              />
              <Panel
                title="Topic results"
                subtitle="Counted across submitted common papers; skipped items receive zero marks."
              >
                {chart(outcomes.data.topics)}
              </Panel>
              <div className="p-actions">
                {outcomes.data.paper.status === "assigned" && (
                  <Button
                    onClick={() => setConfirm("close")}
                    variant="secondary"
                  >
                    Close paper
                  </Button>
                )}
                {outcomes.data.paper.status === "closed" && (
                  <Button onClick={() => setConfirm("release")}>
                    Release results
                  </Button>
                )}
              </div>
            </>
          )}
        </Boundary>
        {confirm && (
          <Confirm
            title={
              confirm === "close"
                ? "Close this assessment?"
                : "Release results to students?"
            }
            action={
              confirm === "close" ? "Close assessment" : "Release results"
            }
            onClose={() => setConfirm(null)}
            busy={a.busy}
            onConfirm={async () => {
              const result = await a.run(
                () => send(`/teacher/papers/${paperId}/${confirm}`),
                confirm === "close" ? "Assessment closed" : "Results released",
              );
              if (result) {
                setConfirm(null);
                outcomes.reload();
              }
            }}
          >
            <p>
              {confirm === "close"
                ? "Students can no longer start or modify this paper. Attempts in progress will be finalized using confirmed answers."
                : "Each learner sees only their own result and any explanations permitted by the assessment rules."}
            </p>
            <ActionStatus a={a} />
          </Confirm>
        )}
      </>
    );
  if (tab === "assessments")
    return (
      <>
        <Heading
          title="Assessments"
          subtitle="Create drafts, assign fixed papers, and review submissions."
          action={
            <Button onClick={() => setBuilding(null)}>
              <Plus size={17} /> New assessment
            </Button>
          }
        />
        <Boundary value={papers}>
          {papers.data?.length ? (
            <Table
              headers={["Assessment", "Class", "Mode", "Status", "Action"]}
              rows={papers.data.map((p) => [
                p.title,
                p.class_name,
                "Fixed paper",
                <Badge key="b" tone={p.status === "draft" ? "neutral" : "sage"}>
                  {label(p.status)}
                </Badge>,
                <Button
                  key="a"
                  variant="secondary"
                  onClick={() =>
                    p.status === "draft" ? setBuilding(p) : setPaperId(p._id)
                  }
                >
                  {p.status === "draft" ? "Edit draft" : "View submissions"}
                </Button>,
              ])}
            />
          ) : (
            <Empty
              title="No assessments yet"
              action={
                <Button onClick={() => setBuilding(null)}>Build a paper</Button>
              }
            >
              Create a fixed paper from published questions and assign it to a
              class.
            </Empty>
          )}
        </Boundary>
      </>
    );
  if (tab === "classes" && create)
    return (
      <>
        <Heading
          title="Create a class"
          subtitle="Choose a name and grade, then share the join code."
          action={
            <Button variant="secondary" onClick={() => setCreate(false)}>
              Back to classes
            </Button>
          }
        />
        <ActionStatus a={a} />
        <Panel className="p-intro">
          <form className="p-stack" onSubmit={addClass}>
            <div className="p-fields">
              <Field label="Class name">
                <input
                  name="name"
                  required
                  minLength={2}
                  maxLength={100}
                  placeholder="8C Mathematics"
                />
              </Field>
              <Field label="Grade">
                <select name="grade_level">
                  {[6, 7, 8, 9, 10, 11, 12].map((x) => (
                    <option key={x}>Grade {x}</option>
                  ))}
                </select>
              </Field>
              <Field label="Subject">
                <input readOnly value="Mathematics" />
              </Field>
              <Field label="Curriculum · optional">
                <select name="curriculum">
                  <option value="">Not specified</option>
                  <option>CBSE</option>
                  <option>ICSE</option>
                  <option>State board</option>
                  <option>Other</option>
                </select>
              </Field>
            </div>
            <div>
              <Button type="submit" disabled={a.busy}>
                Create class
              </Button>
            </div>
          </form>
        </Panel>
      </>
    );
  if (tab === "classes" && learner)
    return (
      <>
        <Heading
          title={detail.data?.name ?? "Student evidence"}
          subtitle={last?.class.name}
          action={
            <Button variant="secondary" onClick={() => setLearner("")}>
              Back to roster
            </Button>
          }
        />
        <Boundary value={detail}>
          {detail.data && (
            <>
              {detail.data.diagnostic ? (
                <div className="p-stats">
                  <Stat
                    title="Diagnostic"
                    value={`${detail.data.diagnostic.score} / ${detail.data.diagnostic.total}`}
                    caption="Common starting check"
                  />
                  <Stat
                    title="Responses"
                    value={detail.data.attempts}
                    caption="Submitted learning evidence"
                  />
                </div>
              ) : (
                <Notice title="No diagnostic results yet">
                  This learner has no submitted diagnostic to report.
                </Notice>
              )}
              {detail.data.diagnostic && (
                <Panel title="Topic evidence">
                  <EvidenceBars
                    items={detail.data.diagnostic.topics.map((t) => ({
                      name: topicName(t.concept_id),
                      count: t.correct,
                      total: t.shown,
                    }))}
                  />
                </Panel>
              )}
              <Panel
                title="Recent learning history"
                subtitle="Sessions from the current class membership period."
              >
                {detail.data.history.length ? (
                  <Table
                    headers={[
                      "Session",
                      "Attempted",
                      "Correct",
                      "Hints",
                      "Date",
                    ]}
                    rows={detail.data.history.map((s) => [
                      label(s.mode),
                      s.attempts,
                      s.correct,
                      s.hints,
                      when(s.date),
                    ])}
                  />
                ) : (
                  <p className="p-muted">
                    No sessions in this enrollment period.
                  </p>
                )}
              </Panel>
            </>
          )}
        </Boundary>
      </>
    );
  if (tab === "classes" && showClass)
    return (
      <>
        <Heading
          title={last?.class.name ?? "Class roster"}
          subtitle={`${last?.class.grade_level ?? "Mathematics"} · ${last?.class.student_count ?? 0} active students`}
          action={
            <>
              <Button variant="secondary" onClick={() => setShowClass(false)}>
                All classes
              </Button>
              <Button onClick={() => setEnroll(!enroll)}>
                <Plus size={17} /> Add students
              </Button>
            </>
          }
        />
        <ActionStatus a={a} />
        <Boundary value={roster}>
          {last && (
            <>
              {enroll && (
                <Panel
                  title="Add students"
                  subtitle="Share this code or add an existing student by email."
                >
                  <div className="p-stack">
                    <div className="p-inline">
                      <Badge>Join code: {last.class.join_code}</Badge>
                      <Button
                        variant="secondary"
                        onClick={() =>
                          a.run(
                            () =>
                              navigator.clipboard.writeText(
                                last.class.join_code,
                              ),
                            "Class code copied",
                          )
                        }
                      >
                        Copy
                      </Button>
                    </div>
                    <form onSubmit={addStudent} className="p-inline">
                      <Field label="Existing student email">
                        <input
                          type="email"
                          name="email"
                          placeholder="student@example.com"
                          required
                        />
                      </Field>
                      <Button type="submit" disabled={a.busy}>
                        Enroll student
                      </Button>
                    </form>
                  </div>
                </Panel>
              )}
              <div className="p-toolbar">
                <Field label="Search roster">
                  <input
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    placeholder="Find a learner…"
                  />
                </Field>
                <Button
                  variant="secondary"
                  onClick={() => navigate("analytics")}
                >
                  Class analytics
                </Button>
              </div>
              {last.students.length ? (
                <Table
                  headers={[
                    "Student",
                    "Diagnostic",
                    "Topic to review",
                    "Last active",
                    "Membership",
                    "Manage",
                  ]}
                  rows={last.students
                    .filter((s) =>
                      s.name.toLowerCase().includes(search.toLowerCase()),
                    )
                    .map((s) => {
                      const w = s.diagnostic?.topics
                        .filter((t) => t.answered > 0)
                        .sort(
                          (a, b) =>
                            a.correct / a.answered - b.correct / b.answered,
                        )[0];
                      return [
                        s.membership === "active" ? (
                          <button
                            className="p-btn link"
                            onClick={() => setLearner(s.id)}
                            key="name"
                          >
                            {s.name} →
                          </button>
                        ) : (
                          s.name
                        ),
                        s.diagnostic
                          ? `${s.diagnostic.score} / ${s.diagnostic.total}`
                          : "Not started",
                        w
                          ? `${topicName(w.concept_id)} · ${w.correct}/${w.answered}`
                          : "No evidence",
                        when(s.last_activity),
                        label(s.membership),
                        s.membership === "active" ? (
                          <Button
                            key="manage"
                            variant="secondary"
                            onClick={() => setRemove(s)}
                          >
                            Remove
                          </Button>
                        ) : (
                          "—"
                        ),
                      ];
                    })}
                />
              ) : (
                <Empty
                  title="No students yet"
                  action={
                    <Button onClick={() => setEnroll(true)}>
                      Add students
                    </Button>
                  }
                >
                  Share the join code or enroll a student who already has an
                  account.
                </Empty>
              )}
            </>
          )}
        </Boundary>
        {remove && (
          <Confirm
            title={`Remove ${remove.name} from this class?`}
            action="Remove student"
            danger
            busy={a.busy}
            onClose={() => setRemove(null)}
            onConfirm={async () => {
              const result = await a.run(
                () => send(`/teacher/classes/${cid}/remove/${remove.id}`),
                "Enrollment ended",
              );
              if (result) {
                setRemove(null);
                classes.reload();
                roster.reload();
              }
            }}
          >
            Their account and prior submissions are preserved. They will lose
            access to active class work.
          </Confirm>
        )}
      </>
    );
  if (tab === "classes")
    return (
      <>
        <Heading
          title="Your classes"
          subtitle="View assigned cohorts and manage enrollment."
          action={
            <Button onClick={() => setCreate(true)}>
              <Plus size={17} /> Create class
            </Button>
          }
        />
        <Boundary value={classes}>
          {classes.data?.length ? (
            <Table
              headers={["Class", "Grade", "Curriculum", "Students", "Action"]}
              rows={classes.data.map((c) => [
                c.name,
                c.grade_level,
                c.curriculum || "Not specified",
                c.student_count,
                <Button
                  key="a"
                  variant="secondary"
                  onClick={() => {
                    setSelected(c._id);
                    setShowClass(true);
                  }}
                >
                  Open class
                </Button>,
              ])}
            />
          ) : (
            <Empty
              title="No classes assigned"
              action={
                <Button onClick={() => setCreate(true)}>
                  Create your first class
                </Button>
              }
            >
              Your roster and class evidence will appear here.
            </Empty>
          )}
        </Boundary>
      </>
    );
  const analytics = tab === "analytics";
  return (
    <>
      <Heading
        title={
          analytics
            ? "Class learning analytics"
            : `Good to see you, ${name.split(" ")[0]}`
        }
        subtitle={
          analytics
            ? "See where a class may need review. Counts show the evidence behind each comparison."
            : "Class priorities and recent learning evidence."
        }
        action={classes.data?.length ? switcher : undefined}
      />
      <Boundary value={classes}>
        {!classes.data?.length ? (
          <Empty
            title="No assigned classes"
            action={
              <Button onClick={() => navigate("classes")}>Open classes</Button>
            }
          >
            Create a class and enroll students to see their learning evidence.
          </Empty>
        ) : (
          <Boundary value={roster}>
            {last && (
              <>
                {!analytics && (
                  <div className="p-stats">
                    <Stat
                      title="Assigned classes"
                      value={classes.data.length}
                    />
                    <Stat
                      title="Active students"
                      value={last.class.student_count}
                      caption={last.class.name}
                    />
                    <Stat
                      title="Diagnostic coverage"
                      value={`${last.diagnostic_count} / ${last.class.student_count}`}
                      caption="Completed by active learners"
                    />
                  </div>
                )}
                <div className="p-grid wide-left">
                  {analytics ? (
                    <Panel
                      title="Topics to revisit"
                      subtitle="Correct answers across the latest available diagnostic for each learner."
                    >
                      {chart(last.topics)}
                      <Button
                        variant="link"
                        onClick={() =>
                          analytics ? viewClass() : navigate("analytics")
                        }
                      >
                        {analytics ? "Open roster" : "Open class analytics"}{" "}
                        <ArrowRight size={17} />
                      </Button>
                    </Panel>
                  ) : (
                    <Panel
                      title="Your next teaching step"
                      subtitle="Review the evidence, then choose the right support for your class."
                    >
                      <p>
                        {last.diagnostic_count} learners have a diagnostic to
                        review. Use Analytics for topic gaps and learner
                        comparisons.
                      </p>
                      <div className="p-actions">
                        <Button onClick={() => navigate("analytics")}>
                          Review class analytics
                        </Button>
                        <Button
                          variant="secondary"
                          onClick={() => navigate("assessments")}
                        >
                          Manage assessments
                        </Button>
                      </div>
                    </Panel>
                  )}
                  {analytics ? (
                    <Panel
                      title="Score distribution"
                      subtitle="Number of learners in each range, out of 12 marks."
                    >
                      {last.diagnostic_count ? (
                        <EvidenceBars
                          items={last.distribution.map((x) => ({
                            name: `${x.low}–${x.high} marks`,
                            count: x.count,
                            total: last.diagnostic_count,
                          }))}
                        />
                      ) : (
                        <p className="p-muted">No results yet.</p>
                      )}
                    </Panel>
                  ) : (
                    <Panel
                      title="Students to check in with"
                      subtitle="Review the response evidence before deciding on support."
                    >
                      <div className="p-list">
                        {last.students
                          .filter(
                            (x) =>
                              x.membership === "active" &&
                              x.diagnostic?.topics.some(
                                (t) =>
                                  t.answered && t.correct / t.answered < 0.5,
                              ),
                          )
                          .slice(0, 5)
                          .map((s) => (
                            <div key={s.id}>
                              <strong>{s.name}</strong>
                              <Button
                                variant="link"
                                onClick={() => {
                                  viewClass();
                                  setTimeout(() => setLearner(s.id), 0);
                                }}
                              >
                                Review evidence →
                              </Button>
                            </div>
                          ))}
                      </div>
                      {!last.students.length && (
                        <p className="p-muted">Add students to see activity.</p>
                      )}
                      {last.students.length > 0 &&
                        !last.students.some(
                          (x) =>
                            x.membership === "active" &&
                            x.diagnostic?.topics.some(
                              (t) => t.answered && t.correct / t.answered < 0.5,
                            ),
                        ) && (
                          <p className="p-muted">
                            No topic has fallen below half correct in the
                            available diagnostics. Keep reviewing as more
                            evidence arrives.
                          </p>
                        )}
                    </Panel>
                  )}
                </div>
                {analytics && (
                  <CompareLearners students={last.students} demo={demo} />
                )}
                {!analytics && (
                  <Panel title="Recent activity">
                    <div className="p-list">
                      {last.students
                        .filter(
                          (s) => s.membership === "active" && s.last_activity,
                        )
                        .sort((a, b) =>
                          String(b.last_activity).localeCompare(
                            String(a.last_activity),
                          ),
                        )
                        .slice(0, 5)
                        .map((s) => (
                          <div key={s.id}>
                            <strong>{s.name}</strong>
                            <small>{when(s.last_activity)}</small>
                          </div>
                        ))}
                    </div>
                    <Button variant="secondary" onClick={viewClass}>
                      Open class
                    </Button>
                  </Panel>
                )}
              </>
            )}
          </Boundary>
        )}
      </Boundary>
    </>
  );
}
