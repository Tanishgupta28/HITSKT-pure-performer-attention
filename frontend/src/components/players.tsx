"use client";
import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Check,
  Clock3,
  Lightbulb,
  PencilLine,
  X,
} from "lucide-react";
import {
  Assessment,
  Feedback,
  Question,
  RequestError,
  percent,
  send,
} from "@/lib/api";
import {
  ActionStatus,
  Badge,
  Boundary,
  Brand,
  Button,
  Confirm,
  EvidenceBars,
  Field,
  Heading,
  Notice,
  Panel,
  Stat,
  label,
  topicName,
  useAction,
  useLoad,
} from "./product";

type Choice = number | string | null;
type Detail = Question & {
  status: string;
  response: Choice;
  answer_key?: Choice;
  explanation?: string;
};
type Result = {
  score: number;
  total: number;
  answered: number;
  accuracy: number | null;
  topics: { concept_id: string; correct: number; shown: number }[];
  questions: Detail[];
};
type Fixed = {
  id: string;
  title: string;
  mode: string;
  status: string;
  index: number;
  total: number;
  positions: string[];
  question: Question | null;
  saved_choice: Choice;
  deadline_at: string | null;
  server_time: string;
  submission_reason?: string;
  visited_until: number;
  rules: { review_allowed: boolean; skips_allowed: boolean };
  released: boolean;
  result: Result | null;
  summary?: { recommendations: { name: string; reason: string }[] };
};
type Review = {
  question: Question;
  choice: Choice;
  correct: boolean;
  feedback: Feedback;
  hints_used: number;
};
function answerText(q: Question, v: Choice | undefined) {
  return v == null
    ? "No answer"
    : typeof v === "number"
      ? (q.choices?.[v] ?? "Selected answer")
      : v;
}
function Options({
  q,
  choice,
  onChoice,
  locked = false,
  feedback,
}: {
  q: Question;
  choice: Choice;
  onChoice: (v: Choice) => void;
  locked?: boolean;
  feedback?: Feedback | null;
}) {
  if (q.question_type === "numeric")
    return (
      <Field
        label="Your answer"
        hint="Enter a number or fraction, such as 5/6."
      >
        <input
          value={choice ?? ""}
          disabled={locked}
          maxLength={100}
          onChange={(e) => onChoice(e.target.value)}
          placeholder="Type your answer"
        />
      </Field>
    );
  return (
    <fieldset className="p-field-group">
      <legend className="p-faint">Choose one answer</legend>
      <div className="p-options" role="group" aria-label="Answer choices">
        {q.choices.map((x, i) => (
          <label
            key={i}
            className={`p-option ${feedback?.answer_index === i ? "right" : feedback && choice === i ? "wrong" : ""}`}
          >
            <input
              type="radio"
              name={`q-${q.id}`}
              checked={choice === i}
              disabled={locked}
              onChange={() => onChoice(i)}
            />
            <span className="p-option-letter" aria-hidden>
              {String.fromCharCode(65 + i)}
            </span>
            <span className="p-option-text">{x}</span>
            {feedback?.answer_index === i ? (
              <small>
                <Check size={15} />{" "}
                {choice === i ? "Your answer · correct" : "Correct answer"}
              </small>
            ) : feedback && choice === i ? (
              <small>
                <X size={15} /> Your answer
              </small>
            ) : null}
          </label>
        ))}
      </div>
    </fieldset>
  );
}
function Top({ back, onLeave }: { back: string; onLeave?: () => void }) {
  return (
    <header className="p-player-top">
      <Brand />
      {onLeave ? (
        <Button variant="secondary" onClick={onLeave}>
          Save & leave
        </Button>
      ) : (
        <Button variant="secondary" href={back}>
          Back to workspace
        </Button>
      )}
    </header>
  );
}
function Report({ data, reload }: { data: Fixed; reload: () => void }) {
  const r = data.result;
  return (
    <>
      <Heading
        title={
          data.released ? `${data.title} · report` : "Assessment submitted"
        }
        subtitle={
          data.released
            ? "Your answers were saved. Use this report to decide what to practise next."
            : data.title
        }
      />
      {data.submission_reason === "deadline" && (
        <Notice title="This assessment closed">
          Only answers confirmed before the deadline were included.
        </Notice>
      )}
      {!r ? (
        <Notice title="Waiting for your teacher to release results" tone="ok">
          Your answers are saved. Marks and permitted feedback will appear here
          when released.{" "}
          <Button variant="secondary" onClick={reload}>
            Check status
          </Button>
        </Notice>
      ) : (
        <>
          <div className="p-stats p-practice-stats">
            <Stat
              title="Test score"
              value={`${r.score} / ${r.total}`}
              caption="Unanswered items count as zero marks"
            />
            <Stat
              title="Question accuracy"
              value={r.accuracy == null ? "—" : percent(r.accuracy)}
              caption={`${r.score} correct of ${r.answered} answered`}
            />
          </div>
          <Panel
            title="Topic evidence"
            subtitle="Correct answers per question shown"
          >
            <EvidenceBars
              items={r.topics.map((x) => ({
                name: topicName(x.concept_id),
                count: x.correct,
                total: x.shown,
              }))}
            />
          </Panel>
          {data.summary?.recommendations?.length ? (
            <Panel title="Suggested revision">
              <div className="p-list">
                {data.summary.recommendations.slice(0, 2).map((x) => (
                  <div key={x.name}>
                    <span>
                      <strong>{x.name}</strong>
                      <small>{x.reason}</small>
                    </span>
                  </div>
                ))}
              </div>
            </Panel>
          ) : null}
          <Panel
            title="Review your responses"
            subtitle="Correctness is shown only after results are available."
          >
            <div className="p-review-list">
              {r.questions.map((item, i) => (
                <details key={item.id}>
                  <summary>
                    {i + 1}. {item.prompt}{" "}
                    <Badge
                      tone={
                        item.status === "correct"
                          ? "sage"
                          : item.status === "incorrect"
                            ? "danger"
                            : "neutral"
                      }
                    >
                      {label(item.status)}
                    </Badge>
                  </summary>
                  <p>
                    Your response:{" "}
                    <strong>{answerText(item, item.response)}</strong>
                  </p>
                  {item.explanation && (
                    <p>
                      Answer:{" "}
                      <strong>{answerText(item, item.answer_key)}</strong> ·{" "}
                      {item.explanation}
                    </p>
                  )}
                </details>
              ))}
            </div>
          </Panel>
        </>
      )}
      <div className="p-actions">
        <Button href={r ? "/?view=practice" : "/?view=assessments"}>
          {r ? "Open learning plan" : "Back to assessments"}
        </Button>
        <Button href="/?view=progress&section=history" variant="secondary">
          Session history
        </Button>
      </div>
    </>
  );
}

export function FixedPlayer({ id }: { id: string }) {
  const loaded = useLoad<Fixed>(`/student/fixed/${id}`),
    a = useAction();
  const [data, setData] = useState<Fixed | null>(null),
    [choice, setChoice] = useState<Choice>(null),
    [modal, setModal] = useState<"leave" | "submit" | null>(null),
    [remain, setRemain] = useState<number | null>(null),
    [expired, setExpired] = useState(false);
  useEffect(() => {
    if (loaded.data) {
      setData(loaded.data);
      setChoice(loaded.data.saved_choice);
    }
  }, [loaded.data]);
  useEffect(() => {
    if (!data?.deadline_at || data.status !== "active") {
      setRemain(null);
      return;
    }
    const deadline = Date.parse(data.deadline_at),
      offset = Date.parse(data.server_time) - Date.now();
    const update = () =>
      setRemain(
        Math.max(0, Math.ceil((deadline - Date.now() - offset) / 1000)),
      );
    update();
    const timer = setInterval(update, 1000);
    return () => clearInterval(timer);
  }, [data?.deadline_at, data?.server_time, data?.status]);
  useEffect(() => {
    if (remain === 0 && data?.status === "active" && !expired) {
      setExpired(true);
      loaded.reload();
    }
  }, [remain, data?.status, expired, loaded.reload]);
  const sync = (v: Fixed) => {
    setData(v);
    setChoice(v.saved_choice);
  };
  const item = (action: string, index: number, v?: Choice) =>
    send<Fixed>(`/student/fixed/${id}/item`, {
      action,
      index,
      ...(v == null ? {} : { choice: v }),
    });
  async function advance(skip = false) {
    if (!data) return;
    await a.run(async () => {
      let v = await item(skip ? "skip" : "answer", data.index, choice);
      sync(v);
      if (v.status !== "active") return;
      if (v.index + 1 === v.total) {
        setModal("submit");
        return;
      }
      v = await item("open", v.index + 1);
      sync(v);
    });
  }
  async function submit() {
    const v = await a.run(async () => {
      if (
        data?.status === "active" &&
        choice != null &&
        choice !== "" &&
        choice !== data.saved_choice
      )
        sync(await item("answer", data.index, choice));
      return send<Fixed>(`/student/fixed/${id}/submit`);
    });
    if (v) {
      sync(v);
      setModal(null);
    }
  }
  const q = data?.question,
    shown = data?.positions.filter((x) => x === "answered").length ?? 0,
    skipped = data?.positions.filter((x) => x === "skipped").length ?? 0;
  async function saveCurrent() {
    if (!data) return;
    if (choice != null && choice !== "" && choice !== data.saved_choice) {
      const saved = await item("answer", data.index, choice);
      sync(saved);
      return saved;
    }
    return data;
  }
  async function openQuestion(index: number) {
    await a.run(async () => {
      const saved = await saveCurrent();
      if (saved?.status === "active") sync(await item("open", index));
    });
  }
  async function prepareSubmission() {
    await a.run(async () => {
      await saveCurrent();
      setModal("submit");
    });
  }
  return (
    <>
      <Top
        back="/?view=assessments"
        onLeave={
          data?.status === "active" ? () => setModal("leave") : undefined
        }
      />
      <main className="p-player-body">
        <ActionStatus a={a} />
        {loaded.error && data && (
          <Notice title="Couldn’t check the server" tone="error">
            {loaded.error}{" "}
            <Button onClick={loaded.reload} variant="secondary">
              Try again
            </Button>
          </Notice>
        )}
        {!data ? (
          <Boundary value={loaded}>
            <Notice title="Assessment unavailable">
              Return to the assessment list.
            </Notice>
          </Boundary>
        ) : data.status === "submitted" ? (
          <Report data={data} reload={loaded.reload} />
        ) : (
          q && (
            <>
              <div className="p-player-progress">
                <div className="p-inline">
                  <Badge tone="neutral">
                    {data.mode === "diagnostic" ? "Diagnostic" : "Fixed paper"}
                  </Badge>
                  <strong>
                    {data.index + 1} / {data.total}
                  </strong>
                  <span className="p-faint">{shown} saved</span>
                </div>
                {remain !== null && (
                  <Badge tone={remain < 60 ? "amber" : "neutral"}>
                    <Clock3 size={15} aria-hidden /> {Math.floor(remain / 60)}:
                    {String(remain % 60).padStart(2, "0")} remaining
                  </Badge>
                )}
                <div
                  className="p-bar-track"
                  role="progressbar"
                  aria-label="Saved answers"
                  aria-valuenow={shown}
                  aria-valuemax={data.total}
                >
                  <i style={{ width: `${(100 * shown) / data.total}%` }} />
                </div>
              </div>
              {expired && (
                <Notice title="Time is up">
                  Only answers confirmed by the server count. Checking your
                  attempt.{" "}
                  <Button onClick={loaded.reload} variant="secondary">
                    Check status
                  </Button>
                </Notice>
              )}
              <div className="p-question-layout">
                <Panel className="p-question">
                  <div className="p-eyebrow">
                    {topicName(q.concept_id)} /{" "}
                    {q.question_type === "numeric"
                      ? "NUMERIC ANSWER"
                      : "MULTIPLE CHOICE"}
                  </div>
                  <h1>{q.prompt}</h1>
                  <Options
                    q={q}
                    choice={choice}
                    onChoice={setChoice}
                    locked={a.busy || expired}
                  />
                  <p className="p-faint">
                    {data.saved_choice !== null && choice === data.saved_choice
                      ? "This answer is saved. Results are still hidden."
                      : "Save this answer before moving on. Hints are disabled."}
                  </p>
                  <footer className="p-q-footer">
                    {data.rules.skips_allowed && (
                      <Button
                        variant="secondary"
                        disabled={a.busy || expired}
                        onClick={() => advance(true)}
                      >
                        Skip question
                      </Button>
                    )}
                    <Button
                      disabled={
                        choice == null || choice === "" || a.busy || expired
                      }
                      onClick={() => advance()}
                    >
                      {data.index + 1 === data.total
                        ? "Save & review"
                        : "Save & next"}
                      <ArrowRight size={17} />
                    </Button>
                  </footer>
                </Panel>
                <Panel
                  className="p-question-nav"
                  title="Question navigator"
                  subtitle={`${shown} answered · ${skipped} skipped · ${data.total - shown - skipped} unanswered`}
                >
                  <div className="p-number-grid">
                    {data.positions.map((status, i) => (
                      <button
                        key={i}
                        type="button"
                        className={status}
                        aria-label={`Question ${i + 1}, ${status}`}
                        aria-current={i === data.index ? "step" : undefined}
                        disabled={
                          a.busy ||
                          expired ||
                          i > data.visited_until + 1 ||
                          (!data.rules.review_allowed && i < data.index) ||
                          (!data.rules.skips_allowed &&
                            i > data.index &&
                            data.saved_choice == null)
                        }
                        onClick={() => openQuestion(i)}
                      >
                        {i + 1}
                        {status === "answered"
                          ? " ✓"
                          : status === "skipped"
                            ? " –"
                            : ""}
                      </button>
                    ))}
                  </div>
                  <p className="p-faint">
                    ✓ Answered · – Skipped · outline: current or unanswered
                  </p>
                  <Button
                    variant="secondary"
                    disabled={expired || a.busy}
                    onClick={prepareSubmission}
                  >
                    Review & submit
                  </Button>
                </Panel>
              </div>
            </>
          )
        )}
        {modal === "leave" && (
          <Confirm
            title="Save and leave?"
            action="Leave assessment"
            onClose={() => setModal(null)}
            busy={a.busy}
            onConfirm={() =>
              a.run(async () => {
                await saveCurrent();
                location.assign("/?view=assessments");
              })
            }
          >
            <p>
              {shown} answers are already confirmed. Your current selection will
              also be saved before you leave.{" "}
              {data?.deadline_at
                ? "Your timer keeps running."
                : "You can resume later."}
            </p>
            <ActionStatus a={a} />
          </Confirm>
        )}
        {modal === "submit" && data && (
          <Confirm
            title="Submit this attempt?"
            action="Submit assessment"
            onClose={() => setModal(null)}
            onConfirm={submit}
            busy={a.busy}
          >
            <p>
              {shown} answered · {skipped} skipped ·{" "}
              {data.total - shown - skipped} unanswered. Only confirmed answers
              count, and you can’t change them after submission.
            </p>
            <ActionStatus a={a} />
          </Confirm>
        )}
      </main>
    </>
  );
}

export function PracticePlayer({ id }: { id: string }) {
  const loaded = useLoad<Assessment>(`/assessments/${id}`),
    a = useAction();
  const [data, setData] = useState<Assessment | null>(null);
  const [q, setQuestion] = useState<Question | null>(null);
  const [choice, setChoice] = useState<Choice>(null);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [hint, setHint] = useState("");
  const [requestId, setRequestId] = useState("");
  const [pending, setPending] = useState(false);
  const [working, setWorking] = useState("");
  const [leave, setLeave] = useState(false);
  const [showReview, setShowReview] = useState(false);
  const [online, setOnline] = useState(true);
  const heading = useRef<HTMLHeadingElement>(null);
  const responseHeading = useRef<HTMLHeadingElement>(null);
  const workingDetails = useRef<HTMLDetailsElement>(null);
  const review = useLoad<Review[]>(
    showReview ? `/assessments/${id}/review` : null,
  );
  const draftKey = (question: Question) =>
    `lumen:practice:${id}:${question.id}`;
  function readDraft(question: Question) {
    try {
      const value = JSON.parse(
        sessionStorage.getItem(draftKey(question)) || "null",
      );
      if (
        !value ||
        typeof value.requestId !== "string" ||
        !/^[0-9a-f-]{36}$/i.test(value.requestId)
      )
        return null;
      const valid =
        question.question_type === "numeric"
          ? typeof value.choice === "string"
          : Number.isInteger(value.choice) &&
            value.choice >= 0 &&
            value.choice < question.choices.length;
      return {
        choice: valid ? (value.choice as Choice) : null,
        working:
          typeof value.working === "string" ? value.working.slice(0, 4000) : "",
        requestId: value.requestId,
        pending: Boolean(value.pending),
      };
    } catch {
      return null;
    }
  }
  function writeDraft(
    nextChoice: Choice,
    nextWorking: string,
    waiting = pending,
  ) {
    if (!q) return;
    try {
      sessionStorage.setItem(
        draftKey(q),
        JSON.stringify({
          choice: nextChoice,
          working: nextWorking,
          requestId,
          pending: waiting,
        }),
      );
    } catch {
      /* Optional draft storage may be unavailable. */
    }
  }
  function installQuestion(nextQuestion: Question | null) {
    setQuestion(nextQuestion);
    const draft = nextQuestion ? readDraft(nextQuestion) : null;
    setChoice(draft?.choice ?? null);
    setWorking(draft?.working ?? "");
    setPending(draft?.pending ?? false);
    setRequestId(draft?.requestId ?? crypto.randomUUID());
    setHint("");
    setFeedback(null);
  }
  useEffect(() => {
    if (loaded.data) {
      setData(loaded.data);
      installQuestion(loaded.data.question);
    }
  }, [loaded.data]);
  useEffect(() => {
    const update = () => setOnline(navigator.onLine);
    update();
    addEventListener("online", update);
    addEventListener("offline", update);
    return () => {
      removeEventListener("online", update);
      removeEventListener("offline", update);
    };
  }, []);
  useEffect(() => {
    if (feedback) {
      if (workingDetails.current) workingDetails.current.open = false;
      responseHeading.current?.focus();
    } else if (q) heading.current?.focus();
  }, [q?.id, feedback]);
  const complete = data?.status === "completed" && !feedback;
  async function submit() {
    if (choice == null || choice === "" || !q || !requestId || !online) return;
    setPending(true);
    writeDraft(choice, working, true);
    const v = await a.run(async () => {
      try {
        return await send<{ assessment: Assessment; feedback: Feedback }>(
          `/assessments/${id}/answers`,
          { question_id: q.id, choice_index: choice, request_id: requestId },
        );
      } catch (e) {
        if (
          !(e instanceof RequestError) ||
          (e.status >= 400 && e.status < 500)
        ) {
          setPending(false);
          writeDraft(choice, working, false);
        }
        throw e;
      }
    });
    if (v) {
      try {
        sessionStorage.removeItem(draftKey(q));
      } catch {
        /* Optional browser storage. */
      }
      setPending(false);
      setData(v.assessment);
      setFeedback(v.feedback);
    }
  }
  function next() {
    installQuestion(data?.question ?? null);
  }
  async function showHint() {
    const v = await a.run(() =>
      send<{ hint: string }>(`/assessments/${id}/hint`),
    );
    if (v) setHint(v.hint);
  }
  return (
    <>
      <Top
        back="/?view=practice"
        onLeave={data && !complete ? () => setLeave(true) : undefined}
      />
      <main className="p-player-body practice p-practice-redesign">
        {!online && (
          <Notice title="You’re offline" tone="warn">
            Keep your working here. Reconnect before checking an answer; only
            confirmed answers count.
          </Notice>
        )}
        <ActionStatus a={a} />
        {!data ? (
          <Boundary value={loaded}>
            <Notice title="Practice session unavailable">
              Return to Learn to start or resume a session.
            </Notice>
          </Boundary>
        ) : complete ? (
          <>
            <Heading
              title="Practice complete"
              subtitle={`${data.mode === "adaptive" ? "Mixed topics" : topicName(data.concept_id)} · Your responses are saved.`}
            />
            <div className="p-stats p-practice-stats">
              <Stat
                title="Your answers"
                value={`${data.correct} / ${data.answered} correct`}
                caption={
                  data.answered
                    ? `${percent(data.correct / data.answered)} accuracy in this practice set, not an exam grade`
                    : "No answers recorded"
                }
              />
              <Stat
                title="Questions completed"
                value={`${data.answered} / ${data.total}`}
                caption="Confirmed on the server"
              />
            </div>
            {data.answered < data.total && (
              <Notice title="You’ve reached the end of the available questions">
                Try a different topic from your learning plan.
              </Notice>
            )}
            <Panel
              title="What next?"
              subtitle={
                data.summary?.recommendations[0]?.reason ??
                "Revisit the explanation, then try another set when you’re ready."
              }
            >
              <div className="p-actions">
                <Button href="/?view=practice">Back to learning plan</Button>
                <Button
                  variant="secondary"
                  onClick={() => setShowReview(!showReview)}
                >
                  {showReview ? "Hide response review" : "Review answers"}
                </Button>
              </div>
            </Panel>
            {showReview && (
              <Boundary value={review}>
                <div className="p-review-list">
                  {review.data?.map((x, i) => (
                    <details key={i}>
                      <summary>
                        {x.question.prompt ?? `Question ${i + 1}`}{" "}
                        <Badge tone={x.correct ? "sage" : "danger"}>
                          {x.correct ? "Correct" : "Incorrect"}
                        </Badge>
                      </summary>
                      <p>
                        Your answer: {answerText(x.question, x.choice)}.{" "}
                        {x.hints_used ? "Hint used." : "No hints."}
                      </p>
                      <p>
                        Correct answer:{" "}
                        {answerText(
                          x.question,
                          x.feedback.numeric_answer ?? x.feedback.answer_index,
                        )}
                      </p>
                      <p>{x.feedback.explanation}</p>
                    </details>
                  ))}
                </div>
              </Boundary>
            )}
          </>
        ) : q ? (
          <>
            <div className="p-practice-heading">
              <div>
                <div className="p-eyebrow">
                  {data.mode === "adaptive"
                    ? "MIXED PRACTICE"
                    : "FOCUSED PRACTICE"}
                </div>
                <p>
                  {topicName(q.concept_id)} <span aria-hidden>·</span>{" "}
                  {data.total} questions
                </p>
              </div>
              <span className="p-practice-counter">
                Question {feedback ? data.answered : data.answered + 1}{" "}
                <span>of {data.total}</span>
              </span>
            </div>
            <div
              className="p-practice-track"
              role="progressbar"
              aria-label="Practice progress"
              aria-valuemin={0}
              aria-valuenow={data.answered}
              aria-valuemax={data.total}
            >
              {Array.from({ length: data.total }, (_, i) => (
                <span
                  key={i}
                  className={
                    i < data.answered
                      ? "done"
                      : i === data.answered
                        ? "current"
                        : ""
                  }
                />
              ))}
            </div>
            <Panel className="p-question p-practice-question">
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  if (feedback) next();
                  else submit();
                }}
              >
                <div className="p-inline p-question-meta">
                  <Badge tone="neutral">
                    {["", "Foundation", "Building", "Stretch"][q.difficulty]}
                  </Badge>
                  <span>Take your time. Focus on the method.</span>
                </div>
                <h1 ref={heading} tabIndex={-1}>
                  {q.prompt}
                </h1>
                <Options
                  q={q}
                  choice={choice}
                  onChoice={(v) => {
                    setChoice(v);
                    writeDraft(v, working);
                  }}
                  locked={a.busy || pending || Boolean(feedback)}
                  feedback={feedback}
                />
                {pending && !a.busy && !feedback && (
                  <p className="p-muted" role="status">
                    We couldn’t confirm this answer. Retry the same selection to
                    check whether it was saved.
                  </p>
                )}
                {!feedback && (
                  <div className="p-practice-tools">
                    <Button
                      variant="link"
                      disabled={a.busy || pending || Boolean(hint) || !online}
                      onClick={showHint}
                    >
                      <Lightbulb size={18} />
                      {hint ? "Hint shown" : "Show a hint"}
                    </Button>
                    {!pending && choice != null && choice !== "" && (
                      <Button
                        variant="link"
                        disabled={a.busy}
                        onClick={() => {
                          setChoice(null);
                          writeDraft(null, working);
                        }}
                      >
                        Clear answer
                      </Button>
                    )}
                  </div>
                )}
                {hint && !feedback && (
                  <Notice title="Hint shown · 1 used" tone="ok">
                    {hint}
                  </Notice>
                )}
                <details className="p-working-space" ref={workingDetails}>
                  <summary>
                    <PencilLine size={17} />
                    Working space <span>Optional · kept in this tab</span>
                  </summary>
                  <label htmlFor="practice-working">Your calculations</label>
                  <textarea
                    id="practice-working"
                    maxLength={4000}
                    rows={5}
                    value={working}
                    onChange={(e) => {
                      setWorking(e.target.value);
                      if (!feedback) writeDraft(choice, e.target.value);
                    }}
                    placeholder="Write the steps you want to try…"
                  />
                  <p>This space is not graded or sent to the learning model.</p>
                </details>
                {feedback && (
                  <section
                    className={`p-answer-feedback ${feedback.correct ? "correct" : "incorrect"}`}
                    aria-labelledby="practice-feedback"
                  >
                    <div className="p-feedback-title">
                      {feedback.correct ? (
                        <Check size={23} />
                      ) : (
                        <Lightbulb size={23} />
                      )}
                      <h2
                        id="practice-feedback"
                        ref={responseHeading}
                        tabIndex={-1}
                      >
                        {feedback.correct
                          ? "That’s correct"
                          : "Let’s work through it"}
                      </h2>
                    </div>
                    {!feedback.correct && (
                      <p className="p-answer-comparison">
                        <span>
                          Your answer: <strong>{answerText(q, choice)}</strong>
                        </span>
                        <span>
                          Correct answer:{" "}
                          <strong>
                            {answerText(
                              q,
                              feedback.numeric_answer ?? feedback.answer_index,
                            )}
                          </strong>
                        </span>
                      </p>
                    )}
                    <p>{feedback.explanation}</p>
                  </section>
                )}
                <footer className="p-q-footer p-practice-footer">
                  <span className="p-status-line" role="status">
                    {feedback ? (
                      <>
                        <Check size={17} />
                        Answer saved
                      </>
                    ) : a.busy ? (
                      "Checking with the server…"
                    ) : (
                      "Choose an answer, then check it."
                    )}
                  </span>
                  <Button
                    type="submit"
                    disabled={
                      a.busy ||
                      (!feedback &&
                        (!online || choice == null || choice === ""))
                    }
                  >
                    {feedback
                      ? data.status === "completed"
                        ? "See practice report"
                        : "Next question"
                      : a.busy
                        ? "Checking…"
                        : pending
                          ? "Retry answer"
                          : "Check answer"}
                  </Button>
                </footer>
              </form>
            </Panel>
            <details className="p-practice-method">
              <summary>How is this practice personalized?</summary>
              <p>
                Bayesian Knowledge Tracing (BKT) updates a topic estimate after
                each checked answer. Your answers and difficulty guide the next
                question. It is a learning estimate, not a grade.
              </p>
            </details>
          </>
        ) : (
          <Notice title="No question available">
            Return to Learn and choose another topic.
          </Notice>
        )}
        {leave && (
          <Confirm
            title="Leave this practice session?"
            action="Leave practice"
            onClose={() => setLeave(false)}
            onConfirm={() => location.assign("/?view=practice")}
          >
            <p>
              Checked answers are saved on the server. Your unchecked selection
              and working stay in this browser tab when storage is available;
              they do not count as submitted answers.
            </p>
          </Confirm>
        )}
      </main>
    </>
  );
}
