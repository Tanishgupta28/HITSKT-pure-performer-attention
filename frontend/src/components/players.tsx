"use client";
import { useEffect, useState } from "react";
import { ArrowRight, Check, Clock3 } from "lucide-react";
import { Assessment, Feedback, Question, percent, send } from "@/lib/api";
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
            <span>
              {String.fromCharCode(65 + i)} · {x}
            </span>
            {feedback?.answer_index === i ? (
              <small>
                <Check size={15} /> Correct
              </small>
            ) : feedback && choice === i ? (
              <small>Your answer</small>
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
          <div className="p-stats">
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
            <Stat
              title="Estimated mastery"
              value="See topic evidence"
              caption="BKT estimates are shown separately in Progress"
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
                    <Button href="/?view=revision" variant="secondary">
                      Open plan
                    </Button>
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
        <Button href={r ? "/?view=revision" : "/?view=assessments"}>
          {r ? "Open revision plan" : "Back to assessments"}
        </Button>
        <Button href="/?view=history" variant="secondary">
          Learning history
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
    const v = await a.run(() => send<Fixed>(`/student/fixed/${id}/submit`));
    if (v) {
      sync(v);
      setModal(null);
    }
  }
  const q = data?.question,
    shown = data?.positions.filter((x) => x === "answered").length ?? 0,
    skipped = data?.positions.filter((x) => x === "skipped").length ?? 0;
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
                        onClick={() =>
                          a.run(async () => sync(await item("open", i)))
                        }
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
                    onClick={() => setModal("submit")}
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
            onConfirm={() => location.assign("/?view=assessments")}
          >
            <p>
              {shown} answers are confirmed. Any current unsaved choice won’t be
              included.{" "}
              {data?.deadline_at
                ? "Your timer keeps running."
                : "You can resume later."}
            </p>
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
  const [data, setData] = useState<Assessment | null>(null),
    [q, setQuestion] = useState<Question | null>(null),
    [choice, setChoice] = useState<Choice>(null),
    [feedback, setFeedback] = useState<Feedback | null>(null),
    [hint, setHint] = useState(""),
    [requestId, setRequestId] = useState(""),
    [leave, setLeave] = useState(false),
    [showReview, setShowReview] = useState(false);
  const review = useLoad<Review[]>(
    showReview ? `/assessments/${id}/review` : null,
  );
  useEffect(() => {
    if (loaded.data) {
      setData(loaded.data);
      setQuestion(loaded.data.question);
      setChoice(null);
      setFeedback(null);
      setHint("");
      setRequestId(crypto.randomUUID());
    }
  }, [loaded.data]);
  const complete = data?.status === "completed" && !feedback;
  async function submit() {
    if (choice == null || choice === "" || !q || !requestId) return;
    const v = await a.run(() =>
      send<{ assessment: Assessment; feedback: Feedback }>(
        `/assessments/${id}/answers`,
        { question_id: q.id, choice_index: choice, request_id: requestId },
      ),
    );
    if (v) {
      setData(v.assessment);
      setFeedback(v.feedback);
    }
  }
  function next() {
    setQuestion(data?.question ?? null);
    setFeedback(null);
    setChoice(null);
    setHint("");
    setRequestId(crypto.randomUUID());
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
        back="/?view=history"
        onLeave={data && !complete ? () => setLeave(true) : undefined}
      />
      <main className="p-player-body practice">
        <ActionStatus a={a} />
        {!data ? (
          <Boundary value={loaded}>
            <Notice title="Practice session unavailable">
              Return to the workspace to start a session.
            </Notice>
          </Boundary>
        ) : complete ? (
          <>
            <Heading
              title="Practice complete"
              subtitle={`${topicName(data.concept_id)} · ${data.mode === "adaptive" ? "Adaptive mock" : "Focused practice"}`}
            />
            <div className="p-stats">
              <Stat
                title="Attempted"
                value={`${data.answered} / ${data.total}`}
                caption="Saved responses"
              />
              <Stat
                title="Correct"
                value={data.correct}
                caption="Practice result, not an exam grade"
              />
              <Stat
                title="Answer accuracy"
                value={
                  data.answered ? percent(data.correct / data.answered) : "—"
                }
                caption="Correct of answered questions"
              />
            </div>
            {data.answered < data.total && (
              <Notice title="No more eligible items in this set">
                Choose another topic or try mixed practice.
              </Notice>
            )}
            <Panel
              title="Check your understanding again later"
              subtitle="Try a fresh set without hints after a short break."
            >
              <div className="p-actions">
                <Button href="/?view=revision">Open revision plan</Button>
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
                      {x.feedback && <p>{x.feedback.explanation}</p>}
                    </details>
                  ))}
                </div>
              </Boundary>
            )}
          </>
        ) : q ? (
          <>
            <div className="p-player-progress">
              <div className="p-inline">
                <Badge tone="neutral">
                  {data.mode === "adaptive"
                    ? "Adaptive mock"
                    : "Adaptive practice"}
                </Badge>
                <strong>
                  {feedback ? data.answered : data.answered + 1} of {data.total}
                </strong>
              </div>
              <span className="p-faint">
                Bayesian knowledge tracing guides live practice.
              </span>
              <div
                className="p-bar-track"
                role="progressbar"
                aria-label="Practice progress"
                aria-valuenow={data.answered}
                aria-valuemax={data.total}
              >
                <i
                  style={{ width: `${(100 * data.answered) / data.total}%` }}
                />
              </div>
            </div>
            <Panel className="p-question">
              <div className="p-eyebrow">
                {topicName(q.concept_id)} /{" "}
                {["", "FOUNDATION", "BUILDING", "STRETCH"][q.difficulty]}
              </div>
              <h1>{q.prompt}</h1>
              <Options
                q={q}
                choice={choice}
                onChoice={setChoice}
                locked={a.busy || !!feedback}
                feedback={feedback}
              />
              {hint && !feedback && (
                <Notice title="Hint shown · 1 used" tone="ok">
                  {hint}
                </Notice>
              )}
              {feedback && (
                <Notice
                  title={
                    feedback.correct ? "Correct" : "Let’s work through that"
                  }
                  tone={feedback.correct ? "ok" : "warn"}
                >
                  {feedback.numeric_answer && (
                    <p>Correct answer: {feedback.numeric_answer}</p>
                  )}
                  {feedback.explanation}
                </Notice>
              )}
              <footer className="p-q-footer">
                {feedback ? (
                  <span className="p-status-line">
                    <Check size={17} /> Answer saved
                  </span>
                ) : (
                  <Button
                    variant="secondary"
                    disabled={a.busy || !!hint}
                    onClick={showHint}
                  >
                    {hint ? "Hint shown" : "Show a hint"}
                  </Button>
                )}
                {feedback ? (
                  <Button onClick={next}>
                    {data.status === "completed"
                      ? "See practice report"
                      : "Next question"}
                    <ArrowRight size={17} />
                  </Button>
                ) : (
                  <Button
                    onClick={submit}
                    disabled={a.busy || choice == null || choice === ""}
                  >
                    {a.busy ? "Saving answer…" : "Check answer"}
                  </Button>
                )}
              </footer>
            </Panel>
          </>
        ) : (
          <Notice title="No question available">
            Return to the workspace and choose another topic.
          </Notice>
        )}
        {leave && (
          <Confirm
            title="Leave this practice session?"
            action="Leave practice"
            onClose={() => setLeave(false)}
            onConfirm={() => location.assign("/?view=today")}
          >
            Previously checked answers are saved. You can resume from your last
            confirmed answer.
          </Confirm>
        )}
      </main>
    </>
  );
}
