"use client";
import { useEffect, useRef, useState } from "react";
import { Check, Lightbulb, PencilLine, X } from "lucide-react";
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
  Field,
  Heading,
  Notice,
  Panel,
  Stat,
  topicName,
  useAction,
  useLoad,
} from "./product";

type Choice = number | string | null;
type Review = {
  question: Question;
  choice: Choice;
  correct: boolean;
  feedback: Feedback | null;
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
export function PracticePlayer({ id }: { id: string }) {
  const loaded = useLoad<Assessment>(`/practice/${id}`),
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
    showReview ? `/practice/${id}/review` : null,
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
          `/practice/${id}/answers`,
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
    const v = await a.run(() => send<{ hint: string }>(`/practice/${id}/hint`));
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
                    ? `${percent(data.correct / data.answered)} accuracy in this practice set`
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
                      {x.feedback ? (
                        <>
                          <p>
                            Correct answer:{" "}
                            {answerText(
                              x.question,
                              x.feedback.numeric_answer ??
                                x.feedback.answer_index,
                            )}
                          </p>
                          <p>{x.feedback.explanation}</p>
                        </>
                      ) : (
                        <p>This attempt has no stored explanation.</p>
                      )}
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
