"use client";
import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCircle2,
  CircleHelp,
  Sprout,
  X,
} from "lucide-react";
import {
  api,
  Assessment,
  AuthError,
  Feedback,
  percent,
  Question,
} from "@/lib/api";
import { Brand, BrandMark } from "@/components/brand";
import { MotionToggle } from "@/components/motion";

export default function AssessmentPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [assessment, setAssessment] = useState<Assessment | null>(null);
  const [shown, setShown] = useState<Question | null>(null);
  const [choice, setChoice] = useState<number | null>(null);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [requestId, setRequestId] = useState("");
  async function load() {
    try {
      const value = await api<Assessment>(`/assessments/${id}`);
      setAssessment(value);
      setShown(value.question);
      setChoice(null);
      setFeedback(null);
      setRequestId(crypto.randomUUID());
      setError("");
    } catch (e) {
      if (e instanceof AuthError) router.push("/");
      else setError((e as Error).message);
    }
  }
  useEffect(() => {
    void load();
  }, [id]);
  async function submit() {
    if (choice === null || !shown || busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await api<{ assessment: Assessment; feedback: Feedback }>(
        `/assessments/${id}/answers`,
        {
          method: "POST",
          body: JSON.stringify({
            question_id: shown.id,
            choice_index: choice,
            request_id: requestId,
          }),
        },
      );
      setAssessment(result.assessment);
      setFeedback(result.feedback);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  function next() {
    setShown(assessment?.question || null);
    setChoice(null);
    setFeedback(null);
    setRequestId(crypto.randomUUID());
  }
  const completed = assessment?.status === "completed" && !feedback;
  return (
    <main className="assessment-page">
      <header className="assessment-header">
        <a href="/" aria-label="Lumen home">
          <Brand />
        </a>
        <div className="assessment-toolbar">
          <MotionToggle />
          <button className="text-button" onClick={() => router.push("/")}>
            <X size={22} />
            Save & leave
          </button>
        </div>
      </header>
      <div className="assessment-container">
        {error && (
          <div className="error" role="alert">
            {error}
            <button onClick={load}>Reload assessment</button>
          </div>
        )}
        {!assessment ? (
          <div className="card loading-card">
            Preparing your learning moment…
          </div>
        ) : completed ? (
          <section className="completion card">
            <span className="completion-icon">
              <BrandMark />
            </span>
            <div className="eyebrow">A LITTLE FURTHER THAN BEFORE</div>
            <h1>That’s progress.</h1>
            <p>You made time to understand. Here’s what we discovered.</p>
            <div className="completion-stats">
              <div>
                <strong>
                  {assessment.correct}/{assessment.total}
                </strong>
                <span>answers correct</span>
              </div>
              <div>
                <strong>{percent(assessment.summary!.accuracy)}</strong>
                <span>assessment accuracy</span>
              </div>
            </div>
            <h3>Your next learning moments</h3>
            <div className="completion-recs">
              {assessment.summary?.recommendations.map((r) => (
                <div key={r.concept_id}>
                  <span className="round-icon">
                    <ArrowRight size={22} />
                  </span>
                  <span>
                    <strong>{r.name}</strong>
                    <small>{r.reason}</small>
                  </span>
                </div>
              ))}
            </div>
            <button
              className="button primary full"
              onClick={() => router.push("/")}
            >
              Back to my learning space
              <ArrowRight size={23} />
            </button>
            <p className="fine-print">
              Concept estimates are a guide for practice. They are not grades.
            </p>
          </section>
        ) : (
          <>
            <div className="assessment-top">
              <span className="pill">
                {assessment.mode === "diagnostic"
                  ? "FINDING YOUR STARTING POINT"
                  : assessment.mode === "practice"
                    ? "FOCUSED PRACTICE"
                    : "ADAPTIVE PRACTICE"}
              </span>
              <span>
                {feedback ? assessment.answered : assessment.answered + 1} of{" "}
                {assessment.total}
              </span>
            </div>
            <div className="assessment-progress">
              <i
                style={{
                  width: `${(assessment.answered / assessment.total) * 100}%`,
                }}
              />
            </div>
            {shown && (
              <section className="question-card card" key={shown.id}>
                <div className="question-meta">
                  <span>{shown.concept_id.replaceAll("_", " ")}</span>
                  <span>
                    {
                      ["", "Foundation", "Building", "Stretching"][
                        shown.difficulty
                      ]
                    }
                    <span className="difficulty-dots">
                      {[1, 2, 3].map((i) => (
                        <i
                          key={i}
                          className={shown.difficulty >= i ? "filled" : ""}
                        />
                      ))}
                    </span>
                  </span>
                </div>
                <h1>{shown.prompt}</h1>
                <p className="question-hint">
                  Take your time. Choose the answer that feels right.
                </p>
                <div
                  className="choices"
                  role="group"
                  aria-label="Answer choices"
                >
                  {shown.choices.map((text, i) => (
                    <button
                      key={i}
                      disabled={!!feedback || busy}
                      className={`choice ${choice === i ? "selected" : ""} ${feedback?.answer_index === i ? "right-answer" : ""} ${feedback && choice === i && !feedback.correct ? "wrong-answer" : ""}`}
                      onClick={() => setChoice(i)}
                    >
                      <span className="choice-letter">
                        {String.fromCharCode(65 + i)}
                      </span>
                      <span>{text}</span>
                      {feedback?.answer_index === i ? (
                        <Check size={24} />
                      ) : (
                        <span className="choice-radio" />
                      )}
                    </button>
                  ))}
                </div>
                {feedback ? (
                  <div
                    className={`answer-feedback ${feedback.correct ? "positive" : "reflect"}`}
                    role="status"
                  >
                    <strong>
                      {feedback.correct ? (
                        <CheckCircle2 size={24} />
                      ) : (
                        <CircleHelp size={24} />
                      )}{" "}
                      {feedback.correct
                        ? "Nicely understood."
                        : "A useful moment to learn."}
                    </strong>
                    <p>{feedback.explanation}</p>
                    <span>
                      {feedback.concept_name} · {percent(feedback.mastery)}{" "}
                      estimated understanding
                    </span>
                  </div>
                ) : null}
                <div className="question-actions">
                  <span>
                    <Sprout size={20} />
                    {feedback
                      ? "Your progress has been saved."
                      : "Every answer helps shape your path."}
                  </span>
                  {feedback ? (
                    <button className="button primary" onClick={next}>
                      {assessment.status === "completed"
                        ? "See my discoveries"
                        : "Next question"}
                      <ArrowRight size={23} />
                    </button>
                  ) : (
                    <button
                      className="button primary"
                      disabled={choice === null || busy}
                      onClick={submit}
                    >
                      {busy ? "Saving…" : "Check my answer"}
                      <ArrowRight size={23} />
                    </button>
                  )}
                </div>
              </section>
            )}
            <div className="assessment-note">
              <CircleHelp size={20} />
              {assessment.prediction?.provider === "hitskt"
                ? "HiTSKT uses your earlier sessions to help select your next question."
                : "Questions adapt to your answers as your concept estimates take shape."}
            </div>
          </>
        )}
      </div>
    </main>
  );
}
