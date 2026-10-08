"use client";

import { useState } from "react";
import { ArrowRight, CalendarDays, Check, Save, Target } from "lucide-react";
import { api, Concept, Dashboard, LearnerProfile, User } from "@/lib/api";

const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const grades = [
  "Not set",
  "Grade 6",
  "Grade 7",
  "Grade 8",
  "Grade 9",
  "Grade 10",
  "Grade 11",
  "Grade 12",
  "Independent learner",
];

export function StudyPlan({
  data,
  onConfigure,
  onReview,
}: {
  data: Dashboard;
  onConfigure: () => void;
  onReview: (conceptId: string) => void;
}) {
  const { answered, target, remaining } = data.weekly_goal;
  return (
    <section className="card study-plan-card">
      <div className="section-heading">
        <div>
          <h2>A little plan, a steady rhythm</h2>
          <p>{data.profile.learning_goal}</p>
        </div>
        <button className="text-button" onClick={onConfigure}>
          Edit my plan <ArrowRight size={15} />
        </button>
      </div>
      {data.profile.source === "sample" && (
        <p className="sample-note">
          Example preferences have been added to get you started. Edit your
          profile to make them yours.
        </p>
      )}
      <div className="plan-layout">
        <div className="goal-summary">
          <span className="eyebrow">
            <Target size={15} /> PAST 7 DAYS
          </span>
          <strong>
            {answered}
            <small> / {target} questions</small>
          </strong>
          <div
            className="progress-track"
            role="progressbar"
            aria-label="Question target progress"
            aria-valuenow={Math.min(answered, target)}
            aria-valuemin={0}
            aria-valuemax={target}
          >
            <i
              style={{ width: `${Math.min(100, (answered / target) * 100)}%` }}
            />
          </div>
          <p>
            {remaining
              ? `${remaining} more answers to reach your target.`
              : "You reached your question target. Keep growing at your pace."}
          </p>
        </div>
        <div className="plan-sessions">
          <span className="eyebrow">
            <CalendarDays size={15} /> YOUR SUGGESTED RHYTHM
          </span>
          <div className="plan-days">
            {data.study_plan.map((session) => (
              <button
                key={session.day}
                onClick={() => onReview(session.concept_id)}
              >
                <span className="day-badge">{session.day}</span>
                <span>
                  <strong>{session.concept_name}</strong>
                  <small>
                    {session.questions} focused questions · review & practice
                  </small>
                </span>
                <ArrowRight size={16} />
              </button>
            ))}
          </div>
          <p className="plan-caption">
            A flexible study suggestion. Choose any day that works for you.
          </p>
        </div>
      </div>
    </section>
  );
}

export function LearnerProfileForm({
  user,
  concepts,
  onSaved,
}: {
  user: User;
  concepts: Concept[];
  onSaved: (user: User) => void;
}) {
  const [draft, setDraft] = useState<LearnerProfile>(user.profile);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  function change<K extends keyof LearnerProfile>(
    key: K,
    value: LearnerProfile[K],
  ) {
    setDraft((old) => ({ ...old, [key]: value }));
    setSaved(false);
  }
  function toggle(key: "study_days" | "focus_concepts", value: string) {
    change(
      key,
      draft[key].includes(value)
        ? draft[key].filter((item) => item !== value)
        : [...draft[key], value],
    );
  }
  async function save(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!draft.study_days.length) {
      setError("Choose at least one study day.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const { source, ...profile } = draft;
      void source;
      const updated = await api<User>("/profile", {
        method: "POST",
        body: JSON.stringify(profile),
      });
      setDraft(updated.profile);
      setSaved(true);
      onSaved(updated);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="card profile-card">
      <div className="section-heading">
        <div>
          <h2>Make this space yours</h2>
          <p>
            Your preferences shape your study plan and the length of new
            adaptive sessions.
          </p>
        </div>
      </div>
      <div className="account-summary">
        <span className="avatar">{user.name[0].toUpperCase()}</span>
        <div>
          <strong>{user.name}</strong>
          <span>{user.email}</span>
        </div>
      </div>
      {draft.source === "sample" && (
        <p className="sample-note">
          This is an example profile, including the grade and curriculum. Update
          these fields and save your own preferences.
        </p>
      )}
      <form onSubmit={save} className="profile-form">
        <div className="profile-fields">
          <label>
            Grade or learning stage
            <select
              value={draft.grade_level}
              onChange={(e) => change("grade_level", e.target.value)}
            >
              {grades.map((grade) => (
                <option key={grade}>{grade}</option>
              ))}
            </select>
          </label>
          <label>
            Curriculum
            <select
              value={draft.curriculum}
              onChange={(e) => change("curriculum", e.target.value)}
            >
              {["Not set", "CBSE", "ICSE", "State board", "Other"].map(
                (curriculum) => (
                  <option key={curriculum}>{curriculum}</option>
                ),
              )}
            </select>
          </label>
          <label className="span-all">
            My learning goal
            <textarea
              required
              minLength={5}
              maxLength={240}
              rows={3}
              value={draft.learning_goal}
              onChange={(e) => change("learning_goal", e.target.value)}
              placeholder="What would you like to feel more confident about?"
            />
            <small>{draft.learning_goal.length}/240 characters</small>
          </label>
          <label>
            7-day question target
            <input
              type="number"
              required
              min={5}
              max={200}
              step={1}
              value={draft.weekly_question_target}
              onChange={(e) =>
                change("weekly_question_target", e.target.valueAsNumber)
              }
            />
            <small>
              Progress counts your actual answers in the past seven days.
            </small>
          </label>
          <label>
            Adaptive session length
            <select
              value={draft.adaptive_session_questions}
              onChange={(e) =>
                change("adaptive_session_questions", Number(e.target.value))
              }
            >
              {[6, 12, 18].map((count) => (
                <option key={count} value={count}>
                  {count} questions
                </option>
              ))}
            </select>
            <small>
              New adaptive sessions use this length. Diagnostics have 12
              questions; focused practice has 6.
            </small>
          </label>
        </div>
        <fieldset>
          <legend>Days I like to study</legend>
          <div className="preference-choices">
            {days.map((day) => (
              <label key={day}>
                <input
                  type="checkbox"
                  checked={draft.study_days.includes(day)}
                  onChange={() => toggle("study_days", day)}
                />
                {day}
              </label>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>Concepts I want to focus on</legend>
          <p className="muted">
            Choose your interests, or leave these empty to follow your
            assessment recommendations.
          </p>
          <div className="preference-choices">
            {concepts.map((concept) => (
              <label key={concept.id}>
                <input
                  type="checkbox"
                  checked={draft.focus_concepts.includes(concept.id)}
                  onChange={() => toggle("focus_concepts", concept.id)}
                />
                {concept.name}
              </label>
            ))}
          </div>
        </fieldset>
        <p className="profile-disclosure">
          Grade and curriculum describe your preferences. This foundations bank
          is not a complete board syllabus.
        </p>
        {error && (
          <div className="error" role="alert">
            {error}
          </div>
        )}
        <div className="profile-actions">
          <button className="button primary" disabled={busy}>
            <Save size={17} />
            {busy ? "Saving…" : "Save my preferences"}
          </button>
          {saved && (
            <span role="status">
              <Check size={17} />
              Preferences saved
            </span>
          )}
        </div>
      </form>
    </section>
  );
}
