export type LearnerProfile = {
  grade_level: string;
  curriculum: string;
  learning_goal: string;
  weekly_question_target: number;
  adaptive_session_questions: number;
  study_days: string[];
  focus_concepts: string[];
  source: "default" | "sample" | "user";
};
export type User = {
  id: string;
  name: string;
  email: string;
  profile: LearnerProfile;
};
export type Concept = {
  id: string;
  name: string;
  description: string;
  lesson: string;
  prerequisites: string[];
  mastery: number;
  attempts: number;
  correct: number;
  confidence: string;
  status: string;
  evidence_count: number;
  steps?: string[];
  worked_example?: string;
  common_mistake?: string;
  application?: string;
};
export type Recommendation = {
  concept_id: string;
  name: string;
  reason: string;
  mastery: number;
  lesson: string;
  minutes: number;
};
export type Question = {
  id: string;
  prompt: string;
  choices: string[];
  difficulty: number;
  concept_id: string;
};
export type Feedback = {
  correct: boolean;
  answer_index: number;
  explanation: string;
  concept_name: string;
  mastery: number;
};
export type Prediction = {
  probability: number;
  provider: string;
  reason?: string;
};
export type Assessment = {
  id: string;
  mode: string;
  status: string;
  concept_id?: string;
  answered: number;
  total: number;
  correct: number;
  question: Question | null;
  prediction: Prediction | null;
  selection_reason: string;
  latest_feedback?: Feedback;
  summary?: {
    accuracy: number;
    average_mastery: number;
    concepts: Concept[];
    recommendations: Recommendation[];
  };
};
export type Dashboard = {
  profile: LearnerProfile;
  weekly_goal: { answered: number; target: number; remaining: number };
  study_plan: {
    day: string;
    concept_id: string;
    concept_name: string;
    questions: number;
  }[];
  content: {
    question_count: number;
    difficulty_levels: number;
    concept_question_counts: Record<string, number>;
  };
  subject: { id: string; name: string; subtitle: string; concepts: Concept[] };
  concepts: Concept[];
  recommendations: Recommendation[];
  stats: {
    answers: number;
    accuracy: number | null;
    completed_assessments: number;
    study_days: number;
  };
  diagnostic_complete: boolean;
  active_assessment: Assessment | null;
  progress: {
    id: string;
    mode: string;
    completed_at: string;
    accuracy: number;
    mastery: number;
  }[];
  recent_answers: {
    question_id: string;
    concept_id: string;
    correct: boolean;
    answered_at: string;
    difficulty: number;
    prediction: Prediction;
  }[];
  model: { hitskt_loaded: boolean; dataset: string | null; note: string };
};

export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      ...options,
      credentials: "include",
      headers: { "Content-Type": "application/json", ...options.headers },
      cache: "no-store",
    });
  } catch {
    throw new Error(
      "Cannot reach the learning service. Check your connection and try again.",
    );
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    if (response.status === 401)
      throw new AuthError(body.detail || "Please sign in");
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : "Please check your information and try again.",
    );
  }
  return response.status === 204 ? (undefined as T) : response.json();
}
export class AuthError extends Error {}
export const percent = (n: number) => `${Math.round(n * 100)}%`;
