export type JunyiSkill = {
  tag: number;
  topic: string;
  stage: string;
  category: "weak" | "intermediate" | "strong" | null;
  unattempted: boolean;
  unavailable_reason: string | null;
  score: number | null;
  original_score: number | null;
  actual_rate: number | null;
  original_attempts: number;
  original_correct: number;
  app_attempts: number;
  app_correct: number;
};

export type JunyiResource = { url: string; title: string; source: string };
export type StudyLookup = { resources: JunyiResource[]; message: string | null };
export type JunyiRequirement = StudyLookup & {
  tag: number;
  topic: string;
  status: "pending" | "satisfied" | "overridden";
  lookup: "waiting" | "loading" | "verified" | "unavailable";
  can_retry: boolean;
};

type PaperStatus = "preparing" | "locked" | "open" | "completed" | "shortage" | "unavailable";
type Result = { correct: number; total: number; accuracy: number };

export type JunyiPaper = {
  id: string;
  sequence: number;
  revision: number;
  status: PaperStatus;
  stage: string | null;
  total: number;
  answered: number;
  message: string | null;
  requirements: JunyiRequirement[];
  question: { id: string; topic: string; prompt: string; options: { id: string; text: string }[] } | null;
  results: Result | null;
  analysis_status: "pending" | "ready" | "unavailable" | null;
  analysis_retry: boolean;
  answers_used: number | null;
  skills: JunyiSkill[];
  can_retry: boolean;
};

export type JunyiAssigned = {
  assigned: true;
  ready: boolean;
  issues: string[];
  stage: string | null;
  active: (Pick<JunyiPaper, "id" | "status" | "sequence"> & { required_tags: number[] }) | null;
  history: (Result & { id: string; sequence: number; completed_at: number; stage: string | null })[];
  answers_used: number | null;
  skills: JunyiSkill[];
};

export type JunyiDashboard = { assigned: false } | JunyiAssigned;
export type JunyiAdmin = { ready: boolean; issues: string[]; bank_questions: number; active: JunyiPaper[] };

export const paperUrl = (id: string) => `/assessment/junyi/${encodeURIComponent(id)}`;
