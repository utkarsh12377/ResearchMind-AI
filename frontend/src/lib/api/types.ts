export interface User {
  id: string;
  email: string;
  full_name: string | null;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
}

export interface Token {
  access_token: string;
  token_type: string;
}

export interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used_at: string | null;
  revoked_at: string | null;
}

export interface ApiKeyCreated extends ApiKey {
  api_key: string;
}

export interface RegisterPayload {
  email: string;
  password: string;
  full_name?: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

/** Shape of the structured error envelope returned by the backend. */
export interface ApiErrorBody {
  error: {
    type: string;
    message: string;
  };
}

// --- Retrieval & research ---

export interface SearchFilters {
  paper_ids?: string[];
  kinds?: string[];
  year_from?: number | null;
  year_to?: number | null;
  sections?: string[];
}

export interface Paper {
  id: string;
  workspace_id: string;
  title: string | null;
  status: "pending" | "processing" | "ready" | "failed";
  original_filename: string;
  content_type: string;
  size_bytes: number;
  checksum: string;
  page_count: number | null;
  abstract: string | null;
  authors: string | null;
  error_message: string | null;
  ocr_page_count: number;
  created_at: string;
  updated_at: string;
}

export interface PaperList {
  items: Paper[];
  total: number;
}

export interface ResearchSource {
  index: number;
  chunk_id: string;
  paper_id: string;
  paper_title: string | null;
  citation: string;
  page_number: number | null;
  content: string;
}

export interface AgentStep {
  agent: string;
  summary: string;
  detail: string;
  duration_ms: number;
}

export interface ResearchPlan {
  intent: string;
  sub_questions: string[];
  reasoning: string;
  needs_web_search: boolean;
}

export interface ResearchResult {
  question: string;
  answer: string;
  plan: ResearchPlan | null;
  sources: ResearchSource[];
  citations: number[];
  confidence: number;
  verification: {
    supported: boolean;
    confidence: number;
    unsupported_claims: string[];
    reason: string;
  } | null;
  revision_count: number;
  usage_tokens: number;
  steps: AgentStep[];
  errors: string[];
}

/** One server-sent event from the research stream. */
export type ResearchEvent =
  | ({ type: "step" } & AgentStep)
  | ({ type: "result" } & ResearchResult);
