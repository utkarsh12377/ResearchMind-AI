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

// --- Search ---

export interface SearchResult {
  chunk_id: string;
  paper_id: string;
  paper_title: string | null;
  content: string;
  section_path: string | null;
  page_number: number | null;
  kind: string;
  citation: string;
  score: number;
  dense_rank: number | null;
  sparse_rank: number | null;
  rerank_score: number | null;
  supporting_sentences: string[];
}

export interface SearchResponse {
  query: string;
  total: number;
  results: SearchResult[];
}

// --- Knowledge graph ---

export interface GraphNode {
  id: string;
  label: string;
  kind: string;
  weight: number;
  metadata: Record<string, unknown>;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  kind: string;
}

export interface NetworkResponse {
  view: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  node_count: number;
  edge_count: number;
}

export interface EntityCount {
  label: string;
  name: string;
  paper_count: number;
}

export interface ConnectedPaper {
  paper_id: string;
  title: string;
  entity_count: number;
}

export interface GraphOverview {
  papers: number;
  counts: Record<string, number>;
  top_entities: EntityCount[];
  most_connected: ConnectedPaper[];
}

export interface GraphQueryResponse {
  question: string;
  cypher: string;
  rows: Record<string, unknown>[];
  error: string | null;
}

// --- Insights ---

export interface ComparisonCell {
  paper_id: string;
  paper_title: string;
  model: string;
  value: number;
  unit: string | null;
  split: string | null;
  is_best: boolean;
}

export interface ComparisonRow {
  dataset: string;
  metric: string;
  spread: number;
  cells: ComparisonCell[];
}

export interface ComparisonResponse {
  rows: ComparisonRow[];
  ungrouped_results: number;
  markdown: string;
}

export interface PaperMatrixRow {
  paper_id: string;
  title: string;
  year: number | null;
  datasets: string[];
  models: string[];
  metrics: string[];
}

export interface YearBucket {
  year: number;
  paper_count: number;
  titles: string[];
}

export interface EntityTrend {
  label: string;
  name: string;
  by_year: Record<string, number>;
  total: number;
  direction: "rising" | "emerging" | "declining" | "steady" | "new";
  first_seen: number | null;
}

export interface ProgressionPoint {
  year: number;
  value: number;
  model: string;
}

export interface Progression {
  dataset: string;
  metric: string;
  points: ProgressionPoint[];
  improvement: number | null;
}

export interface TrendResponse {
  narrative: string;
  timeline: YearBucket[];
  undated_papers: number;
  entity_trends: EntityTrend[];
  progressions: Progression[];
}

export interface NumericConflict {
  model: string;
  dataset: string;
  metric: string;
  left_paper: string;
  left_value: number;
  right_paper: string;
  right_value: number;
  relative_gap: number;
}

export interface ClaimPair {
  relation: "agreement" | "contradiction";
  confidence: number;
  reason: string;
  left_paper: string;
  left_claim: string;
  right_paper: string;
  right_claim: string;
}

export interface ConsistencyResponse {
  summary: string;
  is_consistent: boolean;
  papers_examined: number;
  numeric_conflicts: NumericConflict[];
  contradictions: ClaimPair[];
  agreements: ClaimPair[];
}

export interface StructuralGap {
  kind: string;
  description: string;
  evidence: string[];
  strength: number;
}

export interface StatedGap {
  paper_title: string;
  text: string;
}

export interface SuggestedDirection {
  gap: string;
  rationale: string;
  suggested_direction: string;
  confidence: number;
}

export interface GapResponse {
  topic: string;
  summary: string;
  papers_examined: number;
  structural: StructuralGap[];
  stated: StatedGap[];
  directions: SuggestedDirection[];
}

export interface ReviewSection {
  heading: string;
  focus: string;
  text: string;
  source_indices: number[];
}

export interface ReviewSource {
  index: number;
  paper_id: string;
  title: string;
  authors: string | null;
  year: number | null;
  citation: string;
}

export interface ReviewResponse {
  topic: string;
  title: string;
  sections: ReviewSection[];
  sources: ReviewSource[];
  word_count: number;
  markdown: string;
  bibtex: string;
}
