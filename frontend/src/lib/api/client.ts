import type {
  ApiErrorBody,
  ApiKey,
  ApiKeyCreated,
  ComparisonResponse,
  ConsistencyResponse,
  GapResponse,
  GraphOverview,
  GraphQueryResponse,
  LoginPayload,
  NetworkResponse,
  PaperList,
  PaperMatrixRow,
  RegisterPayload,
  ResearchEvent,
  ResearchResult,
  ReviewResponse,
  SearchFilters,
  SearchResponse,
  Token,
  TrendResponse,
  User,
} from "@/lib/api/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const TOKEN_STORAGE_KEY = "researchmind.access_token";

export interface HealthResponse {
  status: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly type?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }

  get isUnauthorized(): boolean {
    return this.status === 401;
  }
}

/**
 * Token persistence.
 *
 * Trade-off: localStorage is readable by any script on the origin, so a
 * successful XSS can exfiltrate the token. The more secure alternative is an
 * httpOnly, SameSite cookie set by the backend, which JS cannot read at all.
 * We use localStorage here because the API is a separate origin from the
 * frontend and is designed to be callable by non-browser clients too (see the
 * X-API-Key auth path); cross-site cookies would require CORS credentials plus
 * CSRF protection. Revisit if the two are ever served from one origin.
 */
export const tokenStorage = {
  get(): string | null {
    if (typeof window === "undefined") return null;
    return window.localStorage.getItem(TOKEN_STORAGE_KEY);
  },
  set(token: string): void {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
  },
  clear(): void {
    if (typeof window === "undefined") return;
    window.localStorage.removeItem(TOKEN_STORAGE_KEY);
  },
};

interface RequestOptions extends Omit<RequestInit, "body"> {
  body?: unknown;
  /** Send the stored bearer token. Defaults to true. */
  auth?: boolean;
  /** Send `body` as form-urlencoded instead of JSON (OAuth2 login flow). */
  form?: boolean;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { body, auth = true, form = false, headers, ...init } = options;

  const requestHeaders = new Headers(headers);
  if (auth) {
    const token = tokenStorage.get();
    if (token) requestHeaders.set("Authorization", `Bearer ${token}`);
  }

  let requestBody: BodyInit | undefined;
  if (body !== undefined) {
    if (form) {
      requestHeaders.set("Content-Type", "application/x-www-form-urlencoded");
      requestBody = new URLSearchParams(body as Record<string, string>).toString();
    } else {
      requestHeaders.set("Content-Type", "application/json");
      requestBody = JSON.stringify(body);
    }
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: requestHeaders,
      body: requestBody,
    });
  } catch (error) {
    throw new ApiError(
      error instanceof DOMException && error.name === "AbortError"
        ? "Request cancelled"
        : "Unable to reach the backend",
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }

  const text = await response.text();
  const payload: unknown = text ? JSON.parse(text) : null;

  if (!response.ok) {
    const errorBody = payload as ApiErrorBody | null;
    throw new ApiError(
      errorBody?.error?.message ?? `Request failed with status ${response.status}`,
      response.status,
      errorBody?.error?.type,
    );
  }

  return payload as T;
}

export const api = {
  health: () => request<HealthResponse>("/health", { auth: false }),

  register: (payload: RegisterPayload) =>
    request<User>("/api/v1/auth/register", { method: "POST", body: payload, auth: false }),

  // The backend uses the OAuth2 password flow, which expects form-encoded
  // `username`/`password` fields rather than a JSON body.
  login: (payload: LoginPayload) =>
    request<Token>("/api/v1/auth/login", {
      method: "POST",
      body: { username: payload.email, password: payload.password },
      form: true,
      auth: false,
    }),

  me: () => request<User>("/api/v1/auth/me"),

  listApiKeys: () => request<ApiKey[]>("/api/v1/auth/api-keys"),

  createApiKey: (name: string) =>
    request<ApiKeyCreated>("/api/v1/auth/api-keys", { method: "POST", body: { name } }),

  revokeApiKey: (id: string) =>
    request<void>(`/api/v1/auth/api-keys/${id}`, { method: "DELETE" }),

  listPapers: (limit = 50, offset = 0) =>
    request<PaperList>(`/api/v1/papers?limit=${limit}&offset=${offset}`),

  uploadPaper: async (file: File) => {
    // FormData sets its own multipart boundary, so this bypasses request()
    // rather than letting it force a Content-Type.
    const body = new FormData();
    body.append("file", file);

    const headers = new Headers();
    const token = tokenStorage.get();
    if (token) headers.set("Authorization", `Bearer ${token}`);

    const response = await fetch(`${API_BASE_URL}/api/v1/papers`, {
      method: "POST",
      headers,
      body,
    });

    const text = await response.text();
    const payload = text ? JSON.parse(text) : null;
    if (!response.ok) {
      const error = payload as ApiErrorBody | null;
      throw new ApiError(
        error?.error?.message ?? `Upload failed with status ${response.status}`,
        response.status,
        error?.error?.type,
      );
    }
    return payload;
  },

  deletePaper: (id: string) => request<void>(`/api/v1/papers/${id}`, { method: "DELETE" }),

  research: (question: string, filters?: SearchFilters) =>
    request<ResearchResult>("/api/v1/research", {
      method: "POST",
      body: { question, filters: filters ?? {} },
    }),

  search: (query: string, options: { limit?: number; filters?: SearchFilters } = {}) =>
    request<SearchResponse>("/api/v1/search", {
      method: "POST",
      body: {
        query,
        limit: options.limit ?? 10,
        filters: options.filters ?? {},
      },
    }),

  graphOverview: () => request<GraphOverview>("/api/v1/graph/overview"),

  graphNetwork: (view: string, options: { labels?: string[]; minPapers?: number } = {}) => {
    const params = new URLSearchParams({ view });
    if (options.minPapers) params.set("min_papers", String(options.minPapers));
    for (const label of options.labels ?? []) params.append("labels", label);
    return request<NetworkResponse>(`/api/v1/graph/network?${params.toString()}`);
  },

  graphQuery: (question: string) =>
    request<GraphQueryResponse>("/api/v1/graph/query", { method: "POST", body: { question } }),

  comparison: (paperIds: string[] = []) =>
    request<ComparisonResponse>("/api/v1/insights/comparison", {
      method: "POST",
      body: { paper_ids: paperIds },
    }),

  paperMatrix: (paperIds: string[] = []) =>
    request<{ rows: PaperMatrixRow[] }>("/api/v1/insights/matrix", {
      method: "POST",
      body: { paper_ids: paperIds },
    }),

  trends: (topic: string, paperIds: string[] = []) =>
    request<TrendResponse>("/api/v1/insights/trends", {
      method: "POST",
      body: { topic, paper_ids: paperIds },
    }),

  consistency: (paperIds: string[] = [], includeClaims = true) =>
    request<ConsistencyResponse>("/api/v1/insights/consistency", {
      method: "POST",
      body: { paper_ids: paperIds, include_claims: includeClaims },
    }),

  gaps: (topic: string, paperIds: string[] = []) =>
    request<GapResponse>("/api/v1/insights/gaps", {
      method: "POST",
      body: { topic, paper_ids: paperIds },
    }),

  review: (topic: string, paperIds: string[] = [], maxSections = 6) =>
    request<ReviewResponse>("/api/v1/insights/review", {
      method: "POST",
      body: { topic, paper_ids: paperIds, max_sections: maxSections },
    }),
};

/**
 * Stream a research run as server-sent events.
 *
 * Uses fetch rather than EventSource because EventSource cannot send an
 * Authorization header or issue a POST, both of which this endpoint needs.
 */
export async function* streamResearch(
  question: string,
  options: { filters?: SearchFilters; signal?: AbortSignal } = {},
): AsyncGenerator<ResearchEvent> {
  const headers = new Headers({ "Content-Type": "application/json" });
  const token = tokenStorage.get();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(`${API_BASE_URL}/api/v1/research/stream`, {
    method: "POST",
    headers,
    body: JSON.stringify({ question, filters: options.filters ?? {} }),
    signal: options.signal,
  });

  if (!response.ok || !response.body) {
    throw new ApiError(`Research stream failed with status ${response.status}`, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    // SSE frames are separated by a blank line; a partial frame stays in the
    // buffer until the rest arrives.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const line = frame
        .split("\n")
        .find((l) => l.startsWith("data:"));
      if (!line) continue;

      const payload = line.slice(5).trim();
      if (!payload || payload === "[DONE]") continue;

      try {
        yield JSON.parse(payload) as ResearchEvent;
      } catch {
        // A malformed frame shouldn't kill the stream.
      }
    }
  }
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/health", { auth: false, signal });
}
