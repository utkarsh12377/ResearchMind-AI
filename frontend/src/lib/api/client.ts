import type {
  ApiErrorBody,
  ApiKey,
  ApiKeyCreated,
  LoginPayload,
  RegisterPayload,
  Token,
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
};

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/health", { auth: false, signal });
}
