const ENV = import.meta.env ?? {};

export const API_BASE_URL = String(ENV.VITE_API_URL || "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, status, details = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

function csrfToken() {
  if (typeof document === "undefined") return "";
  const entry = document.cookie
    .split("; ")
    .find((item) => item.startsWith("leaflight_csrf="));
  return entry ? decodeURIComponent(entry.slice("leaflight_csrf=".length)) : "";
}

function messageFromPayload(payload, defaultMessage) {
  if (typeof payload?.detail === "string") return payload.detail;
  if (Array.isArray(payload?.detail)) return "Request validation failed.";
  return defaultMessage;
}

async function apiRequest(path, options = {}) {
  const method = String(options.method || "GET").toUpperCase();
  const headers = new Headers(options.headers || {});
  if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
    const token = csrfToken();
    if (token) headers.set("X-CSRF-Token", token);
  }
  const controller = new AbortController();
  const timeout = globalThis.setTimeout(() => controller.abort(), options.timeout ?? 30000);
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      method,
      headers,
      credentials: "include",
      signal: controller.signal,
    });
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new ApiError("The server took too long to respond.", 0);
    }
    throw new ApiError("Cannot reach the Leaflight API.", 0);
  } finally {
    globalThis.clearTimeout(timeout);
  }
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    const error = new ApiError(
      messageFromPayload(payload, `Request failed with status ${response.status}.`),
      response.status,
      payload,
    );
    if (
      response.status === 401
      && path !== "/auth/session"
      && typeof window !== "undefined"
    ) {
      window.dispatchEvent(new CustomEvent("leaflight:unauthorized"));
    }
    throw error;
  }
  return response.status === 204 ? null : payload;
}

export function getGoogleLoginUrl(returnTo = "/") {
  return `${API_BASE_URL}/auth/google/login?return_to=${encodeURIComponent(returnTo)}`;
}

export const getAuthConfig = () => apiRequest("/auth/config");

export async function getSession() {
  try {
    return await apiRequest("/auth/session");
  } catch (error) {
    if (error.status === 401) return null;
    throw error;
  }
}

export const logout = () => apiRequest("/auth/logout", { method: "POST" });
export const getDashboard = () => apiRequest("/dashboard");

export async function predictDisease(file) {
  const formData = new FormData();
  formData.append("file", file);
  return apiRequest("/predict", { method: "POST", body: formData, timeout: 60000 });
}

export const getDiseaseInfo = (className) =>
  apiRequest(`/disease/${encodeURIComponent(className)}`);

export function getHistory(options = {}) {
  const normalized = typeof options === "number" ? { limit: options } : options;
  const params = new URLSearchParams();
  params.set("limit", String(normalized.limit ?? 50));
  params.set("offset", String(normalized.offset ?? 0));
  if (normalized.search) params.set("search", normalized.search);
  return apiRequest(`/history?${params}`);
}

export const getHealth = () => apiRequest("/health");

export const sendFeedback = (payload) =>
  apiRequest("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
