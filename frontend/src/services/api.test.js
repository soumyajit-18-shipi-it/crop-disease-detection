import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { authStateReducer, oauthErrorMessage } from "../authState.js";
import {
  ApiError,
  getDashboard,
  getGoogleLoginUrl,
  getSession,
  getHistory,
  logout,
  predictDisease,
} from "./api.js";


function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  });
}


test("dashboard retrieval returns backend statistics without generated values", async () => {
  const expected = {
    total_scans: 0,
    healthy_scans: 0,
    diseased_scans: 0,
    low_confidence_scans: 0,
    average_confidence: null,
    healthy_percentage: null,
    active_disease_classes: 0,
    latest_scan_at: null,
    disease_distribution: [],
    recent_scans: [],
  };
  globalThis.fetch = async (_url, options) => {
    assert.equal(options.credentials, "include");
    assert.equal(options.method, "GET");
    return jsonResponse(expected);
  };
  assert.deepEqual(await getDashboard(), expected);
});


test("history search is encoded into the real API request", async () => {
  globalThis.fetch = async (url) => {
    assert.match(String(url), /history\?limit=25&offset=0&search=Tomato\+Late/);
    return jsonResponse([]);
  };
  assert.deepEqual(await getHistory({ limit: 25, search: "Tomato Late" }), []);
});


test("prediction sends multipart data and the CSRF token", async () => {
  globalThis.document = { cookie: "leaflight_csrf=csrf-test-token" };
  const expected = { scan_id: 9, class_name: "Tomato_healthy", confidence: 0.98 };
  globalThis.fetch = async (url, options) => {
    assert.match(String(url), /\/predict$/);
    assert.equal(options.method, "POST");
    assert.ok(options.body instanceof FormData);
    assert.equal(options.headers.get("X-CSRF-Token"), "csrf-test-token");
    return jsonResponse(expected);
  };
  const image = new Blob([new Uint8Array([1, 2, 3])], { type: "image/jpeg" });
  assert.deepEqual(await predictDisease(image), expected);
});


test("backend failure raises an error and never returns generated dashboard data", async () => {
  globalThis.fetch = async () => jsonResponse({ detail: "Database unavailable." }, 503);
  await assert.rejects(getDashboard(), (error) => {
    assert.ok(error instanceof ApiError);
    assert.equal(error.status, 503);
    assert.equal(error.message, "Database unavailable.");
    return true;
  });
});


test("missing or expired session restores an unauthenticated state", async () => {
  globalThis.fetch = async () => jsonResponse({ detail: "Session is invalid or expired." }, 401);
  assert.equal(await getSession(), null);
});


test("logout sends only the current CSRF cookie and does not cache it", async () => {
  const seenTokens = [];
  globalThis.document = { cookie: "leaflight_csrf=first-session-csrf" };
  globalThis.fetch = async (_url, options) => {
    seenTokens.push(options.headers.get("X-CSRF-Token"));
    return new Response(null, { status: 204 });
  };

  await logout();
  globalThis.document.cookie = "";
  await logout();

  assert.deepEqual(seenTokens, ["first-session-csrf", null]);
});


test("expected session 401 does not redirect or retry", async () => {
  let requestCount = 0;
  let unauthorizedEvents = 0;
  globalThis.window = { dispatchEvent: () => { unauthorizedEvents += 1; } };
  globalThis.CustomEvent = class CustomEvent {
    constructor(type) { this.type = type; }
  };
  globalThis.fetch = async () => {
    requestCount += 1;
    return jsonResponse({ detail: "Authentication required." }, 401);
  };

  assert.equal(await getSession(), null);
  assert.equal(requestCount, 1);
  assert.equal(unauthorizedEvents, 0);

  await assert.rejects(getDashboard(), (error) => error.status === 401);
  assert.equal(requestCount, 2);
  assert.equal(unauthorizedEvents, 1);
});


test("frontend auth state fully resets after logout", () => {
  const authenticated = {
    activePage: "history",
    session: { user: { email: "farmer@example.test" } },
    error: "stale error",
  };
  assert.deepEqual(
    authStateReducer(authenticated, { type: "logged-out" }),
    { activePage: "dashboard", session: null, error: "" },
  );
  assert.deepEqual(
    authStateReducer(authenticated, { type: "unauthorized" }),
    { activePage: "dashboard", session: null, error: "" },
  );
});


test("OAuth callback errors have clear browser-visible messages", () => {
  assert.match(oauthErrorMessage("state"), /expired or was already used/i);
  assert.match(oauthErrorMessage("provider"), /try again/i);
  assert.match(oauthErrorMessage("unexpected"), /could not be completed/i);
});


test("both login attempts use the configured 127 backend origin", async () => {
  const envExample = await readFile(new URL("../../.env.example", import.meta.url), "utf8");
  const configuredOrigin = envExample
    .split(/\r?\n/)
    .find((line) => line.startsWith("VITE_API_URL="))
    .split("=", 2)[1];
  assert.equal(configuredOrigin, "http://127.0.0.1:8000");
  assert.equal(new URL(getGoogleLoginUrl("/"), configuredOrigin).origin, configuredOrigin);
  assert.equal(new URL(getGoogleLoginUrl("/"), configuredOrigin).origin, configuredOrigin);
  assert.ok(!envExample.includes("localhost"));
});
