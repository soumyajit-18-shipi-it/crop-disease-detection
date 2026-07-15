export const initialAuthState = {
  activePage: "dashboard",
  session: undefined,
  error: "",
};

export function authStateReducer(state, action) {
  switch (action.type) {
    case "session-restored":
      return { ...state, session: action.session, error: "" };
    case "session-failed":
      return { ...state, session: null, error: action.error };
    case "navigate":
      return { ...state, activePage: action.page };
    case "logged-out":
      return {
        activePage: "dashboard",
        session: null,
        error: action.error || "",
      };
    case "unauthorized":
      return { activePage: "dashboard", session: null, error: "" };
    default:
      return state;
  }
}

export const oauthErrorMessages = {
  account: "This Google account could not be linked to the existing Leaflight account.",
  cancelled: "Google sign-in was cancelled.",
  missing_code: "Google did not complete sign-in. Please try again.",
  provider: "Google sign-in could not be completed. Please try again.",
  state: "The sign-in request expired or was already used. Please start a new sign-in.",
};

export function oauthErrorMessage(code) {
  if (!code) return "";
  return oauthErrorMessages[code] || "Google sign-in could not be completed. Please try again.";
}
