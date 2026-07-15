import { useEffect, useReducer } from "react";

import { authStateReducer, initialAuthState, oauthErrorMessage } from "./authState.js";
import GlobalHeader from "./components/GlobalHeader.jsx";
import Sidebar from "./components/Sidebar.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import History from "./pages/History.jsx";
import Login from "./pages/Login.jsx";
import Profile from "./pages/Profile.jsx";
import Scan from "./pages/Scan.jsx";
import { getSession, logout } from "./services/api.js";

const pageTitles = {
  dashboard: "Dashboard",
  scan: "New scan",
  history: "Scan history",
  profile: "Profile",
};

function App() {
  const [{ activePage, session, error }, dispatch] = useReducer(
    authStateReducer,
    initialAuthState,
  );

  const clearAuthQuery = () => {
    const url = new URL(window.location.href);
    url.searchParams.delete("auth");
    url.searchParams.delete("auth_error");
    window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
  };

  useEffect(() => {
    let active = true;
    getSession()
      .then((value) => {
        if (!active) return;
        dispatch({ type: "session-restored", session: value });
        if (value) clearAuthQuery();
      })
      .catch((requestError) => {
        if (active) {
          dispatch({ type: "session-failed", error: requestError.message });
        }
      });
    const unauthorized = () => dispatch({ type: "unauthorized" });
    window.addEventListener("leaflight:unauthorized", unauthorized);
    return () => {
      active = false;
      window.removeEventListener("leaflight:unauthorized", unauthorized);
    };
  }, []);

  const handleLogout = async () => {
    let logoutError = "";
    try {
      await logout();
    } catch (requestError) {
      if (requestError.status !== 401) logoutError = requestError.message;
    } finally {
      clearAuthQuery();
      dispatch({ type: "logged-out", error: logoutError });
    }
  };

  if (session === undefined) {
    return <main className="center-state" role="status">Restoring your secure session…</main>;
  }

  if (!session) {
    const oauthError = oauthErrorMessage(
      new URLSearchParams(window.location.search).get("auth_error"),
    );
    return <Login oauthError={oauthError || error} />;
  }

  const pages = {
    dashboard: (
      <Dashboard
        user={session.user}
        onNavigate={(page) => dispatch({ type: "navigate", page })}
      />
    ),
    scan: <Scan />,
    history: <History />,
    profile: <Profile user={session.user} sessionExpiresAt={session.expires_at} />,
  };

  return (
    <div className="app-shell">
      <GlobalHeader user={session.user} title={pageTitles[activePage]} onLogout={handleLogout} />
      <Sidebar
        activePage={activePage}
        onNavigate={(page) => dispatch({ type: "navigate", page })}
      />
      {pages[activePage]}
    </div>
  );
}

export default App;
