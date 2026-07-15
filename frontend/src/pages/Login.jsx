import { useEffect, useState } from "react";

import { getAuthConfig, getGoogleLoginUrl } from "../services/api.js";

function Login({ oauthError = "" }) {
  const [config, setConfig] = useState(null);
  const [error, setError] = useState(oauthError);

  useEffect(() => {
    let active = true;
    getAuthConfig()
      .then((value) => active && setConfig(value))
      .catch((requestError) => active && setError(requestError.message));
    return () => { active = false; };
  }, []);

  const startLogin = () => {
    window.location.assign(getGoogleLoginUrl("/"));
  };

  return (
    <main className="login-layout">
      <section className="login-card" aria-labelledby="login-title">
        <div className="brand-mark" aria-hidden="true">LL</div>
        <p className="eyebrow">Leaflight</p>
        <h1 id="login-title">Crop health records that belong to you.</h1>
        <p className="login-copy">
          Sign in to analyze leaf images and keep a private history of model-backed scans.
        </p>
        {error && <div className="error-banner" role="alert">{error}</div>}
        {config === null && !error && <p className="loading-copy">Checking authentication configuration…</p>}
        {config?.configured === false && (
          <div className="setup-notice" role="status">
            Google sign-in needs server configuration. Ask the deployer to set the documented OAuth environment variables.
          </div>
        )}
        <button
          className="google-button"
          type="button"
          onClick={startLogin}
          disabled={!config?.configured}
        >
          <span className="google-g" aria-hidden="true">G</span>
          Continue with Google
        </button>
        <p className="privacy-copy">Leaflight stores your profile identity and scan records, not Google access tokens.</p>
      </section>
    </main>
  );
}

export default Login;
