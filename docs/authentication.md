# Google authentication setup

Leaflight uses Google's OAuth 2.0 authorization-code flow in the FastAPI backend. The backend exchanges the one-time code, reads the verified OpenID Connect profile, discards the Google access token, and creates an opaque server-side session. The browser receives an `HttpOnly` session cookie plus a separate CSRF cookie. Session tokens and CSRF tokens are HMAC-hashed in the configured database; raw tokens are never stored.

## Google Cloud configuration

1. In Google Cloud Console, create or select a project.
2. Open **Google Auth Platform**, configure the consent screen, and supply the application name, support email, and developer contact. Choose Internal only for an eligible Google Workspace organization; otherwise choose External and add test users while the application is in testing.
3. Request only the `openid`, `email`, and `profile` scopes. Leaflight does not request offline access.
4. Create an OAuth client with application type **Web application**.
5. Add this local authorized JavaScript origin:

   ```text
   http://127.0.0.1:5173
   ```

6. Add this exact local authorized redirect URI:

   ```text
   http://127.0.0.1:8000/auth/google/callback
   ```

7. For production, add this exact origin and callback:

   ```text
   Authorized JavaScript origin: https://crop-disease-lime.vercel.app
   Authorized redirect URI: https://crop-disease-lime.vercel.app/api/auth/google/callback
   ```

The callback scheme, host, port, path, case, and trailing slash must match `OAUTH_CALLBACK_URL` exactly. Do not add wildcards. Keep the downloaded client configuration and client secret outside the repository.

## Local environment

The backend automatically loads `backend/.env` before constructing application settings. Vite automatically loads `frontend/.env`; that frontend file must contain only `VITE_API_URL`. You may alternatively export the same values in the backend shell:

```powershell
$env:GOOGLE_CLIENT_ID="<google-web-client-id>"
$env:GOOGLE_CLIENT_SECRET="<google-web-client-secret>"
$env:AUTH_SECRET="<at-least-32-random-characters>"
$env:APP_URL="http://127.0.0.1:5173"
$env:OAUTH_CALLBACK_URL="http://127.0.0.1:8000/auth/google/callback"
$env:CORS_ORIGINS="http://127.0.0.1:5173"
$env:COOKIE_SECURE="false"
$env:COOKIE_SAMESITE="lax"
```

Generate `AUTH_SECRET` with a cryptographically secure secret manager or password generator. Never reuse the Google client secret as the application secret.

For the Vite shell, either point requests at the API:

```powershell
$env:VITE_API_URL="http://127.0.0.1:8000"
npm.cmd --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```

or use the development proxy:

```powershell
$env:VITE_DEV_API_TARGET="http://127.0.0.1:8000"
npm.cmd --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```

## Production environment

Production uses Vercel's same-origin `/api` rewrite to Render. Set `APP_URL` and `CORS_ORIGINS` to `https://crop-disease-lime.vercel.app`, set `OAUTH_CALLBACK_URL` to `https://crop-disease-lime.vercel.app/api/auth/google/callback`, leave `COOKIE_SECURE=true` and `COOKIE_SAMESITE=lax`, and leave `COOKIE_DOMAIN` unset. Render uses an SSL Supabase PostgreSQL `DATABASE_URL`; it never uses SQLite or Render's ephemeral filesystem for persistence.

Sessions expire after `SESSION_TTL_HOURS` (168 hours by default). Logout revokes the database session immediately. Expired, revoked, missing, or altered cookies produce HTTP 401. State-changing routes additionally require the matching `X-CSRF-Token` header; a mismatched token produces HTTP 403.

## Failure handling

- Missing OAuth variables: `/auth/config` reports `configured: false`, and `/auth/google/login` returns HTTP 503 with the required variable names.
- Cancelled consent: the user returns to the login screen with a cancellation message.
- Invalid, consumed, or expired state: the callback clears the state cookie, creates no session, and returns to the login screen with a fresh-sign-in message.
- Unverified identity profile: the callback creates no session and returns to the login screen with an account error.
- Duplicate email with a different Google subject: the callback does not silently link accounts and returns to the login screen with an account error.
- Google or database failure: the callback creates no session and returns to the login screen with a retryable provider error.

Tests mock Google's token and user-info responses; they never contact Google. Production must additionally be verified with a live callback through the exact registered Vercel URL.
