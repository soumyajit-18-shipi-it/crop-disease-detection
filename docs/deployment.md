# Production deployment

Leaflight's production architecture is:

- Frontend: Vercel at `https://crop-disease-lime.vercel.app`
- API: Render at `https://crop-disease-api-xcz2.onrender.com`
- Database: Supabase PostgreSQL project `imzoacxlzqrdheftlkhl`
- Authentication: the backend's Google OAuth code flow, opaque server sessions, and double-submit CSRF
- Inference: immutable EfficientNetV2-S ONNX release `v1`

The requested `https://crop-disease.vercel.app` alias is owned by another Vercel account. Vercel rejected the alias assignment, so the service uses the account-owned `https://crop-disease-lime.vercel.app` alias. Do not configure OAuth or backend trust for the unowned hostname.

The frontend uses a same-origin Vercel rewrite. Browser requests go to `/api/*`, which Vercel proxies to Render. This keeps the session and CSRF cookies first-party on `crop-disease-lime.vercel.app` and allows `Secure; SameSite=Lax; Path=/` cookies. The API rewrite is listed before the SPA fallback in `frontend/vercel.json`.

## Local setup

Copy the tracked examples to the ignored real files and fill only local values:

```powershell
Copy-Item backend/.env.example backend/.env
Copy-Item frontend/.env.example frontend/.env
python -m pip install -r backend/requirements.txt
npm.cmd --prefix frontend ci
python scripts/download_model.py --verify-only
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
npm.cmd --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```

Local development uses only these origins:

```text
Frontend: http://127.0.0.1:5173
Backend: http://127.0.0.1:8000
Google callback: http://127.0.0.1:8000/auth/google/callback
```

Set `DATABASE_URL=sqlite:///backend/db/disease_info.db` only for local development or tests. `ENVIRONMENT=production` rejects SQLite and a missing or invalid `DATABASE_URL` fails startup.

## Supabase migrations and backup

The production schema is in `supabase/migrations/`. It uses UUID identities, `timestamp with time zone`, JSONB quality warnings, foreign keys, case-insensitive email uniqueness, session expiry indexes, and atomic OAuth-state deletion. Browser database roles have no table privileges; the backend connects over SSL.

```powershell
npx.cmd --yes supabase@latest login
npx.cmd --yes supabase@latest link --project-ref imzoacxlzqrdheftlkhl
npx.cmd --yes supabase@latest db push --linked --dry-run
npx.cmd --yes supabase@latest db push --linked
npx.cmd --yes supabase@latest migration list --linked
npx.cmd --yes supabase@latest db lint --linked --schema public --fail-on error
```

Supply the database password through the CLI prompt or an ephemeral environment variable; never add it to a command transcript or source file. Do not use `db reset` against the hosted project.

Create an online SQLite backup before a one-time transfer:

```powershell
python -c "import sqlite3; s=sqlite3.connect('backend/db/disease_info.db'); d=sqlite3.connect('backups/disease_info.pre-production.sqlite3'); s.backup(d); d.close(); s.close()"
```

With `DATABASE_URL` set to the SSL PostgreSQL target, migrate genuine records idempotently:

```powershell
python scripts/migrate_sqlite_to_postgres.py --source backups/disease_info.pre-production.sqlite3
```

The utility migrates genuine Google users and their owned scans/feedback. It excludes sessions, OAuth states, unowned legacy rows, and obvious fixtures. Keep the original ignored backup. For a PostgreSQL backup, use Supabase's scheduled backups where available or `pg_dump` with an SSL connection into an encrypted, access-controlled location.

## Render deployment

The root `Dockerfile` pins Python, installs the PostgreSQL driver, downloads the immutable model only when absent, verifies the pinned size and SHA-256 during the build, and runs as an unprivileged user. The runtime command binds to `0.0.0.0:$PORT`. Render health checks use `/health`.

```powershell
render login
render workspace set <workspace-id>
render blueprints validate ./render.yaml
render services create `
  --name crop-disease-api `
  --type web_service `
  --repo https://github.com/soumyajit-18-shipi-it/crop-disease-detection `
  --runtime docker `
  --branch main `
  --plan free `
  --region singapore `
  --health-check-path /health `
  --output json --confirm
```

Set these backend variables through Render's secret environment controls, never plaintext in `render.yaml`:

```text
ENVIRONMENT=production
DATABASE_URL=<supavisor-session-pooler-url-with-ssl>
GOOGLE_CLIENT_ID=<existing-web-client-id>
GOOGLE_CLIENT_SECRET=<existing-web-client-secret>
AUTH_SECRET=<at-least-64-random-characters>
APP_URL=https://crop-disease-lime.vercel.app
CORS_ORIGINS=https://crop-disease-lime.vercel.app
OAUTH_CALLBACK_URL=https://crop-disease-lime.vercel.app/api/auth/google/callback
COOKIE_SECURE=true
COOKIE_SAMESITE=lax
LOG_TO_FILE=false
MAX_UPLOAD_SIZE_MB=10
DATABASE_POOL_MIN_SIZE=1
DATABASE_POOL_MAX_SIZE=5
DATABASE_CONNECT_TIMEOUT_SECONDS=15
FORWARDED_ALLOW_IPS=*
```

The direct Supabase endpoint is IPv6-only for this free project. Render uses the Singapore Supavisor session pooler on port 5432, with SSL required. Do not expose a service-role key to the frontend.

## Vercel deployment

Link the `frontend` directory to the exact Vercel project, set only the safe frontend API value, and deploy production:

```powershell
vercel login
vercel link --cwd frontend
vercel env add VITE_API_URL production --cwd frontend
vercel env ls --cwd frontend
vercel --prod --cwd frontend
```

The production value is `VITE_API_URL=/api`. Framework is Vite, install command is `npm ci`, build command is `npm run build`, and output is `dist`. Backend credentials, database URLs, OAuth secrets, and session material must never be Vercel frontend variables.

## Google OAuth

The existing Google OAuth Web Application must contain these exact production values:

```text
Authorized JavaScript origin: https://crop-disease-lime.vercel.app
Authorized redirect URI: https://crop-disease-lime.vercel.app/api/auth/google/callback
```

Keep the `127.0.0.1` development values if local sign-in is still needed. Do not add `localhost`, preview domains, or the direct Render callback to this architecture.

## Verification

```powershell
python -m pytest -q
npm.cmd --prefix frontend test
npm.cmd --prefix frontend run build
npx.cmd --yes supabase@latest migration list --linked
npx.cmd --yes supabase@latest db lint --linked --schema public --fail-on error
git diff --check
```

Verify live `/health`, `/classes`, `/auth/config`, Google login/refresh/logout repeated three times, a real authenticated image prediction with `mock=false`, user-owned dashboard/history updates, protected API `401` after logout, and no credentials in the frontend bundle or logs.

Uploaded images are validated and held only in memory for inference. The application stores SHA-256 and scan metadata, not image paths or binaries, so no Supabase Storage bucket is required. If image retention is added later, use a private user-scoped bucket and short-lived signed URLs before storing any path.

## Rollback

1. In Vercel, promote the prior healthy production deployment or redeploy its Git commit.
2. In Render, deploy the prior healthy commit and wait for `/health` before restoring traffic.
3. Do not reset Supabase. Database migrations are forward-only; apply a reviewed compensating migration if schema rollback is necessary.
4. Restore data from a verified Supabase/`pg_dump` backup only after taking a fresh backup of the current state and validating row counts and foreign keys in a separate recovery project.
5. Rotate any credential suspected of exposure and update Render before redeploying. Never place recovered secrets in Git or Vercel variables.
