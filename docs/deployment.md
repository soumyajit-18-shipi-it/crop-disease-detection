# Reproducible deployment

Leaflight serves the `efficientnetv2_s_v1` release with backend API version `1.0.0` or newer. Its 80,679,586-byte ONNX binary is intentionally excluded from normal Git history. Git tracks the release manifest, serving metadata, evaluation metrics, and checksum file, while an explicit setup step obtains the immutable binary and verifies it before use.

The active ONNX SHA-256 is:

```text
bd0af61cba3bcc83a59d93348e6e43a539c6b60069203d7ee9d4ee746810beaa
```

The manifest contains the immutable GitHub Release asset URL:

```text
https://github.com/soumyajit-18-shipi-it/crop-disease-detection/releases/download/model-efficientnetv2-s-v1/model.onnx
```

GitHub release immutability locks the published tag and asset. `LEAFLIGHT_MODEL_URL` remains available as an explicit mirror override; do not configure it to a mutable `latest` URL.

## Clean local setup

Leaflight inference uses Python 3.11. The Docker image pins Python 3.11.15; local Python 3.11 is recommended.

```powershell
git clone <repository>
cd crop-disease-detection
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\.venv\Scripts\python.exe scripts/download_model.py
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The URL placeholder above must be replaced with the published asset URL. It is not a repository default.

In another shell, verify:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/classes
```

Prediction is intentionally authenticated. Complete Google sign-in in the frontend, then upload the image there so the scan is associated with that user. The clean-clone verifier uses a session created only inside its temporary test database; there is no runtime authentication bypass.

The downloader supports all release setup modes:

```powershell
# URL from the environment
python scripts/download_model.py

# One-time explicit URL override
python scripts/download_model.py --url "<real-versioned-model-url>"

# No network access; verify the complete local release
python scripts/download_model.py --verify-only
```

If a valid model already exists, the first two commands reuse it without contacting the URL. If it is missing or corrupt, the downloader streams to a temporary file, enforces the manifest size limit, checks SHA-256, and atomically installs it. Failed or interrupted downloads do not replace an existing file.

## Backend configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/releases/efficientnetv2_s_v1/model.onnx` | ONNX file selected for serving |
| `MODEL_METADATA_PATH` | `models/releases/efficientnetv2_s_v1/model.json` | Serving contract for the same release |
| `MODEL_RELEASE_MANIFEST` | `models/releases/efficientnetv2_s_v1/release.json` | Checksums and immutable release identity |
| `LEAFLIGHT_MODEL_URL` | unset | Optional override for the manifest's immutable ONNX URL |
| `DB_PATH` | `backend/db/disease_info.db` | SQLite database file |
| `LOG_DIR` | `backend/logs` | Rotating request logs |
| `CORS_ORIGINS` | local Vite origins | Comma-separated browser origins |
| `MAX_UPLOAD_SIZE_MB` | `10` | Image upload limit |
| `PORT` | `8000` in Docker | Listening port used by the container command |
| `GOOGLE_CLIENT_ID` | unset | Google OAuth Web client ID |
| `GOOGLE_CLIENT_SECRET` | unset | Google OAuth client secret; backend only |
| `AUTH_SECRET` | unset | HMAC secret, at least 32 random characters |
| `APP_URL` | unset | Trusted frontend URL used after OAuth |
| `OAUTH_CALLBACK_URL` | unset | Exact Google-registered callback URL |
| `COOKIE_SECURE` | `true` | Require HTTPS for authentication cookies |
| `COOKIE_SAMESITE` | `lax` | Session cookie same-site policy |
| `SESSION_TTL_HOURS` | `168` | Server-side session lifetime |

The API never downloads a model at startup. It verifies the manifest, checksum file, metadata, metrics, ONNX size, ONNX SHA-256, metadata/manifest compatibility, and ONNX graph contract. A missing or invalid bundle leaves `/health` at HTTP 503 and prediction endpoints fail closed. The startup log tells the operator to run `python scripts/download_model.py`.

## Docker: verified local model before build

Docker uses Method A: obtain and verify the model on the build host, then build from the repository root. The Dockerfile runs verification again inside the build stage. Private URLs and credentials are therefore not passed as build arguments or stored in image layers.

```powershell
python scripts/download_model.py
python scripts/download_model.py --verify-only
docker build -f backend/Dockerfile -t leaflight-api:efficientnetv2-s-v1 .
docker volume create leaflight-data
docker run --rm --name leaflight-api -p 8000:8000 `
  -e PORT=8000 `
  -e CORS_ORIGINS=http://127.0.0.1:5173 `
  -e GOOGLE_CLIENT_ID="<google-web-client-id>" `
  -e GOOGLE_CLIENT_SECRET="<google-web-client-secret>" `
  -e AUTH_SECRET="<at-least-32-random-characters>" `
  -e APP_URL="http://127.0.0.1:5173" `
  -e OAUTH_CALLBACK_URL="http://127.0.0.1:8000/auth/google/callback" `
  -e COOKIE_SECURE=false `
  -e COOKIE_SAMESITE=lax `
  -v leaflight-data:/data `
  leaflight-api:efficientnetv2-s-v1
```

The image runs as the unprivileged `leaflight` user, reports container health through `GET /health`, accepts a platform-supplied `PORT`, and stores SQLite at `/data/disease_info.db`. Mount `/data` as a persistent volume in production. Do not mount over `/app/models`, because the verified release is packaged there.

The build context excludes Git data, virtual environments, frontend dependencies/build output, datasets, logs, databases, temporary files, training artifacts, checkpoints, legacy models, and unrelated model releases. Only the active verified release is eligible for inclusion.

## Railway, Render, or another container platform

1. Run the model download and verification step in a trusted packaging job. Configure `LEAFLIGHT_MODEL_URL` only when using an approved mirror.
2. Build from the repository root with `backend/Dockerfile`.
3. Persist `/data` using the platform's volume feature.
4. Set `CORS_ORIGINS` to the deployed frontend origins. The platform may set `PORT` dynamically.
5. Require `GET /health` to return HTTP 200 before routing traffic.
6. Configure Google OAuth and cookie settings from `docs/authentication.md`. Use HTTPS and `COOKIE_SECURE=true` in production.

For the frontend on Vercel, use `frontend` as the project root, set `VITE_API_URL` to the deployed backend origin, run `npm run build`, and publish `dist`.

## Offline clean-clone simulation

The workflow can also be tested without external network access by serving the existing verified ONNX file from the source checkout to a temporary clean staging copy:

```powershell
python scripts/verify_clean_clone.py
```

The script copies only Git-visible files to a temporary directory (the ignored ONNX file is not copied), starts a temporary loopback HTTP server for the existing verified model, downloads and verifies it in the staging copy, starts the real API with a temporary SQLite database, and checks `GET /health`, `GET /classes`, and an authenticated `POST /predict`. This local test is independent of GitHub availability.

## Troubleshooting

| Symptom | Resolution |
|---|---|
| Model is missing | Run `python scripts/download_model.py`. Use `LEAFLIGHT_MODEL_URL` or `--url` only for an alternate versioned mirror. |
| Checksum or size mismatch | Confirm the URL points to the exact `v1` asset. Do not edit the ONNX file or weaken verification. Correct the published asset/URL and rerun the downloader. |
| Wrong release version | Keep `MODEL_PATH`, `MODEL_METADATA_PATH`, and `MODEL_RELEASE_MANIFEST` in one versioned directory. Use `--release <release-name>` for another published manifest. |
| Download fails | Check DNS/TLS/network access, HTTP authorization supplied by the hosting mechanism, URL expiry, and the `--connect-timeout`/`--read-timeout` settings. The repository does not store credentials. |
| Metadata mismatch | Restore the tracked metadata and manifest for the selected release. Never pair ONNX and metadata files from different exports. |
| `/health` returns 503 | Read the backend startup error, run `python scripts/download_model.py --verify-only`, and verify the persistent database path is writable. |
| Google button is disabled | Set all five required OAuth/application variables and restart the API. Inspect `GET /auth/config`. |
| OAuth `redirect_uri_mismatch` | Register the exact `OAUTH_CALLBACK_URL`, including scheme, host, port, path, case, and trailing slash. |
| Session immediately returns 401 | Keep the SQLite volume persistent, use a stable `AUTH_SECRET`, and ensure the cookie reaches the API origin. |
| Mutating request returns 403 | Restore a session so the CSRF cookie is present; the frontend sends it as `X-CSRF-Token`. |

See `docs/authentication.md` for consent-screen setup, local and production origins, the exact callback, secure-cookie deployment guidance, and auth failure handling.

## Publishing a new model release

Publishing is an explicit promotion step; training output is never activated automatically.

1. Create a new `models/releases/<release-name>/` directory from a completed, validated, ONNX-parity-passing export.
2. Copy only `model.json`, `metrics.json`, `checksum.sha256`, and `release.json` into the Git-visible release structure. Keep `model.onnx` ignored.
3. Calculate exact hashes and size without modifying the assets:

```powershell
Get-FileHash -Algorithm SHA256 models/releases/<release-name>/model.onnx
Get-FileHash -Algorithm SHA256 models/releases/<release-name>/model.json
Get-FileHash -Algorithm SHA256 models/releases/<release-name>/metrics.json
(Get-Item models/releases/<release-name>/model.onnx).Length
```

4. Populate the new manifest with those values, release identity, backend compatibility, validation state, and parity state. Keep class mappings in `model.json` rather than duplicating them.
5. Upload `model.onnx` to an access-controlled or public artifact service under an immutable versioned URL. Configure credentials through the hosting/deployment platform, never Git or the URL itself.
6. Configure and verify the real asset:

```powershell
$env:LEAFLIGHT_MODEL_URL="<real-versioned-model-url>"
python scripts/download_model.py --release <release-name>
python scripts/download_model.py --release <release-name> --verify-only
```

7. Run backend tests, a known-image prediction, the clean-clone simulation, and the Docker smoke test. Only then update application defaults in a reviewed change if the new release is intended to become active.
