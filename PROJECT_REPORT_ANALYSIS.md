# Leaflight Crop Disease Detection

## Formal Project Report Analysis

**Audit date:** 16 July 2026  
**Repository:** crop-disease-detection  
**Audit mode:** Read-only inspection followed by creation of this report only  
**Live-service status:** Needs manual verification; this report does not assume that a public deployment is reachable

### Evidence labels and method

- **Confirmed from code** means a value or behavior is directly implemented in a tracked source or configuration file.
- **Confirmed from generated files or logs** means it is present in a local model artifact, metric file, manifest, database, screenshot, or log.
- **Calculated during this audit** means a read-only script counted files, decoded images, computed hashes, parsed syntax, or summarized an existing artifact.
- **Estimated** identifies a reasoned interpretation rather than a stored fact.
- **Not available** or **Not found in the repository** is used when no defensible value exists.

The audit covered all 134 version-controlled files and the ignored local datasets, manifests, model bundles, checkpoints, databases, backups, logs, and training artifacts. Git internals, virtual environments, dependency trees, caches, and generated build output were excluded from source statistics. Image checks were non-destructive and included decoding, hashes, dimensions, color mode, split leakage, and path existence. Database inspection was read-only and reports only schema and aggregate counts. Secret files and real environment files were inspected only for variable names; no values are reproduced.

**Evidence:** .gitignore, .dockerignore, data/, artifacts/, models/, backend/db/disease_info.db, backups/

---

## 1. Project Overview

### Project identity

| Item | Finding | Confidence and evidence |
|---|---|---|
| Project name | **Leaflight Crop Disease Detection** | Confirmed from code and documentation. Evidence: README.md, frontend/src/pages/Login.jsx, frontend/package.json |
| Project type | Full-stack authenticated crop-leaf image classification and scan-history system, plus a reproducible ML pipeline | Confirmed from code. Evidence: frontend/src/App.jsx, backend/main.py, src/training/train.py |
| ML problem | Single-label, 15-class supervised classification for pepper, potato, and tomato leaf conditions | Confirmed from code and generated files. Evidence: data/class_mapping.json, models/releases/efficientnetv2_s_v1/model.json |
| Active inference release | EfficientNetV2-S v1, ONNX, 300 × 300 RGB input, 15 logits | Confirmed from generated files. Evidence: models/releases/efficientnetv2_s_v1/release.json, models/releases/efficientnetv2_s_v1/model.json |

### Problem statement

Plant disease symptoms can be visually similar, while timely expert diagnosis may be difficult to obtain. Leaflight accepts a crop-leaf photograph, validates its quality, classifies it with a calibrated neural network, and returns the predicted condition, confidence, alternatives, and reviewed guidance where such guidance exists. It also keeps each authenticated user’s scan history and summary statistics.

**Evidence:** backend/api/routes/predict.py, backend/api/model_loader.py, backend/api/routes/system.py, frontend/src/pages/Scan.jsx

### Main objective

The main objective is to provide a reproducible, model-backed decision-support application for early crop-disease screening. It is a classifier and record system, not a laboratory diagnosis or a replacement for an agronomist. A second objective is engineering reproducibility through immutable splits, resumable training, calibrated evaluation, ONNX parity checks, checksummed releases, and fail-closed serving.

**Evidence:** configs/training/phase2_5.yaml, src/training/train.py, src/training/export_onnx.py, src/inference/model_release.py

### Target users

The following are **Estimated** from the implemented workflow:

- Farmers, field workers, and extension personnel needing a preliminary leaf assessment.
- Agriculture students and researchers demonstrating an end-to-end computer-vision system.
- ML engineers comparing backbones and preparing a versioned inference release.
- Reviewers curating field-survey labels before training.

**Evidence:** frontend/src/pages/Scan.jsx, src/data/review_field_survey.py, docs/future_model_roadmap.md

### Real-world use cases

1. Upload or photograph a pepper, potato, or tomato leaf and receive a predicted condition.
2. Identify low-confidence or poor-quality photographs that should be retaken.
3. Review symptoms and treatment information for the six classes with seeded guidance.
4. Review a private scan history and disease distribution for one authenticated account.
5. Normalize and manually validate field-survey labels for a later training iteration.
6. Compare pretrained image backbones and export a parity-tested ONNX release.

**Evidence:** frontend/src/components/UploadCard.jsx, backend/api/routes/predict.py, backend/db/seed_disease_data.py, src/training/benchmark.py

### Current development status

The repository is a **mostly implemented release-engineered MVP**, not a finished research program. The authenticated frontend, API, database abstraction, active ONNX release, validation, prediction persistence, dashboard, history, and core tests are implemented. The comparative Phase 2.5 benchmark is incomplete because ConvNeXt-Base has not been trained and configuration requires every candidate before promotion. Field-survey validation has not begun operationally: all 409 review groups are pending and zero records are eligible. Production configuration exists for Vercel, Render, and Supabase, but live uptime and external OAuth configuration are **Needs manual verification**.

**Evidence:** docs/model_comparison.md, data/manifests/field_survey/validated_manifest.json, render.yaml, frontend/vercel.json

### Major completed and incomplete features

| Feature | Status | Notes | Evidence |
|---|---|---|---|
| Google sign-in and server sessions | Fully implemented | One-use OAuth state, HttpOnly session cookie, CSRF cookie, expiry, revocation | backend/api/routes/auth.py, backend/api/auth.py |
| Single-image prediction | Fully implemented | Real ONNX inference; no mock fallback | backend/api/routes/predict.py, backend/api/model_loader.py |
| Batch prediction | Implemented with limitations | Maximum 10, sequential, not all-or-nothing | backend/api/routes/predict.py |
| Upload security and quality checks | Fully implemented for stated policy | MIME/format, size, dimensions, pixels, EXIF, contrast and warnings | backend/api/routes/predict.py |
| Dashboard and private history | Fully implemented | SQL is scoped by authenticated user ID | backend/api/routes/system.py |
| Disease guidance | Partial | Reviewed content for 6 of 15 classes | backend/db/seed_disease_data.py |
| Feedback persistence | Implemented with validation gaps | Owned scan lookup; free-form fields lack strong constraints | backend/api/routes/system.py, backend/api/schemas.py |
| Reproducible ML pipeline | Fully implemented | Split, augmentation, resume, EMA, calibration, export, comparison | src/training/, src/evaluation/ |
| EfficientNetV2-S release | Complete and active | Metrics, parity, manifest, checksums | models/releases/efficientnetv2_s_v1/ |
| ConvNeXt-Tiny candidate | Complete, not promoted | Retained under generated artifacts | artifacts/training/crop_disease_phase2_5/convnext_tiny/ |
| ConvNeXt-Base | Not started | Blocks formal all-candidate selection | configs/training/phase2_5.yaml, docs/model_comparison.md |
| Field-survey review | Tooling complete; review incomplete | 409 groups pending; zero validated records | data/manifests/field_survey/validated_manifest.json |
| PlantDoc | Optional placeholder | Dataset is not present | configs/training/phase2_5.yaml, data/splits/phase1_split.json |
| CI, E2E, automated accessibility tests | Not found in the repository | No workflow or browser test suite | repository inventory |

---

## 2. Technology Stack

### Frontend technologies

| Name | Version | Purpose | Files where used |
|---|---:|---|---|
| React | Declared ^18.3.1; resolved 18.3.1 | Components, hooks, reducer state | frontend/package.json, frontend/src/App.jsx |
| React DOM | Declared ^18.3.1; resolved 18.3.1 | Browser rendering | frontend/src/main.jsx |
| Vite | Declared ^5.4.11; resolved 5.4.21 | Development server and production build | frontend/vite.config.js |
| Recharts | Declared ^2.15.0; resolved 2.15.4 | Disease distribution chart | frontend/src/components/DiseaseDistributionCard.jsx |
| JavaScript and JSX | ECMAScript modules | Frontend implementation | frontend/src/ |
| Handwritten CSS | Not applicable | Layout, responsive design, motion | frontend/src/styles/index.css |
| Browser Fetch API | Browser-native | JSON and multipart requests, cookies, timeouts | frontend/src/services/api.js |

### Backend technologies

| Name | Version | Purpose | Files where used |
|---|---:|---|---|
| Python | Docker 3.11.15; training artifacts record 3.14.5 | API, data, training, scripts | Dockerfile, backend/, src/ |
| FastAPI | 0.136.3 | HTTP API, dependencies, lifespan | backend/main.py, backend/api/routes/ |
| Uvicorn | 0.48.0 | ASGI server | Dockerfile, README.md |
| Pydantic | 2.12.5 | Settings and API models | backend/config.py, backend/api/schemas.py |
| Pillow | 12.2.0 backend pin | Decode, EXIF, verification | backend/api/routes/predict.py |
| NumPy | 2.4.4 backend pin | Tensors, softmax, image statistics | backend/api/model_loader.py |
| OpenCV headless | 5.0.0.93 | Laplacian sharpness | backend/api/routes/predict.py |
| HTTPX | 0.28.1 | OAuth token/profile requests and tests | backend/api/routes/auth.py |
| python-multipart | 0.0.29 | Multipart upload parsing | backend/api/routes/predict.py |
| python-dotenv | 1.2.2 | Loads backend/.env | backend/config.py |

### Machine-learning and AI technologies

| Name | Version | Purpose | Files where used |
|---|---:|---|---|
| PyTorch | Unpinned; artifact 2.12.1+cu130 | Training, checkpoints, models | requirements.txt, src/training/ |
| torchvision | Unpinned | Declared training dependency; no direct import found | requirements.txt |
| timm | Unpinned; artifact 1.0.28 | Pretrained models and native preprocessing | src/models/model_factory.py |
| Albumentations | Unpinned; artifact 2.0.8 | Image augmentation | src/data/transforms.py |
| scikit-learn | Unpinned; artifact 1.9.0 | Metrics, reports, confusion matrices | src/evaluation/ |
| ONNX | Unpinned | Export and graph checking | src/training/export_onnx.py |
| ONNX Runtime | Backend 1.26.0; artifact 1.27.0 | Production inference and benchmarking | backend/api/model_loader.py |
| Matplotlib | Unpinned | Confusion and reliability plots | src/evaluation/ |
| pandas and openpyxl | Unpinned | Survey CSV/Excel ingestion | src/data/ingest_field_survey.py |

### Database and storage

| Name | Version | Purpose | Files where used |
|---|---:|---|---|
| PostgreSQL | Production; local config major 17 | Users, sessions, scans, feedback, diseases | supabase/migrations/ |
| Supabase | CLI not pinned | Hosted database and local services | supabase/config.toml |
| SQLite | Python standard library | Explicit development/tests and local backup | backend/db/database.py, backups/ |
| psycopg | 3.3.2 with binary/pool | SSL PostgreSQL pool | backend/db/database.py |
| JSONB | PostgreSQL built-in | Quality-warning arrays | supabase/migrations/ |
| Filesystem | Not applicable | Datasets, models, artifacts | data/, models/, artifacts/ |
| Supabase Storage | Enabled locally, unused by app | Uploaded image binaries are not retained | supabase/config.toml, docs/deployment.md |

### APIs and external services

| Service | Version | Purpose | Files where used |
|---|---:|---|---|
| Google OAuth 2.0 and OpenID Connect | Protocol | Login and verified basic profile | backend/api/routes/auth.py |
| Kaggle CLI/API | Not pinned | Optional PlantVillage download | src/data/download_data.py |
| GitHub Releases | HTTP asset | Immutable ONNX download | scripts/download_model.py |
| Vercel | Not pinned | Frontend and same-origin API rewrite | frontend/vercel.json |
| Render | Not pinned | Dockerized FastAPI hosting | render.yaml |
| Supabase hosted PostgreSQL | Not pinned | Managed production database | docs/deployment.md |
| Google Fonts | Remote service | Typography | frontend/src/styles/index.css |
| GitLab CLI MCP | glab not pinned | Development integration | opencode.json |

### Testing tools

| Name | Version | Purpose | Files where used |
|---|---:|---|---|
| pytest | 8.4.2 | Data, ML, release, API, auth, DB tests | tests/, backend/tests/ |
| FastAPI TestClient | FastAPI 0.136.3 | In-process API integration | backend/tests/ |
| Node test runner | Node version not pinned | Frontend service/auth contracts | frontend/src/services/api.test.js |
| ONNX Runtime parity | 1.26.0 in tested backend | Export and graph validation | tests/test_onnx_export.py |

### Deployment and DevOps tools

| Name | Version | Purpose | Files where used |
|---|---:|---|---|
| Docker | Not pinned | Non-root backend image | Dockerfile, backend/Dockerfile |
| Render Blueprint | Platform-managed | Backend service declaration | render.yaml |
| Vercel config | Platform-managed | SPA and API rewrites | frontend/vercel.json |
| Supabase CLI | Docs request latest; not reproducibly pinned | Migration push/list/lint | docs/deployment.md |
| Bash and PowerShell | Host-managed | Setup, training, migration, deployment helpers | scripts/ |

### Development utilities

| Name | Version | Purpose | Files where used |
|---|---:|---|---|
| Git | Not pinned | Version control and clean-clone validation | scripts/verify_clean_clone.py |
| glab | Not pinned | GitLab MCP server | opencode.json |
| PyYAML | Unpinned | Training configuration | src/training/config.py |
| tqdm | Unpinned | Declared but no direct import found | requirements.txt |
| Chrome automation script | Chrome path hard-coded; package declaration absent | Screenshots; current selectors are stale | scripts/capture_frontend_screenshots.mjs |

---

## 3. Repository Structure

### Clean directory tree

~~~text
crop-disease-detection/
├── backend/
│   ├── main.py, config.py
│   ├── api/
│   │   ├── auth.py, model_loader.py, schemas.py
│   │   └── routes/auth.py, disease_info.py, predict.py, system.py
│   ├── db/database.py, seed_disease_data.py, migrations/
│   ├── logs/                         ignored sanitized request logs
│   ├── tests/
│   ├── requirements.txt, requirements-dev.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx, main.jsx, authState.js
│   │   ├── components/              nine reusable components
│   │   ├── pages/                   five logical views
│   │   ├── services/api.js
│   │   └── styles/index.css
│   ├── index.html, vite.config.js
│   ├── package.json, package-lock.json
│   └── vercel.json
├── src/
│   ├── data/                        acquisition, split, registry, review, transforms
│   │   └── review_app/              static survey-review client
│   ├── models/                      BaselineCNN and timm factory
│   ├── training/                    config, engine, train, export, benchmark
│   ├── evaluation/                  metrics, calibration, error analysis
│   ├── inference/                   preprocessing, releases, offline prediction
│   └── utils/seed.py
├── configs/training/                Phase 1 and Phase 2.5 YAML
├── data/
│   ├── raw/                         PlantVillage and field survey
│   ├── processed/                   train, validation, test class folders
│   ├── manifests/field_survey/
│   ├── splits/phase1_split.json
│   └── class_mapping.json
├── models/
│   ├── releases/efficientnetv2_s_v1/
│   ├── checkpoints/                 ignored legacy checkpoint
│   ├── onnx/                        ignored legacy bundle
│   └── model_config.json            obsolete placeholder
├── artifacts/training/              generated checkpoints, metrics, plots, logs
├── scripts/                         operational and verification utilities
├── supabase/config.toml, migrations/
├── tests/                            data and ML pytest suite
├── docs/                             guides, reports, two screenshots
├── backups/                          ignored SQLite backup
├── *.log                              ignored local runtime/training logs
├── Dockerfile, render.yaml
├── requirements.txt, pytest.ini, opencode.json
├── README.md
└── PROJECT_STATUS_AND_TECHNICAL_AUDIT.md
~~~

Generated folders such as .git, .venv, node_modules, dist, caches, and dependency binaries are omitted.

### Folder and major-file purpose

| Path | Purpose |
|---|---|
| backend/main.py | FastAPI entry point, lifespan, CORS, request/security middleware, and router registration. |
| backend/config.py | Environment-backed settings and project-relative path resolution. |
| backend/api/routes/ | HTTP entry layer for OAuth, inference, guidance, health, classes, history, dashboard, and feedback. |
| backend/api/model_loader.py | Release/metadata/graph verification and global ONNX inference service. |
| backend/db/ | SQLite/PostgreSQL adapter, migration, disease seed, and ignored local database. No ORM is used. |
| frontend/src/main.jsx | Mounts App in the browser. |
| frontend/src/App.jsx | Restores session and selects Login or one of four authenticated views. |
| frontend/src/components/ | Upload, result, treatment, history, chart, summary, header, and sidebar elements. |
| frontend/src/pages/ | Dashboard, Scan, History, Profile, and Login logical pages. |
| frontend/src/services/api.js | Central native-fetch client with credentials, CSRF, timeout, and error handling. |
| src/data/ | Dataset download, duplicate-safe split, registry, survey ingestion/cleaning/review, and transforms. |
| src/data/review_app/ | Plain HTML/CSS/JavaScript curation UI; not part of the production React app. |
| src/models/ | BaselineCNN and a seven-architecture timm factory. |
| src/training/ | Training, resume, EMA, AMP, weighting, augmentation, export, comparison, and promotion logic. |
| src/evaluation/ | Classification, calibration, confusion matrix, ROC-AUC, reliability, and error analysis. |
| src/inference/ | Shared release verification, exact preprocessing, and offline prediction tools. |
| configs/training/ | Declarative data/model/optimization/evaluation contracts. |
| data/ | Two locally present dataset families, immutable split, mapping, and survey manifests. |
| artifacts/training/ | One incomplete Phase 1 run and two complete Phase 2.5 candidates. |
| models/releases/ | Backend default release. Large ONNX bytes are local/ignored and fetched for clean deployment. |
| scripts/ | Setup, download, benchmark, migration, smoke, screenshot, deployment, and clone utilities. |
| supabase/ | Local CLI configuration and authoritative production schema. |
| tests/ and backend/tests/ | Data/ML and API/auth/database tests. |
| docs/ | Current targeted guides mixed with stale screenshots; the root audit is severely outdated. |
| backups/ | One ignored pre-production SQLite backup. |

### Entry points

| Area | Entry point | Normal invocation |
|---|---|---|
| Backend | backend.main:app | python -m uvicorn backend.main:app |
| Frontend | index.html → main.jsx → App.jsx | npm run dev or npm run build |
| Training | src/training/train.py | python -m src.training.train |
| Selection/reporting | src/training/benchmark.py | python -m src.training.benchmark |
| Dataset acquisition | src/data/download_data.py | python -m src.data.download_data |
| Dataset split | src/data/split_dataset.py | python -m src.data.split_dataset |
| Survey pipeline | ingest, clean, and review modules | python -m src.data followed by module name |
| Model installation | scripts/download_model.py | python scripts/download_model.py |
| Offline inference | src/inference/predict.py | python -m src.inference.predict |

### Configuration, assets, datasets, and notebooks

- Twenty tracked files meet the audit’s configuration definition: environment examples, Docker/Git/npm/Vite/Vercel/Render/Supabase/pytest/OpenCode configuration, and two training YAML files.
- Assets are primarily two documentation screenshots and generated confusion/reliability plots. There is no separate production frontend image asset directory.
- The two screenshots depict an older, unauthenticated design and do not match current components or screenshot-script selectors.
- **Notebooks: Not found in the repository.**
- Database files found locally are backend/db/disease_info.db and one backup under backups/. Both passed read-only integrity and foreign-key checks.

**Evidence:** docs/screenshots/, scripts/capture_frontend_screenshots.mjs, backend/db/, repository inventory

---

## 4. System Architecture

### Architectural description

The frontend is a React SPA. Locally it calls a configured FastAPI origin; in production Vercel rewrites same-origin /api requests to Render. FastAPI authenticates a server-side session, validates CSRF on mutations, processes image bytes in memory, invokes ONNX Runtime, and writes metadata to SQLite locally or PostgreSQL in production. Image binaries are not stored.

The backend uses a SQL adapter rather than an ORM. SQLite qmark statements are converted to PostgreSQL placeholders, JSON lists become JSONB values, and psycopg pooling is used for PostgreSQL. The active filesystem release is validated for manifest schema, byte size, checksum, metadata, class order, minimum backend version, and ONNX input/output contract before health becomes ready.

**Evidence:** frontend/src/services/api.js, frontend/vercel.json, backend/db/database.py, backend/api/model_loader.py

### High-level system architecture

~~~mermaid
flowchart LR
    U[Authenticated user] --> F[React SPA]
    F -->|Local origin or Vercel API rewrite| A[FastAPI on Render]
    A --> AUTH[Google OAuth and session layer]
    A --> VAL[Upload validation and quality checks]
    VAL --> ORT[ONNX Runtime ModelService]
    ORT --> REL[Verified EfficientNetV2-S v1 files]
    A --> DBX[SQL adapter]
    DBX --> SQL[(Local SQLite)]
    DBX --> PG[(Supabase PostgreSQL over SSL)]
    A -->|JSON| F
    TRAIN[Offline PyTorch pipeline] --> ART[Candidate artifacts]
    ART --> VERIFY[Calibration and ONNX parity]
    VERIFY --> REL
~~~

### User request flow

~~~mermaid
sequenceDiagram
    actor User
    participant UI as React SPA
    participant API as FastAPI
    participant Auth as Session and CSRF
    participant Model as ONNX ModelService
    participant DB as SQLite or PostgreSQL
    User->>UI: Open application
    UI->>API: GET /auth/session
    API->>Auth: Hash token and find active session
    alt no active session
        API-->>UI: 401
        UI-->>User: Google sign-in page
    else active session
        API-->>UI: User and expiry
        UI->>API: GET /dashboard
        API->>DB: User-scoped aggregates
        DB-->>API: Summary
        API-->>UI: Dashboard JSON
        User->>UI: Select image and Analyze
        UI->>API: POST /predict plus CSRF
        API->>Auth: Validate session and CSRF
        API->>Model: Validated RGB tensor
        Model-->>API: Class, confidence, top three
        API->>DB: Insert scan metadata
        DB-->>API: Scan ID
        API-->>UI: Enriched prediction
        UI-->>User: Result and guidance
    end
~~~

### Data flow

~~~mermaid
flowchart TD
    RAW[Raw PlantVillage and field survey] --> INGEST[Download or survey ingestion]
    INGEST --> CLEAN[Normalize and queue manual review]
    CLEAN -->|Accepted only| REG[Multi-source registry]
    RAW --> SPLIT[Duplicate-safe stratified split]
    SPLIT --> MAN[Immutable split manifest]
    MAN --> LOAD[DataLoaders and native preprocessing]
    REG --> LOAD
    LOAD --> TRAIN[Pretrained-backbone training]
    TRAIN --> CKPT[Atomic best and last checkpoints]
    CKPT --> EVAL[Test metrics and calibration]
    EVAL --> ONNX[ONNX export and parity]
    ONNX --> CAND[Candidate comparison]
    CAND -->|All required candidates| RELEASE[Versioned release]
    RELEASE --> API[FastAPI inference]
    API --> SCAN[(User-scoped scan metadata)]
~~~

### Machine-learning prediction flow

~~~mermaid
flowchart TD
    B[Multipart bytes] --> LIMIT[Read limit plus one byte]
    LIMIT --> MIME{Allowed MIME and size?}
    MIME -->|No| E4[4xx response]
    MIME -->|Yes| PIL[Pillow verify and EXIF transpose]
    PIL --> SAFE{64 px minimum and 40 MP maximum?}
    SAFE -->|No| E4
    SAFE -->|Yes| Q[Brightness, contrast, sharpness]
    Q --> PRE[RGB shortest-side resize and center crop]
    PRE --> NORM[Normalize mean and standard deviation]
    NORM --> TENSOR[NCHW float32]
    TENSOR --> ORT[ONNX input images]
    ORT --> LOGITS[15 logits]
    LOGITS --> TEMP[Temperature scaling]
    TEMP --> SOFT[Stable softmax]
    SOFT --> TOP[Top one and top three]
    TOP --> ENRICH[Guidance and status]
    ENRICH --> SAVE[Save user-scoped metadata]
    SAVE --> RESP[Prediction JSON]
~~~

### Authentication flow

1. Login reads /auth/config and navigates to /auth/google/login.
2. The backend generates state, stores only its HMAC hash and safe return path, and sets an HttpOnly state cookie.
3. The callback atomically consumes state, verifies cookie/state, exchanges the code, and requests openid, email, and profile.
4. A verified Google subject/email is upserted. Google access and refresh tokens are not stored.
5. Random session and CSRF tokens are issued; only HMAC hashes are stored.
6. Reads require the session. Mutations also require cookie/header/stored CSRF agreement.
7. Logout revokes the database session and deletes cookies with matching attributes.

**Evidence:** backend/api/routes/auth.py, backend/api/auth.py, frontend/src/services/api.js

### Upload and deployment architecture

The browser accepts JPEG, PNG, or WebP up to 10 MiB. The server remains authoritative, applying MIME/format agreement, decode, EXIF, dimension, pixel, and quality validation. Only hash and metadata are persisted. In production, Vercel serves the SPA and proxies API traffic; Render builds the non-root Docker image, verifies/downloads the model, binds the platform port, and probes /health; Supabase supplies SSL PostgreSQL. Browser roles have RLS enabled and table privileges revoked. No Compose, Kubernetes, worker queue, or CI workflow exists.

**Evidence:** frontend/src/components/UploadCard.jsx, backend/api/routes/predict.py, Dockerfile, render.yaml, supabase/migrations/20260715170000_initial_production_schema.sql

---

## 5. Application Workflow

### End-to-end workflow

1. Vite serves index.html and React mounts App.
2. App sends GET /auth/session with cookies and shows a restoring status.
3. If unauthenticated, Login loads /auth/config and starts Google OAuth. Callback errors are mapped to friendly messages.
4. An authenticated session renders GlobalHeader, Sidebar, and Dashboard. activePage reducer state selects views; there is no React Router.
5. Dashboard sends GET /dashboard. The backend returns user-scoped aggregates and recent rows, never fabricated placeholders.
6. On Scan, the user drags, browses, or opens a camera input. A temporary preview URL is created and later revoked.
7. Scan calls GET /health once on mount to display model readiness/version.
8. Analyze sends FormData to POST /predict, copies the current CSRF cookie to X-CSRF-Token, and uses a 60-second timeout.
9. The backend validates authentication, CSRF, byte size, MIME, format, decoding, dimensions, pixel count, and image content.
10. Release-native preprocessing produces a 300 × 300 RGB NCHW float32 tensor.
11. ONNX Runtime returns logits. Temperature scaling and stable softmax produce confidence and top-three predictions.
12. The result is enriched with reviewed guidance when available and a detection/quality status.
13. A user-scoped scan row is committed with hash and metadata, not image bytes.
14. The response displays preview, class, confidence, alternatives, quality warnings, and available treatment.
15. GET /history and GET /dashboard later return only this user’s rows. History search is escaped server-side and encoded client-side.
16. POST /feedback can save feedback against an owned scan, although no current React form calls it.
17. Logout revokes the session, clears cookies, resets frontend state, and returns to Login.

### Error and loading behavior

| Layer | Behavior | Evidence |
|---|---|---|
| API client | Abort timeout, JSON/text parsing, status-bearing errors, 401 event, no fake data | frontend/src/services/api.js |
| Pages | role=alert errors, role=status loading, dashboard retry, disabled actions | frontend/src/pages/ |
| Auth | 401 missing/expired session, 403 CSRF mismatch, safe OAuth error redirects | backend/api/auth.py |
| Upload | 400/413/422 for type, size, decode, dimensions, pixels, or content | backend/api/routes/predict.py |
| Model | Load failure resets service; /health and prediction return 503 | backend/api/model_loader.py |
| Database | Invalid production configuration fails startup; commits are explicit | backend/db/database.py |
| Batch | Maximum ten sequential files; earlier rows can remain after a later error | backend/api/routes/predict.py |

---

## 6. Frontend Analysis

### Framework and organization

The frontend is React 18 with Vite 5, written in JavaScript/JSX and styled by one large custom stylesheet. It has no React Router, form library, schema validator, global store, Axios, CSS framework, or component library. App uses useReducer for authentication/navigation, while individual pages use local useState, useEffect, and useMemo.

**Evidence:** frontend/package.json, frontend/src/App.jsx, frontend/src/authState.js

### Pages and routes

All views share the same SPA document URL. The term route below therefore means a logical view, not a browser URL route.

| Logical page or route | Purpose | Major components | Related API endpoints |
|---|---|---|---|
| Login | Check OAuth readiness, show errors, begin Google sign-in | Login | GET /auth/config; navigation to GET /auth/google/login; callback handled server-side |
| Application shell | Restore session, handle unauthorized event/logout, select active view | GlobalHeader, Sidebar | GET /auth/session; POST /auth/logout |
| Dashboard | Show real user-specific totals, health ratio, disease chart, and recent scans | FieldOverviewCard, DiseaseDistributionCard, DetectionHistoryCard | GET /dashboard |
| New scan | Select/preview a leaf, check health, run inference, show result/guidance | UploadCard, RecentDetectionCard, DetectionResultCard, TreatmentCard | GET /health; POST /predict |
| Scan history | Search and display up to 200 private rows | DetectionHistoryCard | GET /history |
| Profile | Display session-provided Google identity and dates | Profile | No page-specific call; data comes from GET /auth/session |
| URL behavior | Vercel serves index.html after API rewrites | App | frontend/vercel.json SPA fallback |

**Evidence:** frontend/src/App.jsx, frontend/src/pages/, frontend/vercel.json

### Reusable components

| Component | Purpose | Important inputs |
|---|---|---|
| GlobalHeader | Brand, current title, avatar/initials, logout | user, title, onLogout |
| Sidebar | Four internal navigation buttons | activePage, onNavigate |
| UploadCard | Browse, camera, drag/drop, client checks, Analyze button | selectedFile, onFileSelected, onAnalyze, isLoading |
| RecentDetectionCard | Local preview and compact last result | previewUrl, result, isLoading |
| DetectionResultCard | Prediction class, confidence, status, top-three alternatives | result, isLoading |
| TreatmentCard | Symptoms, treatment, severity, unavailable state | result, isLoading |
| DetectionHistoryCard | Compact/expanded scan rows and optional View All | scans, expanded, onViewAll |
| DiseaseDistributionCard | Responsive Recharts disease distribution | distribution |
| FieldOverviewCard | Healthy/diseased overview and latest scan | summary |

**Evidence:** frontend/src/components/

### State management and important logic

- authStateReducer owns session, activePage, and authentication error. unauthorized and logged-out reset both identity and navigation.
- Each data page owns loading, error, result, and request lifecycle state. Effect cleanup uses an active flag to avoid state updates after unmount.
- Scan owns the selected File, preview object URL, backend health, inference state, and response. Object URLs are revoked.
- API requests always include credentials. Mutation wrappers read the CSRF cookie at request time, preventing stale-token caching.
- A 401 dispatches a window-level leaflight:unauthorized event except where the caller explicitly permits 401 during session restoration.
- Request timeouts are 30 seconds normally and 60 seconds for prediction.

**Evidence:** frontend/src/authState.js, frontend/src/pages/Scan.jsx, frontend/src/services/api.js

### Forms and validation

UploadCard limits client selection to image/jpeg, image/png, and image/webp and 10 MiB. It supports file browse, drag/drop, and a capture-oriented camera input. History uses a controlled search form and trims the query. Login is a button-driven OAuth flow. There is no client schema validation library; server validation is authoritative. One usability inconsistency is that the camera input advertises image capture broadly while acceptFile still rejects types outside the three MIME values.

**Evidence:** frontend/src/components/UploadCard.jsx, frontend/src/pages/History.jsx

### Authentication handling

The frontend does not store tokens in localStorage or sessionStorage. It relies on secure cookies, calls session restoration on load, maps callback query errors, removes auth query parameters after completion, and resets all state on logout/401. The profile picture is fetched from the provider URL with a no-referrer policy.

**Evidence:** frontend/src/App.jsx, frontend/src/pages/Profile.jsx, frontend/src/services/api.js

### UI animation, responsiveness, and accessibility

- CSS includes skeleton shimmer, spinner, fade-in, hover/transition effects, and a prefers-reduced-motion rule.
- Responsive breakpoints occur at 1100, 1000, 860, 640, and 600 pixels; grids collapse and shell/navigation layout adapts.
- Semantic main, section, article, form, label, button, dl/dt/dd elements are used. Errors use role=alert and loading/setup text uses role=status. Images have alt text; stats sections have labels.
- Accessibility limitations: no automated audit, no explicit focus-visible design found, Recharts relies largely on visual output, internal navigation buttons do not provide link/deep-link semantics, and color/contrast has not been measured.

**Evidence:** frontend/src/styles/index.css, frontend/src/pages/, frontend/src/components/

### Error and loading states

Every data page has a loading and error branch. Dashboard offers Retry; Scan clears a prior result before inference; History shows a status; Login disables sign-in until configuration is ready. The client never synthesizes successful dashboard/history values after backend failure.

**Evidence:** frontend/src/pages/Dashboard.jsx, Scan.jsx, History.jsx, Login.jsx, frontend/src/services/api.test.js

### Frontend gaps and stale material

- README.md names Axios, but package.json and api.js use native fetch.
- README.md says availability checks are periodic, but Scan calls health only once per mount.
- The checked-in screenshots show an older unauthenticated navigation/dashboard and do not represent current pages.
- The screenshot script waits for obsolete selectors such as history-panel, dashboard-layout, and chart-panel and hard-codes a local Windows Chrome path.
- Feedback and batch prediction exist in the API client/backend unevenly: sendFeedback exists but no visible form uses it; no batch client wrapper exists.
- The production build succeeds but warns that its main minified JavaScript chunk is 525.55 kB, suggesting route/component code splitting.

**Evidence:** README.md, frontend/package.json, frontend/src/services/api.js, scripts/capture_frontend_screenshots.mjs, build validation on 16 July 2026

---

## 7. Backend Analysis

### Framework, entry point, and route structure

FastAPI is created in backend/main.py with title Crop Disease Detection API and application version 1.0.0. The lifespan configures logging, validates database rules, migrates/seeds SQLite in development, loads the release, and closes a PostgreSQL pool on shutdown. Four route modules are included: authentication, prediction, disease information, and system.

**Evidence:** backend/main.py

### Controllers, services, and middleware

The route functions act as controllers; there is no formal controller/service/repository directory. ModelService is the main service object. DatabaseConnection and its cursor/row adapters form an infrastructure layer. Middleware records sanitized method/path/status/duration and sets X-Content-Type-Options, X-Frame-Options, Referrer-Policy, a camera Permissions-Policy, and no-store for sensitive endpoints. Uvicorn access logging is disabled so OAuth query values are not logged.

**Evidence:** backend/main.py, backend/api/model_loader.py, backend/db/database.py

### Validation and file processing

- Query constraints: history limit 1–200, non-negative offset, search length at most 100.
- Upload constraints: maximum configured bytes, JPEG/PNG/WebP, decoded format agreement, minimum 64 × 64, maximum 40 megapixels, Pillow decompression-bomb handling.
- Quality: grayscale mean, standard deviation contrast, Laplacian variance. Contrast below 2 is rejected; darkness below 25, brightness above 235, contrast below 18, and sharpness below 12 generate warnings.
- Batch size is 1–10 files.
- FeedbackRequest lacks range and length constraints for confidence, class, and message.

**Evidence:** backend/api/routes/predict.py, backend/api/routes/system.py, backend/api/schemas.py

### Authentication and authorization

Google OAuth is the only identity provider; no password is accepted or stored. Session and CSRF tokens are random and HMAC-SHA256 hashed with AUTH_SECRET before database storage. Session-required routes use require_user; mutations use require_csrf. Dashboard, history, scan creation, feedback, and scan-linked feedback are user-scoped. OAuth state is one-use through DELETE RETURNING. Duplicate verified email with a different provider subject is rejected.

**Evidence:** backend/api/auth.py, backend/api/routes/auth.py, backend/api/routes/system.py

### Error handling, logging, and background tasks

HTTPException is used for expected authentication, upload, missing guidance, missing model, and ownership failures. Model loading resets service state and fails closed. The health endpoint exposes database/model readiness and returns 503 when either is unavailable. Standard logging supports console and optional rotating file behavior configured through settings. There are no queues, schedulers, workers, WebSockets, or durable background tasks. FastAPI lifespan is the only startup/shutdown task mechanism.

**Evidence:** backend/main.py, backend/api/model_loader.py, backend/config.py

### API table

| Method | Endpoint | Purpose | Input | Output | Authentication | Main implementation |
|---|---|---|---|---|---|---|
| GET | /health | Database/model readiness | None | HealthResponse; 200 or 503 | Public | backend/api/routes/system.py |
| GET | /classes | Ordered model class names | None | Array of 15 strings; 503 if model absent | Public | backend/api/routes/system.py |
| GET | /auth/config | Tell UI whether Google OAuth is configured | None | AuthConfigResponse | Public | backend/api/routes/auth.py |
| GET | /auth/google/login | Start OAuth | Optional return_to local path | Redirect to provider; state cookie | Public | backend/api/routes/auth.py |
| GET | /auth/google/callback | Complete OAuth | code, state, or error query; state cookie | Redirect to trusted frontend; session/CSRF cookies on success | Public with state verification | backend/api/routes/auth.py |
| GET | /auth/session | Restore current user | Session cookie | SessionResponse | Session required | backend/api/routes/auth.py |
| POST | /auth/logout | Revoke session | Session and CSRF cookie/header | status logged_out; cleared cookies | Session plus CSRF | backend/api/routes/auth.py |
| POST | /predict | Validate/classify/store one image | Multipart file | PredictionResponse | Session plus CSRF | backend/api/routes/predict.py |
| POST | /predict/batch | Classify/store up to ten images | Multipart files list | Array of PredictionResponse | Session plus CSRF | backend/api/routes/predict.py |
| GET | /disease/{class_name} | Fetch reviewed guidance | Class path parameter | DiseaseInfo; 404 if unreviewed | Public | backend/api/routes/disease_info.py |
| GET | /history | Private scan history/search | limit, offset, optional search | Array of ScanHistoryItem | Session required | backend/api/routes/system.py |
| GET | /dashboard | Private aggregates/distribution/recent rows | None | DashboardSummary | Session required | backend/api/routes/system.py |
| POST | /feedback | Save feedback, optionally against owned scan | FeedbackRequest JSON | status received | Session plus CSRF | backend/api/routes/system.py |

### Main business logic

- Prediction status is low_confidence below the configured threshold, otherwise review_recommended when quality warnings exist, healthy when the class contains healthy, and disease_detected otherwise.
- Disease data is reviewed-only. Missing classes are converted into readable crop/disease text but do not receive invented symptoms or treatment.
- Dashboard identifies healthy classes by case-insensitive class-name matching and computes aggregates directly in SQL.
- History escapes backslash, percent, and underscore before LIKE, preventing wildcard expansion by user input.
- Feedback with scan_id retrieves class/confidence from a row owned by the current user.

**Evidence:** backend/api/routes/predict.py, backend/api/routes/disease_info.py, backend/api/routes/system.py

---

## 8. Database Analysis

### Database technology and connection logic

Production uses PostgreSQL through psycopg pooling and requires SSL. Development/test may use an explicitly configured SQLite URL. DATABASE_URL is mandatory; production rejects SQLite. A small adapter normalizes row access and query placeholders. The pool range defaults to 1–5. No ORM or database model class exists.

**Evidence:** backend/config.py, backend/db/database.py

### Production schema table

| Table | Important fields and types | Keys and relationships | Indexes/constraints | Main CRUD |
|---|---|---|---|---|
| app_schema_migrations | version integer, applied_at timestamptz | PK version | RLS; browser privileges revoked | Migration insert/read |
| diseases | class_name text, crop, disease_name, symptoms, recommended_treatment, severity_level | PK class_name | symptoms/treatment not null | Seed delete/upsert; prediction/disease read |
| users | id uuid, name, email, profile_picture, auth_provider, provider_account_id, timestamps | PK id; unique provider pair | Unique lower(email); updated_at trigger | OAuth select/insert/update |
| auth_sessions | id bigint, user_id uuid, token_hash, csrf_token_hash, timestamps, expires_at, revoked_at | PK id; FK users cascade | Unique token hash; user/expiry and active-token indexes | Create, authenticate/update last_seen, revoke |
| oauth_states | state_hash text, return_to, created_at, expires_at | PK state_hash | Expiry index | Insert and atomic delete/consume |
| scans | id bigint, user_id uuid, timestamp, class, confidence, image_hash, file/model/status metadata, quality_warnings JSONB | PK id; FK users cascade; unique id+user | Confidence 0–1, non-negative size, user/timestamp index | Insert prediction; history/dashboard/feedback read |
| feedback | id bigint, user_id uuid, scan_id bigint, timestamp, class, confidence, message | PK id; FK users; composite FK scan_id+user_id to owned scan | Confidence 0–1; user/timestamp index | Insert and migration |

**Evidence:** supabase/migrations/20260715170000_initial_production_schema.sql

### Local SQLite differences

SQLite stores UUIDs/timestamps/quality warnings as TEXT and uses INTEGER AUTOINCREMENT. Its migration conditionally adds columns to legacy scans and feedback tables. It is less strict than PostgreSQL: confidence and file-size CHECK constraints are absent, scans.user_id can remain nullable for legacy data, and users lacks updated_at. backend/db/disease_info.db currently has 7 logical tables, passes integrity_check, has no foreign-key violations, and contains aggregate counts of 2 users, 34 scans, 6 feedback rows, 5 sessions, 6 diseases, no OAuth state, and one migration. The backup has 33 scans and 3 sessions with the other reported counts equal. Values and identities were not inspected or exposed.

**Evidence:** backend/db/database.py, backend/db/migrations/001_auth_and_user_scoping.sql, backend/db/disease_info.db, backups/

### Disease seed

Startup seeding upserts exactly six reviewed rows and deletes disease rows outside that reviewed set: Tomato Early Blight, Tomato Late Blight, Tomato Leaf Mold, Potato Early Blight, Potato Late Blight, and Tomato healthy. Therefore 9 of 15 classifier outputs have no reviewed treatment record.

**Evidence:** backend/db/seed_disease_data.py

### Entity relationships

A user owns many sessions and scans. A user can own many feedback records. Feedback may reference one scan, but its composite foreign key requires that the scan belong to the same user. OAuth states are temporary and not linked by foreign key. Diseases are keyed by the model class string and are looked up logically rather than referenced from scans, preserving scan history even if guidance changes.

### ER diagram

~~~mermaid
erDiagram
    USERS ||--o{ AUTH_SESSIONS : owns
    USERS ||--o{ SCANS : owns
    USERS ||--o{ FEEDBACK : submits
    SCANS ||--o{ FEEDBACK : receives
    DISEASES {
        text class_name PK
        text crop
        text disease_name
        text symptoms
        text recommended_treatment
        text severity_level
    }
    USERS {
        uuid id PK
        text email UK
        text auth_provider
        text provider_account_id UK
    }
    AUTH_SESSIONS {
        bigint id PK
        uuid user_id FK
        text token_hash UK
        text csrf_token_hash
        timestamptz expires_at
        timestamptz revoked_at
    }
    SCANS {
        bigint id PK
        uuid user_id FK
        text predicted_class
        float confidence
        text image_hash
        jsonb quality_warnings
    }
    FEEDBACK {
        bigint id PK
        uuid user_id FK
        bigint scan_id FK
        text predicted_class
        float confidence
        text message
    }
    OAUTH_STATES {
        text state_hash PK
        text return_to
        timestamptz expires_at
    }
~~~

### Migration and initialization

- SQLite migration is applied idempotently during non-production lifespan startup.
- Production schema is a tracked forward migration for Supabase; the application does not auto-apply it.
- Seed data runs during backend startup after database readiness.
- scripts/migrate_sqlite_to_postgres.py migrates genuine Google users and their owned scans/feedback idempotently, excluding sessions, OAuth state, fixtures, unowned legacy records, and secret material.

**Evidence:** backend/main.py, backend/db/database.py, supabase/migrations/, scripts/migrate_sqlite_to_postgres.py

---

## 9. Machine-Learning and AI Analysis

### Problem definition and class ontology

The implemented task is supervised, single-label, multi-class image classification. One leaf image produces one of 15 mutually exclusive labels. The active ontology is:

1. Pepper bell bacterial spot
2. Pepper bell healthy
3. Potato early blight
4. Potato late blight
5. Potato healthy
6. Tomato bacterial spot
7. Tomato early blight
8. Tomato late blight
9. Tomato leaf mold
10. Tomato Septoria leaf spot
11. Tomato two-spotted spider mite
12. Tomato target spot
13. Tomato yellow leaf curl virus
14. Tomato mosaic virus
15. Tomato healthy

**Status:** Confirmed from code.  
**Evidence:** data/class_mapping.json, models/releases/efficientnetv2_s_v1/model.json

### Model architectures and pretrained models

The factory supports seven timm architectures: EfficientNetV2-S, ConvNeXt-Tiny, ConvNeXt-Base, Swin-Tiny, EfficientNet-B0, ResNet-50, and MobileNetV3-Large. The repository also defines BaselineCNN, a three-block 32/64/128-channel convolutional network with batch normalization, ReLU, max pooling, adaptive pooling, 0.25 dropout, and a linear classifier. Phase 2.5 requires pretrained EfficientNetV2-S, ConvNeXt-Tiny, and ConvNeXt-Base, with Swin-Tiny optional.

**Status:** Confirmed from code.  
**Evidence:** src/models/model_factory.py, src/models/baseline_cnn.py, configs/training/phase2_5.yaml

### Feature extraction

Feature extraction is learned end-to-end by the pretrained convolutional backbone. timm replaces each backbone’s classification head with a 15-output head and fine-tunes the network; there is no separate handcrafted color, texture, or shape feature vector. The simple BaselineCNN similarly reduces learned spatial features through adaptive average pooling before its linear classifier.

**Status:** Confirmed from code.  
**Evidence:** src/models/model_factory.py, src/models/baseline_cnn.py, src/training/train.py

### Training procedure

| Item | Phase 2.5 value | Status and evidence |
|---|---|---|
| Split | Immutable 70/15/15 manifest, seed 42 | Confirmed from code/generated manifest. data/splits/phase1_split.json |
| Training source | PlantVillage processed; optional PlantDoc and validated field survey skipped | Confirmed from generated manifest. data/splits/phase1_split.json |
| Native inputs | EfficientNetV2-S 300 px; ConvNeXt-Tiny 224 px | Confirmed from generated files. candidate model.json files |
| Pretraining | timm pretrained weights | Confirmed from code. configs/training/phase2_5.yaml |
| Optimizer | AdamW, betas 0.9/0.999, epsilon 1e-8 | Confirmed from code. configs/training/phase2_5.yaml |
| Base learning rate | 0.0002; ConvNeXt-Base/Swin override 0.0001 | Confirmed from code. configs/training/phase2_5.yaml |
| Weight decay | 0.02; bias/norm excluded | Confirmed from code. configs/training/phase2_5.yaml, src/training/train.py |
| Schedule | Per-step cosine decay, 3 warm-up epochs, start factor 0.01, minimum 1e-6 | Confirmed from code. src/training/train.py |
| Loss | Cross-entropy, 0.1 label smoothing, effective-number class weights beta 0.9999 | Confirmed from code. src/training/engine.py |
| Epochs | Maximum 40 | Confirmed from code. configs/training/phase2_5.yaml |
| Batch/accumulation | EfficientNet 12 × 3 = 36 effective; ConvNeXt-Tiny 8 × 4 = 32 | Confirmed from code and generated metrics |
| Mixed precision | CUDA float16 automatic mixed precision | Confirmed from code. src/training/engine.py |
| Gradient protection | Norm clip 1.0; non-finite gradients skip unsafe AMP optimizer step | Confirmed from code/tests. src/training/engine.py, tests/test_phase2_5_pipeline.py |
| EMA | Enabled, decay 0.9999 | Confirmed from code. src/training/engine.py |
| Early stopping | Validation macro F1, patience 8, minimum delta 0.0001 | Confirmed from code. configs/training/phase2_5.yaml |
| Reproducibility | Seed 42, deterministic warn-only, RNG and DataLoader state in resume checkpoint | Confirmed from code. src/training/train.py |
| Checkpointing | Atomic best inference checkpoint and full last/resume checkpoint | Confirmed from code. src/training/train.py |

### Data preprocessing and augmentation

Evaluation uses metadata-driven RGB shortest-side resize, center crop, bicubic interpolation, float range 0–1, and model-specific normalization. EfficientNetV2-S uses crop percentage 1.0 and mean/std [0.5, 0.5, 0.5]. ConvNeXt-Tiny uses crop percentage 0.95 and ImageNet mean/std.

Training augmentation combines:

- Random resized crop, scale 0.72–1.0 and ratio 0.80–1.25.
- Horizontal flip 0.5 and vertical flip 0.2.
- Shift/scale/rotate/perspective geometry at probability 0.45.
- Weighted brightness/contrast, CLAHE, HSV, RGB shift, and gamma at probability 0.60.
- Motion/Gaussian/defocus blur at 0.16.
- Shadow/fog weather effects at 0.10; rain weight is zero.
- JPEG compression at 0.14 with minimum quality 65.
- Coarse dropout at 0.12.
- MixUp alpha 0.2/probability 0.15 and CutMix alpha 1.0/probability 0.15.

**Status:** Confirmed from code.  
**Evidence:** src/data/transforms.py, configs/training/phase2_5.yaml, src/inference/preprocess_input.py

### Evaluation and model selection

The pipeline calculates accuracy, balanced accuracy, macro/weighted precision-recall-F1, per-class results, Matthews correlation coefficient, Cohen kappa, one-vs-rest ROC-AUC, confusion matrix, classification report, expected calibration error, negative log-likelihood, Brier score, reliability diagram, error analysis, inference latency, peak GPU memory, size, and ONNX parity. Temperature scaling is fitted only on validation logits. Candidate selection weights are 40% validation macro F1, 20% calibration, 15% speed, 15% size, and 10% memory; all required candidates must exist.

**Status:** Confirmed from code.  
**Evidence:** src/evaluation/, src/training/benchmark.py, configs/training/phase2_5.yaml

### Completed model results

All values in this table are **Confirmed from generated files or logs**. They are not estimates.

| Metric | EfficientNetV2-S | ConvNeXt-Tiny | Evidence |
|---|---:|---:|---|
| Status | Complete; active v1 release | Complete candidate; not promoted | run_state.json, docs/model_comparison.md |
| Input | 300 × 300 RGB | 224 × 224 RGB | model.json |
| Test samples | 3,094 | 3,094 | metrics and split manifest |
| Best epoch | 16 | 13 | metrics.json |
| Last epoch | 24 recorded | 21, early stopped | history.json, run_state.json |
| Best-epoch training loss | 0.966336716 | 1.006533523 | history.json |
| Best-epoch training accuracy | 0.945781265 | 0.931465968 | history.json; affected by strong augmentation/MixUp/CutMix |
| Best validation accuracy | 1.000000 | 0.998708428 | metrics.json |
| Best validation macro F1 | 1.000000 | 0.999141453 | metrics.json |
| Test loss | 0.974705146 | 0.971062867 | metrics.json |
| Test accuracy | 0.998707175 | 0.998707175 | metrics.json |
| Balanced accuracy | 0.999062902 | 0.999095165 | metrics.json |
| Macro precision | 0.998752326 | 0.998817656 | metrics.json |
| Macro recall | 0.999062902 | 0.999095165 | metrics.json |
| Macro F1 | 0.998905161 | 0.998953749 | metrics.json |
| Weighted F1 | 0.998707553 | 0.998706947 | metrics.json |
| Matthews correlation | 0.998587758 | 0.998587869 | metrics.json |
| Cohen kappa | 0.998587302 | 0.998587299 | metrics.json |
| Macro ROC-AUC | 0.999953001 | 0.999937757 | metrics.json |
| Calibrated test ECE | 0.001292268 | 0.000515338 | metrics.json |
| Calibrated test NLL | 0.035451943 | 0.011569135 | metrics.json |
| Brier score | 0.002583425 | 0.002694791 | metrics.json |
| Temperature | 0.05; lower-bound review needed | 0.348274971; not at boundary | calibration/model comparison files |
| Misclassifications | 4, derived exactly from 3,094 and accuracy | 4, stored in misclassified_images.json | generated metrics |
| Parameters | 20,196,703 | 27,831,663 | metrics.json |
| ONNX size | 80,679,586 B, 76.94 MiB | 111,404,469 B, 106.24 MiB | metrics.json |
| Best checkpoint size | 81,674,915 B | 111,401,555 B | metrics.json |
| Training time | 16,157.953 s, about 4 h 29 m | 6,150.948 s, about 1 h 43 m | metrics.json |
| Peak training GPU memory | 2,486,524,928 B | 1,451,293,184 B | metrics.json |
| ONNX parity max absolute error | 2.5034e-6, passed | 1.3351e-5, passed | model/metrics JSON |
| Fair 1-thread ONNX CPU median | 734.724 ms | 518.296 ms | fair_benchmark.json |
| Fair CUDA PyTorch median | 82.071 ms | 36.742 ms | fair_benchmark.json |

The older default-thread export measurements, 29.815 ms and 39.546 ms ONNX median respectively, use a different threading methodology and must not be compared with the fair one-thread numbers.

**Evidence:** artifacts/training/crop_disease_phase2_5/efficientnetv2_s/, artifacts/training/crop_disease_phase2_5/convnext_tiny/, artifacts/training/crop_disease_phase2_5/fair_benchmark.json

### Production EfficientNet per-class test results

All rows are **Confirmed from generated files**.

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Pepper bell bacterial spot | 149 | 0.993333 | 1.000000 | 0.996656 |
| Pepper bell healthy | 221 | 1.000000 | 0.995475 | 0.997732 |
| Potato early blight | 150 | 1.000000 | 1.000000 | 1.000000 |
| Potato late blight | 150 | 1.000000 | 1.000000 | 1.000000 |
| Potato healthy | 23 | 1.000000 | 1.000000 | 1.000000 |
| Tomato bacterial spot | 319 | 0.996875 | 1.000000 | 0.998435 |
| Tomato early blight | 150 | 1.000000 | 1.000000 | 1.000000 |
| Tomato late blight | 287 | 1.000000 | 0.996516 | 0.998255 |
| Tomato leaf mold | 143 | 1.000000 | 1.000000 | 1.000000 |
| Tomato Septoria leaf spot | 265 | 1.000000 | 1.000000 | 1.000000 |
| Tomato spider mites | 252 | 1.000000 | 0.996032 | 0.998012 |
| Tomato target spot | 210 | 0.995261 | 1.000000 | 0.997625 |
| Tomato yellow leaf curl virus | 481 | 1.000000 | 0.997921 | 0.998959 |
| Tomato mosaic virus | 56 | 1.000000 | 1.000000 | 1.000000 |
| Tomato healthy | 238 | 0.995816 | 1.000000 | 0.997904 |

**Evidence:** models/releases/efficientnetv2_s_v1/metrics.json

### Confusion matrices and result artifacts

Both completed candidates have generated confusion-matrix images. ConvNeXt also has machine-readable confusion_matrix.json, per-class CSV, four misclassified-image records, confidence distribution, reliability diagram, logits, and reports. EfficientNet release metrics retain full per-class results and a generated confusion-matrix PNG in its training artifact. The matrices are overwhelmingly diagonal because each model made four errors. No external field-test confusion matrix exists.

**Status:** Confirmed from generated files; field benchmark **Not available**.  
**Evidence:** artifacts/training/crop_disease_phase2_5/

### Checkpoints and incomplete/legacy models

| Run | Best checkpoint | Last/resume checkpoint | Checkpoint role |
|---|---:|---:|---|
| Phase 1 EfficientNetV2-S | 325,300,773 B | 325,300,773 B | Old-schema incomplete run |
| Phase 2.5 EfficientNetV2-S | 81,674,915 B | 325,350,828 B | Compact inference best; full optimizer/RNG resume state |
| Phase 2.5 ConvNeXt-Tiny | 111,401,555 B | 445,639,261 B | Epoch-13 best; epoch-21 full resume state |
| Legacy MobileNetV3 | 17,088,329 B | Not available | Legacy best_model.pth only |

**Status:** Confirmed from generated files.  
**Evidence:** artifacts/training/, models/checkpoints/

| Bundle | Finding | Status |
|---|---|---|
| Phase 1 EfficientNetV2-S | best.pt and last.pt are each 325,300,773 B. Historical artifact inspection records epoch 23/30, train loss 0.834294, train accuracy 0.999585, validation loss 1.155973, validation accuracy 0.946400. No final test/calibration/ONNX report. | Confirmed from generated checkpoint as documented in PROJECT_STATUS_AND_TECHNICAL_AUDIT.md; incomplete |
| Active Phase 2.5 EfficientNetV2-S | best/last, metrics, histories, calibration, plots, ONNX; copied into versioned release metadata | Complete |
| Phase 2.5 ConvNeXt-Tiny | best/last plus full reports, logits, parity, checksum, ONNX, resume logs | Complete candidate |
| ConvNeXt-Base | No artifact directory or checkpoint | Not found in repository |
| Swin-Tiny | Optional architecture only; no artifact | Not found in repository |
| Legacy MobileNetV3-Large | 17,088,329 B PyTorch checkpoint and 16,872,152 B ONNX, 224 px/15 classes; no stored evaluation metrics | Generated legacy bundle; not served |
| models/model_config.json | Claims EfficientNet-B0, 6 classes, training pending/mock | Obsolete and contradicted by active release |

### Inference input and output

The active ONNX graph accepts input named images with shape N × 3 × 300 × 300 and float32 RGB values normalized with mean/std 0.5. It returns logits named logits with shape N × 15. ModelService applies temperature 0.05, stable softmax, and returns class_name, confidence, and top_3_predictions. The API adds scan ID/time, guidance, statuses, quality warnings, model name/version, and input dimensions.

**Status:** Confirmed from code and generated release files.  
**Evidence:** backend/api/model_loader.py, backend/api/schemas.py, models/releases/efficientnetv2_s_v1/model.json

### Model deployment and selection conclusion

The active release remains EfficientNetV2-S v1. ConvNeXt-Tiny has marginally higher macro F1 and better calibrated ECE, but is 38.1% larger, has per-class regressions, and cannot be formally selected while ConvNeXt-Base is missing under require_all_candidates. No value is imputed for ConvNeXt-Base.

**Evidence:** docs/model_comparison.md, configs/training/phase2_5.yaml

---

## 10. Dataset Analysis

### Dataset inventory

| Dataset | Format and location | Local files/samples | Classes | Training status |
|---|---|---:|---:|---|
| PlantVillage configured Kaggle mirror | JPEG/PNG folders under data/raw/PlantVillage and data/processed | 20,638 logical samples; raw tree is stored twice byte-identically | 15 | Active and sole source in split |
| Field Survey 2023 | CSV plus JPEG/PNG/WebP/GIF/HEIC under data/raw/PlantVillage/FieldSurvey2023; JSON/CSV manifests | 563 survey rows, 3,510 attachment records, 3,508 local image files | No approved canonical class distribution | Present but hard-gated; zero training records |
| PlantDoc | Configured recursive image folder data/raw/PlantDoc | Not found in repository | Not available | Optional and skipped |

**Evidence:** src/data/download_data.py, configs/training/phase2_5.yaml, data/splits/phase1_split.json, data/manifests/field_survey/

The downloader’s configured PlantVillage source is the public Kaggle dataset slug emmarex/plantdisease, although users may override it. The exact downloaded revision, license acceptance record, and archive checksum are **Not found in the repository**. The field survey is locally collected data described by its CSV/manifests; respondent details are intentionally not reported.

### PlantVillage split and class distribution

The persisted split contains 14,447 train, 3,097 validation, and 3,094 test samples, totaling 20,638 at seed 42.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Pepper bell bacterial spot | 698 | 150 | 149 | 997 |
| Pepper bell healthy | 1,035 | 222 | 221 | 1,478 |
| Potato early blight | 700 | 150 | 150 | 1,000 |
| Potato late blight | 700 | 150 | 150 | 1,000 |
| Potato healthy | 106 | 23 | 23 | 152 |
| Tomato bacterial spot | 1,489 | 319 | 319 | 2,127 |
| Tomato early blight | 700 | 150 | 150 | 1,000 |
| Tomato late blight | 1,336 | 286 | 287 | 1,909 |
| Tomato leaf mold | 666 | 143 | 143 | 952 |
| Tomato Septoria leaf spot | 1,240 | 266 | 265 | 1,771 |
| Tomato spider mites | 1,173 | 251 | 252 | 1,676 |
| Tomato target spot | 983 | 211 | 210 | 1,404 |
| Tomato yellow leaf curl virus | 2,246 | 481 | 481 | 3,208 |
| Tomato mosaic virus | 261 | 56 | 56 | 373 |
| Tomato healthy | 1,114 | 239 | 238 | 1,591 |
| **Total** | **14,447** | **3,097** | **3,094** | **20,638** |

**Status:** Calculated during this audit and cross-checked with the generated split.  
**Evidence:** data/processed/, data/splits/phase1_split.json

### PlantVillage integrity

| Check | Result | Label |
|---|---|---|
| Processed image files | 20,638; 342,230,723 bytes | Calculated |
| Dimensions | All 256 × 256 | Calculated |
| Color modes | 20,637 RGB and 1 RGBA | Calculated |
| Decode failures/zero-byte samples | 0 | Calculated |
| Missing split source paths | 0 of 20,638 | Calculated |
| Unique processed hashes | 20,624 | Calculated |
| Duplicate groups/files | 14 groups, 28 files | Calculated |
| Duplicate leakage across split | 0 groups | Calculated |
| Duplicate label conflict | 0 groups | Calculated |
| Raw mirror | Direct and nested class trees contain the same 20,638 sample paths and all hashes match | Calculated |
| Non-sample raw marker | One zero-byte extensionless marker exists in each raw mirror and is ignored | Calculated |

The duplicate-safe splitter hashes images, groups identical content, and assigns a whole group to one split, which explains why duplicate pairs do not leak across partitions.

**Evidence:** src/data/split_dataset.py, data/raw/PlantVillage/, data/processed/

### Field-survey integrity and distribution

| Item | Result | Label |
|---|---:|---|
| Survey rows | 563 | Confirmed from generated manifest |
| Attachment records | 3,510 | Confirmed from generated manifest |
| Valid records | 3,340 | Confirmed from generated manifest |
| Missing references | 125 | Confirmed from generated manifest |
| Ambiguous references | 2 | Confirmed from generated manifest |
| Invalid image records | 0 | Confirmed from generated manifest |
| Invalid label records | 27 | Confirmed from generated manifest |
| Local image files | 3,508, totaling 2,239,049,918 bytes | Calculated |
| Ingestion-eligible local files | 3,501 JPEG/PNG/WebP; BMP count 0 | Calculated |
| Other local files | 2 GIF and 5 HEIC; ingestion excludes both types | Calculated |
| Decode result | 3,503 Pillow-valid; 5 HEIC not decoded because no HEIC decoder is installed, not proven corrupt | Calculated |
| Unique hashes | 2,898 across all 3,508 files | Calculated |
| Duplicate content | 370 groups containing 980 files across all local types | Calculated |
| Manifest duplicate groups | 357 among referenced/eligible manifest records | Confirmed from generated manifest |
| Raw label buckets | 322; 309 valid normalized values; 94 crop strings | Confirmed from generated manifest |
| Missing disease value | 43 records | Confirmed from generated cleaned manifest |
| Unknown value | 141 records | Confirmed from generated cleaned manifest |
| Multilingual | 890 records | Confirmed from generated cleaned manifest |
| Needs manual review | 3,006 records | Confirmed from generated cleaned manifest |
| Automatically normalized | 2,243 records | Confirmed from generated cleaned manifest |
| Review groups | 409 pending, 0 accepted/replaced/rejected | Confirmed from generated validation manifest |
| Training-eligible records | 0 | Confirmed from generated validation manifest |

A meaningful canonical class distribution is **Not available** because all review groups remain pending and the 322 raw label buckets contain spelling, language, missing, unknown, and compound-label variation. The manifest keeps the detailed raw distributions, but reporting them as model classes would be misleading.

**Evidence:** data/manifests/field_survey/manifest.json, cleaned_manifest.json, validated_manifest.json

### Dataset preprocessing

- split_dataset discovers the most plausible PlantVillage class root, hashes files, performs duplicate-safe stratification, copies files into processed split folders, and writes the class mapping.
- MultiSourceDataset reads the immutable manifest. Optional sources are included only when present and, for the field survey, explicitly validated.
- Decode-time transforms convert images to RGB and apply train or evaluation transforms. Stored processed files themselves remain 256 × 256.
- Augmentation is applied only in the training DataLoader; it does not create extra files or alter the persisted train/validation/test images. The exact probabilities are listed in Section 9.
- Field ingestion indexes only JPEG, PNG, BMP, and WebP by name/relative path, records hashes and metadata, preserves all records, and separates cleaning from human approval.

**Evidence:** src/data/split_dataset.py, src/data/multisource_dataset.py, src/data/ingest_field_survey.py

### Dataset limitations

1. PlantVillage is controlled/background-consistent and may overstate field generalization.
2. Only three crops and 15 classes are active; the broader 38-class roadmap is unimplemented.
3. Severe class imbalance exists: Potato healthy has 152 samples while Tomato yellow leaf curl virus has 3,208.
4. Fourteen duplicate pairs remain within splits, although leakage is prevented.
5. The raw PlantVillage set is stored twice, wasting 342,230,723 bytes.
6. Field-survey data is noisy, multilingual, duplicate-heavy, contains missing files/labels, and has not been expert-approved.
7. Survey metadata includes personal fields; it is ignored by Git but requires access control and minimization.
8. validated_manifest.json stores an absolute local source path, reducing portability and exposing a local filesystem detail; the path is intentionally not reproduced here.
9. PlantDoc is configured but absent.
10. Dataset source licensing/version metadata is not stored in a machine-readable provenance file.

**Evidence:** data/, data/manifests/field_survey/, docs/future_model_roadmap.md

---

## 11. Core Logic and Algorithms

### Major feature logic

| Feature | Input | Processing and algorithm | Output | Important files/functions | Edge cases and errors |
|---|---|---|---|---|---|
| OAuth login | Return path and provider callback | Generate/Hash state, one-use consume, code exchange, verified-profile upsert, random session/CSRF | Secure cookies and frontend redirect | routes/auth.py: google_login, google_callback; auth.py: create_session | Missing config, cancellation, state mismatch/expiry/reuse, duplicate email/subject |
| Session authorization | Session cookie | HMAC hash, active/non-expired/non-revoked DB lookup, last-seen update | AuthContext | backend/api/auth.py: require_user | 401 on absent/expired/revoked |
| CSRF authorization | Session, CSRF cookie and header | Constant-time comparison plus stored-hash verification | AuthContext | backend/api/auth.py: require_csrf | 403 on missing/mismatch |
| Single prediction | Multipart leaf image | Bounded read, decode/security checks, quality assessment, metadata-native preprocessing, ONNX, temperature softmax, guidance, insert | PredictionResponse | predict.py: predict, _validate_upload, _assess_image_quality, _enrich_prediction | Unsupported/oversize/corrupt/bomb/small/huge/contentless/model unavailable |
| Batch prediction | 1–10 files | Reuse single-file pipeline sequentially | Array of responses | predict.py: predict_batch | Partial persistence on later failure; no parallelism |
| Release loading | Manifest and local assets | Safe filenames, schema/version/size/SHA checks, metadata/class/preprocess checks, ONNX graph dimension/name checks | Loaded ModelService or fail-closed state | model_release.py, model_loader.py | Missing/corrupt/mismatched asset or invalid temperature |
| Dashboard | Authenticated user ID | SQL conditional aggregates, distribution grouping, recent five | DashboardSummary | system.py: dashboard | Empty history produces null percentages/confidence |
| History search | User ID, pagination, search | Escape SQL LIKE metacharacters, parameterize, order newest first | ScanHistoryItem array | system.py: history | Bounds enforced; malformed warning JSON becomes empty list |
| Duplicate-safe split | Class image paths and seed | SHA group identical files, stratify whole groups, copy by split | Processed folders, mapping, immutable manifest | split_dataset.py | Root discovery ambiguity; duplicate groups never divided |
| Multi-source loading | YAML sources and manifest | Registered loaders, optional-source skip, validated-only field inclusion | Train/val/test datasets/loaders | registry.py, multisource_dataset.py | Missing required source fails; unvalidated survey yields no samples |
| Training | Config and split | Pretrained model, weighted cross-entropy, augmentation, AdamW/cosine, AMP, EMA, early stop, atomic resume | Best/last checkpoints and history | train.py, engine.py | Signature mismatch refuses resume; non-finite gradients do not poison state |
| Evaluation/calibration | Test/validation logits | Metrics, confusion, temperature optimization, ECE/NLL/Brier, reliability and error reports | JSON/CSV/PNG/NPZ evidence | evaluation/, train.py:_finalize_run | Temperature bounds tracked; missing class probabilities guarded |
| Survey review | Cleaned records and human decisions | Group crop/disease suggestions, append audit decision, regenerate eligibility manifest | Validated manifest | review_field_survey.py: ReviewStore | No uncertain suggestion is automatically approved |

### Prediction pseudocode

~~~text
require active session and valid CSRF
content = read at most maximum_bytes + 1
reject if size or declared MIME is invalid
decode and verify image; require declared/decoded format agreement
apply EXIF orientation; require safe dimensions and pixel count
measure brightness, contrast, and sharpness
reject nearly contentless images; otherwise collect warnings
tensor = preprocess using verified release metadata
logits = ONNX session(tensor)
probabilities = stable_softmax(logits / temperature)
top_three = sort probabilities descending
guidance = reviewed disease lookup or unavailable placeholder
status = confidence/quality/healthy decision rules
insert user-scoped hash and scan metadata
return enriched response
~~~

**Evidence:** backend/api/routes/predict.py, backend/api/model_loader.py

### Duplicate-safe split pseudocode

~~~text
for each class:
    hash every image
    group byte-identical images by hash
shuffle groups deterministically with seed 42
assign entire groups toward 70/15/15 class targets
copy every member of a group to the selected split
write class mapping and immutable sample manifest
~~~

**Evidence:** src/data/split_dataset.py, src/data/multisource_dataset.py

### Training/resume pseudocode

~~~text
load and validate YAML
load immutable split and model-native preprocessing
build pretrained classifier and AdamW parameter groups
if resuming:
    validate architecture, class order, split hash, preprocessing, signature
    restore model, EMA, optimizer, scheduler, scaler, RNG, DataLoader state
for epoch until maximum:
    augment batches, optionally MixUp or CutMix
    compute class-weighted label-smoothed loss under AMP
    accumulate gradients and clip norm
    if gradients are non-finite:
        safely skip AMP step; do not advance scheduler or EMA
    else:
        optimizer step, scheduler step, EMA update
    evaluate validation macro F1
    atomically save last and improved best
    stop after configured patience
evaluate held-out test once, calibrate, export ONNX, verify parity
~~~

**Evidence:** src/training/train.py, src/training/engine.py

### Release installation pseudocode

~~~text
parse release manifest and reject unsafe filenames
if destination exists and size/hash match:
    verify metadata, metrics, checksum list, and return reused
otherwise:
    accept only absolute HTTP(S); forbid embedded credentials and HTTPS downgrade
    stream into a temporary file with size limit and SHA-256
    reject mismatch and clean temporary file
    atomically move verified bytes into release directory
verify all supporting assets before serving
~~~

**Evidence:** scripts/download_model.py, src/inference/model_release.py

---

## 12. Security and Privacy

### Security controls

| Area | Implemented control | Evidence |
|---|---|---|
| Password handling | No password authentication; Google OAuth only | backend/api/routes/auth.py |
| Provider tokens | Google access/refresh tokens are used transiently and not stored | backend/api/routes/auth.py |
| Session tokens | Opaque random value, HttpOnly cookie, HMAC-SHA256 hash in DB, expiry/revocation | backend/api/auth.py |
| CSRF | Double-submit cookie/header plus stored HMAC hash; required on mutations | backend/api/auth.py, frontend/src/services/api.js |
| OAuth state | Random, hashed, expiring, cookie-bound, atomic one-use deletion | backend/api/routes/auth.py |
| Authorization | User-scoped dashboard/history/scans/feedback and owned-scan composite FK | backend/api/routes/system.py, production migration |
| Input validation | Pydantic query models, parameterized SQL, LIKE escaping | schemas.py, system.py |
| File upload | Bounded read, MIME/format match, decode verification, decompression-bomb handling, dimensions/pixels | predict.py |
| CORS | Explicit origin list, credentials, limited methods and headers | backend/main.py |
| Database | Production SSL, pool bounds, RLS enabled, browser table privileges revoked | database.py, production migration |
| Model supply chain | Versioned manifest, size and SHA-256, supporting-asset validation, ONNX graph contract | model_release.py, download_model.py |
| Container | Pinned Python image digest, non-root runtime, health check | Dockerfile |
| Logs | Query-free sanitized request path; Uvicorn access log disabled | backend/main.py |
| Response headers | nosniff, frame deny, referrer policy, camera permissions policy, no-store on sensitive APIs | backend/main.py |

### Environment variables

Required or supported variable names are:

- Core/database: ENVIRONMENT, DATABASE_URL, DATABASE_POOL_MIN_SIZE, DATABASE_POOL_MAX_SIZE, DATABASE_CONNECT_TIMEOUT_SECONDS, FORWARDED_ALLOW_IPS.
- Model: MODEL_PATH, MODEL_METADATA_PATH, MODEL_RELEASE_MANIFEST, LEAFLIGHT_MODEL_URL, LOW_CONFIDENCE_THRESHOLD.
- API/logging: CORS_ORIGINS, MAX_UPLOAD_SIZE_MB, LOG_DIR, LOG_TO_FILE.
- Authentication: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, AUTH_SECRET, APP_URL, OAUTH_CALLBACK_URL, COOKIE_SECURE, COOKIE_SAMESITE, COOKIE_DOMAIN, SESSION_TTL_HOURS.
- Frontend: VITE_API_URL, VITE_DEV_API_TARGET.
- Optional data/test tools: KAGGLE_USERNAME, KAGGLE_KEY, TEST_DATABASE_URL.
- Legacy/backward setting: DB_PATH exists in Settings but does not replace the required DATABASE_URL in current connection validation.

Real .env files and an ignored deployment-secret file are locally present. Only their key names were read. No values are included in this report.

**Evidence:** .env.example, backend/.env.example, frontend/.env.example, backend/config.py, src/data/download_data.py

### Privacy design

Uploads are processed in memory and not retained. The database stores a SHA-256, original base filename, MIME, byte size, class/confidence, model/status metadata, user profile identity, and timestamps. Google tokens are not stored. The profile image is loaded remotely with no-referrer. Production uses a backend-owned database connection rather than exposing service credentials to the browser.

**Evidence:** backend/api/routes/predict.py, backend/api/routes/auth.py, frontend/src/pages/Profile.jsx, docs/deployment.md

### Security and privacy limitations

1. No rate limiting, request concurrency cap, account abuse control, or inference queue is implemented.
2. Synchronous Pillow, OpenCV, ONNX, and SQL work runs inside async prediction handlers and can block the event loop.
3. Feedback class/message length and confidence range are not constrained in Pydantic; PostgreSQL constrains confidence but SQLite does not.
4. The API does not set a Content-Security-Policy or Strict-Transport-Security itself; hosting may add them, but that is **Needs manual verification**.
5. Public health, classes, disease guidance, OpenAPI, and docs endpoints reveal service/model structure by design.
6. Session rows are rejected after expiry but no periodic database purge job exists.
7. The raw field survey contains personal metadata and large image files. It is Git-ignored, but repository-local access controls and retention policy are **Not available**.
8. Original filenames are stored and can contain user-supplied personal text, although path components are removed and length is capped.
9. External Google Fonts and profile-image URLs create network/privacy/CSP dependencies.
10. Dataset ZIP extraction uses extractall without an explicit path-containment precheck. A trusted Kaggle source is assumed.
11. Local PyTorch checkpoints are pickle-based; only trusted local checkpoints should be loaded.
12. Secrets are ignored by Git, but no automated secret-scanning workflow is present.

**Evidence:** backend/api/routes/predict.py, backend/api/schemas.py, src/data/download_data.py, .gitignore

### Local/offline processing

Inference and database work can run locally with SQLite and a local ONNX file; no cloud inference API is used. Google login still requires an external provider unless an alternative auth mechanism is added. Training and dataset acquisition are local, with optional Kaggle/model downloads. This provides partial offline capability, not a fully offline application.

---

## 13. Testing and Quality Assurance

### Test inventory

There are 19 genuine Python test files plus one JavaScript test file, for 20 test files total. backend/tests/conftest.py is support code rather than a test file. Static syntax inspection found 74 pytest test functions and 10 Node test calls, totaling 84 test cases before parametrization; the collected pytest count also equals 74.

| Area | Files/cases | Coverage |
|---|---:|---|
| Data/dataset pipeline | 5 files, 6 test functions | Missing root, registry gating, reproducible split, ingest, cleaning, review |
| ML/model/training/evaluation | 9 files, 27 functions | Baseline shape, transforms, metrics, selection, calibration, phase pipeline, resume smoke, ONNX |
| Model download/release | Included above and backend | Checksum, reuse, failure cleanup, manifest, startup fail-closed |
| Backend API/auth/database | 5 files, 41 functions | OAuth/session/CSRF, user scoping, upload, health, prediction, release, production config |
| Frontend API/auth state | 1 file, 10 tests | Dashboard/history/predict requests, CSRF freshness, 401, logout reset, OAuth messages |

**Evidence:** tests/, backend/tests/, frontend/src/services/api.test.js

### Tests run safely during this audit

| Command/category | Result | Label |
|---|---|---|
| Python tests: tests plus backend/tests | **73 passed, 1 skipped**, 7 warnings, 78.00 s | Confirmed from test run on 16 July 2026 |
| Optional PostgreSQL integration | Skipped because TEST_DATABASE_URL was not configured | Confirmed from test run |
| Frontend Node tests | **10 passed, 0 failed**, about 133 ms | Confirmed from test run |
| Vite production build | Succeeded; 845 modules transformed | Confirmed from build run |
| Combined runnable result | **83 passed, 1 skipped, 0 assertion failures** | Calculated from the final runs |

The first pytest attempt used a sandbox-denied temporary location and produced setup errors unrelated to application assertions. It was rerun in an isolated workspace-local temporary directory, passed as above, and all temporary output was removed.

### Warnings and quality signals

- Starlette warns that its current httpx TestClient integration is deprecated.
- PyTorch warns that the legacy TorchScript-based ONNX exporter is deprecated.
- Two test warnings originate from GUI object cleanup in a non-GUI test process.
- Vite warns that the minified main JavaScript chunk is 525.55 kB; gzip size is 152.28 kB.
- Python coverage percentage is **Not available**; no .coverage report or threshold exists.

### Linting, formatting, typing, and CI

Dedicated ESLint, Prettier, Ruff, Black, mypy, TypeScript, pre-commit, coverage configuration, and CI workflow files are **Not found in the repository**. Python type hints are extensive, but no static checker is configured. The frontend is JavaScript rather than TypeScript.

### Missing test areas

1. No browser E2E path covering real login, image selection, prediction, history, and logout.
2. No component-rendering, visual-regression, responsive, accessibility, or cross-browser tests.
3. No load/concurrency/rate-limit/memory tests for inference and batch requests.
4. No configured live PostgreSQL test in this audit; only portable schema and optional integration logic were checked.
5. No tests for all public disease classes or all six guidance records.
6. Limited feedback validation/route coverage and no frontend feedback UI test.
7. No deployment smoke in CI, container vulnerability scan, secret scan, migration rollback test, or dependency audit.
8. No field generalization/OOD/robustness benchmark.

**Evidence:** repository inventory, pytest output, frontend build output

---

## 14. Deployment and Setup Procedure

The instructions below are based only on tracked repository commands. Placeholders are intentionally used for credentials and deployment identifiers.

### Prerequisites

- Git.
- Python 3.11 for the pinned inference/backend environment. The Docker image is pinned to Python 3.11.15.
- Node.js 18 or newer and npm.
- A Google OAuth Web client for interactive login.
- Network access to the versioned model release unless a verified local ONNX already exists.
- For training only: a separate environment with the unpinned root requirements, adequate disk/RAM, and preferably CUDA.
- Optional: Docker, Supabase CLI, Kaggle CLI, and PostgreSQL client tools.

**Evidence:** README.md, Dockerfile, requirements.txt

### Local development

#### 1. Create and install the inference environment

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
npm.cmd --prefix frontend ci
~~~

Use backend/requirements-dev.txt instead of, or after, runtime requirements when running pytest:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r backend/requirements-dev.txt
~~~

No dependency was installed during this audit.

#### 2. Configure environment variables

Copy names from the tracked examples; do not commit real files. A minimal local configuration requires:

~~~powershell
$env:ENVIRONMENT="development"
$env:DATABASE_URL="<local-sqlite-database-url-from-.env.example>"
$env:GOOGLE_CLIENT_ID="<client-id>"
$env:GOOGLE_CLIENT_SECRET="<client-secret>"
$env:AUTH_SECRET="<at-least-32-random-characters>"
$env:APP_URL="http://127.0.0.1:5173"
$env:OAUTH_CALLBACK_URL="http://127.0.0.1:8000/auth/google/callback"
$env:CORS_ORIGINS="http://127.0.0.1:5173"
$env:COOKIE_SECURE="false"
$env:VITE_API_URL="http://127.0.0.1:8000"
~~~

Register the exact local callback with the OAuth provider. COOKIE_SECURE must be true under production HTTPS.

#### 3. Install or verify the model

~~~powershell
.\.venv\Scripts\python.exe scripts/download_model.py
.\.venv\Scripts\python.exe scripts/download_model.py --verify-only
~~~

MODEL_PATH, MODEL_METADATA_PATH, and MODEL_RELEASE_MANIFEST default to the active v1 directory. LEAFLIGHT_MODEL_URL can override the source when necessary. The installer is checksum- and size-gated.

#### 4. Database setup

For local development, DATABASE_URL must explicitly be SQLite. Backend lifespan creates/migrates the local schema and seeds the six reviewed disease rows. A separate seed command exists:

~~~powershell
.\.venv\Scripts\python.exe backend/db/seed_disease_data.py
~~~

The seed command still requires DATABASE_URL in the environment.

#### 5. Start backend and frontend

Terminal one:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
~~~

Terminal two:

~~~powershell
npm.cmd --prefix frontend run dev -- --host 127.0.0.1 --port 5173
~~~

Open the local frontend and verify /health through the API before prediction.

#### 6. Optional dataset setup

The repository already contains local ignored data. For a clean data setup:

~~~powershell
python -m src.data.download_data
python -m src.data.split_dataset
~~~

If downloading manually, place the archive under data/raw and run:

~~~powershell
python -m src.data.download_data --skip-download
~~~

Warning: the normalization script can replace an existing PlantVillage directory when it moves a newly extracted source. Back up valuable local data first.

Field-survey operations:

~~~powershell
python -m src.data.ingest_field_survey --survey-file "<path-to-survey.xlsx>" --image-root "<path-to-survey-images>"
python -m src.data.clean_field_survey_labels
python -m src.data.review_field_survey --host 127.0.0.1 --port 8765
~~~

Do not include field records in training until a qualified reviewer accepts them.

#### 7. Optional training and comparison

Install the separate root training requirements, then:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.training.train --config configs/training/phase2_5.yaml --architecture convnext_base
.\.venv\Scripts\python.exe -m src.training.benchmark --config configs/training/phase2_5.yaml
~~~

The first command is the documented next benchmark action and has not been run. Do not force-rebuild the split when resuming the existing candidate series.

### Testing and build commands

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests backend/tests -q
npm.cmd --prefix frontend test
npm.cmd --prefix frontend run build
~~~

The Vite output directory is frontend/dist. The commands succeeded in this audit except the optional PostgreSQL test, which requires TEST_DATABASE_URL.

### Production deployment

#### Database

1. Create/select a Supabase PostgreSQL project.
2. Link the CLI using a private project reference without committing it.
3. Review, dry-run, and apply the tracked migration.
4. Lint and list migrations:

~~~powershell
npx.cmd --yes supabase@latest link --project-ref <private-project-reference>
npx.cmd --yes supabase@latest db push --linked --dry-run
npx.cmd --yes supabase@latest db push --linked
npx.cmd --yes supabase@latest migration list --linked
npx.cmd --yes supabase@latest db lint --linked --schema public --fail-on error
~~~

Use an SSL PostgreSQL DATABASE_URL. The migration enables RLS and removes browser-role table access. Production migration is not auto-applied by FastAPI.

#### Backend on Render

render.yaml declares a Singapore-region free Docker web service, root Dockerfile, /health probe, and production environment. Configure secret values in the platform rather than the blueprint:

- DATABASE_URL
- GOOGLE_CLIENT_ID
- GOOGLE_CLIENT_SECRET
- AUTH_SECRET

Review APP_URL, OAUTH_CALLBACK_URL, CORS_ORIGINS, cookie settings, and the tracked backend/frontend origin mapping before deployment. The repository contains public deployment host identifiers, but they are intentionally not repeated here. A deployment configuration is present; actual live state is **Needs manual verification**.

#### Frontend on Vercel

Use frontend as the project root:

~~~text
Install command: npm ci
Build command: npm run build
Output directory: dist
Production VITE_API_URL: /api
~~~

frontend/vercel.json proxies API paths first and then applies the SPA fallback. Never place database credentials, OAuth client secret, AUTH_SECRET, or backend session material in Vercel frontend variables.

### Docker deployment

Build from repository root:

~~~powershell
docker build -t leaflight-api:efficientnetv2-s-v1 .
~~~

Run with all production runtime values supplied by the environment:

~~~powershell
docker run --rm -p 8000:8000 -e PORT=8000 -e ENVIRONMENT=production -e DATABASE_URL="<ssl-postgresql-url>" -e GOOGLE_CLIENT_ID="<id>" -e GOOGLE_CLIENT_SECRET="<secret>" -e AUTH_SECRET="<secret>" -e APP_URL="<https-frontend-origin>" -e OAUTH_CALLBACK_URL="<https-callback>" -e CORS_ORIGINS="<https-frontend-origin>" -e COOKIE_SECURE=true leaflight-api:efficientnetv2-s-v1
~~~

The image downloads the release asset during build because the ONNX binary is excluded by .dockerignore, verifies it twice, runs as user leaflight, supports PORT, and includes an HTTP health check. Docker Compose and a frontend Dockerfile are **Not found in the repository**.

### Setup/deployment documentation issues

| File | Issue |
|---|---|
| scripts/setup_env.sh | Creates a Unix venv and installs both dependency sets, but invokes seeding without first setting required DATABASE_URL. |
| scripts/verify_clean_clone.py | Sets legacy DB_PATH rather than current required DATABASE_URL, so a present-day clean-clone run can fail configuration validation. |
| scripts/deploy.sh | Generic text mentions Railway and direct backend URL; current tracked architecture is Render plus Vercel same-origin proxy. |
| scripts/capture_frontend_screenshots.mjs | Hard-coded local Chrome path and obsolete selectors; not portable/current. |
| README.md | Mostly current setup, but incorrectly lists Axios and periodic health checks. |
| PROJECT_STATUS_AND_TECHNICAL_AUDIT.md | Describes an older unauthenticated, SQLite-only, legacy-MobileNet deployment and must not guide current setup. |

---

## 15. Codebase Statistics

Statistics were measured before this report was created.

### File totals and scope

| Measure | Count | Definition |
|---|---:|---|
| Audit-scope files | 65,642 | Excludes Git, dependency trees, virtual environments, caches, and build directories; includes datasets, artifacts, databases, binaries, and logs because the user requested their inspection |
| Requested file-stat total | 65,630 | Audit scope minus 11 ONNX/checkpoint binaries and one generated logits NPZ |
| Version-controlled files | 134 | git ls-files |
| Tracked source files | 92 | Python, JS, JSX, CSS, HTML, SQL, shell, PowerShell |
| Tracked source lines | Approximately 12,887 | Physical lines including blanks/comments |
| Local data files | 65,432 | data/ including images, manifests, split, mapping |
| Generated training artifact files | 42 | artifacts/ |
| Model-directory files | 9 | models/, including ignored binaries and tracked release metadata |

### Source files and approximate lines by language

| Language | Files | Lines |
|---|---:|---:|
| Python | 60 | 10,560 |
| CSS | 2 | 1,043 |
| JSX | 16 | 553 |
| JavaScript | 5 | 411 |
| SQL | 2 | 159 |
| HTML | 2 | 77 |
| PowerShell | 2 | 61 |
| Shell | 3 | 23 |
| **Total** | **92** | **12,887** |

### Structural counts

| Item | Count | Definition/evidence |
|---|---:|---|
| Reusable frontend components | 9 | frontend/src/components/*.jsx |
| Page components | 5 | Dashboard, Scan, History, Profile, Login |
| Authenticated internal views | 4 | Dashboard, scan, history, profile |
| Browser URL routes | 1 SPA shell | No React Router |
| Backend endpoints | 13 | FastAPI route decorators |
| Production database tables | 7 | Includes migration table; no ORM models |
| Implementable ML architectures | 8 | Seven timm aliases plus BaselineCNN |
| Distinct local model artifact bundles | 4 | Phase 1 EfficientNet, Phase 2.5 EfficientNet, ConvNeXt-Tiny, legacy MobileNet |
| Complete Phase 2.5 candidates | 2 | EfficientNetV2-S and ConvNeXt-Tiny |
| Primary training/execution modules | 7 | train, engine, export, benchmark plus benchmark-candidate, resume-smoke, run-training utilities |
| Configured datasets | 3 | PlantVillage, field survey, PlantDoc |
| Locally present dataset families | 2 | PlantVillage and field survey |
| Training-eligible dataset families | 1 | PlantVillage |
| Genuine test files | 20 | 19 Python plus one JavaScript |
| Test cases | 84 | 74 pytest plus 10 Node |
| Configuration files | 20 | Audit definition described in Section 3 |
| Markdown documentation files | 10 | Before this report |
| Other documentation assets | 2 | Screenshots |

### Largest important tracked source files

| Lines | File | Role |
|---:|---|---|
| 1,042 | src/training/train.py | Full training/resume/finalization orchestration |
| 990 | frontend/src/styles/index.css | Production frontend design |
| 516 | src/training/benchmark.py | Candidate scoring, promotion, reports |
| 473 | backend/api/model_loader.py | Release and ONNX service |
| 452 | backend/tests/test_auth_and_dashboard.py | Auth/user-scoping integration tests |
| 357 | src/data/clean_field_survey_labels.py | Label normalization/reporting |
| 356 | backend/db/database.py | Cross-database adapter/migration |
| 349 | src/data/ingest_field_survey.py | Survey/table/image manifest ingestion |
| 326 | src/data/review_field_survey.py | Review server/store |
| 322 | backend/api/routes/auth.py | OAuth and cookies |
| 318 | src/training/config.py | Typed YAML configuration |
| 281 | scripts/download_model.py | Secure release download |
| 275 | scripts/verify_clean_clone.py | Clone/deployment verifier |
| 255 | src/data/transforms.py | Augmentation and evaluation transforms |
| 253 | src/training/engine.py | Epoch, EMA, weighting, AMP |

**Evidence:** calculated from git-tracked files and local audit inventory on 16 July 2026

---

## 16. Dependencies

### Frontend dependencies

| Package | Declared / resolved version | Purpose | Usage assessment |
|---|---|---|---|
| react | ^18.3.1 / 18.3.1 | Components and hooks | Actively used throughout frontend/src |
| react-dom | ^18.3.1 / 18.3.1 | Browser root renderer | Actively used in main.jsx |
| recharts | ^2.15.0 / 2.15.4 | Disease distribution visualization | Actively used in DiseaseDistributionCard.jsx |
| @vitejs/plugin-react | ^4.3.4 / 4.7.0 | JSX/Fast Refresh build plugin | Actively used in vite.config.js |
| vite | ^5.4.11 / 5.4.21 | Dev server and production build | Actively used by npm scripts |

Axios is not installed or imported even though README.md lists it.

**Evidence:** frontend/package.json, frontend/package-lock.json, frontend/src/

### Backend dependencies

| Package | Version | Purpose | Usage assessment |
|---|---:|---|---|
| fastapi | 0.136.3 | API framework | Directly active |
| uvicorn | 0.48.0 | ASGI server | Directly active |
| pydantic | 2.12.5 | Schemas/settings | Directly active |
| pydantic-core | 2.41.5 | Pydantic runtime | Indirectly active |
| python-multipart | 0.0.29 | File uploads | Directly active |
| python-dotenv | 1.2.2 | Environment files | Directly active |
| httpx | 0.28.1 | OAuth HTTP and tests | Directly active |
| numpy | 2.4.4 | Model/image math | Directly active |
| onnxruntime | 1.26.0 | ONNX inference | Directly active |
| opencv-python-headless | 5.0.0.93 | Sharpness metric | Directly active |
| Pillow | 12.2.0 | Image security/preprocess | Directly active |
| psycopg[binary,pool] | 3.3.2 | PostgreSQL driver/pool | Directly active in production path |
| starlette | 1.2.0 | FastAPI ASGI foundation | Indirectly active |
| anyio | 4.13.0 | Async foundation | Indirectly active |
| click | 8.4.2 | Uvicorn CLI | Indirectly active |
| h11 | 0.16.0 | HTTP protocol | Indirectly active |
| httpcore | 1.0.9 | HTTPX transport | Indirectly active |
| idna | 3.11 | URL internationalization | Indirectly active |
| flatbuffers | 25.12.19 | ONNX Runtime serialization support | Indirectly active |
| protobuf | 5.29.6 | ONNX Runtime/model serialization | Indirectly active |
| packaging | 26.0 | Version/package utilities | Indirect dependency; no direct import |
| annotated-doc | 0.0.4 | FastAPI documentation metadata | Indirectly active |
| annotated-types | 0.7.0 | Pydantic constraints | Indirectly active |
| typing-extensions | 4.15.0 | Backported typing support | Indirectly active |
| typing-inspection | 0.4.2 | Pydantic typing introspection | Indirectly active |

The backend list is fully pinned and includes transitive packages, which aids repeatability but increases maintenance burden.

**Evidence:** backend/requirements.txt, backend imports

### ML dependencies

| Package | Repository version | Purpose | Usage assessment |
|---|---|---|---|
| torch | Unpinned; artifact 2.12.1+cu130 | Training/model/checkpoints | Directly active |
| torchvision | Unpinned | Common vision dependency | Potentially unused directly; no import found |
| timm | Unpinned; artifact 1.0.28 | Pretrained backbones/preprocess | Directly active |
| albumentations | Unpinned; artifact 2.0.8 | Augmentation | Directly active |
| opencv-python | Unpinned | Transform image backend | Active through transform/data path |
| scikit-learn | Unpinned; artifact 1.9.0 | Evaluation metrics | Directly active |
| numpy | Unpinned | Arrays, logits, RNG | Directly active |
| pandas | Unpinned | Survey tabular ingestion | Directly active for field tools |
| openpyxl | Unpinned | pandas Excel engine | Conditionally active for Excel survey input |
| matplotlib | Unpinned | Plots | Directly active |
| Pillow | Unpinned | Images | Directly active |
| PyYAML | Unpinned | YAML configs | Directly active |
| tqdm | Unpinned | Progress utility | Potentially unused; no direct import found |
| onnx | Unpinned | Export/graph | Directly active |
| onnxruntime | Unpinned; artifact 1.27.0 | Parity/benchmark | Directly active |

Root ML dependencies are not locked, so reproducing the artifact environment requires the software versions recorded in model.json in addition to requirements.txt.

**Evidence:** requirements.txt, models/releases/efficientnetv2_s_v1/model.json

### Development dependencies

| Package/tool | Version | Purpose | Usage assessment |
|---|---:|---|---|
| pytest | 8.4.2 | Python tests | Active |
| certifi | 2026.2.25 | TLS trust bundle | Indirectly active |
| colorama | 0.4.6 on Windows | Terminal color | Indirectly active |
| iniconfig | 2.3.0 | pytest configuration | Indirectly active |
| pluggy | 1.6.0 | pytest plugins | Indirectly active |
| Pygments | 2.20.0 | pytest output highlighting | Indirectly active |
| Node test runner | Node built-in | Frontend tests | Active |
| Vite/plugin-react | Resolved 5.4.21 / 4.7.0 | Frontend development/build | Active |

No lint, formatter, type-checker, coverage, E2E, or accessibility-test dependency is declared.

**Evidence:** backend/requirements-dev.txt, frontend/package.json

---

## 17. Important Functions and Classes

| Name | File | Purpose | Inputs | Outputs | Called by |
|---|---|---|---|---|---|
| App | frontend/src/App.jsx | Session restoration and logical view orchestration | Browser state/events | React tree | main.jsx |
| authStateReducer | frontend/src/authState.js | Deterministic auth/navigation transitions | state, action | new state | App |
| apiRequest | frontend/src/services/api.js | Credentials, CSRF, timeout, parse/error behavior | path, options | Promise of payload | All API wrappers |
| getSession/logout | frontend/src/services/api.js | Auth restoration/revocation | Cookies | session/status | App |
| predictDisease | frontend/src/services/api.js | Multipart prediction call | File | PredictionResponse | Scan |
| Scan | frontend/src/pages/Scan.jsx | Preview, health, inference lifecycle | none | React view | App |
| UploadCard | frontend/src/components/UploadCard.jsx | File/camera/drop validation and Analyze | file state/callbacks | UI callbacks | Scan |
| Dashboard | frontend/src/pages/Dashboard.jsx | Load/render private aggregates | user, navigation callback | React view | App |
| lifespan | backend/main.py | Startup validation/migrate/seed/model load and shutdown pool | FastAPI app | async context | FastAPI |
| request_logger | backend/main.py | Sanitize logs and add headers/cache rules | request, call_next | response | FastAPI middleware |
| require_user | backend/api/auth.py | Validate opaque session | Request cookie | AuthContext | Protected routes |
| require_csrf | backend/api/auth.py | Validate session and CSRF | Request and AuthContext | AuthContext | Mutation routes |
| create_session | backend/api/auth.py | Create hashed session/CSRF rows | user_id | raw tokens and expiry | OAuth callback |
| google_login | backend/api/routes/auth.py | Create OAuth state and redirect | request, return_to | RedirectResponse | GET auth route |
| google_callback | backend/api/routes/auth.py | Consume state, fetch/upsert user, set session | request/code/state/error | RedirectResponse | Provider callback |
| ModelMetadata | backend/api/model_loader.py | Validated serving metadata contract | JSON mapping | immutable metadata object | ModelService.load |
| ModelService.load | backend/api/model_loader.py | Verify release and ONNX graph | model/metadata/manifest paths | loaded ORT session | backend lifespan/load_model |
| ModelService.predict | backend/api/model_loader.py | Preprocess, infer, calibrate, top-three | Pillow Image | prediction dictionary | predict route |
| _read_upload | backend/api/routes/predict.py | Bounded upload read and close | UploadFile | bytes | predict/predict_batch |
| _validate_upload | backend/api/routes/predict.py | Security/decode/dimension validation | bytes, MIME | safe RGB Image | prediction routes |
| _assess_image_quality | backend/api/routes/predict.py | Brightness/contrast/sharpness policy | Pillow Image | status and warnings | prediction routes |
| _enrich_prediction | backend/api/routes/predict.py | Guidance/status/persistence/response | inference, content, auth metadata | PredictionResponse | prediction routes |
| dashboard | backend/api/routes/system.py | User-scoped aggregates/distribution/recent | AuthContext | DashboardSummary | GET /dashboard |
| history | backend/api/routes/system.py | Escaped user-scoped pagination/search | query and AuthContext | scan list | GET /history |
| DatabaseConnection | backend/db/database.py | Portable execute/commit adapter | SQLite or psycopg connection | cursor adapter | All DB routes/services |
| connect_database | backend/db/database.py | Choose SQLite or pooled PostgreSQL | optional path/settings | closing connection | Backend DB users |
| migrate_database | backend/db/database.py | Idempotent local schema evolution | optional path | none | lifespan/tests |
| seed_database | backend/db/seed_disease_data.py | Synchronize six reviewed guidance rows | optional DB path | none | lifespan/manual command |
| build_model | src/models/model_factory.py | Construct supported timm classifier | architecture/classes/options | nn.Module | training/offline inference |
| BaselineCNN | src/models/baseline_cnn.py | Small benchmark network | tensor N×3×224×224 | N×classes logits | model tests/experiments |
| build_split_manifest | src/data/multisource_dataset.py | Immutable multi-source, duplicate-aware split | config/force flag | manifest dictionary | training |
| MultiSourceDataset | src/data/multisource_dataset.py | Load manifest samples and transforms | records/split/transform | dataset samples | create_dataloaders |
| build_manifest | src/data/ingest_field_survey.py | Resolve survey attachments/labels and issues | table/image paths/options | manifest JSON | field CLI/tests |
| clean_manifest | src/data/clean_field_survey_labels.py | Normalize without discarding/audit loss | raw manifest/output | cleaned manifest/report | field CLI/tests |
| ReviewStore.decide | src/data/review_field_survey.py | Audit a human group decision and regenerate eligibility | group/decision/replacement/note | decision/status | review HTTP handler |
| run_epoch | src/training/engine.py | Train/evaluate one epoch with AMP/EMA/augmentation | model, loader, loss, optimizer options | metrics | train.py |
| ModelEMA | src/training/engine.py | Maintain exponential moving average weights | model/decay | EMA state | train.py |
| train_architecture | src/training/train.py | Lock and execute one architecture run | config/architecture/resume | artifact directory | CLI/benchmark |
| _finalize_run | src/training/train.py | Test, calibrate, export, and write results | best model/loaders/metadata | final artifacts | training completion |
| fit_temperature | src/evaluation/calibration.py | Minimize validation cross-entropy NLL | logits/labels/iterations | temperature/report | finalization/tests |
| compute_metrics | src/evaluation/metrics.py | Full multi-class metrics | labels/predictions/classes/probabilities | metrics dictionary | evaluation/finalization |
| export_and_verify_onnx | src/training/export_onnx.py | Export opset 18 and compare outputs/latency | model/path/metadata/samples | parity report | finalization/tests |
| load_release_manifest | src/inference/model_release.py | Strict release schema and safe paths | release path | ModelRelease | backend/downloader |
| install_release | scripts/download_model.py | Stream, hash, atomically install model | manifest/url/options | DownloadResult | CLI/Docker/tests |
| benchmark_candidate | scripts/benchmark_candidates.py | Fair CPU/CUDA/ORT latency measurement | run root/architecture/method | candidate metrics | benchmark script |

There are no custom React hook modules; the frontend uses built-in hooks directly.

---

## 18. Features and Contributions

### Fully implemented features

- Google OAuth with verified email, one-use state, server sessions, CSRF, logout, and user scoping.
- Secure single-image ONNX inference with quality warnings, calibrated confidence, top-three output, and persistence.
- Batch endpoint with a ten-file cap.
- Private dashboard, history search, profile, and user-owned feedback API.
- PostgreSQL production schema plus explicit SQLite development compatibility.
- Versioned, checksum-gated, fail-closed model release and Docker download.
- Duplicate-safe immutable data split and multi-source registry.
- Field-survey ingestion, normalization, review queue, decision audit, and training hard gate.
- Resumable pretrained-backbone training with AMP, EMA, class weighting, augmentation, early stopping, evaluation, calibration, and ONNX parity.
- Automated Python/frontend contract tests and a successful production build.

### Partially implemented features

- Disease guidance covers only 6 of 15 classes.
- Batch prediction is sequential and not transactionally atomic.
- Feedback API exists without a current UI form and with weak free-form validation.
- Candidate comparison has two complete results but requires an untrained third result.
- Field data tooling is complete, but no expert decisions are recorded.
- Deployment configuration exists, but no repository CI or recorded live smoke result verifies it continuously.

### Planned or placeholder features

- PlantDoc ingestion, full 38-class ontology, validated field benchmark.
- Segmentation, severity estimation, Grad-CAM/explainability.
- Context-aware weather/location input, multilingual guidance.
- Active/continual learning.
- ConvNeXt-Base and optional Swin-Tiny candidates.

**Evidence:** docs/future_model_roadmap.md, configs/training/phase2_5.yaml

### Technical contributions and engineering decisions

1. **Metadata-native preprocessing:** serving and evaluation use the model’s stored preprocessing contract instead of a hard-coded 224-pixel assumption.
2. **Fail-closed model supply chain:** the backend refuses missing/mismatched release assets rather than switching to mock predictions.
3. **Immutable split and release evidence:** split hash, class order, software versions, calibration, and ONNX parity accompany artifacts.
4. **Human-in-the-loop field gate:** normalization suggestions never become training truth without an explicit reviewed decision.
5. **Portable SQL layer:** one query style supports local SQLite and pooled PostgreSQL without introducing an ORM.
6. **Privacy-oriented inference:** image bytes remain in memory while only hashes/metadata are retained.
7. **Same-origin production API:** Vercel proxying supports cookie-based auth without exposing backend secrets to the browser.

**Evidence:** src/models/model_factory.py, src/inference/model_release.py, src/data/review_field_survey.py, backend/db/database.py, frontend/vercel.json

---

## 19. Challenges and Solutions

Only challenges supported by code, generated artifacts, documentation, or commit messages are included.

| Problem | Technical reason | Implemented solution | Relevant files | Remaining limitation |
|---|---|---|---|---|
| Duplicate images could leak across train/test | Filename-level random split does not recognize byte-identical images | Hash content, group duplicates, assign whole group to one split | src/data/split_dataset.py, tests/test_dataset_registry.py | Fourteen pairs still exist within individual splits |
| Noisy field-survey labels | Missing, multilingual, unknown, compound, and misspelled labels | Preserve records, normalize with trace, group review, require explicit accept/replace | clean_field_survey_labels.py, review_field_survey.py | All 409 groups remain pending |
| Preprocessing mismatch between training and serving | Backbones use different native size/crop/normalization | Resolve timm config and persist exact preprocessing in metadata | model_factory.py, preprocess_input.py | Metadata quality remains release-critical |
| Clean deployments cannot track an 80 MB model in Git | Binary is intentionally ignored | Versioned manifest, checksums, HTTPS downloader, Docker build verification | download_model.py, model_release.py, Dockerfile | Build depends on release-asset availability |
| Serving the wrong or corrupt model | Loose path configuration can pair incompatible graph/metadata | Verify manifest assets, class order, graph names/dimensions, minimum backend version | model_loader.py, release tests | No signature/public-key verification beyond SHA-256 |
| Training interruption and unsafe resume | Optimizer/scheduler/EMA/RNG state can diverge after restart | Atomic last checkpoint, training signature, full state restore, lock validation | train.py, resume_smoke_test.py | Phase 1 old checkpoint lacks current signature |
| Transient non-finite gradient under AMP | Overflow can poison weights or advance scheduler inconsistently | Skip unsafe optimizer step; do not advance scheduler/EMA; regression test | engine.py, test_phase2_5_pipeline.py, docs/training_results.md | Requires continued monitoring on new hardware/models |
| Candidate latency could be compared unfairly | Different thread settings and input sizes distort results | Common host, batch one, one CPU thread, fixed warm-up/iterations, native inputs | scripts/benchmark_candidates.py, fair_benchmark.json | One hardware profile does not represent production fleet |
| Public history lacked ownership in the earlier implementation | Rows were not tied to identities | Google auth, users/sessions, user_id filters, owned-scan FK | auth routes, system.py, production migration | No roles/admin/multi-tenant organization model |
| SQLite is unsuitable for hosted persistence | Ephemeral files and write contention | Production rejects SQLite and uses pooled SSL PostgreSQL | database.py, render.yaml, Supabase migration | Local SQLite schema is less strict |
| Untrusted or huge uploads can exhaust resources | Compressed files can decode to large images; async handler can block | Byte, format, dimension, pixel and decompression-bomb checks | predict.py | No rate limit/queue and inference remains synchronous |
| Windows DataLoader memory pressure | Spawned workers each reserve a scientific-Python runtime | Phase 2.5 uses two workers and disables persistent workers | configs/training/phase2_5.yaml | Host-specific tuning; throughput trade-off |
| Documentation drift | Rapid auth/model/deployment changes left old audit/config/screenshots | New targeted auth/deployment/model docs and versioned release metadata | docs/, models/releases/ | Obsolete root audit/model_config/screenshots remain checked in |

---

## 20. Limitations and Future Scope

### Functional and UI/UX limitations

- Only pepper, potato, and tomato are supported, and treatment guidance is complete for only 6 of 15 model classes.
- No passwordless offline identity, role/admin features, organization sharing, export, notification, or user data deletion interface exists.
- Internal page state has no deep links, browser history, lazy routes, or per-page URL.
- Feedback has no current user interface.
- Health is checked once on Scan mount rather than periodically.
- The 525.55 kB main bundle should be code-split; chart code is a likely lazy-load candidate.
- Current screenshots and automation are outdated.

### Scalability and performance limitations

- Synchronous decode/OpenCV/ONNX/database operations block async route workers.
- Batch processing is sequential and partially committing.
- The active 76.94 MiB ONNX has no server-side request queue, bounded concurrency, cache, or autoscaling guidance.
- No load test establishes throughput, p95 API latency, memory ceiling, or service-level objective.
- SQLite is only suitable for low-concurrency local use.
- Dashboard aggregation has useful user/time indexes but no archival or partitioning plan.

### Security and privacy limitations

- No rate limiting, abuse detection, automated secret scan, dependency audit, CSP, or application-set HSTS.
- Free-form feedback validation is weak.
- Raw field data may include personal information; retention, consent, de-identification, and access policy are not documented.
- Original filenames and provider profile data are retained.
- No user-facing account deletion/data export workflow.

### Dataset and model limitations

- PlantVillage laboratory-style performance is not proof of field performance.
- The field dataset is not validated, and PlantDoc is missing.
- Class imbalance is large; rare-class perfect scores have very small support.
- No external test set, geographic/seasonal domain split, OOD detector, abstention benchmark, adversarial/blur/lighting stress report, or confidence threshold validation exists.
- EfficientNet calibration temperature sits at the optimizer lower bound and deserves refitting/review.
- No severity, localization, segmentation, explainability, or multi-label support.
- Formal model selection is blocked by absent ConvNeXt-Base.

### Testing, deployment, and documentation gaps

- No E2E, component, a11y, visual, load, container, migration recovery, or continuous integration tests.
- Python training dependencies and Supabase CLI are unpinned.
- Live OAuth/Render/Vercel/Supabase state is not reproducibly verified in the repository.
- No Docker Compose, infrastructure-as-code for the database project, frontend container, monitoring/alerting, or centralized log plan.
- README has minor inaccuracies; root technical audit, model_config, screenshots, and several scripts are obsolete.

### Practical future improvements

1. Complete reviewed guidance for all 15 classes and add medical/agronomic disclaimers.
2. Validate field-survey groups with qualified experts; create a privacy-reviewed external field test set before retraining.
3. Train ConvNeXt-Base on the unchanged split, run formal selection, review per-class regressions, and publish a new release only if approved.
4. Add OOD/abstention and threshold evaluation; refit/review EfficientNet calibration.
5. Move CPU-bound prediction work to a bounded executor or worker service; add rate limiting and load tests.
6. Make batch writes atomic or return explicit per-file success/error results.
7. Add frontend routing, lazy loading, feedback UI, accessible chart/table alternative, focus styles, and automated a11y/E2E tests.
8. Add CI for lint/type/test/build/container/release verification, dependency/secret scanning, and PostgreSQL migration smoke.
9. Pin/lock training and CLI dependencies; record dataset provenance/license/checksums.
10. Archive or correct obsolete documentation and make screenshot automation portable.

**Evidence:** implementation and gaps cited in Sections 6–19; docs/future_model_roadmap.md

---

## 21. Report-Ready Content

### Abstract

Leaflight is a full-stack crop disease detection system that classifies leaf images and maintains private scan records. The application combines a React frontend, FastAPI backend, Google OAuth, PostgreSQL/SQLite persistence, and an ONNX EfficientNetV2-S classifier. The active model recognizes 15 conditions across pepper, potato, and tomato leaves. The project also includes a reproducible PyTorch training pipeline with duplicate-safe data splitting, augmentation, checkpoint resume, calibration, evaluation, candidate benchmarking, and ONNX parity verification.

The held-out PlantVillage test set contains 3,094 images. The active EfficientNetV2-S release achieved 99.8707% accuracy and 99.8905% macro F1 in generated artifacts. These results are strong on the stored split but do not establish real-world field accuracy. A collected field survey remains excluded from training because all manual-review groups are pending.

### Introduction

Crop diseases reduce yield and quality when symptoms are not recognized early. Visual screening with machine learning can provide a fast preliminary assessment where expert support is limited. Leaflight was developed to connect an image classifier to a secure, understandable web workflow. It focuses on real inference, reproducible evidence, private user history, and honest failure behavior rather than mocked results.

### Problem statement

Users need a simple way to submit a crop-leaf photograph and receive a consistent preliminary classification. The technical problem includes safe image handling, accurate multi-class inference, meaningful confidence, disease information, private persistence, and reproducible model development.

### Objectives

- Classify supported leaf images into 15 healthy/disease classes.
- Show calibrated confidence, top alternatives, quality warnings, and reviewed guidance.
- Protect user identity and scan history through secure authentication and authorization.
- Store scan metadata without retaining image content.
- Build a repeatable training/evaluation/release process.
- Prepare field data for later expert-validated improvement.

### Proposed solution

The proposed solution is a React SPA connected to a FastAPI API. Google OAuth establishes an opaque server session. A CSRF-protected prediction endpoint validates image bytes, performs quality checks, preprocesses the image according to release metadata, runs an ONNX model, and saves the result to a user-owned record. PostgreSQL supports hosted use, while explicit SQLite supports local development. Offline PyTorch tools build and evaluate candidate models.

### System methodology

PlantVillage images are grouped by content hash and split deterministically into training, validation, and test partitions. Pretrained models use architecture-native preprocessing and strong image augmentation. AdamW, class-weighted label-smoothed loss, mixed precision, gradient accumulation, EMA, cosine scheduling, and early stopping train each candidate. Validation logits fit temperature scaling; the held-out test set is evaluated once after training. The chosen model must pass ONNX parity and release integrity checks before serving.

### Technology stack

The user interface uses React 18, Vite 5, Recharts, native fetch, and responsive CSS. FastAPI, Pydantic, Pillow, OpenCV, NumPy, HTTPX, ONNX Runtime, and Uvicorn implement the API. PyTorch, timm, Albumentations, scikit-learn, Matplotlib, ONNX, pandas, and PyYAML implement the ML pipeline. SQLite supports local work and PostgreSQL/Supabase supports production. Docker, Render, and Vercel define the intended deployment.

### System architecture

The browser communicates only with FastAPI, directly in development or through a same-origin Vercel rewrite in production. FastAPI owns authentication, validation, inference, and database access. The browser has no database credentials. The model is a verified local ONNX release. Training is an offline process whose outputs are promoted separately from runtime.

### Implementation details

The backend is divided into authentication, prediction, disease-information, and system routes. A database adapter supports SQLite and PostgreSQL without an ORM. ModelService validates and loads the release, performs release-native preprocessing, and returns calibrated top-three probabilities. The frontend has five logical pages and nine reusable components. Reducer state controls authentication and internal navigation.

### Dataset description

The active split contains 20,638 PlantVillage images in 15 classes: 14,447 training, 3,097 validation, and 3,094 test. All processed images decode at 256 × 256. Fourteen duplicate pairs exist, but hashing confirms none crosses a split or class. Field-survey storage contains 3,508 local images and 3,510 attachment records, but zero is training-eligible pending expert validation. PlantDoc is configured but absent.

### Model-training procedure

EfficientNetV2-S uses 300-pixel native preprocessing, physical batch 12, three-step accumulation, and a 0.0002 learning rate. ConvNeXt-Tiny uses 224 pixels, batch 8, four-step accumulation, and the same learning rate. Both use pretrained weights, AdamW, cosine decay, warm-up, weighted label-smoothed loss, mixed precision, EMA, augmentation, and macro-F1 early stopping. Best and resume checkpoints are written atomically.

### Results and evaluation

EfficientNetV2-S achieved test accuracy 0.998707, macro precision 0.998752, macro recall 0.999063, macro F1 0.998905, and macro ROC-AUC 0.999953. ConvNeXt-Tiny tied accuracy and achieved macro F1 0.998954. Each made four errors. ConvNeXt had better calibrated ECE and fair one-thread ONNX CPU latency but a 38.1% larger model. Formal selection is incomplete because ConvNeXt-Base is not trained. Results are confirmed from generated files and apply only to the stored test split.

### Testing approach

The repository contains 84 test cases. During this audit, 73 Python tests passed, one optional PostgreSQL test skipped, and all 10 frontend tests passed. The Vite production build also succeeded. Tests cover dataset gating, transformations, model shape, training smoke, checkpoint behavior, metrics, ONNX export, release download, model loading, prediction, authentication, CSRF, user ownership, and frontend request contracts.

### Challenges faced

Supported challenges include preventing duplicate leakage, handling noisy multilingual field labels, persisting model-native preprocessing, distributing a large ignored model safely, resuming interrupted training without state drift, protecting against AMP overflow, comparing candidates under fair latency settings, and moving from public SQLite history to authenticated PostgreSQL-ready ownership.

### Learning outcomes

The project demonstrates how model quality depends on data integrity, split discipline, calibration, and deployment parity rather than accuracy alone. It also shows the importance of secure session design, database ownership, fail-closed releases, reproducible checkpoints, artifact traceability, and keeping documentation synchronized with implementation.

### Limitations

The evidence is based mainly on PlantVillage, not a validated field benchmark. Only 15 classes and six reviewed guidance records are supported. The application lacks rate limiting, worker isolation, end-to-end tests, CI, formal monitoring, and complete field review. Candidate selection and live deployment verification remain incomplete.

### Future scope

Future work should validate field data, expand crops/classes and guidance, complete ConvNeXt-Base comparison, introduce OOD abstention and explainability, improve inference concurrency, add full E2E/accessibility/load testing, lock all environments, and automate deployment/release verification.

### Conclusion

Leaflight is a substantial, functioning full-stack ML MVP with a real, verified inference release and a strong engineering foundation. Its most valuable qualities are reproducibility, fail-closed model loading, private user ownership, and an explicit human gate for uncertain field labels. The next milestone should be field validation and operational assurance rather than reporting higher laboratory accuracy alone.

---

## 22. Evidence and Traceability

### Claim-to-source matrix

| Major claim | Supporting files |
|---|---|
| Current frontend/authenticated workflow | frontend/src/App.jsx, frontend/src/pages/, frontend/src/services/api.js |
| Thirteen API endpoints and policies | backend/api/routes/*.py, backend/api/auth.py |
| PostgreSQL/SQLite architecture | backend/db/database.py, supabase/migrations/20260715170000_initial_production_schema.sql |
| Google OAuth without stored provider tokens | backend/api/routes/auth.py, docs/authentication.md |
| Active EfficientNetV2-S v1 | models/releases/efficientnetv2_s_v1/release.json, model.json |
| Model metrics and parity | models/releases/efficientnetv2_s_v1/metrics.json, artifacts/training/crop_disease_phase2_5/ |
| Training procedure | configs/training/phase2_5.yaml, src/training/train.py, engine.py |
| Dataset counts and immutable split | data/splits/phase1_split.json, data/processed/ |
| Field survey remains ineligible | data/manifests/field_survey/validated_manifest.json |
| Deployment design | Dockerfile, render.yaml, frontend/vercel.json, docs/deployment.md |
| Test inventory/results | tests/, backend/tests/, frontend/src/services/api.test.js, audit test output |
| Security controls | backend/api/auth.py, backend/main.py, backend/api/routes/predict.py |

### Documentation currency cross-check

| Document/artifact | Currency finding |
|---|---|
| README.md | Generally matches active auth/PostgreSQL/release deployment; Axios and periodic health statements are incorrect |
| docs/authentication.md | Matches current Google OAuth/session design |
| docs/deployment.md | Matches current Vercel/Render/Supabase intent, but contains deployment-specific identifiers intentionally omitted here |
| docs/model_comparison.md, production_model.md, training_results.md | Match generated candidate artifacts and incomplete selection |
| docs/future_model_roadmap.md | Clearly describes unimplemented future work rather than current features |
| PROJECT_STATUS_AND_TECHNICAL_AUDIT.md | Obsolete: incorrectly says no auth, SQLite-only, legacy MobileNet served, no release downloader, no frontend tests, and broken Docker |
| models/model_config.json | Obsolete six-class EfficientNet-B0/mock placeholder; not used by current backend |
| docs/screenshots/*.png | Stale pre-authentication design |
| scripts/setup_env.sh and verify_clean_clone.py | Stale database configuration assumptions |

### Unavailable information

- Live production availability, configured external OAuth values, and hosted database migration state: **Needs manual verification**.
- PlantDoc samples/statistics: **Not found in the repository**.
- Field-test accuracy/generalization: **Not found in the repository**.
- ConvNeXt-Base/Swin candidate results: **Not found in the repository**.
- Test coverage percentage and CI status: **Not found in the repository**.
- Dataset license/version checksum manifest: **Not found in the repository**.
- Formal user research, measured accessibility score, load SLO, or incident history: **Not found in the repository**.

### Secret and personal-data handling in this report

No password, OAuth secret, session token, API key, database connection string, deployment secret value, private project reference, private URL, survey respondent identity, or absolute personal path is reproduced. Environment variable names and aggregate database/dataset counts are included because they do not reveal secret values.

---

## 23. Final Summary Tables

### 1. Project completion status

| Area | Status | Evidence-based conclusion |
|---|---|---|
| Full-stack MVP | Mostly complete | Authenticated scan, result, dashboard, history, profile, and persistence work |
| ML training pipeline | Complete core | Reproducible training/evaluation/export exists |
| Production model | Complete active v1 | EfficientNetV2-S release verified and served by default |
| Comparative model selection | Incomplete | ConvNeXt-Base absent; require_all_candidates blocks selection |
| Field-data program | Operationally incomplete | Tooling exists; 409 groups pending, zero eligible |
| Production configuration | Present | Vercel/Render/Supabase/Docker files exist |
| Live production verification | Needs manual verification | External state is not established by repository evidence |
| Quality automation | Partial | Strong Python/API contracts; missing E2E, coverage, lint, CI |

### 2. Feature status

| Feature | Status | Main evidence |
|---|---|---|
| Google login/session/CSRF/logout | Complete | backend/api/auth.py, routes/auth.py |
| Single prediction | Complete | backend/api/routes/predict.py |
| Batch prediction | Partial | Implemented but sequential/non-atomic |
| Quality assessment | Complete for current rules | backend/api/routes/predict.py |
| Dashboard/history/profile | Complete | frontend/src/pages/, system.py |
| Disease guidance | Partial, 6/15 | seed_disease_data.py |
| Feedback | Partial | API exists; UI/validation gaps |
| PostgreSQL production schema | Complete in migration | supabase/migrations/ |
| Model download/release verification | Complete | download_model.py, model_release.py |
| Field label review | Tool complete; decisions absent | review_field_survey.py, validated manifest |
| Explainability/severity/segmentation | Planned only | docs/future_model_roadmap.md |

### 3. Technology stack summary

| Layer | Main technologies | Version highlights |
|---|---|---|
| Frontend | React, React DOM, Vite, Recharts, fetch, CSS | React 18.3.1; Vite resolved 5.4.21 |
| Backend | Python, FastAPI, Uvicorn, Pydantic, Pillow, OpenCV | Python 3.11.15 Docker; FastAPI 0.136.3 |
| Inference | ONNX Runtime, NumPy | Backend ORT 1.26.0 |
| Training | PyTorch, timm, Albumentations, scikit-learn | Artifact PyTorch 2.12.1; timm 1.0.28 |
| Database | PostgreSQL/Supabase, SQLite, psycopg | Local Supabase Postgres major 17; psycopg 3.3.2 |
| Testing | pytest, FastAPI TestClient, Node test | pytest 8.4.2 |
| Deployment | Docker, Render, Vercel, Supabase CLI | Platform versions not pinned |

### 4. API summary

| Method | Endpoint | Auth policy | Purpose |
|---|---|---|---|
| GET | /health | Public | Model/database readiness |
| GET | /classes | Public | Active class order |
| GET | /auth/config | Public | OAuth readiness |
| GET | /auth/google/login | Public/state | Begin OAuth |
| GET | /auth/google/callback | Public/state | Complete OAuth |
| GET | /auth/session | Session | Restore user |
| POST | /auth/logout | Session + CSRF | Revoke session |
| POST | /predict | Session + CSRF | One prediction |
| POST | /predict/batch | Session + CSRF | Up to ten predictions |
| GET | /disease/{class_name} | Public | Reviewed guidance |
| GET | /history | Session | Private scan records |
| GET | /dashboard | Session | Private aggregate summary |
| POST | /feedback | Session + CSRF | Save owned feedback |

### 5. Dataset statistics

| Dataset | Samples/files | Classes | Split | Integrity/training state |
|---|---:|---:|---|---|
| PlantVillage processed | 20,638 | 15 | 14,447 / 3,097 / 3,094 | All decode; 14 duplicate pairs; no cross-split/class leakage |
| PlantVillage raw storage | 20,638 logical, stored twice | 15 | Not applicable | Two byte-identical 342.23 MB trees |
| Field survey | 3,510 records; 3,508 local files | 322 raw label buckets; canonical unavailable | None | 125 missing refs, 2 ambiguous, 409 review groups pending, zero eligible |
| PlantDoc | Not found | Not available | Skipped | Optional configuration only |

### 6. ML model statistics

| Model | Status | Input | Test accuracy | Macro F1 | ONNX size | Fair one-thread ORT median |
|---|---|---:|---:|---:|---:|---:|
| EfficientNetV2-S v1 | Active complete release | 300 | 0.998707 | 0.998905 | 76.94 MiB | 734.724 ms |
| ConvNeXt-Tiny | Complete candidate, not promoted | 224 | 0.998707 | 0.998954 | 106.24 MiB | 518.296 ms |
| Phase 1 EfficientNetV2-S | Incomplete epoch 23/30 | 224 configured | Not available | Not available | Not exported | Not available |
| Legacy MobileNetV3 | Legacy, not served | 224 | Not available | Not available | 16.87 MB | Not available |
| ConvNeXt-Base | Not found | Configured native | Not available | Not available | Not available | Not available |

### 7. Codebase statistics

| Metric | Value |
|---|---:|
| Audit-scope files | 65,642 |
| File-stat total after model/checkpoint/logit binary exclusion | 65,630 |
| Version-controlled files | 134 |
| Tracked source files / lines | 92 / approximately 12,887 |
| Python files / lines | 60 / 10,560 |
| Frontend components / page components | 9 / 5 |
| Backend endpoints | 13 |
| Production tables / ORM models | 7 / 0 |
| Implementable architectures / local artifact bundles | 8 / 4 |
| Configured / present / training-eligible dataset families | 3 / 2 / 1 |
| Test files / test cases | 20 / 84 |
| Configuration files | 20 |
| Pre-report Markdown docs / screenshot docs | 10 / 2 |

### 8. Testing status

| Test/build area | Result | Remaining gap |
|---|---|---|
| Python data/ML/API suite | 73 passed, 1 optional PostgreSQL skipped | Configure TEST_DATABASE_URL for live PostgreSQL case |
| Frontend Node suite | 10 passed, 0 failed | No component/browser tests |
| Production frontend build | Passed, 845 modules | 525.55 kB main chunk warning |
| Coverage | Not available | Add measurement and thresholds |
| Lint/type/format | Not found | Add configured tools |
| CI/CD | Not found | Automate tests/build/container/migration |
| E2E/a11y/load/security | Not found | Add dedicated suites |

### 9. Known issues

| Priority | Issue | Effect | Evidence |
|---:|---|---|---|
| 1 | No validated external/field benchmark | Laboratory metrics may not generalize | field manifest, roadmap |
| 2 | ConvNeXt-Base missing | Formal selection blocked | model_comparison.md |
| 3 | Only 6/15 guidance rows | Incomplete actionable response | seed_disease_data.py |
| 4 | Synchronous prediction in async route | Event-loop blocking under load | predict.py |
| 5 | No rate limit/queue | Abuse and overload risk | backend inventory |
| 6 | Field survey contains pending/possibly personal data | Privacy and label-quality risk | field manifests |
| 7 | Weak feedback schema and no UI | Validation/usability gap | schemas.py, frontend pages |
| 8 | No E2E/CI/coverage/lint | Regression assurance gap | repository inventory |
| 9 | Large single frontend chunk | Slower initial load | verified Vite build |
| 10 | Obsolete audit/config/screenshots/scripts | Developer/reporting confusion | root audit, model_config, screenshots |
| 11 | EfficientNet temperature at lower bound | Calibration deserves review | model comparison/calibration artifact |
| 12 | Raw PlantVillage duplicated on disk | Wastes about 342 MB | calculated hash audit |

### 10. Recommended next steps

| Order | Action | Expected outcome | Acceptance evidence |
|---:|---|---|---|
| 1 | Expert-review and privacy-audit field data; create locked external test set | Honest field-generalization evidence | Approved manifest, provenance, held-out report |
| 2 | Complete reviewed guidance for all 15 classes | Consistent user output | Seed/API tests for every class |
| 3 | Train ConvNeXt-Base on unchanged split and run formal selection | Complete candidate decision | All candidates scored; signed-off report |
| 4 | Review calibration and add OOD/abstention thresholds | Safer confidence behavior | External calibration/OOD metrics |
| 5 | Bound/offload inference, add rate limits, load test, and make batch semantics explicit | Predictable production performance | p95/memory/error SLO report |
| 6 | Add routing/lazy loading, feedback UI, accessible chart alternative, focus styling | Better UX/accessibility and smaller initial bundle | E2E/a11y/build-size checks |
| 7 | Add CI with lint, type checks, 84+ tests, PostgreSQL, build, Docker, migrations, security scans | Continuous reproducibility | Passing protected workflow |
| 8 | Pin training and deployment-tool versions; add dataset provenance/checksums | Reproducible environments and data | Lockfiles and provenance manifest |
| 9 | Add monitoring, backup/restore drill, user export/deletion, retention policy | Operational and privacy readiness | Documented tested procedures |
| 10 | Archive or correct stale audit, model config, screenshots, and scripts | One trustworthy source of truth | Documentation cross-check passes |
