# Leaflight Project Status and Technical Audit

**Audit date:** 2026-07-15  
**Repository:** `C:\Users\soumy\crop-disease-detection`  
**Audited branch/state:** `main`, 19 commits ahead of `origin/main`. At the final integrity check, the worktree was clean except for this requested untracked report. The 19 local commits include a concentrated data/training/backend/test/documentation update made on the audit date.  
**Audit mode:** Read-only inspection plus isolated builds, tests, model parity checks, and localhost smoke tests. No dependencies were installed, no training was started, and no project source was changed. The only added repository file is this requested report.

## Evidence convention

- **Code-confirmed** means the implementation is imported/routed/called in the current source.
- **Artifact-confirmed** means a checkpoint, metric, manifest, SQLite row, or exported model is present on disk.
- **Log-confirmed** means a saved log records the action and result.
- **Executed** means the audit ran the command successfully on 2026-07-15.
- **Configured, not executed** means code/configuration exists without a completed artifact or audit run.
- **Documentation-only** means a document mentions the capability but the implementation or artifact is absent.
- When evidence is absent, this report states **Not found in the codebase.**

# Executive Summary

Leaflight is a full-stack crop-disease image-classification project. Its demonstrable path is React/Vite UI → FastAPI → an ONNX Runtime classifier → SQLite disease guidance/history. It also contains a substantially more mature offline ML system for deterministic dataset splitting, field-survey review, resumable multi-backbone training, calibration, evaluation, ONNX parity checking, and model selection (`README.md:1-3`, `README.md:29-53`).

**Overall project completion: 63%.** This is a weighted evidence score, not an estimate of elapsed effort: runtime MVP 76% × 45%, ML lifecycle 66% × 30%, and assurance/production readiness 36% × 25% = **63.0%**. The detailed rubric is in Section 13.

**Current status:** partially working, under active development, incomplete, conditionally ready for a local demonstration, and not ready for production.

The strongest confirmed results are:

- The main frontend production build succeeded: Vite 5.4.21 transformed 890 modules in 5.90 s.
- The Python suite passed: 23/23 ML/data/training tests in 53.83 s. The isolated backend suite passed: 7/7 tests in 1.09 s.
- A real local smoke test loaded `models/onnx/model.onnx` with `CPUExecutionProvider`, returned `mock: false`, correctly classified a held-out Pepper bacterial-spot image at `0.9999972582`, and persisted the result in an isolated SQLite copy. Saved logs independently record successful `/predict` requests (`backend.live.20260714-114138.err.log:3,37,187`).
- The deployed MobileNetV3 ONNX output matches its PyTorch checkpoint on a two-image random batch (`max absolute logit error = 1.5020e-5`, `allclose(rtol=1e-4, atol=1e-5) = true`).
- A newer Phase 2.5 EfficientNetV2-S run is genuinely complete: 24 epochs were executed before macro-F1 early stopping, best epoch 16, test accuracy `0.998707`, macro F1 `0.998905`, macro ROC-AUC `0.999953`, calibrated test ECE `0.001292`, and ONNX parity passed (`artifacts/training/crop_disease_phase2_5/efficientnetv2_s/run_state.json:2-5`; `metrics.json:2-9,245,251-256,511,517-562`).

The main blockers are:

1. **The evaluated Phase 2.5 model is not the model being served.** The backend serves a 224 px MobileNetV3 bundle (`models/onnx/model.json:2-20`, SHA-256 `65498071...7044E`), while the completed 300 px EfficientNetV2-S ONNX has a different hash (`BD0AF61C...0BEAA`) and remains under ignored `artifacts/`.
2. **A clean clone cannot produce the current backend image.** `.gitignore:18-22` excludes checkpoints, all `artifacts/`, and the entire `models/` tree except already tracked `models/model_config.json`; `backend/Dockerfile:11` nevertheless requires `models/onnx`. There is no model download/release step.
3. **The required benchmark is incomplete.** EfficientNetV2-S is complete, ConvNeXt-Tiny stopped after epoch 6/40, ConvNeXt-Base has not started, and selection requires every configured candidate (`configs/training/phase2_5.yaml:73-75,108,139-145`). No production bundle exists; the saved finalizer says `selected=none` (`phase2.finalize.20260714-145122.out.log:3`).
4. **There is no authentication, authorization, user model, rate limiting, or admin boundary.** All prediction, history, and feedback routes are public.
5. **Nine of 15 disease records contain placeholder expert-review guidance.** The classifier can emit near-1.0 confidence, but there is no out-of-distribution rejection or field-validation benchmark. This is unsuitable for autonomous agricultural treatment decisions.
6. **Production assurance is thin.** There is no CI/CD workflow, frontend test suite, end-to-end suite, coverage report, Python lockfile, migration system, backup plan, frontend container, reverse proxy, monitoring, or deployment health check.

# 1. Project Overview

## Identity, purpose, and users

- **Project name:** Leaflight — Crop Disease Detection (`README.md:1`).
- **Purpose:** Analyze plant-leaf images, classify one of 15 pepper/potato/tomato health or disease labels, display confidence and disease guidance, and retain scan history (`README.md:3`, `data/splits/phase1_split.json:9-24`).
- **Target users:** Farmers, field workers, agricultural students, agronomists, and researchers are inferred from the UI's field-photo instructions, extension-service guidance, and field-survey review pipeline. No formal user-research or persona document was found.
- **Product boundary:** The current application is anonymous and advisory. It is not a diagnosis/treatment authority, multi-user platform, or production MLOps service.

## Major features

- Leaf image selection, drag/drop, camera capture, preview, loading, and error states.
- Real ONNX classification with top-3 probabilities.
- Disease symptoms, static severity, and treatment guidance.
- Recent-scan history and analytics dashboard backed by SQLite.
- Helpful/not-helpful feedback persistence.
- Health, class-list, single prediction, batch prediction, disease lookup, history, and feedback APIs.
- Deterministic PlantVillage split manifest and duplicate-aware data registry.
- Field-survey ingestion, label cleaning, local human-review UI, and a hard training-eligibility gate.
- Resumable PyTorch/timm training with Albumentations, MixUp/CutMix, EMA, AMP, class weighting, calibration, evaluation, ONNX export/parity, latency measurement, and multi-objective selection.

## Architecture

```text
Main runtime
Browser -> React 18/Vite SPA -> Axios -> FastAPI routers
                                      |-> singleton ONNX Runtime session
                                      |-> SQLite diseases/scans/feedback

Offline ML lifecycle
Kaggle/manual/field data -> registry/validation -> persisted split
-> PyTorch+timm training -> evaluation/calibration -> ONNX parity/benchmark
-> intended production bundle (not yet selected or deployed)
```

The runtime relationship is explicitly documented at `README.md:53`. The global `ModelService` is instantiated once (`backend/api/model_loader.py:108`) and loaded during application lifespan (`backend/main.py:26-31`), rather than on each request.

## Technologies

| Layer | Declared/current technology | Version evidence |
|---|---|---|
| Main frontend | React, React DOM, Vite, Axios, Recharts, JavaScript ES modules | Declared React `^18.3.1`, Vite `^5.4.11`; installed React 18.3.1, Vite 5.4.21 (`frontend/package.json:5-19`; audit environment) |
| Styling | Handwritten responsive CSS; Google Fonts; CSS animations | `frontend/src/styles/index.css:1,723-767` |
| Routing/state | Component-local React state only | No React Router or state library in `frontend/package.json`; `frontend/src/App.jsx:7-15` |
| Backend | FastAPI, Uvicorn, Pydantic, Pillow, Python multipart | Installed FastAPI 0.139.0, Uvicorn 0.51.0, Pydantic 2.13.4; requirements are unpinned (`backend/requirements.txt:1-10`) |
| API | REST-style JSON plus multipart uploads | Three routers included at `backend/main.py:60-62` |
| ML | PyTorch, torchvision, timm, Albumentations, scikit-learn | Installed PyTorch 2.12.1+cu130, torchvision 0.27.1+cu130, timm 1.0.28; requirements unpinned (`requirements.txt:1-16`) |
| Serving | ONNX Runtime with CUDA-then-CPU provider selection | `backend/api/model_loader.py:58-71`; audit runtime used CPU |
| Database | SQLite through standard-library `sqlite3`; no ORM | `backend/api/routes/disease_info.py:3,16-19` |
| Logging | Python logging plus rotating file handler | 2 MB × 5 backups (`backend/main.py:19-23`) |
| Deployment | One Python 3.11 backend Dockerfile; documentation for Railway/Render/Vercel | `backend/Dockerfile:1-14`; `docs/deployment.md:3-33` |
| External services | Kaggle CLI for dataset download; optional GitLab CLI MCP config for development | `src/data/download_data.py:58-80`; `opencode.json:2-8` |

The local audit runtime was Python 3.14.5, Node.js 24.13.0, and npm 11.6.2; the backend container instead targets Python 3.11. Installed frontend packages included Axios 1.18.1 and Recharts 2.15.4. These installed versions are environment evidence, not Python lockfile guarantees.

Hugging Face, TensorBoard event files, Weights & Biases runs, MLflow runs, cloud-storage SDKs, and hosted database integrations: **Not found in the codebase.** `.gitignore:23-24` merely reserves `wandb/` and `mlruns/`.

## Important directories

| Path | Responsibility | Audit status |
|---|---|---|
| `frontend/` | React/Vite product UI, components, pages, API client, styles | Implemented; builds; no tests/lint/type checking |
| `backend/` | FastAPI app, routes, model service, schemas, SQLite seed/database, backend tests, Dockerfile | Functional core; security, validation, migration, and deployment gaps |
| `src/data/` | Download, split, transforms, dataset registry, multi-source loaders, field-survey pipeline and review app | Substantial and tested; field survey has zero approved training records |
| `src/models/` | Baseline CNN and timm model factory | Implemented; several backbones are code-only |
| `src/training/` | Config, engine, checkpoint/resume, ONNX export, benchmark/selection | Strong pipeline; required candidate execution incomplete |
| `src/evaluation/` | Metrics, calibration, evaluation, error analysis | Implemented; complete artifact only for one Phase 2.5 candidate |
| `src/inference/` | Canonical preprocessing, PyTorch prediction function, legacy ONNX exporter | Runtime preprocessing reused by backend; CLI exports but does not classify a supplied image |
| `configs/training/` | Phase 1 and Phase 2.5 YAML contracts | Present and tracked |
| `data/processed/` | Current 70/15/15 PlantVillage folders | Present locally, ignored by Git |
| `data/splits/` | Persisted split manifest | `phase1_split.json` is tracked |
| `data/manifests/field_survey/` | Large ingestion/cleaning/validation artifacts | Present locally, ignored by Git, contains personal survey metadata |
| `models/` | Legacy serving checkpoint, ONNX bundle, stale model config | ONNX/checkpoint local and ignored; only stale `model_config.json` tracked |
| `artifacts/training/` | Phase 1 and Phase 2.5 checkpoints, metrics, plots, exports | Present locally, ignored by Git; not reproducible from a clean clone |
| `tests/`, `backend/tests/` | ML/data/training and API tests | 23 + 7 tests passed |
| `docs/` | Deployment, roadmap, training reports, screenshots | Useful but several generated status documents are stale |
| `scripts/` | Setup, deployment guidance, training launcher, screenshot capture | Mixed Bash/Windows assumptions; no full automation |
| `notebooks/` | Legacy notebooks | **Not found in the codebase.** Their deletion is committed in `25b104c` |

# 2. Overall Project Status

| Classification | Finding | Evidence |
|---|---|---|
| Fully working | No | Several individual paths work, but fresh-clone model setup, production selection, auth/security, and deployment are incomplete |
| Partially working | Yes | Build/tests/live API/model smoke tests passed; supporting and production flows remain partial |
| Under active development | Yes | Branch is 19 local commits ahead of origin, most dated on the audit day; ConvNeXt state is interrupted; docs/code/artifacts disagree |
| Incomplete | Yes | Required three-model benchmark is only one complete candidate plus one partial candidate |
| Broken | Not globally | Batch-model-unavailable handling, clean Docker build, and custom `DB_PATH` seeding are specifically broken |
| Demo-ready | Conditional yes | Works locally when ignored ONNX files and SQLite database are already present |
| Production-ready | No | No reproducible model release, auth/rate limiting, CI/CD, migrations, persistent storage plan, or external field validation |

The 63% overall score is deliberately lower than the working demo score. A polished local scan does not establish reproducibility, field generalization, access control, or operational safety.

# 3. Frontend Status

**Frontend completion: 78%.**

## Frontend technologies

- Framework: React 18.3.1 with Vite 5.4.21.
- Language: JavaScript/JSX, not TypeScript.
- Styling: custom CSS, CSS variables/animations, remote Google Fonts (`frontend/src/styles/index.css:1`).
- State management: local `useState`, `useEffect`, and `useMemo`; no global state library.
- Routing: no URL router. `App` switches an in-memory `activePage` map (`frontend/src/App.jsx:7-15`).
- Components: custom components; no Material UI, Tailwind, Bootstrap, or similar library.
- Charts: Recharts (`frontend/src/pages/Dashboard.jsx:2,80-88`).
- Animation: CSS only; reduced-motion handling exists (`frontend/src/styles/index.css:762-767`).
- API transport: Axios with a 30 s timeout (`frontend/src/services/api.js:1-7`).

## Pages, routes, forms, and major components

The three main screens all remain at browser URL `/`; selecting them does not change history or support deep links.

| URL/internal page | Purpose | Status | Data | Relevant files |
|---|---|---|---|---|
| `/`, `activePage="home"` / Scan | Upload/capture a leaf, run prediction, show result and recent scans | Working with minor issues | Real `/health`, `/predict`, `/history` | `frontend/src/App.jsx:14-25`; `pages/Home.jsx`; `components/ImageUpload.jsx`; `PredictionResult.jsx`; `ScanHistory.jsx` |
| `/`, `activePage="dashboard"` | Aggregate scan counts, common class, mean confidence, healthy/diseased ratio, frequency chart, record cards | Working | Real `/history?limit=100` | `App.jsx:28-30`; `pages/Dashboard.jsx:21-110` |
| `/`, `activePage="about"` | Static project/method explanation | Complete static screen | Static | `App.jsx:31-33`; `pages/About.jsx` |
| `http://127.0.0.1:8765/` (separate utility) | Review field-survey groups; filter/search; accept, replace, or reject labels | Code-complete, data workflow not completed | Local `/api/queue`, `/api/decisions`, survey images | `src/data/review_app/index.html:14-58`; `app.js:4-70`; `review_field_survey.py:241-322` |

No modal or dialog component was found. Main forms/controls are the file picker, drag/drop target, camera picker, top-three toggle, and two feedback buttons. The review utility contains reviewer/replacement/note inputs and decision controls.

| Component | Function | Status/issues |
|---|---|---|
| `ImageUpload` | MIME/10 MB validation, drag/drop, picker, camera | Working; accepts WebP in picker while copy says JPG/PNG; drop/camera accepts any `image/*`; keyboard activation missing on the role-button drop zone (`ImageUpload.jsx:21-24,53-62`) |
| `PredictionResult` / `ScannerFrame` | Preview, state, top result, alternatives, guidance, feedback | Working; hardcodes `224 RGB`, so it misdescribes the completed 300 px model (`PredictionResult.jsx:18-24`) |
| `ConfidenceBar` | Percent/bar display | Working; not a semantic progress bar and value is not clamped (`ConfidenceBar.jsx:1-11`) |
| `DiseaseInfoCard` | Symptoms, treatment, severity | Working with backend data; quality limited by placeholders |
| `ScanHistory` | Recent persisted scans | Working with real history (`ScanHistory.jsx:39`) |
| `Dashboard` | Statistics/chart/cards | Working; initial fetch only and no user refresh/filter |

## Feature status

| Feature | Status | Evidence/notes |
|---|---|---|
| Authentication pages | **Not found in the codebase.** | No login/register components or dependency |
| Dashboard | Working | Real history and Recharts; loading/error/empty states |
| Image upload/camera | Working with validation gaps | `ImageUpload.jsx:21-77` |
| ML prediction | Working against current backend | `Home.jsx:36-49`; live API smoke test |
| Results/top-3/confidence | Working | `PredictionResult.jsx:69-105` |
| History/records | Working | Scan page and dashboard call real API |
| Search/filter | Main product: **Not found in the codebase.** Review utility: implemented | `review_app/index.html:20-31`; `app.js:58-65` |
| Forms/validation | Partial | File size/MIME only; no schema/form library |
| Notifications/errors | Partial | Inline errors and feedback status; Home discards the normalized API error detail (`Home.jsx:45`) |
| Loading states | Working | Upload analysis, dashboard skeleton, health checking |
| Responsive design | Implemented in CSS | Breakpoints at 960/860 px; only desktop screenshots were found (`styles/index.css:723-761`) |
| Accessibility | Partial | Reduced motion and several ARIA labels; keyboard drop-zone and confidence semantics are incomplete |
| Dark mode | **Not found in the codebase.** | No theme toggle or alternate scheme |
| Animations | Working | CSS scan/pulse/grow/shimmer animations |
| User profile/admin | **Not found in the codebase.** | No user concept |

## Frontend integration

- API base URL uses `VITE_API_URL` with local hardcoded fallback `http://127.0.0.1:8000` (`frontend/src/services/api.js:3`; `.env.example:1`). This is acceptable for development but must be set at build time for deployment.
- Single prediction, history, health, and feedback request/response shapes match backend schemas. `getDiseaseInfo()` exists but is unused; there are no frontend clients for `/classes` or `/predict/batch` (`api.js:23-63`).
- Authentication tokens are neither stored nor sent because authentication does not exist.
- `Home` marks the backend “online” from `health.status === "ok"` without checking `model_loaded` (`Home.jsx:18-24`). Backend status is based only on database connectivity (`system.py:21-25`), so the UI can show `MODEL LIVE` while predictions return 503.
- The UI renders whatever the backend serves correctly, but it does **not** display the latest Phase 2.5 output because the backend still points to the older bundle.
- Loading and common error states exist, but `Home` replaces useful server errors with one generic message (`Home.jsx:39-46`).
- `URL.createObjectURL()` is never revoked (`Home.jsx:16`), creating a small repeated-upload memory leak.

## Frontend quality and verification

- **Executed build:** success, 890 modules, 5.90 s. Output: HTML 0.41 kB, CSS 11.67 kB, JS 575.70 kB (171.06 kB gzip).
- **Build warning:** the main JS chunk exceeds 500 kB. There is no lazy loading/manual chunking.
- TypeScript errors: not applicable; TypeScript is not used.
- Lint result: **Not found in the codebase.** No lint script/config exists.
- Frontend tests: **Not found in the codebase.**
- Duplicate component implementations, broken imports, dead hyperlinks, and obvious placeholder action buttons: none found. Navigation uses buttons rather than links, so the larger problem is missing URL routing/history.
- Fresh browser console pass: not completed because the in-app browser was unavailable. The saved Vite error log is empty and the saved Vite output reports readiness in 1517 ms (`frontend.live.20260714-114138.out.log:6`), but that is not equivalent to a current console test.
- Repository screenshots from 2026-07-13 visibly confirm the desktop Scan and Dashboard screens (`docs/screenshots/leaflight-scan.png`, `leaflight-dashboard.png`). The dashboard screenshot predates the current About navigation, so screenshots are not a full current-state test.
- The separate review UI has mojibake characters (`Â·`, `â†’`) in `src/data/review_app/app.js:25,29` and permits arbitrary replacement labels rather than restricting them to the ontology (`review_field_survey.py:174-196`).
- The remote Google Fonts import creates an offline/privacy/CSP dependency.

**Frontend score calculation:** navigation/screens 14/15; upload/prediction/results 23/25; history/dashboard/feedback 17/20; integration/resilience 11/15; responsive/accessibility 8/10; test/build/optimization assurance 5/15 = **78/100**.

# 4. Backend and API Status

**Backend completion: 72%.**  
**API integration completion: 78%.**

## Backend technologies and pattern

- Python FastAPI application with modular routers and a singleton inference service; REST-style JSON/multipart API.
- Pydantic response/request models, but without field constraints (`backend/api/schemas.py:6-51`).
- Standard-library SQLite, no ORM/repository abstraction.
- No authentication approach. All routes are public.
- Pillow validation/preprocessing, ONNX Runtime inference.
- Rotating file and console logging (`backend/main.py:19-23`).
- Dependency versions are not pinned in `backend/requirements.txt`; exact versions above describe only the audit environment.

## API endpoint inventory

| Method | Endpoint | Purpose | Auth | Request | Response | Frontend consumer | Status | Evidence |
|---|---|---|---|---|---|---|---|---|
| GET | `/health` | Database/model status | None | None | `{status, model_loaded, model_mode, db_connected}` | Scan health badge | Partial: works, but `status` ignores model availability | `system.py:13-25`; executed `200` with loaded ONNX |
| GET | `/classes` | Class order from loaded metadata | None | None | JSON list of 15 strings; empty if unavailable | None | Working, unused | `system.py:29-31`; executed count 15 |
| POST | `/predict` | Classify one image, enrich guidance, save scan | None | Multipart field `file` | `PredictionResponse` with top 3/guidance/mode/mock | Scan page | Working with validation/scalability issues | `predict.py:57-66`; real audit prediction and saved logs |
| POST | `/predict/batch` | Sequentially classify images | None | Repeated multipart `files` | List of `PredictionResponse` | None | Partial/unused; model-unavailable exception becomes 500 and no batch limit | `predict.py:69-79` |
| GET | `/disease/{class_name}` | Fetch reviewed disease record | None | Path class | `DiseaseInfo` or 404 | API client function exists but unused | Working for reviewed entries; 404 for placeholders/unknowns | `disease_info.py:22-50`; `api.js:34-37` |
| GET | `/history?limit=` | Return newest scans | None | Query integer, clamped 1–200 | List of id/timestamp/class/confidence/hash | Scan and Dashboard | Working | `system.py:34-47` |
| POST | `/feedback` | Store classification feedback | None | JSON `{predicted_class, confidence?, message?}` | `{"status":"received"}` | Result buttons | Partial: weak validation and no scan/user relation | `system.py:50-58`; `PredictionResult.jsx:72-79` |

## Backend feature status

| Feature | Status | Evidence/issues |
|---|---|---|
| Registration/login | **Not found in the codebase.** | No users table or auth route |
| Role authorization/admin | **Not found in the codebase.** | All routes public |
| File upload | Working | Multipart single and batch routes |
| Image validation | Partial | Size after full read, broad MIME, Pillow decode; no dimension/pixel/decompression-bomb or batch-total limit (`predict.py:18-28,59,75`) |
| ML prediction | Working with current local bundle | Singleton ONNX session; fail-closed single endpoint |
| Data storage/history | Working | SQLite scans/feedback |
| Search/filter | **Not found in the main API.** | History only has `limit` |
| Error handling | Partial | Single model unavailable → 503; batch does not catch it; DB and unexpected image exceptions can become 500 |
| Logging | Working basic logging | Method/path/status/duration; exceptions before response bypass the timing log (`main.py:45-57`) |
| Rate limiting | **Not found in the codebase.** | Public CPU-intensive endpoint |
| Input validation | Partial | Unconstrained Pydantic strings/floats (`schemas.py:6-51`) |
| CORS | Configurable | Defaults to two local origins; credentials true and all methods/headers (`main.py:36-42`) |
| Security middleware | **Not found in the codebase.** | No trusted-host, security-header, size/concurrency, or request-ID middleware |
| Health check | Present but semantically weak | Database-only `status`; no process/readiness separation |

## Backend quality findings

1. **Blocking work in async routes:** `await file.read()` is followed by synchronous Pillow, ONNX, and SQLite operations in the event loop. Batch runs sequentially (`predict.py:58-78`).
2. **Memory pressure:** the entire upload is buffered before the 10 MB check. Batch has no file-count or combined-size ceiling.
3. **Broken unavailable behavior:** `/predict/batch` calls `model_service.predict()` without the single-route `ModelUnavailableError` handler.
4. **Relative-path fragility:** model and database defaults depend on process working directory (`config.py:14-15`; `model_loader.py:35-40`).
5. **Custom database path bug:** runtime queries honor `DB_PATH`, but seeding always writes `backend/db/disease_info.db` based on `__file__` (`disease_info.py:13-17`; `seed_disease_data.py:9,75-79`). A custom DB can therefore remain unseeded.
6. **Destructive startup schema action:** every start drops and recreates `diseases` (`seed_disease_data.py:79-90`). This is not a migration system.
7. **Weak schemas:** confidence is not constrained to `[0,1]`, messages/classes have no length limit, and stale response defaults still say `mode="mock"`, `mock=true` (`schemas.py:11-21,48-51`).
8. **Guidance formatting:** `_infer_crop_and_disease` leaves an empty segment for double underscores, producing values such as leading-space `" Bell Bacterial Spot"` (`seed_disease_data.py:68-72`; confirmed live response).
9. **No central exception policy:** database failures and unexpected decode/inference failures use generic 500 responses.
10. **Scalability:** one in-process global session is appropriate for one worker, but there is no worker/process sizing, queue/backpressure, caching, or concurrency plan. SQLite writes can contend under load.

No broken backend import was found: the backend test collection/imports succeeded. Request logs contain method, path, status, and duration rather than bodies or credentials; no secret exposure was found in the inspected logs.

**Backend score calculation:** core endpoint behavior 30/35; real inference 20/20; persistence 10/15; validation/error behavior 7/15; security/operations 5/15 = **72/100**.

**API integration score calculation:** single scan 25/25; health/history 20/25; feedback 9/10; supporting endpoints 7/15; contract/error handling 10/15; current runtime evidence 7/10 = **78/100**.

# 5. Machine Learning Model and Training Status

## Model inventory

| Model | Architecture/framework | Purpose/classes/input/output | Actual status |
|---|---|---|---|
| Current serving model | timm `mobilenetv3_large_100`, PyTorch checkpoint + ONNX | 15 PlantVillage classes; 224×224 RGB; `[N,15]` logits → temperature 1 softmax/top-3 | Real and loadable; checkpoint/ONNX parity passed; **no recorded accuracy, loss, precision, recall, F1, AUC, epoch, or test evaluation** |
| Legacy Phase 1 | timm EfficientNetV2-S (`tf_efficientnetv2_s`) | 15 classes; configured 224 px | Incomplete at epoch 23/30; best and last checkpoints present; no final test/calibration/export artifact |
| Phase 2.5 candidate 1 | timm EfficientNetV2-S | 15 classes; native 300 px; logits, calibrated softmax | Complete/evaluated/exported; best epoch 16; early stopped after epoch 24 |
| Phase 2.5 candidate 2 | timm ConvNeXt-Tiny | 15 classes; native timm preprocessing | Interrupted after epoch 6/40; resumable `last.pt`; no final metrics/calibration/ONNX |
| Phase 2.5 candidate 3 | timm ConvNeXt-Base | 15 classes planned | Configured, not started; **no checkpoint or run directory found** |
| Optional candidate | timm Swin-Tiny | 15 classes planned | Configured optional, not started (`phase2_5.yaml:75`) |
| Other factory models | EfficientNet-B0 and ResNet50 | Generic classifier support | Code-only; no matching artifacts found (`model_factory.py:8-16`) |
| Baseline | Custom `BaselineCNN` | Test/smoke baseline with configurable class count | Code and forward-shape test only; no saved trained model found |
| Stale config entry | `models/model_config.json` claims EfficientNet-B0, six classes, “training pending/mock” | Contradicts current 15-class MobileNetV3 serving metadata | Not a valid model artifact; stale configuration (`model_config.json:2-7`) |

All actual 15-class mappings match `data/class_mapping.json`, the split manifest, current serving metadata (`models/onnx/model.json:5-20`), and Phase 2.5 metadata. No label-index mismatch was found.

The timm transfer-learning runs are configured with pretrained weights. The serving checkpoint metadata records `pretrained=true`; Phase 2.5 resolves each backbone's native timm preprocessing contract (`model_factory.py:47-68`).

## Training status and metrics

| Run | Planned vs actual | Best/latest checkpoint | Actual measured metrics | Platform/config evidence | Status |
|---|---|---|---|---|---|
| Serving MobileNetV3 | Metadata *plans* 2 epochs; actual completed epoch count not recorded | `models/checkpoints/best_model.pth` (17,088,329 B); serving `model.onnx` (16,872,152 B) | **Not found in the codebase.** Only audit parity and one correct smoke prediction are known | Planned batch 64, LR 3e-4, AdamW, image 224, seed 42; device `auto` in checkpoint metadata | Trained weights exist, training/evaluation provenance incomplete |
| Phase 1 EfficientNetV2-S | 23/30 epochs | `best.pt` and `last.pt`, both epoch 23 and 325,300,773 B, different hashes | Epoch 23 train loss `0.834294`, train acc `0.999585`, val loss `1.155973`, val acc `0.946400`, LR `4.706e-5`; 7,656.17 s. Precision/recall/F1/AUC/test: **Not found** | `phase1.yaml:21-55`; actual device not persisted | Incomplete/interrupted; old checkpoint lacks current resume signature, so compatibility is doubtful |
| Phase 2.5 EfficientNetV2-S | 24/40 epochs; valid early stop after 8 epochs without exceeding epoch-16 macro F1 | Best compact checkpoint epoch 16 (81,674,915 B); `last.pt` epoch 24 (325,350,828 B) | Best epoch: train loss `0.966337`, train acc `0.945781`, val loss `0.970266`, val acc/F1 `1.0`. Test loss `0.974705`, accuracy `0.998707`, balanced acc `0.999063`, macro P/R/F1 `0.998752/0.999063/0.998905`, weighted F1 `0.998708`, macro ROC-AUC `0.999953`; 3,094 samples, 4 errors | CUDA; 20,196,703 params; image 300; batch 12 × accumulation 3 = 36; peak GPU 2,486,524,928 B; 16,157.95 s (`metrics.json:517-526`) | Complete |
| Phase 2.5 ConvNeXt-Tiny | 6/40 epochs | `best.pt` 111,403,795 B; `last.pt` 445,633,181 B | Epoch 6 train loss/acc `1.083694/0.907523`; val loss/acc/macro F1 `0.976808/0.996125/0.997236`; LR `1.9682e-4`. No final test metrics | Run state says running from 2026-07-14 but PID 34444 was absent at audit; stale lock remains | Interrupted/resumable, not currently training |
| Phase 2.5 ConvNeXt-Base | 0/40 | **Not found in the codebase.** | **Not found in the codebase.** | Configured batch 2, accumulation 16, LR 1e-4 (`phase2_5.yaml:89-92`) | Not started |

The Phase 2.5 EfficientNetV2-S run used local Windows/CUDA on an NVIDIA RTX 3050 6 GB laptop GPU according to `docs/training_results.md:5-7` and artifact `device="cuda"`. Kaggle is used for dataset acquisition, not as a recorded training platform. Hugging Face training/hosting evidence: **Not found in the codebase.**

No traceback or termination reason was recorded for the incomplete Phase 1 or ConvNeXt-Tiny runs; the cause is **Not found in the codebase.** ConvNeXt-Base's configured batch/accumulation strategy has not been proven on the recorded 6 GB GPU.

### Calibration and export evidence

- Temperature scaling fitted the lower clamp boundary `0.05` (`calibration.json:2-4`; implementation clamp at `calibration.py:86-94`).
- Validation ECE changed from `0.162980` to `0.0001296`; test ECE changed from `0.161257` to `0.001292` (`calibration.json:14,141,270,397`).
- ONNX parity passed at opset 18, batch sizes 1 and 2, maximum logit error `2.5034e-6`, maximum calibrated-probability error `2.1756e-6` (`metrics.json:549-562`).
- Measured median latency: PyTorch CUDA 66.85 ms, PyTorch CPU 625.89 ms, ONNX CPU 29.81 ms / 33.54 images/s (`metrics.json:527-562`).
- The extreme temperature and nearly perfect lab-dataset scores are warning signals, not proof of field performance. There is no OOD/field calibration result.

## Dataset status

### Current training dataset

- **Name/source:** PlantVillage mirror `emmarex/plantdisease` through Kaggle CLI or a manual ZIP (`src/data/download_data.py:6-7,58-104`). Dataset license metadata: **Not found in the codebase.**
- **Samples/classes:** 20,638 images, 15 classes (`README.md:58-60`).
- **Split:** seed 42, 70/15/15; 14,447 train, 3,097 validation, 3,094 test (`data/splits/phase1_split.json:2-7`; audit recount).
- **Optional sources:** PlantDoc and field survey were skipped in the current split (`phase1_split.json:43-45`).

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| `Pepper__bell___Bacterial_spot` | 698 | 150 | 149 | 997 |
| `Pepper__bell___healthy` | 1,035 | 222 | 221 | 1,478 |
| `Potato___Early_blight` | 700 | 150 | 150 | 1,000 |
| `Potato___Late_blight` | 700 | 150 | 150 | 1,000 |
| `Potato___healthy` | 106 | 23 | 23 | 152 |
| `Tomato_Bacterial_spot` | 1,489 | 319 | 319 | 2,127 |
| `Tomato_Early_blight` | 700 | 150 | 150 | 1,000 |
| `Tomato_Late_blight` | 1,336 | 286 | 287 | 1,909 |
| `Tomato_Leaf_Mold` | 666 | 143 | 143 | 952 |
| `Tomato_Septoria_leaf_spot` | 1,240 | 266 | 265 | 1,771 |
| `Tomato_Spider_mites_Two_spotted_spider_mite` | 1,173 | 251 | 252 | 1,676 |
| `Tomato__Target_Spot` | 983 | 211 | 210 | 1,404 |
| `Tomato__Tomato_YellowLeaf__Curl_Virus` | 2,246 | 481 | 481 | 3,208 |
| `Tomato__Tomato_mosaic_virus` | 261 | 56 | 56 | 373 |
| `Tomato_healthy` | 1,114 | 239 | 238 | 1,591 |

### Data quality audit

- A full read-only scan found all 20,638 manifest paths present, no extra processed files, and no unreadable processed images. All were 256×256; 20,637 decoded as RGB and one as RGBA.
- There are 20,624 unique SHA-256 contents: 14 duplicate pairs (28 files). No duplicate group crossed split or label boundaries. Exact content leakage is therefore controlled in the current split; perceptual/near-duplicate and same-plant/background correlation were not evaluated.
- Class imbalance is material: largest training class 2,246 vs smallest 106, a 21.19:1 ratio. Effective-number class weighting is enabled; balanced sampling is intentionally disabled to avoid double correction (`phase2_5.yaml:34-35`; `train.py:623-632`).
- Training augmentation includes geometry, illumination/color, CLAHE, blur/defocus, shadow/fog, JPEG degradation, occlusion, and mutually sampled MixUp/CutMix (`docs/training_results.md:13-19`; `engine.py:104-126`).
- PlantVillage is a controlled/lab-style source. No external field test, geographic holdout, device holdout, or temporal holdout is present. The near-perfect score should not be generalized to field use.
- The raw PlantVillage tree contains a second nested copy of the same 20,638 files; a full hash comparison found all corresponding files identical. It wastes storage but is not used by the current processed split.

### Field-survey dataset

- Ingestion manifest: 563 survey rows, 3,510 image records, 3,340 valid records, 125 missing images, 2 ambiguous image references, 27 invalid labels, and 357 duplicate-image-hash groups (`manifest.json:23-31`).
- Clean-label report: 43 missing-disease records, 141 unknown-value records, 890 multilingual/transliterated records, 3,006 requiring manual review, and 2,243 with at least one automatically normalized field (`dataset_report.md:7-16`). These categories overlap.
- Validation state: all 409 groups pending, 0 accepted/replaced/rejected, 0 validated records (`validated_manifest.json:11-20`). Therefore no field-survey record is eligible for training.
- A filesystem scan found 3,501 image files, 2,892 unique hashes, 369 duplicate-content groups, and 978 files participating in duplicate groups. This source needs substantial deduplication and curation.
- Survey manifests retain fields such as surveyor, college, and farmer name (`validated_manifest.json:36-50`). This is a privacy/retention risk even though the directory is Git-ignored.

## How current inference works

1. The user chooses/drops/captures a file in `ImageUpload`; frontend checks browser MIME and 10 MB size (`ImageUpload.jsx:21-24`).
2. Axios sends multipart field `file` to `/predict` (`frontend/src/services/api.js:23-31`).
3. FastAPI reads all bytes, checks length/MIME, verifies with Pillow, applies EXIF transpose, and converts RGB (`backend/api/routes/predict.py:18-28,58-60`).
4. Startup has already loaded the ONNX file and sibling metadata into the singleton service (`backend/main.py:26-31`; `model_loader.py:34-71`).
5. `preprocess_for_onnx` applies metadata-driven resize/crop/interpolation/normalization, or legacy stretch/ImageNet defaults when metadata omits the contract (`preprocess_input.py:38-94`).
6. ONNX Runtime produces logits; the service divides by saved temperature, applies stable softmax, and selects the top three (`model_loader.py:85-99`).
7. Indexes map through `idx_to_class` in model metadata.
8. The route looks up static disease guidance, inserts class/confidence/image SHA-256 into SQLite, and creates `PredictionResponse` (`predict.py:31-54`).
9. JSON returns class, confidence, alternatives, crop, disease, symptoms, treatment, severity, mode, and mock flag.
10. React displays the result, confidence, alternatives, guidance, feedback controls, and refreshed history (`Home.jsx:36-49`; `PredictionResult.jsx:69-121`).

## ML integration status

- Current trained model connected to backend: yes, but it is the legacy MobileNetV3 bundle.
- Predictions mocked: no. Current service fails closed when unavailable and returns `mock:false` when loaded (`model_loader.py:82-99`). Stale schema/config text still mentions mock mode.
- Model loaded once: yes, at startup.
- CPU/GPU: ONNX chooses CUDA then CPU when available (`model_loader.py:58-61`). Actual saved and audit logs used CPU.
- Training/inference preprocessing: the Phase 2.5 artifact has a complete timm preprocessing contract and shares canonical code. The serving metadata omits that contract, so exact training preprocessing is unrecorded and the backend uses legacy stretch/ImageNet defaults.
- Class order: consistent across current mapping, serving bundle, and completed candidate.
- Confidence: mathematically valid softmax; current serving model is uncalibrated (`temperature=1`). Completed candidate is calibrated in-domain but fitted to boundary `0.05` and lacks OOD safeguards.
- Model export: ONNX exists. TorchScript, TensorFlow Lite, Core ML, and quantized formats: **Not found in the codebase.**
- Latest checkpoint in production: no. The two ONNX hashes differ and no `production/` directory exists.

## ML problems and risks

1. Latest evaluated model is not deployed; serving model has no recorded evaluation metrics.
2. Required benchmark is 1 complete / 1 interrupted / 1 unstarted; selection is correctly blocked.
3. Controlled-source performance is likely optimistic for field photographs; no external test exists.
4. Temperature `0.05` reaches the lower bound and makes probabilities extremely sharp. No OOD, abstention, or confidence-threshold policy exists.
5. Near-duplicate/perceptual leakage was not assessed; exact duplicates are controlled.
6. Severe class imbalance remains despite weighted loss.
7. Field data is unusable for training until reviewed and deduplicated.
8. Several paths call `torch.load(..., weights_only=False)` (`train.py:361,647`; `inference/predict.py:23`; `evaluation/evaluate.py:21`). Untrusted checkpoints can execute pickle payloads.
9. `zipfile.extractall` is used without traversal validation in the dataset downloader, creating a Zip Slip risk for untrusted archives.
10. Python dependencies are unpinned; seeds and split hashes help but do not fully reproduce environments.
11. Phase 1's old checkpoint schema does not contain the current `training_signature` required by `_validate_resume_checkpoint` (`train.py:337-343`), so “resume” is not proven compatible.
12. Root live latency varied from 249.27 ms to 919.88 ms (`backend.live...err.log:37,187`); no production concurrency/load benchmark exists.

## ML component completion

| ML subcomponent | Completion | Basis |
|---|---:|---|
| Dataset preparation | 85% | Complete validated PlantVillage split; exact duplicate/corruption checks; optional real-world data still unusable |
| Training pipeline | 92% | Robust config/resume/EMA/AMP/augmentation/checkpointing and passing smoke tests; unsafe loads/unpinned env remain |
| Actual model training | 38% | Required candidates: 100% complete, 15% of planned epochs but no completion artifact, and 0%; mean 38.3% |
| Evaluation | 35% | Full evaluation for one required candidate; none for two candidates or serving model |
| Model export | 65% | Working legacy ONNX plus robust export for one candidate; no complete selected release |
| Inference pipeline | 82% | Real singleton ONNX, parity and smoke test; stale model/metadata and no OOD policy |
| Backend integration | 80% | Real connected model and correct class order; older bundle served |
| Frontend integration | 78% | Correct response display; wrong model status semantics and 224 px hardcode |
| Deployment readiness | 28% | Model files ignored; no release/download mechanism |
| Overall ML component | 66% | Weighted lifecycle score across the rows above |

# 6. Database Status

**Database completion: 60%.**

## Schema and current contents

Database type is SQLite; the project directly calls `sqlite3` and has no ORM. `backend/db/disease_info.db` passed `PRAGMA integrity_check` during read-only inspection.

| Table | Fields | Relationship/index status | Current local contents |
|---|---|---|---|
| `diseases` | `class_name` PK, crop, disease name, symptoms, treatment, severity | PK only; no relations | 15 rows: 6 reviewed entries, 9 expert-review placeholders |
| `scans` | auto ID, timestamp, predicted class, confidence, image hash | No FK to diseases, no checks, no custom indexes, duplicate hashes allowed | 31 rows, 7 distinct image hashes at pre/post-audit state |
| `feedback` | auto ID, timestamp, predicted class, nullable confidence/message | No FK to scan/class/user, no checks/indexes | 4 rows |

Schema creation is embedded in `seed_database()` (`seed_disease_data.py:75-138`). There are no user or uploaded-file tables; images are not stored, only SHA-256 hashes. Prediction history is implemented, but duplicate scans are intentionally/unintentionally retained.

## Database findings

- Connection works locally and the API health check verifies `SELECT 1`.
- Parameterized SQL is used for class lookup, inserts, and limits; direct SQL injection risk is low (`disease_info.py:24-30`; `predict.py:31-37`; `system.py:38-56`).
- The database is automatically created/seeded on API startup, but this is not migration-safe because `diseases` is dropped each time.
- Migrations/version table: **Not found in the codebase.**
- Seed fixtures separate from runtime seed: **Not found in the codebase.**
- Foreign keys/check constraints: **Not found in the codebase.**
- Backup/restore scripts or documented strategy: **Not found in the codebase.**
- At-rest encryption: **Not found in the codebase.**
- Custom `DB_PATH` is not honored by the seeder, as described in Section 4.
- SQLite files inside an ephemeral container will lose history unless a persistent volume is mounted; deployment docs do not define one.
- Database errors generally propagate as 500 responses; only `/health` catches connection failures.

**Score calculation:** schema/core storage 24/25; connection/CRUD 20/25; consistency/relationships/indexing 7/20; migrations/configuration 4/15; backup/security/error handling 5/15 = **60/100**.

# 7. Authentication and Security Status

**Authentication completion: 0%.**  
**Security completion: 35%.**

## Authentication and authorization

Registration, login, password hashing, JWTs, sessions, token expiration, protected routes, role permissions, users, profiles, and admin accounts: **Not found in the codebase.** This is confirmed by route/schema/database inventory, not merely by missing UI.

## Security controls present

- Local-origin CORS defaults and environment override (`config.py:16-23`).
- Parameterized SQLite queries.
- 10 MB per-file application limit and Pillow verification.
- Model-unavailable single request fails closed with 503.
- Environment-variable configuration examples; no committed real `.env`, credential, PEM, or key file was found.
- Kaggle credentials are expected only through environment variables (`download_data.py:60-67`).
- Review server defaults to loopback and restricts image paths to the manifest allowlist (`review_field_survey.py:270-280,319-320`).
- React text rendering and explicit escaping in much of the review UI reduce XSS exposure.

## Serious risks

| Severity | Risk | Evidence/impact |
|---|---|---|
| Critical for public deployment | Public prediction/history/feedback with no auth or rate limiting | Compute exhaustion, history exposure, feedback spam; every route has `Auth: None` |
| Critical for field advice | High-confidence classifier plus unreviewed guidance and no OOD abstention | 9/15 placeholder records; completed calibration at lower temperature bound; possible harmful treatment reliance |
| High | Deployment/model supply chain is not reproducible | Ignored model artifacts; no checksum-verified download/release in clean setup |
| High | Unsafe PyTorch checkpoint deserialization | Multiple `weights_only=False` calls can execute untrusted pickle code |
| High | Field-survey PII retained in large manifests | Farmer/surveyor/college metadata; no consent, redaction, retention, or access policy |
| High | Upload/batch denial of service | Full buffering before size check; no dimensions/pixel/batch/concurrency/rate limits; synchronous ONNX in async route |
| High | Unvalidated ZIP extraction | `extractall` without archive-path checks |
| Medium | CORS/security middleware | Credentials enabled with wildcard methods/headers; no security headers, TrustedHost, TLS termination, or request IDs |
| Medium | Feedback/schema abuse | Unbounded strings and confidence; no scan relation or deduplication |
| Medium | Public API documentation | FastAPI `/docs` and `/openapi.json` remain enabled by default; no production policy |
| Low/conditional | XSS/CSRF | Main React text is escaped and there is no session. If cookie auth is later added, CSRF protection will be required. Review UI inserts server-created image URLs into `innerHTML`; keep it loopback-only. |

Hardcoded real credentials/API keys: **Not found in the codebase.** Debug mode is not enabled in Docker; `--reload` appears only in local development instructions (`README.md:134-145`).

# 8. Testing and Code Quality Status

**Testing completion: 48%.**

## Commands actually executed

| Command/check | Result | Exact evidence |
|---|---|---|
| `.\.venv\Scripts\python.exe -m pytest tests -p no:cacheprovider -q` | Passed | `23 passed, 4 warnings in 53.83s`; process duration 68.4 s |
| Backend tests against an isolated temp SQLite copy | Passed | `7 passed, 1 warning in 1.09s` |
| `npm.cmd --prefix frontend run build` with temp output | Passed | 890 modules; built in 5.90 s; 575.70 kB JS chunk warning |
| Current serving checkpoint vs ONNX parity | Passed | `[2,15]`; max/mean abs error `1.5020e-5`/`6.2386e-6`; allclose true |
| Isolated localhost health | Passed | `200`, `status=ok`, `model_loaded=true`, `model_mode=onnx`, `db_connected=true` |
| Isolated real `/predict` | Passed | Correct Pepper bacterial spot, confidence `0.9999972582`, `mock=false`, request duration 233.95 ms, history row created in temp DB |
| Current interactive browser/console | Not completed | In-app browser unavailable; no claim of current console success |

### Failed or limited audit attempts

| Attempt | Result/exact error | Interpretation and recovery |
|---|---|---|
| First Vite build with an absolute `C:\tmp` output inside the restricted sandbox | Failed before application compilation: `Cannot read directory "../..": Access is denied` and `Could not resolve "...\\frontend\\vite.config.js"` | Sandbox traversal failure, not a source failure. The same build was rerun with permission to traverse dependencies, still writing only to temp, and succeeded |
| First backend temp-copy attempt using `C:\tmp` | `New-Item: Access ... is denied`; PowerShell treated copy errors as non-terminating and continued | Tests consequently ran once against the repository DB and inserted exactly one identifiable test scan. The audit compared against the recorded pre-test state, deleted only that row, reset `sqlite_sequence`, then reran all seven tests in the writable system temp directory |
| In-app browser connection | Browser surface unavailable | No fallback claim was made. UI evidence is limited to build, source, saved logs/screenshots, and live HTTP/API checks |

Final database integrity after recovery was `ok`, with the recorded pre-audit logical state restored: 15 diseases, 31 scans, 4 feedback rows, maximum scan ID 31, and scan sequence 31. Audit localhost services on ports 5173 and 8000 were stopped and verified unreachable. Docker build, training, dataset rewrite/split, evaluation report generation, and field-review decisions were not executed because they would download, consume substantial compute, or mutate project artifacts; their state was established from code/artifacts instead.

Warnings:

- Two tests emit PyTorch legacy TorchScript ONNX-export deprecation warnings (`src/training/export_onnx.py:29`).
- Backend tests emit a Starlette warning that `httpx` TestClient use is deprecated in favor of `httpx2`.
- Vite warns that the minified main chunk exceeds 500 kB.

## Test inventory and gaps

The 23 root tests cover dataset absence/registry/split reproducibility, field ingestion/cleaning/review gate, metrics/error analysis, baseline shape, model selection, ONNX export/parity, Phase 2.5 config/preprocessing/calibration/augmentation/class weights, RNG restoration, and a one-epoch artifact smoke run. The seven backend tests cover health, valid upload, 503, invalid MIME, unreadable data, oversize, and temperature handling.

Missing areas:

- Frontend component/unit tests: **Not found in the codebase.**
- End-to-end browser tests: **Not found in the codebase.**
- Batch, classes, disease lookup, history bounds, feedback, CORS, database failure, and concurrency API tests are absent.
- Production-model regression/golden-image and OOD tests are absent.
- Docker/fresh-clone deployment tests are absent.
- Load, rate, memory, and security tests are absent.
- Coverage configuration/report: **Not found in the codebase.** No coverage percentage is claimed.
- ESLint/Prettier/Ruff/Black/Mypy/TypeScript/pre-commit configuration: **Not found in the codebase.**
- CI workflow: **Not found in the codebase.**

The backend test that accepts a valid prediction writes to the default database when run from the repository, because it lacks a temp-DB fixture. The audit therefore reran it from an isolated working directory. This should be fixed before CI.

## Code-quality observations

- Positive: modular training code, dataclass configuration, atomic artifact writes, resume signature, deterministic seeds, clear test names, canonical inference preprocessing, and parameterized SQL.
- Negative: unpinned Python dependencies, stale docs/config, no automated style/type gates, blocking async endpoint work, duplicated/stale status concepts, several very large training functions, unsafe checkpoint loads, and mixed Windows/Bash assumptions.
- TODO/FIXME scan: none found. Empty implementation bodies: none found apart from legitimate exception/control constructs.

# 9. Deployment and DevOps Status

**Deployment readiness: 28%.**

## What exists

- Backend Dockerfile based on `python:3.11-slim`, Uvicorn on port 8000 (`backend/Dockerfile:1-14`).
- Deployment notes for Railway/Render and Vercel (`docs/deployment.md`).
- Frontend production build script (`frontend/package.json:6-9`).
- Environment examples for API URL, model paths, DB path, origins, and upload size.
- Rotating local request logs.

## Fresh-environment assessment

**The project cannot be deployed successfully with real inference from a clean clone without manual, undocumented artifact transfer.**

1. `models/onnx` is ignored/untracked, but Docker requires it at `COPY models/onnx ./models/onnx` (`Dockerfile:11`). A clean build fails at that copy or produces no usable model if manually altered.
2. `artifacts/` is ignored and no production bundle has been selected.
3. There is no script to download a versioned model and verify its checksum.
4. There is no `.dockerignore`; the current local build context can include huge datasets, artifacts, `.venv`, and `frontend/node_modules` even though most are not copied.
5. Python dependencies are unpinned, so future image resolution can change.
6. Docker fixes port 8000, while non-Docker Railway instructions use `$PORT` (`docs/deployment.md:13-17`). The target platform behavior must be made explicit.
7. SQLite history has no mounted-volume or backup instructions and is ephemeral on many hosts.
8. Backend startup performs schema seeding rather than migrations.
9. Frontend has no Dockerfile, Vercel project configuration, SPA rewrite configuration, or runtime-config strategy; `VITE_API_URL` is build-time.
10. Health endpoint exists, but Docker has no `HEALTHCHECK`, and the health `status` can be OK without a model.
11. Reverse proxy/TLS, centralized monitoring, alerting, tracing, crash reporting, and backup/restore: **Not found in the codebase.**
12. Docker Compose: **Not found in the codebase.**
13. CI/CD workflows: **Not found in the codebase.**

Platform-specific assumptions include Bash-only `scripts/*.sh`, PowerShell-oriented README commands, a hardcoded Windows Chrome path and `C:\tmp` screenshot profile (`capture_frontend_screenshots.mjs:5-11`), and CUDA-trained artifacts. ONNX CPU serving itself is portable.

The screenshot script also waits for at least one history row (`capture_frontend_screenshots.mjs:129-134`), so it fails on a clean empty database even when the UI is healthy.

# 10. Complete Feature Inventory

| Feature | Description | Frontend status | Backend status | ML dependency | DB dependency | Integration status | Completion | Relevant files | Main issue | Recommended next action |
|---|---|---|---|---|---|---|---:|---|---|---|
| Main navigation | Scan/Dashboard/About switch | Complete | N/A | No | No | Working | 90% | `frontend/src/App.jsx` | No URL routing/deep links | Add React Router or explicit URL state |
| Leaf selection | Picker, drop, camera, preview | Working with minor issues | Upload accepted | Indirect | No | Working | 85% | `ImageUpload.jsx`, `Home.jsx` | Weak format/dimension validation; keyboard issue | Align accepted formats and accessibility |
| Single prediction | Analyze one leaf | Complete UI | Working | Required | Writes scan | Real current model | 80% | `Home.jsx`, `api.js`, `predict.py` | Older unevaluated model served | Deploy approved release bundle |
| Top-3/confidence | Show alternatives and bars | Complete | Working | Required | No | Working | 85% | `PredictionResult.jsx`, `model_loader.py` | No semantic progressbar/OOD policy | Add calibrated abstention and ARIA values |
| Disease guidance | Symptoms/treatment/severity | Complete UI | Partial data | Label required | Diseases | Connected | 45% | `DiseaseInfoCard.jsx`, seed DB | 9/15 placeholders; static severity | Expert-review all classes and version content |
| Low-confidence warning | Warn below 0.60 | Complete | N/A | Confidence | No | Working | 70% | `PredictionResult.jsx:69-95` | Arbitrary threshold, no calibration/OOD basis | Derive threshold from validation/OOD tests |
| Scan history | Recent records | Complete | Working | Prediction source | Scans | Working | 85% | `ScanHistory.jsx`, `system.py` | Public hashes, duplicates, no pagination | Add auth/pagination/filter/dedup policy |
| Analytics dashboard | Counts/chart/history cards | Complete | Working via history | No direct | Scans | Working | 80% | `Dashboard.jsx` | No refresh/search/date filters | Add query/filter API and refresh |
| Feedback | Helpful/not helpful | Complete | Partial | No | Feedback | Connected | 65% | `PredictionResult.jsx`, `system.py` | No scan/user relation, validation, dedup | Link feedback to scan ID and validate |
| Health indicator | API/model/DB status | Partial | Partial | Model status | DB | Semantically wrong | 60% | `Home.jsx`, `system.py` | “Online/live” ignores `model_loaded` | Make readiness require DB and model |
| Class list | Expose model classes | Unused | Working | Metadata | No | Unused | 65% | `system.py:29-31` | Empty when unavailable; no consumer | Use for diagnostics/admin or remove |
| Batch prediction | Multiple files | Not implemented | Partial | Required | Writes scans | Unused | 45% | `predict.py:69-79` | No limits; 500 if model unavailable | Add limits, thread offload, error tests/client |
| Disease lookup API | One class's reviewed info | Client function unused | Working/404 | Label | Diseases | Unused | 60% | `disease_info.py`, `api.js:34-37` | Placeholder entries intentionally 404 | Define consistent guidance contract |
| Authentication | User identity/login | Planned/not found | Not found | No | Users absent | Missing | 0% | Not found | Public sensitive/compute routes | Add only if product needs multi-user/public history |
| Profiles/admin/RBAC | User/admin management | Not found | Not found | No | Tables absent | Missing | 0% | Not found | No authorization boundary | Design roles and protected admin endpoints |
| Main search/filter | Search scan records | Not found | Not found | No | Scans | Missing | 0% | Not found | Dashboard cannot query history | Add indexed query parameters and UI |
| SQLite persistence | Diseases, scans, feedback | UI consumers working | Working basic | No | Core | Connected | 60% | seed/routes/DB | No migrations/FKs/backups; DB_PATH bug | Introduce migration layer and persistent volume |
| Field ingestion | Parse images/metadata and report quality | Utility only | Local script | Future training | Manifest files | Working by tests/artifacts | 75% | `ingest_field_survey.py` | Missing/ambiguous/duplicate/PII records | Redact, dedupe, repair source data |
| Field label cleaning | Normalize without dropping originals | Report only | Local script | Future training | Manifests | Working by tests | 75% | `clean_field_survey_labels.py` | 3,006 records need review | Complete audited review process |
| Field review UI | Queue/search/accept/replace/reject | Partially complete | Local HTTP server | Training gate | JSON/JSONL | Code tested; data 0% reviewed | 35% | `review_app/`, `review_field_survey.py` | 409/409 pending; arbitrary labels; PII/mojibake | Restrict ontology, add reviewer policy, finish review |
| Deterministic split | Persisted 70/15/15 grouping | N/A | Offline | Required | Manifest | Complete for PlantVillage | 95% | `registry.py`, `phase1_split.json` | Optional datasets skipped; near duplicates untested | Add perceptual/source grouping and external set |
| Training engine | Resume, AMP, EMA, weights, augmentations | N/A | Offline | Core | Artifacts | Working/tests pass | 92% | `train.py`, `engine.py` | Unsafe loads/unpinned env | Safe load policy and environment lock |
| Actual benchmark runs | Train required backbones | N/A | Offline | Core | Artifacts | Partial | 38% | Phase 2.5 artifacts | 1 complete, 1 interrupted, 1 unstarted | Resume/finish required candidates |
| Evaluation/calibration | Test metrics, reports, ECE | N/A | Offline | Core | Artifacts | One candidate complete | 35% | `evaluation/`, EFF artifacts | Serving/two candidates unevaluated; no field test | Evaluate all and add external/OOD protocol |
| ONNX export/parity | Export and compare logits | N/A | Serving consumes | Core | Files | Working for serving + one candidate | 65% | `export_onnx.py`, model artifacts | No selected/versioned release | Promote checksummed production bundle |
| Model selection | Weighted quality/calibration/speed/size/memory | N/A | Offline | All candidates | Artifacts | Correctly blocked | 30% | `benchmark.py`, `phase2_5.yaml` | Required candidates incomplete | Finish runs, then promote atomically |
| Docker deployment | Containerized API | N/A | Dockerfile present | Model file required | SQLite file | Broken from clean clone | 28% | `backend/Dockerfile`, deployment docs | Ignored model, no volume/health/CI | Add release retrieval, `.dockerignore`, health/volume |
| Documentation | Setup/architecture/deployment/results | N/A | N/A | N/A | N/A | Partial | 68% | `README.md`, `docs/` | Training docs/model config stale | Generate docs from artifacts and test commands |

# 11. Working and Broken User Flows

| Flow | End-to-end status | Break/mock involvement | Responsible files/evidence |
|---|---|---|---|
| Open app → health/history | Works locally with services/artifacts | No mock; badge can falsely say live when model missing | `Home.jsx:18-34`; `system.py:13-25`; saved screenshots/logs |
| Choose/drop/camera image → preview | Works | No mock | `ImageUpload.jsx`, `Home.jsx:16` |
| Upload → validate → ONNX predict → guidance → UI result | Conditionally working | Real model, not mock; latest model not used. Fresh interactive browser upload was not rerun, but source contracts, saved full-stack logs, isolated API prediction, and screenshots agree | `api.js:23-31`; `predict.py:18-66`; `model_loader.py`; `PredictionResult.jsx`; live log lines 3/37/187 |
| Prediction → SQLite history → recent list | Works | No mock; duplicate scans allowed | `_save_scan`, `/history`, `ScanHistory`; isolated live smoke persisted row |
| Open Dashboard → aggregate real history | Works when backend is available | No static/mock data | `Dashboard.jsx:21-110`; `/history`; screenshot |
| Send helpful/not-helpful feedback | Likely works by connected code and 4 stored rows | No mock; not freshly browser-executed; no scan relation | `PredictionResult.jsx:72-79,117-121`; `system.py:50-58`; DB evidence |
| Batch upload | API-only partial | Real inference; frontend absent; breaks with unhandled model-unavailable error | `predict.py:69-79` |
| Sign up/login/view own history | Broken/not implemented | No mock | **Not found in the codebase.** |
| Review field labels → validated training eligibility | Code/tests work, operational flow incomplete | No mock; all 409 groups remain pending | `review_app/`; `review_field_survey.py`; `validated_manifest.json:11-20` |
| Train/resume Phase 2.5 benchmark → select production model | Partially works | Real artifacts; stops at ConvNeXt-Tiny epoch 6; ConvNeXt-Base absent; selection none | `train.py`, `benchmark.py`, artifacts, finalizer log |
| Fresh clone → Docker deploy → real prediction | Broken | Model bundle absent because ignored; no download step | `.gitignore:18-22`; `Dockerfile:11` |

# 12. Major Bugs, Risks, and Missing Features

## Confirmed bugs or contradictions

1. Health/UI “MODEL LIVE” can be shown while `model_loaded=false` (`Home.jsx:23`; `system.py:22-24`; also acknowledged at `README.md:243`).
2. Batch inference returns an unhandled server error when the model is unavailable (`predict.py:77`).
3. Custom `DB_PATH` is queried but not seeded.
4. Docker clean build requires ignored `models/onnx`.
5. Current UI says `224 RGB` even if the 300 px Phase 2.5 model is deployed.
6. Seed parsing can create leading-space disease display names.
7. Backend tests can mutate the project database when run normally.
8. Review UI contains encoding mojibake.
9. `URL.createObjectURL` is leaked.
10. `models/model_config.json` claims six-class EfficientNet-B0/mock while actual serving metadata is 15-class MobileNetV3/no mock.
11. `docs/training_results.md:23-33` and `docs/model_comparison.md:7-9` say candidates are “not started,” contradicting completed/partial artifacts. `docs/production_model.md` correctly says selection remains blocked.

## Missing/incomplete inventory

- TODO/FIXME comments: none found.
- Empty functions: none found.
- Current mock API responses: none; stale mock defaults/text remain.
- Authentication, profiles, roles, admin, password hashing, tokens: **Not found in the codebase.**
- Main scan search/filter/pagination: **Not found in the codebase.**
- Dark mode: **Not found in the codebase.**
- Frontend tests, E2E tests, accessibility automation: **Not found in the codebase.**
- Lint/type/format/pre-commit/coverage gates: **Not found in the codebase.**
- Database migrations/FKs/backup/restore: **Not found in the codebase.**
- CI/CD workflows and Docker Compose: **Not found in the codebase.**
- Selected production model directory: **Not found in the codebase.**
- ConvNeXt-Base and Swin artifacts: **Not found in the codebase.**
- Serving-model evaluation metrics: **Not found in the codebase.**
- Field/OOD/external benchmark: **Not found in the codebase.**
- Standalone “classify this image” inference CLI: **Not found in the codebase.** `src.inference.predict` CLI only exports ONNX (`predict.py:95-100`).
- Grad-CAM/explainability, segmentation, learned severity, weather, multilingual production UI, active/continual learning: explicitly roadmap-only (`README.md:230`; `docs/future_model_roadmap.md`).
- Current notebooks: **Not found in the codebase.** Their deletion is committed.
- Hugging Face, WandB, TensorBoard, MLflow, or cloud model registry: **Not found in the codebase.**

## Hardcoded/static values

- Local API fallback `127.0.0.1:8000` (`api.js:3`).
- UI `224 RGB` and confidence warning threshold 0.60 (`PredictionResult.jsx:24,69`).
- Backend default ports/origins and Docker port 8000.
- Screenshot Chrome path, debug port 9333, viewport, and `C:\tmp` (`capture_frontend_screenshots.mjs:5-11`).
- Static disease severity/guidance rather than model-derived severity.
- Training config values are intentional configuration, not runtime bugs, but current docs do not reflect actual run state.

# 13. Completion Summary and Calculation

| Component | Completion | Calculation/evidence basis |
|---|---:|---|
| Frontend | 78% | 14/15 screens/navigation + 23/25 scan/results + 17/20 history/dashboard/feedback + 11/15 resilience + 8/10 responsive/a11y + 5/15 assurance |
| Backend | 72% | 30/35 API core + 20/20 inference + 10/15 storage + 7/15 validation/errors + 5/15 security/ops |
| API integration | 78% | Core real scan/history/feedback connected; health semantics, unused endpoints, and error handling reduce score |
| ML dataset preparation | 85% | Complete/corruption-free split and exact duplicate checks; optional field set at 0 eligible, imbalance and domain risks |
| ML training pipeline | 92% | Feature-complete and tests pass; unsafe loading/unpinned reproducibility remain |
| Actual ML training | 38% | Required candidate completion: `(100 + 15 + 0) / 3 = 38.3%`; 15 is ConvNeXt epoch 6/40 without completion artifact |
| ML evaluation | 35% | Full evaluation for one of three required candidates; no serving-model/external-field evaluation |
| Model export | 65% | Two working ONNX bundles and parity evidence, but no complete selected release and two candidates missing |
| ML inference | 82% | Real singleton ONNX, correct mapping, CPU/GPU fallback, parity/smoke; stale model/metadata and no OOD |
| Database | 60% | Functional core tables/CRUD; no migrations, relationships, config consistency, backup, or production persistence |
| Authentication | 0% | No auth implementation at any layer |
| Testing | 48% | 30 tests pass, but no frontend/E2E/coverage/CI/load/security/deployment testing |
| Security | 35% | Basic CORS, parameterized SQL, image decoding; no auth/rate limits/security middleware and significant ML/data risks |
| Documentation | 68% | Strong README/deployment/pipeline descriptions; artifact/status/config documents stale and no generated operational runbook |
| Docker/deployment | 28% | Dockerfile/build docs exist; clean clone lacks model, and CI/volume/health/monitoring are absent |
| Overall project | **63%** | `runtime MVP 76 × 0.45 + ML lifecycle 66 × 0.30 + assurance 36 × 0.25 = 63.0` |

The overall weighting reflects the project's stated product: a working scan demo is important (45%), the ML lifecycle is core intellectual functionality (30%), and production assurance is essential but less visible (25%). Missing auth is scored explicitly under assurance; it does not erase the anonymous local-demo functionality.

# 14. Priority Action Plan

## Critical

| # | Issue | Why it matters | Relevant files | Exact recommended change | Expected result | Dependencies |
|---:|---|---|---|---|---|---|
| 1 | No reproducible model release | Clean clone/Docker cannot serve; wrong model can be paired with metadata | `.gitignore`, `backend/Dockerfile`, `benchmark.py`, deployment docs | Produce a versioned checksummed production bundle outside ignored training artifacts; add an authenticated/reproducible download or release-copy step; verify checksum at startup/build | Fresh environments obtain exactly one ONNX+metadata pair | Artifact registry/release location and model approval |
| 2 | Benchmark and production selection incomplete | Latest single result cannot be called production winner; current serving model lacks metrics | Phase 2.5 config/artifacts, `benchmark.py` | Resume ConvNeXt-Tiny, train ConvNeXt-Base, run selection, and only promote when all required checks pass; add a field/OOD holdout before final approval | Defensible selected model with complete comparison | GPU time, curated external dataset, action 1 |
| 3 | Unsafe public/high-stakes deployment | Open compute/history/feedback and unreviewed advice can be abused or harm users | all backend routes, seed data, frontend result | Before any public exposure, add rate/concurrency limits and an explicit access/history policy; protect user-specific data if introduced; replace all placeholders with expert-reviewed, versioned guidance and a clear advisory disclaimer/abstention path | Safer bounded service and trustworthy content | Product identity decision, agronomist review, privacy policy |

## High priority

| # | Issue | Why it matters | Relevant files | Exact recommended change | Expected result | Dependencies |
|---:|---|---|---|---|---|---|
| 4 | Latest model not served; health lies | Demo can present old results or “live” unavailable model | `model_loader.py`, `system.py`, `Home.jsx`, `PredictionResult.jsx` | Point deployment only at selected bundle; make health `status/readiness` require model and DB; have frontend check `model_loaded`; derive displayed input size from metadata/API | UI and service describe the same loaded model | Actions 1–2 |
| 5 | Docker/runtime not production-safe | Current image is unreproducible, oversized context, fixed port, ephemeral SQLite | `Dockerfile`, deployment docs | Add `.dockerignore`, pinned hashes/lock, model fetch/copy stage, dynamic port strategy, non-root user, Docker health check, and explicit persistent volume/managed DB | Repeatable smaller image with persistent health/history | Action 1; hosting choice |
| 6 | Database lifecycle is ad hoc | Custom paths fail, startup drops a table, relationships and backups are absent | `config.py`, `disease_info.py`, `seed_disease_data.py` | Pass one configured DB path everywhere; add versioned migrations; stop dropping tables; add checks/FKs or documented denormalization; index planned filters; add backup/restore and temp-DB test fixture | Safe upgrades and isolated tests | Schema decision; deployment storage choice |
| 7 | Upload/inference can block or exhaust service | Public files and batch requests can consume memory/event loop/CPU | `predict.py`, schemas, main | Stream/spool with early byte cap; validate allowed formats, decoded dimensions/pixels, batch count/total; offload ONNX/SQLite from async loop; catch batch model errors; add timeouts and rate/load tests | Predictable errors and bounded resource use | Expected traffic/SLOs |

## Medium priority

| # | Issue | Why it matters | Relevant files | Exact recommended change | Expected result | Dependencies |
|---:|---|---|---|---|---|---|
| 8 | Assurance automation missing | Regressions and stale reports are currently manual | `tests/`, `backend/tests/`, `frontend/package.json`, `docs/` | Add frontend unit/a11y tests, API tests for every route, one browser E2E scan with temp DB/model fixture, coverage thresholds, lint/type checks, and CI build/test/Docker smoke; generate training status docs from artifacts | Repeatable quality signal on every change | CI platform; stable fixtures |
| 9 | Field data privacy/quality incomplete | Real-world validation cannot proceed and PII is retained | field manifests/review app/scripts | Redact/pseudonymize personal fields, document consent/retention, restrict replacement to approved ontology, repair mojibake, dedupe images, and complete dual-review/audit workflow | Usable governed field benchmark/training set | Data owner and domain reviewers |

## Optional improvements

| # | Issue | Why it matters | Relevant files | Exact recommended change | Expected result | Dependencies |
|---:|---|---|---|---|---|---|
| 10 | Frontend polish/performance | Better navigation, accessibility, and load size improve usability | `App.jsx`, components, styles | Add URL routing/lazy-loaded dashboard, revoke preview URLs, semantic progress bars, keyboard drop behavior, locally hosted fonts, mobile visual regression, refresh/filter controls, and optional theme | Smaller, accessible, deep-linkable UI | Stable API; design decision |

# 15. Commands and Setup Instructions

Only commands backed by current repository scripts, documentation, or CLI parsers are listed. Commands that train, seed, split, review, or evaluate write files; they were extracted but not executed during this audit.

## Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r backend/requirements.txt
npm.cmd --prefix frontend install
```

Source: `README.md:89-100`. Node 18+ is required. Python packages are unpinned, so these commands are not fully reproducible.

## Configure environment variables

```powershell
$env:MODEL_PATH="models/onnx/model.onnx"
$env:MODEL_METADATA_PATH="models/onnx/model.json"
$env:DB_PATH="backend/db/disease_info.db"
$env:CORS_ORIGINS="http://127.0.0.1:5173"
$env:MAX_UPLOAD_SIZE_MB="10"
$env:VITE_API_URL="http://127.0.0.1:8000"
```

Sources: `backend/.env.example:1-5`, `frontend/.env.example:1`, `README.md:102-120`. The application does not load `.env` automatically; variables must be supplied by the shell/process manager.

## Database

SQLite requires no separate database server. The backend seeds on startup. The explicit repository-supported seed command is:

```powershell
.\.venv\Scripts\python.exe backend/db/seed_disease_data.py
```

Warning: this drops/recreates the `diseases` table. Database migration command: **Not found in the codebase.**

## Download and preprocess PlantVillage

```powershell
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_key"
.\.venv\Scripts\python.exe -m src.data.download_data
.\.venv\Scripts\python.exe -m src.data.split_dataset
```

Manual archive fallback:

```powershell
.\.venv\Scripts\python.exe -m src.data.download_data --skip-download
.\.venv\Scripts\python.exe -m src.data.split_dataset
```

These commands replace/prepare data trees; see `README.md:168-183`.

## Field-survey ingest and review

```powershell
.\.venv\Scripts\python.exe -m src.data.ingest_field_survey `
  --survey-file path\to\survey.xlsx `
  --image-root path\to\survey-images

.\.venv\Scripts\python.exe -m src.data.clean_field_survey_labels
.\.venv\Scripts\python.exe -m src.data.review_field_survey --host 127.0.0.1 --port 8765
```

These exact arguments are documented at `README.md:185-196`. Review decisions mutate JSONL/validated-manifest files.

## Train and resume

Train/resume one required candidate (resume is the default):

```powershell
.\.venv\Scripts\python.exe -m src.training.train --config configs/training/phase2_5.yaml --architecture efficientnetv2_s
.\.venv\Scripts\python.exe -m src.training.train --config configs/training/phase2_5.yaml --architecture convnext_tiny
.\.venv\Scripts\python.exe -m src.training.train --config configs/training/phase2_5.yaml --architecture convnext_base
```

Resume all missing/incomplete required candidates and then attempt selection:

```powershell
.\.venv\Scripts\python.exe -m src.training.benchmark --config configs/training/phase2_5.yaml --train
```

Optional Swin candidate:

```powershell
.\.venv\Scripts\python.exe -m src.training.benchmark --config configs/training/phase2_5.yaml --train --include-optional
```

Sources: `README.md:198-227`; `train.py:847-860`; `benchmark.py:499-512`. Do not pass `--force-split` when resuming current runs.

## Evaluate

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.evaluate --checkpoint models/checkpoints/best_model.pth
```

The parser supports `--checkpoint` (`src/evaluation/evaluate.py:93-101`) and writes report/plot files. For Phase 2.5, final evaluation is normally performed automatically by training finalization.

## Export ONNX

```powershell
.\.venv\Scripts\python.exe -m src.inference.predict --checkpoint models/checkpoints/best_model.pth --output models/onnx/model.onnx
```

Source: `README.md:122-132`. This legacy command exports; it is not a standalone image-classification CLI.

## Start backend and frontend

Backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Frontend:

```powershell
npm.cmd --prefix frontend run dev -- --host 127.0.0.1 --port 5173
```

Sources: `README.md:134-154`.

## Run inference through the supported API

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@path\to\leaf.jpg"
```

Direct image-inference CLI: **Not found in the codebase.**

## Tests, lint, and production build

```powershell
.\.venv\Scripts\python.exe -m pytest
npm.cmd --prefix frontend run build
```

Source: `README.md:232-237`. Frontend test, lint, type-check, formatter, coverage, and pre-commit commands: **Not found in the codebase.**

## Docker and deployment

The Dockerfile and deployment document support building from repository root:

```powershell
docker build -f backend/Dockerfile -t leaflight-backend .
docker run --rm -p 8000:8000 leaflight-backend
```

These commands require `models/onnx/model.onnx` and `model.json` to exist in the build context. They do not work with real inference from the current clean Git checkout because those files are ignored/untracked.

Documentation-only hosting commands emitted by `scripts/deploy.sh`:

```bash
cd frontend && vercel --prod
railway up
```

Required CLIs/configuration are not installed or verified by the repository. Docker Compose command: **Not found in the codebase.**

| Component       | Status | Completion | Evidence | Main Issue | Next Step |
| --------------- | ------ | ---------: | -------- | ---------- | --------- |
| Frontend        | Working with minor/structural issues | 78% | Build passed; React screens and saved screenshots; real Axios calls | No routing/tests; health semantics and accessibility gaps | Add E2E/a11y tests, correct readiness, optimize bundle |
| Backend         | Partially working | 72% | 7/7 tests; live ONNX prediction; routed FastAPI code | Public/blocking endpoints, weak validation/error handling | Bound uploads, offload inference, add security controls |
| API Integration | Core flow working | 78% | Real `/predict`, `/history`, `/health`, `/feedback` wiring | Batch/health issues; supporting endpoints unused | Complete contract/error tests and clients |
| ML Dataset      | Strong core, field set incomplete | 85% | 20,638 validated files; deterministic split; no cross-split exact duplicates | Lab-domain bias, imbalance, 0 field records approved | Curate governed field/OOD benchmark |
| ML Training     | Pipeline strong, runs incomplete | 38% | Actual runs: one complete, one epoch 6/40, one absent; pipeline itself scores 92% and 23 tests pass | Required benchmark unfinished | Resume ConvNeXt-Tiny and train ConvNeXt-Base |
| ML Evaluation   | Partial | 35% | Full metrics/calibration only for Phase 2.5 EfficientNetV2-S | Serving model and two candidates unevaluated; no field test | Evaluate all candidates and external/OOD set |
| ML Inference    | Real but stale | 82% | ONNX/PyTorch parity; correct live held-out prediction; mock false | Older 224 px MobileNet served, no OOD abstention | Deploy selected checksummed bundle and thresholds |
| Database        | Functional basic SQLite | 60% | Integrity OK; 15 diseases, 31 scans, 4 feedback | No migrations/FKs/backups; custom path seed bug | Add migrations, consistent config, persistent storage |
| Authentication  | Not implemented | 0% | No routes, schemas, tables, tokens, or UI found | All endpoints/history public | Define access model before public deployment |
| Testing         | Partial | 48% | 23/23 root + 7/7 backend passed; build passed | No frontend/E2E/coverage/CI/load/security tests | Add isolated cross-layer CI suite |
| Security        | Insufficient for public use | 35% | Parameterized SQL, basic CORS/file checks | No auth/rate limits; unsafe checkpoints; PII/high-stakes guidance | Threat model, rate/auth policy, safe artifacts, privacy review |
| Deployment      | Not fresh-clone ready | 28% | Backend Dockerfile and hosting docs exist | Ignored model, no release fetch, CI, volume, health, monitoring | Build reproducible model-aware deployment pipeline |
| Overall Project | Partially working; conditional demo; not production-ready | **63%** | Build/tests/live model pass; benchmark and production assurance incomplete | Reproducibility, selected model, security, field validation | Execute critical/high-priority plan in order |
