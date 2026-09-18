# ModelOps Platform — Project Documentation

> **This is the project's living technical document.** It is updated at the end of
> every phase with what was built, how to run it, the measured numbers, and the
> commands that keep the stack alive. Interview prep lives in `../PROJECT_RECAP.md`;
> this file answers "how does the project run and what does it contain?"

- Author: Mohammad Yakub · AI & Data Science · KL University
- Guide: `../ModelOps_Platform_Coding_Agent_Guide.pdf`
- Source (linked): `Model-Regression-Detection-System`

---

## 1. What this project is

`modelops-platform` is a production-styled ML lifecycle:

```
Dataset → Validation → Feature Engineering → Training → Experiment Tracking
→ Evaluation → Regression Gate → Model Registry → Deployment → Monitoring
→ Drift Detection → Retraining
```

Each phase adds one stage of that lifecycle. One phase fully working before the
next; commits after every feature; every reported number is measured, never
estimated; at least 2 tests per module (Phase 1 required ≥3).

## 2. Repository layout

```text
.
├── configs/train_config.yaml          # all pipeline knobs (incl. serving bounds)
├── data/
│   ├── raw/dataset_v1.csv             # versioned gold data (SHA-256 tracked)
│   ├── validation_reports/            # generated validation_latest.json
│   ├── gate_reports/                  # generated gate_latest.json
│   ├── benchmarks/                    # Phase 6: serving_benchmark.json
│   ├── drift/                         # Phase 5: drift windows (current_same/shifted.csv)
│   └── drift_reports/                 # Phase 5: generated drift_latest.json
├── scripts/generate_sample_data.py    # deterministic data generator
├── scripts/simulate_drift.py          # Phase 5: drift scenario generator
├── scripts/benchmark.py               # Phase 6: reproducible serving benchmark
├── scripts/demo.py                    # Phase 6: one-command lifecycle demo
├── src/
│   ├── config.py                      # YAML -> typed config
│   ├── pipeline.py                    # step_validate/train_track/gate_deploy
│   ├── validation/validator.py        # Phase 1: schema/leakage/ranges checks
│   ├── training/trainer.py            # Phase 1: sklearn + XGBoost training
│   ├── tracking/tracker.py            # Phase 2: MLflowTracker
│   ├── registry/registry.py           # Phase 2/3: ModelRegistry (stages/rollback)
│   ├── gate/regression_gate.py        # Phase 3: deploy gate
│   ├── serving/api.py                 # Phase 4: FastAPI inference service
│   └── monitoring/                    # Phase 5: monitor.py + drift.py + runner
├── pipelines/airflow/                 # Phase 4: retraining DAG
├── dashboards/
│   ├── prometheus/                    # Phase 5: scrape config (serving:8000/metrics)
│   └── grafana/                       # Phase 5: provisioned datasource + dashboard
├── tests/                             # pytest suites (38 passing)
├── docker/mlflow.Dockerfile
├── docker/serving.Dockerfile          # Phase 4: multi-stage serving image
├── .github/workflows/ci.yml           # Phase 4: tests -> gate -> image
├── docker-compose.yml                 # mlflow + serving + airflow/prometheus/grafana
├── docs/PROJECT.md                    # this file
├── docs/architecture.dot              # editable architecture diagram (graphviz)
├── docs/architecture.png              # rendered PNG for the README
├── PROJECT_RECAP.md                   # week-by-week study recap + interview prep
└── requirements.txt
```

## 3. Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10+ | tested on 3.10 |
| pip | — | |
| Docker | optional | only needed for the containerized MLflow UI |
| disk/network | — | first install pulls mlflow/xgboost wheels |

**Environment quirk (this dev box):** the shell sources ROS (`/opt/ros/humble`),
which sets `PYTHONPATH` and injects ROS pytest plugins into any venv. Prefix
python/pytest commands with `PYTHONPATH=` (empty) to keep them clean.

## 4. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Commands reference

### Run the pipeline (validation + training + gate + deploy)
```bash
PYTHONPATH= .venv/bin/python -m src.pipeline            # validate + train only
PYTHONPATH= .venv/bin/python -m src.pipeline --track    # + MLflow, registry, gate, deploy
```

Outputs:
- `data/validation_reports/validation_latest.json` — validation report
- `data/gate_reports/gate_latest.json` — regression gate report (side-by-side verdicts)
- `artifacts/models/<model>_v1.joblib` + `<model>_v1.json` — model + metadata
- `mlruns.db` (SQLite) + `mlruns/` — MLflow runs, artifacts, registry

### Tests
```bash
PYTHONPATH= .venv/bin/python -m pytest tests/ -q          # full suite (38)
PYTHONPATH= .venv/bin/python -m pytest tests/test_serving.py -q   # a module
```

### Monitoring + drift (Phase 5)
```bash
PYTHONPATH= MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python -m src.monitoring   # runs drift check, logs MLflow run
PYTHONPATH= MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python scripts/simulate_drift.py  # regenerate drift windows
```

### Benchmark + demo (Phase 6)
```bash
# live serving benchmark (start uvicorn first) -> data/benchmarks/serving_benchmark.json
PYTHONPATH= MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python scripts/benchmark.py --n 2000 --concurrency 8
# whole lifecycle in one command (validate -> train -> gate -> serve -> drift)
PYTHONPATH= MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python scripts/demo.py
```

### Serving API (FastAPI, serves the Production model)
```bash
# start (loads the Production model from mlruns.db at startup)
PYTHONPATH= MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/uvicorn src.serving.api:app \
  --host 0.0.0.0 --port 8000
curl -s http://localhost:8000/health     # liveness
curl -s http://localhost:8000/ready      # readiness (503 until model loaded)
curl -s http://localhost:8000/model/info # version + run metrics + served p95/p99
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":45,"income":72000,"credit_score":680,"loan_amount":24000,"employment_years":8,"num_defaults":1,"has_collateral":1}'
```
Restart: `fuser -k 8000/tcp; sleep 1; <start>` · Stop: `fuser -k 8000/tcp`
Interactive docs: http://localhost:8000/docs

### Containerized serving (Docker-enabled host)
```bash
docker compose up -d --build serving    # + mlflow; serving healthcheck on /ready
docker compose --profile airflow up airflow   # Airflow webserver on :8080
docker compose down
```

### Fresh-state rerun (wipe tracked runs + artifacts, regenerate everything)
```bash
rm -rf mlruns mlruns.db artifacts data/validation_reports data/gate_reports
PYTHONPATH= .venv/bin/python scripts/generate_sample_data.py   # only if you want to rebuild the CSV
PYTHONPATH= .venv/bin/python -m src.pipeline --track
```

### MLflow UI (tracking + model registry dashboard)

The UI server reads the same `mlruns.db` the CLI writes (SQLite backing store;
PostgreSQL is the stated production upgrade — SQL either way).

**Start:**
```bash
MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/mlflow server \
  --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000
```
→ open http://localhost:5000 (Experiments → `modelops`; Models → registered models)

**Restart (if it was already running):**
```bash
fuser -k 5000/tcp 2>/dev/null; sleep 2
MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/mlflow server \
  --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000
curl -s http://localhost:5000/health && echo "  <- MLflow UI is UP"
```

**Stop:**
```bash
fuser -k 5000/tcp
```

**Docker variant** (on a host with a running Docker daemon — this box has the CLI
but no `dockerd`, so use the venv command above here):
```bash
docker compose up -d --build mlflow   # start    http://localhost:5000
docker compose down                   # stop
```

### Fresh-state rerun (wipe tracked runs + artifacts, regenerate everything)
```bash
rm -rf mlruns mlruns.db artifacts data/validation_reports
PYTHONPATH= .venv/bin/python scripts/generate_sample_data.py   # only if you want to rebuild the CSV
PYTHONPATH= .venv/bin/python -m src.pipeline --track
```

## 6. Architecture (as built)

```text
train_config.yaml ──► src/config.py ──► typed configs
                        │
raw dataset (CSV) ──────┴──► validation/validator.py ──► validation_latest.json (gate 1)
                                    │  (schema, rows, missing%, ranges, leakage, ids)
                                    ▼
                        training/trainer.py ──► artifacts/models/*.joblib + *.json
                                    │  (sklearn + XGBoost, seeded, hash-lineage)
                                    ▼
                        tracking/tracker.py ──► MLflow run per model (metrics/params/artifact)
                                    │
                                    ▼
                        registry/registry.py ──► registered models, Candidate/Staging/
                                                 Production + rollback (SQL backing store)
                                    │
                                    ▼
                        gate/regression_gate.py ──► gate_latest.json (deploy gate)
                          quality deltas + latency p95 + memory + cost   │
                                            PASS ──► Staging ──► Production (deploy)
                                            FAIL ──► blocked, exit 1
                                                    │
                                                    ▼
                        serving/api.py ──► FastAPI on :8000 (serves Production)
                          /health /ready /model/info /predict (Pydantic-validated)
                                                    │
        CI/CD (.github/workflows/ci.yml) ├── tests ─► validate ─► train ─► gate
                                                    └── gate PASS on main ─► image ─► GHCR
        Airflow (pipelines/airflow/) ──── daily retraining DAG, same step functions
```

Phase 5 (`monitoring/`) adds Prometheus + Grafana + PSI drift; Phase 6 ties it
together with benchmarks and the architecture diagram.

## 7. Phase-by-phase documentation

### Phase 1 — Data Validation + Training Pipeline ✅

**What was built**
- `src/validation/validator.py` — config-driven checks, fail fast:
  schema (expected columns), min row count, missing-value ratio, value ranges,
  leakage (target must not be a feature), id-uniqueness; writes a Pydantic-typed
  JSON report and raises `ValidationError` with the partial report on failure.
- `src/training/trainer.py` — reads `configs/train_config.yaml`; seeds
  `random`/`numpy`/`torch`; stratified 80/20 split; trains sklearn
  `LogisticRegression` + `XGBClassifier`; saves `*.joblib` + metadata JSON
  (params, metrics, data SHA-256, seed, duration); hashes the data file.
- `scripts/generate_sample_data.py` → `data/raw/dataset_v1.csv`: deterministic
  loan-risk classification data (1200 rows × 9 cols, balanced 600/600, seed 42).
- `src/pipeline.py` — validate → train orchestration.

**Why synthetic data**: the linked source project's dataset is an LLM golden-set
(JSON) that sklearn/XGBoost can't train on; a deterministic dataset gives known
true distributions (honest metrics now, reliable PSI drift scenarios in Phase 5).

**Measured (seed 42, 960 train / 240 test)**
- Validation: PASSED, 6/6 checks, data SHA-256 `ffc1f772…8822b3c`
- LogisticRegression: acc 0.7792 · f1 0.7854 · auc 0.8649 · train 0.06 s
- XGBoost (100 est, depth 3): acc 0.7708 · f1 0.7679 · auc 0.8466 · train 14.7 s
- Artifact sizes: LR 4 KB, XGB 116 KB (joblib)
- Tests: 9 passing (~9.5 s); same-seed reruns produce identical metrics

**Key decisions** — committed CSV as gold artifact; config-driven thresholds;
baseline-vs-XGB comparison; reject on validation failure.

**Fix log** — config loader returned a validator instead of a config; ROS
PYTHONPATH broke pytest (documented, not project-patched); dataset first came
out imbalanced (75%) → tuned the ground-truth threshold to 50/50.

---

### Phase 2 — Experiment Tracking + Model Registry ✅

**What was built**
- `src/tracking/tracker.py` — `MLflowTracker`: run lifecycle (FINISHED/FAILED),
  logs params, metrics, model artifact, data lineage; `log_model` routes
  XGBoost to the xgboost flavor, everything else to sklearn, and returns a
  registrable `model_uri`.
- `src/registry/registry.py` — `ModelRegistry`: `register`, `list_versions`
  (each version with its run's real metrics), `promote` (stage-validated),
  `rollback` (demotes current holder, promotes newest older version).
- `src/pipeline.py --track` — validate → train → track → register all models →
  stage best-by-ROC-AUC as **Candidate** (Production is owned by the Phase 3 gate).
- `docker/mlflow.Dockerfile` + `docker-compose.yml` — MLflow UI service.

**Stage model (candidate → staging → production)**
MLflow natively stores only `None`/`Staging`/`Production`/`Archived`. There is no
`Candidate` bucket, so `Candidate` is implemented as a version tag
(`modelops_stage=candidate`) that the registry translates on read/write. The UI's
Models tab will show `None` for candidate versions for that reason.

**One run per model**: early design logged both models into one MLflow run with
identical metric keys, so the LR version showed XGB's numbers. Per-model runs make
every registered version's metrics honest.

**Measured — fresh seed-42 tracked run**
- 2 MLflow runs (both FINISHED) in experiment `modelops`; `mlruns.db` ≈ 860 KB
- Registry: `modelops_logistic_regression` v1 **Candidate** (auc 0.8649) ·
  `modelops_xgboost` v1 **None** (auc 0.8466)
- MLflow UI verified: `GET /health` → 200; registered-models API returns both models
- Tests: **15 passing** (~60 s)

**Docker limitation on this box** — the Docker CLI exists but there is no
`dockerd` daemon, so the container path is validated (`docker compose config`
OK) and the UI is run from the venv. On a Docker-enabled host:
`docker compose up -d --build mlflow`.

**Fix log** — `Candidate` isn't a native MLflow stage (caught by the end-to-end
run, not unit tests) → tag implementation + regression test added; cross-model
metric collision → one run per model.

---

### Phase 3 — Regression Gate ✅ *(the differentiator)*

**What was built**
- `src/gate/regression_gate.py`:
  - `measure_inference()` — real per-request latency p50/p95/p99 (deterministic
    row order, warmup excluded), resident memory footprint delta, and an explicit
    cost model: `$ per 1M predictions = mean_ms/1000 * 1e6 / 3600 * cpu_hour_usd`
    (price from config). Measured at gate time for **both** models on the same
    hardware → apples-to-apples.
  - `evaluate()` — pure, fully testable gate logic; verdicts per dimension:
    quality (accuracy/f1/roc_auc deltas vs production), latency (absolute cap),
    memory (absolute cap), cost (relative cap).
  - `RegressionGate.run()` — loads candidate + production from the registry,
    pulls their measured run metrics, measures both, emits a side-by-side
    `GateReport` with PASS/FAIL.
- `configs/train_config.yaml` → `gate:` block (all thresholds config-driven):
  `min_accuracy_delta: -0.01`, `min_f1_delta: -0.01`, `min_roc_auc_delta: -0.01`,
  `latency_p95_max_ms: 200`, `memory_max_mb: 512`, `cost_max_delta_pct: 10`,
  `benchmark.*` (`n_requests`, `seed`, `cpu_hour_usd`).
- `src/pipeline.py --track` — after staging the best model as Candidate, runs the
  gate; **PASS** → promote Candidate → Staging → Production (deploy);
  **FAIL** → deployment blocked, exit 1, production untouched. The gate report is
  written to `data/gate_reports/gate_latest.json` and logged as an MLflow artifact
  in a dedicated `regression_gate` run.
- `ModelRegistry` + `load_model` (pyfunc, flavor-agnostic) + `latest_in_stage`.

**Bootstrap rule** — with no production reference yet, quality and cost verdicts
SKIP and the gate passes if the candidate meets the absolute latency/memory caps.
First run therefore deploys v1 to Production (mirrors the source project's
"first run seeds the baseline").

**Gate config in YAML**
```yaml
gate:
  quality:
    min_accuracy_delta: -0.01
    min_f1_delta: -0.01
    min_roc_auc_delta: -0.01
  latency_p95_max_ms: 200
  memory_max_mb: 512
  cost_max_delta_pct: 10
  benchmark:
    n_requests: 200
    seed: 42
    cpu_hour_usd: 0.1
```

**Measured — bootstrap deploy (fresh seed-42 run, LR v1 → Production)**
- `data/gate_reports/gate_latest.json`:
  candidate LR v1 · p50 0.63 ms · **p95 0.76 ms** · p99 0.93 ms · memory 0.06 MB ·
  cost $0.018/1M · verdicts quality SKIP / latency PASS / memory PASS / cost SKIP
- **Measured FAIL demo** (policy tightened: `latency_p95_max_ms: 0.5` below the
  real 1.11 ms p95): candidate p95 **1.11 ms** → verdicts
  quality PASS / **latency FAIL** / memory PASS / **cost FAIL** → deployment
  **blocked**, registry `Production` stayed at v1. Real numbers, real block.
- Tests: **23 passing** (~45 s) — 8 new gate tests cover PASS, FAIL on
  quality/latency/memory/cost, bootstrap pass, bootstrap still enforcing absolute caps.

**Key decisions — accuracy alone is not enough**: a candidate that is 0.002 better
on AUC but is 40% slower and 2× more expensive must not ship; the gate captures
both. Statistical-significance framing: the gate uses measured deltas on the same
test split and same benchmark harness, so deltas are comparisons of identical
measurement, which is the prerequisite for a meaningful regression call.

**Fix log** — `all(v == PASS)` rejected SKIP verdicts → bootstrap always failed
(fixed to `all(v != FAIL)`); `ModelSnapshot(**model_dump(), metrics=…)` duplicate
keyword → excluded `metrics` before overriding with registry metrics.

---

### Phase 4 — Serving + Docker + CI/CD ✅

**What was built**
- `src/serving/api.py` — FastAPI inference service with the classic contract:
  | Endpoint | Purpose |
  |---|---|
  | `GET /health` | liveness — process is up |
  | `GET /ready` | readiness — model actually loaded (503 until it is) |
  | `GET /model/info` | model from `Production` stage: name, version, run metrics, served latency p95/p99 |
  | `POST /predict` | one-row inference; Pydantic-validated against **config-driven** feature bounds → structured 422s; 503 when no model |
  - Model resolved once at startup from the registry Production stage via the
    flavor-aware loader (keeps `predict_proba`); a middleware records every
    request's latency so **the API's own** measured p95 is observable.
- `docker/serving.Dockerfile` — multi-stage: deps installed into a clean prefix,
  slim runtime copies only that prefix + the app (independently cacheable layer).
- `docker-compose.yml` — adds a `serving` service (mounts the local SQLite
  registry, healthcheck hits `/ready`) and an optional `airflow` profile service
  (`docker compose --profile airflow up airflow`, one-node standalone).
- `.github/workflows/ci.yml` — push to main + each PR runs `tests → data validation
  → training → regression gate`; only a **gated** main run builds the serving image
  and pushes it to GHCR. The gate report is uploaded as build evidence.
- `pipelines/airflow/modelops_pipeline_dag.py` — scheduled (`@daily`) retraining
  DAG that calls the **same step functions** the CLI uses
  (`step_validate` → `step_train_track` → `step_gate_deploy`); a FAIL verdict
  fails the DAG (red) and leaves production untouched — orchestration status
  doubles as the deployment signal.
- `src/pipeline.py` refactor — three coarse step functions now shared by the
  CLI, CI, and the DAG (single source of truth); `ModelRegistry.load_model`
  is flavor-aware (sklearn → xgboost → pyfunc).

**Health vs readiness** — `/health` only proves the process/ASGI is alive (what
a load balancer probes to restart a dead worker); `/ready` additionally proves
the model is loaded, so traffic is only sent to a truly serving instance.
Liveness/readiness separation is what makes zero-downtime deployments possible
later (a rolling restart keeps `/ready` green only for instances with a model).

**Measured — real HTTP on localhost (Production model = LR, seed 42)**
- `/predict` over 200 requests: p50 **2.96 ms** · p95 **4.42 ms** · p99 **4.87 ms**
  (full HTTP round-trip incl. JSON); middleware-internal p95 **1.49 ms**
- `/model/info` reports served latency from actual traffic, not a benchmark
- Day-2 deploy path proven: running `--track` again produced candidate v3 vs
  production reference v1 → **quality/latency/memory/cost all PASS** → v3 promoted.
- Validated: `docker compose config` OK; CI workflow YAML parses; DAG/API
  compile. (Docker images can't be built here — no `dockerd` — so the container
  path stays pre-validated for a Docker-enabled host/CI.)
- Tests: **30 passing** (~70 s) — 7 new serving tests (health/ready/info/predict,
  422 validation, 503, latency stats).

**Key decisions**
- Serving latency is measured on the API path separately from the gate's
  standalone benchmark — one is operational, the other is the deploy contract.
- The DAG reuses the proof pipeline wholesale rather than reimplementing steps,
  which would risk divergence between "what passed CI" and "what Airflow ships".

**Fix log** — dynamic Pydantic model: FastAPI resolves endpoint annotations via
the module namespace, so a closure-built `ForwardRef('PredictRequest')` was
unresolvable → register the generated model in module globals; naked `TestClient`
(no `with`) never runs lifespan in the deprecated httpx path → wrap in context
manager; Bounds lookup used `cfg.get` on `None` → `(cfg or {}).get`.

### Phase 5 — Monitoring + Drift Detection ✅

**What was built**
- `src/monitoring/monitor.py` — a `Monitor` wrapper over the Prometheus client
  library, exported from the serving API at `GET /metrics`:
  | Metric | Kind | What it tracks |
  |---|---|---|
  | `modelops_prediction_latency_seconds` | Histogram | per-request inference latency (the API's own path) |
  | `modelops_predictions_total` | Counter | predictions, labelled `prediction`, `model_version` |
  | `modelops_prediction_errors_total` | Counter | failed predictions by error class |
  | `modelops_positive_rate` | Gauge | mean predicted probability (a concept-drift canary) |
  | `modelops_feature_value_seconds` | Histogram | per-feature input distributions, labelled `feature` |
  | `modelops_model` | Info | served model name + version |
  | `modelops_feature_psi` / `modelops_drift_alert` | Gauge | latest drift report mirrored into Prometheus |
- `src/monitoring/drift.py` — PSI (Population Stability Index) per feature:
  reference vs current windows binned into equal-width buckets (bins=10,
  EPSILON=0.001 guard); PSI > 0.2 flags a feature; `DriftReport` persists
  `data/drift_reports/drift_latest.json` and is logged as an MLflow `drift_check`
  run (params `psi_threshold`, `n_features`; metrics per-feature PSI; artifact =
  the report JSON).
- `scripts/simulate_drift.py` — builds two **measured** drift windows from the
  gold data: `current_same.csv` (reshuffled, no drift) and `current_shifted.csv`
  (only `income` shifted ×1.5 + noise) and prints a PSI preview before you commit
  to a scenario.
- `src/monitoring/__main__.py` — `python -m src.monitoring` runs the check
  against `monitoring.current_data_path`, logs the MLflow run, prints the verdict.
- Serving integration — `/predict` records each prediction (label, probability,
  features) and failures; `/metrics` exposes everything and mirrors the latest
  drift report into gauges so Grafana has one scrape target.
- `dashboards/` — `prometheus/prometheus.yml` (scrape `serving:8000/metrics`),
  Grafana **auto-provisioning** (datasource → Prometheus) + a dashboard with
  request rate, latency p95 (from the histogram), error rate, positive rate, a
  per-feature PSI bar gauge with the 0.2 threshold warning band, a drift alert
  stat, and the served-model info stat.
- `docker-compose.yml` — `prometheus` (9090) and `grafana` (3000) services added;
  config validated against the Docker version available here.

**Measured — PSI on the real gold data (1200 rows, 7 features)**
- No-drift window (`current_same.csv` vs reference): every feature PSI = 0.0000
- Drifted window (`current_shifted.csv`): **only `income` PSI = 0.636** (threshold
  0.2) → `alert=true`, `flagged=['income']`, runner prints **RETRAIN RECOMMENDED**;
  all six other features stay at 0.0000 — drift is isolated to one feature, which
  is the point of per-feature PSI.
- Verified over real HTTP: 25 `/predict` calls then `/metrics` → all families
  present with correct labels (`modelops_logistic_regression@3`), income PSI
  0.6356 mirrored as a gauge, drift alert 1.0, midnight-blue positive rate 0.168
  for the canonical row.
- Tests: **38 passing** (~75 s) — 8 new monitoring tests (PSI identity ≈ 0,
  shifted > threshold, only-shifted-feature flags, report persistence + alert,
  constant feature cannot alarm, Monitor records, `/metrics` export, drift-gauge
  sync).

**Key decisions**
- PSI over KL/Cramer's V: PSI is the industry-standard drift metric for
  one-dimensional continuous features, easy to threshold and explain in
  interviews ("0.1 stable, 0.1–0.25 monitor, >0.25 drift" — project uses 0.2 as
  "flag for retraining").
- Drift report doubles as the **input** to the Phase-6 retraining decision, not
  just a chart: `alert_retrain_on_drift: true` in config is the mailbox where the
  orchestrator checks before scheduling a retrain.
- `/metrics` is served by the same process as `/predict` so the dashboard shows
  exactly what the running model sees — no separate agent to drift out of sync.
- Prometheus histogram buckets stay at library defaults; the gate already owns
  the rigorous latency benchmark, so the service histogram is for *operations*
  (is it slow right now?), not for the deploy contract.

**Fix log** — the drift-scenario script couldn't import `src` (a script's
`sys.path[0]` is its own directory) → insert the repo root before importing;
`run_drift_check` hard-coded its output dir, which made it untestable with
tmp_path sandboxes → optional `report_dir` override; removed a test dependency
on importing `scripts/` (not a package) by generating drift windows inline.

---

## 8. Planned phases

| Phase | Module | Status |
|---|---|---|
| 4 | Serving + Docker + CI/CD (`src/serving/`, `docker/`, `.github/workflows/`, Airflow) | ✅ done |
| 5 | Monitoring + drift detection (`src/monitoring/`, Prometheus/Grafana, PSI) | ✅ done |
| 6 | Polish + benchmarks + demo (`docs/architecture.*`, `scripts/benchmark.py` + `demo.py`) | ✅ done |

### Phase 6 — Polish + Benchmarks + Demo ✅

**What was built**
- `docs/architecture.dot` → `docs/architecture.png` (graphviz, rendered with `dot`)
  — three planes: **triggers** (CI / CLI / Airflow) → **batch pipeline**
  (validate → train → track/register → gate → deploy), **serving runtime**
  (FastAPI + Monitor), and **observability** (Prometheus → Grafana, drift → alert
  → scheduler → retrain, closing the loop). Embed the PNG in the README; the DOT
  stays the editable source.
- `scripts/benchmark.py` — reproducible serving benchmark against a live uvicorn:
  full-HTTP latency percentiles, sequential + concurrent throughput, JSON out.
- `scripts/demo.py` — the whole lifecycle in one command with live-measured
  numbers at each step; reuses the exact `step_*` functions the CI and DAG use.

**Measured (final published numbers)**
- Serving (LR, seed-42 gold data, 2 000 real HTTP requests, 8-way concurrency):
  | Metric | Value |
  |---|---|
  | Latency avg / p50 / p95 / p99 | 3.09 / **3.02 / 4.22 / 4.77 ms** |
  | Sequential / concurrent throughput | **324 / 578 req/s** |
  | Process RSS (uvicorn) | ~290 MB (flat; no growth over the run) |
  → `data/benchmarks/serving_benchmark.json`
- Model quality on the sealed split (train 960 / test 240, seed 42): LR
  accuracy 0.7792 · f1 0.7854 · ROC-AUC **0.8649**; XGBoost ROC-AUC **0.8466**.
  The pipeline's own rule (best-by-ROC-AUC) therefore stages LR — measurement,
  not preference — and that model serves in Production.
- Gate contract as enforced: quality `min_delta=-0.01` per metric, latency
  p95 ≤ 200 ms, memory ≤ 512 MB, cost delta ≤ 10%; bootstrap runs SKIP
  quality/cost but still enforce the absolute caps.
- Demo run: validate PASS → train (LR 0.05s, XGB 10.4s, both tracked) → LR staged
  Candidate → gate **quality/latency/memory/cost all PASS** → deployed to
  Production → `/predict` `{"prediction":0,"probability":0.167547}` with
  `/metrics` live → drift check **income PSI 0.636 → RETRAIN ALERT**. Total
  ~15 s.
- Tests: **38 passing** (~65 s).

**Fix log / incident (rule 5 — the artifact-store lesson)**
- During a cleanup I removed `./mlruns`, not realizing the model binaries live
  there while the SQLite DB keeps only metadata. Every registered version kept
  its row but lost its artifacts → the gate's reference load failed with
  `MlflowException: No such artifact: ''`. Recovery: retrain → re-register →
  re-promote the best directly to Production (documented as a *recovery*, not a
  regular deploy), then verified `load_model(Production)` + a full demo run.
  For a real deployment the fix is an external artifact backend (S3) so no app
  process ever depends on local `./mlruns` files.

**Key decisions**
- Benchmarks are measured by a committed script against the real uvicorn process
  (not in-process estimates) and stored as a JSON artifact with its host/context.
- The demo shares the pipeline's actual step functions, so "what the demo showed"
  is definitionally "what CI runs" — no demo-only code path.
- Performance numbers are published together with their conditions (host, seed,
  request count), so they are reproducible rather than decorative.

---

## 9. How this document stays current

Every phase ends by updating this file: new modules, their commands, measured
numbers, decisions and fix-log entries — then the README's phase table and
`PROJECT_RECAP.md` are brought in sync in the same commit.