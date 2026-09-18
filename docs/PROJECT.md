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
├── configs/train_config.yaml          # all pipeline knobs
├── data/
│   ├── raw/dataset_v1.csv             # versioned gold data (SHA-256 tracked)
│   ├── processed/                     # (future phases)
│   └── validation_reports/            # generated validation_latest.json
├── scripts/generate_sample_data.py    # deterministic data generator
├── src/
│   ├── config.py                      # YAML -> typed config
│   ├── pipeline.py                    # validate -> train (+ --track)
│   ├── validation/validator.py        # Phase 1: schema/leakage/ranges checks
│   ├── training/trainer.py            # Phase 1: sklearn + XGBoost training
│   ├── tracking/tracker.py            # Phase 2: MLflowTracker
│   ├── registry/registry.py           # Phase 2: ModelRegistry (stages/rollback)
│   ├── gate/    serving/  monitoring/
│   ├── retraining/                    # Phase 3+ (skeletons present)
├── tests/                             # pytest suites (15 passing)
├── docker/mlflow.Dockerfile
├── docker-compose.yml                 # mlflow service (UI)
├── docs/PROJECT.md                    # this file
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

### Run the pipeline (validation + training)
```bash
PYTHONPATH= .venv/bin/python -m src.pipeline            # Phase 1: validate + train
PYTHONPATH= .venv/bin/python -m src.pipeline --track    # Phase 2: + MLflow tracking + registry
```

Outputs:
- `data/validation_reports/validation_latest.json` — validation report
- `artifacts/models/<model>_v1.joblib` + `<model>_v1.json` — model + metadata
- `mlruns.db` (SQLite) + `mlruns/` — MLflow runs, artifacts, registry

### Tests
```bash
PYTHONPATH= .venv/bin/python -m pytest tests/ -q          # full suite
PYTHONPATH= .venv/bin/python -m pytest tests/test_registry* -q   # a module
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
```

Phase 3 (`gate/`) will sit between training and registry promotion;
Phase 4 (`serving/`) exposes the production model; Phase 5 (`monitoring/`)
adds Prometheus + Grafana + PSI drift.

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

## 8. Planned phases

| Phase | Module | Status |
|---|---|---|
| 3 | Regression gate (`src/gate/`) | next |
| 4 | Serving + Docker + CI/CD (`src/serving/`, `docker/`, `.github/workflows/`, Airflow) | planned |
| 5 | Monitoring + drift detection (`src/monitoring/`, Prometheus/Grafana, PSI) | planned |
| 6 | Architecture diagram, measured benchmarks, demo, final README | planned |

## 9. How this document stays current

Every phase ends by updating this file: new modules, their commands, measured
numbers, decisions and fix-log entries — then the README's phase table and
`PROJECT_RECAP.md` are brought in sync in the same commit.