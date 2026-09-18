<div align="center">

# ModelOps Platform

**A production ML lifecycle: validation → training → tracking → regression gate → serving → monitoring → drift → retraining.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green)
![Tests](https://img.shields.io/badge/tests-9%20passing-brightgreen)
![Phase](https://img.shields.io/badge/phase-1%2F6-blueviolet)

</div>

## What it is

Production ML systems don't fail because the code crashes — they fail because bad
data reaches the model, or a "better" model ships even when it regresses quality,
latency, or memory. `modelops-platform` guards every step of the ML lifecycle and
blocks bad deployments automatically.

This project evolves the [Model Regression Detection System](https://github.com/mohammadyakub-ai/model-regression-detection-system)
(a CI/CD-style evaluation harness for LLM features) into a complete, production-style
ML platform.

The full pipeline:

```
Dataset → Validation → Feature Engineering → Training → Experiment Tracking
→ Evaluation → Regression Gate → Model Registry → Deployment → Monitoring
→ Drift Detection → Retraining
```

## Phase status

| Phase | Module | Status |
|-------|--------|--------|
| 1 | Data validation + Training pipeline | ✅ done |
| 2 | Experiment tracking + Model registry (MLflow) | ✅ done (this phase) |
| 3 | Regression gate | ⏳ planned |
| 4 | Serving + Docker + CI/CD | ⏳ planned |
| 5 | Monitoring + drift detection | ⏳ planned |
| 6 | Polish + benchmarks + demo | ⏳ planned |

Live recap and interview-prep notes: **[PROJECT_RECAP.md](PROJECT_RECAP.md)**

## Quick start (Phase 1)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The dev machine sources ROS which pollutes PYTHONPATH — run like this:
PYTHONPATH= python -m src.pipeline                       # validate + train
PYTHONPATH= python -m src.pipeline --track               # + MLflow tracking + register
PYTHONPATH= python -m pytest tests/ -q                   # run tests
```

#### MLflow UI

The tracking server stores to `sqlite:///mlruns.db` and artifacts to `./mlruns`, and the
model registry lives in the same SQL database (SQLite for dev; PostgreSQL is the stated
production target — SQL backend either way).

```bash
# Option A — Docker (requires a running daemon):
docker compose up -d --build mlflow     # http://localhost:5000

# Option B — local server from the venv (no Docker daemon on this box):
MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/mlflow server \
  --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000            # http://localhost:5000
```

> Note: this machine has the Docker CLI but no `dockerd` daemon, so the container
> path is validated (`docker compose config` OK) and runs as-is on a Docker-enabled host.

Known environment quirk: if `source /opt/ros/humble/setup.bash` ran in your shell,
unset `PYTHONPATH` for pip/python commands, otherwise pytest loads ROS plugins and
fails with `ModuleNotFoundError: No module named 'lark'`.

## What Phase 1 delivers

### Data (versioned artifact)

`data/raw/dataset_v1.csv` — 1,200 loan-risk rows, 9 columns, balanced 600/600,
generated deterministically by `scripts/generate_sample_data.py` (seed 42).
The CSV — not the generator — is the gold artifact; it is SHA-256 hashed and the
hash travels with every trained model as lineage.

> The source project's dataset was an LLM golden-set (JSON), not tabular, so it is
> referenced as lineage rather than reused for sklearn/XGBoost training.

### Validation — `src/validation/validator.py`

Config-driven checks, fail fast (raise `ValidationError` with a partial report):

1. file exists
2. schema — required columns present (features + target)
3. row count ≥ `min_rows`
4. missing-value ratio ≤ `max_missing_pct` per column
5. value ranges per `column_ranges` in config
6. leakage — the target column must not be a feature
7. id column uniqueness

Every run writes `data/validation_reports/validation_latest.json` (typed with Pydantic).

### Training — `src/training/trainer.py`

- reads everything from `configs/train_config.yaml`
- seeds `random`, `numpy`, `torch` so the same config + data = same result
- stratified 80/20 split (seed 42)
- trains an sklearn `LogisticRegression` baseline **and** an `XGBClassifier`
- saves `artifacts/models/<model>_v1.joblib` + metadata JSON:
  params, metrics, data SHA-256, seed, training duration, feature list

### Measured results (Phase 1, seed 42, 960 train / 240 test)

| Model | Accuracy | F1 | ROC-AUC | Train time |
|---|---|---|---|---|
| LogisticRegression | 0.7792 | 0.7854 | 0.8649 | 0.06 s |
| XGBoost (100 est, depth 3) | 0.7708 | 0.7679 | 0.8466 | 14.7 s |

- Data file SHA-256: `ffc1f77291204e3aa4e5a2f39e97127c62b351612477ac4d15b6b21cf8822b3c`
- Tests: **15 passed** (~60 s) — validator/trainer Phase 1 + tracking/registry Phase 2

## What Phase 2 delivers

### Experiment tracking — `src/tracking/tracker.py`

`MLflowTracker` owns the run lifecycle (FINISHED / FAILED), logging for every model:

- params — model hyperparameters, split test size/seed, feature count
- metrics — accuracy, F1, ROC-AUC, training duration (all measured at train time)
- lineage — source data file + its SHA-256
- the model artifact itself (`mlflow.sklearn` / `mlflow.xgboost` flavor)

Design detail: **one MLflow run per model**. A single run holding both models made
the registry show xgboost's numbers on the logistic-regression version (the same
metric keys collided). Per-model runs keep every registered version's metrics honest.

### Model registry — `src/registry/registry.py`

`ModelRegistry` provides the lifecycle stages the guide defines —
`candidate -> staging -> production` — plus rollback:

- `register` — new version from a tracked run's model_uri
- `list_versions` — every version with current stage + its run's real metrics
- `promote` — validate + transition to a stage
- `rollback` — demote the current holder, promote the newest older version

**Design decision:** MLflow has no native `Candidate` stage (only `None` /
`Staging` / `Production` / `Archived`). `Candidate` is implemented as a version tag
(`modelops_stage=candidate`) that the registry translates on read/write. Good
interview point: "stages are being deprecated in MLflow; the platform maps its own
stage vocabulary onto tags so nothing is hard-wired to a vendor quirk."

The pipeline stages the best **measured** model (highest ROC-AUC) as `Candidate`,
never `Production` — Production is decided by the regression gate in Phase 3.

### Measured Phase 2 registry state (fresh run, seed 42)

| Registered model | Version | Stage | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|---|
| modelops_logistic_regression | 1 | Candidate | 0.7792 | 0.7854 | 0.8649 |
| modelops_xgboost | 1 | None | 0.7708 | 0.7679 | 0.8466 |

MLflow UI (verified via `GET /health` → 200 and the registered-models API)
shows these runs, versions and metrics.

### Tests

- `tests/test_validator.py` — passes on project data; rejects leakage, missing
  column, out-of-range values, too-few rows, excessive missing values
- `tests/test_trainer.py` — artifacts + metadata written with data hash;
  same-seed runs produce identical metrics (reproducibility); data hash tracked
- `tests/test_tracking_registry.py` — tracker records a FINISHED run with
  metrics/params; register→promote→list; candidate-via-tag; rollback;
  training logs one run per model with clean per-model metrics

## Repository layout

```text
.
├── configs/train_config.yaml     # all pipeline knobs
├── data/raw/dataset_v1.csv       # versioned gold data
├── scripts/generate_sample_data.py
├── src/
│   ├── config.py                 # config loading / typed config assembly
│   ├── pipeline.py               # Phase 1 orchestration: validate -> train
│   ├── validation/validator.py
│   └── training/trainer.py
├── tests/
├── PROJECT_RECAP.md              # week-by-week study recap + interview prep
└── requirements.txt
```

Planned modules (skeleton exists): `tracking/`, `registry/`, `gate/`, `serving/`,
`monitoring/`, `retraining/` + `pipelines/airflow/`, `docker/`, `dashboards/`.

## Design decisions (Phase 1)

| Decision | Rationale |
|---|---|
| Committed CSV as the gold artifact | Data is versioned like code; the hash is lineage |
| Deterministic synthetic data | Known true distributions → honest metrics + reliable drift tests in Phase 5 |
| Config-driven thresholds | Deployment/validation policy lives in YAML, never hard-coded |
| sklearn baseline + XGBoost | Baseline vs powerful model on the same split — an honest comparison story |
| Reject-on-validation-failure | The pipeline refuses to train on bad data instead of "warning" |
| Pydantic for config/report | Typed contracts so CI can consume the JSON report safely |
| SQL store for MLflow | SQLite for dev, PostgreSQL for prod — SQL skills proven in the registry layer |
| One MLflow run per model | Registered versions must carry their own, non-colliding metrics |
| Candidate as a tag, not a stage | MLflow has no native Candidate; platform vocabulary maps onto tags cleanly |
| Best-by-ROC-AUC → Candidate | Promotion is data-driven and measured, never hard-coded

## Limitations (honest)

- Synthetic data: fine for teaching lifecycle skills, not for real lending
- Phase 1 has no feature engineering yet (guide keeps it minimal early on)
- No model persists across phases yet — registry (Phase 2) owns that

## License

MIT — see [LICENSE](LICENSE).