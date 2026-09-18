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
| 1 | Data validation + Training pipeline | ✅ done (this phase) |
| 2 | Experiment tracking + Model registry (MLflow) | ⏳ planned |
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
PYTHONPATH= python -m pytest tests/ -q                   # run tests
```

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
- Tests: **9 passed** (~9.5 s) — validator PASS/FAIL cases + trainer reproducibility

### Tests

- `tests/test_validator.py` — passes on project data; rejects leakage, missing
  column, out-of-range values, too-few rows, excessive missing values
- `tests/test_trainer.py` — artifacts + metadata written with data hash;
  same-seed runs produce identical metrics (reproducibility); data hash tracked

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

## Limitations (honest)

- Synthetic data: fine for teaching lifecycle skills, not for real lending
- Phase 1 has no feature engineering yet (guide keeps it minimal early on)
- No model persists across phases yet — registry (Phase 2) owns that

## License

MIT — see [LICENSE](LICENSE).