# ModelOps Platform — Study Project Recap

> A living document for MY ModelOps study project. I update this file every phase.
> If I can't explain a section in my own words, I haven't learned it yet.

- **Author:** Mohammad Yakub · AI & Data Science · KL University
- **Target:** AI/ML Engineer roles · Hyderabad
- **Guide:** `../ModelOps_Platform_Coding_Agent_Guide.pdf`
- **Source project (linked):** `../Model-Regression-Detection-System`
- **Mission:** Build a full ML lifecycle platform, incrementally. One phase fully working before the next. Commits after every feature. Never claim a metric that wasn't measured. Keep this file updated.

---

## 1. What I'm Building (One Paragraph I Can Say in an Interview)

Production ML systems break in two ways: (1) the code is good but the data is bad
(leakage, drift, schema changes), and (2) a "better" model still gets deployed even
when it's slower or regresses quality. I'm building `modelops-platform`, a complete
ML lifecycle that guards every step — data validation before training, experiment
tracking, a **regression gate** that blocks bad deployments, serving, monitoring,
drift detection, and retraining.

The full pipeline:

```
Dataset → Validation → Feature Engineering → Training → Experiment Tracking
→ Evaluation → Regression Gate → Model Registry → Deployment → Monitoring
→ Drift Detection → Retraining
```

---

## 2. The 12 Skills I Must Prove (and WHERE I Prove Them)

Each phase is deliberately "spoon-fed" below so that every build step teaches a
skill explicitly. This table is my scorecard — I tick a skill only after I've
built something with it AND can talk about it.

| # | Skill | Proven in Phase | Phase | Status |
|---|-------|-----------------|-------|--------|
| 1 | Python | Every module | 1–6 | [ ] |
| 2 | SQL + PostgreSQL | Data validation, registry | 1, 2 | [ ] |
| 3 | PyTorch + sklearn + XGBoost | Training module | 1 | [ ] |
| 4 | MLflow | Tracking + registry | 2 | [ ] |
| 5 | Airflow | Pipeline orchestration | 4 | [ ] |
| 6 | FastAPI | Serving | 4 | [ ] |
| 7 | Docker | All services containerized | 4 | [ ] |
| 8 | GitHub Actions | CI/CD pipeline | 4 | [ ] |
| 9 | Prometheus + Grafana | Monitoring dashboards | 5 | [ ] |
| 10 | Drift detection (PSI) | Monitoring module | 5 | [ ] |
| 11 | PySpark | Large-scale data processing | 6 (scale-up story) | [ ] |
| 12 | AWS S3 | Artifact storage | 6 (scale-up story) | [ ] |

---

## 3. Phase Tracker (Week = Phase)

| Phase | Name | DoD (Definition of Done) | Status |
|-------|------|--------------------------|--------|
| 1 | Data Validation + Training Pipeline | Validator + trainer + ≥3 tests + README | 🔄 In progress |
| 2 | Experiment Tracking + Model Registry | MLflow tracker + registry + UI + tests | ⏳ Planned |
| 3 | Regression Gate | Config-driven gate + report artifact + tests | ⏳ Planned |
| 4 | Serving + Docker + CI/CD | FastAPI + Dockerfile + compose + CI | ⏳ Planned |
| 5 | Monitoring + Drift Detection | Prometheus + PSI drift + Grafana + alert | ⏳ Planned |
| 6 | Polish + Docs + Demo | Architecture png + measured benchmarks + demo | ⏳ Planned |

**Phase rules (the operating contract I follow):**
1. One module fully working before starting the next.
2. Commit after every working feature — not at the end.
3. Never claim a metric that was not measured.
4. At least 2 tests per module (Phase 1 requires ≥3).
5. If something breaks: document why and how it was fixed.
6. Update the README every phase.
7. Thresholds live in config, never hard-coded decisions.
8. Reproducible: explicit seeds, configs, artifact metadata, data hashes.

---

# PHASE 1 — Data Validation + Training Pipeline

## 4a. Recap (What I Actually Did — I fill this in as I work)

> Treated as a changelog. Every commit gets a line here.

- [ ] `git init` + `.gitignore` + folder skeleton
- [ ] `src/validation/validator.py` — schema, missing values, ranges, leakage
- [ ] `src/training/trainer.py` — sklearn + XGBoost from a YAML config
- [ ] `configs/train_config.yaml` — all knobs (path, model, hyperparams, seeds)
- [ ] `tests/` — ≥3 tests
- [ ] README updated with Phase 1 status
- [ ] Measure & record: data size, rows, features, train duration

**Fix log (rule 5):** nothing yet

## 4b. Concepts This Phase Teaches (Spoon-Fed)

- **Data drift vs schema drift:** schema drift = columns/types change (breaks the
  contract); data drift = distribution changes (degrades accuracy). Validation
  catches **schema drift** now; monitoring catches **data drift** in Phase 5.
- **Train/test leakage:** information from the test set (or the future) leaking into
  training. Classic bug: the target column included as a feature, or `scaler.fit()`
  on full data before splitting. Leakage → inflated metrics → garbage in production.
- **Reproducible experiments:** same seed + same config + same data + same code =
  same result. Seeds pin randomness; configs capture every knob; hashes pin data.
- **sklearn vs XGBoost vs PyTorch:** sklearn = fast, simple, tabular/classic ML;
  XGBoost = best for tabular, gradient boosting, handles missing values; PyTorch =
  deep learning, unstructured data (images/text). The platform uses both to justify
  "why both".

## 4c. Skills Demonstrated This Phase (Spoon-Fed)

| Build step | What I type/create | Skill(s) it proves | How I'll explain it in an interview |
|---|---|---|---|
| Project skeleton | `mkdir`, `git init`, `.gitignore` | Python, Git, repo hygiene | "I keep infra as code and commit as I go." |
| Environment | `python3 -m venv .venv` + `requirements.txt` | Python packaging | "I pin dependencies so envs are reproducible." |
| Validator | `src/validation/validator.py` | Python, data quality, pandas | "I validate before training so bad data never reaches a model." |
| Leakage check | target-in-features scan in validator | ML best practice | "Leakage inflates metrics—my validator rejects it." |
| Report JSON | validator writes `validation_reports/*.json` | Python, SQL-ish data QA, serialization | "Every pipeline output is machine-readable for downstream gates." |
| Train config | `configs/train_config.yaml` | Config management, YAML | "I never hard-code; the trainer reads a config file." |
| Seeds | `random/numpy/torch` seeded in trainer | Reproducibility | "Same seed + config + data ⇒ same result, always." |
| Models | sklearn + XGBoost training | Python ML, model selection | "I compare a baseline (sklearn) vs XGBoost on the same split." |
| Artifact + metadata | save `model.pkl`/`.json` + duration log | Model lifecycle | "Every artifact carries metadata; nothing ships untracked." |
| Tests | `tests/` with pytest | pytest, QA | "Each module has tests; Phase 1 demands ≥3." |

## 4d. Step-by-Step Build Plan (Spoon-Fed, in Build Order)

1. **Init repo & skeleton** — `git init`; create folders
   `data/{raw,processed,validation_reports}` `src/{validation,training,tracking,
   registry,gate,serving,monitoring,retraining}` `tests` `configs` `docs`.
   Add `.gitignore` (`.venv/`, `__pycache__/`, `*.pkl`, MLflow dirs, `.env`).
   **Commit 1.**
2. **Create a small training CSV** in `data/raw/` (reuse the source project's
   dataset if compatible) — this becomes the gold data for all later phases.
3. **`configs/train_config.yaml`** — data path, target column, model choice,
   hyperparameters, seed, test_size, validation thresholds
   (e.g. `max_missing_pct: 0.1`, `min_rows: 100`).
4. **`src/validation/validator.py`** — checks in ORDER (fail fast):
   - schema: expected columns + dtypes
   - missing-value ratio per column vs threshold
   - value ranges (min/max vs config)
   - leakage: target column must not appear as a feature
   - writes `validation_reports/validation_latest.json`; raises clear errors on fail.
5. **`src/training/trainer.py`** — read config → seed everything → split
   (train/test) → train sklearn + XGBoost → save artifact + `model_metadata.json`
   (params, metrics, data hash, duration) → also hash the training data.
6. **`tests/test_validator.py` + `tests/test_trainer.py`** (≥3 tests):
   happy path, missing-schema failure, leakage failure, and "trainer reproduces
   with the same seed" test.
7. **Run everything; measure** actual numbers (rows, features, accuracy, seconds).
   Record them in README and this file. **Commit 2.**
8. **Update README** with Phase 1 status.

## 4e. Interview Questions "Prepped" by This Phase

- How do you prevent data leakage? → fail-fast validation that rejects target-in-features;
  fit transforms on train only.
- How do you make training reproducible? → seed all RNGs, config-driven trainer,
  data hash + artifact metadata.
- What would you check before training a model? → schema, missing values, ranges,
  leakage, class balance, data hash.
- What is schema validation and why does it matter? → enforces the data contract so
  downstream code and models never receive unexpected input.

---

# PHASE 2 — Experiment Tracking + Model Registry _(template — fill when started)_

## 5a. Recap
- [ ] `src/tracking/tracker.py` — MLflow run, log params/metrics/artifacts/data-hash
- [ ] `src/registry/registry.py` — register, stage (candidate/staging/production), rollback
- [ ] MLflow UI running via Docker
- [ ] Tests for registry operations

## 5b. Concepts
- What MLflow tracks: params, metrics, artifacts.
- Model registry: why it exists; candidate vs staging vs production.
- Model lineage; versioning ML artifacts.

## 5c. Skills Demonstrated
| Build step | Skill(s) it proves | How I'll explain it |
|---|---|---|
| Tracker integration | Python, MLflow | "I log everything a team needs to compare runs." |
| Registry + rollback | SQL/PostgreSQL metadata, model lifecycle | "I can promote or rollback a model by stage." |
| Docker MLflow | Docker | "The tracking server runs containerized for the team." |

## 5d. Interview Questions
- How do you track experiments at scale? / How do you roll back a bad model? /
  What is model lineage? / How do you version ML artifacts?

---

# PHASE 3 — Regression Gate _(template — fill when started)_

## 6a. Recap
- [ ] `src/gate/regression_gate.py` — compare candidate vs production metrics
- [ ] Config thresholds: `accuracy_min_delta: -0.01`, `latency_p95_max_ms: 200`,
      `memory_max_mb: 512`, `cost_max_delta_pct: 10`
- [ ] Gate report (side-by-side metrics, PASS/FAIL, per-dimension verdicts) as MLflow artifact
- [ ] Gate integrated into training pipeline; tests for PASS and FAIL

## 6b. Concepts
- Meaningful regression vs noise; statistical significance in model comparison.
- Why latency/memory/cost matter as much as accuracy. Deployment gates in CI/CD.

## 6c. Skills Demonstrated
| Build step | Skill(s) it proves | How I'll explain it |
|---|---|---|
| Multi-dimension comparison | Python, engineering judgment | "Accuracy alone misses slow or memory-hungry models." |
| Config-driven thresholds | Config management, CI/CD practice | "Gate rules live in config so deployment policy is auditable." |
| Gate blocks deploy | Automation, CI/CD | "A FAIL report halts the pipeline automatically." |

## 6d. Interview Questions
- What is a meaningful model regression? / Why is accuracy alone not enough? /
  How do you block a bad deployment automatically? / How do you balance quality vs latency vs cost?

---

# PHASE 4 — Serving + Docker + CI/CD _(template — fill when started)_

## 7a. Recap
- [ ] `src/serving/api.py` — `POST /predict`, `GET /health`, `GET /ready`, `GET /model/info`
- [ ] Docker multi-stage build; `docker-compose.yml` for all services
- [ ] `.github/workflows/ci.yml` — tests → validation → training → gate → build image
- [ ] Serving container baseline from guide (`python:3.11-slim`)

## 7b. Concepts
- FastAPI async; health check vs readiness check; zero-downtime deployment;
  multi-stage Docker builds.

## 7c. Skills Demonstrated
| Build step | Skill(s) it proves | How I'll explain it |
|---|---|---|
| FastAPI service | FastAPI, Python async, Pydantic | "Async endpoints keep long inference off the event loop." |
| Health vs ready | SRE/ops concepts | "Health = process alive; ready = model loaded, can serve." |
| Dockerfile + compose | Docker | "One command spins up the whole stack reproducibly." |
| CI workflow | GitHub Actions | "Every push runs validation → train → gate before any deploy." |
| Airflow pipeline | Airflow | "Airflow orchestrates retraining jobs on a schedule." |

## 7d. Interview Questions
- How do you serve a model without blocking the API? / Health vs readiness? /
  How does CI/CD work for ML? / How do you containerize an ML service?

---

# PHASE 5 — Monitoring + Drift Detection _(template — fill when started)_

## 8a. Recap
- [ ] `src/monitoring/monitor.py` — prediction distributions, latency p50/p95/p99,
      error rates, feature distributions, Prometheus export
- [ ] `src/monitoring/drift.py` — PSI per feature; flag PSI > 0.2; drift report;
      retraining alert
- [ ] Grafana dashboard JSON saved (request rate, latency percentiles, prediction
      distribution, drift scores, model version, gate history)

## 8b. Concepts
- Data vs concept vs model drift; PSI; KL divergence; Prometheus scrape vs Grafana
  visualize; operational meaning of alert thresholds.

## 8c. Skills Demonstrated
| Build step | Skill(s) it proves | How I'll explain it |
|---|---|---|
| Metrics middleware | Prometheus, Python | "Every request is measured, nothing is blind." |
| PSI drift | Drift detection, statistics | "PSI>0.2 per feature means the world moved—retrain." |
| Grafana dashboard | Monitoring/observability | "One dashboard shows health, drift, and model version." |

## 8d. Interview Questions
- How do you separate data drift from concept drift? / What is PSI and when do you
  use it? / How do you know when to retrigger training? / What metrics would you monitor?

---

# PHASE 6 — Polish + Documentation + Demo _(template — fill when started)_

## 9a. Recap
- [ ] Architecture diagram (draw.io/Excalidraw) → `docs/*.png`, embedded in README
- [ ] Measured benchmarks published (latency p50/p95/p99, throughput, memory,
      accuracy, gate threshold)
- [ ] 3–5 min demo: training → gate → serving → Grafana → drift alert
- [ ] Final README: what/architecture/pipeline/modules/run/tests/benchmarks/decisions/limitations/linked project

## 9b. Interview Story
Problem (models deployed without quality gates) → What I built (full lifecycle) →
Differentiator (regression gate blocks bad deploys) → Evidence (real latency,
accuracy, drift numbers) → Reflection (what I'd improve).

## 9c. Skills Demonstrated
| Build step | Skill(s) it proves | How I'll explain it |
|---|---|---|
| Benchmarks measured | Honest metrics | "Every number I quote was measured, not estimated." |
| Scale-up story | PySpark, AWS S3 | "At scale I'd add PySpark for data and S3 for artifact storage." |
| Architecture doc | System design, docs | "I can explain any arrow in this diagram." |

---

## 10. How to Use This File

- **At the end of every working session:** update the Phase `Recap` checklist, tick
  the Skills table, and copy real numbers into the relevant phase.
- **Before an interview:** read the same project end-to-end as a "final interview
  story" (section 9b).
- **Golden rule (rule 3):** if a number isn't a real measurement, it doesn't go in
  this file.