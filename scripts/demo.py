"""One-command lifecycle demo (~3 min): the whole platform in five visible steps.

    PYTHONPATH= MLFLOW_DISABLE_AGENT_HINT=1 python scripts/demo.py

  1. validate   — data schema / leakage / range checks against the gold CSV
  2. train+track — sklearn LR + XGBoost on the sealed split, MLflow-tracked,
                   best-by-ROC-AUC staged as Candidate
  3. gate+deploy — regression gate vs the Production reference; PASS promotes
                   Candidate -> Staging -> Production (FAIL would exit 1)
  4. serve      — Production model served by FastAPI; real /predict + /metrics
  5. drift      — PSI reference vs current window -> RETRAIN alert

Every number printed is measured live (no hard-coded claims in this file).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.monitoring.drift import run_drift_check
from src.pipeline import step_gate_deploy, step_train_track, step_validate
from src.registry.registry import ModelRegistry
from src.tracking.tracker import MLflowTracker

CFG_PATH = "configs/train_config.yaml"
ROW = {"age": 45, "income": 72_000, "credit_score": 680, "loan_amount": 24_000,
       "employment_years": 8, "num_defaults": 1, "has_collateral": 1}

_sections: list[tuple[str, float]] = []


def section(tag: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    print(f"\n─── {tag} ───")
    result = fn(*args, **kwargs)
    dur = time.perf_counter() - t0
    _sections.append((tag, dur))
    print(f"✓ {tag} in {dur:.1f}s")
    return result


def main() -> None:
    cfg = load_config(CFG_PATH)
    start = time.perf_counter()

    print("=" * 62)
    print(" modelops-platform — 5-step lifecycle demo (everything measured live)")
    print("=" * 62)
    print(f" gate contract: p95_max={cfg['gate']['latency_p95_max_ms']}ms "
          f"mem_max={cfg['gate']['memory_max_mb']}MB "
          f"quality_min_delta={cfg['gate']['quality']['min_roc_auc_delta']}")

    # 1. validate
    section("1/5 VALIDATE", lambda: step_validate(cfg))

    # 2. train + track + register + stage best as Candidate
    tracker = MLflowTracker()
    trained = section("2/5 TRAIN + TRACK + STAGE CANDIDATE",
                      lambda: step_train_track(cfg, tracker))
    for name, entry in trained["result"]["models"].items():
        m = entry["metrics"]
        print(f"     {name:22s} auc={m['roc_auc']:.4f} acc={m['accuracy']:.4f} "
              f"train/test={m['train_rows']}/{m['test_rows']}")

    # 3. gate
    gated = section("3/5 GATE + DEPLOY",
                    lambda: step_gate_deploy(cfg, trained["registry"],
                                             trained["cand_name"],
                                             trained["cand_version"]))

    # 4. serve
    def _serve():
        from fastapi.testclient import TestClient
        from src.serving.api import create_app

        reg = ModelRegistry()
        app = create_app(registry=reg, model_name="modelops_logistic_regression",
                         config_path=CFG_PATH)
        with TestClient(app) as c:
            pred = c.post("/predict", json=ROW)
            assert pred.status_code == 200, pred.text
            info = c.get("/model/info").json()
            metrics = c.get("/metrics").text
        print(f"     /predict -> {json.dumps(pred.json())}")
        print(f"     /model/info -> model={info.get('model')} version={info.get('version')} "
              f"served_latency_p95={info.get('served_latency_ms', {}).get('p95')}ms")
        lines = [ln for ln in metrics.splitlines()
                 if ln.startswith(("modelops_predictions_total", "modelops_drift_alert",
                                   "modelops_model_info"))]
        print("     /metrics -> " + "; ".join(lines[:3]))
        return pred.json()

    served = section("4/5 SERVE (FastAPI + /metrics)", _serve)

    # 5. drift
    def _drift():
        report = run_drift_check(cfg, model_name="modelops_logistic_regression")
        flagged = sorted(f for f, d in report.feature_psi.items() if d.flag)
        print(f"     reference={report.reference_file} current={report.current_file} "
              f"threshold={report.psi_threshold}")
        print(f"     PSI: " + ", ".join(
            f"{f}={d.psi:.3f}{' ⚠' if d.flag else ''}"
            for f, d in report.feature_psi.items()))
        if report.alert:
            print(f"     ⛔ RETRAIN ALERT — flagged={flagged}")
        else:
            print("     ✅ no drift")
        return report

    report = section("5/5 DRIFT CHECK (PSI)", _drift)

    # summary
    print("\n" + "=" * 62)
    print(f" DEMO COMPLETE in {time.perf_counter() - start:.1f}s")
    for tag, dur in _sections:
        print(f"   {tag:38s} {dur:6.1f}s")
    print("=" * 62)
    ext = f"RETRAIN ALERT → {sorted(f for f, d in report.feature_psi.items() if d.flag)}" if report.alert else "no drift"
    print(f" outcome: production={served['prediction']} | {', '.join(f'{k}={v}' for k, v in gated['report'].verdicts.items())} | {ext}")


if __name__ == "__main__":
    main()