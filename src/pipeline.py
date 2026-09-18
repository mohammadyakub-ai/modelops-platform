"""Pipeline orchestration.

  run():                Phase 1 — validate + train (no MLflow).
  run_with_tracking():  Phase 2/3 — validate -> train -> track -> register ->
                        regression gate -> deploy if PASS (else block).

Deployment contract (Phase 3):
  - the best measured model is registered and staged Candidate
  - the regression gate compares it against the current Production model
    (same family); with no Production yet it bootstraps on absolute caps
  - gate PASS  -> promote Candidate -> Staging -> Production (deploy)
  - gate FAIL  -> model stays Candidate, deployment is blocked, exit 1
The gate report is written to data/gate_reports/ and logged as an MLflow
artifact inside its own mlflow run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import load_config, build_validation_config
from src.gate.regression_gate import GateConfig, RegressionGate
from src.registry.registry import ModelRegistry
from src.tracking.tracker import MLflowTracker
from src.training.trainer import train
from src.validation.validator import DataValidator

GATE_REPORT_DIR = Path("data/gate_reports")
GATE_REPORT_PATTERN = "gate_latest.json"


def _validate(cfg: dict) -> None:
    validator = DataValidator(build_validation_config(cfg))
    report = validator.validate(
        cfg["data"]["raw_path"], report_dir=cfg["output"]["report_dir"]
    )
    print(f"[validation] PASSED rows={report.rows} file_sha256={report.file_sha256[:12]}...")


def _best_model(result: dict) -> str:
    """Choose the best measured model by ROC-AUC. A measurement, not a guess."""
    return max(
        result["models"],
        key=lambda name: result["models"][name]["metrics"]["roc_auc"],
    )


def run(config_path: str = "configs/train_config.yaml") -> dict:
    cfg = load_config(config_path)
    cfg["_config_path"] = config_path
    _validate(cfg)
    result = train(cfg)
    print(f"\n[pipeline] done — data_sha256={result['data_sha256'][:12]}...")
    return result


def _run_gate(cfg: dict, registry: ModelRegistry, candidate: str, version: int) -> dict:
    gate_cfg = GateConfig(
        **cfg["gate"].get("benchmark", {}),
        **cfg["gate"].get("quality", {}),
        latency_p95_max_ms=cfg["gate"]["latency_p95_max_ms"],
        memory_max_mb=cfg["gate"]["memory_max_mb"],
        cost_max_delta_pct=cfg["gate"]["cost_max_delta_pct"],
    )
    gate = RegressionGate(gate_cfg)

    data_path = Path(cfg["data"]["raw_path"])
    features = list(cfg["data"]["features"])
    X = pd.read_csv(data_path)[features]

    report = gate.run(registry, f"modelops_{candidate}", version, X)

    GATE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = GATE_REPORT_DIR / GATE_REPORT_PATTERN
    out.write_text(report.model_dump_json(indent=2))
    return {"report": report, "path": out}


def run_with_tracking(config_path: str = "configs/train_config.yaml") -> dict:
    cfg = load_config(config_path)
    cfg["_config_path"] = config_path
    _validate(cfg)

    tracker = MLflowTracker(experiment_name="modelops")
    result = train(cfg, tracker=tracker)

    registry = ModelRegistry(tracking_uri=tracker.tracking_uri)
    registered = {}
    for name, entry in result["models"].items():
        version = registry.register(entry["model_uri"], name=f"modelops_{name}")
        registered[name] = version["version"]
        print(f"[registry] registered {version['name']} v{version['version']}")

    best = _best_model(result)
    cand_name = f"modelops_{best}"
    cand_version = registered[best]
    registry.promote(cand_name, cand_version, "Candidate")
    print(f"[registry] best by roc_auc={result['models'][best]['metrics']['roc_auc']}: {best} -> Candidate (v{cand_version})")

    # ---------- regression gate ----------
    gate_out = _run_gate(cfg, registry, best, cand_version)
    report, gate_path = gate_out["report"], gate_out["path"]
    reference = report.reference
    ref_txt = (
        f"v{reference.version} (latency p95={reference.latency.p95_ms:.1f}ms, "
        f"auc={reference.metrics.get('roc_auc')})"
        if reference is not None
        else "none (bootstrap)"
    )
    print(
        f"\n[gate] candidate={best} v{cand_version} vs production reference={ref_txt}\n"
        f"[gate] verdicts: quality={report.verdicts['quality']} latency={report.verdicts['latency']} "
        f"memory={report.verdicts['memory']} cost={report.verdicts['cost']} "
        f"-> {'PASS' if report.passed else 'FAIL'}"
    )

    # gate report persisted as an MLflow artifact inside its own run
    with tracker.run(run_name="regression_gate") as _t:
        import mlflow  # local import keeps module import side-effect free

        mlflow.log_param("candidate", cand_name)
        mlflow.log_param("candidate_version", cand_version)
        mlflow.log_metrics(
            {
                "gate_passed": int(report.passed),
                "latency_p95_ms": report.candidate.latency.p95_ms,
                "memory_mb": report.candidate.latency.memory_mb,
                "cost_usd_per_1m": report.candidate.latency.cost_usd_per_1m,
            }
            if report.candidate.latency is not None
            else {"gate_passed": int(report.passed)}
        )
        mlflow.log_artifact(str(gate_path), artifact_path="gate")

    if not report.passed:
        print("[gate] BLOCKED deployment — candidate stays Candidate (exit 1)")
        raise SystemExit(1)

    registry.promote(cand_name, cand_version, "Staging")
    registry.promote(cand_name, cand_version, "Production")
    print(f"[gate] PASS — deployed {best} v{cand_version} to Production")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--track", action="store_true", help="track + register + gate + deploy")
    args = parser.parse_args()
    run_with_tracking() if args.track else run()