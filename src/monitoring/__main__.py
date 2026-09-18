"""Drift check runner: `python -m src.monitoring`.

Loads config, computes PSI per feature (reference vs current window), writes
`data/drift_reports/drift_latest.json`, logs the report as an MLflow artifact
and prints an operational verdict. A retraining alert is raised when any
per-feature PSI exceeds the config threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.config import load_config
from src.monitoring.drift import DriftReport, logged_outcome, run_drift_check

DEFAULT_CONFIG = "configs/train_config.yaml"
REPORT_PATH = Path("data/drift_reports/drift_latest.json")


def _log_drift_run(report: DriftReport) -> None:
    from src.tracking.tracker import MLflowTracker

    with MLflowTracker(experiment_name="modelops").run(run_name="drift_check") as _t:
        import mlflow  # local import keeps module import side-effect free

        mlflow.log_param("model", report.model)
        mlflow.log_param("reference", report.reference_file)
        mlflow.log_param("current", report.current_file)
        mlflow.log_param("psi_threshold", report.psi_threshold)
        mlflow.log_metric("flagged_features", sum(f.flag for f in report.feature_psi.values()))
        mlflow.log_metric("max_psi", max(d.psi for d in report.feature_psi.values()))
        for feature, drift in report.feature_psi.items():
            mlflow.log_metric(f"psi_{feature}", drift.psi)
        mlflow.log_artifact(str(REPORT_PATH), artifact_path="drift")


def main(config_path: str = DEFAULT_CONFIG) -> int:
    cfg = load_config(config_path)
    report = run_drift_check(cfg)
    _log_drift_run(report)

    print(logged_outcome(report))
    flagged = [f for f, d in report.feature_psi.items() if d.flag]
    if flagged and cfg["monitoring"].get("alert_retrain_on_drift", True):
        print(f"[drift] RETRAIN RECOMMENDED — flagged features: {flagged}")
    return 0  # alert is informational; retraining is a separate orchestration decision


if __name__ == "__main__":
    sys.exit(main())