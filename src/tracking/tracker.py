"""Experiment tracking via MLflow.

Owns the run lifecycle: params, metrics, artifacts, and lineage (data file +
data hash) get recorded per run. Everything the tracker receives is passed
straight through from the trainer — nothing is estimated here.

Notes on MLflow 3.x: `log_model` now takes `name=` (artifact paths with "/"
are rejected) and returns a ModelInfo whose `.model_uri` can be registered
in the model registry.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow

DEFAULT_TRACKING_URI = "sqlite:///mlruns.db"
DEFAULT_EXPERIMENT = "modelops"


def _scalar(value: Any) -> Any:
    """mlflow only accepts numbers/strings — collapse everything else safely."""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class MLflowTracker:
    def __init__(self, tracking_uri: str | None = None, experiment_name: str = DEFAULT_EXPERIMENT):
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        self.tracking_uri = tracking_uri or DEFAULT_TRACKING_URI
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.artifacts: dict[str, str] = {}

    @contextmanager
    def run(self, run_name: str | None = None) -> Iterator["MLflowTracker"]:
        """Start an MLflow run; end it FINISHED on success, FAILED on exception."""
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=run_name) as active:
            self.run_id = active.info.run_id
            self.artifacts = {}
            try:
                yield self
            except Exception:
                mlflow.end_run(status="FAILED")
                raise

    def log_data(self, data_file: str | Path, data_sha256: str) -> None:
        """Record data lineage: which file produced this run and its hash."""
        mlflow.log_param("data_file", Path(data_file).name)
        mlflow.log_param("data_sha256", data_sha256)

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params({k: _scalar(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def log_model(self, model: Any, name: str) -> str:
        """Log a trained estimator and return the model_uri for the registry.

        XGBoost models get the xgboost flavor; everything else the sklearn flavor.
        """
        if self.run_id is None:
            raise RuntimeError("log_model() requires an active tracker.run()")
        try:
            from xgboost import XGBModel

            is_xgb = isinstance(model, XGBModel)
        except ImportError:
            is_xgb = False

        if is_xgb:
            info = mlflow.xgboost.log_model(model, name=name)
        else:
            info = mlflow.sklearn.log_model(model, name=name)

        self.artifacts[name] = info.model_uri
        return info.model_uri