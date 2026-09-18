"""Prometheus metrics for the serving service.

Exposes a /metrics endpoint (scraped by Prometheus) with the operational
surface of the model service:

  modelops_prediction_latency_seconds     Histogram  per-request latency
  modelops_predictions_total              Counter    prediction label + version
  modelops_prediction_errors_total        Counter    error label
  modelops_positive_rate                  Gauge      probability mean over last N
  modelops_feature_value_seconds          Histogram  per-feature input distribution
  modelops_model_info                     Info       model name + version

Nothing here is estimated: every value is incremented from a real request or a
real prediction. Standing up the counters is cheap and non-blocking, so the
service hot-path stays untouched by observability overhead worth measuring.
"""

from __future__ import annotations

from collections import deque

from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest

REQUEST_LATENCY_HIST = Histogram(
    "modelops_prediction_latency_seconds",
    "Latency of /predict requests served by the API (seconds)",
    buckets=(0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
)
PREDICTIONS = Counter(
    "modelops_predictions_total",
    "Predictions served, by predicted class and model version",
    ["model_version", "prediction"],
)
PREDICTION_ERRORS = Counter(
    "modelops_prediction_errors_total",
    "Failed predictions, by error label",
    ["model_version", "error"],
)
POSITIVE_RATE = Gauge(
    "modelops_positive_rate",
    "Mean predicted probability over the last observed predict requests",
)
FEATURE_DISTRIBUTION = Histogram(
    "modelops_feature_value_seconds",
    "Distribution of feature values seen by /predict",
    ["feature"],
    buckets=(0, 250, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 250000, float("inf")),
)
MODEL_INFO = Info("modelops_model", "Served model name + version")
DRIFT_PSI = Gauge("modelops_feature_psi", "Latest per-feature PSI from the drift check", ["feature"])
DRIFT_ALERT = Gauge("modelops_drift_alert", "1 when the latest drift check flagged any feature")


def feature_values(request_values: dict[str, float], features: list[str]) -> dict[str, float]:
    return {f: float(request_values[f]) for f in features if f in request_values}


class Monitor:
    """Collects Prometheus metrics from the serving API."""

    def __init__(self) -> None:
        self._positive: deque[float] = deque(maxlen=500)
        self._last_version = "unknown"

    def set_model(self, name: str, version: int) -> None:
        version_str = f"{name}@{version}"
        self._last_version = version_str
        MODEL_INFO.info({"name": name, "version": str(version)})

    def record_latency(self, seconds: float) -> None:
        REQUEST_LATENCY_HIST.observe(seconds)

    def record_prediction(self, version: str, prediction: int, probability: float, features: dict[str, float]) -> None:
        self._last_version = version
        MODEL_INFO.info({"name": _model_name(version), "version": _model_version(version)})
        PREDICTIONS.labels(model_version=version, prediction=str(prediction)).inc()
        self._positive.append(float(probability))
        POSITIVE_RATE.set(sum(self._positive) / len(self._positive))
        for feature, value in features.items():
            try:
                FEATURE_DISTRIBUTION.labels(feature=feature).observe(float(value))
            except (TypeError, ValueError):
                continue

    def record_error(self, version: str, error: str) -> None:
        PREDICTION_ERRORS.labels(model_version=version or "unknown", error=error).inc()

    def sync_drift(self, report_path: str | None = None) -> bool:
        """Load the latest drift report (if any) into Prometheus gauges."""
        from pathlib import Path

        path = Path(report_path or "data/drift_reports/drift_latest.json")
        if not path.exists():
            return False
        import json

        report = json.loads(path.read_text())
        for feature, drift in report.get("feature_psi", {}).items():
            DRIFT_PSI.labels(feature=feature).set(float(drift["psi"]))
        DRIFT_ALERT.set(int(bool(report.get("alert"))))
        return True

    def render(self) -> bytes:
        return generate_latest()


def _model_name(version: str) -> str:
    return version.split("@")[0] if "@" in version else version


def _model_version(version: str) -> str:
    return version.split("@")[1] if "@" in version else "?"