"""Monitoring tests: PSI math (identity ~0, drift > threshold), the drift report
contract, and Prometheus export via the serving /metrics endpoint."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import load_config
from src.monitoring.drift import compute_drift, feature_psi, run_drift_check
from src.monitoring.monitor import Monitor
from src.registry.registry import ModelRegistry
from src.serving.api import create_app

CFG = load_config("configs/train_config.yaml")
FEATURES = list(CFG["data"]["features"])


@pytest.fixture()
def sample() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference = pd.read_csv(CFG["data"]["raw_path"])
    same = reference.sample(frac=1.0, random_state=42).reset_index(drop=True)
    rng = np.random.default_rng(42)
    shifted = reference.copy()
    shifted["income"] = np.clip(
        shifted["income"] * 1.5 + rng.normal(0, 8_000, len(shifted)), 20_000, 250_000
    ).astype(int)
    return reference, same, shifted


# ---------- PSI math ----------

def test_psi_identical_distributions_is_zero():
    rng = np.random.default_rng(7)
    values = rng.normal(50, 10, 5000)
    assert feature_psi(pd.Series(values), pd.Series(values)) == pytest.approx(0.0, abs=1e-6)


def test_psi_shifted_distribution_is_flagged():
    rng = np.random.default_rng(7)
    ref = pd.Series(rng.normal(50, 10, 5000))
    cur = pd.Series(rng.normal(55, 10, 5000))
    assert feature_psi(ref, cur) > 0.2


# ---------- drift report contract ----------

def test_compute_drift_flags_only_shifted_feature(sample):
    reference, same, shifted = sample
    same_drift = compute_drift(reference, same, FEATURES, threshold=0.2)
    shifted_drift = compute_drift(reference, shifted, FEATURES, threshold=0.2)

    assert all(not f.flag for f in same_drift.values()), "no-drift window must not flag"
    assert shifted_drift["income"].flag is True
    flagged = [f for f, d in shifted_drift.items() if d.flag]
    assert flagged == ["income"], f"only income should flag, got {flagged}"


def test_run_drift_check_persists_report(tmp_path, sample):
    reference, _, shifted = sample
    ref_path = tmp_path / "ref.csv"
    cur_path = tmp_path / "cur.csv"
    reference.to_csv(ref_path, index=False)
    shifted.to_csv(cur_path, index=False)

    cfg = dict(CFG)
    cfg["monitoring"] = {
        "reference_data_path": str(ref_path),
        "current_data_path": str(cur_path),
        "psi_threshold": 0.2,
        "psi_bins": 10,
    }
    report = run_drift_check(cfg, report_dir=tmp_path)

    persisted = json.loads((tmp_path / "drift_latest.json").read_text())
    assert persisted["alert"] is True
    assert report.model.startswith("modelops_")
    assert set(persisted) >= {"feature_psi", "alert", "psi_threshold"}


def test_constant_feature_cannot_false_alarm():
    ref = pd.Series(np.zeros(1000))
    cur = pd.Series(np.ones(1000))
    assert feature_psi(ref, cur) == 0.0  # no distribution to drift


# ---------- Prometheus export ----------

def test_monitor_records_prediction_and_error():
    mon = Monitor()
    mon.set_model("modelops_logistic_regression", 1)
    mon.record_prediction("modelops_logistic_regression@1", 0, 0.167547, {"income": 72000.0})
    mon.record_error("modelops_logistic_regression@1", "exception")
    text = mon.render().decode()

    assert "modelops_predictions_total" in text
    assert "modelops_prediction_errors_total" in text
    assert "modelops_positive_rate" in text
    assert "modelops_feature_value_seconds" in text


def test_metrics_exported_by_serving(tmp_path):
    reg = ModelRegistry(tracking_uri=f"sqlite:///{tmp_path}/empty.db")
    app = create_app(registry=reg, model_name="modelops_logistic_regression", config_path=None)
    with TestClient(app) as c:
        r = c.get("/metrics")
    assert r.status_code == 200
    for metric in (
        "modelops_prediction_latency_seconds",
        "modelops_predictions_total",
        "modelops_prediction_errors_total",
        "modelops_feature_psi",
        "modelops_drift_alert",
    ):
        assert metric in r.text


def test_sync_drift_exposes_psi_gauges(tmp_path):
    mon = Monitor()
    report = tmp_path / "drift_latest.json"
    report.write_text(
        json.dumps({"alert": True, "feature_psi": {"income": {"psi": 0.31, "flag": True}}})
    )
    assert mon.sync_drift(str(report)) is True
    text = mon.render().decode()
    assert 'modelops_feature_psi{feature="income"} 0.31' in text
    assert "modelops_drift_alert 1.0" in text