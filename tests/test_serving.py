"""Serving API tests — self-contained: train a tiny LR into a temp registry,
promote it to Production, then exercise the FastAPI contract end-to-end."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import load_config
from src.registry.registry import ModelRegistry
from src.serving.api import create_app
from src.tracking.tracker import MLflowTracker
from src.training.trainer import train

CFG = load_config("configs/train_config.yaml")


@pytest.fixture(scope="session")
def prod_registry(tmp_path_factory) -> ModelRegistry:
    tmp_path = tmp_path_factory.mktemp("reg")
    cfg = dict(CFG)
    cfg["_config_path"] = "configs/train_config.yaml"
    uri = f"sqlite:///{tmp_path}/mlruns.db"
    tracker = MLflowTracker(tracking_uri=uri)
    result = train(cfg, tracker=tracker)
    reg = ModelRegistry(tracking_uri=uri)
    name = "modelops_logistic_regression"
    vr = reg.register(result["models"]["logistic_regression"]["model_uri"], name=name)
    reg.promote(name, vr["version"], "Production")
    return reg


@pytest.fixture()
def client(prod_registry) -> TestClient:
    with TestClient(
        create_app(registry=prod_registry, model_name="modelops_logistic_regression")
    ) as c:
        yield c


def _sample_row() -> dict:
    return {
        "age": 45, "income": 72000, "credit_score": 680,
        "loan_amount": 24000, "employment_years": 8, "num_defaults": 1, "has_collateral": 1,
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "time" in body


def test_ready_when_production_model_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_predict_happy_path_returns_label_and_probability(client):
    r = client.post("/predict", json=_sample_row())
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body and "probability" in body
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_422_on_missing_field(client):
    r = client.post("/predict", json={"age": 45, "income": 72000})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][0] == "body"
    assert r.json()["detail"][0]["loc"][1] == "credit_score"


def test_predict_422_on_out_of_range_value(client):
    row = _sample_row()
    row["credit_score"] = 999
    r = client.post("/predict", json=row)
    assert r.status_code == 422
    assert "credit_score" in str(r.json()["detail"])


def test_model_info_reports_version_metrics_and_served_latency(client):
    for _ in range(20):
        client.post("/predict", json=_sample_row())
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "modelops_logistic_regression"
    assert body["version"] == 1
    assert "roc_auc" in body["metrics"]
    assert set(body["served_latency_ms"]) >= {"n_requests", "avg", "p95", "p99"}
    assert body["served_latency_ms"]["n_requests"] >= 20


def test_not_ready_and_503_when_no_production_model(tmp_path):
    reg = ModelRegistry(tracking_uri=f"sqlite:///{tmp_path}/empty.db")
    app = create_app(registry=reg, model_name="modelops_logistic_regression", config_path=None)
    with TestClient(app) as c:
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json=_sample_row()).status_code == 503