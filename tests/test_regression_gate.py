import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gate.regression_gate import (  # noqa: E402
    GateConfig,
    GateReport,
    LatencyProfile,
    ModelSnapshot,
    evaluate,
    measure_inference,
)


def _cfg(**overrides) -> GateConfig:
    defaults = dict(
        min_accuracy_delta=-0.01,
        min_f1_delta=-0.01,
        min_roc_auc_delta=-0.01,
        latency_p95_max_ms=200,
        memory_max_mb=512,
        cost_max_delta_pct=10,
        n_requests=20,
        seed=42,
        cpu_hour_usd=0.1,
    )
    defaults.update(overrides)
    return GateConfig(**defaults)


def _snapshot(
    name="m",
    version=1,
    accuracy=0.8,
    f1=0.8,
    roc_auc=0.85,
    latency=None,
) -> ModelSnapshot:
    if latency is None:
        latency = LatencyProfile(p50_ms=1.0, p95_ms=2.0, p99_ms=3.0, memory_mb=10.0, cost_usd_per_1m=0.01)
    return ModelSnapshot(
        name=name,
        version=version,
        stage="Production",
        metrics={"accuracy": accuracy, "f1": f1, "roc_auc": roc_auc},
        latency=latency,
    )


def test_gate_passes_when_candidate_meets_all_thresholds():
    cand = _snapshot(version=2, accuracy=0.82, f1=0.81, roc_auc=0.87)
    ref = _snapshot(version=1, accuracy=0.80, f1=0.80, roc_auc=0.85,
                    latency=LatencyProfile(p50_ms=1.0, p95_ms=2.0, p99_ms=3.0, memory_mb=10.0, cost_usd_per_1m=0.01))
    report = evaluate(cand, ref, _cfg())
    assert report.passed is True
    assert report.verdicts == {"quality": "PASS", "latency": "PASS", "memory": "PASS", "cost": "PASS"}


def test_gate_fails_on_quality_regression():
    cand = _snapshot(version=2, accuracy=0.77, f1=0.79, roc_auc=0.84)
    ref = _snapshot(version=1)  # 0.80 / 0.80 / 0.85
    report = evaluate(cand, ref, _cfg())
    assert report.passed is False
    assert report.verdicts["quality"] == "FAIL"
    assert "accuracy=-0.0300" in report.details["quality"]


def test_gate_fails_when_p95_latency_over_cap():
    slow = LatencyProfile(p50_ms=100.0, p95_ms=250.0, p99_ms=400.0, memory_mb=10.0, cost_usd_per_1m=0.5)
    cand = _snapshot(version=2, accuracy=0.85, f1=0.85, roc_auc=0.90, latency=slow)
    ref = _snapshot()
    report = evaluate(cand, ref, _cfg(latency_p95_max_ms=200))
    assert report.passed is False
    assert report.verdicts["latency"] == "FAIL"


def test_gate_fails_when_memory_over_cap():
    big = LatencyProfile(p50_ms=1.0, p95_ms=2.0, p99_ms=3.0, memory_mb=600.0, cost_usd_per_1m=0.01)
    cand = _snapshot(version=2, latency=big)
    ref = _snapshot()
    report = evaluate(cand, ref, _cfg(memory_max_mb=512))
    assert report.passed is False
    assert report.verdicts["memory"] == "FAIL"


def test_gate_fails_when_cost_delta_over_pct():
    ref = _snapshot(latency=LatencyProfile(p50_ms=1, p95_ms=2, p99_ms=3, memory_mb=10, cost_usd_per_1m=0.01))
    costly = LatencyProfile(p50_ms=1, p95_ms=2, p99_ms=3, memory_mb=10, cost_usd_per_1m=0.02)  # +100%
    cand = _snapshot(version=2, latency=costly)
    report = evaluate(cand, ref, _cfg(cost_max_delta_pct=10))
    assert report.passed is False
    assert report.verdicts["cost"] == "FAIL"


def test_gate_bootstraps_without_production_reference():
    cand = _snapshot(version=1)
    report = evaluate(cand, None, _cfg())
    assert report.passed is True
    assert report.verdicts["quality"] == "SKIP"
    assert report.verdicts["cost"] == "SKIP"
    assert report.note != ""


def test_gate_bootstrap_still_fails_absolute_caps():
    slow = LatencyProfile(p50_ms=1, p95_ms=900, p99_ms=1000, memory_mb=10, cost_usd_per_1m=0.1)
    cand = _snapshot(version=1, latency=slow)
    report = evaluate(cand, None, _cfg(latency_p95_max_ms=200))
    assert report.passed is False
    assert report.verdicts["latency"] == "FAIL"


def test_measure_inference_returns_profile(tmp_path):
    from sklearn.linear_model import LogisticRegression

    import numpy as np

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 4)), columns=list("abcd"))

    model = LogisticRegression()
    model.fit(X, (X["a"] > 0).astype(int))
    profile = measure_inference(model.predict, X, _cfg(n_requests=20))
    vals = profile.model_dump()
    assert profile.p95_ms >= 0
    assert profile.memory_mb >= 0
    assert all(isinstance(v, (int, float)) for v in vals.values())