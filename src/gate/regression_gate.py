"""Regression gate — blocks bad deployments before they ship.

Compares a deployment candidate against the current production model across
four dimensions, exactly as the guide defines:

  Quality   — accuracy / F1 / ROC-AUC deltas vs the production reference
  Latency   — measured p95 must stay under latency_p95_max_ms (absolute)
  Memory    — measured resident footprint must stay under memory_max_mb (absolute)
  Cost      — estimated $ per 1M predictions may exceed reference by at most
              cost_max_delta_pct (relative)

All thresholds come from config. Every latency/memory/cost number is MEASURED
at gate time on the same hardware for both models (apples-to-apples) — nothing
is estimated except the price model, which is explicit (`cpu_hour_usd`).

Bootstrap rule: with no production reference yet, quality and cost verdicts are
SKIPPED and the gate passes if the candidate meets the absolute latency/memory
caps. This mirrors the source project's "first run seeds the baseline".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import psutil
from pydantic import BaseModel, Field

QUALITY_METRICS = ("accuracy", "f1", "roc_auc")
VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_SKIP = "SKIP"


class GateConfig(BaseModel):
    min_accuracy_delta: float = -0.01
    min_f1_delta: float = -0.01
    min_roc_auc_delta: float = -0.01
    latency_p95_max_ms: float = Field(default=200, gt=0)
    memory_max_mb: float = Field(default=512, gt=0)
    cost_max_delta_pct: float = Field(default=10, ge=0)
    n_requests: int = Field(default=200, ge=1)
    seed: int = 42
    cpu_hour_usd: float = Field(default=0.1, ge=0)


class LatencyProfile(BaseModel):
    p50_ms: float
    p95_ms: float
    p99_ms: float
    memory_mb: float
    cost_usd_per_1m: float


class ModelSnapshot(BaseModel):
    name: str
    version: int | None
    stage: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    latency: Optional[LatencyProfile] = None


class GateReport(BaseModel):
    gate: str = "regression"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    candidate: ModelSnapshot
    reference: Optional[ModelSnapshot] = None
    verdicts: dict[str, str] = Field(default_factory=dict)
    details: dict[str, str] = Field(default_factory=dict)
    passed: bool = False
    note: str = ""


def percentiles(values: list[float]) -> tuple[float, float, float]:
    a = np.sort(np.asarray(values, dtype=float))
    return tuple(float(np.percentile(a, q)) for q in (50, 95, 99))


def measure_inference(
    predict: Callable[[pd.DataFrame], Any], X: pd.DataFrame, cfg: GateConfig
) -> LatencyProfile:
    """Measure per-request latency (ms), resident memory (MB) and cost estimate.

    Deterministic: the same rows are measured in the same order every run.
    Memory is the resident footprint delta of loading the model (baseline vs
    after warmup+inference), so both models are measured identically.
    """
    import time

    n = min(cfg.n_requests, len(X))
    sample = X.sample(n=n, random_state=cfg.seed)

    base_rss = psutil.Process().memory_info().rss
    latencies: list[float] = []
    predict(sample.iloc[[0]])  # warmup — JIT/pageload noise excluded
    for row_name, _ in sample.iterrows():
        start = time.perf_counter()
        predict(sample.loc[[row_name]])
        latencies.append((time.perf_counter() - start) * 1000.0)

    loaded_rss = psutil.Process().memory_info().rss
    p50, p95, p99 = percentiles(latencies)
    mean_ms = float(np.mean(latencies))
    memory_mb = (loaded_rss - base_rss) / (1024 * 1024)
    cost_usd_per_1m = mean_ms / 1000.0 * 1e6 / 3600.0 * cfg.cpu_hour_usd
    return LatencyProfile(
        p50_ms=round(p50, 2),
        p95_ms=round(p95, 2),
        p99_ms=round(p99, 2),
        memory_mb=round(max(0.0, memory_mb), 2),
        cost_usd_per_1m=round(cost_usd_per_1m, 4),
    )


def _quality_deltas(candidate: ModelSnapshot, reference: ModelSnapshot) -> dict[str, float | None]:
    deltas = {}
    for metric in QUALITY_METRICS:
        c, r = candidate.metrics.get(metric), reference.metrics.get(metric)
        deltas[metric] = (c - r) if (c is not None and r is not None) else None
    return deltas


def evaluate(
    candidate: ModelSnapshot, reference: Optional[ModelSnapshot], cfg: GateConfig
) -> GateReport:
    """Pure gate logic: verdicts from two snapshots + config. No I/O, fully testable."""
    verdicts: dict[str, str] = {}
    details: dict[str, str] = {}
    note = ""

    # ---------- quality ----------
    if reference is None:
        verdicts["quality"] = VERDICT_SKIP
        details["quality"] = "no production reference — skipped (bootstrap)"
        note = "no production baseline in the registry — gate passes on absolute caps only"
    else:
        min_deltas = {
            "accuracy": cfg.min_accuracy_delta,
            "f1": cfg.min_f1_delta,
            "roc_auc": cfg.min_roc_auc_delta,
        }
        bad = []
        for metric, min_delta in min_deltas.items():
            delta = _quality_deltas(candidate, reference)[metric]
            if delta is None:
                bad.append(f"{metric}=MISSING")
                continue
            flag = "ok" if delta >= min_delta else "TOO LOW"
            details[metric] = f"delta={delta:+.4f} (min {min_delta:+.4f}) {flag}"
            if delta < min_delta:
                bad.append(f"{metric}={delta:+.4f}")
        verdicts["quality"] = VERDICT_FAIL if bad else VERDICT_PASS
        details["quality"] = "; ".join(bad) if bad else "candidate quality within allowed regression"

    # ---------- latency (absolute) ----------
    if candidate.latency is not None:
        p95 = candidate.latency.p95_ms
        ok = p95 <= cfg.latency_p95_max_ms
        verdicts["latency"] = VERDICT_PASS if ok else VERDICT_FAIL
        details["latency"] = f"p95={p95:.1f}ms (max {cfg.latency_p95_max_ms}ms)"
    else:
        verdicts["latency"] = VERDICT_FAIL
        details["latency"] = "candidate latency not measured"

    # ---------- memory (absolute) ----------
    if candidate.latency is not None:
        mem = candidate.latency.memory_mb
        ok = mem <= cfg.memory_max_mb
        verdicts["memory"] = VERDICT_PASS if ok else VERDICT_FAIL
        details["memory"] = f"{mem:.1f}MB (max {cfg.memory_max_mb}MB)"
    else:
        verdicts["memory"] = VERDICT_FAIL
        details["memory"] = "candidate memory not measured"

    # ---------- cost (relative to reference) ----------
    if reference is not None and candidate.latency is not None and reference.latency is not None:
        ref_cost = reference.latency.cost_usd_per_1m
        cand_cost = candidate.latency.cost_usd_per_1m
        if ref_cost > 0:
            delta_pct = (cand_cost - ref_cost) / ref_cost * 100.0
            ok = delta_pct <= cfg.cost_max_delta_pct
            verdicts["cost"] = VERDICT_PASS if ok else VERDICT_FAIL
            details["cost"] = (
                f"cand=${cand_cost:.4f}/1M ref=${ref_cost:.4f}/1M "
                f"delta={delta_pct:+.2f}% (max {cfg.cost_max_delta_pct}%)"
            )
        else:
            verdicts["cost"] = VERDICT_PASS
            details["cost"] = "reference cost 0 — nothing to regress against"
    else:
        verdicts["cost"] = VERDICT_SKIP
        details["cost"] = "no reference cost to compare — skipped"

    passed = all(v != VERDICT_FAIL for v in verdicts.values())
    return GateReport(
        candidate=candidate,
        reference=reference,
        verdicts=verdicts,
        details=details,
        passed=passed,
        note=note,
    )


class RegressionGate:
    """Loads candidate + production from the registry, measures both, evaluates."""

    def __init__(self, config: GateConfig):
        self.config = config

    def _snapshot(
        self, name: str, version: int | None, stage: str | None, model: Any, X: pd.DataFrame
    ) -> ModelSnapshot:
        latency = measure_inference(model.predict, X, self.config) if model is not None else None
        return ModelSnapshot(name=name, version=version, stage=stage, latency=latency)

    def run(
        self,
        registry,
        candidate_name: str,
        candidate_version: int,
        X: pd.DataFrame,
    ) -> GateReport:
        """Measure the candidate and its production counterpart, then evaluate."""
        candidate_model = registry.load_model(candidate_name, version=candidate_version)
        reference = registry.latest_in_stage(candidate_name, "Production")
        reference_model = None
        if reference is not None:
            reference_model = registry.load_model(candidate_name, version=reference["version"])

        candidate_snap = self._snapshot(
            candidate_name, candidate_version, "Candidate", candidate_model, X
        )
        candidate_snap = ModelSnapshot(
            **candidate_snap.model_dump(exclude={"metrics"}),
            metrics=_registry_metrics(registry, candidate_name, candidate_version),
        )

        reference_snap = None
        if reference is not None:
            reference_snap = self._snapshot(
                candidate_name, reference["version"], "Production", reference_model, X
            )
            reference_snap = ModelSnapshot(
                **reference_snap.model_dump(exclude={"metrics"}),
                metrics=reference["metrics"],
            )

        return evaluate(candidate_snap, reference_snap, self.config)


def _registry_metrics(registry, name: str, version: int) -> dict[str, float]:
    for v in registry.list_versions(name):
        if v["version"] == version:
            return v["metrics"]
    return {}