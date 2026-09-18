"""FastAPI inference service — serves the Production model from the registry.

Endpoints (the classic ML-service contract):
  GET  /health       process alive check (liveness)
  GET  /ready        model loaded and serving? (readiness, drives load balancer)
  GET  /model/info   model name / version / metrics / observed latency
  POST /predict      one-row inference with Pydantic validation and 422 on bad input

Design notes:
  * model is resolved once at startup from the registry Production stage via the
    flavor (sklearn/xgboost) loader so ``predict_proba`` stays available;
    /ready reports whether that happened.
  * a middleware records every served request's latency so the API's *own*
    measured p95 is observable (not just the gate's standalone benchmark).
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, create_model, Field

from src.config import load_config
from src.monitoring.monitor import Monitor
from src.registry.registry import ModelRegistry

logger = logging.getLogger("modelops.serving")

FEATURES = [
    "age",
    "income",
    "credit_score",
    "loan_amount",
    "employment_years",
    "num_defaults",
    "has_collateral",
]


def _build_predict_request(bounds: dict[str, list[float]]) -> type[BaseModel]:
    """Emit PredictRequest dynamically so feature bounds stay config-driven."""
    fields = {}
    for feature in FEATURES:
        lo, hi = bounds.get(feature, (None, None))
        constraint = {}
        if lo is not None:
            constraint["ge"] = lo
        if hi is not None:
            constraint["le"] = hi
        typ = int if feature in {"age", "credit_score", "num_defaults", "has_collateral"} else float
        fields[feature] = (typ, Field(**constraint))
    return create_model("PredictRequest", __module__=__name__, **fields)


def _p95(ms: list[float]) -> float | None:
    if not ms:
        return None
    return np.percentile(ms, 95)


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(
    registry: ModelRegistry | None = None,
    model_name: str | None = None,
    config_path: str = "configs/train_config.yaml",
    monitor: Monitor | None = None,
) -> FastAPI:
    cfg = load_config(config_path) if config_path else None
    features = list((cfg or {}).get("data", {}).get("features") or FEATURES)
    bounds = (
        (cfg or {}).get("serving", {}).get("feature_bounds")
        or (cfg or {}).get("validation", {}).get("column_ranges")
        or {}
    )
    PredictRequest = _build_predict_request(bounds)
    # FastAPI resolves the endpoint annotation through this module's namespace,
    # not the closure — stash the generated model so the ForwardRef can resolve.
    globals()["PredictRequest"] = PredictRequest
    model_name = model_name or (cfg or {}).get("serving", {}).get("model_name")
    history_len = (cfg or {}).get("serving", {}).get("latency_history", 5000)
    monitor = monitor or Monitor()

    state: dict[str, Any] = {
        "model": None,
        "info": None,
        "loaded_at": None,
        "started_at": time.monotonic(),
        "latencies": deque(maxlen=history_len),
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        reg = registry or ModelRegistry()
        if model_name:
            version = reg.latest_in_stage(model_name, "Production")
            model = reg.load_model(model_name, version=version["version"]) if version else None
            if model is not None:
                state["model"] = model
                state["info"] = {
                    "name": model_name,
                    "version": version["version"],
                    "stage": "Production",
                    "metrics": version.get("metrics", {}),
                    "features": features,
                }
                state["loaded_at"] = _stamp()
                monitor.set_model(model_name, version["version"])
            else:
                logger.warning("no Production model for %s — /ready will report 503", model_name)
        yield

    app = FastAPI(title="modelops-serving", version="0.4.0", lifespan=lifespan)

    @app.middleware("http")
    async def record_latency(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        state["latencies"].append(ms)
        logger.info(
            "%s %s -> %s in %.2fms",
            request.method,
            request.url.path,
            response.status_code,
            ms,
        )
        return response

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "modelops-serving", "time": _stamp()}

    @app.get("/ready")
    async def ready():
        if state["model"] is None:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return {"status": "ready", "model": state["info"]["name"]}

    @app.get("/model/info")
    async def model_info():
        lat = list(state["latencies"])
        info = dict(state["info"] or {}) | {
            "model": (state["info"] or {}).get("name"),
            "uptime_seconds": round(time.monotonic() - state["started_at"], 1),
            "loaded_at": state["loaded_at"],
        }
        if lat:
            info["served_latency_ms"] = {
                "n_requests": len(lat),
                "avg": round(statistics.fmean(lat), 3),
                "p95": round(_p95(lat), 3),
                "p99": round(np.percentile(lat, 99), 3),
            }
        return info

    @app.get("/metrics")
    async def metrics():
        monitor.sync_drift()  # expose the latest drift report as Prometheus gauges
        return Response(content=monitor.render(), media_type="text/plain; version=0.0.4")

    @app.post("/predict")
    async def predict(req: PredictRequest):
        if state["model"] is None:
            return JSONResponse({"error": "model unavailable"}, status_code=503)
        version = f"{state['info']['name']}@{state['info']['version']}"
        values = {f: getattr(req, f) for f in features}
        start = time.perf_counter()
        try:
            row = np.array([[values[f] for f in features]])
            model = state["model"]
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(row)[0, 1])
            else:
                prob = float(model.predict(row)[0])
            prediction = int(prob >= 0.5)
        except Exception as exc:  # noqa: BLE001 - surface structured 500 to client
            monitor.record_error(version, "exception")
            logger.exception("prediction failed")
            return JSONResponse({"error": f"prediction failed: {exc}"}, status_code=500)
        monitor.record_latency(time.perf_counter() - start)
        monitor.record_prediction(version, prediction, prob, values)
        return {"prediction": prediction, "probability": round(prob, 6)}

    return app


# Module-level app for `uvicorn src.serving.api:app`.
app = create_app()