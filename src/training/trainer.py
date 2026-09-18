"""Training module for the ModelOps platform.

Reproducible training:
  - reads everything from configs/train_config.yaml
  - seeds random/numpy/torch/xgboost so the same config + data == same result
  - hashes the training file; the hash travels with the artifact as lineage
  - trains an sklearn baseline + XGBoost on the same split
  - saves model artifact + rich metadata JSON (params, metrics, data hash, duration)
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


class TrainConfigError(Exception):
    pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def make_models(model_configs: dict[str, dict[str, Any]], seed: int) -> dict[str, Any]:
    """Build the model suite from config. Each model exposes fit/predict_proba."""
    models: dict[str, Any] = {}

    if "logistic_regression" in model_configs:
        params = dict(model_configs["logistic_regression"])
        params.setdefault("random_state", seed)
        params.setdefault("max_iter", 1000)
        models["logistic_regression"] = LogisticRegression(**params)

    if "xgboost" in model_configs:
        params = dict(model_configs["xgboost"])
        params.setdefault("random_state", seed)
        params = {k: v for k, v in params.items() if k != "eval_metric"}
        models["xgboost"] = XGBClassifier(**params)

    if not models:
        raise TrainConfigError("No models configured under 'models' in train_config.yaml")

    return models


def train(cfg: dict[str, Any], tracker=None) -> dict[str, Any]:
    """Run validation-resolved training end to end. Returns a result summary.

    If `tracker` (an MLflowTracker) is passed, every model run is also recorded
    in MLflow: params, metrics, data lineage and the logged model artifact.
    """
    data_cfg = cfg["data"]
    split_cfg = cfg.get("split", {})
    seed = cfg.get("reproducibility", {}).get("seed", 42)
    model_configs = cfg.get("models", {})
    output = cfg.get("output", {})

    data_path = Path(data_cfg["raw_path"])
    target = data_cfg["target"]
    features = list(data_cfg["features"])
    model_dir = Path(output.get("model_dir", "artifacts/models"))
    model_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise TrainConfigError(f"Data file missing: {data_path}")

    data_hash = sha256_file(data_path)
    set_seed(seed)

    df = pd.read_csv(data_path)
    X = df[features]
    y = df[target].astype(int)

    test_size = split_cfg.get("test_size", 0.2)
    split_seed = split_cfg.get("seed", seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split_seed, stratify=y
    )

    models = make_models(model_configs, seed)
    results = {}

    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        duration = time.perf_counter() - start

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        }
        train_rows = int(len(X_train))
        test_rows = int(len(X_test))
        metrics["train_rows"] = train_rows
        metrics["test_rows"] = test_rows

        artifact_path = model_dir / f"{name}_v1.joblib"
        joblib.dump(model, artifact_path)

        metadata = {
            "model": name,
            "version": "v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_duration_seconds": round(duration, 3),
            "data_file": str(data_path),
            "data_sha256": data_hash,
            "features": features,
            "target": target,
            "params": model.get_params(),
            "metrics": metrics,
            "seed": seed,
            "config_file": cfg.get("_config_path"),
        }
        meta_path = artifact_path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))

        model_uri = None
        run_id = None
        if tracker is not None:
            with tracker.run(run_name=name) as t:
                t.log_data(data_file=data_path, data_sha256=data_hash)
                t.log_params(
                    {
                        "test_size": split_cfg.get("test_size", 0.2),
                        "split_seed": split_cfg.get("seed", seed),
                        "n_features": len(features),
                        **{k: v for k, v in model.get_params().items()},
                    }
                )
                t.log_metrics(
                    {
                        "accuracy": metrics["accuracy"],
                        "f1": metrics["f1"],
                        "roc_auc": metrics["roc_auc"],
                        "training_duration_seconds": duration,
                    }
                )
                model_uri = t.log_model(model, name=name)
                run_id = t.run_id

        results[name] = {
            "metrics": metrics,
            "artifact": str(artifact_path),
            "metadata": str(meta_path),
            "model_uri": model_uri,
            "run_id": run_id,
        }
        print(
            f"[train] {name:<20} acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
            f"auc={metrics['roc_auc']:.4f} in {duration:.2f}s -> {artifact_path}"
        )

    return {
        "data_sha256": data_hash,
        "train_rows": results[name]["metrics"]["train_rows"],
        "test_rows": results[name]["metrics"]["test_rows"],
        "split": {"test_size": test_size, "seed": split_seed},
        "models": results,
    }


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    train(cfg)