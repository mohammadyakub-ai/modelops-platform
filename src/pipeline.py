"""Phase 2 pipeline: load config -> validate data -> train -> track -> register.

Two entry points:
  - run():                Phase 1 behaviour, no MLflow.
  - run_with_tracking():  Phase 2 behaviour — every run is recorded in MLflow and
                          the trained models are registered, with the best
                          measured candidate staged as Candidate.

Staging the best model as Candidate (never Production) is deliberate: Production
is decided later by the regression gate (Phase 3).
"""

from __future__ import annotations

from src.config import load_config, build_validation_config
from src.registry.registry import ModelRegistry
from src.tracking.tracker import MLflowTracker
from src.training.trainer import train
from src.validation.validator import DataValidator


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


def run_with_tracking(
    config_path: str = "configs/train_config.yaml",
    tracking_uri: str | None = None,
) -> dict:
    cfg = load_config(config_path)
    cfg["_config_path"] = config_path
    _validate(cfg)

    tracker = MLflowTracker(tracking_uri=tracking_uri, experiment_name="modelops")
    result = train(cfg, tracker=tracker)

    registry = ModelRegistry(tracking_uri=tracker.tracking_uri)
    registered = {}
    for name, entry in result["models"].items():
        version = registry.register(entry["model_uri"], name=f"modelops_{name}")
        registered[name] = version["version"]
        print(f"[registry] registered {version['name']} v{version['version']}")

    best = _best_model(result)
    registry.promote(f"modelops_{best}", registered[best], "Candidate")
    print(
        f"[registry] best by roc_auc={result['models'][best]['metrics']['roc_auc']}: "
        f"{best} -> Candidate (v{registered[best]})"
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--track", action="store_true", help="track + register in MLflow")
    args = parser.parse_args()
    run_with_tracking() if args.track else run()