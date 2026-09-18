"""Phase 1 pipeline: load config -> validate data -> train models.

Entry point for Week 1's end-to-end run. Later phases plug tracking, the
regression gate, and retraining into this same orchestration, so the shape of
the pipeline (config -> validate -> train -> report) stays stable.
"""

from __future__ import annotations

from src.config import load_config, build_validation_config
from src.training.trainer import train
from src.validation.validator import DataValidator


def run(config_path: str = "configs/train_config.yaml", require_validation: bool = True) -> dict:
    cfg = load_config(config_path)
    cfg["_config_path"] = config_path

    validator = DataValidator(build_validation_config(cfg))
    report = validator.validate(
        cfg["data"]["raw_path"], report_dir=cfg["output"]["report_dir"]
    )
    print(f"[validation] PASSED rows={report.rows} file_sha256={report.file_sha256[:12]}...")

    if not report.passed and require_validation:
        raise SystemExit("Aborting: validation did not pass.")

    result = train(cfg)
    print(f"\n[pipeline] done — data_sha256={result['data_sha256'][:12]}...")
    return result


if __name__ == "__main__":
    run()