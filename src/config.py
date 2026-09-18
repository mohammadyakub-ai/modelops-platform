"""Configuration loading for the ModelOps platform.

Single responsibility: turn configs/train_config.yaml into typed structures the
pipeline components consume. Splitting config from code is what makes the whole
pipeline reproducible and audit-able.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.validation.validator import ValidationConfig


def load_config(path: str | Path = "configs/train_config.yaml") -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def build_validation_config(cfg: dict[str, Any]) -> ValidationConfig:
    """Merge the data section (target/features/id) into the validation section."""
    data = cfg["data"]
    val = cfg.get("validation", {})
    merged = dict(val)
    merged["target"] = data["target"]
    merged["features"] = data["features"]
    merged["id_column"] = data.get("id_column")
    ranges = {
        col: {"min": lo, "max": hi}
        for col, (lo, hi) in merged.pop("column_ranges", {}).items()
    }
    return ValidationConfig(**merged, column_ranges=ranges)