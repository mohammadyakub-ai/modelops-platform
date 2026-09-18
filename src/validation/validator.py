"""Data validation for the ModelOps platform.

Validates raw data BEFORE any model sees it. Checks, in order (fail fast):
  1. File exists
  2. Schema — required columns present
  3. Row count >= min_rows
  4. Missing-value ratio per column <= max_missing_pct
  5. Value ranges for configured numeric columns
  6. Leakage — the target column must NOT be listed as a feature
  7. Index uniqueness for the id column

Every run writes a JSON report to the report_dir so downstream phases and
CI can consume the same machine-readable artifact.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

REPORT_PATTERN = "validation_latest.json"


class ColumnRange(BaseModel):
    min: float
    max: float


class ValidationConfig(BaseModel):
    min_rows: int = Field(default=100, ge=1)
    max_missing_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    column_ranges: Dict[str, ColumnRange] = Field(default_factory=dict)
    target: str
    features: List[str]
    id_column: Optional[str] = None


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str


class ValidationReport(BaseModel):
    file: str
    file_sha256: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    rows: int
    columns: List[str]
    passed: bool
    checks: List[CheckResult]
    errors: List[str] = Field(default_factory=list)


class ValidationError(Exception):
    """Raised when validation fails; carries the partial report for logging."""

    def __init__(self, message: str, report: Optional[ValidationReport] = None):
        super().__init__(message)
        self.report = report


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class DataValidator:
    def __init__(self, config: ValidationConfig):
        self.config = config

    def validate(self, filepath: str | Path, report_dir: str | Path | None = None) -> ValidationReport:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        file_hash = _sha256(path)
        errors: List[str] = []
        checks: List[CheckResult] = []
        rows = 0
        columns: List[str] = []

        # 1. Load
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 - surface a clear validation error
            report = self._draft(path, file_hash, rows, [], False, [], errors)
            raise ValidationError(f"Could not read {path}: {exc}", report) from exc

        rows = len(df)
        columns = list(df.columns)
        required = self.config.features + [self.config.target]

        # 2. Schema
        missing_cols = [c for c in required if c not in df.columns]
        checks.append(
            CheckResult(
                name="schema",
                passed=not missing_cols,
                detail=f"expected {required}; missing: {missing_cols or 'none'}",
            )
        )

        # 3. Row count
        checks.append(
            CheckResult(
                name="min_rows",
                passed=rows >= self.config.min_rows,
                detail=f"rows={rows}, min_rows={self.config.min_rows}",
            )
        )

        # 4. Missing values
        missing_detail = []
        for col in required:
            if col in df.columns:
                ratio = float(df[col].isna().mean())
                if ratio > self.config.max_missing_pct:
                    missing_detail.append(f"{col}={ratio:.1%}")
        checks.append(
            CheckResult(
                name="missing_values",
                passed=not missing_detail,
                detail=f"max_missing_pct={self.config.max_missing_pct}; " + (
                    "over threshold: " + ", ".join(missing_detail) if missing_detail else "ok"
                ),
            )
        )

        # 5. Value ranges
        range_detail = []
        for col, rng in self.config.column_ranges.items():
            if col in df.columns:
                col_min = float(df[col].min())
                col_max = float(df[col].max())
                if col_min < rng.min or col_max > rng.max:
                    range_detail.append(f"{col} observed=[{col_min},{col_max}] allowed=[{rng.min},{rng.max}]")
        checks.append(
            CheckResult(
                name="value_ranges",
                passed=not range_detail,
                detail="; ".join(range_detail) if range_detail else "all columns within allowed ranges",
            )
        )

        # 6. Leakage — target must not be a feature
        leak = self.config.target in self.config.features
        checks.append(
            CheckResult(
                name="leakage",
                passed=not leak,
                detail=f"target '{self.config.target}' in features: {leak}",
            )
        )

        # 7. Id uniqueness
        id_col = self.config.id_column
        if id_col and id_col in df.columns:
            dupes = int(df[id_col].duplicated().sum())
            checks.append(
                CheckResult(
                    name="id_unique",
                    passed=dupes == 0,
                    detail=f"duplicates in '{id_col}': {dupes}",
                )
            )

        passed = all(c.passed for c in checks)
        if not passed:
            errors = [c.detail for c in checks if not c.passed]

        report = ValidationReport(
            file=str(path),
            file_sha256=file_hash,
            rows=rows,
            columns=columns,
            passed=passed,
            checks=checks,
            errors=errors,
        )

        if report_dir is not None:
            self.write_report(report, report_dir)

        if not passed:
            raise ValidationError(
                f"Validation FAILED for {path}:\n" + "\n".join(f"  - {e}" for e in errors),
                report,
            )
        return report

    def write_report(self, report: ValidationReport, report_dir: str | Path) -> Path:
        out_dir = Path(report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / REPORT_PATTERN
        out.write_text(report.model_dump_json(indent=2))
        return out

    @staticmethod
    def _draft(
        file: str,
        file_hash: str,
        rows: int,
        columns: List[str],
        passed: bool,
        checks: List[CheckResult],
        errors: List[str],
    ) -> ValidationReport:
        return ValidationReport(
            file=file,
            file_sha256=file_hash,
            rows=rows,
            columns=columns,
            passed=passed,
            checks=checks,
            errors=errors,
        )


def validator_from_config(cfg: dict) -> DataValidator:
    """Build a DataValidator from the train_config.yaml validation section."""
    ranges = {
        col: ColumnRange(min=lo, max=hi)
        for col, (lo, hi) in cfg.get("column_ranges", {}).items()
    }
    vc = cfg.copy()
    vc["column_ranges"] = ranges
    return DataValidator(ValidationConfig(**vc))