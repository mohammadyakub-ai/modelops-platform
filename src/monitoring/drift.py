"""PSI drift detection.

Population Stability Index (PSI) measures how much a feature's distribution has
moved between a reference window (what the model was trained on) and a current
window (what production is serving):

    PSI = Σ (actual_pct - expected_pct) * ln(actual_pct / expected_pct)

A per-feature PSI above the config threshold flags drift. Rule of thumb from
credit-risk practice:

    PSI < 0.1  : little/no shift
    0.1–0.25   : moderate — watch
    PSI > 0.25 : significant drift — retrain

The platform flags at `psi_threshold` (0.2) and raises a retraining alert.
The computation is binned (equal-width buckets from the reference distribution)
and `expected_pct` is floored at 0.001 so genuinely empty buckets can't produce
NaN/Infinite PSI. Every PSI in a report is computed, never estimated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

GATE_REPORT_DRIFT_DIR = Path("data/drift_reports")
DRIFT_REPORT_PATTERN = "drift_latest.json"

EPSILON = 0.001  # lower bound for a share, prevents log(0)


def expected_actual_share(
    expected_pct: np.ndarray, actual_pct: np.ndarray
) -> float:
    """PSI over already-share-encoded distributions (pure, testable)."""
    expected_pct = np.clip(np.asarray(expected_pct, dtype=float), EPSILON, None)
    actual_pct = np.clip(np.asarray(actual_pct, dtype=float), EPSILON, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def bucket_shares(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Share of `values` falling in each equal-width bucket defined by `edges`."""
    values = np.asarray(values, dtype=float)
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    return counts / total if total > 0 else np.zeros_like(counts, dtype=float)


def feature_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Equal-width PSI for one feature; reference defines the bucket edges."""
    ref = reference.dropna().astype(float)
    cur = current.dropna().astype(float)
    lo, hi = float(ref.min()), float(ref.max())
    if not np.isfinite(hi - lo) or hi - lo < 1e-9:
        return 0.0  # constant feature cannot drift
    edges = np.linspace(lo, hi, bins + 1)
    return expected_actual_share(bucket_shares(ref, edges), bucket_shares(cur, edges))


class FeatureDrift(BaseModel):
    psi: float
    flag: bool


class DriftReport(BaseModel):
    model: str
    reference_file: str
    current_file: str
    psi_threshold: float
    feature_psi: dict[str, FeatureDrift]
    alert: bool
    binned_at: str
    summary: str = ""


def compute_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    bins: int = 10,
    threshold: float = 0.2,
) -> dict[str, FeatureDrift]:
    out: dict[str, FeatureDrift] = {}
    for feature in features:
        psi = feature_psi(reference_df[feature], current_df[feature], bins=bins)
        out[feature] = FeatureDrift(psi=round(psi, 4), flag=psi > threshold)
    return out


def run_drift_check(cfg: dict, model_name: str = "modelops_logistic_regression", report_dir: str | Path | None = None) -> DriftReport:
    """Compute a drift report and persist it to data/drift_reports/."""
    from datetime import datetime, timezone

    ref_path = Path(cfg["monitoring"]["reference_data_path"])
    cur_path = Path(cfg["monitoring"]["current_data_path"])
    features = list(cfg["data"]["features"])
    threshold = cfg["monitoring"]["psi_threshold"]
    bins = cfg["monitoring"].get("psi_bins", 10)

    reference = pd.read_csv(ref_path)
    current = pd.read_csv(cur_path)
    feature_psi = compute_drift(reference, current, features, bins=bins, threshold=threshold)

    report = DriftReport(
        model=model_name,
        reference_file=ref_path.name,
        current_file=cur_path.name,
        psi_threshold=threshold,
        feature_psi=feature_psi,
        alert=any(f.flag for f in feature_psi.values()),
        binned_at=datetime.now(timezone.utc).isoformat(),
        summary=", ".join(f"{f}={d.psi:.3f}{' ⚠' if d.flag else ''}" for f, d in feature_psi.items()),
    )

    out_dir = Path(report_dir) if report_dir else GATE_REPORT_DRIFT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / DRIFT_REPORT_PATTERN).write_text(report.model_dump_json(indent=2))
    return report


def logged_outcome(report: DriftReport) -> str:
    """Turn a drift report into an operational line for logs/CI."""
    flagged = [f for f, d in report.feature_psi.items() if d.flag]
    action = "RETRAIN ALERT — retraining recommended" if report.alert else "OK — no drift"
    return f"[drift] {action} | flagged={flagged or 'none'} | {report.summary}"