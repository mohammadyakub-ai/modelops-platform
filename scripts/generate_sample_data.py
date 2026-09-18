"""Generate the versioned raw dataset for modelops-platform.

Why a generator + committed CSV?
- The CSV in data/raw/ is the gold, committed artifact. It hashes it exactly; if
  the generator ever changes, the CSV stays stable unless deliberately re-committed.
- Generation is deterministic (seed=42), so the data is fully reproducible.

Target (default_risk) is a KNOWN function of the features, so later phases can
measure model quality honestly and build reliable drift scenarios on top of the
true distributions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

COLUMNS = [
    "loan_id",
    "age",
    "income",
    "credit_score",
    "loan_amount",
    "employment_years",
    "num_defaults",
    "has_collateral",
    "default_risk",
]


def make_default_risk(df: pd.DataFrame) -> pd.Series:
    """Ground-truth score: higher credit/income/longer employment = lower risk."""
    score = (
        df["credit_score"] / 850.0 * 0.5
        + np.log1p(df["income"] / 1000.0) / 6.0 * 0.25
        + np.clip(df["employment_years"] / 40.0, 0, 1) * 0.15
        + df["has_collateral"] * 0.10
    )
    score -= df["num_defaults"] * 0.15
    score -= np.clip(df["loan_amount"] / 50000.0, 0, 1) * 0.10
    prob = 1.0 / (1.0 + np.exp(-6.0 * (0.22 - score)))
    return prob


def generate(n_rows: int = 1200, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "loan_id": np.arange(n_rows),
            "age": rng.integers(21, 71, n_rows),
            "income": np.clip(rng.normal(70_000, 35_000, n_rows), 20_000, 250_000).astype(int),
            "credit_score": np.clip(rng.normal(650, 90, n_rows), 320, 850).astype(int),
            "loan_amount": np.clip(rng.normal(25_000, 15_000, n_rows), 1_000, 80_000).astype(int),
            "employment_years": np.clip(rng.normal(12, 9, n_rows), 0, 42).astype(int),
            "num_defaults": rng.integers(0, 6, n_rows),
            "has_collateral": rng.integers(0, 2, n_rows),
        }
    )
    prob = make_default_risk(df)
    df["default_risk"] = (rng.random(n_rows) < prob).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=str, default="dataset_v1.csv")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / args.out
    df = generate(args.rows, args.seed)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols -> {out}")
    print(f"Target balance: {df['default_risk'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()