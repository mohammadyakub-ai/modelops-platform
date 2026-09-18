"""Deterministic drift scenarios for the monitoring phase.

Writes two "current window" samples next to the reference data so a drift check
can be measured honestly:

  data/drift/current_same.csv     same distribution as reference (PSI ~ 0)
  data/drift/current_shifted.csv  income shifted ~1.5x -> income PSI > 0.2

Only `income` is manipulated (features are drawn independently), so the demo
report flags exactly the drifted feature and nothing else.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW = DATA_DIR / "raw" / "dataset_v1.csv"
OUT_DIR = DATA_DIR / "drift"

SEED = 42


def make_current_same(reference: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Same population, deterministic reshuffle — a genuine no-drift window."""
    return reference.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_current_shifted(reference: pd.DataFrame, income_shift: float = 1.5, seed: int = SEED) -> pd.DataFrame:
    """The economy moved: borrowers now earn ~1.5x. Only income changes."""
    rng = np.random.default_rng(seed)
    df = reference.copy()
    df["income"] = np.clip(
        df["income"] * income_shift + rng.normal(0, 8_000, len(df)),
        20_000,
        250_000,
    ).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    reference = pd.read_csv(RAW)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    same = make_current_same(reference, args.seed)
    shifted = make_current_shifted(reference, args.shift, args.seed)

    same_path = OUT_DIR / "current_same.csv"
    shifted_path = OUT_DIR / "current_shifted.csv"
    same.to_csv(same_path, index=False)
    shifted.to_csv(shifted_path, index=False)

    # quick PSI preview so drift is verified before it's claimed
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from src.monitoring.drift import feature_psi

    features = [c for c in ["age", "income", "credit_score", "loan_amount", "employment_years", "num_defaults", "has_collateral"]]
    psi_same = {f: feature_psi(reference[f], same[f]) for f in features}
    psi_shifted = {f: feature_psi(reference[f], shifted[f]) for f in features}
    print(f"Wrote {same_path} and {shifted_path}")
    print("max|PSI| same   :", round(max(map(abs, psi_same.values())), 4))
    print("max|PSI| shifted:", round(max(map(abs, psi_shifted.values())), 4))
    for f in features:
        mark = "  <-- drifted" if abs(psi_shifted[f]) > 0.2 else ""
        print(f"  psi({f:>16}) same={psi_same[f]:+.4f} shifted={psi_shifted[f]:+.4f}{mark}")


if __name__ == "__main__":
    main()