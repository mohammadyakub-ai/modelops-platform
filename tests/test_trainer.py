import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_sample_data import generate  # noqa: E402
from src.training.trainer import sha256_file, train  # noqa: E402


def _train_config(data_path, model_dir, test_size=0.2, seed=42, xgb_estimators=10):
    return {
        "data": {
            "raw_path": data_path,
            "target": "default_risk",
            "id_column": "loan_id",
            "features": [
                "age",
                "income",
                "credit_score",
                "loan_amount",
                "employment_years",
                "num_defaults",
                "has_collateral",
            ],
        },
        "split": {"test_size": test_size, "seed": seed},
        "reproducibility": {"seed": seed},
        "models": {
            "logistic_regression": {"max_iter": 500},
            "xgboost": {
                "n_estimators": xgb_estimators,
                "max_depth": 2,
                "learning_rate": 0.1,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        },
        "output": {"model_dir": model_dir},
        "_config_path": "tests/fixture_config.yaml",
    }


@pytest.fixture
def small_dataset(tmp_path):
    df = generate(n_rows=300, seed=11)
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


def test_trainer_writes_artifacts_and_metadata(tmp_path, small_dataset):
    model_dir = tmp_path / "models"
    result = train(_train_config(str(small_dataset), model_dir))

    assert {"logistic_regression", "xgboost"} <= set(result["models"])
    for name, entry in result["models"].items():
        artifact = Path(entry["artifact"])
        meta = json.loads(Path(entry["metadata"]).read_text())
        assert artifact.exists(), f"{name} artifact missing"
        assert meta["data_sha256"] == sha256_file(small_dataset)
        assert set(["accuracy", "f1", "roc_auc"]) <= set(meta["metrics"])
        assert meta["metrics"]["train_rows"] + meta["metrics"]["test_rows"] == 300
        assert "training_duration_seconds" in meta


def test_trainer_is_reproducible_with_same_seed(tmp_path, small_dataset):
    cfg = _train_config(str(small_dataset), tmp_path / "m1")
    r1 = train(cfg)
    r2 = train(_train_config(str(small_dataset), tmp_path / "m2"))

    for model in ("logistic_regression", "xgboost"):
        m1 = r1["models"][model]["metrics"]
        m2 = r2["models"][model]["metrics"]
        assert m1 == m2, f"{model} metrics differ across identical runs: {m1} vs {m2}"


def test_trainer_data_hash_tracks_data(tmp_path, small_dataset):
    before = sha256_file(small_dataset)
    cfg = _train_config(str(small_dataset), tmp_path / "m")
    result = train(cfg)
    assert result["data_sha256"] == before