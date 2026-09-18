import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_sample_data import generate  # noqa: E402
from src.config import build_validation_config, load_config  # noqa: E402
from src.validation.validator import (  # noqa: E402
    DataValidator,
    ValidationConfig,
    ValidationError,
)


@pytest.fixture
def small_csv(tmp_path):
    df = generate(n_rows=200, seed=7)
    path = tmp_path / "small.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def default_validator():
    cfg = load_config()
    return DataValidator(build_validation_config(cfg))


def _validator_with(features, target, ranges=None, id_column=None, tmp_path=None):
    return DataValidator(
        ValidationConfig(
            target=target,
            features=features,
            id_column=id_column,
            min_rows=50,
            max_missing_pct=0.05,
            column_ranges=ranges or {},
        )
    )


def test_validator_passes_on_project_data(tmp_path):
    cfg = load_config()
    validator = DataValidator(build_validation_config(cfg))
    report = validator.validate(cfg["data"]["raw_path"], report_dir=tmp_path)

    assert report.passed is True
    assert report.rows == 1200
    assert report.file_sha256  # lineage hash is always present
    assert (tmp_path / "validation_latest.json").exists()


def test_validator_rejects_leakage_when_target_in_features(tmp_path, small_csv):
    validator = _validator_with(
        features=["credit_score", "default_risk", "income"],
        target="default_risk",
        tmp_path=tmp_path,
    )
    with pytest.raises(ValidationError) as exc:
        validator.validate(small_csv)
    assert "leakage" in str(exc.value).lower() or "feature" in str(exc.value).lower()


def test_validator_rejects_missing_required_column(tmp_path, small_csv):
    validator = _validator_with(
        features=["credit_score", "income", "does_not_exist"],
        target="default_risk",
        tmp_path=tmp_path,
    )
    with pytest.raises(ValidationError) as exc:
        validator.validate(small_csv)
    assert "does_not_exist" in str(exc.value)


def test_validator_rejects_values_out_of_range(tmp_path, small_csv):
    validator = DataValidator(
        ValidationConfig(
            target="default_risk",
            features=["credit_score"],
            id_column=None,
            min_rows=50,
            max_missing_pct=0.05,
            column_ranges={"credit_score": {"min": 700, "max": 850}},
        )
    )
    with pytest.raises(ValidationError) as exc:
        validator.validate(small_csv)
    assert "credit_score" in str(exc.value)


def test_validator_rejects_too_few_rows(tmp_path):
    df = generate(20, seed=7)
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    validator = DataValidator(
        ValidationConfig(
            target="default_risk",
            features=["credit_score", "income"],
            min_rows=100,
        )
    )
    with pytest.raises(ValidationError) as exc:
        validator.validate(path)
    assert "min_rows" in str(exc.value)


def test_validator_rejects_excessive_missing_values(tmp_path):
    df = generate(100, seed=7)
    df.loc[df.sample(60, random_state=1).index, "income"] = pd.NA
    path = tmp_path / "missing.csv"
    df.to_csv(path, index=False)
    validator = _validator_with(
        features=["credit_score", "income"],
        target="default_risk",
        tmp_path=tmp_path,
    )
    with pytest.raises(ValidationError) as exc:
        validator.validate(path)
    assert "income" in str(exc.value) or "missing" in str(exc.value)