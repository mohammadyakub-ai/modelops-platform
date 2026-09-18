import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate_sample_data import generate  # noqa: E402
from src.registry.registry import ModelRegistry  # noqa: E402
from src.tracking.tracker import MLflowTracker  # noqa: E402
from src.training.trainer import train  # noqa: E402


def _train_config(data_path, model_dir, seed=1):
    return {
        "data": {
            "raw_path": data_path,
            "target": "default_risk",
            "id_column": "loan_id",
            "features": [
                "credit_score",
                "income",
                "num_defaults",
                "has_collateral",
            ],
        },
        "split": {"test_size": 0.2, "seed": seed},
        "reproducibility": {"seed": seed},
        "models": {"logistic_regression": {"max_iter": 200}},
        "output": {"model_dir": model_dir},
        "_config_path": "tests/fixture_config.yaml",
    }


@pytest.fixture
def registry_uri(tmp_path):
    return f"sqlite:///{tmp_path / 'registry.db'}"


def test_tracker_records_run_and_returns_model_uri(tmp_path):
    tracker = MLflowTracker(tracking_uri=f"sqlite:///{tmp_path / 'run.db'}")
    with tracker.run(run_name="tracker-test") as t:
        t.log_params({"a": 1, "b": "x"})
        t.log_metrics({"acc": 0.5})
        import mlflow

        from sklearn.linear_model import LogisticRegression

        uri = t.log_model(LogisticRegression(), name="probe_lr")
        assert uri.startswith("models:/")
        run_id = t.run_id
    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert dict(run.data.params)["a"] == "1"
    assert run.data.metrics["acc"] == 0.5


def test_registry_register_promote_and_list(tmp_path, registry_uri):
    tracker = MLflowTracker(tracking_uri=registry_uri)
    with tracker.run() as t:
        import mlflow

        from sklearn.linear_model import LogisticRegression

        mlflow.log_metric("acc", 0.9)
        uri = t.log_model(LogisticRegression(), name="reg_lr")

    reg = ModelRegistry(tracking_uri=registry_uri)
    reg.register(uri, name="reg_lr")
    version = reg.list_versions("reg_lr")[0]
    assert version["version"] == 1
    assert version["stage"] == "None"
    assert version["metrics"]["acc"] == 0.9

    reg.promote("reg_lr", version=1, stage="Staging")
    assert reg.list_versions("reg_lr")[0]["stage"] == "Staging"


def test_registry_candidate_stage_via_tag(tmp_path, registry_uri):
    tracker = MLflowTracker(tracking_uri=registry_uri)
    with tracker.run() as t:
        import mlflow

        from sklearn.linear_model import LogisticRegression

        uri = t.log_model(LogisticRegression(), name="cand_lr")

    reg = ModelRegistry(tracking_uri=registry_uri)
    reg.register(uri, name="cand_lr")
    reg.promote("cand_lr", version=1, stage="Candidate")
    assert reg.list_versions("cand_lr")[0]["stage"] == "Candidate"


def test_registry_rollback_promotes_previous_version(tmp_path, registry_uri):
    tracker = MLflowTracker(tracking_uri=registry_uri)
    uris = {}
    for name, acc in (("rl_lr", 0.8), ("rl_lr", 0.9)):
        with tracker.run() as t:
            import mlflow

            from sklearn.linear_model import LogisticRegression

            mlflow.log_metric("acc", acc)
            uris[acc] = t.log_model(LogisticRegression(), name=name)

    reg = ModelRegistry(tracking_uri=registry_uri)
    reg.register(uris[0.8], name="rl_lr")
    reg.register(uris[0.9], name="rl_lr")

    reg.promote("rl_lr", version=2, stage="Production")
    result = reg.rollback("rl_lr", stage="Production")
    assert result == {"name": "rl_lr", "now_production": 1, "demoted": 2}

    stages = {v["version"]: v["stage"] for v in reg.list_versions("rl_lr")}
    assert stages[1] == "Production"
    assert stages[2] == "None"


def test_registry_rejects_invalid_stage(registry_uri):
    reg = ModelRegistry(tracking_uri=registry_uri)
    with pytest.raises(ValueError):
        reg.promote("nope", version=1, stage="NotAStage")


def test_training_logs_one_run_per_model(tmp_path):
    data_path = tmp_path / "t.csv"
    generate(n_rows=150, seed=5).to_csv(data_path, index=False)

    tracker = MLflowTracker(tracking_uri=f"sqlite:///{tmp_path / 'train.db'}")
    result = train(_train_config(str(data_path), tmp_path / "m"), tracker=tracker)

    import mlflow

    lr = result["models"]["logistic_regression"]
    assert lr["model_uri"].startswith("models:/")
    assert lr["run_id"] is not None

    # the model's own run carries its own metrics + lineage (no cross-model bleed)
    run = mlflow.get_run(lr["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.metrics["roc_auc"] == pytest.approx(lr["metrics"]["roc_auc"])
    meta_hash = json.loads(Path(lr["metadata"]).read_text())["data_sha256"]
    assert run.data.params["data_sha256"] == meta_hash