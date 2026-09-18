"""Airflow DAG — scheduled end-to-end retraining for modelops-platform.

Mirrors the CLI/CI contract and reuses the exact same step functions, so a model
that ships from Airflow has gone through byte-identical code to one shipped
from the command line:

  validate_and_prepare -> retrain_track_register -> gate_and_deploy

step_validate  : data validation gate (execution stops here on bad data)
step_train_track: trains, tracks each model in MLflow, registers them, stages
                  the best measured one (ROC-AUC) as Candidate
step_gate_deploy: regression gate -> PASS promotes Candidate -> Staging ->
                  Production; FAIL raises, leaving Production untouched

A FAIL verdict therefore marks the DAG run red with zero extra machinery —
orchestration status doubles as the deployment signal.

Run with the bundled one-node profile:
  docker compose --profile airflow up airflow   # webserver on :8080
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task

from src.config import load_config
from src.pipeline import DEFAULT_CONFIG_PATH, step_gate_deploy, step_train_track, step_validate
from src.tracking.tracker import MLflowTracker

default_args = {
    "owner": "modelops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}


@dag(
    dag_id="modelops_retraining",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    doc_md=__doc__,
)
def modelops_retraining():
    @task
    def validate_and_prepare() -> str:
        cfg = load_config(DEFAULT_CONFIG_PATH)
        step_validate(cfg)
        return DEFAULT_CONFIG_PATH

    @task
    def retrain_track_register(config_path: str) -> dict:
        cfg = load_config(config_path)
        tracker = MLflowTracker(experiment_name="modelops")
        stepped = step_train_track(cfg, tracker)
        return {"cand_name": stepped["cand_name"], "cand_version": stepped["cand_version"]}

    @task
    def gate_and_deploy(config_path: str, candidate: dict) -> str:
        if not candidate:
            raise RuntimeError("no candidate staged — nothing to gate")
        cfg = load_config(config_path)
        from src.registry.registry import ModelRegistry

        registry = ModelRegistry()
        step_gate_deploy(cfg, registry, candidate["cand_name"], int(candidate["cand_version"]))
        return f"deployed {candidate['cand_name']} v{candidate['cand_version']}"

    cfg = validate_and_prepare()
    gate_and_deploy(cfg, retrain_track_register(cfg))


modelops_retraining()