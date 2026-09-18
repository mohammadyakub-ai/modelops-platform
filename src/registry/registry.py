"""Model registry — versioning, staging and rollback for trained models.

Assigns lifecycle stages exactly as the guide defines them:
  candidate -> staging -> production

MLflow only ships with natively-storable stages None/Staging/Production/
Archived — there is no "Candidate" bucket. To keep the guide's vocabulary
without fighting MLflow, "Candidate" is implemented as a version tag
(`modelops_stage=candidate`) that this registry translates on the way in and
out. Staging/Production stay native MLflow stages.

Rollback promotes the newest older version back into the requested stage,
demoting the current holder first so a stage never has two versions.
"""

from __future__ import annotations

import os
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.tracking.tracker import DEFAULT_TRACKING_URI

STAGE_TAG = "modelops_stage"
CANDIDATE = "Candidate"
NATIVE_STAGES = {"None", "Staging", "Production", "Archived"}
VALID_STAGES = NATIVE_STAGES | {CANDIDATE}


class ModelRegistry:
    def __init__(self, tracking_uri: str | None = None):
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        self.tracking_uri = tracking_uri or DEFAULT_TRACKING_URI
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(self.tracking_uri)

    def register(self, model_uri: str, name: str) -> dict[str, Any]:
        """Register a logged model (from a tracker run) as a new version."""
        if not model_uri:
            raise ValueError("model_uri is required to register a model")
        vr = mlflow.register_model(model_uri=model_uri, name=name)
        return {"name": vr.name, "version": int(vr.version)}

    @staticmethod
    def _effective_stage(version) -> str:
        native = version.current_stage
        if native == "None" and version.tags.get(STAGE_TAG) == "candidate":
            return CANDIDATE
        return native

    def versions(self, name: str):
        return sorted(
            self.client.search_model_versions(f"name='{name}'"),
            key=lambda v: int(v.version),
        )

    def list_versions(self, name: str) -> list[dict[str, Any]]:
        """List every version with its platform stage and the run's metrics."""
        out = []
        for v in self.versions(name):
            run = mlflow.get_run(v.run_id) if v.run_id else None
            out.append(
                {
                    "version": int(v.version),
                    "stage": self._effective_stage(v),
                    "run_id": v.run_id,
                    "metrics": dict(run.data.metrics) if run else {},
                }
            )
        return out

    def promote(self, name: str, version: int, stage: str) -> dict[str, Any]:
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'; valid: {sorted(VALID_STAGES)}")
        version = str(version)

        if stage == CANDIDATE:
            cur = self.client.get_model_version(name, version)
            if cur.current_stage != "None":
                self.client.transition_model_version_stage(name, version, "None")
            self.client.set_model_version_tag(name, version, STAGE_TAG, "candidate")
        else:
            self.client.transition_model_version_stage(name, version, stage)
            try:
                self.client.delete_model_version_tag(name, version, STAGE_TAG)
            except Exception:  # noqa: BLE001 - tag may not exist yet
                pass
        return {"name": name, "version": int(version), "stage": stage}

    def _holders(self, name: str, stage: str) -> list:
        vs = self.versions(name)
        if stage == CANDIDATE:
            return [v for v in vs if self._effective_stage(v) == stage]
        return [v for v in vs if v.current_stage == stage]

    def rollback(self, name: str, stage: str = "Production") -> dict[str, Any] | None:
        """Demote the current stage holder, promote the newest older version."""
        holders = self._holders(name, stage)
        if not holders:
            return None
        current = max(holders, key=lambda v: int(v.version))
        older = [v for v in self.versions(name)
                 if int(v.version) < int(current.version)
                 and v.current_stage != "Archived"]
        if not older:
            return None
        target = max(older, key=lambda v: int(v.version))

        if stage == CANDIDATE:
            self.client.delete_model_version_tag(
                name, str(current.version), STAGE_TAG
            )
            self.client.set_model_version_tag(
                name, str(target.version), STAGE_TAG, "candidate"
            )
        else:
            self.client.transition_model_version_stage(
                name, version=str(current.version), stage="None"
            )
            self.client.transition_model_version_stage(
                name, version=str(target.version), stage=stage
            )
        return {
            "name": name,
            "now_production": int(target.version),
            "demoted": int(current.version),
        }