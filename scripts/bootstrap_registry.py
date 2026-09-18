"""Register the bundled deploy snapshot if the registry has no Production model.

Runs at serving-container startup (docker/entrypoint.serving.sh). When the
service is told to read its model from a live tracking server (MLFLOW_TRACKING_URI
set, e.g. docker-compose) the check finds Production there and does nothing.
When no server is reachable (self-contained live demo / Render), the bundled
deploy/model snapshot is re-hydrated into a local sqlite registry and promoted,
so the API serves exactly the packaged production model.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DEPLOY_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT))

import mlflow

from src.registry.registry import ModelRegistry
from src.tracking.tracker import env_tracking_uri

DEPLOY = ROOT / "deploy" / "model"
MANIFEST = DEPLOY / "manifest.json"


def main() -> None:
    if not MANIFEST.exists():
        print("[bootstrap] no bundled snapshot — expecting external registry")
        return

    manifest = json.loads(MANIFEST.read_text())
    name = manifest["name"]

    reg = ModelRegistry(tracking_uri=env_tracking_uri())
    if reg.latest_in_stage(name, "Production") is not None:
        print(f"[bootstrap] Production {name} already present — skipping bundling")
        return

    print(f"[bootstrap] no Production — registering bundled {name}@{manifest['version']}")
    vr = mlflow.register_model(model_uri=str(DEPLOY), name=name)
    reg.promote(name, int(vr.version), "Production")
    print(f"[bootstrap] promoted {name}@v{int(vr.version)} to Production")


if __name__ == "__main__":
    main()