"""Package a registered model as a self-contained deploy snapshot.

Downloads the given (default: Production) model's artifacts from the tracking
server into deploy/model/ — the payload the serving image carries at build time
so the live service runs without external Postgres/S3. The snapshot is a plain
MLflow model dir (MLmodel + flavor files), so `mlflow.register_model` can
re-hydrate it into a registry at container startup (scripts/bootstrap_registry.py).

Usage:
  PYTHONPATH= .venv/bin/python scripts/bake_model.py --name modelops_logistic_regression
  PYTHONPATH= .venv/bin/python scripts/bake_model.py --name modelops_logistic_regression --version 1
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from src.registry.registry import ModelRegistry

ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = ROOT / "deploy" / "model"
MANIFEST = DEPLOY_DIR / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", type=int, default=None)
    args = parser.parse_args()

    registry = ModelRegistry()
    if args.version is None:
        prod = registry.latest_in_stage(args.name, "Production")
        if prod is None:
            raise SystemExit(f"no Production version of '{args.name}' — train and gate first")
        version = prod["version"]
    else:
        version = args.version

    vr = registry.client.get_model_version(args.name, str(version))
    source = vr.source
    if not source:
        raise SystemExit(f"version {args.name}@{version} has no source artifact URI")

    tmp = ROOT / "deploy" / ".snapshot_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    local = mlflow.artifacts.download_artifacts(artifact_uri=source, dst_path=str(tmp))
    local = Path(local)
    if DEPLOY_DIR.exists():
        shutil.rmtree(DEPLOY_DIR)
    DEPLOY_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local, DEPLOY_DIR)
    shutil.rmtree(tmp)

    manifest = {
        "name": args.name,
        "version": version,
        "stage": "Production" if args.version is None else "snapshot",
        "source_artifact_uri": source,
        "tracking_uri": registry.tracking_uri,
        "tracked_at": datetime.now(timezone.utc).isoformat(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"baked {args.name}@{version} -> {DEPLOY_DIR} ({len(list(DEPLOY_DIR.rglob('*')))} files)")


if __name__ == "__main__":
    main()