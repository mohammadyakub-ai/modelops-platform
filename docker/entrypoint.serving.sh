#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] boostrapping model registry (bundles deploy snapshot when no external server)"
python /app/scripts/bootstrap_registry.py

echo "[entrypoint] starting uvicorn"
exec "$@"