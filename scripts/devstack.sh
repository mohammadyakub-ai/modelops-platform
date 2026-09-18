#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT" && pwd)"
TOOLS="${ROOT}/../tools"
PGBIN="${TOOLS}/pg-prefix/usr/lib/postgresql/14/bin"
PGDATA="${TOOLS}/pgdata"
PGPORT="5432"
MINIO_BIN="${TOOLS}/minio"
MINIO_DATA="${TOOLS}/minio-data"
MINIO_PORT="9000"
BUCKET="mlflow-artifacts"
MLFLOW_BIN="${REPO_ROOT}/.venv/bin/mlflow"
MLFLOW_PORT="5000"
BACKEND_URI="postgresql+psycopg2://mlflow@127.0.0.1:${PGPORT}/mlflow"
S3_ROOT="s3://${BUCKET}"

export MLFLOW_S3_ENDPOINT_URL="http://127.0.0.1:${MINIO_PORT}"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin"
export AWS_DEFAULT_REGION="us-east-1"
export MLFLOW_DISABLE_AGENT_HINT="1"

ensure_postgres() {
  if ! pg_isready_port; then
    mkdir -p "${PGDATA}"
    "${PGBIN}/initdb" -D "${PGDATA}" -U mlflow --encoding=UTF8 --auth=trust >/dev/null 2>&1 || true
    mkdir -p /tmp/pgsock
    "${PGBIN}/pg_ctl" -D "${PGDATA}" -l "${TOOLS}/logs/postgres.log" \
      -o "-p ${PGPORT} -c listen_addresses=127.0.0.1 -c unix_socket_directories=/tmp/pgsock" start >/dev/null
  fi
  "${PGBIN}/createdb" -h 127.0.0.1 -p "${PGPORT}" -U mlflow mlflow 2>/dev/null || true
}

pg_isready_port() {
  "${PGBIN}/pg_isready" -h 127.0.0.1 -p "${PGPORT}" -U mlflow >/dev/null 2>&1
}

ensure_minio() {
  if ! port_listening "${MINIO_PORT}"; then
    mkdir -p "${MINIO_DATA}"
    nohup "${MINIO_BIN}" server "${MINIO_DATA}" \
      --address ":${MINIO_PORT}" --console-address ":9001" \
      >"${TOOLS}/logs/minio.log" 2>&1 &
  fi
  for _ in $(seq 1 20); do port_listening "${MINIO_PORT}" && break; sleep 1; done
  "${TOOLS}/mc" alias set local "http://127.0.0.1:${MINIO_PORT}" minioadmin minioadmin >/dev/null 2>&1
  "${TOOLS}/mc" mb --ignore-existing "local/${BUCKET}" >/dev/null
}

ensure_mlflow() {
  if ! port_listening "${MLFLOW_PORT}"; then
    setsid env MLFLOW_S3_ENDPOINT_URL="$MLFLOW_S3_ENDPOINT_URL" \
      AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
      AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
      AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
      nohup "${MLFLOW_BIN}" server \
      --backend-store-uri "${BACKEND_URI}" \
      --default-artifact-root "${S3_ROOT}" \
      --host 127.0.0.1 --port "${MLFLOW_PORT}" \
      >"${TOOLS}/logs/mlflow.log" 2>&1 < /dev/null &
  fi
  for _ in $(seq 1 30); do port_listening "${MLFLOW_PORT}" && break; sleep 1; done
}

port_listening() {
  (command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ":${1} ") || \
    (command -v nc >/dev/null && nc -z 127.0.0.1 "${1}" >/dev/null 2>&1)
}

case "${1:-up}" in
  up)
    mkdir -p "${TOOLS}/logs"
    ensure_postgres
    ensure_minio
    ensure_mlflow
    echo "stack up: postgres :${PGPORT} | minio s3 :${MINIO_PORT} (bucket ${BUCKET}) | mlflow :${MLFLOW_PORT}"
    echo "tracking -> ${BACKEND_URI}   artifacts -> ${S3_ROOT}"
    ;;
  env)
    echo "export MLFLOW_TRACKING_URI=http://127.0.0.1:${MLFLOW_PORT}"
    echo "export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:${MINIO_PORT}"
    echo "export AWS_ACCESS_KEY_ID=minioadmin"
    echo "export AWS_SECRET_ACCESS_KEY=minioadmin"
    echo "export AWS_DEFAULT_REGION=us-east-1"
    echo "export MLFLOW_DISABLE_AGENT_HINT=1"
    ;;
  status)
    echo "postgres: $(pg_isready_port && echo up || echo down)"
    echo "minio:    $(port_listening ${MINIO_PORT} && echo up || echo down)"
    echo "mlflow:   $(port_listening ${MLFLOW_PORT} && echo up || echo down)"
    ;;
  *) echo "usage: $0 {up|env|status}" >&2; exit 2 ;;
esac