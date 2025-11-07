#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

if [[ -z "${PORT:-}" ]]; then
  echo "PORT environment variable is required." >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

echo "[Render] Starting gunicorn on port ${PORT}"
exec gunicorn server:app \
  --workers "${WORKERS:-1}" \
  --threads "${THREADS:-2}" \
  --timeout "${TIMEOUT:-600}" \
  --bind "0.0.0.0:${PORT}" \
  --worker-class gthread \
  --access-logfile '-' \
  --error-logfile '-'


