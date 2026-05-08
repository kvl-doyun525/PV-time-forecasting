#!/usr/bin/env bash
# TSLib(Time-Series-Library)를 vendor/TSLib 에 클론한다. (`pv_model_benchmark_execution.md` §5)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/vendor/TSLib"
REPO="${TSLIB_GIT_URL:-https://github.com/thuml/Time-Series-Library.git}"

mkdir -p "${ROOT}/vendor"

if [[ -d "${DEST}/.git" ]] || [[ -f "${DEST}/run.py" ]]; then
  echo "[setup_vendor] 이미 존재: ${DEST}"
  exit 0
fi

echo "[setup_vendor] cloning → ${DEST}"
git clone --depth 1 "${REPO}" "${DEST}"
echo "[setup_vendor] 완료"
