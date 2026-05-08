#!/usr/bin/env bash
# NVIDIA 드라이버·컨테이너에서 GPU 노출 확인 (`pv_model_benchmark_execution.md` §5)
set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  echo "OK: nvidia-smi 사용 가능"
  exit 0
fi

echo "WARN: nvidia-smi 없음 — 호스트 드라이버 설치 여부 확인"
exit 1
