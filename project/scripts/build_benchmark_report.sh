#!/usr/bin/env bash
# 벤치마크 리포트 생성 (추론 측정 + MD 작성)
# 사용: project/ 루트에서  bash scripts/build_benchmark_report.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash scripts/run_inference_benchmark.sh

python3 src/report/build_leaderboard.py \
  --runs-dir artifacts/training_runs \
  --output artifacts/leaderboard.md

python3 src/report/build_benchmark_report.py \
  --runs-dir artifacts/training_runs \
  --inference-json artifacts/inference_benchmark.json \
  --output artifacts/benchmark_report.md \
  --assets-dir artifacts/report_assets/graphs

echo "리포트: artifacts/benchmark_report.md"
