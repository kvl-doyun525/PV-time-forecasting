#!/usr/bin/env bash
# Seasonal Naive + Persistence (unified 컨테이너)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
COMPOSE=(docker compose -f docker/docker-compose.yml)
MART="${FEATURE_MART:-artifacts/feature_mart_per_site}"

for method in seasonal persistence; do
  out="artifacts/training_runs/${method}"
  echo "=== baseline ${method} ==="
  "${COMPOSE[@]}" run --rm unified \
    python src/train/baseline_seasonal_naive.py \
      --feature-mart "${MART}" \
      --method "${method}" \
      --horizons 24 48 72 \
      --output-dir "${out}"
done

python3 src/report/build_leaderboard.py \
  --runs-dir artifacts/training_runs \
  --output artifacts/leaderboard.md

echo "완료."
