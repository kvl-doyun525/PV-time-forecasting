#!/usr/bin/env bash
# DLinear만 이미 돌린 뒤 집계·리더보드만 다시 할 때
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUNS_GROUP="${RUNS_GROUP:-dlinear_seq_168}"

python3 src/report/aggregate_seeds.py \
  --model dlinear \
  --runs-dir artifacts/training_runs \
  --runs-group "${RUNS_GROUP}" \
  --horizons 24 48 72

python3 src/report/build_leaderboard.py \
  --runs-dir artifacts/training_runs \
  --output artifacts/leaderboard.md

echo "finalize_dlinear 완료 → artifacts/leaderboard.md"
