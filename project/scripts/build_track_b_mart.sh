#!/usr/bin/env bash
# Track B per-site mart enrich (저장소 루트에서 dataset/preprocessor 실행)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}/dataset/preprocessor"

MART="${1:-../../project/artifacts/feature_mart_per_site}"
OUT="${2:-../../project/artifacts/feature_mart_track_b_per_site}"

python run.py enrich-track-b \
  --input-mart-dir "${MART}" \
  --output-dir "${OUT}" \
  "${@:3}"

python run.py update-manifest
