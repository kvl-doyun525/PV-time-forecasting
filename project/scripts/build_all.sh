#!/usr/bin/env bash
# scripts/build_all.sh
# 전체 Docker 이미지 빌드 (프로젝트 루트에서 실행)
# 사용법: bash scripts/build_all.sh [--only unified|time-llm]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ONLY=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --only) ONLY="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

build_image() {
  local name=$1
  local dockerfile=$2
  echo "========================================="
  echo " Building pv-benchmark/${name}:latest"
  echo "========================================="
  docker build -t "pv-benchmark/${name}:latest" -f "${dockerfile}" "$ROOT"
  echo "✓ pv-benchmark/${name}:latest built"
}

case "$ONLY" in
  unified)  build_image "unified"  "docker/unified/Dockerfile" ;;
  time-llm) build_image "time-llm" "docker/time_llm/Dockerfile" ;;
  "")
    echo "=== [1/2] unified (SegRNN + PatchTST + DLinear + LLaMA + Gemma) ==="
    build_image "unified" "docker/unified/Dockerfile"

    echo "=== [2/2] time-llm (transformers==4.31.0 고정) ==="
    build_image "time-llm" "docker/time_llm/Dockerfile"
    ;;
  *) echo "Unknown image: $ONLY"; exit 1 ;;
esac

echo ""
echo "========================================="
echo " 전체 이미지 빌드 완료"
echo "========================================="
docker images | grep pv-benchmark
