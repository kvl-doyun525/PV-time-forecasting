#!/usr/bin/env bash
# LLaMA 3.2 1B LoRA — src/train/train_llama_lora.py 가 있을 때만 실행
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGET="src/train/train_llama_lora.py"
if [[ ! -f "${TARGET}" ]]; then
  echo "ERROR: ${TARGET} 없음 — LoRA 학습 모듈을 먼저 복원하세요." >&2
  exit 1
fi

COMPOSE=(docker compose -f docker/docker-compose.yml)
RUNS_GROUP="${RUNS_GROUP:-llama_lora_seq_168}"
OUT_BASE="artifacts/training_runs/${RUNS_GROUP}"
MODEL_PATH="${LLAMA_MODEL_PATH:-/models/meta-llama/Llama-3.2-1B-Instruct}"

mkdir -p "${OUT_BASE}"
for S in 42 123 2024; do
  out_dir="${OUT_BASE}/h24_seed_${S}"
  echo "=== LLaMA LoRA seed=${S} ==="
  "${COMPOSE[@]}" run --rm unified \
    python "${TARGET}" \
      --model-name-or-path "${MODEL_PATH}" \
      --horizon 24 \
      --seed "${S}" \
      --output-dir "${out_dir}"
done

python3 src/report/aggregate_seeds.py \
  --model llama_lora \
  --runs-dir artifacts/training_runs \
  --runs-group "${RUNS_GROUP}" \
  --horizons 24

python3 src/report/build_leaderboard.py \
  --runs-dir artifacts/training_runs \
  --output artifacts/leaderboard.md

echo "완료."
