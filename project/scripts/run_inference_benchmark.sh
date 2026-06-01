#!/usr/bin/env bash
# 학습 체크포인트 추론 지연 측정 (대표 run, seq_168)
# 사용: project/ 루트에서  bash scripts/run_inference_benchmark.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMPOSE=(docker compose -f docker/docker-compose.yml)
OUT="artifacts/inference_benchmark.json"
TMP_TS="artifacts/inference_benchmark_tslib.json"
TMP_TL="artifacts/inference_benchmark_timellm.json"

echo "=== TSLib 모델 (DLinear, SegRNN, PatchTST) ==="
"${COMPOSE[@]}" run --rm unified \
  python src/benchmark/benchmark_trained_inference.py \
    --representative \
    --models dlinear,segrnn,patchtst \
    --output-json "${TMP_TS}"

echo "=== Time-LLM ==="
"${COMPOSE[@]}" run --rm time-llm \
  python src/benchmark/benchmark_trained_inference.py \
    --representative \
    --models timellm \
    --output-json "${TMP_TL}"

python3 <<'PY'
import json
from pathlib import Path

root = Path(".")
parts = []
for name in ("artifacts/inference_benchmark_tslib.json", "artifacts/inference_benchmark_timellm.json"):
    p = root / name
    if p.is_file():
        parts.append(json.loads(p.read_text(encoding="utf-8")))

if not parts:
    raise SystemExit("벤치마크 결과 없음")

merged = dict(parts[0])
for part in parts[1:]:
    merged.setdefault("entries", []).extend(part.get("entries") or [])

out = root / "artifacts/inference_benchmark.json"
out.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"병합 저장: {out} ({len(merged.get('entries', []))} entries)")
PY

echo "완료."
