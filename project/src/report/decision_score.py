"""종합 의사결정 점수 (`pv_model_benchmark_execution.md` §13.3)."""

from __future__ import annotations

from typing import Any

WEIGHTS: dict[str, float] = {
    "accuracy": 0.50,
    "cpu_latency": 0.20,
    "memory": 0.10,
    "complexity": 0.20,
}


def _inv(x: float, eps: float = 1e-9) -> float:
    return 1.0 / (x + eps)


def compute_decision_score(
    metrics: dict[str, Any],
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """
    metrics 키 예시(모두 선택):
      - daytime_nRMSE 또는 nRMSE: 정확도(낮을수록 좋음 → 역수)
      - warm_p95_ms: 지연(낮을수록 좋음 → 역수)
      - peak_ram_mb: 메모리(낮을수록 좋음 → 역수)
      - complexity_0_10: 운영 복잡도(낮을수록 좋음 → (10-x)/10 가중)
    """
    w = weights or WEIGHTS
    acc = float(metrics.get("daytime_nRMSE") or metrics.get("nRMSE") or 1.0)
    lat = float(metrics.get("warm_p95_ms") or metrics.get("p95_ms") or 1.0)
    mem = float(metrics.get("peak_ram_mb") or metrics.get("peak_ram") or 1.0)
    comp = float(metrics.get("complexity_0_10", 5.0))

    score = 0.0
    score += w.get("accuracy", 0.0) * _inv(acc)
    score += w.get("cpu_latency", 0.0) * _inv(lat / 1000.0)
    score += w.get("memory", 0.0) * _inv(mem / 1024.0)
    score += w.get("complexity", 0.0) * ((10.0 - comp) / 10.0)
    return float(score)
