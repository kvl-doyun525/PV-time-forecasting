#!/usr/bin/env python3
"""
`metrics_test_*h.json` 및 CPU 벤치 JSON을 한 파일로 모은다.

`pv_model_benchmark_execution.md` §12.5 — `cpu_benchmark_report.json` 초안.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_METRIC_RE = re.compile(r"metrics_test_(\d+)h\.json$")


def _load_json(p: Path) -> Any:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="벤치·지표 JSON 합산")
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/training_runs"),
        help="학습 런 루트 (metrics_test_*h.json 탐색)",
    )
    ap.add_argument(
        "--cpu-extra",
        type=Path,
        nargs="*",
        default=(),
        help="추가 CPU 벤치 JSON 경로 (예: artifacts/cpu_benchmark_last.json)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/cpu_benchmark_report.json"),
        help="합산 출력 JSON",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir if args.input_dir.is_absolute() else root / args.input_dir
    out_path = args.output if args.output.is_absolute() else root / args.output

    metrics_index: dict[str, list[dict[str, Any]]] = {}
    if input_dir.is_dir():
        for p in sorted(input_dir.rglob("metrics_test_*h.json")):
            m = _METRIC_RE.search(p.name)
            if not m:
                continue
            h = m.group(1)
            rel = str(p.relative_to(input_dir))
            metrics_index.setdefault(h, []).append(
                {"path": rel, "metrics": _load_json(p)}
            )

    cpu_blocks: list[dict[str, Any]] = []
    extra_paths = list(args.cpu_extra)
    default_last = root / "artifacts" / "cpu_benchmark_last.json"
    if default_last.is_file():
        extra_paths.append(default_last)
    for ep in extra_paths:
        p = ep if ep.is_absolute() else root / ep
        if p.is_file():
            cpu_blocks.append({"path": str(p), "data": _load_json(p)})

    payload = {
        "benchmark_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics_by_horizon": metrics_index,
        "cpu_benchmark_files": cpu_blocks,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[aggregate_results] 저장: {out_path}")


if __name__ == "__main__":
    main()
