#!/usr/bin/env python3
"""학습 런의 `metrics_test_*h.json`을 스캔해 마크다운 리더보드 초안 생성."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_METRIC_RE = re.compile(r"metrics_test_(\d+)h\.json$")


def main() -> None:
    ap = argparse.ArgumentParser(description="정확도 리더보드 (마크다운)")
    ap.add_argument(
        "--training-runs",
        type=Path,
        default=Path("artifacts/training_runs"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/leaderboard.md"),
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    base = args.training_runs if args.training_runs.is_absolute() else root / args.training_runs
    out = args.output if args.output.is_absolute() else root / args.output

    rows: list[tuple[str, str, dict]] = []
    if base.is_dir():
        for p in sorted(base.rglob("metrics_test_*h.json")):
            m = _METRIC_RE.search(p.name)
            if not m:
                continue
            h = m.group(1)
            run_name = p.parent.name
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            rows.append((run_name, h, data))

    lines = [
        "# Accuracy leaderboard (auto)",
        "",
        "| run | horizon_h | MAE | RMSE | daytime_MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for run_name, h, data in sorted(rows, key=lambda x: (x[1], x[0])):
        mae = data.get("MAE", "")
        rmse = data.get("RMSE", "")
        dtm = data.get("daytime_MAE", "")
        lines.append(f"| {run_name} | {h} | {mae} | {rmse} | {dtm} |")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[leaderboard] 저장: {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
