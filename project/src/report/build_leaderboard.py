#!/usr/bin/env python3
"""
`training_runs/**/summary.json` 과 주요 `metrics_test_*h.json`을 읽어
`artifacts/leaderboard.md` 한 페이지로 요약한다.

(에이전트 기록의 `build_leaderboard.py` — `build_accuracy_leaderboard.py` 와 병행 가능)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="요약 리더보드 MD")
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/training_runs"),
        help="학습 루트 (summary.json 재귀 탐색)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/leaderboard.md"),
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    base = args.runs_dir if args.runs_dir.is_absolute() else root / args.runs_dir
    out = args.output if args.output.is_absolute() else root / args.output

    lines = [
        "# Benchmark leaderboard",
        "",
        "## summary.json (per experiment group)",
        "",
        "| group | model | horizon | MAE_mean | MAE_std | daytime_MAE_mean |",
        "|---|---|:---:|---:|---:|---:|",
    ]

    if base.is_dir():
        for summ in sorted(base.rglob("summary.json")):
            try:
                data = json.loads(summ.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            model = data.get("model", "")
            group = str(summ.parent.relative_to(base))
            for hk, block in (data.get("horizons") or {}).items():
                lines.append(
                    f"| {group} | {model} | {hk} | "
                    f"{block.get('MAE_mean', '')} | {block.get('MAE_std', '')} | "
                    f"{block.get('daytime_MAE_mean', '')} |"
                )

    lines.extend(["", "## Raw metrics (recent leaves)", ""])
    if base.is_dir():
        for mpath in sorted(base.rglob("metrics_test_*h.json"))[:80]:
            rel = mpath.relative_to(base)
            try:
                m = json.loads(mpath.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            lines.append(f"- `{rel}` MAE={m.get('MAE', '')} RMSE={m.get('RMSE', '')}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[build_leaderboard] 저장: {out}")


if __name__ == "__main__":
    main()
