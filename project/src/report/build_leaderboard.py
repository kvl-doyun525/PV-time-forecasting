#!/usr/bin/env python3
"""
`training_runs/**/summary.json` 과 `metrics_test_*h.json`을 읽어
`artifacts/leaderboard.md`로 요약한다.

horizon(24h / 48h / 72h)마다 별도 표·절을 둔다.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


_RE_METRICS_H = re.compile(r"metrics_test_(\d+)h\.json$")


def _horizon_sort_key(hk: str) -> int:
    m = re.match(r"(\d+)h", hk)
    return int(m.group(1)) if m else 0


def _fmt_num(x: object) -> str:
    """리더보드 셀용 짧은 숫자 문자열(표 가로 폭·가독성)."""
    if x is None or x == "":
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        v = float(x)
        if abs(v) >= 1e5 or (0 < abs(v) < 1e-4):
            return f"{v:.3e}"
        return f"{v:.4f}".rstrip("0").rstrip(".") or "0"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)[:12]
    if abs(v) >= 1e5 or (0 < abs(v) < 1e-4):
        return f"{v:.3e}"
    return f"{v:.4f}".rstrip("0").rstrip(".") or "0"


def _short_group(s: str, max_len: int = 28) -> str:
    """표 열 폭: 실험 그룹 경로만 잘라 표시."""
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def main() -> None:
    ap = argparse.ArgumentParser(description="요약 리더보드 MD (horizon별 표)")
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
    ap.add_argument(
        "--max-raw-per-horizon",
        type=int,
        default=80,
        metavar="N",
        help="Raw metrics 절에서 horizon당 최대 줄 수 (경로 오름차순)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    base = args.runs_dir if args.runs_dir.is_absolute() else root / args.runs_dir
    out = args.output if args.output.is_absolute() else root / args.output

    lines: list[str] = [
        "# Benchmark leaderboard",
        "",
        "horizon(예측 길이)마다 표를 나눈다. `summary.json`은 seed 집계, Raw는 개별 `metrics_test_*h.json`. "
        "숫자는 표시용으로 소수 4자리(또는 매우 작/큰 값은 과학 표기)로 반올림한다. "
        "긴 `group`/경로는 잘림(`…`).",
        "",
    ]

    # --- summary.json: horizon별 표 ---
    summary_by_h: dict[str, list[dict[str, str]]] = defaultdict(list)
    if base.is_dir():
        for summ in sorted(base.rglob("summary.json")):
            try:
                data = json.loads(summ.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            model = str(data.get("model", ""))
            group = str(summ.parent.relative_to(base))
            for hk, block in (data.get("horizons") or {}).items():
                summary_by_h[str(hk)].append(
                    {
                        "group": _short_group(group),
                        "model": _short_group(model, 22),
                        "MAE_mean": _fmt_num(block.get("MAE_mean")),
                        "MAE_std": _fmt_num(block.get("MAE_std")),
                        "daytime_MAE_mean": _fmt_num(block.get("daytime_MAE_mean")),
                        "daily_energy_error_mean": _fmt_num(
                            block.get("daily_energy_error_mean")
                        ),
                        "daily_energy_error_std": _fmt_num(
                            block.get("daily_energy_error_std")
                        ),
                    }
                )

    lines.append("## summary.json (seed 집계, horizon별)")
    lines.append("")
    if not summary_by_h:
        lines.append("_(summary.json 없음)_")
        lines.append("")
    else:
        for hk in sorted(summary_by_h.keys(), key=_horizon_sort_key):
            rows = summary_by_h[hk]
            rows.sort(key=lambda r: (r["group"], r["model"]))
            lines.append(f"### Horizon {hk}")
            lines.append("")
            lines.append(
                "| group | model | MAE_m | MAE_σ | dayMAE_m | dEerr_m | dEerr_σ |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---:|")
            for r in rows:
                lines.append(
                    f"| {r['group']} | {r['model']} | {r['MAE_mean']} | {r['MAE_std']} | "
                    f"{r['daytime_MAE_mean']} | {r['daily_energy_error_mean']} | "
                    f"{r['daily_energy_error_std']} |"
                )
            lines.append("")

    # --- Raw metrics: horizon별 ---
    lines.append("## Raw metrics (개별 run, horizon별)")
    lines.append("")
    raw_by_h: dict[int, list[tuple[str, str, str, str]]] = defaultdict(list)
    if base.is_dir():
        for mpath in sorted(base.rglob("metrics_test_*h.json")):
            mm = _RE_METRICS_H.search(mpath.name)
            if not mm:
                continue
            hi = int(mm.group(1))
            rel = str(mpath.relative_to(base))
            try:
                m = json.loads(mpath.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            raw_by_h[hi].append(
                (
                    rel,
                    _fmt_num(m.get("MAE")),
                    _fmt_num(m.get("RMSE")),
                    _fmt_num(m.get("daily_energy_error")),
                )
            )

    if not raw_by_h:
        lines.append("_(metrics_test_*h.json 없음)_")
        lines.append("")
    else:
        for hi in sorted(raw_by_h.keys()):
            entries = sorted(raw_by_h[hi], key=lambda t: t[0])
            cap = max(0, int(args.max_raw_per_horizon))
            if cap and len(entries) > cap:
                entries = entries[:cap]
                note = f" _(상위 {cap}개, 경로순)_"
            else:
                note = ""
            lines.append(f"### {hi}h{note}")
            lines.append("")
            for rel, mae, rmse, dee in entries:
                rel_disp = _short_group(rel, 56)
                lines.append(
                    f"- `{rel_disp}` MAE={mae} RMSE={rmse} dEerr={dee}"
                )
            lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[build_leaderboard] 저장: {out}")


if __name__ == "__main__":
    main()
