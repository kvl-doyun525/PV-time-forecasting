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

# (summary.json 키, 표 헤더, 낮을수록 좋음)
_SUMMARY_METRICS: list[tuple[str, str, bool]] = [
    ("MAE_mean", "MAE_m", True),
    ("MAE_std", "MAE_σ", True),
    ("daytime_MAE_mean", "dayMAE_m", True),
    ("daily_energy_error_mean", "dEerr_m", True),
    ("daily_energy_error_std", "dEerr_σ", True),
]

_RANK_STYLE = {
    1: "background-color:#b9f6ca",  # 1위: 연녹
    2: "background-color:#fff59d",  # 2위: 연노랑
}


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


def _parse_metric(x: object) -> float | None:
    if x is None or x == "":
        return None
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _metric_ranks(values: list[float | None], *, lower_better: bool) -> list[int | None]:
    """각 인덱스에 1(최고)·2(차선) 또는 None."""
    ranked = sorted(
        ((i, v) for i, v in enumerate(values) if v is not None),
        key=lambda iv: iv[1],
        reverse=not lower_better,
    )
    out: list[int | None] = [None] * len(values)
    for rank, (idx, _) in enumerate(ranked[:2], start=1):
        out[idx] = rank
    return out


def _styled_cell(text: str, rank: int | None) -> str:
    if rank is None:
        return text
    style = _RANK_STYLE.get(rank)
    if not style:
        return text
    return f'<span style="{style}">{text}</span>'


def _row_label(group: str, model: str) -> str:
    return f"{group} / {model}"


def _emit_best_per_horizon(
    lines: list[str],
    summary_by_h: dict[str, list[dict[str, object]]],
) -> None:
    lines.append("## Best per horizon (summary.json)")
    lines.append("")
    lines.append(
        "seed 집계 기준. **종합 1위**는 `MAE_m` 최소. "
        "아래 표의 1·2위 셀은 연녹·연노랑으로 표시한다."
    )
    lines.append("")

    if not summary_by_h:
        lines.append("_(summary.json 없음)_")
        lines.append("")
        return

    for hk in sorted(summary_by_h.keys(), key=_horizon_sort_key):
        rows = summary_by_h[hk]
        mae_vals = [_parse_metric(r.get("MAE_mean_raw")) for r in rows]
        mae_ranks = _metric_ranks(mae_vals, lower_better=True)
        best_idx = next((i for i, rk in enumerate(mae_ranks) if rk == 1), None)

        lines.append(f"### Horizon {hk}")
        lines.append("")
        if best_idx is not None:
            br = rows[best_idx]
            lines.append(
                f"- **종합 1위 (MAE_m)**: `{br['group']}` / `{br['model']}` "
                f"— MAE_m={br['MAE_mean']}, dayMAE_m={br['daytime_MAE_mean']}, "
                f"dEerr_m={br['daily_energy_error_mean']}"
            )
            lines.append("")

        lines.append("| metric | 1st | value | 2nd | value |")
        lines.append("|---|---|---:|---|---:|")
        for key, header, lower_better in _SUMMARY_METRICS:
            vals = [_parse_metric(r.get(f"{key}_raw")) for r in rows]
            ranks = _metric_ranks(vals, lower_better=lower_better)
            first_label, first_val = "—", "—"
            second_label, second_val = "—", "—"
            for idx, rk in enumerate(ranks):
                if rk == 1:
                    first_label = _row_label(
                        str(rows[idx]["group"]), str(rows[idx]["model"])
                    )
                    first_val = str(rows[idx][key])
                elif rk == 2:
                    second_label = _row_label(
                        str(rows[idx]["group"]), str(rows[idx]["model"])
                    )
                    second_val = str(rows[idx][key])
            lines.append(
                f"| {header} | {first_label} | {first_val} | "
                f"{second_label} | {second_val} |"
            )
        lines.append("")


def _emit_summary_tables(
    lines: list[str],
    summary_by_h: dict[str, list[dict[str, object]]],
) -> None:
    lines.append("## summary.json (seed 집계, horizon별)")
    lines.append("")
    if not summary_by_h:
        lines.append("_(summary.json 없음)_")
        lines.append("")
        return

    for hk in sorted(summary_by_h.keys(), key=_horizon_sort_key):
        rows = summary_by_h[hk]
        rows.sort(key=lambda r: (str(r["group"]), str(r["model"])))

        col_ranks: dict[str, list[int | None]] = {}
        for key, _, lower_better in _SUMMARY_METRICS:
            vals = [_parse_metric(r.get(f"{key}_raw")) for r in rows]
            col_ranks[key] = _metric_ranks(vals, lower_better=lower_better)

        lines.append(f"### Horizon {hk}")
        lines.append("")
        lines.append(
            "| group | model | MAE_m | MAE_σ | dayMAE_m | dEerr_m | dEerr_σ |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for ri, r in enumerate(rows):
            cells = [
                str(r["group"]),
                str(r["model"]),
            ]
            for key, _, _ in _SUMMARY_METRICS:
                cells.append(_styled_cell(str(r[key]), col_ranks[key][ri]))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")


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
        "긴 `group`/경로는 잘림(`…`). "
        "summary 표에서 **연녹=1위**, **연노랑=2위** (지표별, 낮을수록 좋음).",
        "",
    ]

    summary_by_h: dict[str, list[dict[str, object]]] = defaultdict(list)
    if base.is_dir():
        for summ in sorted(base.rglob("summary.json")):
            try:
                data = json.loads(summ.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            model = str(data.get("model", ""))
            group = str(summ.parent.relative_to(base))
            for hk, block in (data.get("horizons") or {}).items():
                row: dict[str, object] = {
                    "group": _short_group(group),
                    "model": _short_group(model, 22),
                }
                for key, _, _ in _SUMMARY_METRICS:
                    raw = block.get(key)
                    row[f"{key}_raw"] = raw
                    row[key] = _fmt_num(raw)
                summary_by_h[str(hk)].append(row)

    _emit_best_per_horizon(lines, summary_by_h)
    _emit_summary_tables(lines, summary_by_h)

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
