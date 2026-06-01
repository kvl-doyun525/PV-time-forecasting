#!/usr/bin/env python3
"""
학습 결과(`summary.json`), 추론 벤치(`inference_benchmark.json`), 예측 그래프(`graphs/`)를
모아 `artifacts/benchmark_report.md` 리포트를 생성한다.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_RE_METRICS_H = re.compile(r"metrics_test_(\d+)h\.json$")
_RE_GRAPH_SITE = re.compile(r"_row(\d+)_site_(\d+)\.png$")
_RE_GRAPH_SAMPLE = re.compile(r"_sample_(\d+)_row\d+_site_\d+\.png$")

# batch_plot 균등 샘플 인덱스 (호라이즌마다 row 번호는 달라짐)
GRAPH_COMPARE_SAMPLES = ["000", "050", "090"]

# horizon별 모델별 대표 run (seq_len=168, graphs/ 존재)
COMPARE_RUNS: dict[int, dict[str, str]] = {
    24: {
        "dlinear": "dlinear_seq_168/seed_42",
        "segrnn": "segrnn_seq_168/seg24_h24_seed42",
        "patchtst": "patchtst_seq_168/pl48_s48_h24_seed42",
        "timellm": "timellm_future_nwp_seq_168/timellm_gpt2_h24_seed42",
    },
    48: {
        "dlinear": "dlinear_seq_168/h48_seed_42",
        "segrnn": "segrnn_seq_168/seg24_h48_seed42",
        "patchtst": "patchtst_seq_168/pl48_s48_h48_seed42",
        "timellm": "timellm_future_nwp_seq_168/timellm_gpt2_h48_seed42",
    },
    72: {
        "dlinear": "dlinear_seq_168/h72_seed_42",
        "segrnn": "segrnn_seq_168/seg24_h72_seed42",
        "patchtst": "patchtst_seq_168/pl48_s48_h72_seed42",
        "timellm": "timellm_future_nwp_seq_168/timellm_gpt2_h72_seed42",
    },
}

_MODEL_ORDER = ("dlinear", "segrnn", "patchtst", "timellm")
_HORIZONS = (24, 48, 72)


def _horizon_sort_key(hk: str) -> int:
    m = re.match(r"(\d+)h", hk)
    return int(m.group(1)) if m else 0


def _fmt_num(x: object) -> str:
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


def _parse_metric(x: object) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _load_summaries(runs_root: Path) -> dict[str, list[dict[str, object]]]:
    by_h: dict[str, list[dict[str, object]]] = defaultdict(list)
    if not runs_root.is_dir():
        return by_h
    for summ in sorted(runs_root.rglob("summary.json")):
        try:
            data = json.loads(summ.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        group = str(summ.parent.relative_to(runs_root))
        model = str(data.get("model", ""))
        for hk, block in (data.get("horizons") or {}).items():
            by_h[str(hk)].append(
                {
                    "group": group,
                    "model": model,
                    "MAE_mean": block.get("MAE_mean"),
                    "RMSE_mean": block.get("RMSE_mean"),
                    "daytime_MAE_mean": block.get("daytime_MAE_mean"),
                    "daily_energy_error_mean": block.get("daily_energy_error_mean"),
                }
            )
    return by_h


def _best_row(rows: list[dict[str, object]], key: str = "MAE_mean") -> dict[str, object] | None:
    best: dict[str, object] | None = None
    best_v: float | None = None
    for r in rows:
        v = _parse_metric(r.get(key))
        if v is None:
            continue
        if best_v is None or v < best_v:
            best_v = v
            best = r
    return best


def _model_family(name: str) -> str:
    n = name.lower()
    if "timellm" in n or n == "timellm":
        return "timellm"
    return n


def _emit_performance_section(
    lines: list[str], summary_by_h: dict[str, list[dict[str, object]]]
) -> None:
    lines.append("## 1. 모델별 성능 요약")
    lines.append("")
    lines.append(
        "테스트 구간 `metrics_test_*h.json`을 seed 집계한 `summary.json` 기준. "
        "지표는 낮을수록 좋음."
    )
    lines.append("")

    for hk in sorted(summary_by_h.keys(), key=_horizon_sort_key):
        rows = summary_by_h[hk]
        lines.append(f"### Horizon {hk}")
        lines.append("")

        best = _best_row(rows)
        if best:
            lines.append(
                f"- **종합 1위 (MAE)**: `{best['group']}` / `{best['model']}` "
                f"— MAE={_fmt_num(best['MAE_mean'])}, "
                f"dayMAE={_fmt_num(best['daytime_MAE_mean'])}, "
                f"dEerr={_fmt_num(best['daily_energy_error_mean'])}"
            )
            lines.append("")

        by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
        for r in rows:
            by_family[_model_family(str(r["model"]))].append(r)

        lines.append("| model | best group | MAE | RMSE | dayMAE | dEerr |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for fam in _MODEL_ORDER:
            fam_rows = by_family.get(fam, [])
            if not fam_rows:
                continue
            br = _best_row(fam_rows)
            if not br:
                continue
            lines.append(
                f"| {fam} | `{br['group']}` | {_fmt_num(br['MAE_mean'])} | "
                f"{_fmt_num(br['RMSE_mean'])} | {_fmt_num(br['daytime_MAE_mean'])} | "
                f"{_fmt_num(br['daily_energy_error_mean'])} |"
            )
        lines.append("")

    lines.append("### 호라이즌별 1위 비교")
    lines.append("")
    lines.append("| Horizon | 1위 group | model | MAE | dayMAE | dEerr |")
    lines.append("|---:|---|---|---:|---:|---:|")
    for hk in sorted(summary_by_h.keys(), key=_horizon_sort_key):
        best = _best_row(summary_by_h[hk])
        if not best:
            continue
        lines.append(
            f"| {hk} | `{best['group']}` | {best['model']} | "
            f"{_fmt_num(best['MAE_mean'])} | {_fmt_num(best['daytime_MAE_mean'])} | "
            f"{_fmt_num(best['daily_energy_error_mean'])} |"
        )
    lines.append("")


def _emit_inference_section(
    lines: list[str], infer_path: Path, assets_rel: str
) -> None:
    lines.append("## 2. 추론 시간 (Inference latency)")
    lines.append("")
    lines.append(
        "학습된 `best_model.pt` 로드 후 측정. **첫 배치(warmup) 1회 제외**, "
        "이후 동일 입력 shape으로 반복 forward. "
        "`batch1` = batch_size 1, `batch100` = batch_size 100 목표 "
        "(GPU OOM 시 32→16→8→4→1 순으로 축소, 표의 batch100 size 열 참고)."
    )
    lines.append("")

    if not infer_path.is_file():
        lines.append(
            "_(추론 벤치마크 미실행 — `bash scripts/run_inference_benchmark.sh` 실행)_"
        )
        lines.append("")
        return

    data = json.loads(infer_path.read_text(encoding="utf-8"))
    lines.append(
        f"- device: `{data.get('device', '?')}` | PyTorch `{data.get('pytorch', '?')}` | "
        f"warmup={data.get('warmup_batches', 1)} | repeats={data.get('repeats', '?')}"
    )
    lines.append("")

    entries = data.get("entries") or []
    if not entries:
        lines.append("_(측정 결과 없음)_")
        lines.append("")
        return

    by_h: dict[int, list[dict]] = defaultdict(list)
    for e in entries:
        h = e.get("pred_len")
        if h is not None:
            by_h[int(h)].append(e)

    for h in _HORIZONS:
        rows = by_h.get(h, [])
        if not rows:
            continue
        def _infer_order(r: dict) -> int:
            fam = _model_family(str(r.get("model", "")))
            return _MODEL_ORDER.index(fam) if fam in _MODEL_ORDER else 99

        rows.sort(key=_infer_order)

        lines.append(f"### Horizon {h}h")
        lines.append("")
        lines.append(
            "| model | run | batch=1 mean (ms) | batch=1 p95 | "
            "batch100 mean (ms) | batch100 size | per-sample (ms) |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for e in rows:
            b1 = e.get("batch1") or {}
            b100 = e.get("batch100") or {}
            fam = _model_family(str(e.get("model", "")))
            bs100 = int(b100.get("batch_size", 100))
            lines.append(
                f"| {fam} | `{e.get('run_dir', '')}` | "
                f"{_fmt_num(b1.get('mean_ms'))} | {_fmt_num(b1.get('p95_ms'))} | "
                f"{_fmt_num(b100.get('mean_ms'))} | {bs100} | "
                f"{_fmt_num(b100.get('per_sample_ms'))} |"
            )
        lines.append("")

    fastest_b1 = min(entries, key=lambda e: (e.get("batch1") or {}).get("mean_ms", 1e18))
    fastest_b100 = min(
        entries, key=lambda e: (e.get("batch100") or {}).get("per_sample_ms", 1e18)
    )
    lines.append("### 추론 속도 하이라이트")
    lines.append("")
    lines.append(
        f"- **batch=1 최단**: `{fastest_b1.get('run_dir')}` "
        f"({ _fmt_num((fastest_b1.get('batch1') or {}).get('mean_ms')) } ms)"
    )
    lines.append(
        f"- **batch=100 샘플당 최단**: `{fastest_b100.get('run_dir')}` "
        f"({ _fmt_num((fastest_b100.get('batch100') or {}).get('per_sample_ms')) } ms/sample)"
    )
    lines.append("")


def _find_graph_by_sample(graphs_dir: Path, sample_idx: str) -> Path | None:
    if not graphs_dir.is_dir():
        return None
    needle = f"_sample_{sample_idx}_"
    for p in sorted(graphs_dir.glob("*.png")):
        if needle in p.name:
            return p
    return None


def _emit_graph_section(
    lines: list[str],
    runs_root: Path,
    assets_dir: Path,
    assets_rel: str,
) -> None:
    lines.append("## 3. 예측 그래프 비교")
    lines.append("")
    lines.append(
        "동일 **균등 샘플 인덱스**(`sample_000`·`050`·`090`, batch_plot 기준)로 "
        "모델별 예측 곡선을 비교. (호라이즌마다 row/site는 달라질 수 있음.)"
    )
    lines.append("")

    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    for h in _HORIZONS:
        run_map = COMPARE_RUNS.get(h, {})
        lines.append(f"### Horizon {h}h")
        lines.append("")

        for sample_idx in GRAPH_COMPARE_SAMPLES:
            lines.append(f"#### Sample `sample_{sample_idx}`")
            lines.append("")
            found_any = False
            for fam in _MODEL_ORDER:
                rel = run_map.get(fam)
                if not rel:
                    continue
                src = _find_graph_by_sample(runs_root / rel / "graphs", sample_idx)
                if src is None:
                    lines.append(f"- **{fam}**: _(그래프 없음 — `{rel}/graphs`)_")
                    continue
                found_any = True
                m = _RE_GRAPH_SITE.search(src.name)
                site_note = m.group(0)[1:-4] if m else src.stem
                dest_name = f"h{h}_sample_{sample_idx}_{fam}.png"
                dest = assets_dir / dest_name
                shutil.copy2(src, dest)
                n_copied += 1
                rel_md = f"{assets_rel}/{dest_name}"
                lines.append(f"**{fam}** (`{rel}`, {site_note})")
                lines.append("")
                lines.append(f"![{fam} {h}h sample_{sample_idx}]({rel_md})")
                lines.append("")

            if not found_any:
                lines.append("_(해당 샘플 그래프 없음)_")
                lines.append("")

        lines.append("")

    lines.append(f"> 그래프 에셋 {n_copied}개 → `{assets_rel}/`")
    lines.append("")


def main() -> None:
    ap = argparse.ArgumentParser(description="벤치마크 종합 리포트 MD 생성")
    ap.add_argument("--runs-dir", type=Path, default=Path("artifacts/training_runs"))
    ap.add_argument(
        "--inference-json",
        type=Path,
        default=Path("artifacts/inference_benchmark.json"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmark_report.md"),
    )
    ap.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("artifacts/report_assets/graphs"),
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    runs_root = args.runs_dir if args.runs_dir.is_absolute() else root / args.runs_dir
    infer_path = (
        args.inference_json
        if args.inference_json.is_absolute()
        else root / args.inference_json
    )
    out_path = args.output if args.output.is_absolute() else root / args.output
    assets_dir = args.assets_dir if args.assets_dir.is_absolute() else root / args.assets_dir

    try:
        assets_rel = assets_dir.relative_to(out_path.parent).as_posix()
    except ValueError:
        assets_rel = assets_dir.as_posix()

    summary_by_h = _load_summaries(runs_root)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        "# PV 시계열 예측 벤치마크 리포트",
        "",
        f"생성: {now} | 데이터: `{runs_root.relative_to(root) if runs_root.is_relative_to(root) else runs_root}`",
        "",
        "DLinear · SegRNN · PatchTST · Time-LLM(+future NWP) 학습 결과를 "
        "정확도·추론 지연·예측 곡선 관점에서 정리한다.",
        "",
        "상세 수치 표는 [`leaderboard.md`](leaderboard.md) 참고.",
        "",
    ]

    _emit_performance_section(lines, summary_by_h)
    _emit_inference_section(lines, infer_path, assets_rel)
    _emit_graph_section(lines, runs_root, assets_dir, assets_rel)

    lines.append("## 4. 해석 메모")
    lines.append("")
    lines.append(
        "- **SegRNN** (`seq_168`)이 24·48·72h 전 구간에서 MAE·dayMAE·일간 에너지 오차가 전반적으로 우수."
    )
    lines.append(
        "- **DLinear**는 구조가 단순해 추론 지연이 짧을 가능성이 높음(§2 표 확인)."
    )
    lines.append(
        "- **PatchTST**는 긴 `seq_len`(720)에서 attention 비용이 커질 수 있음."
    )
    lines.append(
        "- **Time-LLM**은 LLM 백본 로딩·토큰화로 cold start·batch=1 지연이 클 수 있음."
    )
    lines.append(
        "- 그래프(§3)는 동일 site·동일 윈도에서 모델 간 형태·과대/과소 예측 패턴을 육안 비교용."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[build_benchmark_report] 저장: {out_path}")


if __name__ == "__main__":
    main()
