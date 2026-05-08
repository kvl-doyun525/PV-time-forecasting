#!/usr/bin/env python3
"""
동일 실험 그룹(`artifacts/training_runs/<group>/`) 아래 seed별 `metrics_test_{H}h.json`을 모아
`summary.json`을 갱신한다.

glob 패턴은 과거 `seg24_h24_seed42`, `h48_seed_42`, `seed_42` 등 **혼용 네이밍**을 수용한다.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _collect_metric_files(group_root: Path, model: str, horizon: int) -> list[Path]:
    name = f"metrics_test_{horizon}h.json"
    m = model.lower()
    hits: set[Path] = set()

    def add_glob(pattern: str) -> None:
        for p in group_root.glob(pattern):
            if p.is_file():
                hits.add(p)

    if m == "dlinear":
        if horizon == 24:
            add_glob(f"**/seed_*/{name}")
        add_glob(f"**/h{horizon}_seed*/{name}")
        add_glob(f"**/h{horizon}_seed_*/{name}")
    elif m in ("segrnn", "patchtst", "timellm", "time_llm"):
        add_glob(f"**/*h{horizon}_seed*/{name}")
        add_glob(f"**/*h{horizon}_seed_*/{name}")
    elif m in ("llama_lora", "gemma_lora", "llama", "gemma"):
        add_glob(f"**/*h{horizon}_seed*/{name}")
        add_glob(f"**/*seed*{horizon}*/{name}")
    else:
        add_glob(f"**/*h{horizon}_seed*/{name}")

    return sorted(hits)


def _aggregate(files: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        return {}

    keys: set[str] = set()
    for r in rows:
        keys.update(r.keys())

    out: dict[str, Any] = {}
    for k in sorted(keys):
        vals = [r[k] for r in rows if k in r and _is_number(r[k])]
        if not vals:
            continue
        out[f"{k}_mean"] = float(statistics.mean(vals))
        out[f"{k}_std"] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
        out[f"{k}_values"] = vals

    out["n_seeds"] = len(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="seed별 metrics → summary.json")
    ap.add_argument("--model", required=True, help="dlinear|segrnn|patchtst|timellm|llama_lora|gemma_lora …")
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/training_runs"),
        help="training_runs 루트",
    )
    ap.add_argument(
        "--runs-group",
        type=str,
        default="",
        help="하위 그룹 폴더명 (예: segrnn_seq_168). 비우면 --runs-dir 직하위 전체에서 탐색",
    )
    ap.add_argument("--horizons", type=int, nargs="+", default=[24, 48, 72])
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="summary.json 경로 (기본: <runs-dir>/<runs-group>/summary.json 또는 runs-dir/summary_<model>.json)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else root / args.runs_dir
    if args.runs_group:
        group_root = runs_dir / args.runs_group
    else:
        group_root = runs_dir

    if not group_root.is_dir():
        raise SystemExit(f"디렉터리 없음: {group_root}")

    out_path = args.output
    if out_path is None:
        out_path = (
            group_root / "summary.json"
            if args.runs_group
            else runs_dir / f"summary_{args.model}.json"
        )
    else:
        out_path = out_path if out_path.is_absolute() else root / out_path

    horizons_block: dict[str, Any] = {}
    for h in args.horizons:
        files = _collect_metric_files(group_root, args.model, h)
        agg = _aggregate(files)
        if not agg:
            print(f"[aggregate_seeds] horizon={h}h: 결과 없음 (glob 대상: {group_root})")
            continue
        horizons_block[f"{h}h"] = agg
        mae_m = agg.get("MAE_mean", "")
        dtm = agg.get("daytime_MAE_mean", "N/A")
        print(
            f"[aggregate_seeds] horizon={h}h | MAE={mae_m}±{agg.get('MAE_std', '')} | "
            f"daytime_MAE={dtm} | n_seeds={agg.get('n_seeds', 0)}"
        )

    payload = {"model": args.model.lower(), "horizons": horizons_block}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[aggregate_seeds] 집계 완료: {out_path}")


if __name__ == "__main__":
    main()
