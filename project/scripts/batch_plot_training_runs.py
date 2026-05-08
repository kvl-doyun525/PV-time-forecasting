#!/usr/bin/env python3
"""
`training_runs` 하위의 모든 `predictions_test_{H}h.parquet`에 대해
정렬된 행 기준 **균등 간격 k개** 샘플의 시계열 그래프를 생성한다.
(첫 k행이 아님 — 정렬된 전체 행을 N-1 구간에 k개로 등분한 인덱스, 0·N-1 포함)

각 예측 파일이 위치한 run 폴더 안에 `graphs/` 디렉터리를 만들고 PNG를 저장한다.

복구: `recup_dir.7/f567513912.py` (권한 처리·`--time-axis`·스킵 로직) 기준, `--time-axis` 기본은 꺼짐.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.report.plot_forecast_samples import plot_one_sample

PRED_PATTERN = re.compile(r"predictions_test_(\d+)h\.parquet$")


def evenly_spaced_indices(n_rows: int, k: int) -> list[int]:
    """정렬 후 행 번호 [0, n_rows) 구간에서 k개 균등 간격 인덱스 (첫 행·마지막 행 포함)."""
    if n_rows <= 0:
        return []
    k_eff = min(k, n_rows)
    if k_eff == 1:
        return [0]
    out = [round(i * (n_rows - 1) / (k_eff - 1)) for i in range(k_eff)]
    return sorted(set(out))


def horizon_from_filename(path: Path) -> int | None:
    m = PRED_PATTERN.search(path.name)
    if not m:
        return None
    return int(m.group(1))


def discover_prediction_parquets(training_runs_root: Path) -> list[Path]:
    paths = sorted(training_runs_root.rglob("predictions_test_*h.parquet"))
    return [p for p in paths if horizon_from_filename(p) is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description="training_runs 전체 예측 parquet → 균등 샘플 그래프 일괄 생성")
    parser.add_argument("--training-runs", type=Path, default=PROJECT_ROOT / "artifacts" / "training_runs")
    parser.add_argument("--feature-mart", type=Path, default=PROJECT_ROOT / "artifacts" / "feature_mart_per_site")
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--capacity-kw", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--time-axis",
        action="store_true",
        help="x축을 예측 시작 시각 기준 실제 일시(시간별)로 표시",
    )
    parser.add_argument("--dry-run", action="store_true", help="파일 목록만 출력하고 그리지 않음")
    args = parser.parse_args()

    root = args.training_runs.resolve()
    fm = args.feature_mart.resolve()
    if not root.is_dir():
        sys.exit(f"디렉터리 없음: {root}")

    parquet_files = discover_prediction_parquets(root)
    if not parquet_files:
        print(f"[batch_plot] 예측 parquet 없음 under {root}")
        return

    feature_mart_str = str(fm)

    total_png = 0
    for pred_path in parquet_files:
        horizon = horizon_from_filename(pred_path)
        assert horizon is not None

        run_dir = pred_path.parent
        graphs_dir = run_dir / "graphs"
        stem = run_dir.name

        df = pd.read_parquet(pred_path).sort_values(["site_id", "timestamp"]).reset_index(drop=True)
        n = len(df)
        indices = evenly_spaced_indices(n, args.n_samples)

        print(
            f"[batch_plot] {pred_path.relative_to(PROJECT_ROOT)} "
            f"| rows={n} horizon={horizon} | 균등 {len(indices)}개 → {graphs_dir.relative_to(PROJECT_ROOT)}"
        )

        if args.dry_run:
            continue

        try:
            graphs_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print(
                f"[skip] graphs 폴더 생성 불가 (권한): {graphs_dir}\n"
                f"       예: docker run --rm -v \"$PWD/artifacts:/workspace/artifacts\" "
                f"pv-benchmark/unified:latest chmod -R u+rwX /workspace/artifacts/training_runs/segrnn"
            )
            continue

        for seq_i, idx in enumerate(indices):
            row = df.iloc[idx]
            fname = graphs_dir / f"{stem}_sample_{seq_i:03d}_row{idx}_site_{row['site_id']}.png"
            if os.path.exists(fname):
                continue
            fig, ax = plt.subplots(figsize=(10, 4.5))
            plot_one_sample(
                ax,
                sample_idx=idx,
                row=row,
                feature_mart=feature_mart_str,
                seq_len=args.seq_len,
                horizon=horizon,
                compare_rows=[],
                capacity_kw=args.capacity_kw,
                time_axis=args.time_axis,
            )
            fig.tight_layout()

            try:
                fig.savefig(fname, dpi=args.dpi)
                total_png += 1
            except PermissionError as e:
                print(f"[skip] 저장 불가 {fname}: {e}")
                plt.close(fig)
                break
            plt.close(fig)

    if args.dry_run:
        print(f"[dry-run] 대상 parquet {len(parquet_files)}개")
        return

    print(f"[batch_plot] 완료: 총 PNG {total_png}개 생성")


if __name__ == "__main__":
    main()
