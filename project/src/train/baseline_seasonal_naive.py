#!/usr/bin/env python3
"""
Seasonal Naive / Persistence 베이스라인.

- `train_tslib_model.py` 와 동일한 `predictions_test_{H}h.parquet` 형식
  (`site_id`, `timestamp`, `pred_h0` … `pred_h{H-1}`)
- `load_test_windows` 와 동일한 윈도·stride(pred_len)로 테스트 슬라이스 정렬
- 지표는 `evaluate_model.metrics_from_predictions_parquet` 경로

복구: `pv_model_benchmark_execution.md` §6.1–6.2, `reeval_predictions.py` 형식 정합.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from benchmark.evaluate_model import metrics_from_predictions_parquet  # noqa: E402
from datasets.pv_dataset import SingleSiteDataset  # noqa: E402

_SEASONAL_LAG = 7 * 24


def _seasonal_vector(pw: np.ndarray, first_pred_idx: int, pred_len: int) -> np.ndarray:
    out = np.zeros(pred_len, dtype=np.float64)
    for h in range(pred_len):
        j = first_pred_idx + h - _SEASONAL_LAG
        if j < 0:
            raise ValueError("seasonal naive: 인덱스 음수 — seq_len·horizon 확인")
        out[h] = pw[j]
    return np.clip(out, 0.0, 1.0)


def _persistence_vector(pw: np.ndarray, i: int, seq_len: int, pred_len: int) -> np.ndarray:
    block = pw[i + seq_len - 24 : i + seq_len]
    if block.shape[0] != 24:
        raise ValueError("persistence: 과거 24시간 구간 필요")
    tiled = np.tile(block, (pred_len // 24 + 1,))[:pred_len]
    return np.clip(tiled.astype(np.float64), 0.0, 1.0)


def _rows_for_site(
    path: str,
    *,
    seq_len: int,
    pred_len: int,
    method: str,
) -> list[dict]:
    site_id = Path(path).stem
    ds = SingleSiteDataset(
        path,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=pred_len,
    )
    if len(ds) == 0:
        return []

    idx_vals = ds.df.index.values
    pw = ds.df["normalized_power"].to_numpy(dtype=np.float64)
    pw = np.nan_to_num(pw, nan=0.0)

    rows: list[dict] = []
    for j in range(len(ds)):
        i = ds.indices[j]
        first_pred_idx = i + seq_len
        ts = idx_vals[first_pred_idx]
        if method == "seasonal":
            preds = _seasonal_vector(pw, first_pred_idx, pred_len)
        elif method == "persistence":
            preds = _persistence_vector(pw, i, seq_len, pred_len)
        else:
            raise ValueError(method)
        row: dict = {"site_id": site_id, "timestamp": pd.Timestamp(ts)}
        for k in range(pred_len):
            row[f"pred_h{k}"] = float(preds[k])
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Seasonal Naive / Persistence baseline")
    p.add_argument(
        "--feature-mart",
        default="artifacts/feature_mart_per_site",
        help="feature_mart_per_site 루트 (train/valid/test 하위)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="예측·지표 저장 디렉터리",
    )
    p.add_argument(
        "--method",
        choices=("seasonal", "persistence"),
        default="seasonal",
        help="seasonal: 168h lag 동일 시각, persistence: 직전 24h 패턴 반복",
    )
    p.add_argument("--seq-len", type=int, default=168)
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=(24, 48, 72),
        help="예측 길이(시간) 목록",
    )
    p.add_argument(
        "--manifest",
        default="",
        help="선택: split_manifest.yaml (현재는 로깅용)",
    )
    args = p.parse_args()

    mart = args.feature_mart
    if not os.path.isabs(mart):
        mart = str(_PROJECT_ROOT / mart)
    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = str(_PROJECT_ROOT / out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.manifest:
        mf = args.manifest
        if not os.path.isabs(mf):
            mf = str(_PROJECT_ROOT / mf)
        if os.path.isfile(mf):
            print(f"[baseline] manifest 참조: {mf}")

    for pred_len in args.horizons:
        all_rows: list[dict] = []
        for path in sorted(glob.glob(os.path.join(mart, "test", "*.parquet"))):
            all_rows.extend(
                _rows_for_site(
                    path,
                    seq_len=args.seq_len,
                    pred_len=pred_len,
                    method=args.method,
                )
            )
        if not all_rows:
            print(f"[baseline] horizon={pred_len}: 예측 행 없음 — 건너뜀")
            continue

        pred_df = pd.DataFrame(all_rows)
        pred_path = os.path.join(out_dir, f"predictions_test_{pred_len}h.parquet")
        pred_df.to_parquet(pred_path, index=False)
        print(f"[baseline] 저장: {pred_path} ({len(pred_df)} rows)")

        metrics = metrics_from_predictions_parquet(pred_path, mart, pred_len=pred_len)
        if not metrics:
            print(f"[baseline] horizon={pred_len}: 지표 계산 실패 (mart 정렬 확인)")
            continue
        metrics_path = os.path.join(out_dir, f"metrics_test_{pred_len}h.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        extra = ""
        if metrics.get("daytime_MAE") is not None:
            extra = f" daytime_MAE={metrics['daytime_MAE']:.6f}"
        print(
            f"[baseline] method={args.method} H={pred_len} "
            f"MAE={metrics['MAE']:.6f} RMSE={metrics['RMSE']:.6f}{extra}"
        )

    print(f"[baseline] 완료 → {out_dir}")


if __name__ == "__main__":
    main()
