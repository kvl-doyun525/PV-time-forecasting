"""
공통 테스트 지표 계산 (예측 parquet + feature_mart_per_site/test).

`train_tslib_model.py` 평가와 동일한 규칙:
- test parquet는 `ffill().fillna(0.0)` 후 윈도우 정렬·슬라이스
- MAE/RMSE/nRMSE/sMAPE/daily_energy_error는 **flatten(시간×윈도우)** 기준
- `solar_elevation`이 있으면 daytime(>5°) 부분집합에 대해 daytime_MAE / daytime_nRMSE

복구 기준: PhotoRec·에이전트 기록과 기존 `metrics_test_*h.json` 수치 역추적 (2026-05-06 이전 스냅샷과 정합).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_DAYTIME_ELEV_THRESHOLD = 5.0  # split_manifest.yaml: solar_elevation > 5deg
_EPS = 1e-8


def infer_pred_len_from_columns(df: pd.DataFrame) -> int:
    cols = [c for c in df.columns if c.startswith("pred_h")]
    if not cols:
        raise ValueError("pred_h* 컬럼이 없습니다.")
    max_h = max(int(re.sub(r"^pred_h", "", c)) for c in cols)
    return max_h + 1


def pred_horizon_columns(df: pd.DataFrame, pred_len: int) -> list[str]:
    need = [f"pred_h{i}" for i in range(pred_len)]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"예측 컬럼 부족: {miss[:5]}...")
    return need


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    solar_elevation: np.ndarray | None = None,
    daytime_threshold: float = _DAYTIME_ELEV_THRESHOLD,
) -> dict[str, Any]:
    """y_true / y_pred: (N, H) 또는 동일 길이의 1D. 반환값은 JSON 저장용으로 반올림."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"shape 불일치: y_true {yt.shape} vs y_pred {yp.shape}")

    if yt.ndim == 1:
        yt2 = yt.reshape(1, -1)
        yp2 = yp.reshape(1, -1)
    else:
        yt2, yp2 = yt, yp

    daily_energy_error = float(np.mean(np.abs(np.sum(yp2, axis=1) - np.sum(yt2, axis=1))))

    flat_t = yt2.reshape(-1)
    flat_p = yp2.reshape(-1)
    mae = float(np.mean(np.abs(flat_t - flat_p)))
    rmse = float(np.sqrt(np.mean((flat_t - flat_p) ** 2)))
    smape = float(
        100.0
        * np.mean(2.0 * np.abs(flat_p - flat_t) / (np.abs(flat_t) + np.abs(flat_p) + _EPS))
    )

    out: dict[str, Any] = {
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "nRMSE": round(rmse, 6),
        "sMAPE": round(smape, 6),
        "daily_energy_error": round(daily_energy_error, 6),
        "n_samples": int(flat_t.size),
    }

    if solar_elevation is not None:
        el = np.asarray(solar_elevation, dtype=np.float64)
        if el.shape != yt2.shape:
            raise ValueError(f"solar_elevation shape {el.shape} != y_true {yt2.shape}")
        mask = el.reshape(-1) > daytime_threshold
        if np.any(mask):
            err = np.abs(flat_t - flat_p)
            dt_mae = float(np.mean(err[mask]))
            dt_rmse = float(np.sqrt(np.mean(((flat_t - flat_p) ** 2)[mask])))
            out["daytime_MAE"] = round(dt_mae, 6)
            out["daytime_nRMSE"] = round(dt_rmse, 6)
            out["n_daytime_samples"] = int(np.sum(mask))

    return out


def _loc_to_int(idx_loc: int | slice | np.ndarray) -> int:
    if isinstance(idx_loc, slice):
        return int(idx_loc.start or 0)
    if isinstance(idx_loc, (np.ndarray, list)):
        return int(np.asarray(idx_loc).ravel()[0])
    return int(idx_loc)


def metrics_from_predictions_parquet(
    predictions_path: str,
    feature_mart_dir: str,
    pred_len: int | None = None,
) -> dict[str, Any]:
    pred_df = pd.read_parquet(predictions_path)
    horizon = pred_len if pred_len is not None else infer_pred_len_from_columns(pred_df)
    pred_cols = pred_horizon_columns(pred_df, horizon)

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_elev: list[np.ndarray] = []

    for site_id, grp in pred_df.groupby("site_id"):
        test_path = os.path.join(feature_mart_dir, "test", f"{site_id}.parquet")
        if not os.path.exists(test_path):
            continue
        test_full = pd.read_parquet(test_path).ffill().fillna(0.0)
        if "normalized_power" not in test_full.columns:
            continue
        cols = ["normalized_power"]
        if "solar_elevation" in test_full.columns:
            cols.append("solar_elevation")
        test_use = test_full[cols]

        for _, row in grp.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            end_ts = ts + pd.Timedelta(hours=horizon - 1)
            window = test_use.loc[ts:end_ts]
            if len(window) < horizon:
                continue
            all_true.append(window["normalized_power"].values[:horizon].astype(np.float64, copy=False))
            all_pred.append([float(row[c]) for c in pred_cols])
            if "solar_elevation" in window.columns:
                all_elev.append(window["solar_elevation"].values[:horizon].astype(np.float64, copy=False))

    if not all_true:
        return {}

    y_true = np.asarray(all_true, dtype=np.float64)
    y_pred = np.asarray(all_pred, dtype=np.float64)
    y_elev = np.asarray(all_elev, dtype=np.float64) if all_elev else None

    return compute_metrics(y_true, y_pred, solar_elevation=y_elev)


def evaluate_all_sites(
    predictions_parquet: str,
    feature_mart_dir: str,
    horizon: int | None = None,
) -> dict[str, Any]:
    pred_df = pd.read_parquet(predictions_parquet)
    h = horizon if horizon is not None else infer_pred_len_from_columns(pred_df)
    pred_cols = pred_horizon_columns(pred_df, h)

    per_site: dict[str, dict[str, Any]] = {}
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_elev: list[np.ndarray] = []

    for site_id, grp in pred_df.groupby("site_id"):
        test_path = os.path.join(feature_mart_dir, "test", f"{site_id}.parquet")
        if not os.path.exists(test_path):
            continue
        test_full = pd.read_parquet(test_path).ffill().fillna(0.0)
        cols = ["normalized_power"]
        if "solar_elevation" in test_full.columns:
            cols.append("solar_elevation")
        test_use = test_full[cols]

        site_true: list[np.ndarray] = []
        site_pred: list[np.ndarray] = []
        site_elev: list[np.ndarray] = []

        for _, row in grp.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            end_ts = ts + pd.Timedelta(hours=h - 1)
            window = test_use.loc[ts:end_ts]
            if len(window) < h:
                continue
            site_true.append(window["normalized_power"].values[:h].astype(np.float64, copy=False))
            site_pred.append([float(row[c]) for c in pred_cols])
            if "solar_elevation" in window.columns:
                site_elev.append(window["solar_elevation"].values[:h].astype(np.float64, copy=False))

        if not site_true:
            continue

        y_true = np.asarray(site_true, dtype=np.float64)
        y_pred = np.asarray(site_pred, dtype=np.float64)
        y_elev = np.asarray(site_elev, dtype=np.float64) if site_elev else None

        per_site[str(site_id)] = compute_metrics(y_true, y_pred, solar_elevation=y_elev)
        all_true.append(y_true)
        all_pred.append(y_pred)
        if y_elev is not None:
            all_elev.append(y_elev)

    if not all_true:
        return {"overall": {}, "per_site": per_site}

    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)
    y_elev_all = np.concatenate(all_elev, axis=0) if all_elev else None

    overall = compute_metrics(y_true_all, y_pred_all, solar_elevation=y_elev_all)
    return {"overall": overall, "per_site": per_site}


def main() -> None:
    parser = argparse.ArgumentParser(description="모델 예측 결과 평가")
    parser.add_argument("--predictions", required=True, help="predictions parquet 경로")
    parser.add_argument("--feature-mart", required=True, help="feature_mart_per_site 디렉토리")
    parser.add_argument("--horizon", type=int, required=True, help="예측 horizon (시간)")
    parser.add_argument("--output", required=True, help="출력 json 경로")
    args = parser.parse_args()

    results = evaluate_all_sites(
        predictions_parquet=args.predictions,
        feature_mart_dir=args.feature_mart,
        horizon=args.horizon,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[evaluate_model] 저장 완료: {args.output}")
    print("Overall:", json.dumps(results["overall"], indent=2))


if __name__ == "__main__":
    main()
