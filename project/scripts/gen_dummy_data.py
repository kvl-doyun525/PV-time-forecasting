"""
더미 데이터 생성 스크립트.

실제 Feature Mart와 동일한 형태(크기·컬럼·정규화 범위)의 더미 parquet 파일을 생성한다.
모델 파이프라인 smoke test / 속도 측정 전용.

스펙:
  - 해상도: 1H
  - 기간: 3년 (train 2y / valid 6m / test 6m) = 26280 rows
  - Feature 수: 25 (normalized)
  - lookback: 8760 (seq_len)
  - horizon: 24 (pred_len)

사용법:
  python scripts/gen_dummy_data.py
  python scripts/gen_dummy_data.py --rows 26280 --out artifacts/dummy_feature_mart
"""

import argparse
import json
import os
import numpy as np
import pandas as pd

# ── feature 스펙 ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # target
    "normalized_power",
    # weather (ASOS)
    "ta", "rn", "ws", "wd", "hm", "pa", "si", "ss", "dc10Tca",
    # solar (pvlib)
    "solar_elevation", "solar_azimuth", "clearsky_ghi",
    # rolling
    "pv_roll_mean_24h", "pv_roll_std_24h",
    "pv_roll_mean_72h", "pv_roll_std_72h",
    "pv_roll_mean_168h", "pv_roll_std_168h",
    # lag
    "pv_lag_24h", "pv_lag_168h",
    # calendar
    "hour", "dayofweek", "month", "dayofyear", "is_holiday",
]
N_FEATURES = len(FEATURE_COLS)   # 25

SEQ_LEN  = 8760   # lookback (1년)
PRED_LEN = 24     # horizon (1일)
TOTAL_ROWS = 3 * 365 * 24        # 26280  (3년)
TRAIN_ROWS = 2 * 365 * 24        # 17520
VALID_ROWS = 183 * 24            # 4392
TEST_ROWS  = TOTAL_ROWS - TRAIN_ROWS - VALID_ROWS  # 4368


def make_dummy_df(n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2022-01-01 00:00:00")
    ts = pd.date_range(start, periods=n_rows, freq="h")

    data = {}
    for col in FEATURE_COLS:
        if col == "normalized_power":
            # 0~1 범위 (주간 높고 야간 0)
            hour_arr = ts.hour.values
            solar_mask = (hour_arr >= 6) & (hour_arr <= 18)
            v = rng.uniform(0, 1, n_rows)
            v[~solar_mask] = rng.uniform(0, 0.05, (~solar_mask).sum())
            data[col] = v.astype("float32")
        elif col in ("si", "clearsky_ghi", "solar_elevation"):
            hour_arr = ts.hour.values
            solar_mask = (hour_arr >= 6) & (hour_arr <= 18)
            v = rng.uniform(0, 1, n_rows)
            v[~solar_mask] = 0.0
            data[col] = v.astype("float32")
        elif col in ("pv_roll_mean_24h", "pv_roll_mean_72h", "pv_roll_mean_168h",
                     "pv_lag_24h", "pv_lag_168h"):
            data[col] = rng.uniform(0, 1, n_rows).astype("float32")
        elif col in ("pv_roll_std_24h", "pv_roll_std_72h", "pv_roll_std_168h"):
            data[col] = rng.uniform(0, 0.3, n_rows).astype("float32")
        elif col == "hour":
            data[col] = ts.hour.values.astype("float32")
        elif col == "dayofweek":
            data[col] = ts.dayofweek.values.astype("float32")
        elif col == "month":
            data[col] = ts.month.values.astype("float32")
        elif col == "dayofyear":
            data[col] = ts.dayofyear.values.astype("float32")
        elif col == "is_holiday":
            data[col] = rng.integers(0, 2, n_rows).astype("float32")
        elif col == "wd":
            data[col] = rng.uniform(0, 360, n_rows).astype("float32")
        elif col == "solar_azimuth":
            data[col] = rng.uniform(0, 360, n_rows).astype("float32")
        else:
            # 나머지 연속 feature: z-score 정규화 후 범위 모사
            data[col] = rng.normal(0, 1, n_rows).astype("float32")

    df = pd.DataFrame(data, index=ts)
    df.index.name = "timestamp"
    return df


def main():
    parser = argparse.ArgumentParser(description="더미 Feature Mart 생성")
    parser.add_argument("--rows", type=int, default=TOTAL_ROWS,
                        help=f"총 row 수 (기본값: {TOTAL_ROWS})")
    parser.add_argument("--out", type=str,
                        default="artifacts/dummy_feature_mart",
                        help="출력 디렉토리")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows = int(args.rows * TRAIN_ROWS / TOTAL_ROWS)
    valid_rows = int(args.rows * VALID_ROWS / TOTAL_ROWS)
    test_rows  = args.rows - train_rows - valid_rows

    os.makedirs(f"{args.out}/train", exist_ok=True)
    os.makedirs(f"{args.out}/valid", exist_ok=True)
    os.makedirs(f"{args.out}/test",  exist_ok=True)

    df_full = make_dummy_df(args.rows, seed=args.seed)

    splits = {
        "train": df_full.iloc[:train_rows],
        "valid": df_full.iloc[train_rows:train_rows + valid_rows],
        "test":  df_full.iloc[train_rows + valid_rows:],
    }
    for split_name, df in splits.items():
        out_path = f"{args.out}/{split_name}/site_dummy.parquet"
        df.to_parquet(out_path)
        print(f"  [{split_name}] {len(df):>6} rows × {len(df.columns)} cols → {out_path}")

    # scaler_stats (더미 — 모두 mean=0, std=1)
    scaler_stats = {
        "version": "dummy-1.0",
        "fit_end": "dummy",
        "features": {col: {"mean": 0.0, "std": 1.0} for col in FEATURE_COLS},
    }
    stats_path = f"{args.out}/scaler_stats.json"
    with open(stats_path, "w") as f:
        json.dump(scaler_stats, f, indent=2)

    print(f"\n✓ 더미 Feature Mart 생성 완료")
    print(f"  위치    : {args.out}/")
    print(f"  총 rows : {args.rows:,}  (train {train_rows:,} / valid {valid_rows:,} / test {test_rows:,})")
    print(f"  features: {N_FEATURES}  → {FEATURE_COLS}")
    print(f"  seq_len : {SEQ_LEN}  /  pred_len : {PRED_LEN}")


if __name__ == "__main__":
    main()
