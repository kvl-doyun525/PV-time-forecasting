"""
PV wide mart용 Dataset (단일 site parquet).

`scan_fcst_parquet_health.py`의 FEATURE_COLS와 동기화 유지.
복구: recup_dir.7/f567300984.h, track_b_mart_layout 문서 §2.6.7.
"""
from __future__ import annotations

import glob
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, Dataset

FEATURE_COLS: list[str] = [
    "normalized_power",
    "ta", "rn", "ws", "wd", "hm", "pa", "si", "ss", "dc10Tca",
    "solar_elevation", "solar_azimuth", "clearsky_ghi",
    "pv_roll_mean_24h", "pv_roll_mean_72h", "pv_roll_mean_168h",
    "pv_roll_std_24h", "pv_roll_std_72h", "pv_roll_std_168h",
    "pv_lag_24h", "pv_lag_168h",
    "hour", "dayofweek", "month", "dayofyear", "is_holiday",
    "t2m_c", "reh", "wsd", "vec", "tp_mm", "tcc",
]

TARGET_IDX = 0  # normalized_power

FUTURE_NWP_TO_FEATURE_COL_INDEX: dict[str, int] = {
    "tmp": FEATURE_COLS.index("t2m_c"),
    "reh": FEATURE_COLS.index("reh"),
    "wsd": FEATURE_COLS.index("wsd"),
    "vec": FEATURE_COLS.index("vec"),
    "sky": FEATURE_COLS.index("tcc"),
    "pcp": FEATURE_COLS.index("tp_mm"),
}


def encoder_input_channel_count(
    *,
    merge_future_nwp_into_encoder_input: bool = False,
) -> int:
    n = len(FEATURE_COLS)
    return n + 1 if merge_future_nwp_into_encoder_input else n


class SingleSiteDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        seq_len: int = 168,
        pred_len: int = 24,
        stride: int = 1,
        *,
        merge_future_nwp_into_encoder_input: bool = False,
        future_nwp_variable_names: Optional[tuple[str, ...]] = None,
    ) -> None:
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.merge = bool(merge_future_nwp_into_encoder_input)
        self.future_names: tuple[str, ...] = future_nwp_variable_names or (
            "tmp",
            "reh",
            "wsd",
            "vec",
            "sky",
            "pcp",
        )

        self.df = pd.read_parquet(parquet_path)
        if not isinstance(self.df.index, pd.DatetimeIndex):
            if "timestamp" in self.df.columns:
                self.df = self.df.set_index(pd.to_datetime(self.df["timestamp"]))
            else:
                raise ValueError(f"{parquet_path}: DatetimeIndex 또는 timestamp 컬럼 필요")
        self.df.sort_index(inplace=True)

        miss = [c for c in FEATURE_COLS if c not in self.df.columns]
        if miss:
            raise KeyError(f"{parquet_path}: FEATURE_COLS 누락 {miss[:8]}")

        need = self.seq_len + self.pred_len
        self.indices = list(range(0, len(self.df) - need + 1, self.stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        past = (
            self.df.iloc[i : i + self.seq_len][FEATURE_COLS]
            .to_numpy(dtype=np.float64)
        )
        past = np.nan_to_num(past, nan=0.0).astype(np.float32)
        y = (
            self.df.iloc[i + self.seq_len : i + self.seq_len + self.pred_len][FEATURE_COLS]
            .to_numpy(dtype=np.float64)
        )
        y = np.nan_to_num(y, nan=0.0).astype(np.float32)

        if not self.merge:
            return torch.from_numpy(past), torch.from_numpy(y)

        c = len(FEATURE_COLS)
        unified = np.zeros((self.seq_len + self.pred_len, c + 1), dtype=np.float32)
        unified[: self.seq_len, :c] = past
        unified[: self.seq_len, c] = 1.0

        t_end = i + self.seq_len - 1
        row_end = self.df.iloc[t_end]
        for h in range(1, self.pred_len + 1):
            r = self.seq_len + h - 1
            unified[r, c] = 0.0
            for var in self.future_names:
                colname = f"fcst_{var}_{h:03d}"
                if colname not in self.df.columns:
                    continue
                fi = FUTURE_NWP_TO_FEATURE_COL_INDEX[var]
                val = row_end[colname]
                unified[r, fi] = 0.0 if pd.isna(val) else float(val)

        return torch.from_numpy(unified), torch.from_numpy(y)


def build_multisite_dataset(
    feature_mart_dir: str,
    split: str,
    seq_len: int = 168,
    pred_len: int = 24,
    stride: int = 1,
    min_windows: int = 1,
    **dataset_kw: Any,
) -> ConcatDataset:
    pattern = os.path.join(feature_mart_dir, split, "*.parquet")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"parquet 파일 없음: {pattern}")

    datasets = []
    skipped = 0
    for p in paths:
        ds = SingleSiteDataset(
            p,
            seq_len=seq_len,
            pred_len=pred_len,
            stride=stride,
            **dataset_kw,
        )
        if len(ds) < min_windows:
            skipped += 1
            continue
        datasets.append(ds)

    print(
        f"[build_multisite_dataset] split={split}, "
        f"sites={len(datasets)}/{len(paths)}, skipped={skipped}, "
        f"total_windows={sum(len(d) for d in datasets):,}"
    )
    return ConcatDataset(datasets)


def load_test_windows(
    parquet_path: str,
    seq_len: int = 168,
    pred_len: int = 24,
    *,
    merge_future_nwp_into_encoder_input: bool = False,
    future_nwp_variable_names: Optional[tuple[str, ...]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = SingleSiteDataset(
        parquet_path,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=pred_len,
        merge_future_nwp_into_encoder_input=merge_future_nwp_into_encoder_input,
        future_nwp_variable_names=future_nwp_variable_names,
    )
    c_x = encoder_input_channel_count(
        merge_future_nwp_into_encoder_input=merge_future_nwp_into_encoder_input
    )
    seq_model = seq_len + pred_len if merge_future_nwp_into_encoder_input else seq_len
    if len(ds) == 0:
        return np.array([]), np.empty((0, seq_model, c_x)), np.empty((0, pred_len, 1))

    xs, ys, starts = [], [], []
    idx_vals = ds.df.index.values
    for j in range(len(ds)):
        x, y = ds[j]
        xs.append(x.numpy())
        ys.append(y[:, TARGET_IDX : TARGET_IDX + 1].numpy())
        ts_start = idx_vals[ds.indices[j] + seq_len]
        starts.append(ts_start)

    return np.array(starts), np.stack(xs), np.stack(ys)
