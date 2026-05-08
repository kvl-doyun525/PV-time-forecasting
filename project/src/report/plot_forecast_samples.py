#!/usr/bin/env python3
"""
테스트 예측 parquet(`pred_h*`) + feature_mart_per_site/test 의 실측을 합쳐
입력·실측·예측 시계열 샘플 플롯을 생성한다.

복구: PhotoRec 텍스트 조각(`recup_dir.7/f567349552.txt` 등) + 에이전트 대화 기록과 정합.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _loc_int(idx_loc: int | slice | np.ndarray | pd.Index) -> int:
    if isinstance(idx_loc, slice):
        return int(idx_loc.start or 0)
    a = np.asarray(idx_loc).ravel()
    return int(a[0])


def _series_from_row(
    row: pd.Series,
    feature_mart: str,
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    site_id = str(row["site_id"])
    ts = pd.Timestamp(row["timestamp"])
    test_path = os.path.join(feature_mart, "test", f"{site_id}.parquet")
    test_df = (
        pd.read_parquet(test_path)[["normalized_power"]].ffill().fillna(0.0).sort_index()
    )
    pos = _loc_int(test_df.index.get_loc(ts))

    act = test_df["normalized_power"].iloc[pos : pos + horizon].values.astype(float)
    if len(act) != horizon:
        raise ValueError(f"site={site_id} ts={ts}: actual 길이 불일치 ({len(act)}/{horizon})")

    if pos >= seq_len:
        inp = test_df["normalized_power"].iloc[pos - seq_len : pos].values.astype(float)
    else:
        need_train = seq_len - pos
        train_path = os.path.join(feature_mart, "train", f"{site_id}.parquet")
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"입력 길이 부족(pos={pos} < seq_len={seq_len})인데 train 파일 없음: {train_path}"
            )
        train_df = pd.read_parquet(train_path)[["normalized_power"]].ffill().fillna(0.0).sort_index()
        if len(train_df) < need_train:
            raise ValueError(f"site={site_id}: train 행 부족 ({len(train_df)} < 필요 {need_train})")
        tail_tr = train_df["normalized_power"].iloc[-need_train:].values.astype(float)
        head_te = (
            test_df["normalized_power"].iloc[:pos].values.astype(float)
            if pos > 0
            else np.array([], dtype=float)
        )
        inp = np.concatenate([tail_tr, head_te])
    if len(inp) != seq_len:
        raise ValueError(f"site={site_id} ts={ts}: input 길이 불일치 ({len(inp)}/{seq_len})")
    return inp, act, ts


def _pred_from_row(row: pd.Series, horizon: int) -> np.ndarray:
    return np.array([float(row[f"pred_h{i}"]) for i in range(horizon)], dtype=float)


def _maybe_scale(arr: np.ndarray, capacity_kw: float | None) -> np.ndarray:
    if capacity_kw is None or capacity_kw <= 0:
        return arr
    return arr * capacity_kw


def _match_row(df: pd.DataFrame, site_id: str, ts: pd.Timestamp) -> pd.Series | None:
    ts = pd.Timestamp(ts)
    hit = df[
        (df["site_id"].astype(str) == str(site_id))
        & (pd.to_datetime(df["timestamp"], utc=False) == ts)
    ]
    if hit.empty:
        return None
    return hit.iloc[0]


def plot_one_sample(
    ax: plt.Axes,
    sample_idx: int,
    row: pd.Series,
    feature_mart: str,
    seq_len: int,
    horizon: int,
    compare_rows: list[tuple[str, pd.Series | None]],
    capacity_kw: float | None,
    *,
    time_axis: bool = False,
) -> None:
    inp, act, first_ts = _series_from_row(row, feature_mart, seq_len, horizon)
    pred_main = _pred_from_row(row, horizon)

    inp_y = _maybe_scale(inp, capacity_kw)
    act_y = _maybe_scale(act, capacity_kw)
    pred_y = _maybe_scale(pred_main, capacity_kw)

    first_ts = pd.Timestamp(first_ts)
    if time_axis:
        x_in = pd.date_range(end=first_ts - pd.Timedelta(hours=1), periods=seq_len, freq="h")
        x_fc = pd.date_range(start=first_ts, periods=horizon, freq="h")
        vline_x = first_ts
    else:
        x_in = np.arange(seq_len)
        x_fc = np.arange(seq_len, seq_len + horizon)
        vline_x = seq_len - 0.5

    ax.plot(x_in, inp_y, "o-", color="C0", markersize=3, linewidth=1.2, label="input")
    ax.plot(x_fc, act_y, "o-", color="C2", markersize=4, linewidth=1.2, label="actual")
    ax.plot(x_fc, pred_y, "x-", color="C1", markersize=5, linewidth=1.2, label="prediction")

    colors_cmp = ["C3", "C4", "C5", "C6"]
    for k, (label, cr) in enumerate(compare_rows):
        if cr is None:
            continue
        p = _pred_from_row(cr, horizon)
        py = _maybe_scale(p, capacity_kw)
        ax.plot(
            x_fc,
            py,
            "x--",
            color=colors_cmp[k % len(colors_cmp)],
            markersize=4,
            linewidth=1.0,
            label=f"prediction ({label})",
        )

    ax.axvline(vline_x, color="red", linestyle="--", linewidth=1.0, alpha=0.85)
    y_label = "Active power (kW)" if capacity_kw else "Normalized power (0–1)"
    if time_axis:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")
        ax.set_xlabel("Time (forecast start = red line)")
    else:
        ax.set_xlabel("Time index")
    ax.set_ylabel(y_label)
    site = row["site_id"]
    ax.set_title(f"Predictions vs Actuals for Sample {sample_idx} — site {site} @ {row['timestamp']}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="test 샘플 예측 vs 실측 플롯")
    parser.add_argument("--feature-mart", default="artifacts/feature_mart_per_site")
    parser.add_argument("--predictions", required=True, help="predictions_test_{h}h.parquet 경로")
    parser.add_argument("--compare-predictions", nargs="*", default=[], help="추가 예측 parquet (비교 곡선)")
    parser.add_argument("--compare-label", nargs="*", default=[], help="비교 곡선 범례 이름 (순서 일치)")
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--n-plots", type=int, default=5, help="생성할 샘플 수")
    parser.add_argument(
        "--row-offset",
        type=int,
        default=0,
        help="predictions parquet에서 시작 행 오프셋 (정렬 후, --indices 미사용 시)",
    )
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="플롯할 행 번호 (정렬 후 0-base). 예: 5 12 20 — 이미지의 Sample 5는 --indices 5",
    )
    parser.add_argument("--out-dir", default="artifacts/plots/forecast_samples")
    parser.add_argument("--capacity-kw", type=float, default=None, help="정격(kW). 주면 축을 kW로 스케일")
    parser.add_argument(
        "--time-axis",
        action="store_true",
        help="x축을 예측 시작 시각 기준 실제 일시(시간별)로 표시",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--prefix", default="", help="파일명 접두사")
    args = parser.parse_args()

    pred_path = args.predictions
    if not os.path.exists(pred_path):
        raise FileNotFoundError(pred_path)

    df_main = pd.read_parquet(pred_path).sort_values(["site_id", "timestamp"]).reset_index(drop=True)

    cmp_dfs: list[pd.DataFrame] = []
    for p in args.compare_predictions:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        cmp_dfs.append(pd.read_parquet(p))

    labels_cmp = list(args.compare_label)
    if len(labels_cmp) < len(cmp_dfs):
        labels_cmp = labels_cmp + [f"model{i}" for i in range(len(labels_cmp), len(cmp_dfs))]

    stem = Path(pred_path).parent.name
    title_base = args.prefix or stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.indices is not None:
        plot_indices = [(k, ix) for k, ix in enumerate(args.indices) if 0 <= ix < len(df_main)]
    else:
        plot_indices = []
        for i in range(args.n_plots):
            idx = args.row_offset + i
            if idx >= len(df_main):
                break
            plot_indices.append((i, idx))

    for seq_k, idx in plot_indices:
        row = df_main.iloc[idx]

        compare_rows: list[tuple[str, pd.Series | None]] = []
        for j, cdf in enumerate(cmp_dfs):
            cr = _match_row(cdf, str(row["site_id"]), row["timestamp"])
            compare_rows.append((labels_cmp[j], cr))

        fig, ax = plt.subplots(figsize=(10, 4.5))
        plot_one_sample(
            ax,
            sample_idx=idx,
            row=row,
            feature_mart=args.feature_mart,
            seq_len=args.seq_len,
            horizon=args.horizon,
            compare_rows=compare_rows,
            capacity_kw=args.capacity_kw,
            time_axis=args.time_axis,
        )
        fig.tight_layout()
        fname = out_dir / f"{title_base}_row{idx}_site_{row['site_id']}.png"
        fig.savefig(fname, dpi=args.dpi)
        plt.close(fig)
        print(f"저장: {fname}")

    print(f"완료: {len(plot_indices)}개 PNG → {out_dir}")


if __name__ == "__main__":
    main()
