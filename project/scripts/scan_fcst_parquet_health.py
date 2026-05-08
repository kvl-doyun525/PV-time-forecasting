#!/usr/bin/env python3
"""
feature_mart wide Parquet의 fcst_* / NWP fan 적재를 pv_dataset.SingleSiteDataset과
동일한 방식으로 시뮬레이션하고 문제 사이트·열을 찾는다 (torch 불필요).

학습 코드는 `pd.read_parquet(path)` 전체를 읽으므로, 본 스크립트도 기본적으로
전체 DataFrame으로 검사한다 (한 파일당 1회 읽기, pred_len 여러 개는 동일 df로 처리).

  python3 scripts/scan_fcst_parquet_health.py \\
    --mart artifacts/feature_mart_track_b_per_site \\
    --splits train valid test \\
    --pred-lens 24 48 72

pv_dataset.FEATURE_COLS / DEFAULT NWP 이름과 동기화 유지할 것.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── pv_dataset.py 와 동일 (수정 시 양쪽 맞출 것) ─────────────────
FEATURE_COLS = [
    "normalized_power",
    "ta", "rn", "ws", "wd", "hm", "pa", "si", "ss", "dc10Tca",
    "solar_elevation", "solar_azimuth", "clearsky_ghi",
    "pv_roll_mean_24h", "pv_roll_mean_72h", "pv_roll_mean_168h",
    "pv_roll_std_24h", "pv_roll_std_72h", "pv_roll_std_168h",
    "pv_lag_24h", "pv_lag_168h",
    "hour", "dayofweek", "month", "dayofyear", "is_holiday",
    "t2m_c", "reh", "wsd", "vec", "tp_mm", "tcc",
]
def fcst_column_list(names: tuple[str, ...], pred_len: int) -> list[str]:
    out: list[str] = []
    for name in names:
        for h in range(1, pred_len + 1):
            out.append(f"fcst_{name}_{h:03d}")
    return out


@dataclass
class SitePredReport:
    site_id: str
    path: str
    pred_len: int
    issues: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        return len(self.issues) == 0


def scan_parquet_schema_duplicates(path: str) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return []
    sch = pq.read_schema(path)
    seen: dict[str, int] = {}
    for n in sch.names:
        seen[n] = seen.get(n, 0) + 1
    return [n for n, c in seen.items() if c > 1]


def _simulate_fan_for_pred_len(
    df: pd.DataFrame,
    pred_len: int,
    names: tuple[str, ...],
    *,
    site_id: str,
    exhaustive_window_check: bool = False,
) -> SitePredReport:
    rep = SitePredReport(site_id=site_id, path="", pred_len=pred_len)
    fcst_cols = fcst_column_list(names, pred_len)
    need = FEATURE_COLS + fcst_cols
    missing = [c for c in need if c not in df.columns]
    if missing:
        rep.issues.append(f"누락 컬럼 {len(missing)}개: {missing[:15]}...")
        return rep

    for c in fcst_cols:
        s = df[c]
        if s.dtype == object:
            sample = s.dropna().head(500)
            bad: list[str] = []
            for v in sample:
                if isinstance(v, type):
                    bad.append(f"type:{v.__name__}")
                elif not isinstance(v, (bool, int, float, np.floating, np.integer)):
                    bad.append(type(v).__name__)
            tail = f" 예: {bad[:10]}" if bad else ""
            rep.issues.append(f"object dtype 열 {c}{tail}")

    sub = df[need].copy()
    sub = sub.ffill().fillna(0.0)
    T = len(sub)
    H, V = pred_len, len(names)
    fan = np.zeros((T, H, V), dtype=np.float32)
    for vi, name in enumerate(names):
        for h in range(H):
            col = f"fcst_{name}_{h + 1:03d}"
            raw = sub[col].to_numpy(dtype=np.float32, copy=False)
            if raw.dtype == object:
                rep.issues.append(f"to_numpy(float32) 후 object: {col}")
                continue
            fan[:, h, vi] = raw

    if fan.dtype != np.float32:
        rep.issues.append(f"_fcst_fan dtype={fan.dtype} (기대 float32)")

    if fan.dtype == np.float32 and not np.isfinite(fan).all():
        rep.issues.append(f"비유한값 {int(np.sum(~np.isfinite(fan)))}개")

    if fan.dtype == np.float32 and T > 0 and H > 0 and V > 0:
        L_default = 168
        if T >= L_default + H:
            for t_end in (0, T // 2, min(T - 1, T - H - 1)):
                slab = fan[t_end]
                for k in range(H):
                    for vi in range(V):
                        v = slab[k, vi]
                        try:
                            raw = float(v)
                        except TypeError as e:
                            rep.issues.append(
                                f"fan[{t_end},{k},{vi}] float 실패: {e}; value={v!r}"
                            )
                            return rep
                        nwp_name = names[vi]
                        try:
                            if nwp_name == "sky":
                                float((raw - 1.0) / 3.0)
                            else:
                                float(raw)
                        except TypeError as e:
                            rep.issues.append(
                                f"_fan_value_to_feature_col({nwp_name!r}, …) 실패: {e}; raw={raw!r}"
                            )
                            return rep
        if exhaustive_window_check:
            err = _exhaustive_float_probe(fan, pred_len, names, seq_len=168)
            if err:
                rep.issues.append(f"[전체 윈도우] {err}")
    return rep


def _exhaustive_float_probe(
    fan: np.ndarray,
    pred_len: int,
    names: tuple[str, ...],
    seq_len: int,
) -> str | None:
    """모든 윈도우의 t_end에 대해 float(fan[t_end,k,vi]) 검사. fan은 (T,H,V) float32."""
    T, H, V = fan.shape
    L = seq_len
    min_len = L + pred_len
    if T < min_len:
        return None
    for start in range(0, T - min_len + 1, 1):
        t_end = start + L - 1
        slab = fan[t_end]
        for k in range(H):
            for vi in range(V):
                v = slab[k, vi]
                try:
                    raw = float(v)
                except TypeError as e:
                    return f"idx start={start} t_end={t_end} k={k} vi={vi} name={names[vi]}: {e}; v={v!r}"
                name = names[vi]
                try:
                    if name == "sky":
                        float((raw - 1.0) / 3.0)
                    else:
                        float(raw)
                except TypeError as e:
                    return (
                        f"idx start={start} t_end={t_end} k={k} vi={vi} "
                        f"name={names[vi]!r}: {e}; raw={raw!r}"
                    )
    return None


def scan_file_once(
    path: str,
    pred_lens: list[int],
    names: tuple[str, ...],
    *,
    check_column_subset: bool,
    exhaustive_window_check: bool = False,
) -> list[SitePredReport]:
    site_id = os.path.splitext(os.path.basename(path))[0]
    reports: list[SitePredReport] = []

    dups = scan_parquet_schema_duplicates(path)
    dup_msg = f"parquet 스키마 중복 열: {dups[:15]}" if dups else ""

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        for pl in pred_lens:
            r = SitePredReport(site_id=site_id, path=path, pred_len=pl)
            r.issues.append(f"read_parquet 실패: {e}")
            if dup_msg:
                r.issues.append(dup_msg)
            reports.append(r)
        return reports

    prefix: list[str] = []
    if dup_msg:
        prefix.append(dup_msg)

    if check_column_subset:
        max_pl = max(pred_lens)
        fcst_all = fcst_column_list(names, max_pl)
        need_all = FEATURE_COLS + fcst_all
        miss_all = [c for c in need_all if c not in df.columns]
        if not miss_all:
            try:
                df_sub = pd.read_parquet(path, columns=need_all)
            except Exception:
                pass
            else:
                sub_miss = [c for c in need_all if c not in df_sub.columns]
                if sub_miss:
                    prefix.append(
                        f"부분읽기(columns={len(need_all)}) 후 누락: {sub_miss[:12]}..."
                    )
                    prefix.append(
                        "(학습은 전체 read를 사용하므로 당장 학습 실패 원인은 아닐 수 있음)"
                    )

    for pl in pred_lens:
        r = _simulate_fan_for_pred_len(
            df,
            pl,
            names,
            site_id=site_id,
            exhaustive_window_check=exhaustive_window_check,
        )
        r.path = path
        r.issues = prefix + r.issues
        reports.append(r)
    return reports


def main() -> None:
    p = argparse.ArgumentParser(description="fcst_* / NWP fan Parquet 건강 검사")
    p.add_argument("--mart", default="artifacts/feature_mart_track_b_per_site")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    p.add_argument("--pred-lens", nargs="+", type=int, default=[24, 48, 72])
    p.add_argument(
        "--future-nwp-names",
        default="tmp,reh,wsd,vec,sky,pcp",
        help="콤마 구분, 학습 스크립트와 동일하게",
    )
    p.add_argument(
        "--no-column-subset-check",
        action="store_true",
        help="pd.read_parquet(columns=...) 정합성 검사 생략",
    )
    p.add_argument(
        "--exhaustive-window-check",
        action="store_true",
        help="모든 슬라이딩 윈도우(start)에 대해 fan[t_end] float 경로 검사 (느림)",
    )
    p.add_argument("--json-out", default="", help="요약 JSON 저장 경로")
    args = p.parse_args()

    names = tuple(x.strip() for x in args.future_nwp_names.split(",") if x.strip())
    mart = args.mart
    if not os.path.isdir(mart):
        print(f"[fatal] 마트 디렉터리 없음: {mart}", file=sys.stderr)
        sys.exit(1)

    all_bad: list[dict] = []
    files = 0
    for split in args.splits:
        pattern = os.path.join(mart, split, "*.parquet")
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"[warn] 파일 없음: {pattern}")
            continue
        print(f"\n=== split={split} files={len(paths)} pred_lens={args.pred_lens} ===", flush=True)
        bad_here = 0
        for path in paths:
            files += 1
            reps = scan_file_once(
                path,
                args.pred_lens,
                names,
                check_column_subset=not args.no_column_subset_check,
                exhaustive_window_check=args.exhaustive_window_check,
            )
            for rep in reps:
                if not rep.ok():
                    bad_here += 1
                    print(f"  [BAD] {rep.site_id} H={rep.pred_len}:", flush=True)
                    for line in rep.issues[:10]:
                        print(f"        {line}", flush=True)
                    all_bad.append(
                        {
                            "site_id": rep.site_id,
                            "split": split,
                            "pred_len": rep.pred_len,
                            "path": rep.path,
                            "issues": rep.issues,
                        }
                    )
        print(f"  split 요약: 문제 레코드 {bad_here} (site×H 조합)", flush=True)

    summary = {
        "mart": os.path.abspath(mart),
        "splits": args.splits,
        "pred_lens": args.pred_lens,
        "future_nwp_names": names,
        "parquet_files_read": files,
        "problem_entries": len(all_bad),
        "problems": all_bad,
    }
    if args.json_out:
        parent = os.path.dirname(args.json_out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nJSON 저장: {args.json_out}")

    if all_bad:
        print(f"\n총 문제 레코드: {len(all_bad)}", file=sys.stderr)
        sys.exit(2)
    print("\n스캔 완료: 보고된 문제 없음.")
    sys.exit(0)


if __name__ == "__main__":
    main()
