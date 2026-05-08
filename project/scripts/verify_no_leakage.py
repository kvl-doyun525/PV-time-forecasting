#!/usr/bin/env python3
"""
split_manifest.yaml 의 날짜 경계와 feature_mart per-site parquet 인덱스를 대조한다.

`pv_model_benchmark_execution.md` §4.2 — forecast join 전용 컬럼이 없을 때도
**시간 분할 누수** 1차 검증에 사용한다.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


def _load_manifest(path: Path) -> dict:
    if yaml is None:
        raise SystemExit("PyYAML 필요: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="split 경계 vs mart 인덱스 검사")
    ap.add_argument(
        "--feature-mart",
        type=Path,
        default=Path("artifacts/feature_mart_per_site"),
        help="per-site mart 루트 (train/valid/test)",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/split_manifest.yaml"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/leakage_check_result.json"),
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    mart = args.feature_mart if args.feature_mart.is_absolute() else root / args.feature_mart
    man_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    out_path = args.output if args.output.is_absolute() else root / args.output

    mf = _load_manifest(man_path)
    sp = mf.get("split") or {}
    train_end = pd.Timestamp(sp.get("train_end"))
    valid_end = pd.Timestamp(sp.get("valid_end"))
    test_end = pd.Timestamp(sp.get("test_end"))
    data_start = pd.Timestamp(mf.get("data", {}).get("data_start"))

    violations: list[dict] = []

    def check_split(name: str, sub: str, lo: pd.Timestamp | None, hi: pd.Timestamp | None) -> None:
        ddir = mart / sub
        if not ddir.is_dir():
            return
        for path in glob.glob(str(ddir / "*.parquet")):
            raw = pd.read_parquet(path)
            if not isinstance(raw.index, pd.DatetimeIndex):
                if "timestamp" in raw.columns:
                    idx = pd.to_datetime(raw["timestamp"])
                else:
                    violations.append(
                        {"file": path, "error": "DatetimeIndex 또는 timestamp 없음"}
                    )
                    continue
            else:
                idx = raw.index
            mn, mx = pd.Timestamp(idx.min()), pd.Timestamp(idx.max())
            if lo is not None and mn < lo:
                violations.append(
                    {
                        "file": path,
                        "split": name,
                        "kind": "min_before_lower_bound",
                        "min": str(mn),
                        "bound": str(lo),
                    }
                )
            if hi is not None and mx > hi:
                violations.append(
                    {
                        "file": path,
                        "split": name,
                        "kind": "max_after_upper_bound",
                        "max": str(mx),
                        "bound": str(hi),
                    }
                )

    check_split("train", "train", data_start, train_end)
    check_split("valid", "valid", train_end + pd.Timedelta(seconds=1), valid_end)
    check_split("test", "test", valid_end + pd.Timedelta(seconds=1), test_end)

    ok = len(violations) == 0
    report = {
        "pass": ok,
        "manifest": str(man_path),
        "mart": str(mart),
        "n_violations": len(violations),
        "violations": violations[:200],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[verify_no_leakage] pass={ok} → {out_path}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
