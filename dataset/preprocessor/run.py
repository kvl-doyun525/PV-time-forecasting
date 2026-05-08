#!/usr/bin/env python3
"""
PV Feature Mart 전처리 CLI.

복구: recup_dir.7/f567523592.txt, f567522592.txt, f568504408.txt.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyYAML 필요: pip install pyyaml") from e

from config import (
    DATA_END,
    DATA_START,
    FEATURE_MART,
    FEATURE_MART_PER_SITE,
    MANIFEST_PATH,
    PreprocessConfig,
    SNAPSHOT_DIR,
    TEST_END,
    TRACK_B_ERA5_NATIVE_FCST_COLS,
    TRACK_B_HORIZON_MAX,
    TRACK_B_SERVICE_FCST_COLS,
    TRAIN_END,
    VALID_END,
)

log = logging.getLogger(__name__)


def load_manifest() -> dict:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            manifest,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def cmd_build(args: argparse.Namespace) -> None:
    from feature_mart_builder import build_feature_mart

    snap = Path(args.snapshot_dir)
    if args.output_dir:
        out_path = Path(args.output_dir)
    else:
        out_path = (
            FEATURE_MART_PER_SITE
            if args.split_mode == "per_site"
            else FEATURE_MART
        )
    if args.split_mode == "global":
        cfg = PreprocessConfig(
            snapshot_dir=snap,
            feature_mart_dir=out_path,
            global_time_start=args.global_time_start,
            global_time_end=args.global_time_end,
            train_end=args.train_end,
            valid_end=args.valid_end,
            test_end=args.test_end,
            max_interp_gap_hours=args.max_gap,
            min_site_coverage=args.min_coverage,
            use_era5=not args.no_era5,
            split_mode=args.split_mode,
            split_ratios=tuple(args.split_ratios),
            min_split_hours=args.min_split_hours,
        )
    else:
        cfg = PreprocessConfig(
            snapshot_dir=snap,
            feature_mart_per_site_dir=out_path,
            global_time_start=args.global_time_start,
            global_time_end=args.global_time_end,
            train_end=args.train_end,
            valid_end=args.valid_end,
            test_end=args.test_end,
            max_interp_gap_hours=args.max_gap,
            min_site_coverage=args.min_coverage,
            use_era5=not args.no_era5,
            split_mode=args.split_mode,
            split_ratios=tuple(args.split_ratios),
            min_split_hours=args.min_split_hours,
        )
    build_feature_mart(cfg=cfg, site_ids=args.sites)
    log.info("build 완료: split_mode=%s output=%s", args.split_mode, out_path)


def cmd_update_manifest(_args: argparse.Namespace) -> None:
    manifest = load_manifest()
    manifest.setdefault("data", {})
    manifest["data"]["data_start"] = DATA_START
    manifest["data"]["data_end"] = DATA_END
    manifest.setdefault("split", {})
    manifest["split"]["train_end"] = TRAIN_END
    manifest["split"]["valid_end"] = VALID_END
    manifest["split"]["test_end"] = TEST_END
    manifest["created_at"] = datetime.now().strftime("%Y-%m-%d")
    manifest.setdefault("horizons", [24, 48, 72])
    manifest.setdefault("lookback", 8760)
    manifest.setdefault(
        "inference_schedule",
        {"daily_run_time": "05:00", "forecast_issue_lag_hours": 1},
    )
    manifest.setdefault("seeds", [42, 123, 2024])
    manifest.setdefault(
        "evaluation",
        {
            "primary_metrics": ["daytime_MAE", "daytime_nRMSE", "daily_energy_error"],
            "secondary_metrics": ["MAE", "RMSE", "nRMSE", "sMAPE"],
            "daytime_definition": "solar_elevation > 5deg",
            "normalization_base": "capacity_kw",
        },
    )
    save_manifest(manifest)
    log.info("split_manifest.yaml 갱신 완료: %s", MANIFEST_PATH)
    print(f"data_start : {DATA_START}")
    print(f"data_end   : {DATA_END}")
    print(f"train_end  : {TRAIN_END}")
    print(f"valid_end  : {VALID_END}")
    print(f"test_end   : {TEST_END}")


def cmd_quality_check(args: argparse.Namespace) -> None:
    """
    Feature mart 데이터 품질 리포트 출력.

    PV 데이터 특성 반영 기준
    -------------------------
    - 결측률: 야간 NaN은 정상(인버터 꺼짐). 전체 기준 60%로 완화.
              daytime(solar_elevation > 5°) 한정 결측률은 30% 이하 요구.
    - 시간 수: train ≥ 8760h(1년), valid/test ≥ 4380h(6개월)
    - 낮 시간대 0값: PV 설비 고장 판별. daytime 중 normalized_power ≈ 0 비율 < 50%.
    """
    import pandas as pd  # noqa: PLC0415

    fm_dir = Path(args.feature_mart_dir)

    build_report_path = fm_dir / "build_report.json"
    split_mode = "global"
    min_split_hours_from_report = 500
    if build_report_path.exists():
        with open(build_report_path, encoding="utf-8") as _f:
            _br = json.load(_f)
        split_mode = _br.get("split_mode", "global")
        min_split_hours_from_report = _br.get("min_split_hours", 500)

    report: dict = {
        "checked_at": datetime.now().isoformat(),
        "split_mode": split_mode,
        "note": (
            "PV 인버터는 야간에 꺼져 데이터를 미전송(NaN)함. "
            "전체 결측률이 높은 것은 정상이며 daytime 결측률로 품질 판단."
        ),
        "splits": {},
        "issues": [],
        "warnings": [],
    }

    PASS_MISSING_TOTAL = 0.60
    PASS_MISSING_DAYTIME = 0.30
    PASS_DAYTIME_ZEROS = 0.50

    if split_mode == "per_site":
        MIN_HOURS = {
            "train": max(min_split_hours_from_report, 500),
            "valid": max(min_split_hours_from_report, 500),
            "test": max(min_split_hours_from_report, 500),
        }
    else:
        MIN_HOURS = {"train": 8760, "valid": 4380, "test": 4380}

    for split in ("train", "valid", "test"):
        split_dir = fm_dir / split
        if not split_dir.exists():
            report["issues"].append(f"{split} 디렉터리 없음")
            continue

        files = sorted(split_dir.glob("*.parquet"))
        if not files:
            report["issues"].append(f"{split} parquet 파일 없음")
            continue

        n_sites = len(files)
        total_rows = 0
        total_nan = 0
        total_cells = 0
        daytime_nan_sum = 0
        daytime_total_sum = 0
        daytime_zero_sum = 0

        for f in files:
            df = pd.read_parquet(f)
            total_rows += len(df)
            total_cells += df.size
            total_nan += df.isnull().sum().sum()

            if "solar_elevation" in df.columns and "normalized_power" in df.columns:
                valid_mask = df["normalized_power"].notna()
                first_valid = df.index[valid_mask].min() if valid_mask.any() else None
                last_valid = df.index[valid_mask].max() if valid_mask.any() else None

                if first_valid is not None:
                    active = df.loc[first_valid:last_valid]
                    daytime = active[active["solar_elevation"] > 0]
                    if len(daytime) > 0:
                        daytime_nan_sum += int(daytime["normalized_power"].isnull().sum())
                        daytime_zero_sum += int((daytime["normalized_power"].abs() < 0.01).sum())
                        daytime_total_sum += len(daytime)

        missing_rate = total_nan / max(total_cells, 1)
        avg_hours = total_rows / max(n_sites, 1)
        daytime_missing = daytime_nan_sum / max(daytime_total_sum, 1)
        daytime_zeros = daytime_zero_sum / max(daytime_total_sum, 1)

        split_pass = True
        min_h = MIN_HOURS.get(split, 4380)

        if missing_rate > PASS_MISSING_TOTAL:
            report["issues"].append(
                f"[{split}] 전체 결측률 {missing_rate:.1%} > 기준 {PASS_MISSING_TOTAL:.0%}"
            )
            split_pass = False
        elif missing_rate > 0.40:
            report["warnings"].append(
                f"[{split}] 전체 결측률 {missing_rate:.1%} (야간 NaN 포함 정상 범주)"
            )

        if daytime_missing > PASS_MISSING_DAYTIME:
            report["issues"].append(
                f"[{split}] 낮 시간대 결측률 {daytime_missing:.1%} > 기준 {PASS_MISSING_DAYTIME:.0%}"
            )
            split_pass = False

        if avg_hours < min_h:
            report["issues"].append(
                f"[{split}] 평균 시간 {avg_hours:.0f}h < 기준 {min_h}h"
            )
            split_pass = False

        if daytime_zeros > PASS_DAYTIME_ZEROS:
            report["issues"].append(
                f"[{split}] 낮 시간대 0값 비율 {daytime_zeros:.1%} > 기준 {PASS_DAYTIME_ZEROS:.0%}"
                " (설비 이상 의심)"
            )
            split_pass = False

        report["splits"][split] = {
            "n_sites": n_sites,
            "total_rows": total_rows,
            "avg_hours_per_site": avg_hours,
            "missing_rate_total": round(missing_rate, 4),
            "missing_rate_daytime": round(daytime_missing, 4),
            "daytime_zero_ratio": round(daytime_zeros, 4),
            "pass": split_pass,
        }

    overall_pass = len(report["issues"]) == 0
    report["overall_pass"] = overall_pass

    out_path = fm_dir / "quality_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if overall_pass:
        print("\n✓ 품질 검사 통과")
    else:
        print(f"\n✗ 품질 검사 실패: {len(report['issues'])}건 이슈")
        sys.exit(1)


def cmd_enrich_track_b(args: argparse.Namespace) -> None:
    """트랙 B: ERA5(계열) NWP covariate를 기존 mart에 조인해 별도 디렉터리에 저장."""
    from track_b_enrich_mart import (  # noqa: PLC0415
        default_output_dir_for_input,
        enrich_track_b_mart,
    )

    in_dir = Path(args.input_mart_dir)
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir_for_input(in_dir)
    fc_raw = getattr(args, "forecast_parquet", "") or ""
    fc_path = Path(fc_raw) if fc_raw.strip() else None

    value_cols = (
        [c.strip() for c in args.fcst_cols.split(",") if c.strip()]
        if args.fcst_cols
        else (
            list(TRACK_B_SERVICE_FCST_COLS)
            if args.fcst_schema == "shortterm_aligned"
            else list(TRACK_B_ERA5_NATIVE_FCST_COLS)
        )
    )

    log.info(
        "트랙 B enrich: input=%s output=%s join_mode=%s fcst_schema=%s forecast=%s",
        in_dir,
        out_dir,
        args.join_mode,
        args.fcst_schema,
        fc_path,
    )
    report = enrich_track_b_mart(
        in_dir,
        out_dir,
        join_mode=args.join_mode,
        fcst_schema=args.fcst_schema,
        forecast_path=fc_path,
        horizon_max=args.horizon_max,
        value_cols=value_cols,
        site_col=args.site_col,
        site_ids=args.sites,
    )
    print("\n=== 트랙 B enrich 완료 ===")
    print(f"  join_mode     : {report['join_mode']}")
    print(f"  fcst_schema   : {report['fcst_schema']}")
    print(f"  출력 경로     : {out_dir}")
    print(f"  parquet 수   : {report['n_parquet_written']}")
    print(f"  총 행 수      : {report['total_rows']:,}")
    print(f"  예보/ERA5 파일: {report['forecast_path']}")
    print(f"  추가 컬럼 수  : {len(report['added_columns'])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PV Feature Mart 전처리 파이프라인",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="feature mart 전체 빌드")
    p_build.add_argument(
        "--sites", type=int, nargs="+", default=None,
        help="처리할 cid_seq 목록 (미지정 시 전체)",
    )
    p_build.add_argument(
        "--snapshot-dir", default=str(SNAPSHOT_DIR),
        help="dataset_snapshot 디렉터리 경로",
    )
    p_build.add_argument(
        "--output-dir",
        default=None,
        help=(
            "feature mart 출력 루트 (미지정 시 per_site → project/artifacts/feature_mart_per_site, "
            "global → project/artifacts/feature_mart)"
        ),
    )
    p_build.add_argument("--global-time-start", default=DATA_START)
    p_build.add_argument("--global-time-end", default=DATA_END)
    p_build.add_argument("--train-end", default=TRAIN_END)
    p_build.add_argument("--valid-end", default=VALID_END)
    p_build.add_argument("--test-end", default=TEST_END)
    p_build.add_argument("--no-era5", action="store_true", help="ERA5 feature 비활성화")
    p_build.add_argument("--max-gap", type=int, default=1, help="보간 최대 연속 결측 (시간)")
    p_build.add_argument("--min-coverage", type=float, default=0.1, help="site 최소 유효 데이터 비율")
    p_build.add_argument(
        "--split-mode", choices=["global", "per_site"], default="global",
        help=(
            "split 방식: "
            "global=전체 공통 날짜 경계, "
            "per_site=site별 활성 기간에 비율 적용 (연도 무관 계절성 학습에 적합)"
        ),
    )
    p_build.add_argument(
        "--split-ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15],
        metavar=("TRAIN", "VALID", "TEST"),
        help="per_site 모드 train/valid/test 비율 (합산 1.0)",
    )
    p_build.add_argument(
        "--min-split-hours", type=int, default=500,
        help="per_site 모드: 각 split당 최소 시간 수 (미달 site 제외)",
    )

    sub.add_parser("update-manifest", help="split_manifest.yaml 날짜 갱신")

    p_qc = sub.add_parser("quality-check", help="feature mart 품질 검사")
    p_qc.add_argument(
        "--feature-mart-dir", default=str(FEATURE_MART),
        help="feature mart 디렉터리",
    )

    p_tb = sub.add_parser(
        "enrich-track-b",
        help="트랙 B: ERA5(계열) 미래 NWP covariate(fcst_*)를 mart에 추가",
    )
    p_tb.add_argument(
        "--input-mart-dir",
        default=str(FEATURE_MART_PER_SITE),
        help="기존 feature mart 디렉터리 (train/valid/test 포함)",
    )
    p_tb.add_argument(
        "--output-dir",
        default="",
        help="출력 루트 (미지정 시 per_site 입력이면 feature_mart_track_b_per_site)",
    )
    p_tb.add_argument(
        "--join-mode",
        choices=["era5_hourly_valid", "issue_target"],
        default="era5_hourly_valid",
        help=(
            "era5_hourly_valid: era5_nwp_input_raw 시계열에서 valid(t0+h) 값 "
            "(재분석 궤적·실운영 예보 아님). "
            "issue_target: (issue_time,target_time) long 테이블 §3.3 누수 방지 (hindcast 등)."
        ),
    )
    p_tb.add_argument(
        "--fcst-schema",
        choices=["era5_native", "shortterm_aligned"],
        default="era5_native",
        help="era5_native: 원천 격자 컬럼명. shortterm_aligned: tmp/pcp/sky 등 단기 슬롄 proxy.",
    )
    p_tb.add_argument(
        "--forecast-parquet",
        default="",
        help=(
            "join-mode별 입력 parquet 경로. 비우면 era5_hourly_valid→era5_nwp_input_raw, "
            "issue_target→era5_fcst_long"
        ),
    )
    p_tb.add_argument(
        "--horizon-max",
        type=int,
        default=TRACK_B_HORIZON_MAX,
        help="조인할 최대 horizon (시간)",
    )
    p_tb.add_argument(
        "--fcst-cols",
        default="",
        dest="fcst_cols",
        help="쉼표구분 예보 값 컬럼명 (비우면 fcst-schema에 따른 기본 목록)",
    )
    p_tb.add_argument(
        "--site-col",
        default="cid_seq",
        help="예보 테이블의 site 키 컬럼",
    )
    p_tb.add_argument(
        "--sites",
        type=int,
        nargs="+",
        default=None,
        help="처리할 cid_seq만 제한 (미지정 시 전체)",
    )

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "build": cmd_build,
        "update-manifest": cmd_update_manifest,
        "quality-check": cmd_quality_check,
        "enrich-track-b": cmd_enrich_track_b,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
