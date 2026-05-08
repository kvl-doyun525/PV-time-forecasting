"""
기상 데이터 수집 CLI 진입점.

사용 예:
    # site → KMA 격자 / ASOS 지점 매핑 생성
    python run.py mapping

    # ASOS 시간 관측 이력 수집
    python run.py asos --start 2022-01-01 --end 2024-12-31

    # 특정 지점만 수집
    python run.py asos --stations 108 112 156 --start 2023-01-01

    # 증분 수집 (마지막 저장 이후)
    python run.py asos --incremental

    # AWS 관측 수집
    python run.py aws --start 2022-01-01 --end 2024-12-31

    # ERA5 재분석 수집 (학습용 NWP 입력 — 전략 B)
    python run.py era5 --years 2022 2023 2024
    python run.py era5 --years 2022 2023 2024 --no-bias-correct

    # 단기예보 (현재 발표 기준, 전체 site)
    python run.py forecast --type short

    # 초단기예보
    python run.py forecast --type ultra

    # 매핑 + ASOS + AWS + 예보 일괄
    python run.py all --start 2022-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from asos_collector import ASOSCollector
from aws_collector import AWSCollector
from config import ASOS_STATIONS_CSV, collect_config
from era5_collector import ERA5Collector
from kma_forecast_collector import KMAForecastCollector
from kma_mapping import load_asos_stations, run as run_mapping


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _parse_date(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"날짜 형식 오류: '{s}' (YYYY-MM-DD)")


# ─── 서브커맨드 핸들러 ────────────────────────────────────────────────────────

def cmd_mapping(args: argparse.Namespace) -> None:
    """plant_meta.parquet → site_to_kma_grid.csv 생성."""
    output = Path(args.output) if args.output else None
    df = run_mapping(output_path=output)
    print(f"\n매핑 완료: {len(df)} 건")
    print(df[["plant_seq", "cid_seq", "fcst_nx", "fcst_ny",
               "asos_stn_id", "asos_stn_name", "dist_to_asos_km"]].head(10).to_string(index=False))


def cmd_asos(args: argparse.Namespace) -> None:
    """ASOS 시간 관측 이력 수집."""
    collector = ASOSCollector()

    # 지점 목록 결정
    if args.stations:
        station_ids = [str(s) for s in args.stations]
    else:
        # site_to_kma_grid.csv 에서 asos_stn_id 추출
        grid_path = collect_config.output_dir / "site_to_kma_grid.csv"
        if not grid_path.exists():
            print(f"[ERROR] {grid_path} 가 없습니다. 먼저 'python run.py mapping' 을 실행하세요.")
            sys.exit(1)
        grid_df      = pd.read_csv(grid_path, dtype={"asos_stn_id": str})
        station_ids  = grid_df["asos_stn_id"].dropna().unique().tolist()
        print(f"site_to_kma_grid 에서 ASOS 지점 {len(station_ids)} 개 추출")

    output = Path(args.output) if args.output else None
    df = collector.collect_and_save(
        station_ids  = station_ids,
        start_date   = args.start,
        end_date     = args.end,
        output_path  = output,
        incremental  = args.incremental,
    )
    if not df.empty:
        print(f"\nASOS 수집 완료: {len(df):,} 행 / 지점 {df['stnId'].nunique() if 'stnId' in df.columns else '?'} 개")


def cmd_aws(args: argparse.Namespace) -> None:
    """AWS 시간 관측 이력 수집. APIHUB_KEY 미설정 시 건너뜀."""
    from config import api_config as _api_cfg

    if not _api_cfg.has_apihub_key:
        print(
            "[SKIP] APIHUB_KEY 가 설정되지 않아 AWS 수집을 건너뜁니다.\n"
            "       기상청 API허브(https://apihub.kma.go.kr)에서 키를 발급받아\n"
            "       .env 파일의 APIHUB_KEY 에 입력하면 AWS 수집이 활성화됩니다."
        )
        return

    collector = AWSCollector()

    if args.stations:
        station_ids = [str(s) for s in args.stations]
    else:
        grid_path = collect_config.output_dir / "site_to_kma_grid.csv"
        if not grid_path.exists():
            print(f"[ERROR] {grid_path} 가 없습니다. 먼저 'python run.py mapping' 을 실행하세요.")
            sys.exit(1)
        grid_df     = pd.read_csv(grid_path, dtype={"aws_stn_id": str})
        station_ids = grid_df["aws_stn_id"].dropna().unique().tolist()
        print(f"site_to_kma_grid 에서 AWS 지점 {len(station_ids)} 개 추출")

    output = Path(args.output) if args.output else None
    df = collector.collect_and_save(
        station_ids = station_ids,
        start_date  = args.start,
        end_date    = args.end,
        output_path = output,
        incremental = args.incremental,
    )
    if not df.empty:
        print(f"\nAWS 수집 완료: {len(df):,} 행")


def cmd_forecast(args: argparse.Namespace) -> None:
    """단기/초단기예보 수집."""
    grid_path = collect_config.output_dir / "site_to_kma_grid.csv"
    if not grid_path.exists():
        print(f"[ERROR] {grid_path} 가 없습니다. 먼저 'python run.py mapping' 을 실행하세요.")
        sys.exit(1)

    grid_df   = pd.read_csv(grid_path)
    shortterm = (getattr(args, "type", "short") == "short")
    collector = KMAForecastCollector()
    output    = Path(args.output) if getattr(args, "output", None) else None
    incremental = getattr(args, "incremental", False)
    start    = getattr(args, "start", None)
    end      = getattr(args, "end", None)
    kind     = "단기" if shortterm else "초단기"

    if start or end:
        # 날짜 범위 지정 → 과거 이력 수집
        df = collector.collect_range(
            site_grid_map    = grid_df,
            start_date       = start,
            end_date         = end,
            shortterm        = shortterm,
            output_path      = output,
            request_interval = collect_config.request_interval,
            incremental      = incremental,
        )
    else:
        # 날짜 미지정 → 현재 발표 시점 수집
        df = collector.collect_sites_now(
            site_grid_map    = grid_df,
            shortterm        = shortterm,
            output_path      = output,
            request_interval = collect_config.request_interval,
            incremental      = incremental,
        )

    if not df.empty:
        print(f"\n{kind}예보 수집 완료: {len(df):,} 행")


def cmd_era5(args: argparse.Namespace) -> None:
    """ERA5 재분석 수집 (학습용 NWP 입력 — 전략 B, 일사량 제외)."""
    grid_path = collect_config.output_dir / "site_to_kma_grid.csv"
    if not grid_path.exists():
        print(f"[ERROR] {grid_path} 가 없습니다. 먼저 'python run.py mapping' 을 실행하세요.")
        sys.exit(1)

    # site 위경도 목록 로드 (plant_meta.parquet 에서)
    plant_meta_path = collect_config.output_dir / "plant_meta.parquet"
    if not plant_meta_path.exists():
        print(f"[ERROR] {plant_meta_path} 가 없습니다. pv_collector 로 먼저 수집하세요.")
        sys.exit(1)

    plant_meta = pd.read_parquet(plant_meta_path)
    if not {"cid_seq", "latitude", "longitude"}.issubset(plant_meta.columns):
        print("[ERROR] plant_meta.parquet 에 cid_seq / latitude / longitude 컬럼이 없습니다.")
        sys.exit(1)

    # 고유 site 좌표만 사용 (중복 제거)
    site_coords = (
        plant_meta[["cid_seq", "latitude", "longitude"]]
        .drop_duplicates("cid_seq")
        .rename(columns={"latitude": "lat", "longitude": "lon"})
    )
    print(f"대상 site: {len(site_coords)}개")

    years = args.years
    bias_correct = not args.no_bias_correct
    output = Path(args.output) if getattr(args, "output", None) else None

    collector = ERA5Collector()
    df = collector.collect_and_save(
        years         = years,
        site_coords   = site_coords,
        output_path   = output,
        bias_correct  = bias_correct,
    )
    if not df.empty:
        print(f"\nERA5 수집 완료: {len(df):,} 행 / {df['cid_seq'].nunique()} site / {df['timestamp'].dt.year.unique().tolist()} 년")
    else:
        print("[WARN] 수집된 ERA5 데이터가 없습니다. CDS API 키 및 네트워크를 확인하세요.")


def cmd_all(args: argparse.Namespace) -> None:
    """매핑 + ASOS + AWS(선택) + 단기예보 일괄 수집."""
    from config import api_config as _api_cfg

    print("=== 1/4 KMA 격자 매핑 ===")
    cmd_mapping(args)

    print("\n=== 2/4 ASOS 관측 수집 ===")
    cmd_asos(args)

    print("\n=== 3/4 AWS 관측 수집 ===")
    if _api_cfg.has_apihub_key:
        cmd_aws(args)
    else:
        print(
            "[SKIP] APIHUB_KEY 미설정 → AWS 수집 건너뜀\n"
            "       (활성화: .env 에 APIHUB_KEY 입력)"
        )

    print("\n=== 4/4 단기예보 수집 (현재 시점) ===")
    forecast_args = argparse.Namespace(type="short", output=args.output)
    cmd_forecast(forecast_args)

    print("\n=== 전체 수집 완료 ===")


# ─── CLI 정의 ─────────────────────────────────────────────────────────────────

def _add_obs_args(p: argparse.ArgumentParser) -> None:
    """관측 수집 공통 인자."""
    p.add_argument("--start",  metavar="YYYY-MM-DD", type=_parse_date, default=None,
                   help=f"수집 시작일 (기본: {collect_config.start_date})")
    p.add_argument("--end",    metavar="YYYY-MM-DD", type=_parse_date, default=None,
                   help=f"수집 종료일 (기본: {collect_config.end_date})")
    p.add_argument("--stations", metavar="N", type=str, nargs="+",
                   help="수집할 지점 번호 (미지정 시 site_to_kma_grid 에서 자동 추출)")
    p.add_argument("--incremental", action="store_true",
                   help="마지막 저장 이후 데이터만 추가 수집")
    p.add_argument("--output", metavar="PATH", help="저장 경로 override")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weather_collector",
        description="기상청 공공데이터 API로 기상 관측/예보 데이터를 수집합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG 로그 출력")

    subs = parser.add_subparsers(dest="command", required=True)

    # mapping
    p_map = subs.add_parser("mapping", help="site → KMA 격자 / ASOS 지점 매핑 CSV 생성")
    p_map.add_argument("--output", metavar="PATH", help="저장 경로 override")

    # asos
    p_asos = subs.add_parser("asos", help="ASOS 시간 관측 이력 수집")
    _add_obs_args(p_asos)

    # aws
    p_aws = subs.add_parser("aws", help="AWS 시간 관측 이력 수집")
    _add_obs_args(p_aws)

    # forecast
    p_fcst = subs.add_parser(
        "forecast",
        help="단기/초단기예보 수집 (--start/--end 미지정 시 현재 발표 기준)",
    )
    p_fcst.add_argument("--type", choices=["short", "ultra"], default="short",
                         help="short=단기예보, ultra=초단기예보 (기본: short)")
    p_fcst.add_argument("--start", metavar="YYYY-MM-DD", type=_parse_date, default=None,
                         help="과거 이력 수집 시작일 (미지정 시 현재 시점만 수집)")
    p_fcst.add_argument("--end",   metavar="YYYY-MM-DD", type=_parse_date, default=None,
                         help="과거 이력 수집 종료일")
    p_fcst.add_argument("--incremental", action="store_true",
                         help="이미 수집된 issue_time / cid_seq 건너뜀")
    p_fcst.add_argument("--output", metavar="PATH", help="저장 경로 override")

    # era5
    p_era5 = subs.add_parser(
        "era5",
        help="ERA5 재분석 수집 (학습용 NWP 입력 — 전략 B, 일사량 제외)",
    )
    p_era5.add_argument(
        "--years", metavar="YYYY", type=int, nargs="+",
        default=[2022, 2023, 2024],
        help="수집할 연도 목록 (기본: 2022 2023 2024)",
    )
    p_era5.add_argument(
        "--no-bias-correct", action="store_true",
        help="바이어스 교정 생략 (원본 ERA5 만 저장)",
    )
    p_era5.add_argument("--output", metavar="PATH", help="저장 경로 override")

    # all
    p_all = subs.add_parser("all", help="매핑 + ASOS + AWS + ERA5 + 단기예보 일괄 수집")
    _add_obs_args(p_all)

    return parser


_HANDLERS = {
    "mapping":  cmd_mapping,
    "asos":     cmd_asos,
    "aws":      cmd_aws,
    "era5":     cmd_era5,
    "forecast": cmd_forecast,
    "all":      cmd_all,
}


def main() -> None:
    parser  = build_parser()
    args    = parser.parse_args()
    _setup_logging(args.verbose)
    handler = _HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()
