"""
PV 데이터 수집 CLI 진입점.

사용 예:
    # DB 연결 테스트
    python run.py test

    # DB 데이터 범위 요약 출력
    python run.py summary

    # 발전소·CID 메타 수집
    python run.py meta

    # 시계열 전체 수집 (config 기본 기간)
    python run.py hourly

    # 날짜 범위 지정
    python run.py hourly --start 2023-01-01 --end 2023-12-31

    # 특정 CID 만 수집
    python run.py hourly --cid 101 102 103

    # 증분 수집 (마지막 저장 이후부터)
    python run.py hourly --incremental

    # 메타 + 시계열 일괄
    python run.py all --start 2023-01-01 --end 2023-12-31

    # 출력 경로 override
    python run.py all --output /tmp/pv_data
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# 수집기 모듈 임포트
from collector import PvCollector
from config import CollectConfig, DbConfig, collect_config, db_config
from db import test_connection


# ─── 로깅 설정 ────────────────────────────────────────────────────────────────
def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ─── 날짜 파싱 ────────────────────────────────────────────────────────────────
def _parse_date(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"날짜 형식 오류: '{s}' (YYYY-MM-DD 형식 필요)")


# ─── 서브커맨드 핸들러 ────────────────────────────────────────────────────────
def cmd_test(args: argparse.Namespace) -> None:
    """DB 연결 테스트."""
    ok = test_connection()
    sys.exit(0 if ok else 1)


def cmd_summary(args: argparse.Namespace) -> None:
    """DB 데이터 범위 및 CID 목록 출력."""
    collector = _make_collector(args)
    collector.show_db_summary()


def cmd_meta(args: argparse.Namespace) -> None:
    """발전소·CID 메타데이터 수집."""
    collector = _make_collector(args)
    df = collector.collect_meta(save=True)
    print(f"\n메타 수집 완료: {len(df):,} 건 (CID {df['cid_seq'].nunique():,} 개)")
    print(df[["plant_seq", "cid_seq", "plant_nm", "area_1", "area_2",
               "latitude", "longitude", "module_capacity_kw"]].head(10).to_string(index=False))


def cmd_hourly(args: argparse.Namespace) -> None:
    """시간 집계 시계열 수집."""
    collector = _make_collector(args)
    df = collector.collect_hourly(
        start_date  = args.start,
        end_date    = args.end,
        cid_list    = args.cid or None,
        incremental = args.incremental,
        save        = True,
    )
    if df.empty:
        print("수집된 데이터가 없습니다.")
        return

    print(f"\n시계열 수집 완료: {len(df):,} 건 / CID {df['cid_seq'].nunique():,} 개")
    print(f"  기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print(df[["timestamp", "cid_seq", "pow_gen_kwh", "cur_pow_kw"]].tail(5).to_string(index=False))


def cmd_all(args: argparse.Namespace) -> None:
    """메타 + 시계열 일괄 수집."""
    collector = _make_collector(args)
    result = collector.collect_all(
        start_date  = args.start,
        end_date    = args.end,
        cid_list    = args.cid or None,
        incremental = getattr(args, "incremental", False),
    )
    meta   = result["meta"]
    hourly = result["hourly"]
    print(f"\n─── 수집 완료 ─────────────────────────────────────")
    print(f"  메타     : {len(meta):,} 건 / CID {meta['cid_seq'].nunique():,} 개")
    if not hourly.empty:
        print(f"  시계열   : {len(hourly):,} 건")
        print(f"  기간     : {hourly['timestamp'].min()} ~ {hourly['timestamp'].max()}")
    print(f"  저장 위치: {collector.out_dir}")
    print(f"───────────────────────────────────────────────────\n")


# ─── 공통 헬퍼 ────────────────────────────────────────────────────────────────
def _make_collector(args: argparse.Namespace) -> PvCollector:
    output_dir = Path(args.output) if getattr(args, "output", None) else None
    return PvCollector(output_dir=output_dir)


# ─── CLI 정의 ─────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pv_collector",
        description="사내 DB에서 태양광(삼상) 발전 데이터를 수집합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG 로그 출력")

    subs = parser.add_subparsers(dest="command", required=True)

    # ── test ──
    subs.add_parser("test", help="DB 연결 테스트")

    # ── summary ──
    p_sum = subs.add_parser("summary", help="DB 데이터 범위 및 CID 요약")
    p_sum.add_argument("--output", metavar="DIR", help="출력 경로 override")

    # ── meta ──
    p_meta = subs.add_parser("meta", help="발전소·CID 메타데이터 수집")
    p_meta.add_argument("--output", metavar="DIR", help="출력 경로 override")

    # ── hourly ──
    p_hour = subs.add_parser("hourly", help="시간 집계 시계열 수집")
    _add_time_args(p_hour)
    p_hour.add_argument(
        "--incremental", action="store_true",
        help="기존 파일의 마지막 timestamp 이후만 수집",
    )

    # ── all ──
    p_all = subs.add_parser("all", help="메타 + 시계열 일괄 수집")
    _add_time_args(p_all)
    p_all.add_argument(
        "--incremental", action="store_true",
        help="기존 파일의 마지막 timestamp 이후만 수집",
    )

    return parser


def _add_time_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--start", metavar="YYYY-MM-DD", type=_parse_date,
        default=None, help=f"수집 시작일 (기본: {collect_config.start_date})",
    )
    p.add_argument(
        "--end", metavar="YYYY-MM-DD", type=_parse_date,
        default=None, help=f"수집 종료일 (기본: {collect_config.end_date})",
    )
    p.add_argument(
        "--cid", metavar="N", type=int, nargs="+",
        help="수집할 cid_seq 목록 (미지정 시 전체)",
    )
    p.add_argument(
        "--batch-days", metavar="N", type=int, default=None,
        help=f"배치 단위 일수 (기본: {collect_config.batch_days}일)",
    )
    p.add_argument("--output", metavar="DIR", help="출력 경로 override")


# ─── 메인 ─────────────────────────────────────────────────────────────────────
_HANDLERS = {
    "test":    cmd_test,
    "summary": cmd_summary,
    "meta":    cmd_meta,
    "hourly":  cmd_hourly,
    "all":     cmd_all,
}


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    _setup_logging(args.verbose)

    # batch_days override 처리
    if getattr(args, "batch_days", None):
        collect_config.batch_days = args.batch_days

    handler = _HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
