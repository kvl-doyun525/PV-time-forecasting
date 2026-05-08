"""
DB 접속 설정 및 수집 기본값 로더.

우선순위: 환경변수 > .env 파일 > 하드코딩 기본값
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

# 이 파일 위치에서 .env 탐색
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)

# project/artifacts/dataset_snapshot 기본 출력 경로
_DEFAULT_OUTPUT = (
    Path(__file__).parent.parent.parent
    / "project" / "artifacts" / "dataset_snapshot"
)


@dataclass
class DbConfig:
    host:     str = field(default_factory=lambda: os.environ.get("DB_HOST", ""))
    port:     int = field(default_factory=lambda: int(os.environ.get("DB_PORT", "3306")))
    user:     str = field(default_factory=lambda: os.environ.get("DB_USER", ""))
    password: str = field(default_factory=lambda: os.environ.get("DB_PASSWORD", ""))
    database: str = field(default_factory=lambda: os.environ.get("DB_NAME", ""))
    charset:  str = field(default_factory=lambda: os.environ.get("DB_CHARSET", "utf8mb4"))

    def validate(self) -> None:
        missing = [f for f in ("host", "user", "password", "database") if not getattr(self, f)]
        if missing:
            raise ValueError(
                f"DB 접속 정보 누락: {missing}\n"
                f".env 파일을 생성하거나 환경변수를 설정해주세요. (.env.example 참고)"
            )


@dataclass
class CollectConfig:
    """수집 동작 파라미터."""

    # 시계열 수집 기간
    start_date: date = field(
        default_factory=lambda: _parse_date(os.environ.get("COLLECT_START_DATE", ""))
        or date(2022, 1, 1)
    )
    end_date: date = field(
        default_factory=lambda: _parse_date(os.environ.get("COLLECT_END_DATE", ""))
        or date.today()
    )

    # 한 번에 fetch 할 날짜 범위 (일)
    batch_days: int = field(
        default_factory=lambda: int(os.environ.get("COLLECT_BATCH_DAYS", "30"))
    )

    # 출력 디렉토리
    output_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("OUTPUT_DIR", "") or _DEFAULT_OUTPUT)
    )

    # phase_type 필터 (삼상 고정)
    phase_type: str = "tp"


def _parse_date(s: str) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


# ── 싱글턴 인스턴스 ────────────────────────────────────────────────────────────
db_config      = DbConfig()
collect_config = CollectConfig()
