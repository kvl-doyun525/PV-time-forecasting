"""
기상청 API 설정 및 수집 기본값 로더.

우선순위: 환경변수 > .env 파일 > 하드코딩 기본값
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from urllib.parse import unquote

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)

_DEFAULT_OUTPUT = (
    Path(__file__).parent.parent.parent
    / "project" / "artifacts" / "dataset_snapshot"
)

# weather_api 소스 경로 (WeatherAPI 클래스 재사용)
WEATHER_API_DIR = Path(__file__).parent.parent.parent / "weather_api"

# ASOS 지점 목록 (번들 CSV)
ASOS_STATIONS_CSV = Path(__file__).parent / "asos_stations.csv"


def _decode_key(raw: str) -> str:
    """URL 인코딩된 API 키를 디코딩 (weather_api.py 동일 처리)."""
    if not raw:
        return ""
    return unquote(raw) if "%" in raw else raw


@dataclass
class ApiConfig:
    """공공데이터포털 / 기상청 API허브 키."""

    # 단기예보 / 초단기예보 키 (공공데이터포털)
    kma_key: str = field(
        default_factory=lambda: _decode_key(os.environ.get("KMA_API_KEY", ""))
    )
    # ASOS 관측 키 (공공데이터포털, KMA_API_KEY 와 동일 키 사용 가능)
    asos_key: str = field(
        default_factory=lambda: _decode_key(
            os.environ.get("ASOS_API_KEY", "") or os.environ.get("KMA_API_KEY", "")
        )
    )
    # AWS 시간통계 키 (기상청 API허브 — 공공데이터포털과 별도 발급)
    # 비어있으면 AWS 수집을 건너뜀
    apihub_key: str = field(
        default_factory=lambda: _decode_key(os.environ.get("APIHUB_KEY", ""))
    )

    @property
    def has_apihub_key(self) -> bool:
        """AWS 수집 가능 여부."""
        return bool(self.apihub_key)

    def validate_kma(self) -> None:
        if not self.kma_key:
            raise ValueError(
                "KMA_API_KEY 가 설정되지 않았습니다. .env 파일을 확인하세요. (.env.example 참고)"
            )

    def validate_asos(self) -> None:
        if not self.asos_key:
            raise ValueError(
                "ASOS_API_KEY (또는 KMA_API_KEY) 가 설정되지 않았습니다."
            )

    def validate_apihub(self) -> None:
        if not self.apihub_key:
            raise ValueError(
                "APIHUB_KEY 가 설정되지 않았습니다.\n"
                "기상청 API허브(https://apihub.kma.go.kr)에서 키를 발급받아 .env 에 입력하세요.\n"
                "AWS 수집을 건너뛰려면 'python run.py aws' 대신 'python run.py asos' 를 사용하세요."
            )


@dataclass
class CollectConfig:
    """관측 수집 동작 파라미터."""

    start_date: date = field(
        default_factory=lambda: _parse_date(os.environ.get("OBS_START_DATE", ""))
        or date(2022, 1, 1)
    )
    end_date: date = field(
        default_factory=lambda: _parse_date(os.environ.get("OBS_END_DATE", ""))
        or date.today()
    )
    batch_days: int = field(
        default_factory=lambda: int(os.environ.get("OBS_BATCH_DAYS", "30"))
    )
    output_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("OUTPUT_DIR", "") or _DEFAULT_OUTPUT)
    )
    # API 요청 간 대기 (초) — rate limit 방지
    request_interval: float = 0.2


def _parse_date(s: str) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


api_config     = ApiConfig()
collect_config = CollectConfig()
