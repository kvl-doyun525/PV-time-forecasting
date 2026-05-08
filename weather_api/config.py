"""
설정 파일
API 키와 기본 설정을 관리합니다.
YAML 파일과 Python 코드 모두 지원합니다.
"""
import os
from typing import Optional, Dict, Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("경고: PyYAML이 설치되지 않았습니다. YAML 파일 기능을 사용하려면 'pip install pyyaml'을 실행하세요.")

from pathlib import Path


class Config:
    """설정 클래스"""
    
    # YAML 파일 경로
    CONFIG_YAML = "config.yaml"
    LOCATIONS_YAML = "locations.yaml"
    
    # 기상청 단기예보 API 키
    # 공공데이터포털(data.go.kr)에서 발급받은 키를 입력하세요
    WEATHER_SERVICE_KEY: str = os.getenv("WEATHER_API_KEY", "8fsu6XbEIR2%2FTjMWUjVqFzwHaCvbImmfl%2F3qxIqx8CGQirKOh7WTMkZvk1yla9JPTWQwVdP41XN28t2scncceQ%3D%3D")
    
    # 일출몰 정보 API 키 (기상청 API와 동일한 키 사용 가능)
    SUN_MOON_SERVICE_KEY: Optional[str] = os.getenv("SUN_MOON_API_KEY", "8fsu6XbEIR2%2FTjMWUjVqFzwHaCvbImmfl%2F3qxIqx8CGQirKOh7WTMkZvk1yla9JPTWQwVdP41XN28t2scncceQ%3D%3D")
    
    # 조력 정보 API 키 (국립해양조사원 API)
    TIDE_SERVICE_KEY: Optional[str] = os.getenv("TIDE_API_KEY", "8fsu6XbEIR2%2FTjMWUjVqFzwHaCvbImmfl%2F3qxIqx8CGQirKOh7WTMkZvk1yla9JPTWQwVdP41XN28t2scncceQ%3D%3D")
    
    # 기본 위치 설정
    DEFAULT_LATITUDE: float = 37.5665  # 서울
    DEFAULT_LONGITUDE: float = 126.9780  # 서울
    DEFAULT_LOCATION: str = "서울"
    
    # API 요청 타임아웃 (초)
    REQUEST_TIMEOUT: int = 30
    
    # 데이터 저장 경로
    DATA_DIR: str = "data"
    
    # YAML 설정 캐시
    _yaml_config: Optional[Dict[str, Any]] = None
    _locations_config: Optional[Dict[str, Any]] = None
    
    @classmethod
    def load_yaml_config(cls) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        if not YAML_AVAILABLE:
            return {}
        
        if cls._yaml_config is None:
            config_path = Path(cls.CONFIG_YAML)
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        cls._yaml_config = yaml.safe_load(f) or {}
                except Exception as e:
                    print(f"YAML 설정 파일 로드 실패: {e}")
                    cls._yaml_config = {}
            else:
                cls._yaml_config = {}
        return cls._yaml_config
    
    @classmethod
    def load_locations_config(cls) -> Dict[str, Any]:
        """지역 정보 YAML 파일 로드"""
        if not YAML_AVAILABLE:
            return {}
        
        if cls._locations_config is None:
            locations_path = Path(cls.LOCATIONS_YAML)
            if locations_path.exists():
                try:
                    with open(locations_path, 'r', encoding='utf-8') as f:
                        cls._locations_config = yaml.safe_load(f) or {}
                except Exception as e:
                    print(f"지역 정보 YAML 파일 로드 실패: {e}")
                    cls._locations_config = {}
            else:
                cls._locations_config = {}
        return cls._locations_config
    
    @classmethod
    def get_location_info(cls, location_name: str) -> Optional[Dict[str, Any]]:
        """지역명으로 지역 정보 조회"""
        locations = cls.load_locations_config()
        cities = locations.get("major_cities", [])
        
        for city in cities:
            if city.get("name") == location_name:
                return city
        return None
    
    @classmethod
    def get_tide_station_info(cls, station_code: str) -> Optional[Dict[str, Any]]:
        """조석 관측소 코드로 관측소 정보 조회"""
        locations = cls.load_locations_config()
        stations = locations.get("tide_observation_stations", [])
        
        for station in stations:
            if station.get("code") == station_code:
                return station
        return None
    
    @classmethod
    def get_all_cities(cls) -> list:
        """모든 도시 정보 반환"""
        locations = cls.load_locations_config()
        return locations.get("major_cities", [])
    
    @classmethod
    def get_all_tide_stations(cls) -> list:
        """모든 조석 관측소 정보 반환"""
        locations = cls.load_locations_config()
        return locations.get("tide_observation_stations", [])
    
    @classmethod
    def validate(cls) -> bool:
        """설정 유효성 검사"""
        # YAML 파일에서 API 키 확인
        if YAML_AVAILABLE:
            yaml_config = cls.load_yaml_config()
            api_keys = yaml_config.get("api_keys", {})
            
            # YAML 파일의 API 키가 있으면 사용 (환경 변수 우선)
            if api_keys.get("weather", {}).get("key"):
                if not os.getenv("WEATHER_API_KEY"):
                    cls.WEATHER_SERVICE_KEY = api_keys["weather"]["key"]
            
            if api_keys.get("sun_moon", {}).get("key"):
                if not os.getenv("SUN_MOON_API_KEY"):
                    cls.SUN_MOON_SERVICE_KEY = api_keys["sun_moon"]["key"]
            
            if api_keys.get("tide", {}).get("key"):
                if not os.getenv("TIDE_API_KEY"):
                    cls.TIDE_SERVICE_KEY = api_keys["tide"]["key"]
            
            # YAML 파일의 기본 위치 설정 확인
            default_loc = yaml_config.get("default_location", {})
            if default_loc:
                if not hasattr(cls, '_default_location_loaded'):
                    cls.DEFAULT_LOCATION = default_loc.get("name", cls.DEFAULT_LOCATION)
                    cls.DEFAULT_LATITUDE = default_loc.get("latitude", cls.DEFAULT_LATITUDE)
                    cls.DEFAULT_LONGITUDE = default_loc.get("longitude", cls.DEFAULT_LONGITUDE)
                    cls._default_location_loaded = True
        
        if cls.WEATHER_SERVICE_KEY == "YOUR_WEATHER_API_KEY_HERE" or not cls.WEATHER_SERVICE_KEY:
            print("경고: 기상청 API 키가 설정되지 않았습니다.")
            print("공공데이터포털(data.go.kr)에서 API 키를 발급받아 설정하세요.")
            print("  - config.yaml 파일에 입력하거나")
            print("  - 환경 변수 WEATHER_API_KEY로 설정하세요.")
            return False
        return True
