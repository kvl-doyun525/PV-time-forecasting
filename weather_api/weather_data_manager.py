"""
기상 데이터 저장 및 관리 모듈
과거 실측 데이터와 예보 데이터를 저장하고 관리합니다.
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path


class WeatherDataManager:
    """기상 데이터 저장 및 관리 클래스"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 하위 디렉토리 생성
        (self.data_dir / "forecast").mkdir(exist_ok=True)
        (self.data_dir / "sensor").mkdir(exist_ok=True)
        (self.data_dir / "historical").mkdir(exist_ok=True)
    
    def save_forecast_data(self, forecast_data: Dict, location_name: str = None):
        """
        예보 데이터 저장
        
        Args:
            forecast_data: 예보 데이터 딕셔너리
            location_name: 지역명 (파일명에 사용)
        """
        if location_name is None:
            location_name = forecast_data.get("location", {}).get("name", "unknown")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forecast_{location_name}_{timestamp}.json"
        filepath = self.data_dir / "forecast" / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(forecast_data, f, ensure_ascii=False, indent=2)
        
        return str(filepath)
    
    def save_sensor_data(self, sensor_data: Dict, location_name: str, 
                        sensor_type: str = "weather"):
        """
        센서 실측 데이터 저장
        
        Args:
            sensor_data: 센서 데이터 딕셔너리
            location_name: 지역명
            sensor_type: 센서 타입 (weather, solar, wind 등)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sensor_{sensor_type}_{location_name}_{timestamp}.json"
        filepath = self.data_dir / "sensor" / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(sensor_data, f, ensure_ascii=False, indent=2)
        
        return str(filepath)
    
    def load_historical_data(self, location_name: str, 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[Dict]:
        """
        과거 데이터 로드
        
        Args:
            location_name: 지역명
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            과거 데이터 리스트
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()
        
        historical_data = []
        
        # forecast 디렉토리에서 데이터 로드
        forecast_dir = self.data_dir / "forecast"
        for filepath in forecast_dir.glob(f"forecast_{location_name}_*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # 날짜 필터링
                    file_date_str = filepath.stem.split("_")[-1]  # 타임스탬프
                    file_date = datetime.strptime(file_date_str, "%Y%m%d_%H%M%S")
                    
                    if start_date <= file_date <= end_date:
                        historical_data.append(data)
            except Exception as e:
                print(f"파일 로드 실패 {filepath}: {e}")
        
        # sensor 디렉토리에서 데이터 로드
        sensor_dir = self.data_dir / "sensor"
        for filepath in sensor_dir.glob(f"sensor_*_{location_name}_*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    file_date_str = filepath.stem.split("_")[-1]
                    file_date = datetime.strptime(file_date_str, "%Y%m%d_%H%M%S")
                    
                    if start_date <= file_date <= end_date:
                        historical_data.append(data)
            except Exception as e:
                print(f"파일 로드 실패 {filepath}: {e}")
        
        return sorted(historical_data, key=lambda x: x.get("timestamp", ""))
    
    def get_latest_forecast(self, location_name: str) -> Optional[Dict]:
        """
        최신 예보 데이터 조회
        
        Args:
            location_name: 지역명
            
        Returns:
            최신 예보 데이터 또는 None
        """
        forecast_dir = self.data_dir / "forecast"
        matching_files = list(forecast_dir.glob(f"forecast_{location_name}_*.json"))
        
        if not matching_files:
            return None
        
        # 가장 최신 파일
        latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"파일 로드 실패 {latest_file}: {e}")
            return None
    
    def merge_sensor_and_forecast(self, location_name: str, 
                                  forecast_data: Dict) -> Dict:
        """
        센서 실측 데이터와 예보 데이터 병합
        
        Args:
            location_name: 지역명
            forecast_data: 예보 데이터
            
        Returns:
            병합된 데이터
        """
        # 최신 센서 데이터 로드
        sensor_dir = self.data_dir / "sensor"
        sensor_files = list(sensor_dir.glob(f"sensor_*_{location_name}_*.json"))
        
        sensor_data = None
        if sensor_files:
            latest_sensor = max(sensor_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_sensor, "r", encoding="utf-8") as f:
                    sensor_data = json.load(f)
            except:
                pass
        
        merged = forecast_data.copy()
        merged["sensor_data"] = sensor_data
        merged["merged_at"] = datetime.now().isoformat()
        
        return merged
