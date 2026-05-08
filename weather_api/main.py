"""
기상 데이터 수집 메인 프로그램
"""
import json
from datetime import datetime
from weather_api import WeatherDataCollector
from config import Config
from utils import print_forecast_summary, save_to_json


def main():
    """메인 함수"""
    # 설정 검증
    if not Config.validate():
        print("\n설정 파일(config.py)에 API 키를 입력하거나")
        print("환경 변수로 설정하세요:")
        print("  export WEATHER_API_KEY='your_api_key'")
        print("  export SUN_MOON_API_KEY='your_api_key'")
        print("  export TIDE_API_KEY='your_api_key'")
        return
    
    # 데이터 수집기 생성
    collector = WeatherDataCollector(
        weather_service_key=Config.WEATHER_SERVICE_KEY,
        sun_moon_service_key=Config.SUN_MOON_SERVICE_KEY,
        tide_service_key=Config.TIDE_SERVICE_KEY
    )
    
    # 기본 위치로 데이터 수집
    print(f"\n{Config.DEFAULT_LOCATION} 지역의 기상 데이터를 수집합니다...")
    print(f"위도: {Config.DEFAULT_LATITUDE}, 경도: {Config.DEFAULT_LONGITUDE}\n")
    
    try:
        # 종합 예보 데이터 조회
        forecast_data = collector.get_comprehensive_forecast(
            latitude=Config.DEFAULT_LATITUDE,
            longitude=Config.DEFAULT_LONGITUDE,
            location=Config.DEFAULT_LOCATION
        )
        
        # 결과 출력
        print_forecast_summary(forecast_data)
        
        # JSON 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weather_forecast_{timestamp}.json"
        save_to_json(forecast_data, filename)
        
        print(f"\n총 {len(forecast_data['forecast_24h'])}개의 예보 데이터를 수집했습니다.")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        print("\nAPI 키가 올바른지 확인하세요.")
        print("공공데이터포털(data.go.kr)에서 API 키를 발급받을 수 있습니다.")


if __name__ == "__main__":
    main()
