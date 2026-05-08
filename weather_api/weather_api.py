"""
한국 기상청 공공데이터 API를 사용한 기상 데이터 수집 모듈
"""
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import math
from urllib.parse import unquote


class WeatherAPI:
    """기상청 단기예보 API 클래스"""
    
    # 초단기예보 (1~6시간)
    ULTRA_SRT_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
    # 단기예보 (3일간, 3시간 간격)
    VILAGE_FCST_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    
    BASE_URL = ULTRA_SRT_URL  # 기본값
    
    # 기상 코드 매핑
    SKY_CODE = {
        "1": "맑음",
        "3": "구름많음",
        "4": "흐림"
    }
    
    PTY_CODE = {
        "0": "없음",
        "1": "비",
        "2": "비/눈",
        "3": "눈",
        "4": "소나기"
    }
    
    def __init__(self, service_key: str):
        """
        Args:
            service_key: 공공데이터포털에서 발급받은 API 인증키
        """
        # API 키가 이미 URL 인코딩되어 있으면 디코딩
        # requests가 자동으로 인코딩하므로 이중 인코딩 방지
        self.service_key = unquote(service_key) if '%' in service_key else service_key
    
    def get_grid_coordinates(self, latitude: float, longitude: float) -> tuple:
        """
        위경도를 기상청 격자 좌표로 변환
        
        Args:
            latitude: 위도
            longitude: 경도
            
        Returns:
            (nx, ny) 격자 좌표 튜플
        """
        # 기상청 격자 변환 공식
        RE = 6371.00877  # 지구 반경(km)
        GRID = 5.0  # 격자 간격(km)
        SLAT1 = 30.0  # 표준 위도 1
        SLAT2 = 60.0  # 표준 위도 2
        OLON = 126.0  # 기준점 경도
        OLAT = 38.0  # 기준점 위도
        XO = 43  # 기준점 X좌표
        YO = 136  # 기준점 Y좌표
        
        DEGRAD = math.pi / 180.0
        RADDEG = 180.0 / math.pi
        
        re = RE / GRID
        slat1 = SLAT1 * DEGRAD
        slat2 = SLAT2 * DEGRAD
        olon = OLON * DEGRAD
        olat = OLAT * DEGRAD
        
        sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
        sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
        sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
        sf = math.pow(sf, sn) * math.cos(slat1) / sn
        ro = math.tan(math.pi * 0.25 + olat * 0.5)
        ro = re * sf / math.pow(ro, sn)
        
        ra = math.tan(math.pi * 0.25 + (latitude) * DEGRAD * 0.5)
        ra = re * sf / math.pow(ra, sn)
        theta = longitude * DEGRAD - olon
        if theta > math.pi:
            theta -= 2.0 * math.pi
        if theta < -math.pi:
            theta += 2.0 * math.pi
        theta *= sn
        
        nx = int(ra * math.sin(theta) + XO + 0.5)
        ny = int(ro - ra * math.cos(theta) + YO + 0.5)
        
        return nx, ny
    
    def get_base_time(self) -> tuple:
        """
        현재 시간 기준으로 가장 최근 예보 발표 시간 반환
        
        Returns:
            (base_date, base_time) 튜플 (YYYYMMDD, HHMM 형식)
        """
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # 예보 발표 시간: 00, 03, 06, 09, 12, 15, 18, 21시
        forecast_times = [0, 3, 6, 9, 12, 15, 18, 21]
        
        # 현재 시간 이전의 가장 최근 발표 시간 찾기
        base_hour = 21  # 기본값
        for ft in reversed(forecast_times):
            if hour >= ft:
                base_hour = ft
                break
        
        # 만약 오늘 발표 시간이 없으면 어제 21시
        if hour < 0:
            base_date = (now - timedelta(days=1)).strftime("%Y%m%d")
            base_time = "2100"
        else:
            base_date = now.strftime("%Y%m%d")
            base_time = f"{base_hour:02d}00"
        
        return base_date, base_time
    
    def _adjust_short_term_time(self, base_time: str) -> str:
        """
        단기예보 발표 시간 조정
        단기예보는 02, 05, 08, 11, 14, 17, 20, 23시에 발표
        
        Args:
            base_time: 초단기예보 base_time (HHMM 형식)
            
        Returns:
            단기예보 base_time (HHMM 형식)
        """
        hour = int(base_time[:2])
        
        # 단기예보 발표 시간: 02, 05, 08, 11, 14, 17, 20, 23시
        short_term_times = [2, 5, 8, 11, 14, 17, 20, 23]
        
        # 현재 시간 이전의 가장 최근 발표 시간 찾기
        adjusted_hour = 23  # 기본값
        for stt in reversed(short_term_times):
            if hour >= stt:
                adjusted_hour = stt
                break
        
        return f"{adjusted_hour:02d}00"
    
    def get_weather_forecast(self, nx: int, ny: int, 
                            base_date: Optional[str] = None, 
                            base_time: Optional[str] = None,
                            use_short_term: bool = False) -> List[Dict]:
        """
        기상 예보 데이터 조회
        
        Args:
            nx: 격자 X 좌표
            ny: 격자 Y 좌표
            base_date: 예보 발표 기준날짜 (YYYYMMDD), None이면 자동 계산
            base_time: 예보 발표 기준시간 (HHMM), None이면 자동 계산
            use_short_term: True면 단기예보(3일), False면 초단기예보(6시간)
            
        Returns:
            예보 데이터 리스트
        """
        if base_date is None or base_time is None:
            base_date, base_time = self.get_base_time()
        
        # 단기예보는 발표 시간이 다름 (02, 05, 08, 11, 14, 17, 20, 23시)
        if use_short_term:
            base_time = self._adjust_short_term_time(base_time)
            api_url = self.VILAGE_FCST_URL
        else:
            api_url = self.ULTRA_SRT_URL
        
        params = {
            "serviceKey": self.service_key,
            "pageNo": "1",
            "numOfRows": "1000",  # 단기예보는 데이터가 많으므로 증가
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": str(nx),
            "ny": str(ny)
        }
        
        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["response"]["header"]["resultCode"] != "00":
                error_msg = data["response"]["header"]["resultMsg"]
                raise Exception(f"API 오류: {error_msg}")
            
            items = data["response"]["body"]["items"]["item"]
            if not items:
                return []
            
            # 시간별로 데이터 정리
            forecast_dict = {}
            for item in items:
                fcst_time = item["fcstTime"]
                if fcst_time not in forecast_dict:
                    # 날짜를 YYYY-MM-DD 형식으로 변환
                    fcst_date = item["fcstDate"]
                    formatted_date = f"{fcst_date[:4]}-{fcst_date[4:6]}-{fcst_date[6:8]}"
                    formatted_time = f"{fcst_time[:2]}:{fcst_time[2:]}"
                    forecast_dict[fcst_time] = {
                        "date": item["fcstDate"],
                        "time": fcst_time,
                        "datetime": f"{formatted_date} {formatted_time}"
                    }
                
                category = item["category"]
                value = item["fcstValue"]
                
                if category == "T1H":  # 기온
                    forecast_dict[fcst_time]["temperature"] = float(value)
                elif category == "REH":  # 습도
                    forecast_dict[fcst_time]["humidity"] = float(value)
                elif category == "SKY":  # 하늘상태
                    forecast_dict[fcst_time]["sky"] = self.SKY_CODE.get(value, value)
                    forecast_dict[fcst_time]["sky_code"] = int(value)  # 태양광 예측용
                elif category == "PTY":  # 강수형태
                    forecast_dict[fcst_time]["precipitation_type"] = self.PTY_CODE.get(value, value)
                    forecast_dict[fcst_time]["precipitation_type_code"] = int(value)
                elif category == "PCP":  # 강수량 (1시간 강수량, mm)
                    if value and value != "강수없음":
                        try:
                            # "1.5mm" 형식 처리
                            if "mm" in str(value):
                                forecast_dict[fcst_time]["precipitation_amount"] = float(str(value).replace("mm", ""))
                            else:
                                forecast_dict[fcst_time]["precipitation_amount"] = float(value)
                        except:
                            forecast_dict[fcst_time]["precipitation_amount"] = 0.0
                    else:
                        forecast_dict[fcst_time]["precipitation_amount"] = 0.0
                elif category == "POP":  # 강수확률 (%)
                    forecast_dict[fcst_time]["precipitation_probability"] = float(value)
                elif category == "WSD":  # 풍속
                    forecast_dict[fcst_time]["wind_speed"] = float(value)
                elif category == "VEC":  # 풍향
                    forecast_dict[fcst_time]["wind_direction"] = float(value)
                elif category == "WAV":  # 파고
                    forecast_dict[fcst_time]["wave_height"] = float(value)
                elif category == "TMN":  # 최저기온
                    forecast_dict[fcst_time]["min_temperature"] = float(value)
                elif category == "TMX":  # 최고기온
                    forecast_dict[fcst_time]["max_temperature"] = float(value)
                elif category == "SNO":  # 적설량 (1시간 적설량, cm)
                    if value and value != "적설없음":
                        try:
                            if "cm" in str(value):
                                forecast_dict[fcst_time]["snowfall"] = float(str(value).replace("cm", ""))
                            else:
                                forecast_dict[fcst_time]["snowfall"] = float(value)
                        except:
                            forecast_dict[fcst_time]["snowfall"] = 0.0
                    else:
                        forecast_dict[fcst_time]["snowfall"] = 0.0
                elif category == "TMP":  # 1시간 기온 (단기예보용)
                    if "temperature" not in forecast_dict[fcst_time]:
                        forecast_dict[fcst_time]["temperature"] = float(value)
            
            # 리스트로 변환하고 시간순 정렬
            forecasts = list(forecast_dict.values())
            forecasts.sort(key=lambda x: x["datetime"])
            
            return forecasts
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API 요청 실패: {str(e)}")
    
    def get_24hour_forecast(self, latitude: float, longitude: float, 
                           use_short_term: bool = True) -> List[Dict]:
        """
        24시간 예보 데이터 조회
        
        하이브리드 방식:
        - 처음 6시간: 초단기예보 사용 (더 정확)
        - 6시간 이후: 단기예보 사용 (24시간 채우기)
        
        Args:
            latitude: 위도
            longitude: 경도
            use_short_term: True면 하이브리드 방식 (초단기+단기), False면 초단기예보만 사용
            
        Returns:
            24시간 예보 데이터 리스트
        """
        nx, ny = self.get_grid_coordinates(latitude, longitude)
        now = datetime.now()
        
        if use_short_term:
            # 하이브리드 방식: 초단기예보(6시간) + 단기예보(나머지)
            all_forecasts = []
            
            # 1. 초단기예보로 처음 6시간 데이터 수집 (더 정확)
            try:
                ultra_forecasts = self.get_weather_forecast(nx, ny, use_short_term=False)
                ultra_end_time = now + timedelta(hours=6)
                
                for fcst in ultra_forecasts:
                    fcst_time = datetime.strptime(fcst["datetime"], "%Y-%m-%d %H:%M")
                    # 현재 시간 이후이고 6시간 이내인 데이터만 선택
                    if now < fcst_time <= ultra_end_time:
                        all_forecasts.append(fcst)
            except Exception as e:
                print(f"초단기예보 조회 실패: {e}, 단기예보만 사용합니다.")
                ultra_forecasts = []
            
            # 2. 단기예보로 6시간 이후 ~ 24시간 데이터 수집
            # 단기예보는 이미 1시간 단위로 제공되므로 보간 불필요
            try:
                short_forecasts = self.get_weather_forecast(nx, ny, use_short_term=True)
                # 6시간 후의 정시를 기준으로 설정 (예: 10:39 -> 16:00)
                short_start_hour = (now + timedelta(hours=6)).replace(minute=0, second=0, microsecond=0)
                target_time = now + timedelta(hours=24)
                
                for fcst in short_forecasts:
                    fcst_time = datetime.strptime(fcst["datetime"], "%Y-%m-%d %H:%M")
                    # 6시간 후 정시 이후(포함)이고 24시간 이내인 데이터 선택
                    # 초단기예보와 겹치는 시간은 중복 제거에서 초단기예보가 우선
                    if fcst_time >= short_start_hour and fcst_time <= target_time:
                        all_forecasts.append(fcst)
            except Exception as e:
                print(f"단기예보 조회 실패: {e}")
                # 단기예보 실패 시 초단기예보로 24시간 채우기 시도
                if not ultra_forecasts:
                    use_short_term = False
                else:
                    # 초단기예보만으로는 6시간만 있으므로 단기예보 없이는 24시간 불가
                    # 초단기예보만 반환
                    all_forecasts.sort(key=lambda x: x["datetime"])
                    return all_forecasts
            
            # 하이브리드 방식: 초단기예보 + 단기예보 데이터 병합
            # 중복 제거 및 정렬 (초단기예보가 우선)
            seen = set()
            unique_forecasts = []
            # 초단기예보를 먼저 추가 (더 정확하므로 우선)
            for fcst in all_forecasts:
                key = fcst["datetime"]
                fcst_time = datetime.strptime(fcst["datetime"], "%Y-%m-%d %H:%M")
                # 현재 시간 이후이고 24시간 이내인 데이터만 선택
                if now < fcst_time <= now + timedelta(hours=24):
                    if key not in seen:
                        seen.add(key)
                        unique_forecasts.append(fcst)
            
            unique_forecasts.sort(key=lambda x: x["datetime"])
            
            # 단기예보는 이미 1시간 단위이므로 보간 불필요
            # 현재 시간 기준으로 24시간까지만 반환
            return unique_forecasts
        
        if not use_short_term:
            # 초단기예보만 사용 (6시간까지만 제공)
            forecasts = self.get_weather_forecast(nx, ny, use_short_term=False)
            
            # 24시간 데이터를 위해 여러 번 호출
            all_forecasts = []
            base_date, base_time = self.get_base_time()
            current_time = datetime.strptime(f"{base_date} {base_time}", "%Y%m%d %H%M")
            
            for i in range(4):  # 최대 4번 호출
                forecast_date = (current_time + timedelta(hours=i*6)).strftime("%Y%m%d")
                forecast_hour = (current_time + timedelta(hours=i*6)).hour
                
                forecast_times = [0, 3, 6, 9, 12, 15, 18, 21]
                base_hour = 21
                for ft in reversed(forecast_times):
                    if forecast_hour >= ft:
                        base_hour = ft
                        break
                
                base_time_str = f"{base_hour:02d}00"
                
                try:
                    fcst = self.get_weather_forecast(nx, ny, forecast_date, base_time_str, use_short_term=False)
                    all_forecasts.extend(fcst)
                except:
                    continue
            
            # 중복 제거 및 정렬
            seen = set()
            unique_forecasts = []
            for fcst in all_forecasts:
                key = fcst["datetime"]
                if key not in seen:
                    seen.add(key)
                    unique_forecasts.append(fcst)
            
            unique_forecasts.sort(key=lambda x: x["datetime"])
            
            # 현재 시간 기준으로 24시간까지만 필터링
            now = datetime.now()
            target_time = now + timedelta(hours=24)
            filtered_forecasts = [
                f for f in unique_forecasts 
                if now < datetime.strptime(f["datetime"], "%Y-%m-%d %H:%M") <= target_time
            ]
            
            # 정확히 24시간 데이터가 있도록 보간 (현재 시간부터 1시간 간격)
            if filtered_forecasts:
                return self._interpolate_forecast(filtered_forecasts, start_time=now, hours=24)
            else:
                return filtered_forecasts[:24]
    
    def _interpolate_forecast(self, forecasts: List[Dict], start_time: Optional[datetime] = None, hours: int = 24) -> List[Dict]:
        """
        3시간 간격 예보 데이터를 1시간 단위로 선형 보간
        
        Args:
            forecasts: 예보 데이터 리스트 (시간 순으로 정렬되어 있어야 함)
            start_time: 보간 시작 시간 (None이면 현재 시간)
            hours: 보간할 시간 수
            
        Returns:
            보간된 예보 데이터 리스트
        """
        if not forecasts:
            return []
        
        if start_time is None:
            start_time = datetime.now()
        
        # forecasts를 시간 순으로 정렬
        sorted_forecasts = sorted(forecasts, key=lambda x: datetime.strptime(x["datetime"], "%Y-%m-%d %H:%M"))
        
        # 시간 리스트 생성
        forecast_times = [datetime.strptime(f["datetime"], "%Y-%m-%d %H:%M") for f in sorted_forecasts]
        
        interpolated = []
        
        for i in range(hours):
            target_time = start_time + timedelta(hours=i)
            target_str = target_time.strftime("%Y-%m-%d %H:%M")
            
            # 정확히 일치하는 데이터가 있는지 확인
            exact_match = None
            for j, ft in enumerate(forecast_times):
                if ft == target_time:
                    exact_match = sorted_forecasts[j].copy()
                    exact_match["datetime"] = target_str
                    interpolated.append(exact_match)
                    break
            
            if exact_match:
                continue
            
            # 이전과 다음 데이터 찾기
            prev_idx = None
            next_idx = None
            
            for j, ft in enumerate(forecast_times):
                if ft < target_time:
                    prev_idx = j
                elif ft > target_time:
                    next_idx = j
                    break
            
            # 선형 보간 수행
            if prev_idx is not None and next_idx is not None:
                # 이전과 다음 데이터 사이를 선형 보간
                prev_fcst = sorted_forecasts[prev_idx]
                next_fcst = sorted_forecasts[next_idx]
                prev_time = forecast_times[prev_idx]
                next_time = forecast_times[next_idx]
                
                # 시간 비율 계산
                total_diff = (next_time - prev_time).total_seconds()
                target_diff = (target_time - prev_time).total_seconds()
                ratio = target_diff / total_diff if total_diff > 0 else 0
                
                # 보간된 데이터 생성
                interpolated_fcst = {
                    "date": target_time.strftime("%Y%m%d"),
                    "time": target_time.strftime("%H%M"),
                    "datetime": target_str
                }
                
                # 수치 데이터 선형 보간
                numeric_fields = ["temperature", "humidity", "wind_speed", "wind_direction", 
                                 "precipitation_amount", "precipitation_probability", "wave_height",
                                 "min_temperature", "max_temperature", "snowfall"]
                
                for field in numeric_fields:
                    if field in prev_fcst and field in next_fcst:
                        prev_val = prev_fcst[field]
                        next_val = next_fcst[field]
                        interpolated_fcst[field] = prev_val + (next_val - prev_val) * ratio
                    elif field in prev_fcst:
                        interpolated_fcst[field] = prev_fcst[field]
                    elif field in next_fcst:
                        interpolated_fcst[field] = next_fcst[field]
                
                # 카테고리 데이터는 가장 가까운 값 사용
                category_fields = ["precipitation_type", "precipitation_type_code", "sky", "sky_code"]
                for field in category_fields:
                    if field in prev_fcst:
                        interpolated_fcst[field] = prev_fcst[field]
                    elif field in next_fcst:
                        interpolated_fcst[field] = next_fcst[field]
                
                interpolated.append(interpolated_fcst)
                
            elif prev_idx is not None:
                # 이전 데이터만 있는 경우 (마지막 데이터 사용)
                interpolated_fcst = sorted_forecasts[prev_idx].copy()
                interpolated_fcst["datetime"] = target_str
                interpolated_fcst["date"] = target_time.strftime("%Y%m%d")
                interpolated_fcst["time"] = target_time.strftime("%H%M")
                interpolated.append(interpolated_fcst)
                
            elif next_idx is not None:
                # 다음 데이터만 있는 경우 (다음 데이터 사용)
                interpolated_fcst = sorted_forecasts[next_idx].copy()
                interpolated_fcst["datetime"] = target_str
                interpolated_fcst["date"] = target_time.strftime("%Y%m%d")
                interpolated_fcst["time"] = target_time.strftime("%H%M")
                interpolated.append(interpolated_fcst)
            
            else:
                # 데이터가 없는 경우 (첫 번째 데이터 사용)
                if sorted_forecasts:
                    interpolated_fcst = sorted_forecasts[0].copy()
                    interpolated_fcst["datetime"] = target_str
                    interpolated_fcst["date"] = target_time.strftime("%Y%m%d")
                    interpolated_fcst["time"] = target_time.strftime("%H%M")
                    interpolated.append(interpolated_fcst)
        
        return interpolated


class SunMoonAPI:
    """일출몰 시간 API 클래스 - 한국천문연구원 출몰시각 정보
    
    공공데이터포털의 일출일몰정보 조회서비스를 사용하여 일출, 일몰, 월출, 월몰 시간을 수집합니다.
    
    참고 문서: https://www.data.go.kr/data/15012688/openapi.do
    
    API 정보:
    - getAreaRiseSetInfo: 지역별 해달 출몰시각 정보조회 (지역명 기반)
    - getLCRiseSetInfo: 위치별 해달 출몰시각 정보조회 (위경도 기반)
    - 응답 형식: XML (기본)
    
    제공 데이터:
    - sunrise: 일출 시간
    - sunset: 일몰 시간
    - moonrise: 월출 시간
    - moonset: 월몰 시간
    - suntransit: 일중 시간
    - moontransit: 월중 시간
    - civilm/civile: 시민박명 (아침/저녁)
    - nautm/naute: 항해박명 (아침/저녁)
    - astm/aste: 천문박명 (아침/저녁)
    
    대체 방법:
    - API 실패 시 calculate_sunrise_sunset() 메서드로 천문학적 계산 수행
    
    사용 예시:
        sun_moon_api = SunMoonAPI(service_key="your_api_key")
        sun_moon_info = sun_moon_api.get_sun_moon_info(location="서울", date="20260130")
    """
    
    # 지역별 해달 출몰시각 정보조회 API
    AREA_BASE_URL = "http://apis.data.go.kr/B090041/openapi/service/RiseSetInfoService/getAreaRiseSetInfo"
    # 위치별 해달 출몰시각 정보조회 API
    LOCATION_BASE_URL = "http://apis.data.go.kr/B090041/openapi/service/RiseSetInfoService/getLCRiseSetInfo"
    
    # 기본 URL (하위 호환성을 위해 유지)
    BASE_URL = AREA_BASE_URL
    
    def __init__(self, service_key: str):
        """
        Args:
            service_key: 공공데이터포털에서 발급받은 API 인증키
        """
        # API 키가 이미 URL 인코딩되어 있으면 디코딩
        self.service_key = unquote(service_key) if '%' in service_key else service_key
    
    def calculate_sunrise_sunset(self, latitude: float, longitude: float, 
                                 date: Optional[datetime] = None) -> Dict:
        """
        위경도 기반 일출몰 시간 계산 (API 실패 시 대체 방법)
        
        Args:
            latitude: 위도
            longitude: 경도
            date: 조회 날짜, None이면 오늘
            
        Returns:
            일출몰 정보 딕셔너리
        """
        if date is None:
            date = datetime.now()
        
        # 간단한 일출몰 계산 공식 (근사치)
        # 실제로는 더 정확한 천문 계산이 필요하지만, 기본적인 계산 제공
        day_of_year = date.timetuple().tm_yday
        
        # 태양의 적위 계산 (근사)
        declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365))
        
        # 시간 각도 계산
        lat_rad = math.radians(latitude)
        dec_rad = math.radians(declination)
        
        # 일출몰 시간 각도
        hour_angle = math.acos(-math.tan(lat_rad) * math.tan(dec_rad))
        hour_angle_deg = math.degrees(hour_angle)
        
        # 시간 변환 (시간 단위)
        sunrise_hour = 12 - (hour_angle_deg / 15)
        sunset_hour = 12 + (hour_angle_deg / 15)
        
        # 시간 정규화
        if sunrise_hour < 0:
            sunrise_hour += 24
        if sunset_hour >= 24:
            sunset_hour -= 24
        
        sunrise_min = int((sunrise_hour - int(sunrise_hour)) * 60)
        sunset_min = int((sunset_hour - int(sunset_hour)) * 60)
        
        return {
            "date": date.strftime("%Y%m%d"),
            "latitude": latitude,
            "longitude": longitude,
            "sunrise": f"{int(sunrise_hour):02d}{sunrise_min:02d}",
            "sunset": f"{int(sunset_hour):02d}{sunset_min:02d}",
            "method": "calculated"
        }
    
    def get_sun_moon_info(self, location: str, date: Optional[str] = None) -> Dict:
        """
        일출몰 시간 조회 (한국천문연구원 출몰시각 정보)
        참고: https://www.data.go.kr/data/15012688/openapi.do
        
        Args:
            location: 지역명 (예: "서울", "부산")
            date: 조회 날짜 (YYYYMMDD), None이면 오늘
            
        Returns:
            일출몰 정보 딕셔너리
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        params = {
            "serviceKey": self.service_key,
            "locdate": date,
            "location": location
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # 응답이 XML인 경우 처리 (공공데이터포털 기본 형식)
            content_type = response.headers.get('Content-Type', '').lower()
            response_text = response.text
            
            if 'xml' in content_type or response_text.strip().startswith('<?xml'):
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response_text)
                
                # XML 응답 파싱
                # 공공데이터포털 XML 응답 형식: <response><header>...</header><body><items><item>...</item></items></body></response>
                header = root.find('header')
                if header is not None:
                    result_code = header.find('resultCode')
                    if result_code is not None and result_code.text != "00":
                        result_msg = header.find('resultMsg')
                        error_msg = result_msg.text if result_msg is not None else "알 수 없는 오류"
                        raise Exception(f"API 오류: {error_msg}")
                
                body = root.find('body')
                if body is None:
                    raise Exception("응답에 body가 없습니다.")
                
                items = body.find('items')
                if items is None:
                    raise Exception("응답에 items가 없습니다.")
                
                item = items.find('item')
                if item is None:
                    raise Exception("응답에 item이 없습니다.")
                
                # XML 요소에서 데이터 추출
                def get_text(elem, tag, default=""):
                    found = elem.find(tag)
                    if found is not None and found.text:
                        return found.text.strip()
                    return default
                
                return {
                    "date": get_text(item, "locdate", date),
                    "location": get_text(item, "location", location),
                    "longitude": get_text(item, "longitude", ""),
                    "latitude": get_text(item, "latitude", ""),
                    "sunrise": get_text(item, "sunrise", ""),
                    "suntransit": get_text(item, "suntransit", ""),  # 일중
                    "sunset": get_text(item, "sunset", ""),
                    "moonrise": get_text(item, "moonrise", ""),
                    "moontransit": get_text(item, "moontransit", ""),  # 월중
                    "moonset": get_text(item, "moonset", ""),
                    "civilm": get_text(item, "civilm", ""),  # 시민박명(아침)
                    "civile": get_text(item, "civile", ""),  # 시민박명(저녁)
                    "nautm": get_text(item, "nautm", ""),  # 항해박명(아침)
                    "naute": get_text(item, "naute", ""),  # 항해박명(저녁)
                    "astm": get_text(item, "astm", ""),  # 천문박명(아침)
                    "aste": get_text(item, "aste", ""),  # 천문박명(저녁)
                    "method": "api"
                }
            else:
                # JSON 응답 처리
                data = response.json()
                
                if "response" in data:
                    header = data["response"].get("header", {})
                    if header.get("resultCode") != "00":
                        error_msg = header.get("resultMsg", "알 수 없는 오류")
                        raise Exception(f"API 오류: {error_msg}")
                    
                    body = data["response"].get("body", {})
                    items = body.get("items", {})
                    
                    if isinstance(items, dict) and "item" in items:
                        item = items["item"]
                    else:
                        raise Exception("응답 데이터에 item이 없습니다.")
                    
                    return {
                        "date": item.get("locdate", date),
                        "location": item.get("location", location),
                        "sunrise": item.get("sunrise", ""),
                        "sunset": item.get("sunset", ""),
                        "moonrise": item.get("moonrise", ""),
                        "moonset": item.get("moonset", ""),
                        "method": "api"
                    }
                else:
                    raise Exception("예상하지 못한 API 응답 형식")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API 요청 실패: {str(e)}")
        except Exception as e:
            if "API 오류" in str(e) or "응답" in str(e):
                raise
            raise Exception(f"응답 파싱 실패: {str(e)}")
    
    def get_sun_moon_info_by_location(self, latitude: float, longitude: float, 
                                      date: Optional[str] = None, 
                                      dn_yn: str = "Y") -> Dict:
        """
        위치별 해달 출몰시각 정보조회 (위경도 기반)
        참고: https://www.data.go.kr/data/15012688/openapi.do
        
        Args:
            latitude: 위도 (실수 형식: 37.5666660 또는 도분 형식: 3700)
            longitude: 경도 (실수 형식: 126.9833330 또는 도분 형식: 12800)
            date: 조회 날짜 (YYYYMMDD), None이면 오늘
            dn_yn: 10진수 여부 ("Y": 실수 형식, "N": 도분 형식)
                   실수 형식 예: 129.1257996, 35.3694613
                   도분 형식 예: 127도 05분 -> 12705
                   잘못 전달하면 동서남북 최외곽 지역이 조회됨
        
        Returns:
            일출몰 정보 딕셔너리
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        # 위경도 형식 변환
        if dn_yn == "Y":
            # 실수 형식 그대로 사용
            lat_str = str(latitude)
            lon_str = str(longitude)
        else:
            # 도분 형식으로 변환 (예: 37.5665 -> 3756, 126.978 -> 12697)
            lat_deg = int(latitude)
            lat_min = int((latitude - lat_deg) * 60)
            lat_str = f"{lat_deg:02d}{lat_min:02d}"
            
            lon_deg = int(longitude)
            lon_min = int((longitude - lon_deg) * 60)
            lon_str = f"{lon_deg:03d}{lon_min:02d}"
        
        params = {
            "serviceKey": self.service_key,
            "locdate": date,
            "longitude": lon_str,
            "latitude": lat_str,
            "dnYn": dn_yn
        }
        
        try:
            response = requests.get(self.LOCATION_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # 응답이 XML인 경우 처리 (공공데이터포털 기본 형식)
            content_type = response.headers.get('Content-Type', '').lower()
            response_text = response.text
            
            if 'xml' in content_type or response_text.strip().startswith('<?xml'):
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response_text)
                
                # XML 응답 파싱
                header = root.find('header')
                if header is not None:
                    result_code = header.find('resultCode')
                    if result_code is not None and result_code.text != "00":
                        result_msg = header.find('resultMsg')
                        error_msg = result_msg.text if result_msg is not None else "알 수 없는 오류"
                        raise Exception(f"API 오류: {error_msg}")
                
                body = root.find('body')
                if body is None:
                    raise Exception("응답에 body가 없습니다.")
                
                items = body.find('items')
                if items is None:
                    raise Exception("응답에 items가 없습니다.")
                
                item = items.find('item')
                if item is None:
                    raise Exception("응답에 item이 없습니다.")
                
                # XML 요소에서 데이터 추출
                def get_text(elem, tag, default=""):
                    found = elem.find(tag)
                    if found is not None and found.text:
                        return found.text.strip()
                    return default
                
                return {
                    "date": get_text(item, "locdate", date),
                    "location": get_text(item, "location", ""),
                    "longitude": get_text(item, "longitude", ""),
                    "longitudeNum": get_text(item, "longitudeNum", ""),
                    "latitude": get_text(item, "latitude", ""),
                    "latitudeNum": get_text(item, "latitudeNum", ""),
                    "sunrise": get_text(item, "sunrise", ""),
                    "suntransit": get_text(item, "suntransit", ""),  # 일중
                    "sunset": get_text(item, "sunset", ""),
                    "moonrise": get_text(item, "moonrise", ""),
                    "moontransit": get_text(item, "moontransit", ""),  # 월중
                    "moonset": get_text(item, "moonset", ""),
                    "civilm": get_text(item, "civilm", ""),  # 시민박명(아침)
                    "civile": get_text(item, "civile", ""),  # 시민박명(저녁)
                    "nautm": get_text(item, "nautm", ""),  # 항해박명(아침)
                    "naute": get_text(item, "naute", ""),  # 항해박명(저녁)
                    "astm": get_text(item, "astm", ""),  # 천문박명(아침)
                    "aste": get_text(item, "aste", ""),  # 천문박명(저녁)
                    "method": "api_location"
                }
            else:
                # JSON 응답 처리
                data = response.json()
                
                if "response" in data:
                    header = data["response"].get("header", {})
                    if header.get("resultCode") != "00":
                        error_msg = header.get("resultMsg", "알 수 없는 오류")
                        raise Exception(f"API 오류: {error_msg}")
                    
                    body = data["response"].get("body", {})
                    items = body.get("items", {})
                    
                    if isinstance(items, dict) and "item" in items:
                        item = items["item"]
                    else:
                        raise Exception("응답 데이터에 item이 없습니다.")
                    
                    return {
                        "date": item.get("locdate", date),
                        "location": item.get("location", ""),
                        "longitude": item.get("longitude", ""),
                        "longitudeNum": item.get("longitudeNum", ""),
                        "latitude": item.get("latitude", ""),
                        "latitudeNum": item.get("latitudeNum", ""),
                        "sunrise": item.get("sunrise", ""),
                        "suntransit": item.get("suntransit", ""),
                        "sunset": item.get("sunset", ""),
                        "moonrise": item.get("moonrise", ""),
                        "moontransit": item.get("moontransit", ""),
                        "moonset": item.get("moonset", ""),
                        "civilm": item.get("civilm", ""),
                        "civile": item.get("civile", ""),
                        "nautm": item.get("nautm", ""),
                        "naute": item.get("naute", ""),
                        "astm": item.get("astm", ""),
                        "aste": item.get("aste", ""),
                        "method": "api_location"
                    }
                else:
                    raise Exception("예상하지 못한 API 응답 형식")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API 요청 실패: {str(e)}")
        except Exception as e:
            if "API 오류" in str(e) or "응답" in str(e):
                raise
            raise Exception(f"응답 파싱 실패: {str(e)}")


class TideAPI:
    """조력(조수) 정보 API 클래스 - 공공데이터포털 조석 예측 정보
    
    공공데이터포털의 조석 예측 정보 API를 사용하여 조력(조수) 정보를 수집합니다.
    
    참고 문서: https://www.data.go.kr/data/15142507/openapi.do
    
    API 정보:
    - Base URL: apis.data.go.kr/1192136/surveyTideLevel
    - API 엔드포인트: GetSurveyTideLevelApiService
    - 응답 형식: JSON (GetSurveyTideLevelApiService_response)
    
    제공 데이터:
    - tdlvHgt: 예측조위 (m)
    - bscTdlvHgt: 실측조위 (m)
    - obsvtrNm: 관측소명
    - obsrvnDt: 관측일시
    - lat: 위도
    - lot: 경도
    
    사용 예시:
        tide_api = TideAPI(service_key="your_api_key")
        tide_data = tide_api.get_tide_info(location_code="DT_0001", date="20260130")
    """
    
    # 공공데이터포털 조석 예측 정보 API
    BASE_URL = "http://apis.data.go.kr/1192136/surveyTideLevel/GetSurveyTideLevelApiService"
    
    def __init__(self, service_key: str):
        """
        Args:
            service_key: 공공데이터포털 조석 예측 정보 API 인증키
        """
        # API 키가 이미 URL 인코딩되어 있으면 디코딩
        if service_key:
            self.service_key = unquote(service_key) if '%' in service_key else service_key
        else:
            self.service_key = None
    
    def get_tide_info(self, location_code: str = None, date: Optional[str] = None, 
                     hourly: bool = True, future_only: bool = False,
                     page_no: int = 1, num_of_rows: int = 300, time_interval: int = None) -> List[Dict]:
        """
        조력 정보 조회 (공공데이터포털 조석 예측 정보)
        참고: https://www.data.go.kr/data/15142507/openapi.do
        
        Args:
            location_code: 관측소 코드 (공공데이터포털 문서 참고, None이면 전체)
            date: 조회 날짜 (YYYYMMDD), None이면 오늘
            hourly: True이면 시간 단위로 필터링 (기본값: True)
            future_only: True이면 미래 시간만 반환 (기본값: False)
            
        Returns:
            조력 정보 리스트 (시간 단위로 필터링됨)
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        # 공공데이터포털 조석 예측 정보 API 파라미터
        # API 문서에 따른 정확한 파라미터 사용
        params = {
            "serviceKey": self.service_key if self.service_key else "",
            "type": "json",  # json/xml 중 선택
            "pageNo": "1",
            "numOfRows": "300"  # 최대값: 300
        }
        
        # 필수 파라미터: obsCode
        if location_code:
            params["obsCode"] = location_code
        else:
            raise Exception("관측소 코드(obsCode)는 필수입니다.")
        
        # 선택 파라미터: reqDate (조위관측일, 기본값: 현재일시)
        if date:
            params["reqDate"] = date
        
        # 선택 파라미터: pageNo, numOfRows
        params["pageNo"] = str(page_no)
        params["numOfRows"] = str(min(num_of_rows, 300))  # 최대값: 300
        
        # 선택 파라미터: min (출력되는 시간 간격, 기본값: 1)
        if time_interval is not None:
            params["min"] = str(time_interval)
        elif hourly:
            params["min"] = "60"  # 60분 = 1시간 간격
        else:
            params["min"] = "1"  # 1분 간격
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # 응답이 XML인 경우 처리
            content_type = response.headers.get('Content-Type', '').lower()
            response_text = response.text
            
            if 'xml' in content_type or response_text.strip().startswith('<?xml'):
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response_text)
                
                # XML 응답 파싱
                header = root.find('header')
                if header is not None:
                    result_code = header.find('resultCode')
                    if result_code is not None and result_code.text != "00":
                        result_msg = header.find('resultMsg')
                        error_msg = result_msg.text if result_msg is not None else "알 수 없는 오류"
                        raise Exception(f"API 오류: {error_msg}")
                
                body = root.find('body')
                if body is None:
                    raise Exception("응답에 body가 없습니다.")
                
                items = body.find('items')
                if items is None:
                    raise Exception("응답에 items가 없습니다.")
                
                tide_data = []
                item_list = items.findall('item')
                
                def get_text(elem, tag, default=""):
                    found = elem.find(tag)
                    return found.text if found is not None and found.text else default
                
                def get_float(elem, tag, default=0.0):
                    text = get_text(elem, tag, str(default))
                    try:
                        return float(text)
                    except:
                        return default
                
                for item in item_list:
                    tide_data.append({
                        "time": get_text(item, "obsrvnDt", ""),  # 관측일시
                        "tide_level": get_float(item, "tdlvHgt", 0.0),  # 예측조위
                        "measured_tide_level": get_float(item, "bscTdlvHgt", 0.0),  # 실측조위
                        "obs_name": get_text(item, "obsvtrNm", ""),  # 관측소
                        "latitude": get_float(item, "lat", 0.0),  # 위도
                        "longitude": get_float(item, "lot", 0.0)  # 경도
                    })
                
                # 시간 단위로 필터링
                if hourly:
                    tide_data = self._filter_hourly(tide_data)
                
                # 미래 시간만 필터링
                if future_only:
                    tide_data = self._filter_future_only(tide_data)
                
                return tide_data
            else:
                # JSON 응답 처리
                data = response.json()
                
                # 응답 구조 확인
                # 가능한 형식:
                # 1. GetSurveyTideLevelApiService_response
                # 2. response
                # 3. 직접 header, body 구조
                response_data = None
                
                if "GetSurveyTideLevelApiService_response" in data:
                    response_data = data["GetSurveyTideLevelApiService_response"]
                elif "response" in data:
                    response_data = data["response"]
                elif "header" in data and "body" in data:
                    # 직접 header, body 구조인 경우
                    response_data = data
                else:
                    raise Exception(f"예상하지 못한 API 응답 형식: {list(data.keys())}")
                
                header = response_data.get("header", {})
                result_code = header.get("resultCode", "")
                if result_code != "00":
                    error_msg = header.get("resultMsg", "알 수 없는 오류")
                    # 데이터가 없는 경우 (예: 페이지 범위 초과) 빈 리스트 반환
                    if result_code in ["03", "NODATA_ERROR"]:
                        return []
                    raise Exception(f"API 오류 (코드: {result_code}): {error_msg}")
                
                body = response_data.get("body", {})
                if not body:
                    # body가 없으면 빈 리스트 반환
                    return []
                
                items = body.get("items", {})
                
                if isinstance(items, dict) and "item" in items:
                    item_list = items["item"]
                    if not isinstance(item_list, list):
                        item_list = [item_list]
                else:
                    item_list = []
                
                tide_data = []
                for item in item_list:
                    tide_data.append({
                        "time": item.get("obsrvnDt", ""),  # 관측일시
                        "tide_level": float(item.get("tdlvHgt", 0)),  # 예측조위
                        "measured_tide_level": float(item.get("bscTdlvHgt", 0)),  # 실측조위
                        "obs_name": item.get("obsvtrNm", ""),  # 관측소
                        "latitude": float(item.get("lat", 0)),  # 위도
                        "longitude": float(item.get("lot", 0))  # 경도
                    })
                
                # 시간 단위로 필터링
                if hourly:
                    tide_data = self._filter_hourly(tide_data)
                
                # 미래 시간만 필터링
                if future_only:
                    tide_data = self._filter_future_only(tide_data)
                
                return tide_data
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    # 여러 가능한 응답 형식 시도
                    if "GetSurveyTideLevelApiService_response" in error_data:
                        header = error_data["GetSurveyTideLevelApiService_response"].get("header", {})
                    elif "response" in error_data:
                        header = error_data["response"].get("header", {})
                    else:
                        header = {}
                    error_msg = header.get("resultMsg", str(e))
                    raise Exception(f"API 요청 실패: {error_msg}")
                except:
                    raise Exception(f"API 요청 실패: {str(e)}, 응답: {e.response.text[:200] if e.response else 'N/A'}")
            else:
                raise Exception(f"API 요청 실패: {str(e)}")
        except Exception as e:
            if "API 오류" in str(e) or "응답" in str(e):
                raise
            raise Exception(f"응답 파싱 실패: {str(e)}")
    
    def _filter_hourly(self, tide_data: List[Dict]) -> List[Dict]:
        """
        조력 데이터를 시간 단위로 필터링 (매 시간 정각 데이터만 선택)
        
        Args:
            tide_data: 원본 조력 데이터 리스트
            
        Returns:
            시간 단위로 필터링된 조력 데이터 리스트
        """
        if not tide_data:
            return tide_data
        
        # 시간별로 그룹화하고 각 시간의 첫 번째 데이터(정각) 선택
        hourly_data = {}
        
        for item in tide_data:
            time_str = item.get("time", "")
            if not time_str:
                continue
            
            try:
                # 시간 문자열 파싱 (형식: "YYYY-MM-DD HH:MM" 또는 "YYYYMMDDHHMM")
                if " " in time_str:
                    # "2026-01-30 00:00" 형식
                    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                elif len(time_str) == 12:
                    # "202601300000" 형식
                    dt = datetime.strptime(time_str, "%Y%m%d%H%M")
                elif len(time_str) == 14:
                    # "2026-01-30 00:00:00" 형식
                    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                else:
                    continue
                
                # 시간 키 생성 (YYYY-MM-DD HH:00 형식)
                hour_key = dt.strftime("%Y-%m-%d %H:00")
                
                # 해당 시간의 첫 번째 데이터만 저장
                if hour_key not in hourly_data:
                    hourly_data[hour_key] = item.copy()
                    hourly_data[hour_key]["time"] = hour_key
                    
            except ValueError:
                # 파싱 실패 시 스킵
                continue
        
        # 시간 순서로 정렬
        sorted_data = sorted(hourly_data.values(), key=lambda x: x.get("time", ""))
        return sorted_data
    
    def _filter_future_only(self, tide_data: List[Dict]) -> List[Dict]:
        """
        미래 시간만 필터링 (현재 시간 이후의 데이터만 반환)
        
        Args:
            tide_data: 조력 데이터 리스트
            
        Returns:
            미래 시간만 포함된 조력 데이터 리스트
        """
        if not tide_data:
            return tide_data
        
        now = datetime.now()
        future_data = []
        
        for item in tide_data:
            time_str = item.get("time", "")
            if not time_str:
                continue
            
            try:
                # 시간 문자열 파싱
                if " " in time_str:
                    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                elif len(time_str) == 12:
                    dt = datetime.strptime(time_str, "%Y%m%d%H%M")
                elif len(time_str) == 14:
                    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                else:
                    continue
                
                # 현재 시간 이후인지 확인
                if dt > now:
                    future_data.append(item)
                    
            except ValueError:
                continue
        
        return future_data


class TideForecastAPI:
    """조석 예보(고, 저조) API 클래스 - 해양수산부 국립해양조사원
    
    조석 예보지점의 고조, 저조 정보(조위, 시각)를 제공하는 API입니다.
    
    참고 문서: https://www.data.go.kr/data/15156018/openapi.do
    
    API 정보:
    - Base URL: apis.data.go.kr/1192136/tideFcstHghLw
    - API 엔드포인트: GetTideFcstHghLwApiService
    - 응답 형식: JSON/XML
    
    제공 데이터:
    - obsvtrNm: 예보지점명
    - lot: 예보지점 경도
    - lat: 예보지점 위도
    - predcDt: 예측일시
    - predcTdlvVl: 예측조위값(cm)
    - extrSe: 극치구분 (1: 오전 고조, 2: 오전 저조, 3: 오후 고조, 4: 오후 저조)
    
    사용 예시:
        forecast_api = TideForecastAPI(service_key="your_api_key")
        forecast_data = forecast_api.get_tide_forecast(obs_code="DT_0001", req_date="20260130")
    """
    
    # 공공데이터포털 조석 예보(고, 저조) API
    # 참고: https://www.data.go.kr/data/15156018/openapi.do
    BASE_URL = "http://apis.data.go.kr/1192136/tideFcstHghLw/GetTideFcstHghLwApiService"
    
    def __init__(self, service_key: str):
        """
        Args:
            service_key: 공공데이터포털 조석 예보 API 인증키
        """
        # API 키가 이미 URL 인코딩되어 있으면 디코딩
        if service_key:
            self.service_key = unquote(service_key) if '%' in service_key else service_key
        else:
            self.service_key = None
    
    def get_tide_forecast(self, obs_code: str, req_date: Optional[str] = None,
                         page_no: int = 1, num_of_rows: int = 300) -> List[Dict]:
        """
        조석 예보(고, 저조) 정보 조회
        
        Args:
            obs_code: 예보지점 코드 (필수, 예: "DT_0001")
            req_date: 요청일자 (YYYYMMDD), None이면 오늘
            page_no: 페이지 번호 (기본값: 1)
            num_of_rows: 한 페이지 결과 수 (기본값: 300, 최대값: 300)
            
        Returns:
            조석 예보 정보 리스트 (고조, 저조 정보)
        """
        if req_date is None:
            req_date = datetime.now().strftime("%Y%m%d")
        
        # API 파라미터
        params = {
            "serviceKey": self.service_key if self.service_key else "",
            "type": "json",
            "pageNo": str(page_no),
            "numOfRows": str(min(num_of_rows, 300)),  # 최대값: 300
            "obsCode": obs_code,
            "reqDate": req_date
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            # 응답이 XML인 경우 처리
            content_type = response.headers.get('Content-Type', '').lower()
            response_text = response.text
            
            if 'xml' in content_type or response_text.strip().startswith('<?xml'):
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response_text)
                
                # XML 응답 파싱
                header = root.find('header')
                if header is not None:
                    result_code = header.find('resultCode')
                    if result_code is not None and result_code.text != "00":
                        result_msg = header.find('resultMsg')
                        error_msg = result_msg.text if result_msg is not None else "알 수 없는 오류"
                        raise Exception(f"API 오류: {error_msg}")
                
                body = root.find('body')
                if body is None:
                    return []
                
                items = body.find('items')
                if items is None:
                    return []
                
                forecast_data = []
                item_list = items.findall('item')
                
                def get_text(elem, tag, default=""):
                    found = elem.find(tag)
                    return found.text if found is not None and found.text else default
                
                def get_float(elem, tag, default=0.0):
                    text = get_text(elem, tag, str(default))
                    try:
                        return float(text)
                    except:
                        return default
                
                def get_int(elem, tag, default=0):
                    text = get_text(elem, tag, str(default))
                    try:
                        return int(text)
                    except:
                        return default
                
                for item in item_list:
                    extr_se = get_int(item, "extrSe", 0)
                    extr_se_name = {
                        1: "오전 고조",
                        2: "오전 저조",
                        3: "오후 고조",
                        4: "오후 저조"
                    }.get(extr_se, f"알 수 없음({extr_se})")
                    
                    forecast_data.append({
                        "obs_name": get_text(item, "obsvtrNm", ""),  # 예보지점명
                        "longitude": get_float(item, "lot", 0.0),  # 경도
                        "latitude": get_float(item, "lat", 0.0),  # 위도
                        "prediction_time": get_text(item, "predcDt", ""),  # 예측일시
                        "tide_level": get_float(item, "predcTdlvVl", 0.0),  # 예측조위값(cm)
                        "tide_type_code": extr_se,  # 극치구분 코드
                        "tide_type": extr_se_name  # 극치구분 이름
                    })
                
                return forecast_data
            else:
                # JSON 응답 처리
                data = response.json()
                
                # 응답 구조 확인
                response_data = None
                
                if "GetTideFcstHghLwApiService_response" in data:
                    response_data = data["GetTideFcstHghLwApiService_response"]
                elif "response" in data:
                    response_data = data["response"]
                elif "header" in data and "body" in data:
                    response_data = data
                else:
                    raise Exception(f"예상하지 못한 API 응답 형식: {list(data.keys())}")
                
                header = response_data.get("header", {})
                result_code = header.get("resultCode", "")
                if result_code != "00":
                    error_msg = header.get("resultMsg", "알 수 없는 오류")
                    # 데이터가 없는 경우 빈 리스트 반환
                    if result_code in ["03", "NODATA_ERROR"]:
                        return []
                    raise Exception(f"API 오류 (코드: {result_code}): {error_msg}")
                
                body = response_data.get("body", {})
                if not body:
                    return []
                
                items = body.get("items", {})
                
                if isinstance(items, dict) and "item" in items:
                    item_list = items["item"]
                    if not isinstance(item_list, list):
                        item_list = [item_list]
                else:
                    item_list = []
                
                forecast_data = []
                extr_se_names = {
                    1: "오전 고조",
                    2: "오전 저조",
                    3: "오후 고조",
                    4: "오후 저조"
                }
                
                for item in item_list:
                    extr_se = int(item.get("extrSe", 0))
                    forecast_data.append({
                        "obs_name": item.get("obsvtrNm", ""),  # 예보지점명
                        "longitude": float(item.get("lot", 0)),  # 경도
                        "latitude": float(item.get("lat", 0)),  # 위도
                        "prediction_time": item.get("predcDt", ""),  # 예측일시
                        "tide_level": float(item.get("predcTdlvVl", 0)),  # 예측조위값(cm)
                        "tide_type_code": extr_se,  # 극치구분 코드
                        "tide_type": extr_se_names.get(extr_se, f"알 수 없음({extr_se})")  # 극치구분 이름
                    })
                
                return forecast_data
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    if "GetTideFcstHghLwApiService_response" in error_data:
                        header = error_data["GetTideFcstHghLwApiService_response"].get("header", {})
                    elif "response" in error_data:
                        header = error_data["response"].get("header", {})
                    else:
                        header = {}
                    error_msg = header.get("resultMsg", str(e))
                    raise Exception(f"API 요청 실패: {error_msg}")
                except:
                    raise Exception(f"API 요청 실패: {str(e)}, 응답: {e.response.text[:200] if e.response else 'N/A'}")
            else:
                raise Exception(f"API 요청 실패: {str(e)}")
        except Exception as e:
            if "API 오류" in str(e) or "응답" in str(e):
                raise
            raise Exception(f"응답 파싱 실패: {str(e)}")


class TideForecastTimeAPI:
    """조석 예보(시계열) API 클래스 - 해양수산부 국립해양조사원
    
    조석 예보지점에 대한 시계열 조석 정보(조위, 시각)를 제공하는 API입니다.
    
    참고 문서: https://www.data.go.kr/data/15156022/openapi.do
    
    API 정보:
    - Base URL: apis.data.go.kr/1192136/tideFcstTime
    - API 엔드포인트: GetTideFcstTimeApiService
    - 응답 형식: JSON/XML
    
    제공 데이터:
    - obsvtrNm: 예보지점명
    - lot: 예보지점 경도
    - lat: 예보지점 위도
    - predcDt: 예측일시
    - tdlvHgt: 조위높이(cm)
    
    사용 예시:
        time_api = TideForecastTimeAPI(service_key="your_api_key")
        time_data = time_api.get_tide_forecast_time(obs_code="DT_0001", req_date="20260130")
    """
    
    # 공공데이터포털 조석 예보(시계열) API
    BASE_URL = "http://apis.data.go.kr/1192136/tideFcstTime/GetTideFcstTimeApiService"
    
    def __init__(self, service_key: str):
        """
        Args:
            service_key: 공공데이터포털 조석 예보(시계열) API 인증키
        """
        # API 키가 이미 URL 인코딩되어 있으면 디코딩
        if service_key:
            self.service_key = unquote(service_key) if '%' in service_key else service_key
        else:
            self.service_key = None
    
    def get_tide_forecast_time(self, obs_code: str, req_date: Optional[str] = None,
                               page_no: int = 1, num_of_rows: int = 300, 
                               time_interval: int = 1) -> List[Dict]:
        """
        조석 예보(시계열) 정보 조회
        
        Args:
            obs_code: 예보지점 코드 (필수, 예: "DT_0001")
            req_date: 요청일자 (YYYYMMDD), None이면 오늘
            page_no: 페이지 번호 (기본값: 1)
            num_of_rows: 한 페이지 결과 수 (기본값: 300, 최대값: 300)
            time_interval: 시간 간격(분) (기본값: 1, 최대값: 60)
            
        Returns:
            조석 예보 시계열 정보 리스트
        """
        if req_date is None:
            req_date = datetime.now().strftime("%Y%m%d")
        
        # API 파라미터
        params = {
            "serviceKey": self.service_key if self.service_key else "",
            "type": "json",
            "pageNo": str(page_no),
            "numOfRows": str(min(num_of_rows, 300)),  # 최대값: 300
            "obsCode": obs_code,
            "reqDate": req_date,
            "min": str(min(time_interval, 60))  # 최대값: 60
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            
            # 403 오류인 경우 상세 메시지 제공
            if response.status_code == 403:
                error_msg = "API 권한 오류 (403 Forbidden)"
                if response.text.strip().startswith('<?xml'):
                    import xml.etree.ElementTree as ET
                    try:
                        root = ET.fromstring(response.text)
                        result_msg = root.find('.//resultMsg')
                        if result_msg is not None:
                            error_msg = f"API 권한 오류: {result_msg.text}"
                    except:
                        pass
                raise Exception(f"{error_msg}. 조석 예보(시계열) API는 별도의 API 키가 필요할 수 있습니다. 공공데이터포털(https://www.data.go.kr/data/15156022/openapi.do)에서 해당 서비스에 대한 활용신청을 확인하세요.")
            
            response.raise_for_status()
            
            # 응답이 XML인 경우 처리
            content_type = response.headers.get('Content-Type', '').lower()
            response_text = response.text
            
            if 'xml' in content_type or response_text.strip().startswith('<?xml'):
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response_text)
                
                # XML 응답 파싱
                header = root.find('header')
                if header is not None:
                    result_code = header.find('resultCode')
                    if result_code is not None and result_code.text != "00":
                        result_msg = header.find('resultMsg')
                        error_msg = result_msg.text if result_msg is not None else "알 수 없는 오류"
                        raise Exception(f"API 오류: {error_msg}")
                
                body = root.find('body')
                if body is None:
                    return []
                
                items = body.find('items')
                if items is None:
                    return []
                
                forecast_data = []
                item_list = items.findall('item')
                
                def get_text(elem, tag, default=""):
                    found = elem.find(tag)
                    return found.text if found is not None and found.text else default
                
                def get_float(elem, tag, default=0.0):
                    text = get_text(elem, tag, str(default))
                    try:
                        return float(text)
                    except:
                        return default
                
                for item in item_list:
                    forecast_data.append({
                        "obs_name": get_text(item, "obsvtrNm", ""),  # 예보지점명
                        "longitude": get_float(item, "lot", 0.0),  # 경도
                        "latitude": get_float(item, "lat", 0.0),  # 위도
                        "prediction_time": get_text(item, "predcDt", ""),  # 예측일시
                        "tide_level": get_float(item, "tdlvHgt", 0.0)  # 조위높이(cm)
                    })
                
                return forecast_data
            else:
                # JSON 응답 처리
                data = response.json()
                
                # 응답 구조 확인
                response_data = None
                
                if "GetTideFcstTimeApiService_response" in data:
                    response_data = data["GetTideFcstTimeApiService_response"]
                elif "response" in data:
                    response_data = data["response"]
                elif "header" in data and "body" in data:
                    response_data = data
                else:
                    raise Exception(f"예상하지 못한 API 응답 형식: {list(data.keys())}")
                
                header = response_data.get("header", {})
                result_code = header.get("resultCode", "")
                if result_code != "00":
                    error_msg = header.get("resultMsg", "알 수 없는 오류")
                    # 데이터가 없는 경우 빈 리스트 반환
                    if result_code in ["03", "NODATA_ERROR"]:
                        return []
                    raise Exception(f"API 오류 (코드: {result_code}): {error_msg}")
                
                body = response_data.get("body", {})
                if not body:
                    return []
                
                items = body.get("items", {})
                
                if isinstance(items, dict) and "item" in items:
                    item_list = items["item"]
                    if not isinstance(item_list, list):
                        item_list = [item_list]
                else:
                    item_list = []
                
                forecast_data = []
                for item in item_list:
                    forecast_data.append({
                        "obs_name": item.get("obsvtrNm", ""),  # 예보지점명
                        "longitude": float(item.get("lot", 0)),  # 경도
                        "latitude": float(item.get("lat", 0)),  # 위도
                        "prediction_time": item.get("predcDt", ""),  # 예측일시
                        "tide_level": float(item.get("tdlvHgt", 0))  # 조위높이(cm)
                    })
                
                return forecast_data
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    if "GetTideFcstTimeApiService_response" in error_data:
                        header = error_data["GetTideFcstTimeApiService_response"].get("header", {})
                    elif "response" in error_data:
                        header = error_data["response"].get("header", {})
                    else:
                        header = {}
                    error_msg = header.get("resultMsg", str(e))
                    raise Exception(f"API 요청 실패: {error_msg}")
                except:
                    raise Exception(f"API 요청 실패: {str(e)}, 응답: {e.response.text[:200] if e.response else 'N/A'}")
            else:
                raise Exception(f"API 요청 실패: {str(e)}")
        except Exception as e:
            if "API 오류" in str(e) or "응답" in str(e):
                raise
            raise Exception(f"응답 파싱 실패: {str(e)}")


class WeatherDataCollector:
    """통합 기상 데이터 수집 클래스"""
    
    def __init__(self, weather_service_key: str, 
                 sun_moon_service_key: Optional[str] = None,
                 tide_service_key: Optional[str] = None):
        """
        Args:
            weather_service_key: 기상청 API 인증키
            sun_moon_service_key: 일출몰 API 인증키 (선택)
            tide_service_key: 조력 API 인증키 (선택, 조석 시계열 API에 사용)
        """
        self.weather_api = WeatherAPI(weather_service_key)
        self.sun_moon_api = SunMoonAPI(sun_moon_service_key) if sun_moon_service_key else None
        self.tide_api = TideAPI(tide_service_key) if tide_service_key else None
        # 조석 시계열 API를 기본으로 사용
        self.tide_forecast_time_api = TideForecastTimeAPI(tide_service_key) if tide_service_key else None
    
    def get_comprehensive_forecast(self, latitude: float, longitude: float, 
                                   location: str = "서울") -> Dict:
        """
        24시간 종합 예보 데이터 조회
        
        Args:
            latitude: 위도
            longitude: 경도
            location: 지역명 (일출몰 조회용)
            
        Returns:
            종합 예보 데이터 딕셔너리
        """
        result = {
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "name": location
            },
            "forecast_24h": [],
            "sun_moon_info": None,
            "tide_info": None
        }
        
        # 24시간 예보
        result["forecast_24h"] = self.weather_api.get_24hour_forecast(latitude, longitude)
        
        # 일출몰 정보
        # 1순위: 위경도 기반 API (더 정확)
        # 2순위: 지역명 기반 API
        # 3순위: 계산 방법
        if self.sun_moon_api:
            try:
                # 위경도 기반 API 시도 (실수 형식 사용)
                result["sun_moon_info"] = self.sun_moon_api.get_sun_moon_info_by_location(
                    latitude, longitude, dn_yn="Y"
                )
            except Exception as e1:
                try:
                    # 지역명 기반 API 시도
                    result["sun_moon_info"] = self.sun_moon_api.get_sun_moon_info(location)
                except Exception as e2:
                    # API 실패 시 계산 방법 사용
                    try:
                        result["sun_moon_info"] = self.sun_moon_api.calculate_sunrise_sunset(
                            latitude, longitude
                        )
                    except Exception as e3:
                        print(f"일출몰 정보 조회 실패 (위경도: {e1}, 지역명: {e2}, 계산: {e3})")
        else:
            # API 키가 없어도 계산 방법 사용
            try:
                sun_moon_calc = SunMoonAPI("")
                result["sun_moon_info"] = sun_moon_calc.calculate_sunrise_sunset(
                    latitude, longitude
                )
            except Exception as e:
                print(f"일출몰 시간 계산 실패: {e}")
        
        # 조력 정보 (조석 시계열 API 사용 - 실행 시간 기준 24시간)
        if self.tide_forecast_time_api:
            try:
                # 인천 관측소 코드 예시 (실제로는 위치에 맞게 변경 필요)
                # 실행 시간 기준으로 24시간의 미래 예측 데이터 수집
                now = datetime.now()
                target_time = now + timedelta(hours=24)
                
                # 오늘과 내일 날짜로 조회하여 24시간 데이터 확보
                today = now.strftime("%Y%m%d")
                tomorrow = (now + timedelta(days=1)).strftime("%Y%m%d")
                
                all_tide_data = []
                
                # 오늘 날짜 조회
                try:
                    today_data = self.tide_forecast_time_api.get_tide_forecast_time(
                        obs_code="DT_0001",  # 인천 관측소 (실제로는 위치에 맞게 변경 필요)
                        req_date=today,
                        time_interval=60,  # 1시간 간격
                        num_of_rows=300
                    )
                    all_tide_data.extend(today_data)
                except Exception as e:
                    print(f"오늘 조력 데이터 조회 실패: {e}")
                
                # 내일 날짜도 조회 (24시간을 채우기 위해)
                try:
                    tomorrow_data = self.tide_forecast_time_api.get_tide_forecast_time(
                        obs_code="DT_0001",
                        req_date=tomorrow,
                        time_interval=60,
                        num_of_rows=300
                    )
                    all_tide_data.extend(tomorrow_data)
                except Exception as e:
                    print(f"내일 조력 데이터 조회 실패: {e}")
                
                # 현재 시간 기준으로 24시간 이후까지의 데이터만 필터링
                future_24h_data = []
                
                for item in all_tide_data:
                    pred_time = item.get("prediction_time", "")
                    if not pred_time:
                        continue
                    
                    try:
                        # 시간 문자열 파싱
                        if " " in pred_time:
                            pred_dt = datetime.strptime(pred_time, "%Y-%m-%d %H:%M")
                        elif len(pred_time) == 12:
                            pred_dt = datetime.strptime(pred_time, "%Y%m%d%H%M")
                        elif len(pred_time) == 14:
                            pred_dt = datetime.strptime(pred_time, "%Y-%m-%d %H:%M:%S")
                        else:
                            continue
                        
                        # 현재 시간 이후이고 24시간 이내인 데이터만 선택
                        if now < pred_dt <= target_time:
                            future_24h_data.append(item)
                    except ValueError:
                        continue
                
                # 시간 순으로 정렬
                future_24h_data.sort(key=lambda x: x.get("prediction_time", ""))
                
                result["tide_info"] = future_24h_data
                
            except Exception as e:
                print(f"조력 정보 조회 실패: {e}")
                # 실패 시 기존 TideAPI로 대체 시도
                if self.tide_api:
                    try:
                        result["tide_info"] = self.tide_api.get_tide_info(
                            "DT_0001",
                            hourly=True,
                            future_only=True
                        )
                    except Exception as e2:
                        print(f"기존 조력 API도 실패: {e2}")
        
        return result


if __name__ == "__main__":
    # 사용 예시
    # API 키는 공공데이터포털에서 발급받아야 합니다
    SERVICE_KEY = "YOUR_API_KEY_HERE"
    
    collector = WeatherDataCollector(
        weather_service_key=SERVICE_KEY,
        sun_moon_service_key=SERVICE_KEY,
        tide_service_key=None  # 조력 API는 별도 키 필요
    )
    
    # 서울 좌표
    seoul_lat = 37.5665
    seoul_lon = 126.9780
    
    try:
        data = collector.get_comprehensive_forecast(seoul_lat, seoul_lon, "서울")
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"오류 발생: {e}")
