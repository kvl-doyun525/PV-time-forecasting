# PV 발전 예측 모델 5종 비교 실험 계획

## 0. 관련 문서

| 문서 | 역할 |
|---|---|
| `pv_model_benchmark_plan.md` | **본 문서** — 실험 전체 설계 및 원칙 |
| `pv_model_benchmark_execution.md` | 구체적 수행 절차서 (단계별 실행 가이드) |

---

## 1. 문서 목적

이 문서는 동일한 데이터와 가능한 한 동일한 컴퓨팅 조건에서 다음 5개 모델 계열을 비교 실험하기 위한 계획서이다.

- SegRNN
- PatchTST
- Time-LLM
- LLaMA 계열 소형 모델
- GEMMA 계열 소형 모델

핵심 목표는 다음 3가지다.

1. **정확도 비교**: 동일한 PV/기상/달력/설비 정보를 사용했을 때 예측 정확도 비교
2. **운영성 비교**: GPU로 학습한 모델을 CPU 16코어 환경에서 실제 운영 가능한지 비교
3. **실무 의사결정**: 최종적으로 어떤 모델을 예측 본체로 채택하고, 어떤 모델을 설명/리포트 계층으로 분리할지 결정

---

## 2. 실험 범위 및 전제

### 2.1 공통 전제

- 데이터 도메인: **태양광 발전(PV)**
- 데이터 해상도: **1시간 단위**
- 입력 길이: **과거 1년(8760시간)**
- 학습 데이터: 최소 **2년 이상**, site 수 **10개 이상** 확보를 기준으로 한다
- 단일 site 데이터만 존재하는 경우 sliding window로 sample 수 확대 허용
- 학습 환경: **GPU (RTX 4080 또는 4090)**
- 추론 환경: **최신 CPU 16코어**, GPU 없이 CPU만 사용
- 개발 언어/프레임워크: **Python + PyTorch 중심**
- OS 후보: Windows 또는 Ubuntu(Docker), 단 **공식 벤치마크 기준 환경은 Ubuntu + Docker**를 우선 권장

### 2.2 비교 대상 모델

#### A. 시계열 전용 모델
- **SegRNN**
- **PatchTST**

#### B. LLM/LLM-reprogramming 계열
- **Time-LLM**
- **LLaMA 계열 소형 모델**
- **GEMMA 계열 소형 모델**

### 2.3 LLM 모델 선정 원칙

사용 모델은 반드시 다음 조건을 만족하도록 선정한다.

- 공식 문서 또는 공식 모델 카드에서 **CPU / edge / mobile / on-device** 실행 적합성이 언급된 최신 소형 모델일 것
- Python + PyTorch + Hugging Face 또는 공식 런타임으로 재현 가능할 것
- GPU 학습 및 CPU 추론 파이프라인 분리가 가능할 것

**1차 후보**

- **LLaMA**: `Llama 3.2 1B-Instruct`
- **GEMMA**: `Gemma 4 E2B-it`
- **Time-LLM backbone**: 공식 지원하는 작은 backbone 우선 (`GPT-2` 1차, 필요 시 `BERT` 추가)

> 참고: Time-LLM은 현재 공식 구현에서 Llama-7B 외에 GPT-2와 BERT도 지원한다. CPU 비교군에서는 작은 backbone을 우선 적용하는 것이 현실적이다.

---

## 3. 왜 Ubuntu + Docker를 기준 환경으로 두는가

실험 자체는 Windows에서도 가능하지만, **최종 기준 환경은 Ubuntu + Docker**를 권장한다.

### 이유

1. **재현성**
   - 컨테이너 단위로 모델 family별 의존성 충돌을 분리하기 쉽다.

2. **GPU 학습 구성의 명확성**
   - NVIDIA Container Toolkit은 Linux 기준 설치 및 운영 가이드가 가장 안정적이다.

3. **Windows 편차 감소**
   - Windows + WSL2 + Docker Desktop은 가능하지만, 파일시스템/런타임 차이로 성능 편차가 생길 수 있다.

### 권장 운영 방식

- **공식 비교/리더보드 채택 결과**: Ubuntu + Docker 결과만 채택
- **개발 편의 환경**: Windows + WSL2 + Docker 허용
- **학습/추론 명령 체계**: 가능하면 Docker Compose 또는 Makefile로 통일

---

## 4. 데이터 소스 계획

이번 실험의 가장 중요한 포인트는 **동일한 데이터 계약(data contract)** 을 먼저 고정하는 것이다.

### 4.1 사내 DB에서 수집할 데이터

필수 컬럼 예시:

- `site_id`
- `timestamp` (KST 기준 1시간 단위)
- `pv_power_kw` 또는 발전량
- `capacity_kw` (설비 정격 용량)
- 발전 관련 부가 feature
  - 인버터/PCS 상태
  - 설비 운영 상태
  - curtailment 관련 정보
  - 유지보수/장애 flag
  - site 메타정보
- 지역 정보
  - 주소
  - 위도/경도
  - 행정구역

### 4.2 기상청에서 수집할 데이터

#### 관측 데이터
- **ASOS 시간자료**: 기온, 강수량, 기압, 습도, 일사, 일조 등
- **AWS 시간자료**: 지역 촘촘도 보강용

#### 예보 데이터
- **기상청 단기예보 조회서비스(2.0)**
  - 1시간 단위 상세 예보
  - 격자 기반 예보
  - 운영 시점 기준 미래 horizon 구성에 사용

#### 지점/구역 정보
- site 좌표를 **ASOS/AWS 지점** 또는 **예보 격자**에 매핑하기 위한 구역/지점 정보

### 4.3 추가 계산 feature

- **pvlib 기반 태양 위치**
  - solar zenith / elevation / azimuth
- **clear-sky irradiance**
- 일출/일몰 시간
- daytime mask

### 4.4 데이터 전처리 정책

결측값과 이상치 처리 원칙을 미리 고정하지 않으면 모델 간 비교가 불공정해진다.

#### 결측값 처리
- **PV 발전량**: 연속 결측 1시간 이하 → 선형 보간. 초과 시 해당 구간 학습 제외 (마스킹)
- **기상 관측**: 인근 지점 값 또는 ERA5 보완 허용. 단, 보완 여부를 flag 컬럼으로 기록
- **기상 예보**: 예보 자체 결측은 이전 issue run으로 대체 허용

#### 이상치 처리
- **PV 발전량**: `capacity_kw` 초과 값 → 클리핑 후 flag 기록
- **야간 발전**: 일출~일몰 외 구간의 비정상 양수 값 → 0 클리핑
- **기상 관측**: 물리적 허용 범위 초과 값(예: 기온 > 50℃) → 결측 처리 후 보간

#### 정규화
- 모델 입력의 PV 발전량은 `pv_power_kw / capacity_kw` (0~1 범위, 1 초과 시 클리핑) 사용
- 기상 피처는 train 구간 통계(mean/std)로 z-score 정규화 후 test에 동일 적용

---

## 5. 가장 중요한 데이터 원칙: forecast leakage 금지

이 프로젝트에서 가장 중요한 실험 통제 규칙은 아래 한 줄이다.

> **미래 시점의 weather feature는 반드시 예측 시점 이전에 발표된 예보만 사용한다.**

즉, 예측 시점 `t0`에서 `t0+1 ~ t0+H`를 예측할 때:

- 미래 관측값을 feature로 사용하면 안 된다.
- `issue_time <= t0` 인 최신 예보만 사용한다.
- 학습/검증/테스트 모두 동일한 규칙을 지킨다.

### 구현 규칙

1. 각 예측 시점 `t0`를 기준으로
2. `issue_time <= t0` 를 만족하는 최신 forecast run을 선택하고
3. 그 run 안에서 `target_time in (t0+1, ..., t0+H)` 를 join한다.

이 규칙을 위반하면 오프라인 성능이 과대평가되어 운영 재현성이 크게 떨어진다.

---

## 6. 데이터 파이프라인 설계

### 6.1 원천 테이블

예시 테이블 구조:

- `pv_raw_hourly`
- `plant_meta`
- `kma_obs_asos_hourly`
- `kma_obs_aws_hourly`
- `kma_fcst_shortterm`
- `site_to_kma_grid`

### 6.2 Feature Mart (공통 학습 데이터셋)

최종 모델 입력용 키는 아래처럼 정의한다.

- `site_id`
- `base_time`
- `horizon`

각 row는 다음 정보를 가진다.

#### 과거 시계열
- 과거 1년 PV 발전량
- 과거 weather observation
- 과거 설비 상태/이벤트 정보

#### 미래 입력
- 미래 24h / 48h / 72h weather forecast
- future calendar feature
- future solar position / clear-sky feature

#### 메타 데이터
- 설비 정격용량
- 지역/위도/경도
- site category

#### target
- `normalized_power = pv_power_kw / capacity_kw`
- 필요 시 `pv_power_kw`도 병행 저장

### 6.3 추천 feature 예시

#### 과거 feature
- PV 발전량
- 기온, 습도, 풍속, 풍향, 기압
- 강수량
- 일사/일조 (가용 시)
- 구름량/운량 계열
- 설비 상태 flag
- curtailment / outage flag

#### 미래 feature
- 예보 기온
- 예보 강수형태/강수확률/강수량
- 예보 풍속/풍향
- 예보 하늘상태/운량
- 미래 solar position / clear-sky irradiance
- 달력 feature (hour, weekday, month, holiday)

#### 파생 feature
- 최근 24h / 72h / 168h rolling 통계
- 전일 동일시각 값
- 전주 동일시각 값
- capacity factor
- daytime/nighttime mask

---

## 7. 예측 horizon 및 운영 시나리오

1차 벤치마크에서는 horizon을 고정하는 것이 좋다.

### 권장 horizon
- **24시간**
- **48시간**
- **72시간**

### 운영 시나리오
- 하루 1회 예측
- KMA 단기예보 발행 시각을 반영해 **하루 1회 대표 실행 시각**을 정한다.
- PV 운영 관점에서는 **새벽 또는 이른 아침** 실행이 유리하다.

### 이유
- 24h는 가장 실무적인 기본 단위
- 48h/72h는 예보 오차 증가에 대한 모델 robustness를 보기 좋음
- KMA 단기예보의 공개 범위와 자연스럽게 연결됨

---

## 8. 데이터 분할 및 검증 전략

### 8.1 권장 split

시간 누수를 막기 위해 **시간 기반 분할**을 사용한다.

예시:

- Train: 과거 구간의 70%
- Valid: 이후 15%
- Test: 마지막 15%

또는 site generalization이 중요하면 아래 두 축을 같이 본다.

- **시간 일반화**: 같은 site의 미래 구간 예측
- **site 일반화**: unseen site 또는 일부 site 홀드아웃

### 8.2 추천 검증 방식

#### A. 기본 검증
- Train / Valid / Test 고정 분할

#### B. 운영형 검증
- **daily rolling backtest**
- 매일 같은 시각에 예측했다고 가정하고 결과 누적

### 8.3 seed 정책

- 각 실험은 최소 **3개 seed** 반복 (`seed: 42, 123, 2024` 고정)
- 리더보드는 평균과 표준편차 함께 기록
- LLM 계열은 generation temperature=0 (greedy) 로 고정하여 seed 의존성 최소화

### 8.4 최소 데이터 요구 사항

분할이 의미 있으려면 아래 조건을 만족해야 한다.

- **전체 기간**: 최소 **3년** 권장 (부득이한 경우 2년)
- **site 수**: 최소 **5개** (일반화 검증용 hold-out 가능 수준)
- **결측률**: 각 site 기준 **10% 미만** 권장, 초과 시 해당 site 제외 또는 별도 분류
- Train/Valid/Test 각 구간 모두 **계절 1순환(1년) 이상** 포함 권장

---

## 9. 모델별 실험 전략

## 9.1 SegRNN

### 역할
- CPU 환경에서 가장 유력한 실전형 예측 본체 후보

### 구현 방안
- 공식 SegRNN repo 또는 TSLib 포함 구현 사용

### 기대 특성
- 경량
- inference latency 우수
- 운영 단순성 우수

### 실험 포인트
- segment length 후보: 24, 48
- horizon별 latency 변화
- site 수 증가 시 batch 처리 효율

---

## 9.2 PatchTST

### 역할
- 정확도와 긴 입력 길이 대응을 함께 보는 주력 비교군

### 구현 방안
- TSLib 기반 구현 사용

### 기대 특성
- 긴 look-back 처리 유리
- 다변량 feature 대응력 우수
- CPU에서는 SegRNN보다 다소 느릴 가능성

### 실험 포인트
- patch length 후보: 24, 48
- patch stride 후보: 24, 48
- multivariate feature 수 변화에 따른 정확도/속도 변화

---

## 9.3 Time-LLM

### 역할
- LLM reprogramming 방식의 중간 지대 평가

### 구현 방안
- 공식 Time-LLM repo 사용
- 1차 backbone: `GPT-2`
- 2차 optional backbone: `BERT`
- Llama-7B backbone은 논문 재현용 보조 실험으로만 고려

### 기대 특성
- raw numeric text serialization 방식보다 빠름
- 그러나 frozen PLM forward 비용 때문에 SegRNN/PatchTST보다 무거울 가능성 높음

### 실험 포인트
- 작은 backbone에서 CPU latency가 실무 허용 범위인지 확인
- 같은 horizon에서 PatchTST 대비 정확도/latency trade-off 확인

---

## 9.4 LLaMA 계열

### 1차 후보
- **Llama 3.2 1B-Instruct**

### 역할
- 예측용 LLM 실험군
- 또는 설명/리포트 생성 레이어 후보

### 학습 방식
- full fine-tune 대신 **LoRA/QLoRA** 우선

### 입력/출력 형식
#### 입력
- 최근 168h raw PV
- 과거 1년 통계 요약
- 설비 정보
- 달력 정보
- 미래 weather forecast

#### 출력
- 다음 horizon의 normalized power를 담은 **고정 길이 JSON 배열**

예시:

```json
[0.00, 0.00, 0.01, 0.05, 0.17, 0.32, ...]
```

### 추가 평가 항목
- JSON 파싱 성공률
- 출력 길이 일치율
- 값 범위 위반율 (0 미만 또는 1 초과)

### 출력 파싱 실패 시 fallback 정책
1. **재시도**: 동일 프롬프트로 최대 2회 재생성
2. **부분 복구**: 파싱 가능한 부분만 사용하고 나머지는 직전 예측값으로 대체
3. **전체 실패**: 해당 시점은 `NaN` 처리 후 실패 건수를 리더보드에 별도 기록
4. 재시도 포함 latency도 측정 항목에 포함

---

## 9.5 GEMMA 계열

### 1차 후보
- **Gemma 4 E2B-it**

### 역할
- CPU / edge 친화적 최신 소형 LLM 실험군
- LLaMA 대비 CPU 추론 효율 비교군

### 학습 방식
- LoRA 또는 QLoRA 우선

### 입력/출력 형식
- LLaMA와 동일한 prompt / output JSON 구조 사용

### 실험 포인트
- 동일 프롬프트 구조에서 LLaMA 1B 대비 정확도/속도 비교
- CPU warm/cold start latency 비교

---

## 10. 공정한 비교를 위한 2개 트랙

LLM 계열과 시계열 전용 모델은 입력 표현이 다르므로, 아래 두 트랙으로 나누는 것이 공정하다.

### 트랙 A. 실전 배포 기준
각 모델이 가장 자연스러운 입력 형식을 사용한다.

- SegRNN / PatchTST: raw numeric time series
- Time-LLM: official reprogramming 방식
- LLaMA / GEMMA: 구조화된 prompt + JSON output

### 트랙 B. 공통 정보량 기준
모든 모델에 아래 정보를 동일하게 제공한다.

- 최근 168h raw series
- 과거 30일 일별 집계
- 과거 365일 통계 요약
- 미래 72h weather forecast
- 설비 및 calendar feature

### 권장 판단 기준
- **최종 운영 선택은 트랙 A를 우선**
- 트랙 B는 진단용/해석용으로 사용

---

## 11. 학습 환경 계획 (GPU)

### 공통 원칙
- 학습은 GPU 서버 또는 워크스테이션에서 수행
- GPU 후보: **RTX 4080 / RTX 4090**
- 모델 family별 의존성 충돌을 피하기 위해 **개별 컨테이너** 사용

### 추천 컨테이너 구성

| 컨테이너 | 포함 모델 | 기반 이미지 권장 |
|---|---|---|
| `env-tslib` | SegRNN, PatchTST, DLinear(baseline) | `pytorch/pytorch:2.3.x-cuda12.1-cudnn8-runtime` |
| `env-time-llm` | Time-LLM | `pytorch/pytorch:2.1.x-cuda12.1-cudnn8-runtime` |
| `env-llama` | Llama 3.2 1B + LoRA | `pytorch/pytorch:2.3.x-cuda12.1-cudnn8-devel` |
| `env-gemma` | Gemma 4 E2B + LoRA/QLoRA | `pytorch/pytorch:2.3.x-cuda12.1-cudnn8-devel` |

> Time-LLM은 transformers 버전 제약이 있으므로 별도 이미지에서 독립적으로 고정한다.

### 이유
- Time-LLM은 특정 torch / transformers 조합 요구
- Llama 계열과 Gemma 계열은 최신 transformers 요구 가능성 큼
- 한 환경에 전부 몰아넣으면 재현성 저하 가능

---

## 12. CPU 추론 벤치마크 계획

CPU 추론은 이번 프로젝트의 핵심 결과물이다.

### 12.1 하드웨어 고정
- 최신 CPU 16코어
- GPU 비활성화
- 동일 메모리 조건
- 동일 OS/컨테이너 설정
- `torch.set_num_threads(16)` 등 스레드 정책 고정

### 12.2 측정 항목

#### 지연(latency)
- **cold start latency**
  - 모델 로드 + 첫 추론
- **warm start latency**
  - 모델 로드 후 반복 추론
- p50 / p95 latency

#### 처리량(throughput)
- batch=1
- batch=N (예: 8, 16, 32)
- samples/sec

#### 메모리
- peak RSS RAM
- model size on disk
- 로드 직후 RAM

#### LLM 전용 추가 항목
- total generation time
- output parse success rate
- output validity rate

### 12.3 비교 방식

#### 온라인 운영 관점
- batch=1 latency 우선

#### 배치 운영 관점
- 동일 시점에 여러 site를 동시에 추론하는 throughput 측정

---

## 13. 평가 지표 설계

PV 예측은 야간이 0에 가깝기 때문에 일반 오차 지표만으로 판단하면 왜곡될 수 있다.

### 기본 지표
- MAE
- RMSE
- nRMSE (capacity 기준)
- sMAPE 또는 보조 지표

### PV 특화 지표
- **daytime MAE**
- **daytime nRMSE**
- **일 누적 발전량 오차**
- **site별 평균 오차**

### 운영성 지표
- CPU p50 latency
- CPU p95 latency
- peak RAM
- cold start time
- output validity

### 종합 의사결정 점수 예시
- 정확도: 50%
- CPU 지연: 20%
- 메모리 사용량: 10%
- 운영 복잡도: 20%

---

## 14. 반드시 포함할 baseline

본선 5개 모델 외에도 아래 baseline은 반드시 같이 실행한다.

- Seasonal Naive
- Persistence
- DLinear

### 이유
- 데이터 파이프라인 sanity check
- 과도한 모델 복잡도에 대한 기준선 확보
- 실제로는 단순 모델이 더 강할 가능성 점검

---

## 15. 추천 실험 순서

### 0단계: 계약 고정
- target horizon 확정 (24/48/72)
- 운영 예측 시각 확정
- 평가 지표 확정
- split 정책 확정

### 1단계: 데이터 수집 파이프라인
- 사내 DB snapshot 추출
- KMA 관측/예보 수집
- site ↔ KMA grid/station 매핑
- feature mart 생성

### 2단계: baseline 검증
- Seasonal Naive
- Persistence
- DLinear

### 3단계: 시계열 전용 모델
- SegRNN 학습/평가
- PatchTST 학습/평가

### 4단계: LLM-reprogramming
- Time-LLM(GPT-2 backbone) 학습/평가
- optional: BERT backbone 추가

### 5단계: 소형 LLM
- Llama 3.2 1B-Instruct + LoRA
- Gemma 4 E2B-it + LoRA/QLoRA

### 6단계: CPU 리더보드 확정
- accuracy + latency + RAM + 운영성 종합 판단
- 최종 후보 1~2개 선정

---

## 16. 코드/리포지토리 구조 제안

```text
project/
  docker/
    tslib/
    time_llm/
    llama/
    gemma/
  conf/
    data/
    model/
    benchmark/
  src/
    ingestion/
    weather/
    features/
    datasets/
    train/
    infer/
    benchmark/
    report/
  notebooks/
  artifacts/
  scripts/
  tests/
```

### 산출물 예시
- `dataset_snapshot/`
- `feature_mart/`
- `split_manifest.yaml`
- `training_runs/`
- `cpu_benchmark_report.json`
- `leaderboard.md`

---

## 17. 리스크 및 사전 대응

### 리스크 1. 기상 예보 join 오류
- 대응: `issue_time` 기반 조인 로직을 별도 테스트

### 리스크 2. site 좌표 ↔ KMA 지점/격자 매핑 오류
- 대응: 매핑 테이블을 별도 버전 관리

### 리스크 3. 컨테이너 의존성 충돌
- 대응: 모델 family별 이미지 분리

### 리스크 4. LLM output format 불안정
- 대응: JSON schema validation 및 재시도 정책 도입

### 리스크 5. horizon 증가에 따른 CPU 지연 급증
- 대응: batch=1 / batch=N / horizon별 latency 표 별도 작성

---

## 18. 최종 추천 방향

현 시점의 실험 우선순위는 아래와 같다.

1. **SegRNN**: CPU 추론 우선 후보
2. **PatchTST**: 정확도와 긴 입력 대응 주력 후보
3. **Time-LLM (small backbone)**: LLM-reprogramming 중간 실험군
4. **LLaMA 3.2 1B-Instruct**: 소형 LLM 비교군
5. **Gemma 4 E2B-it**: 최신 edge/on-device LLM 비교군

실무적으로 가장 가능성 높은 운영 구조는 아래일 가능성이 높다.

- **예측 본체**: SegRNN 또는 PatchTST
- **설명/보고/QA 레이어**: LLaMA 또는 GEMMA

단, 실제 선택은 반드시 아래 리더보드 이후 결정한다.

- 정확도 리더보드
- CPU latency 리더보드
- RAM/모델크기 리더보드
- 운영복잡도 점수

---

## 19. 바로 다음 액션 아이템

### 필수 결정 사항
1. OS 기준 환경을 Ubuntu + Docker로 확정할지 여부
2. 사내 DB 스키마/접속 방식 확인
3. site 위치 정보(좌표 또는 주소) 확보 여부 확인
4. KMA API key 및 사용 채널 확정
5. 대표 운영 예측 시각 확정

### 바로 착수 가능한 작업
1. DB 스키마 인벤토리 정리
2. KMA API 수집 스크립트 프로토타입 작성
3. site ↔ KMA grid/station 매핑 테이블 설계
4. feature mart 스키마 정의
5. baseline(DLinear/naive) 먼저 연결

---

## 20. 참고 자료

### 모델/라이브러리
- SegRNN 공식 저장소: <https://github.com/lss-1138/SegRNN>
- Time-Series-Library(TSLib): <https://github.com/thuml/Time-Series-Library>
- Time-LLM 공식 저장소: <https://github.com/KimMeen/Time-LLM>
- Llama 3.2 1B 모델 카드: <https://huggingface.co/meta-llama/Llama-3.2-1B>
- Meta Llama 3.2 모델 카드(GitHub): <https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md>
- Gemma 4 모델 카드: <https://ai.google.dev/gemma/docs/core/model_card_4>
- Gemma 4 소개: <https://deepmind.google/models/gemma/gemma-4/>

### 기상청 / 공공데이터
- 기상청 API 허브: <https://apihub.kma.go.kr/>
- 기상자료개방포털: <https://data.kma.go.kr/svc/main.do>
- ASOS 시간자료 조회서비스: <https://www.data.go.kr/data/15057210/openapi.do>
- AWS 기상관측자료 조회서비스: <https://www.data.go.kr/data/15057084/openapi.do>
- 기상청 단기예보 조회서비스(2.0): <https://data.kma.go.kr/api/selectApiDetail.do?openApiNo=421&pgmNo=42>
- 단기예보 조회서비스(공공데이터포털): <https://www.data.go.kr/data/15084084/openapi.do>
- 관측지점/예보구역 정보 서비스(영문 포털): <https://www.data.go.kr/en/data/15057111/openapi.do>

### PV 모델링
- pvlib python 문서: <https://pvlib-python.readthedocs.io/en/stable/index.html>
- pvlib GitHub: <https://github.com/pvlib/pvlib-python>

### 컨테이너 / 실행 환경
- NVIDIA Container Toolkit 설치 가이드: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
- Docker Desktop GPU 지원 문서: <https://docs.docker.com/desktop/features/gpu/>

---

## 21. 부록: 실험 결과 표 예시

### 21.1 Accuracy Leaderboard

| Model | Horizon | MAE | RMSE | nRMSE | Daytime MAE | Daily Energy Error |
|---|---:|---:|---:|---:|---:|---:|
| SegRNN | 24h |  |  |  |  |  |
| PatchTST | 24h |  |  |  |  |  |
| Time-LLM(GPT-2) | 24h |  |  |  |  |  |
| Llama 3.2 1B | 24h |  |  |  |  |  |
| Gemma 4 E2B | 24h |  |  |  |  |  |

### 21.2 CPU Benchmark Leaderboard

| Model | Horizon | Batch | Cold Start(s) | Warm p50(ms) | Warm p95(ms) | Throughput(samples/s) | Peak RAM(GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SegRNN | 24h | 1 |  |  |  |  |  |
| PatchTST | 24h | 1 |  |  |  |  |  |
| Time-LLM(GPT-2) | 24h | 1 |  |  |  |  |  |
| Llama 3.2 1B | 24h | 1 |  |  |  |  |  |
| Gemma 4 E2B | 24h | 1 |  |  |  |  |  |

### 21.3 운영 의사결정 표

| Model | Accuracy | CPU Latency | RAM | Complexity | Final Score | Decision |
|---|---:|---:|---:|---:|---:|---|
| SegRNN |  |  |  |  |  |  |
| PatchTST |  |  |  |  |  |  |
| Time-LLM |  |  |  |  |  |  |
| LLaMA |  |  |  |  |  |  |
| GEMMA |  |  |  |  |  |  |

