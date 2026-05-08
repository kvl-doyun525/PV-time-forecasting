# LLM 모델 다운로드 가이드

이 문서는 벤치마크에 필요한 LLM 모델(Llama 3.2 1B-Instruct, Gemma 4 E2B-it)을  
호스트의 `artifacts/models/` 디렉토리에 다운로드하는 절차를 설명한다.

---

## HF 토큰이 필요한가?

> **결론: 다운로드 시 1회만 필요. 로컬 실행 시에는 불필요.**

Llama·Gemma는 Hugging Face의 **gated model**(라이선스 동의 필요 모델)이다.  
다운로드할 때 계정 인증이 필요하지만, 일단 `artifacts/models/`에 저장된 후에는  
**HF 서버에 접근하지 않으므로 토큰·인터넷 없이 로컬에서 바로 실행** 가능하다.

| 단계 | HF 토큰 필요 | 비고 |
|------|:---:|------|
| 최초 다운로드 | ✅ 필요 | 라이선스 동의 계정 확인 (1회) |
| 로컬 실행·추론 | ❌ 불필요 | 로컬 경로에서 직접 로드 |
| 이미지 빌드 | ❌ 불필요 | 빌드 시 모델 포함 안 함 |

> **토큰 발급은 무료**: [huggingface.co](https://huggingface.co) 계정 가입 후 즉시 발급 가능.  
> **gate 승인 소요 시간**: Llama는 수 분 내 자동 승인, Gemma는 즉시 승인되는 경우가 많다.

---

## 사전 조건

### 1. Hugging Face 계정 및 토큰

1. [huggingface.co](https://huggingface.co) 에 로그인 (무료 가입)
2. **Settings → Access Tokens → New token** 에서 `Read` 권한 토큰 발급 (무료)
3. 아래 두 모델의 **gate 접근 승인** 요청 완료 (라이선스 동의, 무료):

| 모델 | 승인 요청 URL |
|------|--------------|
| Llama 3.2 1B-Instruct | https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct |
| Gemma 4 E2B-it | https://huggingface.co/google/gemma-4-e2b-it |

> gate 승인 후 `HF_TOKEN` 없이는 다운로드가 불가능하지만, 로컬 실행에는 영향 없다.

---

### 2. `.env` 파일에 토큰 설정

`project/.env` 파일을 열어 `HF_TOKEN` 값을 실제 토큰으로 교체한다.

```bash
# project/ 루트에서 실행
vi .env
```

```dotenv
# Hugging Face 토큰 (Llama gate 통과 후 발급)
HF_TOKEN=hf_실제토큰값으로_교체
```

---

## 모델 다운로드

모든 명령은 `project/` 디렉토리에서 실행한다.

```bash
cd /disk1/krems/time_forecasting/project
```

---

### Llama 3.2 1B-Instruct

저장 경로: `artifacts/models/models--meta-llama--Llama-3.2-1B-Instruct/`

```bash
docker run --rm --gpus all \
  --env-file .env \
  -v $(pwd)/artifacts/models:/models \
  pv-benchmark/unified:latest \
  python -c "
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

token    = os.environ['HF_TOKEN']
model_id = 'meta-llama/Llama-3.2-1B-Instruct'

print(f'Downloading {model_id} ...')
AutoTokenizer.from_pretrained(model_id, token=token, cache_dir='/models')
AutoModelForCausalLM.from_pretrained(
    model_id, token=token, cache_dir='/models', torch_dtype='auto'
)
print('Llama 3.2 1B download complete')
"
```

---

### Gemma 4 E2B-it

저장 경로: `artifacts/models/gemma-4-e2b-it/`

> `transformers AutoTokenizer`로 직접 다운로드하면 Gemma 4 토크나이저 포맷 파싱 중  
> 오류가 발생할 수 있다. `huggingface_hub.snapshot_download`로 파일만 받는다.

```bash
docker run --rm --gpus all \
  --env-file .env \
  -v $(pwd)/artifacts/models:/models \
  pv-benchmark/unified:latest \
  python -c "
import os
from huggingface_hub import snapshot_download

token    = os.environ['HF_TOKEN']
model_id = 'google/gemma-4-e2b-it'

print(f'Downloading {model_id} ...')
path = snapshot_download(
    repo_id=model_id,
    token=token,
    local_dir='/models/gemma-4-e2b-it',
    ignore_patterns=['*.msgpack', '*.h5', 'flax_*', 'tf_*'],
)
print(f'Gemma 4 E2B download complete → {path}')
"
```

---

## 다운로드 완료 확인

```bash
# 모델 디렉토리 목록 확인
ls -lh artifacts/models/

# 용량 확인
du -sh artifacts/models/*/
```

실제 저장 경로 및 용량:

| 모델 | 경로 | 실측 크기 |
|------|------|----------|
| Llama 3.2 1B-Instruct | `artifacts/models/models--meta-llama--Llama-3.2-1B-Instruct/` | ~2.4 GB |
| Gemma 4 E2B-it | `artifacts/models/gemma-4-e2b-it/` | ~9.6 GB |

---

## 로컬 모델 로드 (토큰 불필요)

다운로드 완료 후에는 HF 모델 ID 대신 **로컬 경로를 직접 지정**한다.  
토큰·인터넷 연결 없이 오프라인으로 실행된다.

```bash
# Llama — 로컬 경로로 로드 (토큰 없음)
docker run --rm --gpus all \
  -v $(pwd)/artifacts/models:/models \
  -v $(pwd)/vendor/TSLib:/workspace/vendor/TSLib \
  pv-benchmark/unified:latest \
  python -c "
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer

# Llama는 cache_dir 구조 → snapshots/{hash}/ 에 저장됨
snapshot = sorted(glob.glob('/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*'))[-1]
tokenizer = AutoTokenizer.from_pretrained(snapshot)
model     = AutoModelForCausalLM.from_pretrained(snapshot, dtype='auto', device_map='auto')
print('Llama 3.2 1B loaded on:', next(model.parameters()).device)
"

# Gemma — 로컬 경로로 로드 (토큰 없음)
# snapshot_download 로 받았으므로 artifacts/models/gemma-4-e2b-it/ 에 직접 저장됨
docker run --rm --gpus all \
  -v $(pwd)/artifacts/models:/models \
  -v $(pwd)/vendor/TSLib:/workspace/vendor/TSLib \
  pv-benchmark/unified:latest \
  python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

gemma_path = '/models/gemma-4-e2b-it'
tokenizer = AutoTokenizer.from_pretrained(gemma_path)
model     = AutoModelForCausalLM.from_pretrained(gemma_path, dtype='auto', device_map='auto')
print('Gemma 4 E2B loaded on:', next(model.parameters()).device)
"
```

> **모델별 저장 경로 차이**:  
> - Llama: `AutoModelForCausalLM.from_pretrained(..., cache_dir=...)` → `models--org--name/snapshots/{hash}/`  
> - Gemma: `snapshot_download(local_dir=...)` → 지정한 경로에 직접 저장

---

## 트러블슈팅

### `401 Unauthorized` 오류

- `.env` 파일의 `HF_TOKEN` 값을 확인한다.
- 토큰이 `Read` 권한을 포함하는지 확인한다.

### `403 Forbidden` — gate not passed

- 해당 모델의 HF 페이지에서 gate 승인을 요청했는지 확인한다.
- 승인 이메일을 수신했는지 확인한다.

### 디스크 공간 부족

```bash
df -h /disk1
```

- `artifacts/models/` 에 최소 **20 GB** 여유 공간이 필요하다.

### 네트워크 타임아웃

- HuggingFace CDN이 느린 경우 `HF_HUB_DOWNLOAD_TIMEOUT` 환경변수를 늘린다.

```bash
docker run --rm --gpus all \
  --env-file .env \
  -e HF_HUB_DOWNLOAD_TIMEOUT=300 \
  -v $(pwd)/artifacts/models:/models \
  pv-benchmark/unified:latest \
  python -c "..."
```
