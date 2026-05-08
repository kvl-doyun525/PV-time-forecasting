# Time-LLM 전용: torch 2.1.x + transformers 4.31.0 고정
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY docker/time_llm/requirements-time-llm.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-time-llm.txt

# Time-LLM 공식 저장소 설치
RUN git clone https://github.com/KimMeen/Time-LLM.git /workspace/TimeLLM \
    && cd /workspace/TimeLLM \
    && pip install --no-cache-dir -r requirements.txt || true

# GPT-2 backbone 사전 캐시 (빌드 시 다운로드 → 런타임 인터넷 불필요)
RUN python -c "\
from transformers import GPT2Model, GPT2Tokenizer; \
GPT2Model.from_pretrained('gpt2'); \
GPT2Tokenizer.from_pretrained('gpt2'); \
print('GPT-2 backbone cached')"

# src/, conf/, artifacts/ 는 런타임에 호스트 볼륨으로 마운트
# (-v $(pwd)/src:/workspace/src 등) — 이미지에 포함하지 않는다

ENV PYTHONPATH=/workspace:/workspace/TimeLLM
