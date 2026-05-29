# Time-LLM 공식 requirements.txt 의 torch==2.2.2 와 정렬 (CUDA 12.1)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git curl libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 런타임에 ../artifacts 가 마운트되면 /workspace/artifacts 가 가려지므로,
# HF 캐시는 볼륨 밖 /opt 에 두고 학습 코드는 HF_HOME 이 있으면 덮어쓰지 않는다.
ENV HF_HOME=/opt/huggingface_cache

COPY docker/time_llm/requirements-time-llm.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-time-llm.txt

# Time-LLM 소스만 클론 (의존성은 requirements-time-llm.txt 가 공식 핀과 동일하게 설치)
RUN git clone --depth 1 https://github.com/KimMeen/Time-LLM.git /workspace/TimeLLM

# TimeLLM.Model 과 동일한 HF id 로 빌드 시 캐시 (런타임 재다운로드 완화)
RUN mkdir -p "$HF_HOME" && python -c "\
from transformers import GPT2Model, GPT2Tokenizer, GPT2TokenizerFast; \
GPT2Model.from_pretrained('openai-community/gpt2'); \
GPT2Tokenizer.from_pretrained('openai-community/gpt2'); \
GPT2TokenizerFast.from_pretrained('openai-community/gpt2'); \
print('HF openai-community/gpt2 cached under', __import__('os').environ.get('HF_HOME'))"

# src/, conf/, artifacts/ 는 런타임에 호스트 볼륨으로 마운트
# (-v $(pwd)/src:/workspace/src 등) — 이미지에 포함하지 않는다

ENV PYTHONPATH=/workspace:/workspace/TimeLLM
