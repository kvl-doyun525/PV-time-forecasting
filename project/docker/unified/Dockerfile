# SegRNN, PatchTST, DLinear, LLaMA 3.2 1B, Gemma 4 E2B 통합 이미지
# bitsandbytes CUDA 컴파일을 위해 -devel 베이스 사용
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git curl wget build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY docker/unified/requirements-unified.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-unified.txt

# transformers 5.x(Gemma 4 지원)이 torch>=2.4 를 요구하므로
# 호스트 CUDA 드라이버(12.2)와 호환되는 torch 2.4.1+cu121 로 재고정
RUN pip install --no-cache-dir --force-reinstall \
    "torch==2.4.1+cu121" \
    "torchvision==0.19.1+cu121" \
    "torchaudio==2.4.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

# bitsandbytes CUDA 컴파일 확인
RUN python -c "import bitsandbytes; print('bitsandbytes OK:', bitsandbytes.__version__)"

# src/, conf/, artifacts/, vendor/TSLib 는 런타임에 호스트 볼륨으로 마운트
# scripts/setup_vendor.sh 실행 후 호스트의 vendor/TSLib 을 컨테이너에서 사용
# HF 토큰도 런타임에 환경변수로 주입 (-e HF_TOKEN=xxx)

ENV PYTHONPATH=/workspace:/workspace/vendor/TSLib
