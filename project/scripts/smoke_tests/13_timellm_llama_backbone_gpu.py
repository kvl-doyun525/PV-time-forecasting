"""
Time-LLM + LLaMA 3.2 1B 백본 GPU 추론 테스트

Time-LLM 아키텍처 구조:
  ① 시계열 패치 임베딩 (Linear projection → backbone hidden_size)
  ② 동결된 LLM backbone (frozen LlamaModel)
  ③ 출력 레이어 (Linear → pred_len)

note: time-llm 공식 이미지 transformers=4.31.0 은 Llama-3.2 rope_scaling 미지원
      → unified:latest 이미지 (transformers>=4.40) 로 실행

실행: docker run --gpus all -v ... pv-benchmark/unified:latest \
        python /workspace/smoke_tests/13_timellm_llama_backbone_gpu.py
"""
import time, torch, glob, sys, numpy as np
import torch.nn as nn
from transformers import LlamaModel, AutoConfig

sys.path.insert(0, '/workspace/smoke_tests')
from smoke_logger import setup_log
from util_sampler import UtilSampler, measure_util_loop
setup_log('13_timellm_llama_backbone_gpu')

# ─── 설정 ──────────────────────────────────────────────────────────────────
SEQ_LEN   = 512   # 시계열 입력 길이
PATCH_LEN = 16    # 패치 크기 (Time-LLM 기본값)
PRED_LEN  = 24    # 예측 길이
N_FEAT    = 26    # 피처 수 (dummy feature mart 기준)
N_RUNS    = 5     # 추론 반복 횟수

MODEL_SNAPSHOT = sorted(
    glob.glob('/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*')
)[-1]


# ─── Time-LLM + LLaMA backbone 미니 구현 ──────────────────────────────────
class TimeLLM_LLaMA(nn.Module):
    """
    Time-LLM 논문의 핵심 파이프라인을 최소 구현:
      patch_embed → frozen LlamaModel → output_proj
    backbone hidden_size: 2048 (LLaMA 3.2 1B)
    """
    def __init__(self, llm_model: LlamaModel, hidden_size: int,
                 seq_len: int, patch_len: int, pred_len: int, n_feat: int):
        super().__init__()
        n_patches = seq_len // patch_len      # 패치 수
        patch_dim  = patch_len * n_feat       # 패치 1개의 raw 차원

        self.patch_embed = nn.Linear(patch_dim, hidden_size)
        self.llm_backbone = llm_model
        self.output_proj  = nn.Linear(n_patches * hidden_size, pred_len)

        # backbone 파라미터 동결 (Time-LLM 핵심)
        for p in self.llm_backbone.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, seq_len, n_feat) float16
        return: (B, pred_len)
        """
        B, S, F = x.shape
        P = S // PATCH_LEN
        x_patch = x.reshape(B, P, PATCH_LEN * F)          # (B, n_patches, patch_dim)
        emb     = self.patch_embed(x_patch)                # (B, n_patches, hidden)

        out = self.llm_backbone(inputs_embeds=emb)         # LlamaModel forward
        h   = out.last_hidden_state                        # (B, n_patches, hidden)
        h   = h.reshape(B, -1)                             # (B, n_patches*hidden)
        return self.output_proj(h)                         # (B, pred_len)


# ─── 메인 ─────────────────────────────────────────────────────────────────
print(f'LLaMA 3.2 1B backbone 로드 시작: {MODEL_SNAPSHOT}', flush=True)
t0 = time.time()
llm_cfg = AutoConfig.from_pretrained(MODEL_SNAPSHOT)
llm_model = LlamaModel.from_pretrained(
    MODEL_SNAPSHOT,
    torch_dtype=torch.float16,
    device_map='auto',
)
load_sec = time.time() - t0
print(f'LLaMA 로드 완료: {load_sec:.1f}s', flush=True)
print(f'  hidden_size={llm_cfg.hidden_size}, layers={llm_cfg.num_hidden_layers}', flush=True)

model = TimeLLM_LLaMA(
    llm_model  = llm_model,
    hidden_size= llm_cfg.hidden_size,
    seq_len    = SEQ_LEN,
    patch_len  = PATCH_LEN,
    pred_len   = PRED_LEN,
    n_feat     = N_FEAT,
).half().cuda()   # 전체 float16 으로 통일 (backbone 포함)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'파라미터: {trainable:,} / {total:,} ({100*trainable/total:.2f}% 학습 가능)', flush=True)

# 더미 입력 (batch=1)
x_dummy = torch.randn(1, SEQ_LEN, N_FEAT, dtype=torch.float16).cuda()

model.eval()
torch.cuda.reset_peak_memory_stats()

# warm-up 1회 제외
with torch.no_grad():
    model(x_dummy)
    torch.cuda.synchronize()
print('  (warm-up 완료)', flush=True)

# 본 측정
times = []
with torch.no_grad():
    for i in range(N_RUNS):
        t = time.time()
        out = model(x_dummy)
        torch.cuda.synchronize()
        elapsed = (time.time() - t) * 1000
        times.append(elapsed)
        print(f'  forward #{i+1}: {elapsed:.1f} ms  out={out.shape}', flush=True)

print('  (사용률 측정 중 ~3s ...)', flush=True)
util_s, n_util, dur_s = measure_util_loop(
    lambda: (model(x_dummy), torch.cuda.synchronize()),
    min_seconds=3.0, interval_ms=500, gpu=True,
)
peak_mem = torch.cuda.max_memory_allocated() / 1024**3

print()
print('[Time-LLM + LLaMA 3.2 1B backbone GPU 결과]')
print(f'  backbone       : LLaMA 3.2 1B (hidden=2048, layers={llm_cfg.num_hidden_layers})')
print(f'  seq_len={SEQ_LEN}, patch_len={PATCH_LEN}, n_patches={SEQ_LEN//PATCH_LEN}')
print(f'  backbone load  : {load_sec:.1f}s')
print(f'  trainable param: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')
print(f'  forward latency: {np.mean(times):.1f} ms  (avg {N_RUNS}회, warm-up 제외)')
print(f'  forward p95    : {np.percentile(times, 95):.1f} ms')
print(f'  peak GPU mem   : {peak_mem:.3f} GB')
print(f'  util 측정      : {n_util}회/{dur_s:.1f}s 동안 폴링')
print(util_s.summary(mode='gpu'))
print('  PASS')
