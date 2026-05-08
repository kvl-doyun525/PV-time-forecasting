import time, torch, sys, numpy as np
from transformers import GPT2Model

assert not torch.cuda.is_available(), 'GPU should be disabled'

sys.path.insert(0, '/workspace/smoke_tests')
from smoke_logger import setup_log
from util_sampler import UtilSampler
setup_log('10_timellm_cpu')

print(f'Time-LLM (GPT-2 backbone) CPU 테스트 시작', flush=True)
print(f'torch threads: {torch.get_num_threads()}', flush=True)

print('GPT-2 backbone 로드 시작 ...', flush=True)
t0 = time.time()
backbone = GPT2Model.from_pretrained('gpt2')
backbone.eval()
load_sec = time.time() - t0
print(f'GPT-2 로드 완료: {load_sec:.1f}s', flush=True)

# warm-up 1회 제외 (seq=512로 최악 케이스 warm-up)
with torch.no_grad():
    backbone(inputs_embeds=torch.randn(1, 512, 768))
print('  (warm-up 완료)', flush=True)

# seq=64 (GPU 테스트와 동일 조건)
sampler_64 = UtilSampler(interval_ms=100, gpu=False).start()
times_64 = []
with torch.no_grad():
    for i in range(10):
        t = time.time()
        out = backbone(inputs_embeds=torch.randn(1, 64, 768))
        elapsed = (time.time()-t)*1000
        times_64.append(elapsed)
        if (i+1) % 3 == 0:
            print(f'  seq=64  #{i+1:2d}: {elapsed:.1f} ms', flush=True)
sampler_64.stop()

# seq=512 (실제 Time-LLM 권장 입력 토큰 수에 근접)
sampler_512 = UtilSampler(interval_ms=100, gpu=False).start()
times_512 = []
with torch.no_grad():
    for i in range(5):
        t = time.time()
        out = backbone(inputs_embeds=torch.randn(1, 512, 768))
        elapsed = (time.time()-t)*1000
        times_512.append(elapsed)
        print(f'  seq=512 #{i+1}: {elapsed:.1f} ms', flush=True)
sampler_512.stop()

print()
print('[Time-LLM CPU 결과 (GPT-2 backbone, batch=1, warm-up 제외)]')
print(f'  backbone load   : {load_sec:.1f}s')
print(f'  seq=64  mean    : {np.mean(times_64):.1f} ms  p50={np.percentile(times_64,50):.1f}  p95={np.percentile(times_64,95):.1f}')
print(f'  seq=64  CPU util: avg={sampler_64.cpu_avg:.0f}%  peak={sampler_64.cpu_peak:.0f}%')
print(f'  seq=512 mean    : {np.mean(times_512):.1f} ms  p50={np.percentile(times_512,50):.1f}  p95={np.percentile(times_512,95):.1f}')
print(f'  seq=512 CPU util: avg={sampler_512.cpu_avg:.0f}%  peak={sampler_512.cpu_peak:.0f}%')
print('  CPU mode: PASS')
