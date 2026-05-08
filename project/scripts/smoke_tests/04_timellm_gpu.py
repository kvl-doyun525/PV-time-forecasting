import time, torch, sys, numpy as np
from transformers import GPT2Model

sys.path.insert(0, '/workspace/smoke_tests')
from smoke_logger import setup_log
from util_sampler import UtilSampler, measure_util_loop
setup_log('04_timellm_gpu')

print('GPT-2 backbone 로드 시작 ...', flush=True)
t0 = time.time()
backbone = GPT2Model.from_pretrained('gpt2').cuda()
load_sec = time.time() - t0
print(f'GPT-2 로드 완료: {load_sec:.1f}s', flush=True)

backbone.eval()
# warm-up 1회 제외
with torch.no_grad():
    backbone(inputs_embeds=torch.randn(1, 64, 768).cuda()); torch.cuda.synchronize()

times = []
with torch.no_grad():
    for i in range(5):
        t = time.time()
        out = backbone(inputs_embeds=torch.randn(1, 64, 768).cuda())
        torch.cuda.synchronize()
        elapsed = (time.time()-t)*1000
        times.append(elapsed)
        print(f'  forward #{i+1}: {elapsed:.1f} ms', flush=True)

print('  (사용률 측정 중 ~3s ...)', flush=True)
util_s, n_util, dur_s = measure_util_loop(
    lambda: (backbone(inputs_embeds=torch.randn(1, 64, 768).cuda()), torch.cuda.synchronize()),
    min_seconds=3.0, interval_ms=500, gpu=True,
)
peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print()
print('[Time-LLM GPU 결과]')
print(f'  backbone load   : {load_sec:.1f}s')
print(f'  forward latency : {np.mean(times):.2f} ms  (avg 5회, seq=64)')
print(f'  forward p95     : {np.percentile(times,95):.2f} ms')
print(f'  peak GPU mem    : {peak_mem:.3f} GB')
print(f'  util 측정       : {n_util}회/{dur_s:.1f}s 동안 폴링')
print(util_s.summary(mode='gpu'))
print('  PASS')
