import time, torch, sys, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

assert not torch.cuda.is_available(), 'GPU should be disabled'

sys.path.insert(0, '/workspace/smoke_tests')
from smoke_logger import setup_log
from util_sampler import UtilSampler
setup_log('09_gemma_cpu')

gemma_path = '/models/gemma-4-e2b-it'
print(f'Gemma 4 E2B CPU 로드 시작: {gemma_path}', flush=True)
print(f'torch threads: {torch.get_num_threads()}', flush=True)

tokenizer = AutoTokenizer.from_pretrained(gemma_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# CPU에서는 float16 대신 bfloat16 또는 float32 사용
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    gemma_path,
    dtype=torch.bfloat16,   # CPU에서 bfloat16 지원 (AVX-512)
    device_map='cpu',
    low_cpu_mem_usage=True,
)
model.eval()
load_sec = time.time() - t0
print(f'CPU 로드 완료: {load_sec:.1f}s', flush=True)

dummy_prompt = (
    'Time series solar power: '
    + ', '.join([f'{v:.2f}' for v in np.random.rand(24).tolist()])
    + '. Forecast next 24h:'
)
inputs = tokenizer(dummy_prompt, return_tensors='pt', truncation=True, max_length=256)

# warm-up 1회 제외
with torch.no_grad():
    model(**inputs, labels=inputs['input_ids'])
print('  (warm-up 완료)', flush=True)

sampler = UtilSampler(interval_ms=500, gpu=False).start()
times = []
with torch.no_grad():
    for i in range(3):
        t = time.time()
        out = model(**inputs, labels=inputs['input_ids'])
        elapsed = (time.time()-t)*1000
        times.append(elapsed)
        print(f'  forward #{i+1}: {elapsed:.0f} ms', flush=True)
sampler.stop()

print()
print('[Gemma 4 E2B CPU 결과]')
print(f'  load time   : {load_sec:.1f}s')
print(f'  forward mean: {np.mean(times):.0f} ms')
print(f'  forward p50 : {np.percentile(times,50):.0f} ms')
print(f'  forward p95 : {np.percentile(times,95):.0f} ms')
print(sampler.summary(mode='cpu'))
print('  CPU mode: PASS')
