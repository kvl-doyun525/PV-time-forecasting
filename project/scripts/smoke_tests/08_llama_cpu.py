import time, torch, glob, sys, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

assert not torch.cuda.is_available(), 'GPU should be disabled'

sys.path.insert(0, '/workspace/smoke_tests')
from smoke_logger import setup_log
from util_sampler import UtilSampler
setup_log('08_llama_cpu')

print('LLaMA CPU 로드 시작 ...', flush=True)
snapshot = sorted(glob.glob('/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*'))[-1]

tokenizer = AutoTokenizer.from_pretrained(snapshot)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32)
model.eval()
load_sec = time.time() - t0
print(f'CPU 로드 완료: {load_sec:.1f}s', flush=True)

prompt = 'Solar power forecast: ' + ', '.join([f'{v:.2f}' for v in np.random.rand(24)])
inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)

# warm-up 1회 제외
with torch.no_grad():
    model(**inputs, labels=inputs['input_ids'])
print('  (warm-up 완료)', flush=True)

sampler = UtilSampler(interval_ms=200, gpu=False).start()
times = []
with torch.no_grad():
    for i in range(5):
        t = time.time()
        out = model(**inputs, labels=inputs['input_ids'])
        elapsed = (time.time()-t)*1000
        times.append(elapsed)
        print(f'  forward #{i+1}: {elapsed:.0f} ms', flush=True)
sampler.stop()

print()
print('[LLaMA 3.2 1B CPU 결과]')
print(f'  load time   : {load_sec:.1f}s')
print(f'  forward mean: {np.mean(times):.0f} ms  (5회, warm-up 제외)')
print(f'  forward p50 : {np.percentile(times,50):.0f} ms')
print(f'  forward p95 : {np.percentile(times,95):.0f} ms')
print(sampler.summary(mode='cpu'))
print('  CPU mode: PASS')
