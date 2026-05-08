import time, torch, glob, sys, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, '/workspace/smoke_tests')
from smoke_logger import setup_log
from util_sampler import UtilSampler, measure_util_loop
setup_log('05_llama_gpu')

print('LLaMA 3.2 1B 모델 로드 시작 ...', flush=True)
snapshot = sorted(glob.glob('/models/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*'))[-1]
print(f'  snapshot: {snapshot}', flush=True)

tokenizer = AutoTokenizer.from_pretrained(snapshot)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

t0 = time.time()
base_model = AutoModelForCausalLM.from_pretrained(
    snapshot, dtype=torch.float16, device_map='auto'
)
load_sec = time.time() - t0
print(f'모델 로드 완료: {load_sec:.1f}s', flush=True)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
    target_modules=['q_proj', 'v_proj'], lora_dropout=0.05, bias='none'
)
model = get_peft_model(base_model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f'LoRA 적용: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)', flush=True)

dummy_prompt = (
    'The following is a time-series of solar power generation: '
    + ', '.join([f'{v:.2f}' for v in np.random.rand(24).tolist()])
    + '. Predict the next 24 hours:'
)
inputs = tokenizer(dummy_prompt, return_tensors='pt', truncation=True, max_length=512)
inputs = {k: v.to('cuda') for k, v in inputs.items()}

model.eval()
# warm-up 1회 제외
with torch.no_grad():
    model(**inputs, labels=inputs['input_ids']); torch.cuda.synchronize()
print('  (warm-up 완료)', flush=True)

times = []
with torch.no_grad():
    for i in range(5):
        t = time.time()
        out = model(**inputs, labels=inputs['input_ids'])
        torch.cuda.synchronize()
        elapsed = (time.time()-t)*1000
        times.append(elapsed)
        print(f'  forward #{i+1}: {elapsed:.0f} ms', flush=True)

print('  (사용률 측정 중 ~3s ...)', flush=True)
util_s, n_util, dur_s = measure_util_loop(
    lambda: (model(**inputs, labels=inputs['input_ids']), torch.cuda.synchronize()),
    min_seconds=3.0, interval_ms=500, gpu=True,
)
peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print()
print('[LLaMA 3.2 1B GPU 결과]')
print(f'  model load      : {load_sec:.1f}s')
print(f'  LoRA params     : {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')
print(f'  forward latency : {np.mean(times):.0f} ms  (avg 5회, warm-up 제외)')
print(f'  forward p95     : {np.percentile(times,95):.0f} ms')
print(f'  peak GPU mem    : {peak_mem:.3f} GB')
print(f'  util 측정       : {n_util}회/{dur_s:.1f}s 동안 폴링')
print(util_s.summary(mode='gpu'))
print('  PASS')
