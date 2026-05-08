"""
Time-LLM 백본 GPT-2 변형 GPU 추론 비교 테스트
gpt2 (117M) / gpt2-medium (345M) / gpt2-large (774M) / gpt2-xl (1.5B)
"""
import time, torch, sys, numpy as np
from transformers import GPT2Model, GPT2Config

sys.path.insert(0, '/workspace/smoke_tests')
from smoke_logger import setup_log
from util_sampler import UtilSampler
setup_log('11_timellm_gpt2_variants_gpu')

VARIANTS = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
SEQ_LEN = 64   # GPU 테스트와 동일 조건

results = {}

for variant in VARIANTS:
    print(f'\n{"="*50}', flush=True)
    print(f'[{variant}] 로드 시작 ...', flush=True)
    try:
        t0 = time.time()
        model = GPT2Model.from_pretrained(variant).cuda()
        load_sec = time.time() - t0
        cfg = model.config
        approx_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'  로드: {load_sec:.1f}s  |  params: {approx_params:.0f}M  |  layers={cfg.n_layer}, hidden={cfg.n_embd}', flush=True)

        model.eval()
        torch.cuda.reset_peak_memory_stats()

        # warm-up 1회 제외
        with torch.no_grad():
            model(inputs_embeds=torch.randn(1, SEQ_LEN, cfg.n_embd).cuda())
            torch.cuda.synchronize()

        sampler = UtilSampler(interval_ms=100).start()
        times = []
        with torch.no_grad():
            for i in range(10):
                t = time.time()
                model(inputs_embeds=torch.randn(1, SEQ_LEN, cfg.n_embd).cuda())
                torch.cuda.synchronize()
                elapsed = (time.time()-t)*1000
                times.append(elapsed)
                if (i+1) % 3 == 0:
                    print(f'  forward #{i+1}: {elapsed:.2f} ms', flush=True)
        sampler.stop()

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        results[variant] = {
            'params_M':   approx_params,
            'load_sec':   load_sec,
            'mean_ms':    float(np.mean(times)),
            'p50_ms':     float(np.percentile(times, 50)),
            'p95_ms':     float(np.percentile(times, 95)),
            'peak_gb':    peak_mem,
            'gpu_avg':    sampler.gpu_avg,
            'gpu_peak':   sampler.gpu_peak,
            'cpu_avg':    sampler.cpu_avg,
            'cpu_peak':   sampler.cpu_peak,
        }
        print(f'  → mean={np.mean(times):.2f}ms  p95={np.percentile(times,95):.2f}ms  mem={peak_mem:.3f}GB', flush=True)
        print(sampler.summary(mode='gpu'), flush=True)

        del model
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        print(f'  ✗ OOM — GPU 메모리 부족', flush=True)
        results[variant] = {'error': 'OOM'}
    except Exception as e:
        print(f'  ✗ ERROR: {e}', flush=True)
        results[variant] = {'error': str(e)}

print(f'\n{"="*60}')
print(f'[GPT-2 변형 GPU 추론 비교 (seq={SEQ_LEN}, batch=1, warm-up 제외)]')
print(f'{"모델":<14} {"파라미터":>9} {"로드":>7} {"mean":>9} {"p95":>9} {"GPU mem":>9}')
print('-'*60)
for v, r in results.items():
    if 'error' in r:
        print(f'{v:<14} {r["error"]}')
    else:
        print(f'{v:<14} {r["params_M"]:>7.0f}M {r["load_sec"]:>6.1f}s {r["mean_ms"]:>8.2f}ms {r["p95_ms"]:>8.2f}ms {r["peak_gb"]:>8.3f}GB')
print('  PASS')
