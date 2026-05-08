"""
Time-LLM 백본 GPT-2 변형 CPU 추론 비교 테스트 (warm-up 제외)
gpt2 (117M) / gpt2-medium (345M) / gpt2-large (774M)
※ gpt2-xl(1.5B)은 CPU에서 매우 느릴 수 있으므로 선택적 실행
"""
import time, torch, sys, numpy as np
from transformers import GPT2Model

assert not torch.cuda.is_available(), 'GPU should be disabled'

sys.path.insert(0, '/workspace/smoke_tests')
from smoke_logger import setup_log
from util_sampler import UtilSampler
setup_log('12_timellm_gpt2_variants_cpu')

print(f'torch threads: {torch.get_num_threads()}', flush=True)

VARIANTS = ['gpt2', 'gpt2-medium', 'gpt2-large']
SEQ_LENS = [64, 512]

results = {}

for variant in VARIANTS:
    print(f'\n{"="*50}', flush=True)
    print(f'[{variant}] CPU 로드 시작 ...', flush=True)
    try:
        t0 = time.time()
        model = GPT2Model.from_pretrained(variant)
        model.eval()
        load_sec = time.time() - t0
        cfg = model.config
        approx_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'  로드: {load_sec:.1f}s  |  params: {approx_params:.0f}M', flush=True)

        row = {'params_M': approx_params, 'load_sec': load_sec}

        for seq_len in SEQ_LENS:
            # warm-up 1회 제외
            with torch.no_grad():
                model(inputs_embeds=torch.randn(1, seq_len, cfg.n_embd))
            print(f'  warm-up (seq={seq_len}) 완료', flush=True)

            n_runs = 10 if seq_len == 64 else 5
            sampler = UtilSampler(interval_ms=100, gpu=False).start()
            times = []
            with torch.no_grad():
                for i in range(n_runs):
                    t = time.time()
                    model(inputs_embeds=torch.randn(1, seq_len, cfg.n_embd))
                    elapsed = (time.time()-t)*1000
                    times.append(elapsed)
                    if (i+1) % (n_runs//2) == 0:
                        print(f'  seq={seq_len} #{i+1}: {elapsed:.1f}ms', flush=True)
            sampler.stop()

            row[f'seq{seq_len}_mean']     = float(np.mean(times))
            row[f'seq{seq_len}_p50']      = float(np.percentile(times, 50))
            row[f'seq{seq_len}_p95']      = float(np.percentile(times, 95))
            row[f'seq{seq_len}_cpu_avg']  = sampler.cpu_avg
            row[f'seq{seq_len}_cpu_peak'] = sampler.cpu_peak
            print(f'  seq={seq_len} → mean={np.mean(times):.1f}ms  p95={np.percentile(times,95):.1f}ms'
                  f'  CPU avg={sampler.cpu_avg:.0f}%  peak={sampler.cpu_peak:.0f}%', flush=True)

        results[variant] = row
        del model

    except Exception as e:
        print(f'  ✗ ERROR: {e}', flush=True)
        results[variant] = {'error': str(e)}

print(f'\n{"="*65}')
print(f'[GPT-2 변형 CPU 추론 비교 (batch=1, warm-up 제외)]')
print(f'{"모델":<14} {"params":>8} {"seq=64 mean":>12} {"seq=64 p95":>11} {"seq=512 mean":>13} {"seq=512 p95":>12}')
print('-'*65)
for v, r in results.items():
    if 'error' in r:
        print(f'{v:<14} {r["error"]}')
    else:
        print(f'{v:<14} {r["params_M"]:>6.0f}M  {r["seq64_mean"]:>11.1f}ms {r["seq64_p95"]:>10.1f}ms {r["seq512_mean"]:>12.1f}ms {r["seq512_p95"]:>11.1f}ms')
print('  CPU mode: PASS')
