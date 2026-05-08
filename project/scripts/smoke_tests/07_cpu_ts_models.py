import time, torch, sys, numpy as np
import pandas as pd

assert not torch.cuda.is_available(), 'GPU should be disabled'

sys.path.insert(0, '/workspace/smoke_tests')
from smoke_logger import setup_log
from util_sampler import UtilSampler, measure_util_loop
setup_log('07_cpu_ts_models')

sys.path.insert(0, '/workspace/vendor/TSLib')

df = pd.read_parquet('/workspace/artifacts/dummy_feature_mart/train/site_dummy.parquet')
data = torch.tensor(df.values, dtype=torch.float32)
SEQ_LEN, PRED_LEN, N_FEAT = 8760, 24, data.shape[1]
x = data[:SEQ_LEN].unsqueeze(0)
print(f'CPU 테스트 입력: {x.shape}, torch threads: {torch.get_num_threads()}', flush=True)

results = {}

# DLinear
print('DLinear CPU 추론 시작 ...', flush=True)
from models.DLinear import Model as DL
class DArgs:
    task_name = 'long_term_forecast'
    seq_len = SEQ_LEN; pred_len = PRED_LEN; enc_in = N_FEAT
    individual = False; moving_avg = 25

m = DL(DArgs()).eval()
with torch.no_grad():
    m(x, None, None, None)  # warm-up
# DLinear CPU는 0.3ms/call → 3초 동안 ~10000회 실행해야 의미있는 샘플 수 확보
util_dl, n_dl, dur_dl = measure_util_loop(
    lambda: m(x, None, None, None), min_seconds=3.0, interval_ms=300, gpu=False,
)
# 별도 latency 측정 (20회)
times = []
with torch.no_grad():
    for i in range(20):
        t = time.time(); m(x, None, None, None); elapsed = (time.time()-t)*1000
        times.append(elapsed)
        if (i+1) % 5 == 0:
            print(f'  DLinear #{i+1:2d}: {elapsed:.1f} ms', flush=True)
results['DLinear'] = {
    'mean': float(np.mean(times)),
    'p50':  float(np.percentile(times, 50)),
    'p95':  float(np.percentile(times, 95)),
    'cpu_avg': util_dl.cpu_avg, 'cpu_peak': util_dl.cpu_peak,
}

# SegRNN
print('SegRNN CPU 추론 시작 ...', flush=True)
from models.SegRNN import Model as SR
class SArgs:
    task_name = 'long_term_forecast'
    seq_len = SEQ_LEN; pred_len = PRED_LEN; enc_in = N_FEAT; d_model = 512
    dropout = 0.1; seg_len = 24; rnn_type = 'gru'; dec_way = 'pmf'
    channel_id = False; revin = False

m = SR(SArgs()).eval()
with torch.no_grad():
    m(x, None, None, None)  # warm-up
# SegRNN CPU ~70ms/call → 3초면 ~42회, 충분
sampler_sr = UtilSampler(interval_ms=300, gpu=False).start()
times = []
with torch.no_grad():
    for i in range(5):
        t = time.time(); m(x, None, None, None); elapsed = (time.time()-t)*1000
        times.append(elapsed)
        print(f'  SegRNN #{i+1}: {elapsed:.1f} ms', flush=True)
sampler_sr.stop()
results['SegRNN'] = {
    'mean': float(np.mean(times)),
    'p50':  float(np.percentile(times, 50)),
    'p95':  float(np.percentile(times, 95)),
    'cpu_avg': sampler_sr.cpu_avg, 'cpu_peak': sampler_sr.cpu_peak,
}

# PatchTST
print('PatchTST CPU 추론 시작 ...', flush=True)
from models.PatchTST import Model as PT
class PArgs:
    task_name = 'long_term_forecast'
    seq_len = SEQ_LEN; pred_len = PRED_LEN; enc_in = N_FEAT; c_out = 1
    d_model = 128; n_heads = 4; e_layers = 2; d_ff = 256; dropout = 0.1
    fc_dropout = 0.1; head_dropout = 0.0; patch_len = 16; stride = 8
    padding_patch = 'end'; revin = True; affine = False; subtract_last = False
    decomposition = False; kernel_size = 25; individual = False; factor = 1
    activation = 'gelu'

m = PT(PArgs()).eval()
with torch.no_grad():
    m(x, None, None, None)  # warm-up
# PatchTST CPU ~320ms/call → 5회 = 1.6초, 약간 부족하므로 sampler 유지
sampler_pt = UtilSampler(interval_ms=300, gpu=False).start()
times = []
with torch.no_grad():
    for i in range(5):
        t = time.time(); m(x, None, None, None); elapsed = (time.time()-t)*1000
        times.append(elapsed)
        print(f'  PatchTST #{i+1}: {elapsed:.1f} ms', flush=True)
sampler_pt.stop()
results['PatchTST'] = {
    'mean': float(np.mean(times)),
    'p50':  float(np.percentile(times, 50)),
    'p95':  float(np.percentile(times, 95)),
    'cpu_avg': sampler_pt.cpu_avg, 'cpu_peak': sampler_pt.cpu_peak,
}

print()
print('[CPU 추론 latency + 사용률 결과 (batch=1, seq_len=8760)]')
print(f'  {"모델":<12} {"mean":>9} {"p50":>9} {"p95":>9} {"CPU avg":>9} {"CPU peak":>10}')
print('  ' + '-'*55)
for name, r in results.items():
    print(f'  {name:<12} {r["mean"]:>8.1f}ms {r["p50"]:>8.1f}ms {r["p95"]:>8.1f}ms'
          f' {r["cpu_avg"]:>8.0f}%  {r["cpu_peak"]:>8.0f}%')
print('  CPU mode: PASS')
