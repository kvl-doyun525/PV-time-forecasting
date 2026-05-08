import time, torch, sys, numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/smoke_tests')
from smoke_logger import setup_log
from util_sampler import UtilSampler, measure_util_loop
setup_log('02_segrnn_gpu')

sys.path.insert(0, '/workspace/vendor/TSLib')
from models.SegRNN import Model

df = pd.read_parquet('/workspace/artifacts/dummy_feature_mart/train/site_dummy.parquet')
data = torch.tensor(df.values, dtype=torch.float32)
SEQ_LEN, PRED_LEN, N_FEAT = 8760, 24, data.shape[1]
print(f'데이터 로드: {data.shape}', flush=True)

class Args:
    task_name = 'long_term_forecast'
    seq_len = SEQ_LEN; pred_len = PRED_LEN; enc_in = N_FEAT; d_model = 512
    dropout = 0.1; seg_len = 24; rnn_type = 'gru'; dec_way = 'pmf'
    channel_id = False; revin = False

model = Model(Args()).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()
print('모델 초기화 완료 (SegRNN)', flush=True)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for step, i in enumerate(range(0, len(data) - SEQ_LEN - PRED_LEN - 32, 32)):
    x = torch.stack([data[j:j+SEQ_LEN] for j in range(i, i+32)]).cuda()
    y = torch.stack([data[j+SEQ_LEN:j+SEQ_LEN+PRED_LEN, 0:1] for j in range(i, i+32)]).cuda()
    pred = model(x, None, None, None)[:, :PRED_LEN, 0:1]
    loss = criterion(pred, y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (step+1) % 3 == 0:
        print(f'  step {step+1} / loss={loss.item():.4f}', flush=True)
    if step >= 9:
        break
epoch_sec = time.time() - t0

model.eval()
x_s = data[:SEQ_LEN].unsqueeze(0).cuda()
# warm-up 1회 제외
with torch.no_grad():
    model(x_s, None, None, None); torch.cuda.synchronize()

times = []
with torch.no_grad():
    for _ in range(10):
        t = time.time(); model(x_s, None, None, None); torch.cuda.synchronize()
        times.append((time.time()-t)*1000)

print('  (사용률 측정 중 ~3s ...)', flush=True)
util_s, n_util, dur_s = measure_util_loop(
    lambda: (model(x_s, None, None, None), torch.cuda.synchronize()),
    min_seconds=3.0, interval_ms=500, gpu=True,
)
peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print()
print('[SegRNN GPU 결과]')
print(f'  10 steps time : {epoch_sec:.1f}s')
print(f'  infer latency : {np.mean(times):.2f} ms  (avg 10회)')
print(f'  infer p95     : {np.percentile(times,95):.2f} ms')
print(f'  peak GPU mem  : {peak_mem:.3f} GB')
print(f'  util 측정     : {n_util}회/{dur_s:.1f}s 동안 폴링')
print(util_s.summary(mode='gpu'))
print('  PASS')
