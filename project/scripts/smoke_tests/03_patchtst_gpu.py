import time, torch, sys, numpy as np
import pandas as pd

sys.path.insert(0, '/workspace/smoke_tests')
from smoke_logger import setup_log
from util_sampler import UtilSampler, measure_util_loop
setup_log('03_patchtst_gpu')

sys.path.insert(0, '/workspace/vendor/TSLib')
from models.PatchTST import Model

df = pd.read_parquet('/workspace/artifacts/dummy_feature_mart/train/site_dummy.parquet')
data = torch.tensor(df.values, dtype=torch.float32)
SEQ_LEN, PRED_LEN, N_FEAT = 8760, 24, data.shape[1]
print(f'데이터 로드: {data.shape}', flush=True)

class Args:
    task_name = 'long_term_forecast'
    seq_len = SEQ_LEN; pred_len = PRED_LEN; enc_in = N_FEAT; c_out = 1
    d_model = 128; n_heads = 4; e_layers = 2; d_ff = 256; dropout = 0.1
    fc_dropout = 0.1; head_dropout = 0.0; patch_len = 16; stride = 8
    padding_patch = 'end'; revin = True; affine = False; subtract_last = False
    decomposition = False; kernel_size = 25; individual = False; factor = 1
    activation = 'gelu'

model = Model(Args()).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()
print('모델 초기화 완료 (PatchTST)', flush=True)

# PatchTST는 seq_len=8760일 때 attention map이 크므로 batch_size=2 사용
BATCH = 2
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for step, i in enumerate(range(0, len(data) - SEQ_LEN - PRED_LEN - BATCH, BATCH)):
    x = torch.stack([data[j:j+SEQ_LEN] for j in range(i, i+BATCH)]).cuda()
    y = torch.stack([data[j+SEQ_LEN:j+SEQ_LEN+PRED_LEN, 0:1] for j in range(i, i+BATCH)]).cuda()
    pred = model(x, None, None, None)[:, :PRED_LEN, 0:1]
    loss = criterion(pred, y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    print(f'  step {step+1} / loss={loss.item():.4f}', flush=True)
    if step >= 5:
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
print('[PatchTST GPU 결과]')
print(f'  6 steps time  : {epoch_sec:.1f}s')
print(f'  infer latency : {np.mean(times):.2f} ms  (avg 10회)')
print(f'  infer p95     : {np.percentile(times,95):.2f} ms')
print(f'  peak GPU mem  : {peak_mem:.3f} GB')
print(f'  util 측정     : {n_util}회/{dur_s:.1f}s 동안 폴링')
print(util_s.summary(mode='gpu'))
print('  PASS')
