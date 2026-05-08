"""CPU 벤치마크용 스레드·GPU 비활성화 (`pv_model_benchmark_execution.md` §12.1)."""

from __future__ import annotations

import os

import torch


def setup_cpu_benchmark(*, num_threads: int = 16) -> None:
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if torch.cuda.is_available():
        raise RuntimeError("CPU 벤치마크는 CUDA 비가용이어야 합니다 (CUDA_VISIBLE_DEVICES 확인).")
    if torch.get_num_threads() != num_threads:
        # 일부 BLAS 빌드는 상한을 낮출 수 있음 — 치명적이지 않으면 경고만
        pass
