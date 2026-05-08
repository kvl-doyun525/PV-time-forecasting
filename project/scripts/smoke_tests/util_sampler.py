"""
GPU/CPU 사용률 백그라운드 폴링 유틸리티.

사용법:
    from util_sampler import UtilSampler

    sampler = UtilSampler()
    sampler.start()
    # ... 추론 루프 ...
    sampler.stop()
    print(sampler.summary())

참고:
- GPU util: pynvml 우선, 없으면 nvidia-smi subprocess 폴백
- CPU util: psutil (cpu_percent, 전체 코어 평균)
- 폴링 간격 기본값 100ms → 매우 빠른 모델(<1ms/call)은 샘플 수가 적어 정확도 낮음
"""
import threading
import time
import subprocess
import numpy as np

try:
    import psutil as _psutil
    _psutil.cpu_percent()   # 첫 호출 초기화 (항상 0.0 반환)
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# GPU 방법 선택: pynvml > nvidia-smi subprocess
_GPU_METHOD = None
_NVML_HANDLE = None

try:
    import pynvml as _nvml
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _nvml.nvmlInit()
    _NVML_HANDLE = _nvml.nvmlDeviceGetHandleByIndex(0)
    _GPU_METHOD = "pynvml"
except Exception:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0:
            _GPU_METHOD = "smi"
    except Exception:
        pass


def _read_gpu_util() -> float:
    """현재 GPU 사용률(%) 반환. 측정 불가 시 -1."""
    if _GPU_METHOD == "pynvml":
        try:
            u = _nvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
            return float(u.gpu)
        except Exception:
            return -1.0
    elif _GPU_METHOD == "smi":
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1,
            )
            vals = [float(x.strip()) for x in r.stdout.strip().split("\n") if x.strip()]
            return vals[0] if vals else -1.0
        except Exception:
            return -1.0
    return -1.0


class UtilSampler:
    """
    백그라운드 스레드로 GPU/CPU 사용률을 주기적으로 수집.

    Parameters
    ----------
    interval_ms : 폴링 간격 (밀리초). 기본 100ms.
    gpu         : GPU 사용률 수집 여부 (False = CPU only 모드)
    """

    def __init__(self, interval_ms: int = 100, gpu: bool = True):
        self.interval = interval_ms / 1000.0
        self._use_gpu = gpu and (_GPU_METHOD is not None)
        self._gpu_samples: list[float] = []
        self._cpu_samples: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None

    # ── 제어 ────────────────────────────────────────────────
    def start(self) -> "UtilSampler":
        if _HAS_PSUTIL:
            _psutil.cpu_percent()   # 갱신 초기화
        self._running = True
        self._gpu_samples.clear()
        self._cpu_samples.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> "UtilSampler":
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        return self

    # ── 결과 프로퍼티 ────────────────────────────────────────
    @property
    def gpu_avg(self) -> float:
        valid = [v for v in self._gpu_samples if v >= 0]
        return float(np.mean(valid)) if valid else 0.0

    @property
    def gpu_peak(self) -> float:
        valid = [v for v in self._gpu_samples if v >= 0]
        return float(np.max(valid)) if valid else 0.0

    @property
    def cpu_avg(self) -> float:
        return float(np.mean(self._cpu_samples)) if self._cpu_samples else 0.0

    @property
    def cpu_peak(self) -> float:
        return float(np.max(self._cpu_samples)) if self._cpu_samples else 0.0

    @property
    def n_samples(self) -> int:
        return len(self._cpu_samples)

    def summary(self, mode: str = "gpu") -> str:
        """
        결과 문자열 반환.
        mode='gpu' : GPU + CPU
        mode='cpu' : CPU only
        """
        lines = []
        if mode == "gpu" and self._use_gpu:
            lines.append(
                f"  GPU util       : avg={self.gpu_avg:.0f}%  peak={self.gpu_peak:.0f}%"
                f"  (n={self.n_samples} samples)"
            )
        lines.append(
            f"  CPU util       : avg={self.cpu_avg:.0f}%  peak={self.cpu_peak:.0f}%"
            + ("" if mode == "gpu" else f"  (n={self.n_samples} samples)")
        )
        return "\n".join(lines)

    # ── 내부 폴링 ────────────────────────────────────────────
    def _loop(self) -> None:
        while self._running:
            if self._use_gpu:
                self._gpu_samples.append(_read_gpu_util())
            if _HAS_PSUTIL:
                self._cpu_samples.append(_psutil.cpu_percent())
            time.sleep(self.interval)


def measure_util_loop(
    fn,
    min_seconds: float = 3.0,
    interval_ms: int = 500,
    gpu: bool = True,
):
    """
    fn 을 min_seconds 이상 반복 실행하며 GPU/CPU 사용률을 측정.
    latency 측정과 별도로 '부하 지속 상태'의 실제 사용률을 측정하는 용도.

    Returns
    -------
    sampler   : UtilSampler (결과 프로퍼티 참조 가능)
    n_calls   : 측정 동안 호출된 횟수
    elapsed_s : 실제 경과 시간(초)
    """
    sampler = UtilSampler(interval_ms=interval_ms, gpu=gpu)
    sampler.start()
    t0 = time.time()
    n_calls = 0
    while (time.time() - t0) < min_seconds:
        fn()
        n_calls += 1
    elapsed_s = time.time() - t0
    sampler.stop()
    return sampler, n_calls, elapsed_s
