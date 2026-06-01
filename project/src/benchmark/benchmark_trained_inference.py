#!/usr/bin/env python3
"""
학습된 `best_model.pt` 체크포인트로 추론 지연을 측정한다.

- 첫 배치(warmup) 1회는 제외
- batch_size=1, batch_size=100 각각 반복 측정
- `--representative` 이면 리포트용 대표 run만 (모델×horizon×seq_168 계열)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
_TSLIB_ROOT = _PROJECT_ROOT / "vendor" / "TSLib"

for _p in (str(_SRC_ROOT), str(_SRC_ROOT / "train"), str(_TSLIB_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.pv_dataset import encoder_input_channel_count  # noqa: E402
from train_tslib_model import (  # noqa: E402
    build_configs,
    build_model,
    set_seed,
    _patch_timellm_patch_embedding_input_dtype,
)

_RE_METRICS_H = re.compile(r"metrics_test_(\d+)h\.json$")

# (model_key, horizon) → training_runs 상대 경로 (seq_len=168 대표)
REPRESENTATIVE_RUNS: dict[tuple[str, int], str] = {
    ("dlinear", 24): "dlinear_seq_168/seed_42",
    ("dlinear", 48): "dlinear_seq_168/h48_seed_42",
    ("dlinear", 72): "dlinear_seq_168/h72_seed_42",
    ("segrnn", 24): "segrnn_seq_168/seg24_h24_seed42",
    ("segrnn", 48): "segrnn_seq_168/seg24_h48_seed42",
    ("segrnn", 72): "segrnn_seq_168/seg24_h72_seed42",
    ("patchtst", 24): "patchtst_seq_168/pl48_s48_h24_seed42",
    ("patchtst", 48): "patchtst_seq_168/pl48_s48_h48_seed42",
    ("patchtst", 72): "patchtst_seq_168/pl48_s48_h72_seed42",
    ("timellm", 24): "timellm_future_nwp_seq_168/timellm_gpt2_h24_seed42",
    ("timellm", 48): "timellm_future_nwp_seq_168/timellm_gpt2_h48_seed42",
    ("timellm", 72): "timellm_future_nwp_seq_168/timellm_gpt2_h72_seed42",
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _stats_ms(samples: list[float]) -> dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def _forward(model: torch.nn.Module, x: torch.Tensor) -> None:
    with torch.no_grad():
        out = model(x, None, None, None)
        if isinstance(out, tuple):
            _ = out[0]
        else:
            _ = out


def _bench_batch_size(
    model: torch.nn.Module,
    *,
    batch_size: int,
    seq_len: int,
    enc_in: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    dtype = torch.float32
    x = torch.randn(batch_size, seq_len, enc_in, dtype=dtype, device=device)

    for _ in range(warmup):
        _forward(model, x)
    _sync(device)

    times: list[float] = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        _forward(model, x)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    out = _stats_ms(times)
    out["batch_size"] = float(batch_size)
    out["per_sample_ms"] = out["mean_ms"] / batch_size
    return out


def _bench_batch100_with_fallback(
    model: torch.nn.Module,
    *,
    target_batch: int,
    seq_len: int,
    enc_in: int,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    for bs in (target_batch, 32, 16, 8, 4, 1):
        if bs > target_batch:
            continue
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return _bench_batch_size(
                model,
                batch_size=bs,
                seq_len=seq_len,
                enc_in=enc_in,
                device=device,
                warmup=warmup,
                repeats=repeats,
            )
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
    raise RuntimeError(f"batch100 측정 실패 (seq_len={seq_len}, target={target_batch})")


def _load_model_from_checkpoint(
    ckpt_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_dict = dict(ckpt.get("args") or {})
    if not args_dict:
        raise ValueError(f"체크포인트에 args 없음: {ckpt_path}")

    args = Namespace(**args_dict)
    merge_nwp = bool(getattr(args, "merge_future_nwp_into_encoder_input", False))
    seq_len_model = int(args.seq_len) + (int(args.pred_len) if merge_nwp else 0)
    enc_in = encoder_input_channel_count()

    configs = build_configs(args, enc_in=enc_in, seq_len_model=seq_len_model)
    model = build_model(str(args.model), configs, args)
    model.load_state_dict(ckpt["model_state"])
    model.float()
    if str(args.model) == "TimeLLM":
        _patch_timellm_patch_embedding_input_dtype(model)
    model.to(device)
    model.eval()

    meta = {
        "model": str(args.model),
        "seq_len": int(args.seq_len),
        "seq_len_model": seq_len_model,
        "pred_len": int(args.pred_len),
        "enc_in": enc_in,
        "merge_future_nwp": merge_nwp,
        "as_float32": str(args.model) == "TimeLLM",
    }
    return model, meta


def _horizon_from_run(run_dir: Path) -> int | None:
    for p in run_dir.glob("metrics_test_*h.json"):
        m = _RE_METRICS_H.search(p.name)
        if m:
            return int(m.group(1))
    return None


def _model_key(name: str) -> str:
    n = name.lower()
    return "timellm" if n == "timellm" else n


def _discover_runs(
    runs_root: Path,
    *,
    representative: bool,
    model_filter: set[str] | None = None,
) -> list[Path]:
    if representative:
        out: list[Path] = []
        for (model_key, _h), rel in REPRESENTATIVE_RUNS.items():
            if model_filter and model_key not in model_filter:
                continue
            d = runs_root / rel
            if (d / "best_model.pt").is_file():
                out.append(d)
        return sorted(set(out))

    found: list[Path] = []
    for ckpt in sorted(runs_root.rglob("best_model.pt")):
        found.append(ckpt.parent)
    return found


def _bench_one_run(
    run_dir: Path,
    runs_root: Path,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    ckpt_path = run_dir / "best_model.pt"
    rel = str(run_dir.relative_to(runs_root))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    set_seed(seed)
    model, meta = _load_model_from_checkpoint(ckpt_path, device)

    b1 = _bench_batch_size(
        model,
        batch_size=1,
        seq_len=meta["seq_len_model"],
        enc_in=meta["enc_in"],
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    b100 = _bench_batch100_with_fallback(
        model,
        target_batch=100,
        seq_len=meta["seq_len_model"],
        enc_in=meta["enc_in"],
        device=device,
        warmup=warmup,
        repeats=repeats,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pred_len = meta["pred_len"] or _horizon_from_run(run_dir)
    return {
        "run_dir": rel,
        "model": meta["model"],
        "model_key": meta["model"].lower().replace("timellm", "timellm"),
        "seq_len": meta["seq_len"],
        "pred_len": pred_len,
        "merge_future_nwp": meta["merge_future_nwp"],
        "batch1": b1,
        "batch100": b100,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="학습 체크포인트 추론 지연 측정")
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/training_runs"),
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/inference_benchmark.json"),
    )
    ap.add_argument("--warmup", type=int, default=1, help="제외할 warmup forward 횟수")
    ap.add_argument("--repeats", type=int, default=20, help="측정 반복 횟수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--representative",
        action="store_true",
        help="리포트용 대표 run만 측정 (seq_168 계열)",
    )
    ap.add_argument(
        "--models",
        default="",
        help="쉼표 구분 모델 필터 (dlinear,segrnn,patchtst,timellm)",
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
    )
    args = ap.parse_args()

    runs_root = args.runs_dir if args.runs_dir.is_absolute() else _PROJECT_ROOT / args.runs_dir
    out_path = (
        args.output_json
        if args.output_json.is_absolute()
        else _PROJECT_ROOT / args.output_json
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_filter = {m.strip().lower() for m in args.models.split(",") if m.strip()}

    run_dirs = _discover_runs(
        runs_root, representative=args.representative, model_filter=model_filter or None
    )
    entries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        try:
            entry = _bench_one_run(
                run_dir,
                runs_root,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"[infer_bench] skip {run_dir}: {exc}", flush=True)
            continue

        entries.append(entry)
        print(
            f"[infer_bench] {entry['run_dir']} "
            f"b1={entry['batch1']['mean_ms']:.2f}ms "
            f"b100={entry['batch100']['mean_ms']:.2f}ms",
            flush=True,
        )

    payload = {
        "device": str(device),
        "pytorch": torch.__version__,
        "warmup_batches": int(args.warmup),
        "repeats": int(args.repeats),
        "representative_only": bool(args.representative),
        "entries": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[infer_bench] 저장: {out_path} ({len(entries)} runs)")


if __name__ == "__main__":
    main()
