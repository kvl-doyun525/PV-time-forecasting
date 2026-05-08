#!/usr/bin/env python3
"""
학습 가중치 없이(구조·입력 shape만) CPU 추론 지연을 측정한다.

- TSLib: `train_tslib_model.build_model` + 더미 입력
- Time-LLM / LLaMA / Gemma: transformers 설치 시 백본만 경량 측정, 없으면 스킵

결과 JSON은 `aggregate_results.py` 가 수집할 수 있도록 스키마를 맞춘다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
_TSLIB_ROOT = _PROJECT_ROOT / "vendor" / "TSLib"

for _p in (str(_SRC_ROOT), str(_TSLIB_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark.cpu_setup import setup_cpu_benchmark  # noqa: E402
from datasets.pv_dataset import encoder_input_channel_count  # noqa: E402
from train_tslib_model import build_configs, build_model, set_seed  # noqa: E402


def _latency_stats(forward: Callable[[], None], *, warmup: int, runs: int) -> dict[str, float]:
    for _ in range(warmup):
        forward()
    ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        forward()
        ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(ms, dtype=np.float64)
    return {
        "warm_p50_ms": float(np.percentile(arr, 50)),
        "warm_p95_ms": float(np.percentile(arr, 95)),
        "warm_mean_ms": float(np.mean(arr)),
        "warm_std_ms": float(np.std(arr)),
    }


def _bench_tslib(name: str, args: Namespace, *, seed: int) -> dict[str, Any]:
    set_seed(seed)
    enc_in = encoder_input_channel_count()
    configs = build_configs(args, enc_in=enc_in, seq_len_model=args.seq_len)
    model = build_model(name, configs, args)
    model.eval()
    x = torch.randn(1, args.seq_len, enc_in, dtype=torch.float32)

    def forward() -> None:
        with torch.no_grad():
            out = model(x, None, None, None)
            if isinstance(out, tuple):
                _ = out[0]
            else:
                _ = out

    stats = _latency_stats(forward, warmup=3, runs=20)
    t0 = time.perf_counter()
    with torch.no_grad():
        out0 = model(x, None, None, None)
        if isinstance(out0, tuple):
            _ = out0[0]
    stats["cold_start_ms"] = (time.perf_counter() - t0) * 1000.0
    stats["model"] = name
    stats["seq_len"] = int(args.seq_len)
    stats["pred_len"] = int(args.pred_len)
    stats["enc_in"] = int(enc_in)
    return stats


def _bench_gpt2_backbone(*, seed: int) -> dict[str, Any] | None:
    try:
        from transformers import GPT2Model  # type: ignore
    except Exception:
        return None
    set_seed(seed)
    m = GPT2Model.from_pretrained("gpt2")
    m.eval()
    hidden = m.config.hidden_size
    x = torch.randn(1, 64, hidden)

    def forward() -> None:
        with torch.no_grad():
            m(inputs_embeds=x)

    stats = _latency_stats(forward, warmup=2, runs=10)
    t0 = time.perf_counter()
    with torch.no_grad():
        m(inputs_embeds=x)
    stats["cold_start_ms"] = (time.perf_counter() - t0) * 1000.0
    stats["model"] = "time_llm_gpt2_backbone_stub"
    stats["note"] = "GPT2Model inputs_embeds, seq=64 (Time-LLM 백본 대용)"
    return stats


def _run_all(args: Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {"results": {}}
    ts_args = Namespace(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_ff=256,
        dropout=0.1,
        seg_len=24,
        patch_len=16,
        stride=8,
    )
    for name in ("DLinear", "SegRNN", "PatchTST"):
        out["results"][name.lower()] = {
            f"horizon_{args.pred_len}h": _bench_tslib(name, ts_args, seed=args.seed)
        }
    gpt = _bench_gpt2_backbone(seed=args.seed)
    if gpt:
        out["results"]["time_llm_gpt2"] = {f"horizon_{args.pred_len}h": gpt}
    else:
        out["results"]["time_llm_gpt2"] = {
            f"horizon_{args.pred_len}h": {"skipped": True, "reason": "transformers 미설치"}
        }
    out["results"]["llama_1b"] = {
        f"horizon_{args.pred_len}h": {
            "skipped": True,
            "reason": "CPU 벤치 전용 스텁 — train_llama_lora 체크포인트 연동 시 확장",
        }
    }
    out["results"]["gemma_e2b"] = {
        f"horizon_{args.pred_len}h": {
            "skipped": True,
            "reason": "CPU 벤치 전용 스텁 — train_gemma 연동 시 확장",
        }
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="CPU 추론 벤치마크 (구조 기준)")
    p.add_argument(
        "--model",
        default="dlinear",
        help="dlinear|segrnn|patchtst|time_llm|all",
    )
    p.add_argument("--seq-len", type=int, default=168)
    p.add_argument("--pred-len", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-json",
        default="",
        help="결과 JSON 경로 (기본: artifacts/cpu_benchmark_last.json)",
    )
    args_ns = p.parse_args()

    setup_cpu_benchmark()
    key = args_ns.model.lower().replace("-", "_")
    default_out = _PROJECT_ROOT / "artifacts" / "cpu_benchmark_last.json"
    out_path = Path(args_ns.output_json) if args_ns.output_json else default_out
    if not out_path.is_absolute():
        out_path = _PROJECT_ROOT / out_path

    if key == "all":
        payload = _run_all(args_ns)
    elif key in ("dlinear", "segrnn", "patchtst"):
        name = {"dlinear": "DLinear", "segrnn": "SegRNN", "patchtst": "PatchTST"}[key]
        ts_args = Namespace(
            seq_len=args_ns.seq_len,
            pred_len=args_ns.pred_len,
            d_model=128,
            n_heads=8,
            e_layers=2,
            d_ff=256,
            dropout=0.1,
            seg_len=24,
            patch_len=16,
            stride=8,
        )
        payload = {
            "results": {
                key: {f"horizon_{args_ns.pred_len}h": _bench_tslib(name, ts_args, seed=args_ns.seed)}
            }
        }
    elif key in ("time_llm", "timellm"):
        gpt = _bench_gpt2_backbone(seed=args_ns.seed)
        payload = {
            "results": {
                "time_llm_gpt2": {
                    f"horizon_{args_ns.pred_len}h": gpt or {"skipped": True}
                }
            }
        }
    else:
        raise SystemExit(f"지원하지 않는 --model: {args_ns.model}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[cpu_bench] 저장: {out_path}")


if __name__ == "__main__":
    main()
