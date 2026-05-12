#!/usr/bin/env python3
"""
TSLib 기반 시계열 예측 학습 스크립트.

복구: recup_dir.7/f567522392.txt, f567522496.txt, f567522624.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# train_tslib_model.py → parents[2] == project 루트 (호스트: .../project, Docker: /workspace)
_PROJECT_DIR = Path(__file__).resolve().parents[2]
_TSLIB_ROOT = _PROJECT_DIR / "vendor" / "TSLib"
_SRC_ROOT = _PROJECT_DIR / "src"

for _p in (str(_TSLIB_ROOT), str(_SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark.evaluate_model import compute_metrics
from datasets.pv_dataset import (
    TARGET_IDX,
    build_multisite_dataset,
    encoder_input_channel_count,
    load_test_windows,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_configs(
    args: Namespace,
    *,
    enc_in: int | None = None,
    seq_len_model: int | None = None,
) -> SimpleNamespace:
    enc = enc_in if enc_in is not None else int(encoder_input_channel_count())
    slen = int(seq_len_model if seq_len_model is not None else args.seq_len)
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=slen,
        pred_len=int(args.pred_len),
        enc_in=enc,
        dec_in=enc,
        c_out=enc,
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        e_layers=int(args.e_layers),
        d_ff=int(args.d_ff),
        dropout=float(args.dropout),
        moving_avg=25,
        factor=1,
        activation="gelu",
        features="MS",
        embed="timeF",
        freq="h",
        label_len=0,
        num_class=0,
        seg_len=int(args.seg_len),
    )


# Time-LLM KimMeen/Time-LLM: Model.__init__ 가 huggingface 기본 체크포인트 hidden과 맞춤
_TIMELLM_LLM_DIM: dict[str, int] = {
    "GPT2": 768,
    "BERT": 768,
    "LLAMA": 4096,
}


def _augment_configs_for_timellm(configs: SimpleNamespace, args: Namespace) -> None:
    """공식 TimeLLM.Model 이 읽는 configs 필드 보강( llm_dim·patch_len 등 )."""
    key = str(args.llm_model)
    if key not in _TIMELLM_LLM_DIM:
        raise ValueError(
            f"TimeLLM: llm_model={key!r} — 지원 키: {sorted(_TIMELLM_LLM_DIM)}"
        )
    configs.llm_dim = _TIMELLM_LLM_DIM[key]
    configs.llm_model = key
    configs.llm_layers = int(args.llm_layers)
    configs.patch_len = int(args.patch_len)
    configs.stride = int(args.stride)
    configs.prompt_domain = False
    configs.content = ""


def _timellm_repo_root() -> Path:
    """KimMeen/Time-LLM 클론 루트(models/, layers/ 포함)."""
    for root in (
        _PROJECT_DIR / "vendor" / "TimeLLM",
        Path("/workspace/TimeLLM"),
    ):
        r = root.resolve()
        if (r / "models" / "TimeLLM.py").is_file() and (r / "layers" / "Embed.py").is_file():
            return r
    raise ModuleNotFoundError(
        "TimeLLM 저장소 없음: project/vendor/TimeLLM 또는 /workspace/TimeLLM 에 "
        "models/TimeLLM.py 와 layers/Embed.py 가 있어야 함"
    )


def _prepend_sys_path_front(path: Path) -> None:
    s = str(path.resolve())
    while s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)


def _patch_timellm_patch_embedding_input_dtype(model: nn.Module) -> None:
    """
    TimeLLM.forecast 가 patch_embedding 호출 전 x_enc.to(bfloat16) 을 강제하는데,
    학습 스크립트는 model.float() 로 패치 경로는 float32 유지 → dtype 불일치.
    patch_embedding **입력**을 해당 서브모듈 가중치 dtype 으로 맞춘다(state_dict 키 유지).
    """
    pe = getattr(model, "patch_embedding", None)
    if pe is None or not isinstance(pe, nn.Module):
        return
    orig_forward = pe.forward

    def forward_with_input_dtype(x):  # type: ignore[no-untyped-def]
        wdt = next(pe.parameters()).dtype
        return orig_forward(x.to(dtype=wdt))

    pe.forward = forward_with_input_dtype  # type: ignore[method-assign]


def build_model(model_name: str, configs: SimpleNamespace, args: Namespace) -> nn.Module:
    if model_name == "DLinear":
        from models.DLinear import Model
    elif model_name == "SegRNN":
        from models.SegRNN import Model
    elif model_name == "PatchTST":
        from models.PatchTST import Model

        return Model(configs, patch_len=int(args.patch_len), stride=int(args.stride))
    elif model_name == "TimeLLM":
        _augment_configs_for_timellm(configs, args)
        # TSLib이 sys.path 앞에 있으면 `layers.Embed`가 TSLib PatchEmbedding( padding 필수 )로
        # 잡혀 Time-LLM 호출 시그니처와 충돌한다 → Time-LLM 루트를 반드시 맨 앞에 둔다.
        _prepend_sys_path_front(_timellm_repo_root())
        from models.TimeLLM import Model
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")
    return Model(configs)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    target_idx: int = TARGET_IDX,
    *,
    epoch: int | None = None,
    epochs: int | None = None,
    log_batch_every: int = 0,
    phase: str = "train",
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    ep = epoch if epoch is not None else "?"
    ne = epochs if epochs is not None else "?"
    for bi, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)
        y_target = y[:, :, target_idx : target_idx + 1]

        optimizer.zero_grad()
        out = model(x, None, None, None)
        if isinstance(out, tuple):
            out = out[0]
        out_target = out[:, :, target_idx : target_idx + 1]
        loss = criterion(out_target, y_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        if log_batch_every > 0 and (bi % log_batch_every == 0 or bi == n_batches):
            print(
                f"[batch] epoch {ep}/{ne} {phase} step {bi}/{n_batches} "
                f"batch_loss={loss.item():.6f}",
                flush=True,
            )
    return total_loss / max(len(loader), 1)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_idx: int = TARGET_IDX,
    *,
    epoch: int | None = None,
    epochs: int | None = None,
    log_batch_every: int = 0,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = len(loader)
    ep = epoch if epoch is not None else "?"
    ne = epochs if epochs is not None else "?"
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)
            y_target = y[:, :, target_idx : target_idx + 1]
            out = model(x, None, None, None)
            if isinstance(out, tuple):
                out = out[0]
            out_target = out[:, :, target_idx : target_idx + 1]
            step_loss = criterion(out_target, y_target).item()
            total_loss += step_loss

            if log_batch_every > 0 and (bi % log_batch_every == 0 or bi == n_batches):
                print(
                    f"[batch] epoch {ep}/{ne} valid step {bi}/{n_batches} "
                    f"batch_loss={step_loss:.6f}",
                    flush=True,
                )
    return total_loss / max(len(loader), 1)


def generate_predictions(
    model: nn.Module,
    feature_mart_dir: str,
    seq_len: int,
    pred_len: int,
    device: torch.device,
    batch_size: int = 256,
    *,
    merge_future_nwp_into_encoder_input: bool = False,
    future_nwp_variable_names: tuple[str, ...] | None = None,
    as_float32: bool = False,
    align_window_start_to_midnight: bool = True,
) -> list[dict]:
    import glob

    model.eval()
    all_rows: list[dict] = []

    test_paths = sorted(glob.glob(os.path.join(feature_mart_dir, "test", "*.parquet")))
    for path in test_paths:
        site_id = os.path.splitext(os.path.basename(path))[0]
        starts, X, Y = load_test_windows(
            path,
            seq_len=seq_len,
            pred_len=pred_len,
            merge_future_nwp_into_encoder_input=merge_future_nwp_into_encoder_input,
            future_nwp_variable_names=future_nwp_variable_names,
            align_window_start_to_midnight=align_window_start_to_midnight,
        )
        if len(starts) == 0:
            continue

        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[i : i + batch_size]).to(device)
                if as_float32:
                    xb = xb.float()
                out = model(xb, None, None, None)
                if isinstance(out, tuple):
                    out = out[0]
                preds.append(out[:, :, TARGET_IDX].cpu().numpy())

        preds_np = np.clip(np.concatenate(preds, axis=0), 0.0, 1.0)

        for j, ts in enumerate(starts):
            row = {"site_id": site_id, "timestamp": pd.Timestamp(ts)}
            for k in range(pred_len):
                row[f"pred_h{k}"] = float(preds_np[j, k])
            all_rows.append(row)

    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="TSLib 모델 학습")
    parser.add_argument(
        "--model", required=True, choices=["DLinear", "SegRNN", "PatchTST", "TimeLLM"]
    )
    parser.add_argument("--feature-mart", default="artifacts/feature_mart_per_site")
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument(
        "--train-window-stride",
        type=int,
        default=24,
        metavar="ROWS",
        help=(
            "train 윈도 시작 간격(행). 자정 정렬 ON이면 **가장 이른 자정(00:00)에서 시작하는 첫 윈도**를 잡은 뒤, "
            "i0, i0+stride, i0+2*stride … 만 사용(이후 시작 시각은 stride에 따라 자정이 아닐 수 있음). "
            "stride=24·1시간 마트면 이후 시작도 자정에 맞춰짐. "
            "행 0부터 stride만 쓰려면 `--no-midnight-window-align` . "
            "valid·test 슬라이스도 동일 규칙(시작 간격은 pred_len). "
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seg-len", type=int, default=24)
    parser.add_argument("--patch-len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--llm-model", type=str, default="GPT2", choices=["GPT2", "LLAMA", "BERT"])
    parser.add_argument("--llm-layers", type=int, default=6)
    parser.add_argument("--llm-model-path", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--merge-future-nwp-into-encoder-input",
        action="store_true",
        help=(
            "입력 x를 [B, L+H, C'] 로 구성: 앞 L(--seq-len)행은 FEATURE_COLS, "
            "뒤 H(--pred-len)행은 t_end 시각의 wide fan에서 미래 NWP를 동일 채널에 채움."
        ),
    )
    parser.add_argument(
        "--future-nwp-variable-names",
        type=str,
        default="tmp,reh,wsd,vec,sky,pcp",
        metavar="NAMES",
        help="fan에서 읽을 미래 NWP 슬롄(콤마).",
    )
    parser.add_argument(
        "--log-batch-every",
        type=int,
        default=0,
        metavar="N",
        help="N 배치마다 train/valid step 로그 출력 (0이면 비활성)",
    )
    parser.add_argument(
        "--no-midnight-window-align",
        action="store_true",
        help="첫 윈도만 자정(00:00)에 맞추는 동작 끔(윈도 시작 i=0,stride,2*stride…).",
    )
    args = parser.parse_args()

    align_midnight = not args.no_midnight_window_align
    merge_nwp = args.merge_future_nwp_into_encoder_input
    future_nwp_names = tuple(
        p.strip() for p in args.future_nwp_variable_names.split(",") if p.strip()
    )
    seq_len_model = args.seq_len + (args.pred_len if merge_nwp else 0)
    enc_in_eff = encoder_input_channel_count(
        merge_future_nwp_into_encoder_input=merge_nwp
    )

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] model={args.model}, seed={args.seed}, device={device}")
    print(f"[train] context_len(L)={args.seq_len}, pred_len(H)={args.pred_len}")
    print(
        f"[train] window_stride(rows): train={args.train_window_stride}, "
        f"valid={args.pred_len} (non-overlap valid steps); "
        f"start_align_midnight={align_midnight} (첫 윈도만 00:00 앵커)"
    )
    if merge_nwp:
        print(
            f"[train] merge_future_nwp_into_encoder_input: "
            f"model_seq_len={seq_len_model}, enc_in={enc_in_eff}, "
            f"future_nwp_variable_names={future_nwp_names}"
        )

    ds_kw: dict = {}
    if merge_nwp:
        ds_kw["merge_future_nwp_into_encoder_input"] = True
        ds_kw["future_nwp_variable_names"] = future_nwp_names

    mart = args.feature_mart
    if not os.path.isabs(mart):
        # 상대 경로는 project/ 기준 (Docker에서 repo 전체가 아닌 /workspace만 마운트됨)
        mart = str(_PROJECT_DIR / mart)

    if args.train_window_stride < 1:
        raise SystemExit("--train-window-stride must be >= 1")

    train_ds = build_multisite_dataset(
        mart,
        "train",
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.train_window_stride,
        align_window_start_to_midnight=align_midnight,
        **ds_kw,
    )
    valid_ds = build_multisite_dataset(
        mart,
        "valid",
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.pred_len,
        align_window_start_to_midnight=align_midnight,
        **ds_kw,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    configs = build_configs(
        args, enc_in=enc_in_eff, seq_len_model=seq_len_model
    )
    model = build_model(args.model, configs, args).to(device)
    if args.model == "TimeLLM":
        model.float()
        _patch_timellm_patch_embedding_input_dtype(model)
        print("[train] TimeLLM: float32 가중치 사용, patch_embedding 입력 dtype 정렬")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] 파라미터 수: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_cnt = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch=epoch,
            epochs=args.epochs,
            log_batch_every=args.log_batch_every,
            phase="train",
        )
        val_loss = validate(
            model,
            valid_loader,
            criterion,
            device,
            epoch=epoch,
            epochs=args.epochs,
            log_batch_every=args.log_batch_every,
        )
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"[epoch] epoch {epoch:3d}/{args.epochs} | "
            f"train={train_loss:.6f} val={val_loss:.6f} | "
            f"{elapsed:.1f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"  [EarlyStopping] epoch {epoch} at patience={args.patience}")
                break

    with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if args.model == "TimeLLM":
        model.float()
        _patch_timellm_patch_embedding_input_dtype(model)
    print("[train] best model 로드 완료, 테스트 예측 생성 중...")

    rows = generate_predictions(
        model,
        mart,
        args.seq_len,
        args.pred_len,
        device,
        batch_size=args.batch_size * 2,
        merge_future_nwp_into_encoder_input=merge_nwp,
        future_nwp_variable_names=future_nwp_names if merge_nwp else None,
        as_float32=(args.model == "TimeLLM"),
        align_window_start_to_midnight=align_midnight,
    )

    if rows:
        import pandas as pd

        pred_df = pd.DataFrame(rows)
        pred_path = os.path.join(args.output_dir, f"predictions_test_{args.pred_len}h.parquet")
        pred_df.to_parquet(pred_path, index=False)
        print(f"[train] 예측 저장: {pred_path} ({len(pred_df)} rows)")

        pred_cols = [f"pred_h{i}" for i in range(args.pred_len)]
        all_true, all_pred, all_elev = [], [], []
        for site_id, grp in pred_df.groupby("site_id"):
            test_path = os.path.join(mart, "test", f"{site_id}.parquet")
            if not os.path.exists(test_path):
                continue
            raw = pd.read_parquet(test_path)
            cols = ["normalized_power"]
            has_se = "solar_elevation" in raw.columns
            if has_se:
                cols.append("solar_elevation")
            test_df = raw[cols].ffill().fillna(0.0)
            for _, row in grp.iterrows():
                ts = row["timestamp"]
                end_ts = ts + pd.Timedelta(hours=args.pred_len - 1)
                window = test_df.loc[ts:end_ts]
                if len(window) < args.pred_len:
                    continue
                all_true.append(window["normalized_power"].values[: args.pred_len])
                all_pred.append([row[c] for c in pred_cols])
                if has_se:
                    all_elev.append(window["solar_elevation"].values[: args.pred_len])

        if all_true:
            y_true_arr = np.array(all_true)
            y_pred_arr = np.array(all_pred)
            solar_elev_arr = (
                np.array(all_elev) if len(all_elev) == len(all_true) else None
            )
            metrics = compute_metrics(
                y_true_arr, y_pred_arr, solar_elevation=solar_elev_arr
            )
            metrics_path = os.path.join(
                args.output_dir, f"metrics_test_{args.pred_len}h.json"
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            extra = ""
            if metrics.get("daytime_MAE") is not None:
                extra = f" daytime_MAE={metrics['daytime_MAE']:.4f}"
            print(f"[train] MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f}{extra}")

    print(f"[train] 완료 → {args.output_dir}")


if __name__ == "__main__":
    main()
