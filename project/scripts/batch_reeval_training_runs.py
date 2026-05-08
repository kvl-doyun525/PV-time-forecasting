#!/usr/bin/env python3
"""
`artifacts/training_runs` (또는 지정 루트) 이하의 모든
`predictions_test_{H}h.parquet`에 대해 `metrics_test_{H}h.json`을
오프라인으로 다시 계산한다 (`metrics_from_predictions_parquet`).

`predictions_test_*h.parquet` 가 없는 run 은 탐색 대상에 없다 (건너뜀).

사용 예:
    cd project
    python3 scripts/batch_reeval_training_runs.py

하위 디렉터리만:
    python3 scripts/batch_reeval_training_runs.py \\
        --training-runs artifacts/training_runs/segrnn

미리 확인만:
    python3 scripts/batch_reeval_training_runs.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.evaluate_model import metrics_from_predictions_parquet  # noqa: E402

PRED_PATTERN = re.compile(r"predictions_test_(\d+)h\.parquet$")


def horizon_from_filename(path: Path) -> int | None:
    m = PRED_PATTERN.search(path.name)
    if not m:
        return None
    return int(m.group(1))


def discover_prediction_parquets(training_runs_root: Path) -> list[Path]:
    paths = sorted(training_runs_root.rglob("predictions_test_*h.parquet"))
    return [p for p in paths if horizon_from_filename(p) is not None]


def _rel_project(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="training_runs 전체 예측 parquet → metrics_test_*h.json 일괄 재평가",
    )
    parser.add_argument(
        "--training-runs",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "training_runs",
        help="탐색 루트 (재귀)",
    )
    parser.add_argument(
        "--feature-mart",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "feature_mart_per_site",
        help="feature_mart_per_site 경로",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일 목록만 출력하고 지표 계산·저장 안 함",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="첫 오류 시 즉시 종료",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="성공 행 출력 생략 (실패·요약만)",
    )
    args = parser.parse_args()

    root = args.training_runs.resolve()
    fm = str(args.feature_mart.resolve())

    if not root.is_dir():
        sys.exit(f"디렉터리 없음: {root}")
    if not args.feature_mart.resolve().is_dir():
        sys.exit(f"디렉터리 없음: {args.feature_mart}")

    parquet_files = discover_prediction_parquets(root)
    if not parquet_files:
        print(f"[batch_reeval] 예측 parquet 없음 under {root}")
        return

    ok = 0
    failed: list[tuple[str, str]] = []

    for pred_path in parquet_files:
        rel = _rel_project(pred_path)
        h = horizon_from_filename(pred_path)
        assert h is not None
        out_path = pred_path.parent / f"metrics_test_{h}h.json"

        if args.dry_run:
            print(f"[dry-run] {rel} → {_rel_project(out_path)}")
            ok += 1
            continue

        try:
            metrics = metrics_from_predictions_parquet(str(pred_path), fm, pred_len=h)
            if not metrics:
                raise RuntimeError("집계된 윈도우 없음 (feature_mart/test 타임스탬프 불일치 등)")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            if not args.quiet:
                extra = ""
                if metrics.get("daytime_MAE") is not None:
                    extra = f" daytime_MAE={metrics['daytime_MAE']:.6f}"
                print(
                    f"[ok] {rel} | MAE={metrics['MAE']:.6f}{extra} → "
                    f"{out_path.relative_to(PROJECT_ROOT) if out_path.is_relative_to(PROJECT_ROOT) else out_path}"
                )
            ok += 1
        except Exception as e:
            msg = f"{e!s}"
            failed.append((str(rel), msg))
            print(f"[fail] {rel}: {msg}", file=sys.stderr)
            if not args.quiet:
                traceback.print_exc()
            if args.fail_fast:
                sys.exit(1)

    print(
        f"[batch_reeval] 완료 | 성공 {ok} / 실패 {len(failed)} "
        f"(예측 parquet {len(parquet_files)}개 처리 대상)"
    )
    if failed:
        print("[batch_reeval] 실패 목록:", file=sys.stderr)
        for path, err in failed:
            print(f"  - {path}: {err}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
