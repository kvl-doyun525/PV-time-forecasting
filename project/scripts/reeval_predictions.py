#!/usr/bin/env python3
"""
예측 parquet 만으로 테스트 지표를 다시 계산한다 (학습 재실행 불필요).

`train_tslib_model` 가 저장하는 `predictions_test_{H}h.parquet` 형식과 동일해야 한다:
    site_id, timestamp, pred_h0 … pred_h{H-1}

평가 규칙은 `train_tslib_model.py` 마지막 평가 블록과 동일하며,
`metrics_from_predictions_parquet()` → `compute_metrics()` 경로를 사용한다.

사용 예:
    cd project
    python3 scripts/reeval_predictions.py \\
        --predictions artifacts/training_runs/segrnn/seg24_h24_seed42/predictions_test_24h.parquet \\
        --feature-mart artifacts/feature_mart_per_site

출력 경로 미지정 시 같은 디렉터리에 `metrics_test_{H}h.json` 저장.

추가로 site별 분해 결과가 필요하면 `--detail-output`:

    python3 scripts/reeval_predictions.py \\
        --predictions .../predictions_test_24h.parquet \\
        --feature-mart artifacts/feature_mart_per_site \\
        --detail-output .../metrics_eval_detail.json

(site별 분해는 `evaluate_all_sites()` 경로이며, overall 과 단일 집계와 미세 차이 있을 수 있음)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.evaluate_model import (  # noqa: E402
    evaluate_all_sites,
    infer_pred_len_from_columns,
    metrics_from_predictions_parquet,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="예측 parquet 오프라인 재평가")
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="predictions_test_{H}h.parquet 경로",
    )
    parser.add_argument(
        "--feature-mart",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "feature_mart_per_site",
        help="feature_mart_per_site 루트",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="예측 길이(시간). 생략 시 pred_h* 컬럼으로 추론",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="metrics JSON 경로 (기본: 예측 파일과 같은 폴더의 metrics_test_{H}h.json)",
    )
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=None,
        help="optional: overall + per_site 구조 JSON (evaluate_all_sites)",
    )
    args = parser.parse_args()

    pred_path = args.predictions.resolve()
    if not pred_path.is_file():
        raise SystemExit(f"파일 없음: {pred_path}")

    fm = str(args.feature_mart.resolve())
    pred_str = str(pred_path)

    import pandas as pd

    pred_df = pd.read_parquet(pred_str)
    h = args.horizon if args.horizon is not None else infer_pred_len_from_columns(pred_df)

    metrics = metrics_from_predictions_parquet(pred_str, fm, pred_len=h)
    if not metrics:
        raise SystemExit(
            "집계된 윈도우가 없습니다. feature_mart/test 의 site parquet 경로·타임스탬프를 확인하세요."
        )

    out_path = args.output
    if out_path is None:
        out_path = pred_path.parent / f"metrics_test_{h}h.json"
    else:
        out_path = out_path.resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    extra = ""
    if metrics.get("daytime_MAE") is not None:
        extra = f" daytime_MAE={metrics['daytime_MAE']:.6f}"
    print(f"MAE={metrics['MAE']:.6f} RMSE={metrics['RMSE']:.6f}{extra}")
    print(f"저장: {out_path}")

    if args.detail_output is not None:
        detail = evaluate_all_sites(pred_str, fm, horizon=h)
        detail_path = args.detail_output.resolve()
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2, ensure_ascii=False)
        print(f"상세(overall+per_site): {detail_path}")


if __name__ == "__main__":
    main()
