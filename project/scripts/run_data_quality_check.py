#!/usr/bin/env python3
"""`dataset/preprocessor/run.py quality-check` 래퍼 (`pv_model_benchmark_execution.md` §4.1)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="feature mart 품질 검사 래퍼")
    ap.add_argument(
        "--feature-mart-dir",
        type=Path,
        required=True,
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    run_py = repo / "dataset" / "preprocessor" / "run.py"
    cmd = [
        sys.executable,
        str(run_py),
        "quality-check",
        "--feature-mart-dir",
        str(args.feature_mart_dir.resolve()),
    ]
    raise SystemExit(subprocess.call(cmd, cwd=str(run_py.parent)))


if __name__ == "__main__":
    main()
