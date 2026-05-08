#!/usr/bin/env python3
"""
지정한 폴더들 아래의 .png 파일을 삭제한다.

  python3 scripts/delete_pngs_in_folders.py recup_dir.1 recup_dir.2
  python3 scripts/delete_pngs_in_folders.py --dry-run /path/to/a /path/to/b
  python3 scripts/delete_pngs_in_folders.py --no-recursive ./graphs_only_here
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def iter_pngs(root: Path, recursive: bool) -> list[Path]:
    if not root.is_dir():
        return []
    if recursive:
        return sorted(root.rglob("*.pdf"))
    return sorted(root.glob("*.pdf"))


def main() -> None:
    p = argparse.ArgumentParser(description="여러 폴더에서 PNG 파일 삭제")
    p.add_argument(
        "folders",
        nargs="+",
        type=Path,
        help="대상 디렉터리(여러 개 가능)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="삭제하지 않고 목록만 출력",
    )
    p.add_argument(
        "--no-recursive",
        action="store_true",
        help="각 폴더 직속 .png만 (하위 폴더 제외)",
    )
    args = p.parse_args()
    recursive = not args.no_recursive

    deleted = 0
    errors: list[tuple[Path, str]] = []

    for folder in args.folders:
        root = folder.expanduser().resolve()
        paths = iter_pngs(root, recursive)
        if not paths:
            print(f"[skip] PNG 없음 또는 디렉터리 아님: {root}", file=sys.stderr)
            continue
        for path in paths:
            rel = path
            try:
                rel = path.relative_to(Path.cwd())
            except ValueError:
                pass
            if args.dry_run:
                print(f"[dry-run] {rel}")
                deleted += 1
                continue
            try:
                path.unlink()
                print(f"[del] {rel}")
                deleted += 1
            except OSError as e:
                errors.append((path, str(e)))

    print(f"완료: {'(시뮬)' if args.dry_run else ''} 처리 {deleted}개 PNG")
    if errors:
        print("오류:", file=sys.stderr)
        for path, msg in errors:
            print(f"  {path}: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
