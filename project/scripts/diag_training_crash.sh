#!/usr/bin/env bash
# 학습/도커 프로세스가 조용히 죽었을 때 커널·docker 로그에서 힌트 조회
# 사용: bash scripts/diag_training_crash.sh [journalctl --since 인자, 기본: "2 hours ago"]
# 예: bash scripts/diag_training_crash.sh "2026-05-14 17:20:00"
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SINCE="${1:-2 hours ago}"

echo "=== 범위: journalctl --since '$SINCE' ==="
echo ""

echo "=== kernel (-k): segfault / OOM / GPU (nvrm|xid) ==="
journalctl -k --since "$SINCE" --no-pager 2>/dev/null | grep -iE 'segfault|oom|killed process|out of memory|nvrm|xid|gpu has fallen' || echo "(해당 없음)"
echo ""

echo "=== kernel (-k): python / torch 관련 (최근 40줄) ==="
journalctl -k --since "$SINCE" --no-pager 2>/dev/null | grep -iE 'python|torch|cuda' | tail -40 || echo "(해당 없음)"
echo ""

echo "=== user: docker / containerd (task-delete, shim, oom) ==="
journalctl --since "$SINCE" --no-pager 2>/dev/null | grep -iE 'dockerd|containerd|task-delete|shim disconnected|oom-kill' | tail -40 || echo "(해당 없음)"
echo ""

echo "=== dmesg (root 권한이면 OOM 등 추가 확인) ==="
if dmesg -T 2>/dev/null | tail -5 >/dev/null; then
  dmesg -T 2>/dev/null | grep -iE 'oom|killed process|segfault|nvrm|xid' | tail -20 || true
else
  echo "dmesg: 권한 없음 → sudo dmesg -T | grep -iE 'oom|segfault|nvrm'"
fi
