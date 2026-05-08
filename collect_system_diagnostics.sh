#!/usr/bin/env bash
# 시스템 프리즈·OOM·GPU·IRQ 재현 시 로그를 파일로 수집한다.
#
# 사용법:
#   ./collect_system_diagnostics.sh snapshot          # 1회 스냅샷 (sudo 권장: dmesg/journal 일부)
#   ./collect_system_diagnostics.sh start             # 백그라운드 연속 수집 시작
#   ./collect_system_diagnostics.sh stop              # start로 띄운 수집 중지
#   LOG_ROOT=/path ./collect_system_diagnostics.sh snapshot
#
# 환경변수:
#   LOG_ROOT   로그 루트 (기본: 이 스크립트가 있는 디렉터리 아래 logs/freeze_diagnostics)
#   NV_INTERVAL nvidia-smi 샘플 간격 초 (기본: 1)
#   VM_INTERVAL vmstat 간격 초 (기본: 1)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs/freeze_diagnostics}"
NV_INTERVAL="${NV_INTERVAL:-1}"
VM_INTERVAL="${VM_INTERVAL:-1}"
PID_FILE="$LOG_ROOT/.collect_pids"
STATE_DIR=""

ts_dir() { date +%Y%m%d_%H%M%S; }

ensure_logdir() {
  local sub="$1"
  STATE_DIR="$LOG_ROOT/$sub"
  mkdir -p "$STATE_DIR"
  echo "$STATE_DIR"
}

run_sudo() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
}

# ── 스냅샷 (1회) ───────────────────────────────────────────────────────────
cmd_snapshot() {
  local out
  out="$(ensure_logdir "$(ts_dir)_snapshot")"
  echo "[snapshot] 출력: $out"

  {
    echo "=== date ==="
    date -Is
    echo "=== uname -a ==="
    uname -a
    echo "=== uptime ==="
    uptime
    echo "=== free -h ==="
    free -h || true
    echo "=== df -h ==="
    df -h || true
  } >"$out/host_summary.txt" 2>&1

  run_sudo dmesg -T >"$out/dmesg.txt" 2>&1 || dmesg -T >"$out/dmesg_no_sudo.txt" 2>&1 || true

  journalctl -b -0 --no-pager -n 800 >"$out/journal_last800.txt" 2>&1 || true
  run_sudo journalctl -b -0 -k --no-pager -n 400 >"$out/journal_kernel_last400.txt" 2>&1 || true

  journalctl -u systemd-oomd --no-pager -n 200 >"$out/journal_systemd_oomd_last200.txt" 2>&1 || true

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -q >"$out/nvidia_smi_query.txt" 2>&1 || true
    nvidia-smi --query-gpu=timestamp,name,driver_version,temperature.gpu,power.draw,clocks.sm,utilization.gpu,utilization.memory,memory.used,memory.total \
      --format=csv >"$out/nvidia_smi_query_csv.txt" 2>&1 || true
  else
    echo "nvidia-smi 없음" >"$out/nvidia_smi_skipped.txt"
  fi

  cat /proc/interrupts >"$out/proc_interrupts.txt" 2>&1 || true
  lsmod >"$out/lsmod.txt" 2>&1 || true

  if command -v docker >/dev/null 2>&1; then
    docker ps -a >"$out/docker_ps_a.txt" 2>&1 || true
    docker info >"$out/docker_info.txt" 2>&1 || true
  fi

  if command -v sar >/dev/null 2>&1; then
    sar -u -r -S 1 3 >"$out/sar_sample_3s.txt" 2>&1 || true
  else
    echo "sar 없음 (apt install sysstat)" >"$out/sar_skipped.txt"
  fi

  echo "[snapshot] 완료: $out"
}

# ── 연속 수집 (백그라운드) ─────────────────────────────────────────────────
cmd_start() {
  if [[ -f "$PID_FILE" ]]; then
    echo "이미 수집 중일 수 있음: $PID_FILE 가 존재합니다. ./collect_system_diagnostics.sh stop 후 다시 시도하세요." >&2
    exit 1
  fi

  STATE_DIR="$(ensure_logdir "$(ts_dir)_follow")"
  echo "[start] 출력: $STATE_DIR"
  mkdir -p "$LOG_ROOT"
  echo "$STATE_DIR" >"$PID_FILE.dir"

  : >"$PID_FILE"
  local pids=()

  journalctl -b -0 --no-pager -n 300 >"$STATE_DIR/journal_bootstrap.txt" 2>&1 || true
  run_sudo dmesg -T >"$STATE_DIR/dmesg_at_start.txt" 2>&1 || true

  # 커널/전체 로그 스트림 (커널 전체는 systemd-journal 그룹 또는 sudo 권장)
  journalctl -k -f --no-hostname -n 0 >"$STATE_DIR/journal_kernel_follow.log" 2>&1 &
  pids+=($!)
  journalctl -f --no-hostname -n 0 >"$STATE_DIR/journal_all_follow.log" 2>&1 &
  pids+=($!)

  journalctl -u systemd-oomd -f --no-pager -n 0 >"$STATE_DIR/journal_oomd_follow.log" 2>&1 &
  pids+=($!)

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,name,temperature.gpu,power.draw,clocks.sm,utilization.gpu,utilization.memory,memory.used,memory.total \
      --format=csv -l "$NV_INTERVAL" >"$STATE_DIR/nvidia_smi_loop.csv" 2>&1 &
    pids+=($!)
  fi

  vmstat "$VM_INTERVAL" >"$STATE_DIR/vmstat.log" 2>&1 &
  pids+=($!)

  if command -v sar >/dev/null 2>&1; then
    sar -u -r -S "$VM_INTERVAL" >"$STATE_DIR/sar_follow.log" 2>&1 &
    pids+=($!)
  fi

  if command -v docker >/dev/null 2>&1; then
    docker events >"$STATE_DIR/docker_events.log" 2>&1 &
    pids+=($!)
  fi

  for pid in "${pids[@]}"; do
    echo "$pid" >>"$PID_FILE"
  done

  echo "[start] 백그라운드 PID: ${pids[*]}"
  echo "[start] PID 목록: $PID_FILE"
  echo "[start] 중지: $0 stop"
}

cmd_stop() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "PID 파일 없음: $PID_FILE (start를 먼저 실행했는지 확인)" >&2
    exit 1
  fi
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
      echo "종료: PID $pid"
      kill "$pid" 2>/dev/null || true
    fi
  done <"$PID_FILE"
  rm -f "$PID_FILE"
  echo "[stop] 완료 (자식 프로세스가 남았다면 수동 확인: ps aux | grep journalctl)"
}

usage() {
  cat <<EOF
사용법: $0 {snapshot|start|stop}

  snapshot  현재 시점 1회 덤프 (OOM/IRQ/GPU/메모리/도커 등)
  start     journal/nvidia/vmstat/sar/docker events 를 백그라운드로 파일에 기록
  stop      start 로 기록한 PID 종료

환경변수: LOG_ROOT=경로  NV_INTERVAL=초  VM_INTERVAL=초

start/stop는 동일 사용자로 실행할 것 (sudo ./script start 했으면 sudo ./script stop).
커널 링 버퍼 전체는 권한이 없을 수 있으면: sudo ./collect_system_diagnostics.sh snapshot
EOF
}

main() {
  local cmd="${1:-}"
  case "$cmd" in
    snapshot) cmd_snapshot ;;
    start)    cmd_start ;;
    stop)     cmd_stop ;;
    *)        usage; exit 1 ;;
  esac
}

main "$@"
