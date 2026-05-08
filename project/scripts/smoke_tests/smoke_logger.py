"""스모크 테스트 공통 로거 - stdout/stderr를 파일과 콘솔에 동시 출력."""
import sys
import os
import atexit
from datetime import datetime


class _Tee:
    """stdout/stderr를 파일과 터미널에 동시 출력."""
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile  = logfile

    def write(self, data):
        self.terminal.write(data)
        self.terminal.flush()
        self.logfile.write(data)
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def fileno(self):
        return self.terminal.fileno()


def _write_footer(logfile, success: bool) -> None:
    status = "완료" if success else "실패"
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        # stdout/stderr를 원본으로 복원한 뒤 파일 닫기
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        logfile.write(f"\n{status}: {ts}\n")
        logfile.flush()
        logfile.close()
    except Exception:
        pass


def setup_log(script_name: str, log_dir: str = "/workspace/logs/smoke_test") -> str:
    """
    로그 파일을 열고 stdout/stderr를 파일과 콘솔에 동시 기록하도록 설정.
    atexit 핸들러를 등록하므로 스크립트가 정상/비정상 종료 시 모두 footer가 기록됨.

    Parameters
    ----------
    script_name : 예) '01_dlinear_gpu'
    log_dir     : 컨테이너 내부 로그 디렉토리 (기본값: /workspace/logs/smoke_test)

    Returns
    -------
    log_path : 생성된 로그 파일의 절대 경로
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{script_name}.log")

    logfile = open(log_path, "w", buffering=1, encoding="utf-8")
    logfile.write(
        f"=== {script_name} ===\n"
        f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    logfile.flush()

    sys.stdout = _Tee(sys.__stdout__, logfile)
    sys.stderr = _Tee(sys.__stderr__, logfile)

    # 스크립트 종료 시 자동으로 footer 기록
    atexit.register(_write_footer, logfile, True)

    return log_path
