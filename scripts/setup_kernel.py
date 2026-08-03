"""전용 Jupyter 커널 등록.

기본 `python3` 커널을 쓰면 어느 환경에서 실행됐는지 노트북만 보고 알 수 없다.
`banggoot-model` 이름으로 등록해 감사 흔적을 명확히 한다.

    .venv\\Scripts\\python.exe scripts/setup_kernel.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

KERNEL_NAME = "banggoot-model"
DISPLAY_NAME = "Python 3.11 (banggoot-model)"
REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    running = Path(sys.executable).resolve()

    if venv_python.exists() and running != venv_python.resolve():
        print(f"경고: 이 스크립트를 전용 venv 로 실행해야 한다.\n"
              f"  현재: {running}\n  기대: {venv_python}")
        return 1

    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install", "--user",
         "--name", KERNEL_NAME, "--display-name", DISPLAY_NAME],
        check=True,
    )
    print(f"\n커널 등록 완료: {KERNEL_NAME}")
    print(f"  interpreter: {sys.executable}")
    print("\n노트북 실행:")
    print(f"  .venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute --inplace \\")
    print(f"      --ExecutePreprocessor.kernel_name={KERNEL_NAME} notebooks/00_setup_env_check.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
