"""
fashion-hit-engine launcher.

This is the REAL entry point. start.bat is a tiny wrapper that only does:
    python launch.py

Keeping all logic in Python avoids every Windows batch encoding trap
(BOM, codepage mismatch, LF vs CRLF, non-ASCII echo text, etc.).
"""

from __future__ import annotations

import subprocess
import sys
import os
import shutil
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "fashion-hit-engine"
VENV_DIR = ".venv"
REQUIREMENTS = "requirements.txt"
APP_ENTRY = "app.py"
HOST = "localhost"
PORT = 8501

TSINGHUA_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(msg: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {msg}\n{bar}\n")


def _step(n: int, total: int, text: str) -> None:
    print(f"[{n}/{total}] {text}")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, capture output, return CompletedProcess."""
    print(f"    > {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def _fail(msg: str, hint: str = "") -> None:
    _banner(f"ERROR: {msg}")
    if hint:
        print(hint)
    print("\nPress Enter to exit...")
    try:
        input()
    except EOFError:
        pass
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1: Python check
# ---------------------------------------------------------------------------

def check_python() -> None:
    _step(1, 4, "Checking Python...")

    version = sys.version_info
    print(f"    Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 10):
        _fail(
            f"Python {version.major}.{version.minor} is too old.",
            "Please install Python 3.10+ from https://www.python.org/downloads/\n"
            "IMPORTANT: Check 'Add Python to PATH' during installation.",
        )


# ---------------------------------------------------------------------------
# Step 2: venv
# ---------------------------------------------------------------------------

def setup_venv() -> str:
    _step(2, 4, "Setting up virtual environment...")

    venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
    if not os.name == "nt":  # Unix fallback (for dev machines)
        venv_python = os.path.join(VENV_DIR, "bin", "python")

    if os.path.isfile(venv_python):
        print("    venv exists, skipping creation.")
        return venv_python

    print("    Creating venv...")
    result = _run([sys.executable, "-m", "venv", VENV_DIR])
    if result.returncode != 0:
        _fail(
            "Failed to create virtual environment.",
            "Most likely: Python was installed WITHOUT 'Add Python to PATH'.\n"
            "Fix: Reinstall Python and CHECK 'Add Python to PATH'.",
        )
    print("    venv created.")
    return venv_python


# ---------------------------------------------------------------------------
# Step 3: install deps
# ---------------------------------------------------------------------------

def install_deps(venv_python: str) -> None:
    _step(3, 4, "Installing dependencies...")

    if not os.path.isfile(REQUIREMENTS):
        _fail(f"'{REQUIREMENTS}' not found. Make sure this file is complete.")

    pip_upgrade = _run(
        [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
        capture_output=True, text=True,
    )
    if pip_upgrade.returncode != 0:
        print("    (pip upgrade skipped, will try install anyway)")

    # First attempt: default index
    result = _run(
        [venv_python, "-m", "pip", "install", "-r", REQUIREMENTS],
    )
    if result.returncode == 0:
        print("    Dependencies OK.")
        return

    # Second attempt: Tsinghua mirror (China users)
    print("\n    Default mirror failed. Retrying with Tsinghua mirror...\n")
    result2 = _run(
        [venv_python, "-m", "pip", "install", "-r", REQUIREMENTS, "-i", TSINGHUA_MIRROR],
    )
    if result2.returncode == 0:
        print("    Dependencies OK (via Tsinghua mirror).")
        return

    _fail(
        "Failed to install dependencies (both mirrors failed).",
        "Check your internet connection, then run again.\n"
        "Or manually: .venv\\Scripts\\pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple",
    )


# ---------------------------------------------------------------------------
# Step 4: launch streamlit
# ---------------------------------------------------------------------------

def launch_app(venv_python: str) -> None:
    _step(4, 4, "Starting web app...")

    banner = f"""
    ============================================================
      {APP_NAME}  is running!
      Open  http://{HOST}:{PORT}  in your browser.
      Close THIS window to stop the app.
    ============================================================
"""
    print(banner)

    cmd = [
        venv_python, "-m", "streamlit", "run", APP_ENTRY,
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.port", str(PORT),
        "--server.address", HOST,
    ]

    # Replace current process so Ctrl-C / window-close kills streamlit directly
    if sys.platform == "win32":
        os.execv(venv_python, cmd)
    else:
        os.execv(venv_python, cmd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    _banner(f"{APP_NAME}  ·  First run downloads dependencies (2-5 min)")

    check_python()
    venv_python = setup_venv()
    install_deps(venv_python)
    launch_app(venv_python)


if __name__ == "__main__":
    main()
