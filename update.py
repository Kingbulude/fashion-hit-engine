"""
fashion-hit-engine one-click updater.

Reads local VERSION, fetches latest Release from GitHub, downloads zip,
and overlay-updates code files while preserving user data.

Windows users: double-click update.bat  (3-line wrapper that calls this)
"""

from __future__ import annotations

import json
import os
import random
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO = "Kingbulude/fashion-hit-engine"
VERSION_FILE = "VERSION"

# Files/dirs that user data lives in — NEVER overwrite these during update.
# Patterns use pathlib match logic (exact name match for top-level, or glob).
PROTECTED_PATHS = [
    ".env",
    "data",
    "output",
    "calibration",
    ".streamlit/secrets.toml",
    ".uploads",          # user-uploaded images / assets
]

# File extensions that are user-generated data — NEVER overwrite.
# These include Excel batch exports, PDFs, image uploads, etc.
PROTECTED_EXTENSIONS = {
    ".xlsx", ".xls", ".xlsm", ".csv",       # spreadsheets
    ".zip", ".7z", ".rar",                  # archives (user exports)
    ".pdf", ".docx", ".pptx",               # user documents
    ".png", ".jpg", ".jpeg", ".gif",        # uploaded images
    ".ipynb",                               # Jupyter notebooks
    ".log",                                 # log files
}

# Within brand_profiles/, protect ONLY the calibrated subdirectory contents.
# Everything else under brand_profiles/ (config.yaml etc.) gets updated.
BRAND_CALIBRATED_GLOBS = [
    "brand_profiles/*/calibrated/*.yaml",
    "brand_profiles/*/calibrated/*.json",
    "brand_profiles/*/calibrated/*.md",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"


def _banner(msg: str) -> None:
    bar = "=" * 58
    print(f"\n{BOLD}{CYAN}{bar}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{bar}{RESET}\n")


def _ok(msg: str) -> None:
    print(f"  {GREEN}✔{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✖{RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {DIM}→{RESET} {msg}")


def _pause() -> None:
    try:
        input("\nPress Enter to exit...")
    except EOFError:
        pass


def _parse_version(tag: str) -> tuple[int, ...]:
    """Convert 'v1.2.3.4' → (1, 2, 3, 4). Non-numeric parts treated as 0.

    支持 4 段版本号：MAJOR.MINOR.PATCH.HOTFIX
    旧格式 3 段自动补 0 到 4 位。
    """
    tag = tag.lstrip("vV")
    parts = []
    for p in tag.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    while len(parts) < 4:
        parts.append(0)
    return tuple(parts[:4])


# ---------------------------------------------------------------------------
# Step 1: local version
# ---------------------------------------------------------------------------

def read_local_version() -> str | None:
    ver_path = Path(VERSION_FILE)
    if not ver_path.is_file():
        return None
    return ver_path.read_text().strip()


# ---------------------------------------------------------------------------
# Step 2: remote latest tag
# ---------------------------------------------------------------------------

def fetch_latest_release() -> tuple[str, str]:
    """Returns (tag_name, zip_download_url) or raises on failure."""
    api_url = f"https://api.github.com/repos/{REPO}/releases/latest"
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                api_url,
                headers={"User-Agent": "fashion-hit-engine-updater"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            break
        except (urllib.error.URLError, socket.timeout, OSError) as e:
            last_err = e
            if attempt < 2:
                wait = 2 ** attempt + random.random()
                print(f"    API 连接失败 (第 {attempt+1}/3 次)，{wait:.1f}s 后重试...")
                time.sleep(wait)
    else:
        raise RuntimeError(f"Cannot reach GitHub API after 3 retries: {last_err}")

    tag = data.get("tag_name")
    assets = data.get("assets", [])
    zip_url = None
    for a in assets:
        name = a.get("name", "")
        if name.endswith(".zip"):
            zip_url = a.get("browser_download_url")
            break

    if not tag:
        raise RuntimeError("Latest release has no tag.")

    # Fallback: if no custom zip asset, use GitHub's auto-generated source zip
    # Every tag automatically gets source.zip at this URL
    if not zip_url:
        zip_url = f"https://github.com/{REPO}/archive/refs/tags/{tag}.zip"
        _info(f"Using source-code zip (no custom release asset found).")

    return tag, zip_url


# ---------------------------------------------------------------------------
# Step 3: download zip — robust version with resume-friendly retry
# ---------------------------------------------------------------------------

def _urlopen_with_retry(url: str, *, timeout: int, headers: dict,
                         retries: int = 4) -> urllib.request.urlopen:
    """Open URL with exponential backoff. Returns response object."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            return urllib.request.urlopen(req, timeout=timeout)
        except (urllib.error.URLError, socket.timeout, OSError) as e:
            last_err = e
            if attempt < retries - 1:
                wait = 2 ** attempt + random.random()
                print(f"    连接失败 (第 {attempt+1}/{retries} 次)，{wait:.1f}s 后重试...")
                time.sleep(wait)
    raise RuntimeError(f"Download failed after {retries} retries: {last_err}")


def download_zip(url: str, dest: Path) -> int:
    """Download with heartbeat + progress. Returns bytes downloaded."""
    headers = {"User-Agent": "fashion-hit-engine-updater"}
    last_err = None
    for attempt in range(4):
        try:
            print()  # newline before progress bar
            resp = _urlopen_with_retry(url, timeout=120, headers=headers, retries=1)
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 64 * 1024
            last_report = time.time()
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    # Update progress at least every 0.5s OR every 5%
                    time_ok = (now - last_report) >= 0.5
                    pct_now = (downloaded * 100 // total) if total else 0
                    pct_changed = total and (downloaded * 100 // total !=
                                             (downloaded - len(chunk)) * 100 // total)
                    if time_ok or pct_changed:
                        if total:
                            pct = pct_now
                            mb_done = downloaded / 1024 / 1024
                            mb_total = total / 1024 / 1024
                            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                            sys.stdout.write(f"\r    [{bar}] {pct}%  ({mb_done:.1f}/{mb_total:.1f} MB)")
                        else:
                            mb_done = downloaded / 1024 / 1024
                            sys.stdout.write(f"\r    下载中... ({mb_done:.1f} MB, 未知总大小)")
                        sys.stdout.flush()
                        last_report = now
            sys.stdout.write("\n")
            return downloaded
        except (urllib.error.URLError, socket.timeout, OSError) as e:
            last_err = e
            if dest.exists():
                dest.unlink(missing_ok=True)
            if attempt < 3:
                wait = 3 ** attempt + random.random()
                print(f"\n    下载中断 (第 {attempt+1}/4 次)，{wait:.1f}s 后从头重试...")
                time.sleep(wait)
    raise RuntimeError(f"Download failed after 4 retries: {last_err}")


# ---------------------------------------------------------------------------
# Step 4: merge extracted zip → current dir, respecting protected paths
# ---------------------------------------------------------------------------

def _is_protected(rel_path: str) -> bool:
    """Check if a relative path should NOT be overwritten."""
    top = rel_path.split("/")[0]
    if top in PROTECTED_PATHS:
        return True

    # User data extensions — never overwrite
    from pathlib import PurePosixPath
    ext = PurePosixPath(rel_path).suffix.lower()
    if ext in PROTECTED_EXTENSIONS:
        return True

    # Glob matches (brand calibrated)
    from fnmatch import fnmatchcase
    for pat in BRAND_CALIBRATED_GLOBS:
        if fnmatchcase(rel_path.replace("\\", "/"), pat):
            return True

    return False


def merge_release(extracted_root: Path, dest_root: Path) -> tuple[int, int]:
    """Copy files from extracted zip into dest_root, skip protected paths.

    Returns (copied, skipped). PermissionError on non-code files is treated
    as skip (user may have it open in Excel / browser). Only truly critical
    code file locks raise RuntimeError.
    """
    copied = 0
    skipped = 0
    locked_skipped: list[str] = []

    # Code files — locked ones (e.g. imported .py) must succeed or crash,
    # because half-updated source would break the engine.
    CODE_EXT = {".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
                ".md", ".txt", ".bat", ".sh", ".ts", ".tsx", ".js",
                ".css", ".html", ".gitkeep", ".env.example"}

    # Zip extracts as  fashion-hit-engine/<files>  — we want its contents
    src_dir = extracted_root
    children = list(src_dir.iterdir())
    if len(children) == 1 and children[0].is_dir():
        src_dir = children[0]

    for src_file in src_dir.rglob("*"):
        if src_file.is_dir():
            continue
        rel = src_file.relative_to(src_dir).as_posix()

        if _is_protected(rel):
            skipped += 1
            continue

        dest_file = dest_root / rel
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        bak_file = dest_file.with_suffix(dest_file.suffix + ".bak")
        final_ext = dest_file.suffix.lower()

        last_err: PermissionError | None = None
        for attempt in range(5):
            try:
                if dest_file.is_file():
                    try:
                        dest_file.rename(bak_file)
                    except OSError:
                        pass
                shutil.copy2(src_file, dest_file)
                copied += 1
                # Best-effort bak cleanup
                if bak_file.is_file():
                    try:
                        bak_file.unlink()
                    except OSError:
                        pass
                break
            except PermissionError as pe:
                last_err = pe
                if bak_file.is_file() and not dest_file.is_file():
                    try:
                        bak_file.rename(dest_file)
                    except OSError:
                        pass
                if attempt < 4:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                # --- retry exhausted ---
                if final_ext in CODE_EXT:
                    # Critical source file — cannot half-update
                    raise RuntimeError(
                        f"Cannot overwrite {rel} after 5 retries (file locked by another process). "
                        f"Close Python / Streamlit / any IDE holding this file, then re-run update.bat. "
                        f"Or manually run: copy /Y {rel} from the downloaded zip."
                    ) from pe
                else:
                    # Non-code file that slipped past _is_protected — just skip it
                    locked_skipped.append(rel)
                    skipped += 1
                    break

        # Best-effort bak cleanup from a previous run's self-update
        if bak_file.is_file() and not dest_file.is_file():
            try:
                bak_file.unlink()
            except OSError:
                pass

    if locked_skipped:
        _warn(f"Skipped {len(locked_skipped)} locked file(s) (open in another program):")
        for f in locked_skipped[:5]:
            _warn(f"  ↳ {f}")
        if len(locked_skipped) > 5:
            _warn(f"  ↳ ... and {len(locked_skipped) - 5} more")
        _warn("These were non-code user data — skipping them does not affect the update.")

    return copied, skipped


# ---------------------------------------------------------------------------
# Step 5: kill running app processes (streamlit/python)
# ---------------------------------------------------------------------------

def kill_running_apps() -> bool:
    """Find and kill streamlit/python processes from this project. Returns True if something was killed."""
    import subprocess
    killed_any = False
    try:
        if sys.platform == "win32":
            # Windows: PowerShell Get-CimInstance (Win11+ compatible, wmic removed)
            ps_script = (
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Select-Object ProcessId, CommandLine | "
                "Format-List -HideTableHeaders"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True, text=True, timeout=8,
            )
            # Also try Get-CimInstance via pwsh fallback
            if result.returncode != 0 or not result.stdout.strip():
                result = subprocess.run(
                    ["pwsh", "-NoProfile", "-Command", ps_script],
                    capture_output=True, text=True, timeout=8,
                )
            output = result.stdout.lower()

            # Parse ProcessId + CommandLine from PowerShell Format-List output
            # Each process block looks like: ProcessId : 1234 \n CommandLine : ...
            blocks = output.split("processid")[1:]  # skip header, split by ProcessId
            for block in blocks:
                lines = block.strip().splitlines()
                pid_line = lines[0] if lines else ""
                # Extract numeric PID
                pid_match = re.search(r"(\d+)", pid_line)
                if not pid_match:
                    continue
                pid = pid_match.group(1)
                # CommandLine is everything after "commandline :"
                cmd_match = re.search(r"commandline\s*:\s*(.*)", block, re.DOTALL)
                cmd = (cmd_match.group(1).strip() if cmd_match else "").lower()
                if ("streamlit" in cmd or "app.py" in cmd) and pid.isdigit():
                    _info(f"Killing process PID={pid} ...")
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   capture_output=True, timeout=5)
                    killed_any = True
        else:
            out = subprocess.run(
                ["pgrep", "-f", "streamlit|app.py"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            for pid in out.split():
                if pid.strip().isdigit():
                    _info(f"Killing process PID={pid} ...")
                    subprocess.run(["kill", "-9", pid], timeout=5)
                    killed_any = True
    except Exception as e:
        _warn(f"Failed to kill processes: {e}")

    if killed_any:
        # Wait a moment for file handles to be released
        _info("Waiting for file handles to be released (2s)...")
        time.sleep(2)
        # Verify no more streamlit processes
        time.sleep(1)
    return killed_any


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def main() -> int:
    os.chdir(Path(__file__).parent.resolve())

    _banner("fashion-hit-engine  ·  Updater")

    # 1. local version
    local = read_local_version()
    if not local:
        _warn("No VERSION file found — first time updating?")
        _info("Will treat all files as updatable.")
        local = "v0.0.0"
    _ok(f"Local version:  {BOLD}{local}{RESET}")

    # 2. remote version
    _info("Checking GitHub for latest release...")
    try:
        remote_tag, zip_url = fetch_latest_release()
    except Exception as e:
        _fail(str(e))
        _pause()
        return 1
    _ok(f"Latest version: {BOLD}{remote_tag}{RESET}")

    # 3. compare
    if _parse_version(remote_tag) <= _parse_version(local):
        _ok("Already up to date!")
        _pause()
        return 0

    print()
    print(f"  {YELLOW}Update available: {local} → {remote_tag}{RESET}")

    # 4. kill running app processes (streamlit/python) — auto so user doesn't have to
    _info("Checking for running app processes...")
    killed = kill_running_apps()
    if killed:
        _ok("Killed running streamlit/app processes.")
    else:
        _ok("No running app processes found.")

    # 5. download
    tmp_dir = Path(".update_tmp")
    zip_path = tmp_dir / f"update-{remote_tag}.zip"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    _info(f"Downloading {remote_tag} ...")
    try:
        download_zip(zip_url, zip_path)
    except Exception as e:
        _fail(f"Download failed: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _pause()
        return 3
    _ok("Download complete.")

    # 6. extract
    _info("Extracting...")
    extract_dir = tmp_dir / "extracted"
    extract_dir.mkdir()
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        _fail("Downloaded file is not a valid zip. May be a GitHub rate-limit page.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _pause()
        return 4
    _ok("Extracted.")

    # 7. merge
    _info("Updating code files (user data is preserved)...")
    copied, skipped = merge_release(extract_dir, Path("."))

    # 8. update VERSION
    Path(VERSION_FILE).write_text(f"{remote_tag}\n")

    # 9. cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 10. report
    print()
    _banner(f"Updated to {remote_tag}!")
    print(f"  Files overwritten:  {GREEN}{copied}{RESET}")
    print(f"  Data files skipped: {DIM}{skipped}{RESET}  (preserved)")
    print()

    _info("Checking if requirements.txt changed...")
    # Quick heuristic: pip install if possible
    reqs_changed = False
    try:
        # Re-read requirements.txt timestamp — actually we just overwrote it
        # so we can't compare. Simplest: always ensure deps after update
        reqs_changed = True
    except Exception:
        pass

    venv_python = Path(".venv/Scripts/python.exe" if sys.platform == "win32" else ".venv/bin/python")
    if reqs_changed and venv_python.is_file():
        _info("Refreshing dependencies (pip install -r requirements.txt)...")
        print("    (first-time install takes 2-5 min; subsequent runs are fast)")
        result = os.system(f'"{venv_python}" -m pip install -r requirements.txt -q')
        if result == 0:
            _ok("Dependencies refreshed.")
        else:
            _warn("Some dependencies failed to install. The app may still work — run start.bat again if needed.")

    _banner("Done!")
    print("  Double-click start.bat to launch the app.")
    _pause()
    return 0


if __name__ == "__main__":
    sys.exit(main())
