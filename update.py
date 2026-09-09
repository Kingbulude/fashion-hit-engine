"""
fashion-hit-engine one-click updater.

Reads local VERSION, fetches latest Release from GitHub, downloads zip,
and overlay-updates code files while preserving user data.

Windows users: double-click update.bat  (3-line wrapper that calls this)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
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
]

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
    """Convert 'v1.2.3' → (1, 2, 3). Non-numeric parts treated as 0."""
    tag = tag.lstrip("vV")
    parts = []
    for p in tag.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


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
    req = urllib.request.Request(api_url, headers={"User-Agent": "fashion-hit-engine-updater"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach GitHub: {e.reason}")

    tag = data.get("tag_name")
    assets = data.get("assets", [])
    zip_url = None
    for a in assets:
        name = a.get("name", "")
        if name.endswith(".zip"):
            zip_url = a.get("browser_download_url")
            break

    if not tag or not zip_url:
        raise RuntimeError(f"Latest release has no tag or no zip asset.")

    return tag, zip_url


# ---------------------------------------------------------------------------
# Step 3: download zip
# ---------------------------------------------------------------------------

def download_zip(url: str, dest: Path) -> int:
    """Download with a simple progress indicator. Returns bytes downloaded."""
    req = urllib.request.Request(url, headers={"User-Agent": "fashion-hit-engine-updater"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 64 * 1024
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    mb_done = downloaded / 1024 / 1024
                    mb_total = total / 1024 / 1024
                    sys.stdout.write(f"\r    {pct}%  ({mb_done:.1f}/{mb_total:.1f} MB)")
                    sys.stdout.flush()
        sys.stdout.write("\n")
        return downloaded


# ---------------------------------------------------------------------------
# Step 4: merge extracted zip → current dir, respecting protected paths
# ---------------------------------------------------------------------------

def _is_protected(rel_path: str) -> bool:
    """Check if a relative path should NOT be overwritten."""
    # Top-level exact matches
    top = rel_path.split("/")[0]
    if top in PROTECTED_PATHS:
        return True

    # Glob matches
    from fnmatch import fnmatchcase
    for pat in BRAND_CALIBRATED_GLOBS:
        # fnmatch wants forward slashes on all platforms
        if fnmatchcase(rel_path.replace("\\", "/"), pat):
            return True

    return False


def merge_release(extracted_root: Path, dest_root: Path) -> int:
    """Copy files from extracted zip into dest_root, skip protected paths."""
    copied = 0
    skipped = 0

    # Zip extracts as  fashion-hit-engine/<files>  — we want its contents
    src_dir = extracted_root
    children = list(src_dir.iterdir())
    # If the zip has one top-level folder, drill into it
    if len(children) == 1 and children[0].is_dir():
        src_dir = children[0]

    for src_file in src_dir.rglob("*"):
        if src_file.is_dir():
            continue  # we only copy files; parent dirs auto-created
        rel = src_file.relative_to(src_dir).as_posix()

        if _is_protected(rel):
            skipped += 1
            continue

        dest_file = dest_root / rel
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)
        copied += 1

        # also write any missing parent .gitkeep files for calibrated dirs

    return copied, skipped


# ---------------------------------------------------------------------------
# Step 5: check if streamlit is running
# ---------------------------------------------------------------------------

def find_running_streamlit() -> str | None:
    """Return process id or None."""
    import subprocess
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["wmic", "process", "where", "name='python.exe'", "get", "processid,commandline"],
                capture_output=True, text=True, timeout=5,
            ).stdout
        else:
            out = subprocess.run(
                ["pgrep", "-af", "streamlit"],
                capture_output=True, text=True, timeout=5,
            ).stdout
        for line in out.splitlines():
            if "streamlit" in line.lower() or "app.py" in line.lower():
                return line.strip()
    except Exception:
        pass
    return None


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

    # 4. check if streamlit running
    streamlit_proc = find_running_streamlit()
    if streamlit_proc:
        print()
        _warn("Streamlit appears to be running.")
        _warn("Please close the app window first, then re-run update.bat.")
        _pause()
        return 2

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
