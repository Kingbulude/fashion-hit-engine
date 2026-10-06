#!/usr/bin/env python3
"""一键发版：commit + push main + tag + GitHub Release。

用法:
    python release.py patch              # v1.4.74 → v1.4.75 (默认)
    python release.py minor              # v1.4.74 → v1.5.0
    python release.py major              # v1.4.74 → v2.0.0
    python release.py patch "fix: 修 bug"  # 自定义 commit 信息

前置条件:
    - 必须在 main 分支（否则报错退出，绝不偷偷推 agent 分支）
    - 工作区可以有改动，但必须先 git add
    - 环境变量 GH_TOKEN 或 .env 里有 GITHUB_TOKEN（用于创建 Release）
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO = "Kingbulude/fashion-hit-engine"
VERSION_FILE = "VERSION"
REMOTE = "origin"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run(args: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command. Returns CompletedProcess."""
    return subprocess.run(args, check=check, capture_output=capture, text=True)

def _banner(msg: str) -> None:
    bar = "=" * 56
    print(f"\n{bar}\n  {msg}\n{bar}\n")

def _ok(msg: str) -> None:
    print(f"  ✅  {msg}")

def _fail(msg: str) -> None:
    print(f"  ❌  {msg}")
    sys.exit(1)

def _warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")

def _info(msg: str) -> None:
    print(f"  →  {msg}")

# ---------------------------------------------------------------------------
# Step 1: 强制在 main 分支
# ---------------------------------------------------------------------------
def enforce_main_branch() -> None:
    """硬门控：不在 main 就报错退出，绝不误推 agent 分支。"""
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture=True).stdout.strip()
    if branch != "main":
        _fail(
            f"当前在 '{branch}' 分支，发版必须在 main 分支。\n"
            f"     git checkout main && git pull origin main\n"
            f"     然后再 python release.py"
        )
    _ok(f"当前在 main 分支（硬门控已通过）")

# ---------------------------------------------------------------------------
# Step 2: 读当前版本 → bump
# ---------------------------------------------------------------------------
def read_current_version() -> str:
    ver_file = Path(VERSION_FILE)
    if not ver_file.is_file():
        _fail(f"VERSION 文件不存在：{ver_file.resolve()}")
    return ver_file.read_text().strip()

def bump_version(current: str, level: str) -> str:
    """v1.4.74 + patch → v1.4.75；minor → v1.5.0；major → v2.0.0"""
    m = re.match(r"^v?(\d+)\.(\d+)\.(\d+)", current)
    if not m:
        _fail(f"VERSION 文件格式错误：'{current}'（期望 vX.Y.Z）")
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if level == "major":
        major += 1; minor = 0; patch = 0
    elif level == "minor":
        minor += 1; patch = 0
    elif level == "patch":
        patch += 1
    else:
        _fail(f"level 必须是 patch / minor / major，收到 '{level}'")
    return f"v{major}.{minor}.{patch}"

# ---------------------------------------------------------------------------
# Step 3: 确认远端 main 是最新的
# ---------------------------------------------------------------------------
def ensure_remote_synced() -> None:
    """fetch + 对比 HEAD 和 origin/main，落后就退出。"""
    _run(["git", "fetch", REMOTE], capture=True)
    local_head = _run(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    remote_head = _run(["git", "rev-parse", f"{REMOTE}/main"], capture=True).stdout.strip()
    if local_head != remote_head:
        _warn("本地 main 和 origin/main 不一致（可能是之前 agent 分支 cherry-pick 后的状态）")
        _info("自动执行 git pull --rebase origin main 同步...")
        r = _run(["git", "pull", "--rebase", REMOTE, "main"], check=False)
        if r.returncode != 0:
            _fail("pull --rebase 有冲突，手动解决后再发版")
    _ok("本地 main 与 origin/main 已同步")

# ---------------------------------------------------------------------------
# Step 4: commit + push
# ---------------------------------------------------------------------------
def commit_and_push(new_version: str, commit_msg: str, token: str) -> None:
    _run(["git", "add", "-A"])
    # 有 VERSION 文件改动 → commit
    r = _run(["git", "status", "--porcelain"], capture=True)
    if not r.stdout.strip():
        _warn("没有任何文件改动（包括 VERSION），跳过 commit")
    else:
        _info(f"git commit: {commit_msg}")
        _run(["git", "commit", "-m", commit_msg])

    _info(f"git push {REMOTE} main")
    auth_url = f"https://Kingbulude:{token}@github.com/{REPO}"
    old_url = _run(["git", "remote", "get-url", REMOTE], capture=True).stdout.strip()
    try:
        _run(["git", "remote", "set-url", REMOTE, auth_url])
        _run(["git", "push", REMOTE, "main"])
    finally:
        _run(["git", "remote", "set-url", REMOTE, old_url])
    _ok("main 已推送")

# ---------------------------------------------------------------------------
# Step 5: tag
# ---------------------------------------------------------------------------
def create_and_push_tag(new_version: str, token: str) -> None:
    # 如果本地已存在同版本 tag，删除（用户可能重复跑）
    r = _run(["git", "tag", "-l", new_version], capture=True)
    if r.stdout.strip():
        _info(f"本地已存在 {new_version}，删除重建")
        _run(["git", "tag", "-d", new_version])

    # 如果远端已存在，删除重建（之前 agent 分支的 tag 指向错 commit）
    auth_url = f"https://Kingbulude:{token}@github.com/{REPO}"
    old_url = _run(["git", "remote", "get-url", REMOTE], capture=True).stdout.strip()
    try:
        _run(["git", "remote", "set-url", REMOTE, auth_url])
        _run(["git", "push", REMOTE, f":refs/tags/{new_version}"], check=False)  # 删远端旧 tag（可能不存在）

        tag_msg = ""
        r2 = _run(["git", "log", "--oneline", "-1"], capture=True)
        tag_msg = r2.stdout.strip()[:100]
        _run(["git", "tag", "-a", new_version, "-m", tag_msg])
        _run(["git", "push", REMOTE, new_version])
    finally:
        _run(["git", "remote", "set-url", REMOTE, old_url])
    _ok(f"tag {new_version} 已推送")

# ---------------------------------------------------------------------------
# Step 6: GitHub Release
# ---------------------------------------------------------------------------
def create_github_release(new_version: str, token: str) -> None:
    import urllib.request, json
    commit_msg = _run(["git", "log", "--oneline", "-1"], capture=True).stdout.strip()
    body = f"**Release:** {commit_msg}\n\nAuto-generated by release.py"

    data = json.dumps({
        "tag_name": new_version,
        "name": new_version,
        "body": body,
        "draft": False,
        "prerelease": False,
    }).encode()

    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases",
        data=data,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "fashion-hit-engine-release",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read())
    _ok(f"GitHub Release: {result.get('html_url', '?')}")

# ---------------------------------------------------------------------------
# Step 7: 验证
# ---------------------------------------------------------------------------
def verify_release(new_version: str, token: str) -> None:
    import urllib.request, json
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases/latest",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        latest = json.loads(resp.read())
    latest_tag = latest.get("tag_name", "?")
    if latest_tag == new_version:
        _ok(f"✅ releases/latest = {latest_tag}  (匹配新版本)")
    else:
        _fail(f"releases/latest = {latest_tag}，不是刚发的 {new_version}！")

# ---------------------------------------------------------------------------
# Step 8: 更新 VERSION 文件 + 最终状态
# ---------------------------------------------------------------------------
def update_version_file(new_version: str) -> None:
    Path(VERSION_FILE).write_text(f"{new_version}\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = sys.argv[1:]
    level = args[0] if args else "patch"
    custom_msg = args[1] if len(args) > 1 else None

    _banner(f"发版流程 · {REPO}")

    # 0. token
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        # 尝试从 .env 读
        env_path = Path(".env")
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("GH_TOKEN=") or line.startswith("GITHUB_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not token:
        _fail("需要 GH_TOKEN 环境变量或 .env 里有 GH_TOKEN")

    # 1. 强制 main
    enforce_main_branch()

    # 2. bump version
    current = read_current_version()
    new_version = bump_version(current, level)
    _ok(f"VERSION {current} → {new_version}  ({level})")
    update_version_file(new_version)

    # 3. 同步
    ensure_remote_synced()

    # 4. commit + push
    commit_msg = custom_msg or f"chore({new_version}): release {new_version}"
    commit_and_push(new_version, commit_msg, token)

    # 5. tag
    create_and_push_tag(new_version, token)

    # 6. Release
    create_github_release(new_version, token)

    # 7. 验证 updater 能看到
    verify_release(new_version, token)

    _banner(f"✅  {new_version} 发版完成！")
    print("  本地用户运行 update.bat 就能收到更新通知。\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
