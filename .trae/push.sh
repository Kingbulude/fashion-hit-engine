#!/usr/bin/env bash
# ================================================================
# push.sh — 一次性推送脚本（version bump + commit + tag + push main + push tag）
# 用法：
#   bash .trae/push.sh                      # patch 版本自动 bump
#   bash .trae/push.sh "feat: 加了 xxx"    # 指定 commit message
#   bash .trae/push.sh "" v1.4.41.0         # 指定版本号
#
# Token 放在 .trae/token 文件里（chmod 600），不会硬编码进脚本。
# Cloudflare Pages 监听 main 分支和 tag 推送，两条都推 → 稳触发部署。
# ================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

TOKEN_FILE=".trae/token"
TOKEN=""
if [[ -f "$TOKEN_FILE" ]]; then
    TOKEN=$(cat "$TOKEN_FILE" | tr -d '\n')
fi
if [[ -z "$TOKEN" ]]; then
    echo "❌ 未找到 token，请先写 .trae/token 文件："
    echo "   echo 'ghp_xxxx' > .trae/token && chmod 600 .trae/token"
    exit 1
fi

COMMIT_MSG="${1:-auto: $(date -Iseconds)}"
FORCE_VERSION="${2:-}"

echo "═══ Fashion Hit Engine Push ═══"

# ---------- 1. VERSION bump ----------
CURRENT=$(cat VERSION | tr -d '\n')
echo "当前 VERSION: $CURRENT"

if [[ -n "$FORCE_VERSION" ]]; then
    NEW_VERSION="$FORCE_VERSION"
else
    # 自动 bump patch: v1.4.40.1 → v1.4.40.2
    IFS='.' read -r v M m p <<< "${CURRENT#v}"
    NEW_VERSION="v${v}.${M}.${m}.$((p + 1))"
fi
echo "新版本: $NEW_VERSION"
echo "$NEW_VERSION" > VERSION

# ---------- 2. git 检查 ----------
git add -A
if git diff --cached --quiet; then
    echo "⚠️ 没有代码改动，只 bump VERSION 文件"
    git add VERSION
fi

git commit -m "$COMMIT_MSG (VERSION $NEW_VERSION)"

# ---------- 3. tag ----------
git tag -a "$NEW_VERSION" -m "$COMMIT_MSG"
echo "✅ tag $NEW_VERSION 已创建 → $(git rev-parse --short "$NEW_VERSION")"

# ---------- 4. push ----------
REPO_URL="https://Kingbulude:${TOKEN}@github.com/Kingbulude/fashion-hit-engine"
CLEAN_URL="https://github.com/Kingbulude/fashion-hit-engine"

# 恢复清理（即使中间失败也执行）
trap "git remote set-url origin ${CLEAN_URL}" EXIT

git remote set-url origin "$REPO_URL"
echo ""
echo "正在 push main ..."
git push origin main
echo ""
echo "正在 push tag $NEW_VERSION ..."
git push origin "$NEW_VERSION"

# ---------- 5. 结果 ----------
echo ""
echo "═════ 推送成功 ═════"
echo "  main → $(git rev-parse --short HEAD)"
echo "  tag  → $NEW_VERSION"
echo ""
echo "Cloudflare Pages 应该会在 2-3 分钟内自动部署"
echo "部署后访问：https://fashion-hit-engine.pages.dev"
