#!/usr/bin/env bash
# ============================================================
# fashion-hit-engine · Mac/Linux 一键启动器
# 双击或 ./start.sh 运行：检查 Python → 建虚拟环境 → 装依赖 → 启动 Web UI
# ============================================================
set -e
cd "$(dirname "$0")"

echo
echo "============================================================"
echo "  fashion-hit-engine · 服装爆款预测引擎"
echo "  正在准备运行环境，首次启动需下载依赖（约2分钟）..."
echo "============================================================"
echo

# ---------- 1. 检查 Python ----------
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "[错误] 未检测到 Python，请先安装 Python 3.10+"
    echo "  Mac:   brew install python3"
    echo "  Linux: sudo apt install python3 python3-venv"
    exit 1
fi

# ---------- 2. 建虚拟环境 ----------
if [ ! -f ".venv/bin/python" ]; then
    echo "[1/3] 创建虚拟环境..."
    "$PYTHON" -m venv .venv
else
    echo "[1/3] 虚拟环境已存在，跳过"
fi

# ---------- 3. 装依赖 ----------
echo "[2/3] 检查并安装依赖..."
".venv/bin/python" -m pip install --quiet --upgrade pip
".venv/bin/python" -m pip install --quiet -r requirements.txt

# ---------- 4. 启动 ----------
echo "[3/3] 启动 Web 应用..."
echo
echo "============================================================"
echo "  浏览器将自动打开 http://localhost:8501"
echo "  关闭此窗口或按 Ctrl+C 停止应用"
echo "============================================================"
echo

".venv/bin/python" -m streamlit run app.py --server.headless=true --browser.gatherUsageStats=false
