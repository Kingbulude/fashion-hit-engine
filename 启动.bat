@echo off
chcp 65001 >nul
REM ============================================================
REM fashion-hit-engine · Windows 一键启动器
REM 双击此文件即可：检查 Python → 建虚拟环境 → 装依赖 → 启动 Web UI
REM ============================================================

setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   fashion-hit-engine · 服装爆款预测引擎
echo   正在准备运行环境，首次启动需下载依赖（约2分钟）...
echo ============================================================
echo.

REM ---------- 1. 检查 Python ----------
where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+：
    echo   下载地址：https://www.python.org/downloads/
    echo   安装时请勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

REM ---------- 2. 建虚拟环境（首次） ----------
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 创建虚拟环境...
    python -m venv .venv
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
) else (
    echo [1/3] 虚拟环境已存在，跳过
)

REM ---------- 3. 装依赖（首次或 requirements 变更） ----------
echo [2/3] 检查并安装依赖...
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip >nul 2>&1
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo [错误] 依赖安装失败，请检查网络或手动运行：
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM ---------- 4. 启动 Streamlit ----------
echo [3/3] 启动 Web 应用...
echo.
echo ============================================================
echo   浏览器将自动打开 http://localhost:8501
echo   如未自动打开，请手动访问该地址
echo   关闭此窗口即可停止应用
echo ============================================================
echo.

".venv\Scripts\python.exe" -m streamlit run app.py --server.headless=true --browser.gatherUsageStats=false

REM 异常退出时暂停，方便看错误
if errorlevel 1 (
    echo.
    echo [应用已停止] 如有错误请查看上方日志
    pause
)

endlocal
