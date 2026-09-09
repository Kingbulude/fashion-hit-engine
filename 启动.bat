@echo off
REM ============================================================
REM fashion-hit-engine · Windows 一键启动器 v2
REM 修复闪退：goto 分支 + 延迟扩展 + 去掉 --quiet
REM ============================================================
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

title fashion-hit-engine 启动器

REM 切到脚本所在目录（含中文路径也能处理）
cd /d "%~dp0"

echo.
echo ============================================================
echo   fashion-hit-engine 服装爆款预测引擎
echo   首次启动需下载依赖（约2-5分钟），请耐心等待...
echo ============================================================
echo.

REM ---------- 1. 检查 Python ----------
echo [1/4] 检查 Python...
python --version >nul 2>&1
if errorlevel 1 goto :no_python
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo   已安装：!PY_VER!

REM ---------- 2. 建虚拟环境 ----------
echo.
echo [2/4] 准备虚拟环境...
if exist ".venv\Scripts\python.exe" (
    echo   虚拟环境已存在，跳过
    goto :venv_done
)
echo   创建虚拟环境中...
python -m venv .venv
if errorlevel 1 goto :venv_fail
echo   虚拟环境创建完成
:venv_done

REM ---------- 3. 装依赖 ----------
echo.
echo [3/4] 安装依赖（首次较慢，请勿关闭窗口）...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :pip_fail
echo   依赖安装完成

REM ---------- 4. 启动 ----------
echo.
echo [4/4] 启动 Web 应用...
echo.
echo ============================================================
echo   浏览器将打开 http://localhost:8501
echo   关闭此黑窗口即可停止应用
echo ============================================================
echo.

".venv\Scripts\python.exe" -m streamlit run app.py --server.headless=true --browser.gatherUsageStats=false

echo.
echo ============================================================
echo 应用已停止。按任意键关闭窗口。
echo ============================================================
pause >nul
goto :eof

REM ============================================================
REM 错误分支
REM ============================================================
:no_python
echo.
echo [错误] 未检测到 Python
echo.
echo 请先安装 Python 3.10+：
echo   下载地址 https://www.python.org/downloads/
echo.
echo 安装时务必勾选 "Add Python to PATH"（页面底部复选框）
echo.
pause
exit /b 1

:venv_fail
echo.
echo [错误] 虚拟环境创建失败
echo 可能原因：Python 安装时未勾选 "Add Python to PATH"
echo 修复：重装 Python，勾选 Add to PATH
echo.
pause
exit /b 1

:pip_fail
echo.
echo [错误] 依赖安装失败
echo 可能原因：网络问题
echo.
echo 国内用户建议加镜像源：
echo   ".venv\Scripts\pip" install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo.
pause
exit /b 1
