@echo off
REM ============================================================
REM fashion-hit-engine · 诊断脚本
REM 双击此文件查看环境信息，方便排查启动问题
REM ============================================================
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title fashion-hit-engine 诊断

cd /d "%~dp0"

echo ============================================================
echo   fashion-hit-engine 环境诊断
echo   把下面信息发给我，我帮你排查问题
echo ============================================================
echo.

echo [1] Python 版本：
where python 2>nul
if errorlevel 1 (
    echo   × 未找到 python 命令
    echo   × Python 未安装或未加入 PATH
) else (
    python --version 2>&1
    echo   √ Python 可用
)
echo.

echo [2] 当前目录：
echo   %CD%
echo.

echo [3] 项目文件检查：
if exist "app.py" (echo   √ app.py 存在) else (echo   × app.py 不存在)
if exist "requirements.txt" (echo   √ requirements.txt 存在) else (echo   × requirements.txt 不存在)
if exist "src\pipeline.py" (echo   √ src\pipeline.py 存在) else (echo   × src\pipeline.py 不存在)
if exist "brand_profiles\tongzhuang-outdoor\profile.yaml" (echo   √ 品牌配置存在) else (echo   × 品牌配置缺失)
echo.

echo [4] 虚拟环境：
if exist ".venv\Scripts\python.exe" (
    echo   √ .venv 已创建
    ".venv\Scripts\python.exe" --version 2>&1
) else (
    echo   × .venv 未创建（首次启动会自动创建）
)
echo.

echo [5] streamlit 是否已装（在 venv 内）：
if exist ".venv\Scripts\streamlit.exe" (
    echo   √ streamlit 已装
    ".venv\Scripts\streamlit.exe" --version 2>&1
) else (
    echo   × streamlit 未装（启动器会自动装，需联网）
)
echo.

echo [6] 网络连通性测试（pypi）：
ping -n 2 pypi.org >nul 2>&1
if errorlevel 1 (
    echo   × 无法连接 pypi.org（网络问题，建议用镜像源）
) else (
    echo   √ 可连接 pypi.org
)
echo.

echo ============================================================
echo 诊断完成。按任意键关闭。
echo 如有问题，截图此窗口发给我。
echo ============================================================
pause >nul
