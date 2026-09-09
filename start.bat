@echo off
REM ============================================================
REM fashion-hit-engine Windows Launcher (ENGLISH, ASCII-only)
REM This file has NO Chinese characters, so encoding issues are impossible.
REM If the Chinese launcher fails, try this one.
REM ============================================================
setlocal enabledelayedexpansion

title fashion-hit-engine

cd /d "%~dp0"

echo.
echo ============================================================
echo   fashion-hit-engine
echo   First run needs to download dependencies (2-5 min).
echo ============================================================
echo.

REM ---------- 1. Check Python ----------
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 goto :no_python
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo   Found: !PY_VER!

REM ---------- 2. Create venv ----------
echo.
echo [2/4] Setting up virtual environment...
if exist ".venv\Scripts\python.exe" (
    echo   venv already exists, skipping.
    goto :venv_done
)
echo   Creating venv...
python -m venv .venv
if errorlevel 1 goto :venv_fail
echo   venv created.
:venv_done

REM ---------- 3. Install dependencies ----------
echo.
echo [3/4] Installing dependencies (please wait)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :pip_fail
echo   Dependencies installed.

REM ---------- 4. Launch ----------
echo.
echo [4/4] Starting web app...
echo.
echo ============================================================
echo   Browser will open http://localhost:8501
echo   Close this window to stop the app.
echo ============================================================
echo.

".venv\Scripts\python.exe" -m streamlit run app.py --server.headless=true --browser.gatherUsageStats=false

echo.
echo ============================================================
echo App stopped. Press any key to close.
echo ============================================================
pause >nul
goto :eof

REM ============================================================
REM Error handlers
REM ============================================================
:no_python
echo.
echo [ERROR] Python not found.
echo.
echo Please install Python 3.10+ from https://www.python.org/downloads/
echo IMPORTANT: Check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:venv_fail
echo.
echo [ERROR] Failed to create virtual environment.
echo Likely cause: Python was installed WITHOUT "Add to PATH".
echo Fix: Reinstall Python and check "Add Python to PATH".
echo.
pause
exit /b 1

:pip_fail
echo.
echo [ERROR] Failed to install dependencies.
echo Likely cause: network issue.
echo.
echo In China, try Tsinghua mirror:
echo   ".venv\Scripts\pip" install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo.
pause
exit /b 1
