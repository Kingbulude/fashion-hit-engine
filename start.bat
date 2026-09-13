@echo off
cd /d "%~dp0"

echo ============================================
echo   fashion-hit-engine starting...
echo ============================================
echo.

rem Try python first, then py launcher
python launch.py
if not errorlevel 9009 goto :done

echo.
echo Python not found as "python", trying "py"...
py launch.py
if not errorlevel 9009 goto :done

echo.
echo ============================================
echo   [ERROR] Python is NOT installed or NOT in PATH
echo ============================================
echo.
echo Please install Python 3.10+ from:
echo   https://www.python.org/downloads/
echo.
echo IMPORTANT: Check "Add Python to PATH" during install.
echo.

:done
echo.
echo ----------------------------------------
echo Press any key to close this window...
pause >nul
