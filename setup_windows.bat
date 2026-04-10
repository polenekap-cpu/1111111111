@echo off
echo ============================================
echo   AI Playlist Generator - Windows Setup
echo ============================================
echo.

:: Try to find Python 3.12 or 3.11 via py launcher (preferred)
set PYTHON=
py -3.12 --version > nul 2>&1
if not errorlevel 1 (
    set PYTHON=py -3.12
    goto found_python
)
py -3.11 --version > nul 2>&1
if not errorlevel 1 (
    set PYTHON=py -3.11
    goto found_python
)

:: Fall back to plain "python" but check version
python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Install Python 3.11 or 3.12 from https://www.python.org/downloads/
    echo Check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Check that the version is not 3.13+
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=2 delims=." %%m in ('echo %PYVER%') do set PYMINOR=%%m
if %PYMINOR% GEQ 13 (
    echo [ERROR] Python %PYVER% is not supported by Kivy yet.
    echo.
    echo Kivy requires Python 3.11 or 3.12.
    echo Please install one of these versions from https://www.python.org/downloads/
    echo Then re-run this script.
    echo.
    echo Tip: you can have multiple Python versions installed at the same time.
    pause
    exit /b 1
)
set PYTHON=python

:found_python
for /f "tokens=2" %%v in ('%PYTHON% --version 2^>^&1') do set PYVER=%%v
echo [OK] Using Python %PYVER%
echo.

if not exist ".venv" (
    echo Creating virtual environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)
echo.

echo Installing dependencies (may take 1-3 minutes)...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [OK] All dependencies installed.
echo.
echo ============================================
echo   Setup complete! Run the app via run.bat
echo ============================================
pause
