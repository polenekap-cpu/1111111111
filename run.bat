@echo off
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup_windows.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python main.py
if errorlevel 1 (
    echo.
    echo App exited with an error.
    pause
)
