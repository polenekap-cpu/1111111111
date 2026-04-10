@echo off
chcp 65001 > nul

if not exist ".venv\Scripts\activate.bat" (
    echo [ОШИБКА] Виртуальное окружение не найдено.
    echo Сначала запустите setup_windows.bat
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python main.py
if errorlevel 1 (
    echo.
    echo Приложение завершилось с ошибкой.
    pause
)
