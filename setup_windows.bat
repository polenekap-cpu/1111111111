@echo off
chcp 65001 > nul
echo ============================================
echo   AI Playlist Generator — Windows Setup
echo ============================================
echo.

:: Check Python
python --version > nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден.
    echo Скачайте Python 3.10 или 3.11 с https://www.python.org/downloads/
    echo При установке ОБЯЗАТЕЛЬНО поставьте галочку "Add Python to PATH"
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER%
echo.

:: Create venv
if not exist ".venv" (
    echo Создаём виртуальное окружение...
    python -m venv .venv
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось создать виртуальное окружение.
        pause
        exit /b 1
    )
    echo [OK] Виртуальное окружение создано.
) else (
    echo [OK] Виртуальное окружение уже существует.
)
echo.

:: Activate venv and install deps
echo Устанавливаем зависимости (может занять 1-3 минуты)...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if errorlevel 1 (
    echo [ОШИБКА] Не удалось установить зависимости.
    pause
    exit /b 1
)
echo.
echo [OK] Все зависимости установлены.
echo.
echo ============================================
echo   Установка завершена!
echo   Теперь запускайте приложение через run.bat
echo ============================================
pause
