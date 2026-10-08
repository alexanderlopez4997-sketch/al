@echo off
rem Start the Meridian web dashboard (Windows). First run creates .venv and installs
rem requirements.txt; later runs only reinstall when requirements.txt changes.
cd /d "%~dp0"

where python >nul 2>nul || (echo python not found - install Python 3.9+ first. & pause & exit /b 1)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtualenv in .venv ...
    python -m venv .venv || (pause & exit /b 1)
)
fc /b requirements.txt .venv\.requirements.installed >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies ...
    .venv\Scripts\python -m pip install --quiet --upgrade pip
    .venv\Scripts\python -m pip install --quiet -r requirements.txt || (pause & exit /b 1)
    copy /y requirements.txt .venv\.requirements.installed >nul
)

rem Keep one login across launches: web_server.py writes a generated password to .env once.
set MERIDIAN_SAVE_PASSWORD=1
.venv\Scripts\python web_server.py %*
pause
