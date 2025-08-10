@echo off
setlocal enabledelayedexpansion

REM Get the directory where the batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Set UTF-8 encoding
chcp 65001 > nul

echo Starting D^&D Campaign Assistant...
echo Working Directory: %SCRIPT_DIR%
echo.

REM Check for virtual environment
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo Error: Virtual environment activation script not found
    echo Path: "%SCRIPT_DIR%.venv\Scripts\activate.bat"
    goto :error
)

REM Check for required packages
echo Checking required packages...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing Streamlit and other requirements...
    pip install -r requirements.txt
)

REM Install package in development mode
echo Installing package in development mode...
pip install -e .

REM Clear Streamlit cache and storage
echo Clearing Streamlit cache and storage...
python -m streamlit cache clear
rmdir /s /q "%APPDATA%\Streamlit" 2>nul
rmdir /s /q "%USERPROFILE%\.streamlit" 2>nul
mkdir "%APPDATA%\Streamlit"
mkdir "%USERPROFILE%\.streamlit"

REM Kill any existing Streamlit processes
taskkill /F /IM "streamlit.exe" 2>nul
taskkill /F /IM "python.exe" /FI "WINDOWTITLE eq streamlit" 2>nul

REM Run the application
echo Starting application...
python -m streamlit run dnd_campaign_assistant\app.py

goto :end

:error
echo.
echo An error occurred while starting the application
pause
exit /b 1

:end
endlocal
pause
