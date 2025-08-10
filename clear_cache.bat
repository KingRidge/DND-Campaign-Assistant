@echo off
echo Clearing Streamlit cache...

REM Clear Streamlit cache directories
rmdir /s /q "%APPDATA%\Streamlit" 2>nul
rmdir /s /q "%USERPROFILE%\.streamlit" 2>nul

REM Clear Python cache
rmdir /s /q "__pycache__" 2>nul
rmdir /s /q "dnd_campaign_assistant\__pycache__" 2>nul

echo Cache cleared!
pause
