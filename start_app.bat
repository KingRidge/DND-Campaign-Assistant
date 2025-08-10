@echo off
chcp 65001 > nul
echo Starting D^&D Campaign Assistant...
echo.
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
    if exist ".venv\Scripts\streamlit.exe" (
        ".venv\Scripts\streamlit.exe" run app.py
    ) else (
        echo Error: streamlit.exe not found in virtual environment
        echo Installing streamlit...
        .venv\Scripts\pip.exe install streamlit
        if exist ".venv\Scripts\streamlit.exe" (
            ".venv\Scripts\streamlit.exe" run app.py
        ) else (
            echo Failed to install streamlit
        )
    )
) else (
    echo Virtual environment not found
    echo Please ensure the virtual environment is set up correctly
)
pause
