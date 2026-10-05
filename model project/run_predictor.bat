@echo off
REM Emotion & Stress Predictor Launcher
echo ========================================
echo   Emotion and Stress Level Predictor
echo ========================================
echo.
echo Choose an interface:
echo 1) Desktop GUI (Tkinter)
echo 2) Web Interface (Flask)
echo 3) Exit
echo.

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Starting Desktop GUI...
    ".venv\Scripts\python.exe" emotion_predictor_gui.py
) else if "%choice%"=="2" (
    echo.
    echo Starting Web Interface...
    echo Please install Flask first if not done:
    echo   pip install flask pillow
    echo.
    ".venv\Scripts\python.exe" app.py
) else if "%choice%"=="3" (
    echo Exiting...
) else (
    echo Invalid choice. Exiting...
)

pause
