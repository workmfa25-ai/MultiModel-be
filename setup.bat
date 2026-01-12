@echo off
REM Sound Detection Setup Script for Windows
REM Run this script to set up the entire system

echo ============================================================
echo Sound Event Detection System - Setup
echo ============================================================
echo.

REM Check if virtual environment exists
if exist "sounds\Scripts\activate.bat" (
    echo [OK] Virtual environment already exists
) else (
    echo [1/4] Creating virtual environment...
    python -m venv sounds
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

echo.
echo [2/4] Activating virtual environment...
call sounds\Scripts\activate.bat

echo.
echo [3/4] Installing dependencies...
echo This may take a few minutes...
pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo [OK] Dependencies installed

echo.
echo [4/4] Downloading BEATs model...
echo This will download ~500MB, please be patient...
python download_model.py

if errorlevel 1 (
    echo [WARNING] Model download may have failed
    echo You can try downloading manually later
)

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Activate environment: sounds\Scripts\activate
echo   2. Test detection: python test_detection.py
echo   3. Run CLI: python detect_sounds.py --input ../audio_files/audio1.mp3
echo   4. Start API: python app.py
echo.
echo Press any key to exit...
pause >nul
