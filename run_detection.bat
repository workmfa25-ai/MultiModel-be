@echo off
REM YAMNet Sound Detection - Quick Run Script

echo ============================================================
echo YAMNet Sound Detection System
echo ============================================================
echo.

REM Activate virtual environment
call .\sounds\Scripts\activate.bat

REM Check if TensorFlow is installed
python -c "import tensorflow" 2>nul
if errorlevel 1 (
    echo Installing TensorFlow and TensorFlow Hub...
    python -m pip install tensorflow tensorflow-hub
    echo.
)

REM Run detection
if "%1"=="" (
    echo Usage: run_detection.bat path\to\audio.mp3
    echo.
    echo Example: run_detection.bat ..\audio_files\audio1.mp3
    pause
) else (
    echo Running detection on: %1
    echo.
    python detect_yamnet.py --input %1
    echo.
    echo ============================================================
    echo Detection complete! Check output\reports\ for results
    echo ============================================================
    pause
)
