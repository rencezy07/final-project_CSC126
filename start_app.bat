@echo off
echo Starting Aerial Threat Detection System...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or later
    pause
    exit /b 1
)

REM Check if Node.js is available
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js
    pause
    exit /b 1
)

REM Install Python dependencies if needed
echo Checking Python dependencies...
pip install flask flask-socketio opencv-python ultralytics torch torchvision --quiet

REM Install Node.js dependencies if needed
echo Checking Node.js dependencies...
cd electron-app
npm install --silent
cd ..

REM Start the Python detection server in background
echo Starting detection server...
start /B python src/detection_server.py

REM Wait a moment for server to start
timeout /t 3 /nobreak >nul

REM Start the Electron app
echo Starting Electron app...
cd electron-app
npm start

REM Clean up - stop the Python server when Electron app closes
echo Cleaning up...
taskkill /f /im python.exe >nul 2>&1