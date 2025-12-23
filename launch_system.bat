@echo off
title Aerial Threat Detection System - Auto Launcher
color 0A

echo.
echo ================================================
echo     AERIAL THREAT DETECTION SYSTEM 
echo         Auto Launcher v1.0
echo ================================================
echo.

REM Navigate to the project directory
cd /d "c:\Users\lilir\OneDrive\Desktop\UPDATED-cscFinal"

echo [1/3] Checking Python dependencies...
cd backend
pip install -q -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install Python dependencies
    pause
    exit /b 1
)
echo ✅ Python dependencies verified

echo.
echo [2/3] Starting Backend API Server...
start "API Server" /min cmd /k "python api_server.py"

REM Wait for backend to start
timeout /t 5 /nobreak >nul

echo.
echo [3/3] Launching Desktop Application...
cd ..\frontend

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing Electron dependencies...
    npm install
)

REM Start Electron app
start "Aerial Threat Detection" cmd /c "npm start"

echo.
echo ================================================
echo ✅ System launched successfully!
echo.
echo Backend API: http://127.0.0.1:5000
echo Desktop App: Should open automatically
echo.
echo Press any key to exit this launcher...
echo ================================================
pause >nul