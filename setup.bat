@echo off
REM Aerial Threat Detection System Setup Script for Windows
REM This script automates the installation and setup process

setlocal enabledelayedexpansion

echo 🚁 Aerial Threat Detection System Setup
echo ========================================

REM Color codes don't work well in batch, so we'll use simple text
set "INFO=[INFO]"
set "WARNING=[WARNING]"
set "ERROR=[ERROR]"
set "STEP=[STEP]"

echo %STEP% Checking Python installation...

REM Check for Python 3
python --version >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo %INFO% Found Python !PYTHON_VERSION!
    set PYTHON_CMD=python
) else (
    python3 --version >nul 2>&1
    if %errorlevel% == 0 (
        for /f "tokens=2" %%i in ('python3 --version 2^>^&1') do set PYTHON_VERSION=%%i
        echo %INFO% Found Python !PYTHON_VERSION!
        set PYTHON_CMD=python3
    ) else (
        echo %ERROR% Python is not installed. Please install Python 3.8 or higher.
        pause
        exit /b 1
    )
)

echo %STEP% Checking Node.js installation...

node --version >nul 2>&1
if %errorlevel% == 0 (
    for /f %%i in ('node --version') do set NODE_VERSION=%%i
    echo %INFO% Found Node.js !NODE_VERSION!
) else (
    echo %ERROR% Node.js is not installed. Please install Node.js 16 or higher.
    pause
    exit /b 1
)

echo %STEP% Checking for trained model file...

if exist "yolo11s.pt" (
    echo %INFO% Found trained model: yolo11s.pt
    for %%A in (yolo11s.pt) do echo %INFO% Model size: %%~zA bytes
) else (
    echo %WARNING% Model file 'yolo11s.pt' not found in project root
    echo %WARNING% Please ensure your trained model is placed as 'yolo11s.pt'
    set /p continue="Continue without model file? (y/n): "
    if /i not "!continue!"=="y" (
        echo %ERROR% Setup cancelled. Please add your model file and try again.
        pause
        exit /b 1
    )
)

echo %STEP% Setting up Python virtual environment...

if not exist "venv" (
    echo %INFO% Creating virtual environment...
    %PYTHON_CMD% -m venv venv
) else (
    echo %INFO% Virtual environment already exists
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo %INFO% Activated virtual environment

echo %INFO% Upgrading pip...
pip install --upgrade pip

echo %STEP% Installing Python dependencies...

cd backend

if exist "requirements.txt" (
    echo %INFO% Installing packages from requirements.txt...
    pip install -r requirements.txt
    echo %INFO% Python dependencies installed successfully
) else (
    echo %ERROR% requirements.txt not found in backend directory
    pause
    exit /b 1
)

cd ..

echo %STEP% Installing Node.js dependencies...

cd frontend

if exist "package.json" (
    echo %INFO% Installing npm packages...
    npm install
    echo %INFO% Node.js dependencies installed successfully
) else (
    echo %ERROR% package.json not found in frontend directory
    pause
    exit /b 1
)

cd ..

echo %STEP% Testing backend setup...

call venv\Scripts\activate.bat

cd backend

echo %INFO% Testing API imports...
%PYTHON_CMD% -c "from api_server import app; print('✅ API server imports successful')"

cd ..

echo %STEP% Testing frontend setup...

cd frontend

echo %INFO% Checking Electron installation...
npx electron --version

cd ..

echo %STEP% Creating startup scripts...

REM Backend startup script
echo @echo off > start_backend.bat
echo echo 🚁 Starting Aerial Threat Detection Backend... >> start_backend.bat
echo call venv\Scripts\activate.bat >> start_backend.bat
echo cd backend >> start_backend.bat
echo python api_server.py >> start_backend.bat
echo pause >> start_backend.bat

REM Frontend startup script
echo @echo off > start_frontend.bat
echo echo 🚁 Starting Aerial Threat Detection Frontend... >> start_frontend.bat
echo cd frontend >> start_frontend.bat
echo npm start >> start_frontend.bat
echo pause >> start_frontend.bat

REM Combined startup script
echo @echo off > start_system.bat
echo echo 🚁 Starting Aerial Threat Detection System... >> start_system.bat
echo echo Starting backend and frontend services... >> start_system.bat
echo start "Backend" cmd /k start_backend.bat >> start_system.bat
echo timeout /t 5 /nobreak ^> nul >> start_system.bat
echo start "Frontend" cmd /k start_frontend.bat >> start_system.bat

echo %INFO% Created startup scripts:
echo %INFO%   - start_backend.bat  (backend only)
echo %INFO%   - start_frontend.bat (frontend only)
echo %INFO%   - start_system.bat   (complete system)

echo.
echo 🎉 Setup completed successfully!
echo ================================
echo.
echo %INFO% Your Aerial Threat Detection system is ready to use!
echo.
echo 📋 Next steps:
echo 1. Ensure your trained model 'yolo11s.pt' is in the project root
echo 2. Start the system using one of these methods:
echo.
echo    Option A - Start complete system:
echo    start_system.bat
echo.
echo    Option B - Start services separately:
echo    Terminal 1: start_backend.bat
echo    Terminal 2: start_frontend.bat
echo.
echo    Option C - Manual startup:
echo    Terminal 1: cd backend ^&^& python api_server.py
echo    Terminal 2: cd frontend ^&^& npm start
echo.
echo 🌐 Access points:
echo • Frontend: Electron application (auto-opens)
echo • Backend API: http://localhost:5000/api/health
echo.
echo 📖 Documentation:
echo • README.md - Complete documentation
echo • Help menu in application (F1)
echo.
echo %INFO% Enjoy your aerial threat detection system! 🚁

pause