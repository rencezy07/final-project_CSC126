@echo off
echo ====================================
echo Aerial Threat Detection System
echo ====================================
echo.

echo Starting Flask Detection Server...
start "Detection Server" cmd /k "cd /d %~dp0 && python src/detection_server.py --debug --host localhost --port 5000"
timeout /t 3 /nobreak > nul

echo Starting Electron GUI Application...
start "Electron App" cmd /k "cd /d %~dp0/electron-app && npm start"

echo.
echo ====================================
echo Both services are starting...
echo - Flask Server: http://localhost:5000
echo - Electron App: Will open automatically
echo ====================================
echo.
echo Press any key to close this launcher...
pause > nul