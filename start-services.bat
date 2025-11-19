@echo off
REM Quick start script for Comply-AI services (Windows Batch)
REM Usage: start-services.bat

echo ========================================
echo   Comply-AI Services Startup Script
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

echo Starting services...
echo.

echo Starting Detector Orchestration Service (port 8000)...
start "Detector Orchestration" python start_detector_orchestration.py
timeout /t 2 /nobreak >nul

echo Starting Analysis Service (port 8001)...
start "Analysis Service" python start_analysis_service.py
timeout /t 2 /nobreak >nul

echo Starting Mapper Service (port 8003)...
start "Mapper Service" python start_mapper_service.py
timeout /t 3 /nobreak >nul

echo.
echo Waiting for services to initialize...
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo   Services started!
echo ========================================
echo.
echo Service URLs:
echo   - Detector Orchestration: http://localhost:8000/docs
echo   - Analysis Service:       http://localhost:8001/docs
echo   - Mapper Service:         http://localhost:8003/docs
echo.
echo Health Check URLs:
echo   - http://localhost:8000/health
echo   - http://localhost:8001/health
echo   - http://localhost:8003/health
echo.
echo Press any key to exit...
pause >nul

