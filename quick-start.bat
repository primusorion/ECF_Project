@echo off
REM Quick Start Script for Predictive Maintenance System (Windows)
REM This script sets up and starts all services

echo ========================================
echo Predictive Maintenance System
echo Quick Start (Windows)
echo ========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo [OK] Docker is running
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Create directories
echo Creating directories...
if not exist "models\saved_models" mkdir "models\saved_models"
if not exist "models\edge_optimized" mkdir "models\edge_optimized"
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "data\synthetic" mkdir "data\synthetic"
if not exist "logs" mkdir "logs"
if not exist "config\grafana\dashboards" mkdir "config\grafana\dashboards"
if not exist "config\grafana\datasources" mkdir "config\grafana\datasources"

echo [OK] Directories created
echo.

REM Install dependencies
set /p install_deps="Install Python dependencies? (y/n): "
if /i "%install_deps%"=="y" (
    echo Installing dependencies...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    echo [OK] Dependencies installed
    echo.
)

REM Generate data and train models
set /p gen_data="Generate sample data and train models? (y/n): "
if /i "%gen_data%"=="y" (
    echo Generating sample data...
    python main.py --mode generate --samples 10000
    
    echo Training models...
    python main.py --mode train
    echo [OK] Models trained
    echo.
)

REM Choose deployment mode
echo Choose deployment mode:
echo 1) Full stack (API + Dashboard + Database + MQTT + Monitoring)
echo 2) Minimal (API + Dashboard only)
echo 3) Inference only (optimized edge deployment)
set /p mode="Select mode (1/2/3): "

if "%mode%"=="1" (
    echo Starting full stack...
    docker-compose -f docker-compose.full.yml up -d
) else if "%mode%"=="2" (
    echo Starting minimal stack...
    docker-compose up -d
) else if "%mode%"=="3" (
    echo Starting inference stack...
    docker-compose -f docker-compose.inference.yml up -d
) else (
    echo Invalid option. Exiting.
    pause
    exit /b 1
)

echo.
echo Waiting for services to start...
timeout /t 15 /nobreak >nul

echo.
echo Checking service health...
echo.

REM Check services
curl -s -o nul -w "[OK] API Server is running - http://localhost:8000/health\n" http://localhost:8000/health || echo [WARNING] API Server not responding yet
curl -s -o nul -w "[OK] Dashboard is running - http://localhost:8050\n" http://localhost:8050 || echo [WARNING] Dashboard not responding yet

if "%mode%"=="1" (
    curl -s -o nul -w "[OK] Grafana is running - http://localhost:3000\n" http://localhost:3000 || echo [WARNING] Grafana not responding yet
    curl -s -o nul -w "[OK] Prometheus is running - http://localhost:9090\n" http://localhost:9090 || echo [WARNING] Prometheus not responding yet
)

echo.
echo ========================================
echo Predictive Maintenance System Started!
echo ========================================
echo.
echo Access Points:
echo   API:            http://localhost:8000
echo   API Docs:       http://localhost:8000/docs
echo   Dashboard:      http://localhost:8050

if "%mode%"=="1" (
    echo   Grafana:        http://localhost:3000 ^(admin/admin^)
    echo   Prometheus:     http://localhost:9090
    echo   MLflow:         http://localhost:5000
)

echo.
echo Commands:
echo   View logs:      docker-compose logs -f
echo   Stop services:  docker-compose down
echo   Restart:        docker-compose restart
echo.
echo Documentation:
echo   Project Overview:  PROJECT_OVERVIEW.md
echo   Docker Guide:      DOCKER_RUN_GUIDE.md
echo   Improvements:      IMPROVEMENTS_GUIDE.md
echo.
echo Happy predicting!
echo.
pause
