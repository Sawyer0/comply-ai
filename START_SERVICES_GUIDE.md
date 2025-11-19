# How to Start Comply-AI Services

This guide shows you all the ways to start the three microservices for future reference.

## Prerequisites

1. **Python 3.11+** installed
2. **Required packages** installed:
   ```powershell
   pip install fastapi uvicorn httpx pydantic
   ```

## Method 1: Start All Services at Once (Recommended)

The easiest way to start all three services:

```powershell
# From the project root directory
python start_all_services.py
```

This will:
- Start all 3 services in separate processes
- Display service URLs and health check endpoints
- Keep running until you press `Ctrl+C` to stop all services

**Service Ports:**
- Detector Orchestration: http://localhost:8000
- Analysis Service: http://localhost:8001
- Mapper Service: http://localhost:8002

## Method 2: Start Services Individually

If you want to start services one at a time or in separate terminal windows:

### Option A: Using Python Scripts

```powershell
# Terminal 1 - Detector Orchestration Service
python start_detector_orchestration.py

# Terminal 2 - Analysis Service
python start_analysis_service.py

# Terminal 3 - Mapper Service
python start_mapper_service.py
```

### Option B: Using PowerShell Background Jobs

```powershell
# Start all services in background
Start-Process python -ArgumentList "start_detector_orchestration.py" -WindowStyle Hidden
Start-Process python -ArgumentList "start_analysis_service.py" -WindowStyle Hidden
Start-Process python -ArgumentList "start_mapper_service.py" -WindowStyle Hidden
```

## Method 3: Using Docker Compose

If you have Docker installed, you can use Docker Compose:

```powershell
# Start all services with Docker
docker-compose -f docker-compose.microservices.yml up --build

# Or start in detached mode (background)
docker-compose -f docker-compose.microservices.yml up -d --build
```

## Method 4: Manual Uvicorn Commands

For development with auto-reload:

```powershell
# Terminal 1 - Detector Orchestration
uvicorn start_detector_orchestration:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Analysis Service
uvicorn start_analysis_service:app --host 0.0.0.0 --port 8001 --reload

# Terminal 3 - Mapper Service
uvicorn start_mapper_service:app --host 0.0.0.0 --port 8002 --reload
```

## Verify Services Are Running

After starting, verify all services are healthy:

```powershell
# Check all health endpoints
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

Or open in your browser:
- Detector Orchestration: http://localhost:8000/docs
- Analysis Service: http://localhost:8001/docs
- Mapper Service: http://localhost:8002/docs

## Service Endpoints Reference

### Detector Orchestration Service (Port 8000)
- **API Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health
- **Main Endpoint**: POST http://localhost:8000/api/v1/orchestrate

### Analysis Service (Port 8001)
- **API Docs**: http://localhost:8001/docs
- **Health**: http://localhost:8001/health
- **Main Endpoint**: POST http://localhost:8001/api/v1/analyze

### Mapper Service (Port 8002)
- **API Docs**: http://localhost:8002/docs
- **Health**: http://localhost:8002/health
- **Main Endpoint**: POST http://localhost:8002/api/v1/map
- **Taxonomy**: GET http://localhost:8002/api/v1/taxonomy

## Stopping Services

### If using `start_all_services.py`:
- Press `Ctrl+C` in the terminal where it's running

### If using individual processes:
```powershell
# Find and stop Python processes
Get-Process python | Where-Object {$_.CommandLine -like "*start_*"} | Stop-Process

# Or stop by port (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### If using Docker Compose:
```powershell
docker-compose -f docker-compose.microservices.yml down
```

## Troubleshooting

### Port Already in Use
If you get a "port already in use" error:
```powershell
# Find what's using the port
netstat -ano | findstr :8000

# Kill the process (replace <PID> with actual PID)
taskkill /PID <PID> /F
```

### Missing Dependencies
```powershell
pip install fastapi uvicorn httpx pydantic
```

### Services Not Starting
1. Check Python version: `python --version` (needs 3.11+)
2. Check if ports are available
3. Check for error messages in the terminal output

## Quick Start Script (PowerShell)

Save this as `start-services.ps1`:

```powershell
# Quick start script for Comply-AI services
Write-Host "Starting Comply-AI Services..." -ForegroundColor Green

# Install dependencies if needed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
pip install fastapi uvicorn httpx pydantic --quiet

# Start services
Write-Host "Starting services..." -ForegroundColor Yellow
Start-Process python -ArgumentList "start_detector_orchestration.py" -WindowStyle Hidden
Start-Sleep -Seconds 2
Start-Process python -ArgumentList "start_analysis_service.py" -WindowStyle Hidden
Start-Sleep -Seconds 2
Start-Process python -ArgumentList "start_mapper_service.py" -WindowStyle Hidden

# Wait for services to start
Write-Host "Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check health
Write-Host "`nChecking service health..." -ForegroundColor Cyan
$services = @(
    @{Name="Detector Orchestration"; Port=8000},
    @{Name="Analysis Service"; Port=8001},
    @{Name="Mapper Service"; Port=8002}
)

foreach ($service in $services) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$($service.Port)/health" -UseBasicParsing -TimeoutSec 2
        Write-Host "✓ $($service.Name) - Healthy" -ForegroundColor Green
    } catch {
        Write-Host "✗ $($service.Name) - Not responding" -ForegroundColor Red
    }
}

Write-Host "`nServices started! Access API docs at:" -ForegroundColor Green
Write-Host "  • http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "  • http://localhost:8001/docs" -ForegroundColor Cyan
Write-Host "  • http://localhost:8002/docs" -ForegroundColor Cyan
```

Run it with:
```powershell
.\start-services.ps1
```

