# Quick start script for Comply-AI services
# Usage: .\start-services.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Comply-AI Services Startup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "start_detector_orchestration.py")) {
    Write-Host "Error: Please run this script from the project root directory" -ForegroundColor Red
    exit 1
}

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  Error: Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Install dependencies if needed
Write-Host "`nChecking dependencies..." -ForegroundColor Yellow
$packages = @("fastapi", "uvicorn", "httpx", "pydantic")
$missing = @()

foreach ($package in $packages) {
    $installed = python -m pip show $package 2>&1
    if ($LASTEXITCODE -ne 0) {
        $missing += $package
    }
}

if ($missing.Count -gt 0) {
    Write-Host "  Installing missing packages: $($missing -join ', ')" -ForegroundColor Yellow
    python -m pip install $missing --quiet
    Write-Host "  Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  All dependencies installed" -ForegroundColor Green
}

# Check if services are already running
Write-Host "`nChecking for existing services..." -ForegroundColor Yellow
$ports = @(8000, 8001, 8002)
$running = @()

foreach ($port in $ports) {
    $connection = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connection) {
        $running += $port
    }
}

if ($running.Count -gt 0) {
    Write-Host "  Warning: Services already running on ports: $($running -join ', ')" -ForegroundColor Yellow
    $response = Read-Host "  Do you want to continue anyway? (y/n)"
    if ($response -ne "y") {
        Write-Host "  Aborted" -ForegroundColor Red
        exit 0
    }
}

# Start services
Write-Host "`nStarting services..." -ForegroundColor Yellow

Write-Host "  Starting Detector Orchestration Service (port 8000)..." -ForegroundColor Cyan
Start-Process python -ArgumentList "start_detector_orchestration.py" -WindowStyle Hidden
Start-Sleep -Seconds 2

Write-Host "  Starting Analysis Service (port 8001)..." -ForegroundColor Cyan
Start-Process python -ArgumentList "start_analysis_service.py" -WindowStyle Hidden
Start-Sleep -Seconds 2

Write-Host "  Starting Mapper Service (port 8002)..." -ForegroundColor Cyan
Start-Process python -ArgumentList "start_mapper_service.py" -WindowStyle Hidden
Start-Sleep -Seconds 3

# Wait for services to initialize
Write-Host "`nWaiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check health
Write-Host "`nChecking service health..." -ForegroundColor Yellow
$services = @(
    @{Name="Detector Orchestration"; Port=8000; URL="http://localhost:8000/docs"},
    @{Name="Analysis Service"; Port=8001; URL="http://localhost:8001/docs"},
    @{Name="Mapper Service"; Port=8002; URL="http://localhost:8002/docs"}
)

$allHealthy = $true
foreach ($service in $services) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$($service.Port)/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        Write-Host "  ✓ $($service.Name) - Healthy" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ $($service.Name) - Not responding" -ForegroundColor Red
        $allHealthy = $false
    }
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
if ($allHealthy) {
    Write-Host "  All services started successfully!" -ForegroundColor Green
} else {
    Write-Host "  Some services may still be starting..." -ForegroundColor Yellow
    Write-Host "  Wait a few seconds and check again" -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nService URLs:" -ForegroundColor Cyan
foreach ($service in $services) {
    Write-Host "  • $($service.Name): $($service.URL)" -ForegroundColor White
}

Write-Host "`nHealth Check URLs:" -ForegroundColor Cyan
Write-Host "  • http://localhost:8000/health" -ForegroundColor White
Write-Host "  • http://localhost:8001/health" -ForegroundColor White
Write-Host "  • http://localhost:8002/health" -ForegroundColor White

Write-Host "`nTo stop services, use:" -ForegroundColor Yellow
Write-Host "  Get-Process python | Where-Object {\$_.Path -like '*comply-ai*'} | Stop-Process" -ForegroundColor Gray
Write-Host ""

