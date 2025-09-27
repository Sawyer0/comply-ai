#!/usr/bin/env python3
"""
Start all microservices for testing.
This script starts all three services on different ports.
"""

import subprocess
import sys
import time
import os
from pathlib import Path


def start_service(script_name, service_name):
    """Start a service in a separate process"""
    print(f"Starting {service_name}...")

    # Use the virtual environment python
    venv_python = Path("microservices-env/Scripts/python.exe")
    if not venv_python.exists():
        venv_python = Path("microservices-env/bin/python")  # Unix

    if venv_python.exists():
        python_cmd = str(venv_python)
    else:
        python_cmd = sys.executable

    try:
        process = subprocess.Popen(
            [python_cmd, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"✅ {service_name} started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start {service_name}: {e}")
        return None


def main():
    """Start all services"""
    print("🚀 Starting all microservices...")
    print("=" * 50)

    services = [
        (
            "start_detector_orchestration.py",
            "Detector Orchestration Service (Port 8000)",
        ),
        ("start_analysis_service.py", "Analysis Service (Port 8001)"),
        ("start_mapper_service.py", "Mapper Service (Port 8002)"),
    ]

    processes = []

    for script, name in services:
        process = start_service(script, name)
        if process:
            processes.append((process, name))
        time.sleep(2)  # Give each service time to start

    print("\n" + "=" * 50)
    print("🎉 All services started!")
    print("\n📋 Service URLs:")
    print("• Detector Orchestration: http://localhost:8000/docs")
    print("• Analysis Service:       http://localhost:8001/docs")
    print("• Mapper Service:         http://localhost:8002/docs")

    print("\n🔍 Health Check URLs:")
    print("• Detector Orchestration: http://localhost:8000/health")
    print("• Analysis Service:       http://localhost:8001/health")
    print("• Mapper Service:         http://localhost:8002/health")

    print("\n⚡ API Endpoints to Test:")
    print("• POST http://localhost:8000/api/v1/orchestrate")
    print("• POST http://localhost:8001/api/v1/analyze")
    print("• POST http://localhost:8002/api/v1/map")
    print("• GET  http://localhost:8002/api/v1/taxonomy")

    print("\n⏹️  Press Ctrl+C to stop all services")

    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
        for process, name in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ Stopped {name}")
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
                try:
                    process.kill()
                except:
                    pass
        print("🏁 All services stopped")


if __name__ == "__main__":
    main()
