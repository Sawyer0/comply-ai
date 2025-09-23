#!/usr/bin/env python3
"""Demo script showing CLI and API integration."""

import json
import subprocess
import time
from pathlib import Path

import requests


def run_cli_command(command: list[str]) -> tuple[int, str, str]:
    """Run a CLI command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"


def check_api_health(base_url: str = "http://localhost:8000") -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def demo_cli_usage():
    """Demonstrate CLI usage."""
    print("=== CLI Demo ===")
    
    # Check API health using CLI
    print("1. Checking API health with CLI...")
    exit_code, stdout, stderr = run_cli_command([
        "python", "-m", "src.llama_mapper.cli.main", "api", "health"
    ])
    
    if exit_code == 0:
        print("✓ API is healthy")
        print(stdout)
    else:
        print("✗ API health check failed")
        print(f"Error: {stderr}")
        return False
    
    # Map single detector output
    print("\n2. Mapping single detector output...")
    sample_file = Path("examples/sample_data/detector_output.json")
    if sample_file.exists():
        exit_code, stdout, stderr = run_cli_command([
            "python", "-m", "src.llama_mapper.cli.main", "api", "map",
            "--input", str(sample_file),
            "--output", "mapped_result.json"
        ])
        
        if exit_code == 0:
            print("✓ Single mapping completed")
            print(stdout)
        else:
            print("✗ Single mapping failed")
            print(f"Error: {stderr}")
    else:
        print("✗ Sample file not found")
    
    # Batch mapping
    print("\n3. Batch mapping...")
    batch_file = Path("examples/sample_data/batch_input.json")
    if batch_file.exists():
        exit_code, stdout, stderr = run_cli_command([
            "python", "-m", "src.llama_mapper.cli.main", "api", "batch-map",
            "--input", str(batch_file),
            "--output", "batch_results.json"
        ])
        
        if exit_code == 0:
            print("✓ Batch mapping completed")
            print(stdout)
        else:
            print("✗ Batch mapping failed")
            print(f"Error: {stderr}")
    else:
        print("✗ Batch file not found")
    
    # Get metrics
    print("\n4. Getting API metrics...")
    exit_code, stdout, stderr = run_cli_command([
        "python", "-m", "src.llama_mapper.cli.main", "api", "metrics",
        "--format", "table"
    ])
    
    if exit_code == 0:
        print("✓ Metrics retrieved")
        print(stdout)
    else:
        print("✗ Metrics retrieval failed")
        print(f"Error: {stderr}")
    
    return True


def demo_api_usage():
    """Demonstrate direct API usage."""
    print("\n=== API Demo ===")
    
    base_url = "http://localhost:8000"
    
    # Check API health
    print("1. Checking API health...")
    if check_api_health(base_url):
        print("✓ API is healthy")
    else:
        print("✗ API is not responding")
        return False
    
    # Map single detector output
    print("\n2. Mapping single detector output via API...")
    sample_file = Path("examples/sample_data/detector_output.json")
    if sample_file.exists():
        with open(sample_file, 'r') as f:
            payload = json.load(f)
        
        try:
            response = requests.post(
                f"{base_url}/map",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            print("✓ Single mapping completed via API")
            print(f"Detector: {result.get('detector', 'unknown')}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            
        except requests.RequestException as e:
            print(f"✗ API mapping failed: {e}")
    else:
        print("✗ Sample file not found")
    
    # Batch mapping
    print("\n3. Batch mapping via API...")
    batch_file = Path("examples/sample_data/batch_input.json")
    if batch_file.exists():
        with open(batch_file, 'r') as f:
            payload = json.load(f)
        
        try:
            response = requests.post(
                f"{base_url}/map/batch",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            print("✓ Batch mapping completed via API")
            print(f"Total requests: {len(result.get('results', []))}")
            print(f"Errors: {len(result.get('errors', []))}")
            
        except requests.RequestException as e:
            print(f"✗ API batch mapping failed: {e}")
    else:
        print("✗ Batch file not found")
    
    # Get metrics
    print("\n4. Getting metrics via API...")
    try:
        response = requests.get(f"{base_url}/metrics/summary", timeout=10)
        response.raise_for_status()
        
        metrics = response.json()
        print("✓ Metrics retrieved via API")
        print(f"Metrics keys: {list(metrics.keys())}")
        
    except requests.RequestException as e:
        print(f"✗ API metrics retrieval failed: {e}")
    
    # Get alerts
    print("\n5. Getting alerts via API...")
    try:
        response = requests.get(f"{base_url}/metrics/alerts", timeout=10)
        response.raise_for_status()
        
        alerts = response.json()
        print("✓ Alerts retrieved via API")
        print(f"Alert count: {alerts.get('count', 0)}")
        
    except requests.RequestException as e:
        print(f"✗ API alerts retrieval failed: {e}")
    
    return True


def demo_combined_usage():
    """Demonstrate combined CLI and API usage."""
    print("\n=== Combined Usage Demo ===")
    
    # Use CLI to start server (in background)
    print("1. Starting API server with CLI...")
    # Note: In a real scenario, you'd start this in a separate process
    
    # Wait a moment for server to start
    time.sleep(2)
    
    # Use CLI to check health
    print("2. Checking health with CLI...")
    exit_code, stdout, stderr = run_cli_command([
        "python", "-m", "src.llama_mapper.cli.main", "api", "health"
    ])
    
    if exit_code == 0:
        print("✓ Server is healthy")
    else:
        print("✗ Server health check failed")
        return False
    
    # Use API directly for processing
    print("3. Processing with API...")
    sample_file = Path("examples/sample_data/detector_output.json")
    if sample_file.exists():
        with open(sample_file, 'r') as f:
            payload = json.load(f)
        
        try:
            response = requests.post(
                "http://localhost:8000/map",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            print("✓ Processing completed via API")
            
            # Save result using CLI
            with open("api_result.json", 'w') as f:
                json.dump(result, f, indent=2)
            print("✓ Result saved to api_result.json")
            
        except requests.RequestException as e:
            print(f"✗ API processing failed: {e}")
    
    # Use CLI for monitoring
    print("4. Monitoring with CLI...")
    exit_code, stdout, stderr = run_cli_command([
        "python", "-m", "src.llama_mapper.cli.main", "api", "metrics",
        "--format", "json"
    ])
    
    if exit_code == 0:
        print("✓ Monitoring data retrieved")
    else:
        print("✗ Monitoring failed")
    
    return True


def main():
    """Main demo function."""
    print("Llama Mapper CLI and API Integration Demo")
    print("=" * 50)
    
    # Check if API server is running
    if not check_api_health():
        print("⚠️  API server is not running. Please start it first:")
        print("   python -m src.llama_mapper.cli.main serve")
        print("   or")
        print("   mapper serve")
        return
    
    # Run demos
    try:
        demo_cli_usage()
        demo_api_usage()
        demo_combined_usage()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey takeaways:")
        print("- CLI is great for development, testing, and monitoring")
        print("- API is perfect for application integration")
        print("- Both can be used together for comprehensive workflows")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")


if __name__ == "__main__":
    main()
