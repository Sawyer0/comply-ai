#!/usr/bin/env python3
"""Test script for the simplified API client commands."""

import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> tuple[int, str, str]:
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


def test_api_client_commands():
    """Test the API client commands."""
    print("Testing API Client Commands")
    print("=" * 40)
    
    # Test 1: Show help
    print("\n1. Testing help command...")
    exit_code, stdout, stderr = run_command([
        "python", "-m", "src.llama_mapper.cli.main", "api-client", "--help"
    ])
    
    if exit_code == 0:
        print("✓ Help command works")
        print("Available commands:")
        for line in stdout.split('\n'):
            if 'Commands:' in line or (line.strip() and not line.startswith('Usage:')):
                print(f"  {line}")
    else:
        print("✗ Help command failed")
        print(f"Error: {stderr}")
    
    # Test 2: Test API connectivity (this will fail if server isn't running)
    print("\n2. Testing API connectivity...")
    exit_code, stdout, stderr = run_command([
        "python", "-m", "src.llama_mapper.cli.main", "api-client", "test"
    ])
    
    if exit_code == 0:
        print("✓ API test command works")
        print(stdout)
    else:
        print("⚠ API test failed (expected if server isn't running)")
        print(f"Error: {stderr}")
    
    # Test 3: Test health check
    print("\n3. Testing health check...")
    exit_code, stdout, stderr = run_command([
        "python", "-m", "src.llama_mapper.cli.main", "api-client", "health"
    ])
    
    if exit_code == 0:
        print("✓ Health check command works")
        print(stdout)
    else:
        print("⚠ Health check failed (expected if server isn't running)")
        print(f"Error: {stderr}")
    
    # Test 4: Test metrics
    print("\n4. Testing metrics command...")
    exit_code, stdout, stderr = run_command([
        "python", "-m", "src.llama_mapper.cli.main", "api-client", "metrics"
    ])
    
    if exit_code == 0:
        print("✓ Metrics command works")
        print(stdout)
    else:
        print("⚠ Metrics command failed (expected if server isn't running)")
        print(f"Error: {stderr}")
    
    # Test 5: Test with custom endpoint
    print("\n5. Testing custom endpoint...")
    exit_code, stdout, stderr = run_command([
        "python", "-m", "src.llama_mapper.cli.main", "api-client", "test",
        "--endpoint", "/metrics", "--method", "GET"
    ])
    
    if exit_code == 0:
        print("✓ Custom endpoint test works")
        print(stdout)
    else:
        print("⚠ Custom endpoint test failed (expected if server isn't running)")
        print(f"Error: {stderr}")


def main():
    """Main test function."""
    print("API Client Commands Test")
    print("=" * 30)
    print("Note: Some tests will fail if the API server isn't running.")
    print("Start the server with: mapper serve")
    print()
    
    try:
        test_api_client_commands()
        
        print("\n" + "=" * 40)
        print("Test completed!")
        print("\nTo test with a running server:")
        print("1. Start the server: mapper serve")
        print("2. Run this test again: python examples/test_api_client.py")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")


if __name__ == "__main__":
    main()
