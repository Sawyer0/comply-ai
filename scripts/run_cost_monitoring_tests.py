#!/usr/bin/env python3
"""Test runner for cost monitoring system."""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all cost monitoring tests."""
    print("Running Cost Monitoring System Tests")
    print("=" * 40)

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Test commands to run
    test_commands = [
        {
            "name": "Unit Tests - Metrics Collector",
            "command": [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/cost_monitoring/test_metrics_collector.py",
                "-v",
                "--tb=short",
            ],
        },
        {
            "name": "Unit Tests - Guardrails",
            "command": [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/cost_monitoring/test_guardrails.py",
                "-v",
                "--tb=short",
            ],
        },
        {
            "name": "Unit Tests - Autoscaling",
            "command": [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/cost_monitoring/test_autoscaling.py",
                "-v",
                "--tb=short",
            ],
        },
        {
            "name": "Unit Tests - Analytics",
            "command": [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/cost_monitoring/test_analytics.py",
                "-v",
                "--tb=short",
            ],
        },
        {
            "name": "Integration Tests - System",
            "command": [
                sys.executable,
                "-m",
                "pytest",
                "tests/integration/cost_monitoring/test_cost_monitoring_system.py",
                "-v",
                "--tb=short",
            ],
        },
        {
            "name": "Integration Tests - CLI Commands",
            "command": [
                sys.executable,
                "-m",
                "pytest",
                "tests/integration/cost_monitoring/test_cli_commands.py",
                "-v",
                "--tb=short",
            ],
        },
        {
            "name": "All Cost Monitoring Tests",
            "command": [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/cost_monitoring/",
                "tests/integration/cost_monitoring/",
                "-v",
                "--tb=short",
                "--cov=src/llama_mapper/cost_monitoring",
                "--cov-report=term-missing",
            ],
        },
    ]

    results = []

    for test_info in test_commands:
        print(f"\n{test_info['name']}")
        print("-" * len(test_info["name"]))

        try:
            result = subprocess.run(
                test_info["command"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                print("✓ PASSED")
                results.append((test_info["name"], True, None))
            else:
                print("✗ FAILED")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                results.append((test_info["name"], False, result.stderr))

        except subprocess.TimeoutExpired:
            print("✗ TIMEOUT")
            results.append((test_info["name"], False, "Test timed out"))
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append((test_info["name"], False, str(e)))

    # Print summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)

    passed = 0
    failed = 0

    for name, success, error in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status} - {name}")
        if not success and error:
            print(f"    Error: {error[:100]}...")

        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed + failed}, Passed: {passed}, Failed: {failed}")

    if failed > 0:
        print("\nSome tests failed. Please check the output above for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed! 🎉")


if __name__ == "__main__":
    run_tests()
