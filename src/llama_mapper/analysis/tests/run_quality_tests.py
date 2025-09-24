#!/usr/bin/env python3
"""
Quality alerting system test runner.

This script provides comprehensive testing for the quality alerting system
including unit tests, integration tests, and performance tests.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest


def run_tests(
    test_type: str = "all",
    verbose: bool = False,
    coverage: bool = False,
    performance: bool = False,
    parallel: bool = False,
) -> Dict[str, Any]:
    """
    Run quality alerting system tests.

    Args:
        test_type: Type of tests to run (unit, integration, performance, all)
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        performance: Include performance tests
        parallel: Run tests in parallel

    Returns:
        Dictionary with test results
    """
    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    # Add test directory
    test_dir = Path(__file__).parent

    # Determine which tests to run
    if test_type == "unit":
        cmd.append(str(test_dir / "unit"))
    elif test_type == "integration":
        cmd.append(str(test_dir / "integration"))
    elif test_type == "performance":
        cmd.append(str(test_dir / "performance"))
    else:  # all
        cmd.append(str(test_dir))

    # Add options
    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(
            [
                "--cov=src.llama_mapper.analysis.quality",
                "--cov-report=html",
                "--cov-report=term",
            ]
        )

    if parallel:
        cmd.extend(["-n", "auto"])

    # Add markers
    if not performance:
        cmd.extend(["-m", "not performance"])

    # Add other useful options
    cmd.extend(["--tb=short", "--strict-markers", "--disable-warnings"])

    print(f"Running command: {' '.join(cmd)}")

    # Run tests
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration": end_time - start_time,
        "command": " ".join(cmd),
    }


def run_linting() -> Dict[str, Any]:
    """Run linting checks on the quality alerting system."""
    quality_dir = Path(__file__).parent.parent / "quality"

    # Run flake8
    flake8_cmd = [
        "flake8",
        str(quality_dir),
        "--max-line-length=88",
        "--extend-ignore=E203,W503",
    ]
    flake8_result = subprocess.run(flake8_cmd, capture_output=True, text=True)

    # Run mypy
    mypy_cmd = ["mypy", str(quality_dir), "--ignore-missing-imports"]
    mypy_result = subprocess.run(mypy_cmd, capture_output=True, text=True)

    return {
        "flake8": {
            "returncode": flake8_result.returncode,
            "stdout": flake8_result.stdout,
            "stderr": flake8_result.stderr,
        },
        "mypy": {
            "returncode": mypy_result.returncode,
            "stdout": mypy_result.stdout,
            "stderr": mypy_result.stderr,
        },
    }


def run_type_checking() -> Dict[str, Any]:
    """Run type checking on the quality alerting system."""
    quality_dir = Path(__file__).parent.parent / "quality"

    cmd = ["mypy", str(quality_dir), "--strict", "--ignore-missing-imports"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def validate_imports() -> Dict[str, Any]:
    """Validate that all imports work correctly."""
    try:
        from ..quality import (
            AlertManager,
            AlertSeverity,
            CompositeAlertHandler,
            DegradationType,
            EmailAlertHandler,
            LoggingAlertHandler,
            QualityAlertingConfig,
            QualityAlertingSettings,
            QualityAlertingSystem,
            QualityDegradationDetector,
            QualityMetric,
            QualityMetricType,
            QualityMonitor,
            QualityThreshold,
            SlackAlertHandler,
            WebhookAlertHandler,
        )

        return {"success": True, "message": "All imports successful"}
    except ImportError as e:
        return {"success": False, "message": f"Import failed: {e}"}


def run_quick_smoke_test() -> Dict[str, Any]:
    """Run a quick smoke test to verify basic functionality."""
    try:
        from ..quality import (
            QualityAlertingSystem,
            QualityMetric,
            QualityMetricType,
        )

        # Create system
        system = QualityAlertingSystem(
            monitoring_interval_seconds=1, max_metrics_per_type=100
        )

        # Add a metric
        metric = QualityMetric(
            metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
            value=0.95,
            timestamp=time.time(),
            labels={"service": "smoke_test"},
        )

        system.process_metric(metric)

        # Check system status
        status = system.get_system_status()

        return {"success": True, "message": "Smoke test passed", "status": status}

    except Exception as e:
        return {"success": False, "message": f"Smoke test failed: {e}"}


def generate_test_report(results: Dict[str, Any]) -> str:
    """Generate a test report."""
    report = []
    report.append("=" * 80)
    report.append("QUALITY ALERTING SYSTEM TEST REPORT")
    report.append("=" * 80)
    report.append("")

    # Test results
    if "tests" in results:
        test_result = results["tests"]
        report.append("TEST RESULTS:")
        report.append(f"  Command: {test_result['command']}")
        report.append(f"  Duration: {test_result['duration']:.2f} seconds")
        report.append(f"  Return Code: {test_result['returncode']}")
        report.append("")

        if test_result["returncode"] == 0:
            report.append("  ✅ All tests passed!")
        else:
            report.append("  ❌ Some tests failed!")
            if test_result["stderr"]:
                report.append("  Error output:")
                report.append(test_result["stderr"])
        report.append("")

    # Linting results
    if "linting" in results:
        linting = results["linting"]
        report.append("LINTING RESULTS:")

        flake8 = linting["flake8"]
        if flake8["returncode"] == 0:
            report.append("  ✅ Flake8: No issues found")
        else:
            report.append("  ❌ Flake8: Issues found")
            if flake8["stdout"]:
                report.append(flake8["stdout"])

        mypy = linting["mypy"]
        if mypy["returncode"] == 0:
            report.append("  ✅ MyPy: No issues found")
        else:
            report.append("  ❌ MyPy: Issues found")
            if mypy["stdout"]:
                report.append(mypy["stdout"])
        report.append("")

    # Import validation
    if "imports" in results:
        imports = results["imports"]
        if imports["success"]:
            report.append("✅ Import validation: All imports successful")
        else:
            report.append(f"❌ Import validation: {imports['message']}")
        report.append("")

    # Smoke test
    if "smoke_test" in results:
        smoke = results["smoke_test"]
        if smoke["success"]:
            report.append("✅ Smoke test: Basic functionality verified")
        else:
            report.append(f"❌ Smoke test: {smoke['message']}")
        report.append("")

    # Overall status
    all_passed = all(
        [
            results.get("tests", {}).get("returncode", 0) == 0,
            results.get("linting", {}).get("flake8", {}).get("returncode", 0) == 0,
            results.get("linting", {}).get("mypy", {}).get("returncode", 0) == 0,
            results.get("imports", {}).get("success", False),
            results.get("smoke_test", {}).get("success", False),
        ]
    )

    if all_passed:
        report.append("🎉 OVERALL STATUS: ALL CHECKS PASSED")
    else:
        report.append("⚠️  OVERALL STATUS: SOME CHECKS FAILED")

    report.append("=" * 80)

    return "\n".join(report)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run quality alerting system tests")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "performance", "all"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Enable coverage reporting"
    )
    parser.add_argument(
        "--performance", "-p", action="store_true", help="Include performance tests"
    )
    parser.add_argument(
        "--parallel", "-j", action="store_true", help="Run tests in parallel"
    )
    parser.add_argument(
        "--linting", "-l", action="store_true", help="Run linting checks"
    )
    parser.add_argument(
        "--smoke", "-s", action="store_true", help="Run smoke test only"
    )
    parser.add_argument(
        "--report", "-r", action="store_true", help="Generate detailed report"
    )

    args = parser.parse_args()

    results = {}

    # Run smoke test if requested
    if args.smoke:
        print("Running smoke test...")
        results["smoke_test"] = run_quick_smoke_test()
        print("Smoke test completed.")
        return

    # Run tests
    print(f"Running {args.type} tests...")
    results["tests"] = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage,
        performance=args.performance,
        parallel=args.parallel,
    )
    print("Tests completed.")

    # Run linting if requested
    if args.linting:
        print("Running linting checks...")
        results["linting"] = run_linting()
        print("Linting completed.")

    # Validate imports
    print("Validating imports...")
    results["imports"] = validate_imports()
    print("Import validation completed.")

    # Run smoke test
    print("Running smoke test...")
    results["smoke_test"] = run_quick_smoke_test()
    print("Smoke test completed.")

    # Generate report if requested
    if args.report:
        report = generate_test_report(results)
        print("\n" + report)

        # Save report to file
        report_file = Path(__file__).parent / "test_report.txt"
        with open(report_file, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {report_file}")

    # Exit with appropriate code
    test_failed = results.get("tests", {}).get("returncode", 0) != 0
    if test_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
