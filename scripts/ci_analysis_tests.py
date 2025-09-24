#!/usr/bin/env python3
"""
CI test runner for Analysis Module.

This script runs all analysis module tests including unit tests,
integration tests, and HTTP replay tests for comprehensive validation.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_command(
    cmd: List[str], cwd: Optional[Path] = None
) -> subprocess.CompletedProcess:
    """
    Run a command and return the result.

    Args:
        cmd: Command to run
        cwd: Working directory

    Returns:
        CompletedProcess result
    """
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result


def run_unit_tests(verbose: bool = False) -> Dict[str, Any]:
    """
    Run unit tests for analysis module.

    Args:
        verbose: Whether to run with verbose output

    Returns:
        Test results summary
    """
    print("🧪 Running unit tests...")

    cmd = ["python", "-m", "pytest", "tests/unit/test_analysis_validation.py"]
    if verbose:
        cmd.append("-v")

    result = run_command(cmd)

    return {
        "type": "unit_tests",
        "command": " ".join(cmd),
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def run_integration_tests(verbose: bool = False) -> Dict[str, Any]:
    """
    Run integration tests for analysis module.

    Args:
        verbose: Whether to run with verbose output

    Returns:
        Test results summary
    """
    print("🔗 Running integration tests...")

    cmd = ["python", "-m", "pytest", "tests/integration/test_analysis_http_replay.py"]
    if verbose:
        cmd.append("-v")

    result = run_command(cmd)

    return {
        "type": "integration_tests",
        "command": " ".join(cmd),
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def run_http_replay_tests(
    golden_cases_path: str = "tests/fixtures/analysis_golden_cases.json",
    min_pass_rate: float = 0.8,
) -> Dict[str, Any]:
    """
    Run HTTP replay tests.

    Args:
        golden_cases_path: Path to golden test cases
        min_pass_rate: Minimum pass rate required

    Returns:
        Test results summary
    """
    print("🎭 Running HTTP replay tests...")

    cmd = [
        "python",
        "scripts/run_http_replay_tests.py",
        "--golden-cases",
        golden_cases_path,
        "--min-pass-rate",
        str(min_pass_rate),
        "--no-fail",  # Don't exit on failure, we'll handle it
    ]

    result = run_command(cmd)

    # Try to parse JSON output if available
    test_results = None
    try:
        if result.stdout:
            # Look for JSON in stdout
            lines = result.stdout.split("\n")
            for line in lines:
                if line.strip().startswith("{"):
                    test_results = json.loads(line)
                    break
    except json.JSONDecodeError:
        pass

    return {
        "type": "http_replay_tests",
        "command": " ".join(cmd),
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
        "test_results": test_results,
    }


def run_lint_checks() -> Dict[str, Any]:
    """
    Run linting checks for analysis module.

    Returns:
        Lint results summary
    """
    print("🔍 Running lint checks...")

    # Run flake8 on analysis module
    cmd = [
        "python",
        "-m",
        "flake8",
        "src/llama_mapper/analysis/",
        "--max-line-length=88",
    ]
    result = run_command(cmd)

    return {
        "type": "lint_checks",
        "command": " ".join(cmd),
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def run_type_checks() -> Dict[str, Any]:
    """
    Run type checking for analysis module.

    Returns:
        Type check results summary
    """
    print("📝 Running type checks...")

    cmd = ["python", "-m", "mypy", "src/llama_mapper/analysis/"]
    result = run_command(cmd)

    return {
        "type": "type_checks",
        "command": " ".join(cmd),
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def run_all_tests(
    include_lint: bool = True,
    include_type_check: bool = True,
    verbose: bool = False,
    min_pass_rate: float = 0.8,
) -> Dict[str, Any]:
    """
    Run all analysis module tests.

    Args:
        include_lint: Whether to include lint checks
        include_type_check: Whether to include type checks
        verbose: Whether to run with verbose output
        min_pass_rate: Minimum pass rate for HTTP replay tests

    Returns:
        Comprehensive test results
    """
    print("🚀 Starting comprehensive analysis module testing...")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    results = []
    overall_success = True

    # Run lint checks
    if include_lint:
        lint_result = run_lint_checks()
        results.append(lint_result)
        if not lint_result["success"]:
            overall_success = False
        print()

    # Run type checks
    if include_type_check:
        type_result = run_type_checks()
        results.append(type_result)
        if not type_result["success"]:
            overall_success = False
        print()

    # Run unit tests
    unit_result = run_unit_tests(verbose)
    results.append(unit_result)
    if not unit_result["success"]:
        overall_success = False
    print()

    # Run integration tests
    integration_result = run_integration_tests(verbose)
    results.append(integration_result)
    if not integration_result["success"]:
        overall_success = False
    print()

    # Run HTTP replay tests
    replay_result = run_http_replay_tests(min_pass_rate=min_pass_rate)
    results.append(replay_result)
    if not replay_result["success"]:
        overall_success = False
    print()

    # Generate summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_success": overall_success,
        "total_tests": len(results),
        "passed_tests": len([r for r in results if r["success"]]),
        "failed_tests": len([r for r in results if not r["success"]]),
        "results": results,
    }

    # Print summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Overall Success: {'✅ PASS' if overall_success else '❌ FAIL'}")
    print(f"Total Test Suites: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print()

    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} {result['type']}")
        if not result["success"] and result["stderr"]:
            print(f"   Error: {result['stderr'][:200]}...")

    print()

    # HTTP replay test details
    replay_result = next((r for r in results if r["type"] == "http_replay_tests"), None)
    if replay_result and replay_result.get("test_results"):
        test_results = replay_result["test_results"]
        print("🎭 HTTP Replay Test Details:")
        print(f"   Pass Rate: {test_results.get('pass_rate', 0):.2%}")
        print(f"   Total Tests: {test_results.get('total_tests', 0)}")
        print(f"   Passed: {test_results.get('passed_tests', 0)}")
        print(f"   Failed: {test_results.get('failed_tests', 0)}")
        print(f"   Errors: {test_results.get('error_tests', 0)}")

    return summary


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive analysis module tests"
    )
    parser.add_argument("--no-lint", action="store_true", help="Skip lint checks")
    parser.add_argument("--no-type-check", action="store_true", help="Skip type checks")
    parser.add_argument(
        "--verbose", action="store_true", help="Run tests with verbose output"
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.8,
        help="Minimum pass rate for HTTP replay tests (default: 0.8)",
    )
    parser.add_argument("--output", help="Output file for test results (JSON format)")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument(
        "--integration-only", action="store_true", help="Run only integration tests"
    )
    parser.add_argument(
        "--replay-only", action="store_true", help="Run only HTTP replay tests"
    )

    args = parser.parse_args()

    try:
        if args.unit_only:
            result = run_unit_tests(args.verbose)
            summary = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_success": result["success"],
                "total_tests": 1,
                "passed_tests": 1 if result["success"] else 0,
                "failed_tests": 0 if result["success"] else 1,
                "results": [result],
            }
        elif args.integration_only:
            result = run_integration_tests(args.verbose)
            summary = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_success": result["success"],
                "total_tests": 1,
                "passed_tests": 1 if result["success"] else 0,
                "failed_tests": 0 if result["success"] else 1,
                "results": [result],
            }
        elif args.replay_only:
            result = run_http_replay_tests(min_pass_rate=args.min_pass_rate)
            summary = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_success": result["success"],
                "total_tests": 1,
                "passed_tests": 1 if result["success"] else 0,
                "failed_tests": 0 if result["success"] else 1,
                "results": [result],
            }
        else:
            summary = run_all_tests(
                include_lint=not args.no_lint,
                include_type_check=not args.no_type_check,
                verbose=args.verbose,
                min_pass_rate=args.min_pass_rate,
            )

        # Save results to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"Test results saved to: {output_path}")

        # Exit with appropriate code
        if summary["overall_success"]:
            print("🎉 All tests passed!")
            sys.exit(0)
        else:
            print("💥 Some tests failed!")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
