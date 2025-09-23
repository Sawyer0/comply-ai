#!/usr/bin/env python3
"""
Script to run HTTP replay tests for Analysis Module API conformance.

This script can be used in CI/CD pipelines to validate API conformance
using golden test cases.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from tests.integration.test_analysis_http_replay import HTTPReplayTester
from src.llama_mapper.analysis.api.factory import create_analysis_app


def run_http_replay_tests(
    golden_cases_path: str,
    output_file: Optional[str] = None,
    fail_on_error: bool = True,
    min_pass_rate: float = 0.8
) -> Dict[str, Any]:
    """
    Run HTTP replay tests using golden test cases.
    
    Args:
        golden_cases_path: Path to golden test cases JSON file
        output_file: Optional output file for test results
        fail_on_error: Whether to exit with error code on test failures
        min_pass_rate: Minimum pass rate required (0.0 to 1.0)
        
    Returns:
        Test results summary
    """
    print("Starting HTTP replay tests for Analysis Module API...")
    
    # Create analysis app
    print("Creating analysis app...")
    app = create_analysis_app()
    
    # Create replay tester
    print("Initializing HTTP replay tester...")
    tester = HTTPReplayTester(app, golden_cases_path)
    
    # Run all tests
    print("Running golden test cases...")
    summary = tester.run_all_tests()
    
    # Print results
    print(f"\n{'='*60}")
    print("HTTP REPLAY TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Errors: {summary['error_tests']}")
    print(f"Pass rate: {summary['pass_rate']:.2%}")
    print(f"{'='*60}")
    
    # Print failed tests
    failed_tests = [r for r in summary['results'] if r['status'] == 'fail']
    if failed_tests:
        print(f"\nFAILED TESTS ({len(failed_tests)}):")
        print("-" * 40)
        for test in failed_tests:
            print(f"❌ {test['test_id']}")
            for validation in test.get('validations', []):
                if not validation['passed']:
                    print(f"   - {validation['field']}: expected {validation['expected']}, got {validation['actual']}")
        print()
    
    # Print error tests
    error_tests = [r for r in summary['results'] if r['status'] == 'error']
    if error_tests:
        print(f"\nERROR TESTS ({len(error_tests)}):")
        print("-" * 40)
        for test in error_tests:
            print(f"💥 {test['test_id']}: {test.get('error', 'Unknown error')}")
        print()
    
    # Print passed tests summary
    passed_tests = [r for r in summary['results'] if r['status'] == 'pass']
    if passed_tests:
        print(f"\nPASSED TESTS ({len(passed_tests)}):")
        print("-" * 40)
        for test in passed_tests:
            print(f"✅ {test['test_id']}")
        print()
    
    # Save results to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Test results saved to: {output_path}")
    
    # Check pass rate
    if summary['pass_rate'] < min_pass_rate:
        print(f"❌ Pass rate {summary['pass_rate']:.2%} is below minimum {min_pass_rate:.2%}")
        if fail_on_error:
            sys.exit(1)
    else:
        print(f"✅ Pass rate {summary['pass_rate']:.2%} meets minimum requirement {min_pass_rate:.2%}")
    
    # Check for any failures
    if summary['failed_tests'] > 0 or summary['error_tests'] > 0:
        print(f"❌ {summary['failed_tests']} failed tests and {summary['error_tests']} error tests")
        if fail_on_error:
            sys.exit(1)
    else:
        print("✅ All tests passed!")
    
    return summary


def validate_golden_cases(golden_cases_path: str) -> bool:
    """
    Validate golden test cases file.
    
    Args:
        golden_cases_path: Path to golden test cases JSON file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(golden_cases_path, 'r') as f:
            golden_cases = json.load(f)
        
        # Check required fields
        required_fields = ["version", "description", "metadata", "test_cases"]
        for field in required_fields:
            if field not in golden_cases:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Check test cases structure
        test_cases = golden_cases["test_cases"]
        if not isinstance(test_cases, list):
            print("❌ test_cases must be a list")
            return False
        
        if len(test_cases) == 0:
            print("❌ No test cases found")
            return False
        
        # Validate each test case
        for i, test_case in enumerate(test_cases):
            required_test_fields = ["id", "endpoint", "method", "request", "expected_response"]
            for field in required_test_fields:
                if field not in test_case:
                    print(f"❌ Test case {i}: Missing required field: {field}")
                    return False
        
        print(f"✅ Golden test cases file is valid ({len(test_cases)} test cases)")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in golden test cases file: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Golden test cases file not found: {golden_cases_path}")
        return False
    except Exception as e:
        print(f"❌ Error validating golden test cases: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run HTTP replay tests for Analysis Module API")
    parser.add_argument(
        "--golden-cases",
        default="tests/fixtures/analysis_golden_cases.json",
        help="Path to golden test cases JSON file"
    )
    parser.add_argument(
        "--output",
        help="Output file for test results (JSON format)"
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Don't exit with error code on test failures"
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.8,
        help="Minimum pass rate required (default: 0.8)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate golden test cases file, don't run tests"
    )
    
    args = parser.parse_args()
    
    # Validate golden cases file
    if not validate_golden_cases(args.golden_cases):
        sys.exit(1)
    
    if args.validate_only:
        print("✅ Golden test cases validation completed successfully")
        sys.exit(0)
    
    # Run tests
    try:
        summary = run_http_replay_tests(
            golden_cases_path=args.golden_cases,
            output_file=args.output,
            fail_on_error=not args.no_fail,
            min_pass_rate=args.min_pass_rate
        )
        
        print(f"\n🎉 HTTP replay tests completed successfully!")
        print(f"   Pass rate: {summary['pass_rate']:.2%}")
        print(f"   Total tests: {summary['total_tests']}")
        
    except Exception as e:
        print(f"❌ Error running HTTP replay tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
