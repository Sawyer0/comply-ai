#!/usr/bin/env python3
"""
Performance test runner for detector orchestration service.

This script provides a unified interface to run different types of performance tests:
- Smoke tests (light load, quick validation)
- Load tests (normal operational load)
- Stress tests (high load to find breaking points)
- Fault tolerance tests (test system behavior under failures)
- Scalability tests (test detector coordination under load)
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml


class PerformanceTestRunner:
    """Runner for detector orchestration performance tests"""

    def __init__(self, config_path: str = "test-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.test_results = {}
        self.start_time = datetime.now(timezone.utc)

    def _load_config(self) -> Dict[str, Any]:
        """Load test configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def run_smoke_test(self) -> Dict[str, Any]:
        """Run smoke test to verify basic functionality"""
        print("🔥 Running smoke test...")

        # Run k6 smoke test
        result = self._run_k6_test(
            test_type="smoke",
            vus=5,
            duration="2m",
            description="Basic functionality test"
        )

        # Run locust smoke test
        locust_result = self._run_locust_test(
            test_type="smoke",
            users=5,
            spawn_rate=1,
            duration=120
        )

        return {
            "test_type": "smoke",
            "timestamp": self.start_time.isoformat(),
            "k6_result": result,
            "locust_result": locust_result,
            "status": "success" if result["success"] and locust_result["success"] else "failed"
        }

    def run_load_test(self) -> Dict[str, Any]:
        """Run load test with normal operational load"""
        print("⚡ Running load test...")

        # Run k6 load test
        result = self._run_k6_test(
            test_type="load",
            vus=50,
            duration="10m",
            description="Normal operational load test"
        )

        # Run locust load test
        locust_result = self._run_locust_test(
            test_type="load",
            users=50,
            spawn_rate=5,
            duration=600
        )

        return {
            "test_type": "load",
            "timestamp": self.start_time.isoformat(),
            "k6_result": result,
            "locust_result": locust_result,
            "status": "success" if result["success"] and locust_result["success"] else "failed"
        }

    def run_stress_test(self) -> Dict[str, Any]:
        """Run stress test with high load"""
        print("💥 Running stress test...")

        # Run k6 stress test
        result = self._run_k6_test(
            test_type="stress",
            vus=200,
            duration="5m",
            description="High load stress test"
        )

        # Run locust stress test
        locust_result = self._run_locust_test(
            test_type="stress",
            users=200,
            spawn_rate=20,
            duration=300
        )

        return {
            "test_type": "stress",
            "timestamp": self.start_time.isoformat(),
            "k6_result": result,
            "locust_result": locust_result,
            "status": "success" if result["success"] and locust_result["success"] else "failed"
        }

    def run_fault_tolerance_test(self) -> Dict[str, Any]:
        """Run fault tolerance tests"""
        print("🛡️ Running fault tolerance test...")

        # Run locust fault tolerance test
        result = self._run_locust_test(
            test_type="fault_tolerance",
            users=30,
            spawn_rate=5,
            duration=300,
            user_class="FaultToleranceUser"
        )

        return {
            "test_type": "fault_tolerance",
            "timestamp": self.start_time.isoformat(),
            "locust_result": result,
            "status": "success" if result["success"] else "failed"
        }

    def run_scalability_test(self) -> Dict[str, Any]:
        """Run scalability tests for detector coordination"""
        print("📈 Running scalability test...")

        # Test with different batch sizes and concurrent users
        results = []

        for batch_size in [1, 5, 10, 20]:
            for users in [10, 25, 50]:
                result = self._run_locust_test(
                    test_type=f"scalability_batch_{batch_size}_users_{users}",
                    users=users,
                    spawn_rate=max(users // 10, 1),
                    duration=180,
                    environment_vars={
                        "ORCHESTRATION_PERF_BATCH_SIZE": str(batch_size)
                    }
                )
                results.append({
                    "batch_size": batch_size,
                    "users": users,
                    "result": result
                })

        return {
            "test_type": "scalability",
            "timestamp": self.start_time.isoformat(),
            "test_cases": results,
            "status": "success" if all(r["result"]["success"] for r in results) else "failed"
        }

    def _run_k6_test(self, test_type: str, vus: int, duration: str, description: str) -> Dict[str, Any]:
        """Run k6 performance test"""
        try:
            env = os.environ.copy()
            env.update({
                "K6_SMOKE_VUS": str(vus) if test_type == "smoke" else "",
                "K6_LOAD_START_VUS": str(vus // 4) if test_type == "load" else "",
                "K6_LOAD_TARGET_VUS": str(vus) if test_type == "load" else "",
                "K6_STRESS_PEAK_VUS": str(vus) if test_type == "stress" else "",
                "K6_SMOKE_DURATION": duration if test_type == "smoke" else "2m",
                "K6_LOAD_DURATION": duration if test_type == "load" else "7m",
                "PERF_BASE_URL": "http://orchestration:8000",
                "ORCHESTRATION_API_KEY_HEADER": "X-API-Key",
                "ORCHESTRATION_PERF_API_KEY": "test-key",
                "ORCHESTRATION_PERF_TENANT_ID": "perf-tenant",
                "ORCHESTRATION_POLICY_BUNDLE": "default"
            })

            cmd = [
                "docker", "run", "--rm", "--network", "host",
                "-v", f"{Path(__file__).parent}/../k6:/scripts",
                "-e", f"K6_SMOKE_VUS={vus}",
                "-e", f"K6_SMOKE_DURATION={duration}",
                "-e", "PERF_BASE_URL=http://localhost:8000",
                "grafana/k6:latest",
                "run", "/scripts/orchestration_load.js"
            ]

            print(f"   Running k6 {test_type} test: {description}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=1800  # 30 minute timeout
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "command": " ".join(cmd)
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test timed out",
                "timeout": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _run_locust_test(self, test_type: str, users: int, spawn_rate: int,
                        duration: int, user_class: str = "OrchestrationUser",
                        environment_vars: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Run Locust performance test"""
        try:
            env = os.environ.copy()
            if environment_vars:
                env.update(environment_vars)

            env.update({
                "ORCHESTRATION_PERF_TENANT_ID": "perf-tenant",
                "ORCHESTRATION_PERF_API_KEY": "test-key",
                "ORCHESTRATION_PERF_API_KEY_HEADER": "X-API-Key",
                "LOCUST_HOST": "http://localhost:8000",
                "LOCUST_LOGLEVEL": "INFO"
            })

            cmd = [
                "docker", "run", "--rm", "--network", "host",
                "-v", f"{Path(__file__).parent}/..:/mnt/locust",
                "-e", f"LOCUST_USERS={users}",
                "-e", f"LOCUST_SPAWN_RATE={spawn_rate}",
                "-e", f"LOCUST_RUN_TIME={duration}s",
                "-e", "LOCUST_HOST=http://localhost:8000",
                "locustio/locust:latest",
                "-f", f"/mnt/locust/perf/locustfile_orchestration.py",
                "--user-class", user_class,
                "--headless"
            ]

            print(f"   Running Locust {test_type} test with {users} users")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=duration + 300  # Add 5 minutes buffer
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "command": " ".join(cmd)
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test timed out",
                "timeout": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance test types"""
        print("🚀 Starting comprehensive performance test suite...")

        results = {
            "test_suite": "detector_orchestration_performance",
            "timestamp": self.start_time.isoformat(),
            "config": self.config,
            "results": {}
        }

        # Run each test type
        test_types = [
            ("smoke", self.run_smoke_test),
            ("load", self.run_load_test),
            ("stress", self.run_stress_test),
            ("fault_tolerance", self.run_fault_tolerance_test),
            ("scalability", self.run_scalability_test)
        ]

        for test_name, test_func in test_types:
            try:
                result = test_func()
                results["results"][test_name] = result
                print(f"✅ {test_name} test completed")
            except Exception as e:
                results["results"][test_name] = {
                    "status": "error",
                    "error": str(e)
                }
                print(f"❌ {test_name} test failed: {e}")

        # Generate summary
        results["summary"] = self._generate_summary(results["results"])

        # Save results
        self._save_results(results)

        return results

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test summary"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.get("status") == "success")
        failed_tests = total_tests - passed_tests

        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "duration_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds()
        }

    def _save_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"performance_test_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📊 Test results saved to: {filename}")

    def generate_report(self, results: Dict[str, Any]):
        """Generate human-readable test report"""
        summary = results.get("summary", {})
        print("\n" + "="*80)
        print("PERFORMANCE TEST REPORT")
        print("="*80)
        print(f"Test Suite: {results.get('test_suite', 'Unknown')}")
        print(f"Timestamp: {results.get('timestamp', 'Unknown')}")
        print(f"Duration: {summary.get('duration_seconds', 0)".1f"} seconds")
        print()

        print("SUMMARY:")
        print(f"  Total Tests: {summary.get('total_tests', 0)}")
        print(f"  Passed: {summary.get('passed', 0)}")
        print(f"  Failed: {summary.get('failed', 0)}")
        print(f"  Success Rate: {summary.get('success_rate', 0)".1%"}")
        print()

        print("DETAILED RESULTS:")
        for test_name, result in results.get("results", {}).items():
            status = result.get("status", "unknown")
            status_icon = "✅" if status == "success" else "❌"
            print(f"  {status_icon} {test_name}: {status.upper()}")

            if status == "success" and "k6_result" in result:
                k6_result = result["k6_result"]
                if k6_result.get("success"):
                    print("    📊 K6 test: PASSED")
                else:
                    print("    📊 K6 test: FAILED")

            if status == "success" and "locust_result" in result:
                locust_result = result["locust_result"]
                if locust_result.get("success"):
                    print("    🦗 Locust test: PASSED")
                else:
                    print("    🦗 Locust test: FAILED")

        print("="*80)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Run detector orchestration performance tests")
    parser.add_argument(
        "--test-type",
        choices=["smoke", "load", "stress", "fault-tolerance", "scalability", "all"],
        default="all",
        help="Type of test to run"
    )
    parser.add_argument(
        "--config",
        default="test-config.yaml",
        help="Path to test configuration file"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report from existing results"
    )

    args = parser.parse_args()

    if args.report_only:
        # Find latest results file
        results_files = list(Path(".").glob("performance_test_results_*.json"))
        if not results_files:
            print("❌ No test result files found")
            return 1

        latest_file = max(results_files, key=lambda f: f.stat().st_mtime)
        print(f"📊 Loading results from: {latest_file}")

        with open(latest_file, 'r') as f:
            results = json.load(f)

        runner = PerformanceTestRunner()
        runner.generate_report(results)
        return 0

    # Run tests
    runner = PerformanceTestRunner(args.config)

    try:
        if args.test_type == "all":
            results = runner.run_all_tests()
        elif args.test_type == "smoke":
            results = {"results": {"smoke": runner.run_smoke_test()}}
        elif args.test_type == "load":
            results = {"results": {"load": runner.run_load_test()}}
        elif args.test_type == "stress":
            results = {"results": {"stress": runner.run_stress_test()}}
        elif args.test_type == "fault-tolerance":
            results = {"results": {"fault_tolerance": runner.run_fault_tolerance_test()}}
        elif args.test_type == "scalability":
            results = {"results": {"scalability": runner.run_scalability_test()}}

        runner.generate_report(results)
        return 0

    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
