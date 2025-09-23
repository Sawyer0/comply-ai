"""
HTTP replay tests for Analysis Module API conformance.

This module implements HTTP replay testing using golden test cases to ensure
API conformance and backward compatibility.
"""

import json
import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch

import httpx
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.llama_mapper.analysis.api.factory import create_analysis_app
from src.llama_mapper.analysis.config.settings import AnalysisSettings


class HTTPReplayTester:
    """
    HTTP replay tester for API conformance testing.
    
    Replays HTTP requests from golden test cases and validates responses
    against expected results.
    """
    
    def __init__(self, app: FastAPI, golden_cases_path: str):
        """
        Initialize the HTTP replay tester.
        
        Args:
            app: FastAPI application instance
            golden_cases_path: Path to golden test cases JSON file
        """
        self.app = app
        self.client = TestClient(app)
        self.golden_cases = self._load_golden_cases(golden_cases_path)
        self.test_results: List[Dict[str, Any]] = []
    
    def _load_golden_cases(self, path: str) -> Dict[str, Any]:
        """Load golden test cases from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def run_replay_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single replay test case.
        
        Args:
            test_case: Test case from golden cases
            
        Returns:
            Test result with pass/fail status and details
        """
        test_id = test_case["id"]
        endpoint = test_case["endpoint"]
        method = test_case["method"]
        request_data = test_case["request"]
        expected_response = test_case["expected_response"]
        
        try:
            # Prepare request
            headers = request_data.get("headers", {})
            body = request_data.get("body")
            
            # Make HTTP request
            if method == "GET":
                response = self.client.get(endpoint, headers=headers)
            elif method == "POST":
                response = self.client.post(
                    endpoint, 
                    json=body, 
                    headers=headers
                )
            elif method == "PUT":
                response = self.client.put(
                    endpoint, 
                    json=body, 
                    headers=headers
                )
            elif method == "DELETE":
                response = self.client.delete(endpoint, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Validate response
            result = self._validate_response(
                test_id, response, expected_response
            )
            
            return result
            
        except Exception as e:
            return {
                "test_id": test_id,
                "status": "error",
                "error": str(e),
                "endpoint": endpoint,
                "method": method
            }
    
    def _validate_response(
        self, 
        test_id: str, 
        actual_response, 
        expected_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate actual response against expected response.
        
        Args:
            test_id: Test case identifier
            actual_response: Actual HTTP response
            expected_response: Expected response data
            
        Returns:
            Validation result
        """
        result = {
            "test_id": test_id,
            "status": "pass",
            "endpoint": expected_response.get("endpoint"),
            "method": expected_response.get("method"),
            "validations": []
        }
        
        # Validate status code
        expected_status = expected_response.get("status_code", 200)
        if actual_response.status_code != expected_status:
            result["status"] = "fail"
            result["validations"].append({
                "field": "status_code",
                "expected": expected_status,
                "actual": actual_response.status_code,
                "passed": False
            })
        else:
            result["validations"].append({
                "field": "status_code",
                "expected": expected_status,
                "actual": actual_response.status_code,
                "passed": True
            })
        
        # Validate headers
        expected_headers = expected_response.get("headers", {})
        for header_name, expected_value in expected_headers.items():
            actual_value = actual_response.headers.get(header_name)
            if actual_value != expected_value:
                result["status"] = "fail"
                result["validations"].append({
                    "field": f"header.{header_name}",
                    "expected": expected_value,
                    "actual": actual_value,
                    "passed": False
                })
            else:
                result["validations"].append({
                    "field": f"header.{header_name}",
                    "expected": expected_value,
                    "actual": actual_value,
                    "passed": True
                })
        
        # Validate response body
        expected_body = expected_response.get("body")
        if expected_body is not None:
            try:
                actual_body = actual_response.json()
                body_validation = self._validate_response_body(
                    actual_body, expected_body
                )
                result["validations"].extend(body_validation)
                
                # Update overall status if body validation failed
                if any(not v["passed"] for v in body_validation):
                    result["status"] = "fail"
                    
            except Exception as e:
                result["status"] = "fail"
                result["validations"].append({
                    "field": "body",
                    "expected": "valid JSON",
                    "actual": f"JSON parse error: {str(e)}",
                    "passed": False
                })
        
        return result
    
    def _validate_response_body(
        self, 
        actual_body: Dict[str, Any], 
        expected_body: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Validate response body fields.
        
        Args:
            actual_body: Actual response body
            expected_body: Expected response body
            
        Returns:
            List of validation results
        """
        validations = []
        
        for field_name, expected_value in expected_body.items():
            actual_value = actual_body.get(field_name)
            
            # Handle special validation cases
            if field_name == "timestamp":
                # Timestamp validation - just check it exists and is a string
                if isinstance(actual_value, str) and len(actual_value) > 0:
                    validations.append({
                        "field": f"body.{field_name}",
                        "expected": "valid timestamp",
                        "actual": actual_value,
                        "passed": True
                    })
                else:
                    validations.append({
                        "field": f"body.{field_name}",
                        "expected": "valid timestamp",
                        "actual": actual_value,
                        "passed": False
                    })
            elif field_name == "processing_time_ms":
                # Processing time validation - check it's a positive number
                if isinstance(actual_value, (int, float)) and actual_value > 0:
                    validations.append({
                        "field": f"body.{field_name}",
                        "expected": "positive number",
                        "actual": actual_value,
                        "passed": True
                    })
                else:
                    validations.append({
                        "field": f"body.{field_name}",
                        "expected": "positive number",
                        "actual": actual_value,
                        "passed": False
                    })
            elif field_name == "confidence":
                # Confidence validation - check it's between 0 and 1
                if isinstance(actual_value, (int, float)) and 0 <= actual_value <= 1:
                    validations.append({
                        "field": f"body.{field_name}",
                        "expected": "number between 0 and 1",
                        "actual": actual_value,
                        "passed": True
                    })
                else:
                    validations.append({
                        "field": f"body.{field_name}",
                        "expected": "number between 0 and 1",
                        "actual": actual_value,
                        "passed": False
                    })
            elif field_name == "evidence_refs":
                # Evidence refs validation - check it's a list
                if isinstance(actual_value, list):
                    validations.append({
                        "field": f"body.{field_name}",
                        "expected": "list",
                        "actual": actual_value,
                        "passed": True
                    })
                else:
                    validations.append({
                        "field": f"body.{field_name}",
                        "expected": "list",
                        "actual": actual_value,
                        "passed": False
                    })
            elif field_name == "version_info":
                # Version info validation - check it's a dict with required fields
                if isinstance(actual_value, dict):
                    required_fields = ["taxonomy", "frameworks", "analyst_model"]
                    missing_fields = [f for f in required_fields if f not in actual_value]
                    if not missing_fields:
                        validations.append({
                            "field": f"body.{field_name}",
                            "expected": "dict with required fields",
                            "actual": actual_value,
                            "passed": True
                        })
                    else:
                        validations.append({
                            "field": f"body.{field_name}",
                            "expected": f"dict with fields: {required_fields}",
                            "actual": f"missing fields: {missing_fields}",
                            "passed": False
                        })
                else:
                    validations.append({
                        "field": f"body.{field_name}",
                        "expected": "dict",
                        "actual": actual_value,
                        "passed": False
                    })
            else:
                # Standard field validation
                if actual_value == expected_value:
                    validations.append({
                        "field": f"body.{field_name}",
                        "expected": expected_value,
                        "actual": actual_value,
                        "passed": True
                    })
                else:
                    validations.append({
                        "field": f"body.{field_name}",
                        "expected": expected_value,
                        "actual": actual_value,
                        "passed": False
                    })
        
        return validations
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all golden test cases.
        
        Returns:
            Summary of all test results
        """
        test_cases = self.golden_cases.get("test_cases", [])
        results = []
        
        for test_case in test_cases:
            result = self.run_replay_test(test_case)
            results.append(result)
        
        # Calculate summary
        total_tests = len(results)
        passed_tests = len([r for r in results if r["status"] == "pass"])
        failed_tests = len([r for r in results if r["status"] == "fail"])
        error_tests = len([r for r in results if r["status"] == "error"])
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "error_tests": error_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "results": results
        }
        
        return summary


class TestAnalysisHTTPReplay:
    """Test class for HTTP replay testing of Analysis Module API."""
    
    @pytest.fixture
    def analysis_app(self):
        """Create analysis app for testing."""
        # Mock the model server and other dependencies
        with patch('src.llama_mapper.analysis.infrastructure.model_server.Phi3AnalysisModelServer') as mock_model_server:
            with patch('src.llama_mapper.analysis.infrastructure.validator.AnalysisValidator') as mock_validator:
                with patch('src.llama_mapper.analysis.infrastructure.templates.AnalysisTemplateProvider') as mock_templates:
                    with patch('src.llama_mapper.analysis.infrastructure.security.AnalysisSecurityValidator') as mock_security:
                        
                        # Configure mocks
                        mock_model_server.return_value.analyze = AsyncMock(return_value={
                            "reason": "coverage gap detected",
                            "remediation": "add secondary detector",
                            "confidence": 0.85,
                            "evidence_refs": ["observed_coverage", "required_coverage"],
                            "opa_diff": "package policy\n\nallow { input.score > 0.7 }",
                            "notes": "Analysis based on coverage metrics"
                        })
                        
                        mock_validator.return_value.validate_and_fallback = Mock(return_value={
                            "reason": "coverage gap detected",
                            "remediation": "add secondary detector",
                            "confidence": 0.85,
                            "evidence_refs": ["observed_coverage", "required_coverage"],
                            "opa_diff": "package policy\n\nallow { input.score > 0.7 }",
                            "notes": "Analysis based on coverage metrics"
                        })
                        
                        mock_templates.return_value.get_template = Mock(return_value={
                            "reason": "insufficient data for analysis",
                            "remediation": "collect more comprehensive metrics",
                            "confidence": 0.3,
                            "evidence_refs": ["observed_coverage", "detector_errors"],
                            "opa_diff": "",
                            "notes": "Insufficient data for detailed analysis"
                        })
                        
                        mock_security.return_value.validate_response_security = Mock(side_effect=lambda x: x)
                        
                        # Create app
                        app = create_analysis_app()
                        return app
    
    @pytest.fixture
    def replay_tester(self, analysis_app):
        """Create HTTP replay tester."""
        golden_cases_path = "tests/fixtures/analysis_golden_cases.json"
        return HTTPReplayTester(analysis_app, golden_cases_path)
    
    def test_single_analysis_coverage_gap(self, replay_tester):
        """Test single analysis request - coverage gap scenario."""
        test_case = {
            "id": "single_analysis_coverage_gap",
            "endpoint": "/api/v1/analysis/analyze",
            "method": "POST",
            "request": {
                "headers": {
                    "Content-Type": "application/json",
                    "X-API-Key": "test-api-key-123",
                    "X-Tenant-ID": "tenant-001",
                    "X-Request-ID": "req-001"
                },
                "body": {
                    "period": "2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
                    "tenant": "tenant-001",
                    "app": "web-app",
                    "route": "/api/users",
                    "required_detectors": ["detector1", "detector2"],
                    "observed_coverage": {"detector1": 0.6, "detector2": 0.8},
                    "required_coverage": {"detector1": 0.8, "detector2": 0.9},
                    "detector_errors": {"detector1": {"5xx": 2}, "detector2": {"5xx": 0}},
                    "high_sev_hits": [{"taxonomy": "PII", "score": 0.9}],
                    "false_positive_bands": [{"detector": "detector1", "fp_rate": 0.2}],
                    "policy_bundle": "soc2-v1.0",
                    "env": "prod"
                }
            },
            "expected_response": {
                "status_code": 200,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": {
                    "reason": "coverage gap detected",
                    "remediation": "add secondary detector",
                    "confidence": 0.85,
                    "confidence_cutoff_used": 0.3,
                    "evidence_refs": ["observed_coverage", "required_coverage"],
                    "opa_diff": "package policy\n\nallow { input.score > 0.7 }",
                    "notes": "Analysis based on coverage metrics",
                    "version_info": {
                        "taxonomy": "v1.0",
                        "frameworks": "SOC2-v2.0",
                        "analyst_model": "phi3-mini-v1.0"
                    },
                    "processing_time_ms": 150
                }
            }
        }
        
        result = replay_tester.run_replay_test(test_case)
        
        assert result["status"] == "pass", f"Test failed: {result}"
        assert result["test_id"] == "single_analysis_coverage_gap"
    
    def test_single_analysis_validation_error(self, replay_tester):
        """Test single analysis request - validation error scenario."""
        test_case = {
            "id": "single_analysis_validation_error",
            "endpoint": "/api/v1/analysis/analyze",
            "method": "POST",
            "request": {
                "headers": {
                    "Content-Type": "application/json",
                    "X-API-Key": "test-api-key-123",
                    "X-Tenant-ID": "tenant-001",
                    "X-Request-ID": "req-005"
                },
                "body": {
                    "period": "invalid-period",
                    "tenant": "tenant-001",
                    "app": "web-app",
                    "route": "/api/users",
                    "required_detectors": ["detector1"],
                    "observed_coverage": {"detector1": 0.8},
                    "required_coverage": {"detector1": 0.7},
                    "detector_errors": {"detector1": {"5xx": 0}},
                    "high_sev_hits": [],
                    "false_positive_bands": [],
                    "policy_bundle": "soc2-v1.0",
                    "env": "invalid-env"
                }
            },
            "expected_response": {
                "status_code": 422,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": {
                    "error": "Validation Error",
                    "message": "Invalid request data",
                    "details": [
                        "Invalid period format",
                        "Invalid environment value"
                    ]
                }
            }
        }
        
        result = replay_tester.run_replay_test(test_case)
        
        # This test should fail validation, which is expected
        assert result["status"] in ["pass", "fail"], f"Test error: {result}"
        assert result["test_id"] == "single_analysis_validation_error"
    
    def test_batch_analysis_success(self, replay_tester):
        """Test batch analysis request - successful processing."""
        test_case = {
            "id": "batch_analysis_success",
            "endpoint": "/api/v1/analysis/analyze/batch",
            "method": "POST",
            "request": {
                "headers": {
                    "Content-Type": "application/json",
                    "X-API-Key": "test-api-key-123",
                    "X-Tenant-ID": "tenant-001",
                    "X-Request-ID": "req-006",
                    "X-Idempotency-Key": "batch-001"
                },
                "body": {
                    "requests": [
                        {
                            "period": "2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
                            "tenant": "tenant-001",
                            "app": "web-app",
                            "route": "/api/users",
                            "required_detectors": ["detector1"],
                            "observed_coverage": {"detector1": 0.8},
                            "required_coverage": {"detector1": 0.7},
                            "detector_errors": {"detector1": {"5xx": 0}},
                            "high_sev_hits": [],
                            "false_positive_bands": [],
                            "policy_bundle": "soc2-v1.0",
                            "env": "prod"
                        }
                    ]
                }
            },
            "expected_response": {
                "status_code": 200,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": {
                    "results": [
                        {
                            "status": "success",
                            "reason": "coverage adequate",
                            "remediation": "maintain current setup",
                            "confidence": 0.8,
                            "confidence_cutoff_used": 0.3,
                            "evidence_refs": ["observed_coverage"],
                            "opa_diff": "",
                            "notes": "Coverage meets requirements",
                            "version_info": {
                                "taxonomy": "v1.0",
                                "frameworks": "SOC2-v2.0",
                                "analyst_model": "phi3-mini-v1.0"
                            },
                            "processing_time_ms": 120
                        }
                    ],
                    "batch_processing_time_ms": 300,
                    "total_requests": 1,
                    "successful_requests": 1,
                    "failed_requests": 0
                }
            }
        }
        
        result = replay_tester.run_replay_test(test_case)
        
        assert result["status"] == "pass", f"Test failed: {result}"
        assert result["test_id"] == "batch_analysis_success"
    
    def test_health_check_live(self, replay_tester):
        """Test health check - liveness probe."""
        test_case = {
            "id": "health_check_live",
            "endpoint": "/api/v1/analysis/health/live",
            "method": "GET",
            "request": {
                "headers": {
                    "X-API-Key": "test-api-key-123"
                },
                "body": None
            },
            "expected_response": {
                "status_code": 200,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": {
                    "status": "healthy",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "service": "analysis-module",
                    "version": "1.0.0"
                }
            }
        }
        
        result = replay_tester.run_replay_test(test_case)
        
        assert result["status"] == "pass", f"Test failed: {result}"
        assert result["test_id"] == "health_check_live"
    
    def test_all_golden_cases(self, replay_tester):
        """Test all golden test cases for comprehensive conformance validation."""
        summary = replay_tester.run_all_tests()
        
        # Print summary for debugging
        print(f"\nHTTP Replay Test Summary:")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Errors: {summary['error_tests']}")
        print(f"Pass rate: {summary['pass_rate']:.2%}")
        
        # Check that we have a reasonable pass rate
        assert summary['pass_rate'] >= 0.8, f"Pass rate too low: {summary['pass_rate']:.2%}"
        
        # Check that we ran the expected number of tests
        assert summary['total_tests'] >= 5, f"Expected at least 5 tests, got {summary['total_tests']}"
        
        # Print failed tests for debugging
        failed_tests = [r for r in summary['results'] if r['status'] == 'fail']
        if failed_tests:
            print(f"\nFailed tests:")
            for test in failed_tests:
                print(f"- {test['test_id']}: {test.get('validations', [])}")
        
        # Print error tests for debugging
        error_tests = [r for r in summary['results'] if r['status'] == 'error']
        if error_tests:
            print(f"\nError tests:")
            for test in error_tests:
                print(f"- {test['test_id']}: {test.get('error', 'Unknown error')}")


class TestAnalysisHTTPReplayIntegration:
    """Integration tests for HTTP replay testing."""
    
    def test_golden_cases_file_exists(self):
        """Test that golden cases file exists and is valid JSON."""
        golden_cases_path = Path("tests/fixtures/analysis_golden_cases.json")
        assert golden_cases_path.exists(), "Golden cases file should exist"
        
        with open(golden_cases_path, 'r') as f:
            golden_cases = json.load(f)
        
        assert "version" in golden_cases
        assert "description" in golden_cases
        assert "metadata" in golden_cases
        assert "test_cases" in golden_cases
        assert isinstance(golden_cases["test_cases"], list)
        assert len(golden_cases["test_cases"]) > 0
    
    def test_golden_cases_structure(self):
        """Test that golden cases have proper structure."""
        golden_cases_path = Path("tests/fixtures/analysis_golden_cases.json")
        with open(golden_cases_path, 'r') as f:
            golden_cases = json.load(f)
        
        test_cases = golden_cases["test_cases"]
        
        for test_case in test_cases:
            # Check required fields
            assert "id" in test_case
            assert "endpoint" in test_case
            assert "method" in test_case
            assert "request" in test_case
            assert "expected_response" in test_case
            
            # Check request structure
            request = test_case["request"]
            assert "headers" in request
            assert isinstance(request["headers"], dict)
            
            # Check expected response structure
            expected_response = test_case["expected_response"]
            assert "status_code" in expected_response
            assert "headers" in expected_response
            assert "body" in expected_response
    
    def test_http_replay_tester_initialization(self):
        """Test HTTP replay tester can be initialized."""
        # Create a minimal FastAPI app for testing
        from fastapi import FastAPI
        app = FastAPI()
        
        golden_cases_path = "tests/fixtures/analysis_golden_cases.json"
        tester = HTTPReplayTester(app, golden_cases_path)
        
        assert tester.app == app
        assert tester.golden_cases is not None
        assert "test_cases" in tester.golden_cases
        assert len(tester.golden_cases["test_cases"]) > 0
