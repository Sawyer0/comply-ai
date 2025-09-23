#!/usr/bin/env python3
"""
Script to generate golden test cases for Analysis Module HTTP replay testing.

This script creates comprehensive test cases covering various scenarios
for API conformance testing.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone


def generate_analysis_golden_cases() -> Dict[str, Any]:
    """Generate comprehensive golden test cases for Analysis Module API."""
    
    test_cases = []
    
    # Single Analysis Test Cases
    test_cases.extend(_generate_single_analysis_cases())
    
    # Batch Analysis Test Cases
    test_cases.extend(_generate_batch_analysis_cases())
    
    # Error Handling Test Cases
    test_cases.extend(_generate_error_cases())
    
    # Health Check Test Cases
    test_cases.extend(_generate_health_check_cases())
    
    # Edge Case Test Cases
    test_cases.extend(_generate_edge_cases())
    
    return {
        "version": "1.0",
        "description": "Golden test cases for Analysis Module API conformance testing",
        "metadata": {
            "created": datetime.now(timezone.utc).isoformat(),
            "analysis_module_version": "1.0.0",
            "total_cases": len(test_cases),
            "endpoints_covered": ["/analyze", "/analyze/batch", "/health/live", "/health/ready"],
            "test_categories": {
                "single_analysis": len([c for c in test_cases if "single_analysis" in c["id"]]),
                "batch_analysis": len([c for c in test_cases if "batch_analysis" in c["id"]]),
                "error_cases": len([c for c in test_cases if "error" in c["id"]]),
                "health_checks": len([c for c in test_cases if "health" in c["id"]]),
                "edge_cases": len([c for c in test_cases if "edge" in c["id"]])
            }
        },
        "test_cases": test_cases
    }


def _generate_single_analysis_cases() -> List[Dict[str, Any]]:
    """Generate single analysis test cases."""
    cases = []
    
    # Coverage gap scenarios
    coverage_scenarios = [
        {
            "id": "single_analysis_coverage_gap_severe",
            "description": "Single analysis - severe coverage gap",
            "observed_coverage": {"detector1": 0.3, "detector2": 0.4},
            "required_coverage": {"detector1": 0.8, "detector2": 0.9},
            "expected_reason": "severe coverage gap detected",
            "expected_remediation": "immediate detector deployment required"
        },
        {
            "id": "single_analysis_coverage_gap_moderate",
            "description": "Single analysis - moderate coverage gap",
            "observed_coverage": {"detector1": 0.6, "detector2": 0.7},
            "required_coverage": {"detector1": 0.8, "detector2": 0.9},
            "expected_reason": "coverage gap detected",
            "expected_remediation": "add secondary detector"
        },
        {
            "id": "single_analysis_coverage_adequate",
            "description": "Single analysis - adequate coverage",
            "observed_coverage": {"detector1": 0.9, "detector2": 0.95},
            "required_coverage": {"detector1": 0.8, "detector2": 0.9},
            "expected_reason": "coverage adequate",
            "expected_remediation": "maintain current setup"
        }
    ]
    
    for scenario in coverage_scenarios:
        case = {
            "id": scenario["id"],
            "endpoint": "/api/v1/analysis/analyze",
            "method": "POST",
            "description": scenario["description"],
            "request": {
                "headers": {
                    "Content-Type": "application/json",
                    "X-API-Key": "test-api-key-123",
                    "X-Tenant-ID": "tenant-001",
                    "X-Request-ID": f"req-{scenario['id']}"
                },
                "body": {
                    "period": "2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
                    "tenant": "tenant-001",
                    "app": "web-app",
                    "route": "/api/users",
                    "required_detectors": ["detector1", "detector2"],
                    "observed_coverage": scenario["observed_coverage"],
                    "required_coverage": scenario["required_coverage"],
                    "detector_errors": {"detector1": {"5xx": 0}, "detector2": {"5xx": 0}},
                    "high_sev_hits": [],
                    "false_positive_bands": [],
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
                    "reason": scenario["expected_reason"],
                    "remediation": scenario["expected_remediation"],
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
        cases.append(case)
    
    # False positive scenarios
    fp_scenarios = [
        {
            "id": "single_analysis_false_positive_high",
            "description": "Single analysis - high false positive rate",
            "fp_rate": 0.4,
            "expected_reason": "false positive rate high",
            "expected_remediation": "tune detector thresholds"
        },
        {
            "id": "single_analysis_false_positive_moderate",
            "description": "Single analysis - moderate false positive rate",
            "fp_rate": 0.2,
            "expected_reason": "false positive rate acceptable",
            "expected_remediation": "monitor for trends"
        }
    ]
    
    for scenario in fp_scenarios:
        case = {
            "id": scenario["id"],
            "endpoint": "/api/v1/analysis/analyze",
            "method": "POST",
            "description": scenario["description"],
            "request": {
                "headers": {
                    "Content-Type": "application/json",
                    "X-API-Key": "test-api-key-123",
                    "X-Tenant-ID": "tenant-001",
                    "X-Request-ID": f"req-{scenario['id']}"
                },
                "body": {
                    "period": "2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
                    "tenant": "tenant-001",
                    "app": "web-app",
                    "route": "/api/users",
                    "required_detectors": ["detector1"],
                    "observed_coverage": {"detector1": 0.9},
                    "required_coverage": {"detector1": 0.8},
                    "detector_errors": {"detector1": {"5xx": 0}},
                    "high_sev_hits": [],
                    "false_positive_bands": [{"detector": "detector1", "fp_rate": scenario["fp_rate"]}],
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
                    "reason": scenario["expected_reason"],
                    "remediation": scenario["expected_remediation"],
                    "confidence": 0.75,
                    "confidence_cutoff_used": 0.3,
                    "evidence_refs": ["false_positive_bands"],
                    "opa_diff": "package policy\n\nallow { input.fp_rate < 0.3 }",
                    "notes": "False positive analysis",
                    "version_info": {
                        "taxonomy": "v1.0",
                        "frameworks": "SOC2-v2.0",
                        "analyst_model": "phi3-mini-v1.0"
                    },
                    "processing_time_ms": 120
                }
            }
        }
        cases.append(case)
    
    return cases


def _generate_batch_analysis_cases() -> List[Dict[str, Any]]:
    """Generate batch analysis test cases."""
    cases = []
    
    # Successful batch processing
    cases.append({
        "id": "batch_analysis_success_mixed",
        "endpoint": "/api/v1/analysis/analyze/batch",
        "method": "POST",
        "description": "Batch analysis - mixed success scenarios",
        "request": {
            "headers": {
                "Content-Type": "application/json",
                "X-API-Key": "test-api-key-123",
                "X-Tenant-ID": "tenant-001",
                "X-Request-ID": "req-batch-001",
                "X-Idempotency-Key": "batch-mixed-001"
            },
            "body": {
                "requests": [
                    {
                        "period": "2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
                        "tenant": "tenant-001",
                        "app": "web-app",
                        "route": "/api/users",
                        "required_detectors": ["detector1"],
                        "observed_coverage": {"detector1": 0.9},
                        "required_coverage": {"detector1": 0.8},
                        "detector_errors": {"detector1": {"5xx": 0}},
                        "high_sev_hits": [],
                        "false_positive_bands": [],
                        "policy_bundle": "soc2-v1.0",
                        "env": "prod"
                    },
                    {
                        "period": "2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
                        "tenant": "tenant-001",
                        "app": "web-app",
                        "route": "/api/orders",
                        "required_detectors": ["detector2"],
                        "observed_coverage": {"detector2": 0.6},
                        "required_coverage": {"detector2": 0.8},
                        "detector_errors": {"detector2": {"5xx": 1}},
                        "high_sev_hits": [{"taxonomy": "PII", "score": 0.8}],
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
                    },
                    {
                        "status": "success",
                        "reason": "coverage gap detected",
                        "remediation": "add secondary detector",
                        "confidence": 0.85,
                        "confidence_cutoff_used": 0.3,
                        "evidence_refs": ["observed_coverage", "required_coverage"],
                        "opa_diff": "package policy\n\nallow { input.score > 0.7 }",
                        "notes": "Coverage below requirements",
                        "version_info": {
                            "taxonomy": "v1.0",
                            "frameworks": "SOC2-v2.0",
                            "analyst_model": "phi3-mini-v1.0"
                        },
                        "processing_time_ms": 150
                    }
                ],
                "batch_processing_time_ms": 300,
                "total_requests": 2,
                "successful_requests": 2,
                "failed_requests": 0
            }
        }
    })
    
    # Batch with partial failures
    cases.append({
        "id": "batch_analysis_partial_failure",
        "endpoint": "/api/v1/analysis/analyze/batch",
        "method": "POST",
        "description": "Batch analysis - partial failure scenario",
        "request": {
            "headers": {
                "Content-Type": "application/json",
                "X-API-Key": "test-api-key-123",
                "X-Tenant-ID": "tenant-001",
                "X-Request-ID": "req-batch-002",
                "X-Idempotency-Key": "batch-partial-002"
            },
            "body": {
                "requests": [
                    {
                        "period": "2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
                        "tenant": "tenant-001",
                        "app": "web-app",
                        "route": "/api/users",
                        "required_detectors": ["detector1"],
                        "observed_coverage": {"detector1": 0.9},
                        "required_coverage": {"detector1": 0.8},
                        "detector_errors": {"detector1": {"5xx": 0}},
                        "high_sev_hits": [],
                        "false_positive_bands": [],
                        "policy_bundle": "soc2-v1.0",
                        "env": "prod"
                    },
                    {
                        "period": "invalid-period",
                        "tenant": "tenant-001",
                        "app": "web-app",
                        "route": "/api/orders",
                        "required_detectors": ["detector2"],
                        "observed_coverage": {"detector2": 0.6},
                        "required_coverage": {"detector2": 0.8},
                        "detector_errors": {"detector2": {"5xx": 1}},
                        "high_sev_hits": [],
                        "false_positive_bands": [],
                        "policy_bundle": "soc2-v1.0",
                        "env": "invalid-env"
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
                    },
                    {
                        "status": "error",
                        "error_type": "validation_error",
                        "message": "Invalid request data",
                        "details": ["Invalid period format", "Invalid environment value"]
                    }
                ],
                "batch_processing_time_ms": 200,
                "total_requests": 2,
                "successful_requests": 1,
                "failed_requests": 1
            }
        }
    })
    
    return cases


def _generate_error_cases() -> List[Dict[str, Any]]:
    """Generate error handling test cases."""
    cases = []
    
    # Authentication errors
    cases.append({
        "id": "error_missing_api_key",
        "endpoint": "/api/v1/analysis/analyze",
        "method": "POST",
        "description": "Error - missing API key",
        "request": {
            "headers": {
                "Content-Type": "application/json",
                "X-Tenant-ID": "tenant-001"
            },
            "body": {
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
        },
        "expected_response": {
            "status_code": 401,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "error": "Unauthorized",
                "message": "API key required"
            }
        }
    })
    
    # Rate limiting
    cases.append({
        "id": "error_rate_limit_exceeded",
        "endpoint": "/api/v1/analysis/analyze",
        "method": "POST",
        "description": "Error - rate limit exceeded",
        "request": {
            "headers": {
                "Content-Type": "application/json",
                "X-API-Key": "test-api-key-123",
                "X-Tenant-ID": "tenant-001"
            },
            "body": {
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
        },
        "expected_response": {
            "status_code": 429,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "error": "Rate limit exceeded",
                "message": "Maximum 60 requests per minute allowed"
            }
        }
    })
    
    return cases


def _generate_health_check_cases() -> List[Dict[str, Any]]:
    """Generate health check test cases."""
    cases = []
    
    # Liveness probe
    cases.append({
        "id": "health_check_live",
        "endpoint": "/api/v1/analysis/health/live",
        "method": "GET",
        "description": "Health check - liveness probe",
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
    })
    
    # Readiness probe
    cases.append({
        "id": "health_check_ready",
        "endpoint": "/api/v1/analysis/health/ready",
        "method": "GET",
        "description": "Health check - readiness probe",
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
                "status": "ready",
                "timestamp": "2024-01-01T12:00:00Z",
                "service": "analysis-module",
                "version": "1.0.0",
                "dependencies": {
                    "model_server": "healthy",
                    "database": "healthy",
                    "cache": "healthy"
                }
            }
        }
    })
    
    return cases


def _generate_edge_cases() -> List[Dict[str, Any]]:
    """Generate edge case test cases."""
    cases = []
    
    # Empty batch request
    cases.append({
        "id": "edge_case_empty_batch",
        "endpoint": "/api/v1/analysis/analyze/batch",
        "method": "POST",
        "description": "Edge case - empty batch request",
        "request": {
            "headers": {
                "Content-Type": "application/json",
                "X-API-Key": "test-api-key-123",
                "X-Tenant-ID": "tenant-001",
                "X-Idempotency-Key": "batch-empty-001"
            },
            "body": {
                "requests": []
            }
        },
        "expected_response": {
            "status_code": 400,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "error": "Bad Request",
                "message": "Batch request cannot be empty"
            }
        }
    })
    
    # Large batch request
    cases.append({
        "id": "edge_case_large_batch",
        "endpoint": "/api/v1/analysis/analyze/batch",
        "method": "POST",
        "description": "Edge case - large batch request (101 items)",
        "request": {
            "headers": {
                "Content-Type": "application/json",
                "X-API-Key": "test-api-key-123",
                "X-Tenant-ID": "tenant-001",
                "X-Idempotency-Key": "batch-large-001"
            },
            "body": {
                "requests": [
                    {
                        "period": "2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
                        "tenant": "tenant-001",
                        "app": "web-app",
                        "route": f"/api/users/{i}",
                        "required_detectors": ["detector1"],
                        "observed_coverage": {"detector1": 0.8},
                        "required_coverage": {"detector1": 0.7},
                        "detector_errors": {"detector1": {"5xx": 0}},
                        "high_sev_hits": [],
                        "false_positive_bands": [],
                        "policy_bundle": "soc2-v1.0",
                        "env": "prod"
                    }
                    for i in range(101)
                ]
            }
        },
        "expected_response": {
            "status_code": 400,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "error": "Bad Request",
                "message": "Batch size exceeds maximum of 100 items"
            }
        }
    })
    
    return cases


def main():
    """Main function to generate golden test cases."""
    parser = argparse.ArgumentParser(description="Generate Analysis Module golden test cases")
    parser.add_argument(
        "-o", "--output",
        default="tests/fixtures/analysis_golden_cases.json",
        help="Output file path for golden test cases"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON output"
    )
    
    args = parser.parse_args()
    
    # Generate golden test cases
    golden_cases = generate_analysis_golden_cases()
    
    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if args.pretty:
            json.dump(golden_cases, f, indent=2, sort_keys=True)
        else:
            json.dump(golden_cases, f)
    
    print(f"Generated {golden_cases['metadata']['total_cases']} golden test cases")
    print(f"Saved to: {output_path}")
    
    # Print summary
    categories = golden_cases['metadata']['test_categories']
    print("\nTest categories:")
    for category, count in categories.items():
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
