#!/usr/bin/env python3
"""
Test script to verify all microservice endpoints are working.
"""

import requests
import json
import time


def test_endpoint(method, url, data=None, description=""):
    """Test a single endpoint"""
    print(f"\n🧪 Testing: {description}")
    print(f"   {method} {url}")

    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
        elif method.upper() == "POST":
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=data, headers=headers, timeout=10)
        else:
            print(f"   ❌ Unsupported method: {method}")
            return False

        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            print(f"   ✅ SUCCESS")
            # Print first 200 chars of response
            response_text = response.text[:200]
            if len(response.text) > 200:
                response_text += "..."
            print(f"   Response: {response_text}")
            return True
        else:
            print(f"   ❌ FAILED")
            print(f"   Error: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"   ❌ CONNECTION ERROR - Service not running?")
        return False
    except requests.exceptions.Timeout:
        print(f"   ❌ TIMEOUT")
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False


def main():
    """Test all endpoints"""
    print("🚀 Testing Microservice Endpoints")
    print("=" * 50)

    # Test data
    orchestrate_data = {
        "content": "John Doe's email is john.doe@example.com",
        "detector_types": ["presidio", "deberta"],
        "tenant_id": "test-tenant",
    }

    analyze_data = {
        "detector_results": [
            {
                "detector_id": "presidio",
                "detector_type": "pii",
                "findings": [
                    {
                        "type": "PII.Contact.Email",
                        "confidence": 0.95,
                        "location": {"start": 0, "end": 20},
                    }
                ],
                "confidence": 0.95,
            }
        ],
        "analysis_type": "risk_assessment",
    }

    map_data = {
        "detector_outputs": [
            {
                "detector_id": "presidio",
                "detector_type": "pii",
                "findings": [
                    {
                        "type": "EMAIL_ADDRESS",
                        "confidence": 0.95,
                        "location": {"start": 0, "end": 20},
                    }
                ],
                "confidence": 0.95,
            }
        ],
        "target_framework": "soc2",
    }

    # Test cases
    tests = [
        # Health checks first
        ("GET", "http://localhost:8000/health", None, "Detector Orchestration Health"),
        ("GET", "http://localhost:8001/health", None, "Analysis Service Health"),
        ("GET", "http://localhost:8002/health", None, "Mapper Service Health"),
        # Root endpoints
        ("GET", "http://localhost:8000/", None, "Detector Orchestration Root"),
        ("GET", "http://localhost:8001/", None, "Analysis Service Root"),
        ("GET", "http://localhost:8002/", None, "Mapper Service Root"),
        # GET endpoints
        ("GET", "http://localhost:8000/api/v1/detectors", None, "List Detectors"),
        ("GET", "http://localhost:8002/api/v1/taxonomy", None, "Get Taxonomy"),
        ("GET", "http://localhost:8002/api/v1/frameworks", None, "List Frameworks"),
        (
            "GET",
            "http://localhost:8001/api/v1/quality/metrics",
            None,
            "Quality Metrics",
        ),
        # POST endpoints
        (
            "POST",
            "http://localhost:8000/api/v1/orchestrate",
            orchestrate_data,
            "Orchestrate Detectors",
        ),
        (
            "POST",
            "http://localhost:8001/api/v1/analyze",
            analyze_data,
            "Analyze Results",
        ),
        ("POST", "http://localhost:8002/api/v1/map", map_data, "Map to Taxonomy"),
    ]

    passed = 0
    total = len(tests)

    for method, url, data, description in tests:
        success = test_endpoint(method, url, data, description)
        if success:
            passed += 1
        time.sleep(0.5)  # Small delay between tests

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All endpoints working correctly!")
    else:
        print("⚠️  Some endpoints failed - check service logs")

    print("\n💡 If you got 'Method Not Allowed' errors:")
    print("   • Make sure you're using POST for POST endpoints")
    print("   • Check that the services are actually running")
    print("   • Try the FastAPI docs: http://localhost:800X/docs")


if __name__ == "__main__":
    main()
