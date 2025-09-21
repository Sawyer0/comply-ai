#!/usr/bin/env python3
"""
Mock detector service for testing detector orchestration.

This service simulates various detector behaviors including:
- Variable response times
- Configurable success/failure rates
- Timeout scenarios
- Different capabilities and content type support
"""

import asyncio
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

# Configuration from environment variables
DETECTOR_NAME = os.getenv("DETECTOR_NAME", "mock-detector")
SUCCESS_RATE = float(os.getenv("SUCCESS_RATE", "0.95"))  # 95% success rate
BASE_RESPONSE_TIME_MS = int(os.getenv("BASE_RESPONSE_TIME_MS", "200"))
MAX_RESPONSE_TIME_MS = int(os.getenv("MAX_RESPONSE_TIME_MS", "1000"))
MOCK_CAPABILITIES = os.getenv("MOCK_CAPABILITIES", "detection").split(",")
TIMEOUT_PROBABILITY = float(os.getenv("TIMEOUT_PROBABILITY", "0.0"))
BATCH_MODE = os.getenv("BATCH_MODE", "false").lower() == "true"

# Request counter for tracking
request_count = 0
error_count = 0

app = FastAPI(
    title=f"Mock Detector - {DETECTOR_NAME}",
    description="Mock detector service for testing orchestration",
    version="1.0.0"
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global request_count, error_count

    return {
        "status": "healthy",
        "detector": DETECTOR_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "requests_processed": request_count,
        "errors": error_count,
        "success_rate": (request_count - error_count) / max(request_count, 1),
        "capabilities": MOCK_CAPABILITIES
    }

@app.get("/")
async def root():
    """Root endpoint with detector information"""
    return {
        "name": DETECTOR_NAME,
        "capabilities": MOCK_CAPABILITIES,
        "supported_content_types": ["text", "document", "code"],
        "health": "ok"
    }

@app.get("/capabilities")
async def get_capabilities():
    """Get detector capabilities"""
    return {
        "detector": DETECTOR_NAME,
        "capabilities": MOCK_CAPABILITIES,
        "supported_content_types": ["text", "document", "code"],
        "timeout_ms": BASE_RESPONSE_TIME_MS,
        "version": "1.0.0"
    }

def generate_mock_detection(content: str, content_type: str) -> Dict[str, Any]:
    """Generate mock detection results based on content analysis"""
    results = []

    # Simulate different detection types based on capabilities
    if "toxicity_detection" in MOCK_CAPABILITIES:
        if any(word in content.lower() for word in ["hate", "violent", "attack", "threat", "abuse"]):
            results.append({
                "category": "toxicity",
                "subcategory": "hate_speech",
                "confidence": round(random.uniform(0.8, 0.95), 3),
                "severity": "high"
            })

    if "pii_detection" in MOCK_CAPABILITIES:
        pii_patterns = ["email", "phone", "ssn", "address", "credit_card"]
        for pattern in pii_patterns:
            if pattern.replace("_", " ") in content.lower():
                results.append({
                    "category": "pii",
                    "subcategory": pattern,
                    "confidence": round(random.uniform(0.9, 0.99), 3),
                    "severity": "medium"
                })

    if "security_scan" in MOCK_CAPABILITIES:
        security_keywords = ["vulnerability", "exploit", "injection", "xss", "csrf"]
        for keyword in security_keywords:
            if keyword in content.lower():
                results.append({
                    "category": "security",
                    "subcategory": keyword,
                    "confidence": round(random.uniform(0.85, 0.98), 3),
                    "severity": "high"
                })

    if "code_analysis" in MOCK_CAPABILITIES and content_type == "code":
        code_issues = ["sql_injection", "hardcoded_secret", "unsafe_deserialization"]
        for issue in code_issues:
            if issue.replace("_", " ") in content.lower():
                results.append({
                    "category": "code_security",
                    "subcategory": issue,
                    "confidence": round(random.uniform(0.9, 0.99), 3),
                    "severity": "critical"
                })

    # Always add some baseline detection
    if not results:
        results.append({
            "category": "general",
            "subcategory": "content_analysis",
            "confidence": round(random.uniform(0.1, 0.3), 3),
            "severity": "low"
        })

    return results

async def simulate_processing_time() -> float:
    """Simulate variable processing time"""
    global request_count

    # Base processing time
    processing_time = BASE_RESPONSE_TIME_MS / 1000.0

    # Add some randomness
    if MAX_RESPONSE_TIME_MS > BASE_RESPONSE_TIME_MS:
        additional_time = random.uniform(0, (MAX_RESPONSE_TIME_MS - BASE_RESPONSE_TIME_MS) / 1000.0)
        processing_time += additional_time

    # Simulate occasional slow responses
    if random.random() < 0.1:  # 10% slow responses
        processing_time *= random.uniform(2, 5)

    # Simulate occasional timeouts
    if random.random() < TIMEOUT_PROBABILITY:
        processing_time = 30.0  # 30 second timeout

    return processing_time

async def simulate_failure() -> bool:
    """Determine if this request should fail"""
    global error_count

    # Random failure based on success rate
    if random.random() > SUCCESS_RATE:
        error_count += 1
        return True

    # Additional failure scenarios
    if random.random() < 0.02:  # 2% random failures
        error_count += 1
        return True

    return False

@app.post("/detect")
async def detect_content(request: Dict[str, Any]):
    """Main detection endpoint"""
    global request_count

    request_count += 1

    try:
        # Extract content from request
        content = request.get("content", "")
        content_type = request.get("content_type", "text")
        metadata = request.get("metadata", {})

        # Simulate processing time
        processing_time = await simulate_processing_time()

        # Simulate failure
        if await simulate_failure():
            # Random error types
            error_types = ["internal_error", "timeout", "invalid_input", "service_unavailable"]
            error_type = random.choice(error_types)

            raise HTTPException(
                status_code=500 if error_type == "internal_error" else 408 if error_type == "timeout" else 400,
                detail=f"Mock detector {error_type}: {DETECTOR_NAME}"
            )

        # Wait for simulated processing time
        await asyncio.sleep(processing_time)

        # Generate mock detection results
        detections = generate_mock_detection(content, content_type)

        # Calculate overall confidence
        overall_confidence = sum(d["confidence"] for d in detections) / len(detections) if detections else 0.0

        response = {
            "detector": DETECTOR_NAME,
            "version": "1.0.0",
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": int(processing_time * 1000),
            "content_type": content_type,
            "detections": detections,
            "overall_confidence": round(overall_confidence, 3),
            "total_findings": len(detections),
            "metadata": {
                **metadata,
                "mock_detector": True,
                "simulated_processing_time_ms": int(processing_time * 1000)
            }
        }

        return JSONResponse(status_code=200, content=response)

    except HTTPException:
        raise
    except Exception as e:
        error_count += 1
        raise HTTPException(status_code=500, detail=f"Unexpected error in {DETECTOR_NAME}: {str(e)}")

@app.post("/detect/batch")
async def detect_batch(request: List[Dict[str, Any]]):
    """Batch detection endpoint"""
    global request_count

    if not BATCH_MODE:
        raise HTTPException(status_code=400, detail="Batch mode not enabled for this detector")

    results = []

    for item in request:
        request_count += 1

        try:
            content = item.get("content", "")
            content_type = item.get("content_type", "text")
            metadata = item.get("metadata", {})

            # Simulate processing time
            processing_time = await simulate_processing_time()

            # Simulate failure
            if await simulate_failure():
                error_types = ["internal_error", "timeout", "invalid_input"]
                error_type = random.choice(error_types)

                results.append({
                    "status": "error",
                    "error": f"Mock detector {error_type}: {DETECTOR_NAME}",
                    "detector": DETECTOR_NAME,
                    "request_id": str(uuid.uuid4()),
                    "content_type": content_type
                })
                error_count += 1
                continue

            # Wait for simulated processing time
            await asyncio.sleep(processing_time)

            # Generate mock detection results
            detections = generate_mock_detection(content, content_type)

            results.append({
                "status": "success",
                "detector": DETECTOR_NAME,
                "version": "1.0.0",
                "request_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_time_ms": int(processing_time * 1000),
                "content_type": content_type,
                "detections": detections,
                "total_findings": len(detections),
                "metadata": {
                    **metadata,
                    "mock_detector": True,
                    "batch_item": True
                }
            })

        except Exception as e:
            error_count += 1
            results.append({
                "status": "error",
                "error": f"Exception in {DETECTOR_NAME}: {str(e)}",
                "detector": DETECTOR_NAME,
                "request_id": str(uuid.uuid4())
            })

    return {
        "detector": DETECTOR_NAME,
        "batch_size": len(request),
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_mode": True
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "detector": DETECTOR_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detector": DETECTOR_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    print(f"🚀 Starting mock detector: {DETECTOR_NAME}")
    print(f"   Success rate: {SUCCESS_RATE}")
    print(f"   Response time: {BASE_RESPONSE_TIME_MS}ms - {MAX_RESPONSE_TIME_MS}ms")
    print(f"   Capabilities: {', '.join(MOCK_CAPABILITIES)}")
    print(f"   Timeout probability: {TIMEOUT_PROBABILITY}")
    print(f"   Batch mode: {BATCH_MODE}")

    uvicorn.run(
        "mock_detector:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
