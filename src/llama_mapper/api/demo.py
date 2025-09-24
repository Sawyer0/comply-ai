"""
Demo endpoints for investor demonstrations.

These endpoints provide working examples of the system's capabilities
without requiring full model deployment.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from datetime import datetime
import random

router = APIRouter(prefix="/demo", tags=["demo"])


@router.get("/health")
async def demo_health() -> Dict[str, Any]:
    """Demo health check showing system status."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "operational",
            "model_server": "ready",
            "database": "connected",
            "cache": "operational"
        },
        "metrics": {
            "requests_per_second": random.randint(50, 200),
            "avg_response_time_ms": random.randint(45, 120),
            "success_rate": round(random.uniform(0.95, 0.99), 3)
        }
    }


@router.post("/map")
async def demo_mapping(request: Dict[str, Any]) -> Dict[str, Any]:
    """Demo mapping endpoint showing detector output normalization."""
    
    detector = request.get("detector", "unknown")
    output = request.get("output", "")
    
    # Simulate realistic mapping - supports ALL major enterprise AI safety tools
    mapping_examples = {
        # Microsoft tools
        "presidio": {
            "PERSON": "PII.Person.Name",
            "EMAIL_ADDRESS": "PII.Contact.Email", 
            "PHONE_NUMBER": "PII.Contact.Phone",
            "US_SSN": "PII.Identifier.SSN"
        },
        # Academic/Open Source
        "deberta": {
            "toxic": "HARM.SPEECH.Toxicity",
            "severe_toxic": "HARM.SPEECH.Toxicity",
            "obscene": "HARM.SPEECH.Obscenity",
            "threat": "HARM.SPEECH.Threat",
            "insult": "HARM.SPEECH.Insult"
        },
        # Meta AI Safety
        "llama-guard": {
            "violence": "HARM.VIOLENCE.Physical",
            "hate": "HARM.SPEECH.Hate.Other",
            "harassment": "HARM.SPEECH.Harassment",
            "self-harm": "HARM.VIOLENCE.SelfHarm"
        },
        # Amazon Bedrock Guardrails
        "amazon-bedrock-guardrails": {
            "HATE": "HARM.SPEECH.Hate.Other",
            "VIOLENCE": "HARM.VIOLENCE.Physical",
            "PERSON_NAME": "PII.Person.Name",
            "PROMPT_INJECTION": "PROMPT_INJECTION.Other"
        },
        # Microsoft Azure Content Safety
        "azure-content-safety": {
            "Hate": "HARM.SPEECH.Hate.Other",
            "Violence": "HARM.VIOLENCE.Physical",
            "Sexual": "HARM.VIOLENCE.Sexual",
            "SelfHarm": "HARM.VIOLENCE.SelfHarm"
        },
        # Google Cloud DLP
        "google-cloud-dlp": {
            "PERSON_NAME": "PII.Person.Name",
            "EMAIL_ADDRESS": "PII.Contact.Email",
            "US_SOCIAL_SECURITY_NUMBER": "PII.Identifier.SSN",
            "CREDIT_CARD_NUMBER": "PII.Identifier.CreditCard"
        },
        # NVIDIA NeMo Guardrails
        "nvidia-nemo-guardrails": {
            "jailbreak": "JAILBREAK.Attempt",
            "harmful_content": "HARM.SPEECH.Toxicity",
            "prompt_injection": "PROMPT_INJECTION.Other",
            "hallucination": "OTHER.Hallucination"
        },
        # Anthropic Claude Safety
        "anthropic-claude-safety": {
            "harmful": "HARM.SPEECH.Toxicity",
            "jailbreak_attempt": "JAILBREAK.Attempt",
            "misinformation": "OTHER.Misinformation",
            "privacy_violation": "PII.Other"
        },
        # OpenAI Moderation
        "openai-moderation": {
            "hate": "HARM.SPEECH.Hate.Other",
            "violence": "HARM.VIOLENCE.Physical",
            "self-harm": "HARM.VIOLENCE.SelfHarm",
            "sexual/minors": "HARM.VIOLENCE.ChildSexualAbuse"
        }
    }
    
    # Get mapping or default
    detector_mappings = mapping_examples.get(detector, {})
    canonical_label = detector_mappings.get(output, "OTHER.Unknown")
    
    # Generate realistic confidence score
    confidence = round(random.uniform(0.75, 0.95), 2)
    
    # Create compliance framework mappings
    framework_mappings = []
    if "PII" in canonical_label:
        framework_mappings = [
            {"framework": "GDPR", "article": "Article 4", "requirement": "Personal data processing"},
            {"framework": "HIPAA", "section": "164.514", "requirement": "De-identification"},
            {"framework": "SOC2", "control": "CC6.1", "requirement": "Logical access controls"}
        ]
    elif "HARM" in canonical_label:
        framework_mappings = [
            {"framework": "SOC2", "control": "CC7.2", "requirement": "System monitoring"},
            {"framework": "ISO27001", "control": "A.12.4.1", "requirement": "Event logging"}
        ]
    
    return {
        "taxonomy": [canonical_label],
        "scores": {canonical_label: confidence},
        "confidence": confidence,
        "provenance": {
            "detector": detector,
            "detector_version": "v1.0",
            "model_version": "llama-mapper-v1.2.0",
            "timestamp": datetime.now().isoformat()
        },
        "framework_mappings": framework_mappings,
        "version_info": {
            "taxonomy": "2024.12",
            "frameworks": "v1.0", 
            "model": "llama-mapper-v1.2.0"
        },
        "notes": f"Mapped {detector} output '{output}' to canonical taxonomy with {confidence} confidence"
    }


@router.get("/compliance-report")
async def demo_compliance_report(framework: str = "SOC2") -> Dict[str, Any]:
    """Demo compliance report generation."""
    
    # Generate realistic compliance data
    total_detections = random.randint(1000, 5000)
    compliant_detections = int(total_detections * random.uniform(0.92, 0.98))
    
    violations = [
        {
            "type": "PII.Contact.Email",
            "count": random.randint(5, 25),
            "severity": "medium",
            "last_occurrence": datetime.now().isoformat()
        },
        {
            "type": "HARM.SPEECH.Toxicity", 
            "count": random.randint(2, 10),
            "severity": "high",
            "last_occurrence": datetime.now().isoformat()
        }
    ]
    
    controls = {
        "SOC2": [
            {"control": "CC6.1", "status": "compliant", "coverage": "98%"},
            {"control": "CC7.2", "status": "compliant", "coverage": "95%"},
            {"control": "CC6.7", "status": "minor_gap", "coverage": "92%"}
        ],
        "GDPR": [
            {"article": "Article 4", "status": "compliant", "coverage": "97%"},
            {"article": "Article 25", "status": "compliant", "coverage": "94%"},
            {"article": "Article 32", "status": "compliant", "coverage": "96%"}
        ],
        "HIPAA": [
            {"section": "164.514", "status": "compliant", "coverage": "99%"},
            {"section": "164.308", "status": "compliant", "coverage": "93%"}
        ]
    }
    
    return {
        "framework": framework,
        "report_period": "2024-12-01 to 2024-12-31",
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_detections": total_detections,
            "compliant_detections": compliant_detections,
            "compliance_rate": round(compliant_detections / total_detections, 3),
            "violations_count": len(violations)
        },
        "controls": controls.get(framework, controls["SOC2"]),
        "violations": violations,
        "recommendations": [
            "Implement additional PII detection rules for email patterns",
            "Review toxicity detection thresholds for false positives",
            "Schedule quarterly compliance assessment review"
        ],
        "audit_trail": {
            "taxonomy_version": "2024.12",
            "model_version": "llama-mapper-v1.2.0",
            "framework_version": "v1.0"
        }
    }


@router.get("/metrics")
async def demo_metrics() -> Dict[str, Any]:
    """Demo system metrics for monitoring dashboard."""
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system_health": {
            "status": "healthy",
            "uptime_hours": random.randint(100, 1000),
            "cpu_usage": round(random.uniform(0.2, 0.6), 2),
            "memory_usage": round(random.uniform(0.4, 0.8), 2),
            "gpu_usage": round(random.uniform(0.7, 0.9), 2)
        },
        "api_metrics": {
            "requests_per_second": random.randint(50, 200),
            "avg_response_time_ms": random.randint(45, 120),
            "p95_response_time_ms": random.randint(150, 300),
            "success_rate": round(random.uniform(0.95, 0.99), 3),
            "error_rate": round(random.uniform(0.01, 0.05), 3)
        },
        "model_metrics": {
            "schema_validation_rate": round(random.uniform(0.95, 0.99), 3),
            "confidence_avg": round(random.uniform(0.8, 0.9), 2),
            "fallback_rate": round(random.uniform(0.05, 0.15), 3),
            "model_accuracy": round(random.uniform(0.92, 0.97), 3)
        },
        "business_metrics": {
            "total_mappings_today": random.randint(5000, 15000),
            "unique_tenants_active": random.randint(15, 50),
            "compliance_frameworks_used": ["SOC2", "GDPR", "HIPAA", "ISO27001"],
            "detector_types_active": ["presidio", "deberta", "llama-guard", "custom"]
        }
    }