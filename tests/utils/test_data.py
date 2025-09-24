"""
Test data factory for generating consistent test data across all services.

This module provides utilities for creating test data that works across:
- Core Mapper Service
- Detector Orchestration Service
- Analysis Service
"""

import uuid
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class DetectorType(Enum):
    """Supported detector types."""
    PRESIDIO = "presidio"
    DEBERTA = "deberta"
    CUSTOM = "custom"
    SPACY = "spacy"
    TRANSFORMERS = "transformers"


class EntityType(Enum):
    """Supported entity types."""
    PERSON = "PERSON"
    PHONE_NUMBER = "PHONE_NUMBER"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    CREDIT_CARD = "CREDIT_CARD"
    SSN = "SSN"
    IP_ADDRESS = "IP_ADDRESS"
    DATE_TIME = "DATE_TIME"
    LOCATION = "LOCATION"
    ORGANIZATION = "ORGANIZATION"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    SOC2 = "SOC2"
    ISO27001 = "ISO27001"
    HIPAA = "HIPAA"
    GDPR = "GDPR"
    CCPA = "CCPA"


@dataclass
class TestDataConfig:
    """Configuration for test data generation."""
    use_synthetic: bool = True
    privacy_scrubbing: bool = True
    tenant_isolation: bool = True
    deterministic_seed: Optional[int] = None
    default_confidence: float = 0.95
    default_framework: ComplianceFramework = ComplianceFramework.SOC2


class TestDataFactory:
    """Factory for generating consistent test data."""
    
    def __init__(self, 
                 use_synthetic: bool = True,
                 privacy_scrubbing: bool = True,
                 seed: Optional[int] = None):
        self.config = TestDataConfig(
            use_synthetic=use_synthetic,
            privacy_scrubbing=privacy_scrubbing,
            deterministic_seed=seed
        )
        
        if seed:
            random.seed(seed)
        
        # Privacy-safe synthetic data
        self.synthetic_names = [
            "John Doe", "Jane Smith", "Bob Johnson", "Alice Brown",
            "Charlie Davis", "Diana Wilson", "Eve Miller", "Frank Garcia"
        ]
        
        self.synthetic_emails = [
            "test.user@example.com", "sample@test.org", "demo@example.net",
            "user123@example.com", "testdata@sample.org"
        ]
        
        self.synthetic_phones = [
            "555-0123", "555-0456", "555-0789", "555-0012", "555-0345"
        ]
        
        self.synthetic_content_templates = [
            "Hello {name}, your account {email} has been updated.",
            "Contact {name} at {phone} for more information.",
            "User {name} with email {email} registered successfully.",
            "Please verify your identity with SSN {ssn}.",
            "Payment processed for card ending in {credit_card}."
        ]
    
    def create_detector_output(self, 
                             detector_type: str = None,
                             confidence: float = None,
                             findings: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a detector output for testing."""
        if detector_type is None:
            detector_type = random.choice(list(DetectorType)).value
        
        if confidence is None:
            confidence = self.config.default_confidence
        
        if findings is None:
            findings = [self._create_random_finding()]
        
        return {
            "detector_type": detector_type,
            "confidence": confidence,
            "findings": findings,
            "metadata": {
                "processing_time_ms": random.randint(10, 100),
                "model_version": f"{detector_type}-v1.0.{random.randint(0, 9)}",
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": str(uuid.uuid4())
            }
        }
    
    def create_mapping_payload(self,
                             detector_outputs: List[Dict[str, Any]] = None,
                             framework: str = None,
                             tenant_id: str = None) -> Dict[str, Any]:
        """Create a mapping payload for testing."""
        if detector_outputs is None:
            detector_outputs = [
                self.create_detector_output(DetectorType.PRESIDIO.value),
                self.create_detector_output(DetectorType.DEBERTA.value)
            ]
        
        if framework is None:
            framework = self.config.default_framework.value
        
        if tenant_id is None:
            tenant_id = f"test_tenant_{random.randint(1000, 9999)}"
        
        return {
            "detector_outputs": detector_outputs,
            "framework": framework,
            "tenant_id": tenant_id,
            "correlation_id": str(uuid.uuid4()),
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "source": "test_data_factory",
                "version": "1.0.0"
            }
        }
    
    def create_tenant_data(self, tenant_id: str) -> Dict[str, Any]:
        """Create tenant-specific test data with isolation."""
        return {
            "tenant_id": tenant_id,
            "tenant_config": {
                "name": f"Test Tenant {tenant_id}",
                "compliance_frameworks": [f.value for f in ComplianceFramework],
                "privacy_settings": {
                    "data_retention_days": 90,
                    "pii_redaction": True,
                    "audit_logging": True
                }
            },
            "test_scenarios": self._create_tenant_test_scenarios(tenant_id),
            "expected_results": self._create_tenant_expected_results(tenant_id)
        }
    
    def create_cross_service_test_data(self, scenario_name: str) -> Dict[str, Any]:
        """Create test data for cross-service scenarios."""
        scenarios = {
            "detection_to_analysis_workflow": self._create_detection_workflow_data(),
            "batch_processing_workflow": self._create_batch_processing_data(),
            "chaos_testing_scenario": self._create_chaos_testing_data(),
            "performance_testing_scenario": self._create_performance_testing_data()
        }
        
        return scenarios.get(scenario_name, {})
    
    def create_golden_dataset(self, dataset_name: str, size: int = 100) -> List[Dict[str, Any]]:
        """Create golden dataset for regression testing."""
        dataset = []
        
        for i in range(size):
            # Create deterministic data if seed is set
            if self.config.deterministic_seed:
                random.seed(self.config.deterministic_seed + i)
            
            data_point = {
                "id": f"{dataset_name}_{i:04d}",
                "input": self.create_mapping_payload(),
                "expected_output": self._create_expected_mapping_result(),
                "metadata": {
                    "dataset": dataset_name,
                    "index": i,
                    "created_at": datetime.utcnow().isoformat()
                }
            }
            
            dataset.append(data_point)
        
        return dataset
    
    def create_security_test_payloads(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create payloads for security testing."""
        return {
            "sql_injection": [
                self._create_malicious_payload("sql_injection", "'; DROP TABLE users; --"),
                self._create_malicious_payload("sql_injection", "1' OR '1'='1"),
                self._create_malicious_payload("sql_injection", "admin'--")
            ],
            "xss": [
                self._create_malicious_payload("xss", "<script>alert('xss')</script>"),
                self._create_malicious_payload("xss", "javascript:alert('xss')"),
                self._create_malicious_payload("xss", "<img src=x onerror=alert('xss')>")
            ],
            "path_traversal": [
                self._create_malicious_payload("path_traversal", "../../etc/passwd"),
                self._create_malicious_payload("path_traversal", "..\\..\\windows\\system32\\config\\sam"),
                self._create_malicious_payload("path_traversal", "/etc/hosts")
            ],
            "overflow": [
                self._create_malicious_payload("overflow", "A" * 10000),
                self._create_malicious_payload("overflow", "B" * 100000)
            ]
        }
    
    def create_performance_test_data(self, 
                                   batch_size: int = 100,
                                   complexity: str = "medium") -> List[Dict[str, Any]]:
        """Create data for performance testing."""
        complexity_configs = {
            "simple": {
                "detector_count": 1,
                "findings_count": 1,
                "content_length": 100
            },
            "medium": {
                "detector_count": 3,
                "findings_count": 5,
                "content_length": 1000
            },
            "complex": {
                "detector_count": 5,
                "findings_count": 20,
                "content_length": 10000
            }
        }
        
        config = complexity_configs.get(complexity, complexity_configs["medium"])
        test_data = []
        
        for i in range(batch_size):
            detector_outputs = []
            for j in range(config["detector_count"]):
                findings = [
                    self._create_random_finding() 
                    for _ in range(config["findings_count"])
                ]
                detector_outputs.append(
                    self.create_detector_output(
                        detector_type=list(DetectorType)[j % len(DetectorType)].value,
                        findings=findings
                    )
                )
            
            payload = self.create_mapping_payload(detector_outputs=detector_outputs)
            payload["performance_metadata"] = {
                "batch_index": i,
                "complexity": complexity,
                "expected_processing_time_ms": config["detector_count"] * 50
            }
            
            test_data.append(payload)
        
        return test_data
    
    def _create_random_finding(self) -> Dict[str, Any]:
        """Create a random finding for testing."""
        entity_type = random.choice(list(EntityType))
        
        # Generate appropriate synthetic data based on entity type
        if entity_type == EntityType.PERSON:
            text = random.choice(self.synthetic_names)
        elif entity_type == EntityType.EMAIL_ADDRESS:
            text = random.choice(self.synthetic_emails)
        elif entity_type == EntityType.PHONE_NUMBER:
            text = random.choice(self.synthetic_phones)
        elif entity_type == EntityType.SSN:
            text = "XXX-XX-XXXX"  # Always redacted
        elif entity_type == EntityType.CREDIT_CARD:
            text = "**** **** **** 1234"  # Always masked
        else:
            text = "[REDACTED]"
        
        # Apply privacy scrubbing if enabled
        if self.config.privacy_scrubbing:
            text = "[REDACTED]"
        
        return {
            "entity_type": entity_type.value,
            "confidence": random.uniform(0.8, 0.99),
            "start": 0,
            "end": len(text),
            "text": text
        }
    
    def _create_tenant_test_scenarios(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Create test scenarios for a specific tenant."""
        return [
            {
                "scenario_name": "pii_detection",
                "input_data": self.create_mapping_payload(tenant_id=tenant_id),
                "expected_framework": ComplianceFramework.SOC2.value
            },
            {
                "scenario_name": "security_analysis",
                "input_data": self.create_mapping_payload(
                    framework=ComplianceFramework.ISO27001.value,
                    tenant_id=tenant_id
                ),
                "expected_framework": ComplianceFramework.ISO27001.value
            }
        ]
    
    def _create_tenant_expected_results(self, tenant_id: str) -> Dict[str, Any]:
        """Create expected results for tenant scenarios."""
        return {
            "pii_detection": {
                "canonical_result": {
                    "category": "pii",
                    "subcategory": "person_name",
                    "confidence": 0.95
                }
            },
            "security_analysis": {
                "canonical_result": {
                    "category": "security",
                    "subcategory": "credentials",
                    "confidence": 0.90
                }
            }
        }
    
    def _create_detection_workflow_data(self) -> Dict[str, Any]:
        """Create data for detection to analysis workflow."""
        return {
            "workflow_type": "detection_to_analysis",
            "orchestration_payload": {
                "content": "Hello John Doe, please contact us at john@example.com",
                "detectors": ["presidio", "deberta"],
                "framework": "SOC2",
                "tenant_id": "workflow_test_tenant",
                "auto_map": True,
                "correlation_id": str(uuid.uuid4())
            },
            "expected_stages": [
                "orchestration_processing",
                "detector_execution",
                "result_aggregation",
                "mapping_execution", 
                "analysis_processing"
            ]
        }
    
    def _create_batch_processing_data(self) -> Dict[str, Any]:
        """Create data for batch processing workflow."""
        batch_size = 50
        batch_items = []
        
        for i in range(batch_size):
            batch_items.append({
                "item_id": f"batch_item_{i:03d}",
                "content": f"Sample content {i} with PII data",
                "metadata": {"batch_index": i}
            })
        
        return {
            "workflow_type": "batch_processing", 
            "batch_data": {
                "items": batch_items,
                "batch_size": batch_size,
                "processing_mode": "parallel",
                "tenant_id": "batch_test_tenant"
            }
        }
    
    def _create_chaos_testing_data(self) -> Dict[str, Any]:
        """Create data for chaos testing scenarios."""
        return {
            "workflow_type": "chaos_testing",
            "chaos_scenarios": [
                {
                    "name": "mapper_service_failure",
                    "target": "core_mapper",
                    "failure_type": "service_crash",
                    "test_payload": self.create_mapping_payload(),
                    "expected_behavior": "fallback_to_rules"
                },
                {
                    "name": "orchestration_timeout",
                    "target": "detector_orchestration", 
                    "failure_type": "timeout",
                    "test_payload": {
                        "content": "Test content for timeout scenario",
                        "detectors": ["presidio"],
                        "timeout_ms": 1000
                    },
                    "expected_behavior": "timeout_handling"
                }
            ]
        }
    
    def _create_performance_testing_data(self) -> Dict[str, Any]:
        """Create data for performance testing scenarios."""
        return {
            "workflow_type": "performance_testing",
            "load_test_configs": [
                {
                    "name": "baseline_load",
                    "concurrent_users": 10,
                    "duration_seconds": 30,
                    "test_data": self.create_performance_test_data(100, "simple")
                },
                {
                    "name": "stress_test",
                    "concurrent_users": 100,
                    "duration_seconds": 60,
                    "test_data": self.create_performance_test_data(1000, "medium")
                },
                {
                    "name": "endurance_test",
                    "concurrent_users": 50,
                    "duration_seconds": 300,
                    "test_data": self.create_performance_test_data(500, "complex")
                }
            ]
        }
    
    def _create_expected_mapping_result(self) -> Dict[str, Any]:
        """Create expected mapping result for golden datasets."""
        return {
            "canonical_result": {
                "category": "pii",
                "subcategory": "person_name",
                "confidence": self.config.default_confidence,
                "taxonomy_version": "1.0.0"
            },
            "framework_mappings": [
                {
                    "control_id": "CC6.1",
                    "control_name": "Logical Access Controls",
                    "risk_level": "medium"
                }
            ],
            "confidence_score": self.config.default_confidence
        }
    
    def _create_malicious_payload(self, attack_type: str, payload: str) -> Dict[str, Any]:
        """Create malicious payload for security testing."""
        base_payload = self.create_mapping_payload()
        
        # Inject malicious content into various fields
        malicious_payload = {
            **base_payload,
            "malicious_content": payload,
            "attack_type": attack_type,
            "detector_outputs": [
                {
                    **base_payload["detector_outputs"][0],
                    "findings": [
                        {
                            "entity_type": "MALICIOUS",
                            "confidence": 0.99,
                            "start": 0,
                            "end": len(payload),
                            "text": payload
                        }
                    ]
                }
            ]
        }
        
        return malicious_payload
