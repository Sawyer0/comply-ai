"""
Tests for audit trail and compliance mapping functionality.
"""

import pytest
from datetime import datetime, timedelta, UTC
from unittest.mock import Mock, patch

from src.llama_mapper.reporting.audit_trail import AuditTrailManager, IncidentData
from src.llama_mapper.reporting.models import AuditRecord, LineageRecord
from src.llama_mapper.data.frameworks import FrameworkMapper
from src.llama_mapper.data.taxonomy import TaxonomyLoader
from src.llama_mapper.data.detectors import DetectorConfigLoader


class TestAuditTrailManager:
    """Test cases for AuditTrailManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_framework_mapper = Mock(spec=FrameworkMapper)
        self.mock_taxonomy_loader = Mock(spec=TaxonomyLoader)
        self.mock_detector_loader = Mock(spec=DetectorConfigLoader)
        
        self.audit_manager = AuditTrailManager(
            framework_mapper=self.mock_framework_mapper,
            taxonomy_loader=self.mock_taxonomy_loader,
            detector_loader=self.mock_detector_loader
        )
    
    def test_create_audit_record(self):
        """Test creating an audit record."""
        # Mock framework mappings
        mock_framework_mapping = Mock()
        mock_framework_mapping.mappings = {
            "HARM.SPEECH.Toxicity": ["SOC2:CC7.2", "ISO27001:A.12.4.1"]
        }
        self.mock_framework_mapper.load_framework_mapping.return_value = mock_framework_mapping
        
        record = self.audit_manager.create_audit_record(
            tenant_id="test-tenant",
            detector_type="deberta-toxicity",
            taxonomy_hit="HARM.SPEECH.Toxicity",
            confidence_score=0.95,
            model_version="v1.0.0",
            mapping_method="model"
        )
        
        assert record.tenant_id == "test-tenant"
        assert record.detector_type == "deberta-toxicity"
        assert record.taxonomy_hit == "HARM.SPEECH.Toxicity"
        assert record.confidence_score == 0.95
        assert record.mapping_method == "model"
        assert len(record.framework_mappings) == 2
        assert "SOC2:CC7.2" in record.framework_mappings
        assert "ISO27001:A.12.4.1" in record.framework_mappings
    
    def test_create_lineage_record(self):
        """Test creating a lineage record."""
        record = self.audit_manager.create_lineage_record(
            detector_name="deberta-toxicity",
            detector_version="v1",
            original_label="toxic",
            canonical_label="HARM.SPEECH.Toxicity",
            confidence_score=0.95,
            mapping_method="model",
            model_version="v1.0.0",
            tenant_id="test-tenant"
        )
        
        assert record.detector_name == "deberta-toxicity"
        assert record.original_label == "toxic"
        assert record.canonical_label == "HARM.SPEECH.Toxicity"
        assert record.confidence_score == 0.95
        assert record.mapping_method == "model"
        assert record.tenant_id == "test-tenant"
    
    def test_record_and_resolve_incident(self):
        """Test recording and resolving incidents."""
        # Record incident
        incident = self.audit_manager.record_incident(
            taxonomy_label="HARM.SPEECH.Toxicity",
            detector_type="deberta-toxicity",
            tenant_id="test-tenant",
            severity="high"
        )
        
        assert incident.taxonomy_label == "HARM.SPEECH.Toxicity"
        assert incident.detector_type == "deberta-toxicity"
        assert incident.severity == "high"
        assert not incident.is_resolved()
        
        # Resolve incident
        success = self.audit_manager.resolve_incident(incident.incident_id)
        assert success
        assert incident.is_resolved()
        assert incident.resolution_time_hours is not None
        assert incident.resolution_time_hours > 0
    
    def test_generate_compliance_report(self):
        """Test generating compliance report."""
        # Set up mocks
        self.mock_framework_mapper.get_compliance_controls_for_label.return_value = {
            "SOC2": [{"control_id": "CC7.2", "description": "Test control"}]
        }
        self.mock_framework_mapper.get_version_info.return_value = {"version": "v1.0"}
        
        # Mock the framework mapping to return actual list instead of Mock
        mock_framework_mapping = Mock()
        mock_framework_mapping.mappings = {
            "HARM.SPEECH.Toxicity": ["SOC2:CC7.2", "ISO27001:A.12.4.1"]
        }
        self.mock_framework_mapper.load_framework_mapping.return_value = mock_framework_mapping
        
        # Create some test data
        self.audit_manager.create_audit_record(
            tenant_id="test-tenant",
            detector_type="deberta-toxicity",
            taxonomy_hit="HARM.SPEECH.Toxicity",
            confidence_score=0.95,
            model_version="v1.0.0",
            mapping_method="model"
        )
        
        incident = self.audit_manager.record_incident(
            taxonomy_label="HARM.SPEECH.Toxicity",
            detector_type="deberta-toxicity",
            tenant_id="test-tenant"
        )
        self.audit_manager.resolve_incident(incident.incident_id)
        
        # Generate report
        report = self.audit_manager.generate_compliance_report(tenant_id="test-tenant")
        
        assert report.metadata.tenant_id == "test-tenant"
        assert report.metadata.report_type == "compliance"
        assert len(report.audit_records) == 1
        assert report.coverage_metrics.total_incidents == 1
        assert report.coverage_metrics.covered_incidents == 1
        assert report.coverage_metrics.coverage_percentage == 100.0
    
    def test_generate_coverage_report(self):
        """Test generating coverage report."""
        # Set up mocks
        mock_taxonomy = Mock()
        mock_taxonomy.get_all_label_names.return_value = {
            "HARM.SPEECH.Toxicity", "HARM.SPEECH.Hate.Other", "PII.Contact.Email"
        }
        mock_taxonomy.version = "2025.09"
        self.mock_taxonomy_loader.load_taxonomy.return_value = mock_taxonomy
        
        mock_detector_configs = {
            "deberta-toxicity": Mock(),
            "openai-moderation": Mock()
        }
        self.mock_detector_loader.load_all_detector_configs.return_value = mock_detector_configs
        
        # Create some audit records
        self.audit_manager.create_audit_record(
            tenant_id="test-tenant",
            detector_type="deberta-toxicity",
            taxonomy_hit="HARM.SPEECH.Toxicity",
            confidence_score=0.95,
            model_version="v1.0.0",
            mapping_method="model"
        )
        
        # Generate report
        report = self.audit_manager.generate_coverage_report(tenant_id="test-tenant")
        
        assert report.metadata.tenant_id == "test-tenant"
        assert report.metadata.report_type == "coverage"
        assert report.total_taxonomy_labels == 3
        assert report.covered_labels == 1
        assert len(report.uncovered_labels) == 2
        assert report.coverage_percentage == pytest.approx(33.33, rel=1e-2)
    
    def test_get_lineage_for_label(self):
        """Test getting lineage records for a specific label."""
        # Create lineage records
        self.audit_manager.create_lineage_record(
            detector_name="deberta-toxicity",
            detector_version="v1",
            original_label="toxic",
            canonical_label="HARM.SPEECH.Toxicity",
            confidence_score=0.95,
            mapping_method="model",
            tenant_id="test-tenant"
        )
        
        self.audit_manager.create_lineage_record(
            detector_name="openai-moderation",
            detector_version="v1",
            original_label="hate",
            canonical_label="HARM.SPEECH.Hate.Other",
            confidence_score=0.88,
            mapping_method="model",
            tenant_id="test-tenant"
        )
        
        # Get lineage for specific label
        lineage = self.audit_manager.get_lineage_for_label(
            "HARM.SPEECH.Toxicity", 
            tenant_id="test-tenant"
        )
        
        assert len(lineage) == 1
        assert lineage[0].canonical_label == "HARM.SPEECH.Toxicity"
        assert lineage[0].original_label == "toxic"
        assert lineage[0].detector_name == "deberta-toxicity"
    
    def test_validate_compliance_mappings(self):
        """Test validating compliance mappings."""
        # Set up mocks
        mock_taxonomy = Mock()
        mock_taxonomy.get_all_label_names.return_value = {
            "HARM.SPEECH.Toxicity", "HARM.SPEECH.Hate.Other"
        }
        self.mock_taxonomy_loader.load_taxonomy.return_value = mock_taxonomy
        
        mock_framework_mapping = Mock()
        mock_framework_mapping.mappings = {
            "HARM.SPEECH.Toxicity": ["SOC2:CC7.2"]
            # Note: HARM.SPEECH.Hate.Other is missing from mappings
        }
        self.mock_framework_mapper.load_framework_mapping.return_value = mock_framework_mapping
        self.mock_framework_mapper.validate_against_taxonomy.return_value = {
            'valid': ["HARM.SPEECH.Toxicity"],
            'invalid': []
        }
        
        # Validate mappings
        validation = self.audit_manager.validate_compliance_mappings()
        
        assert validation['total_taxonomy_labels'] == 2
        assert validation['total_mapped_labels'] == 1
        assert validation['mapping_coverage_percentage'] == 50.0
        assert "HARM.SPEECH.Hate.Other" in validation['unmapped_taxonomy_labels']


class TestIncidentData:
    """Test cases for IncidentData."""
    
    def test_incident_resolution_time_calculation(self):
        """Test incident resolution time calculation."""
        incident = IncidentData(
            incident_id="test-incident",
            taxonomy_label="HARM.SPEECH.Toxicity",
            detector_type="deberta-toxicity",
            timestamp=datetime.now(UTC)
        )
        
        # Initially not resolved
        assert not incident.is_resolved()
        assert incident.calculate_resolution_time() is None
        
        # Resolve after 2 hours
        incident.resolved_at = incident.timestamp + timedelta(hours=2)
        resolution_time = incident.calculate_resolution_time()
        
        assert incident.is_resolved()
        assert resolution_time == pytest.approx(2.0, rel=1e-2)
        assert incident.resolution_time_hours == pytest.approx(2.0, rel=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])