#!/usr/bin/env python3
"""
Demonstration of the audit trail and compliance mapping functionality.

This script shows how to use the AuditTrailManager to:
1. Create audit records for detector mappings
2. Track lineage from detector outputs to canonical labels
3. Record and resolve incidents
4. Generate compliance and coverage reports
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_mapper.reporting.audit_trail import AuditTrailManager
from llama_mapper.data.frameworks import FrameworkMapper
from llama_mapper.data.taxonomy import TaxonomyLoader
from llama_mapper.data.detectors import DetectorConfigLoader


def main():
    """Demonstrate audit trail functionality."""
    print("🔍 Audit Trail and Compliance Mapping Demo")
    print("=" * 50)
    
    # Initialize the audit trail manager
    print("\n1. Initializing AuditTrailManager...")
    try:
        framework_mapper = FrameworkMapper()
        taxonomy_loader = TaxonomyLoader()
        detector_loader = DetectorConfigLoader()
        
        audit_manager = AuditTrailManager(
            framework_mapper=framework_mapper,
            taxonomy_loader=taxonomy_loader,
            detector_loader=detector_loader
        )
        print("✅ AuditTrailManager initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Create some sample audit records
    print("\n2. Creating sample audit records...")
    
    # Record 1: DeBERTa toxicity detection
    audit_record1 = audit_manager.create_audit_record(
        tenant_id="demo-tenant",
        detector_type="deberta-toxicity",
        taxonomy_hit="HARM.SPEECH.Toxicity",
        confidence_score=0.95,
        model_version="mapper-lora@v1.0.0",
        mapping_method="model",
        metadata={"request_id": "req-001", "source": "chat-api"}
    )
    print(f"✅ Created audit record: {audit_record1.event_id}")
    
    # Record 2: OpenAI Moderation hate speech
    audit_record2 = audit_manager.create_audit_record(
        tenant_id="demo-tenant",
        detector_type="openai-moderation",
        taxonomy_hit="HARM.SPEECH.Hate.Other",
        confidence_score=0.88,
        model_version="mapper-lora@v1.0.0",
        mapping_method="model",
        metadata={"request_id": "req-002", "source": "content-filter"}
    )
    print(f"✅ Created audit record: {audit_record2.event_id}")
    
    # Record 3: PII detection with fallback
    audit_record3 = audit_manager.create_audit_record(
        tenant_id="demo-tenant",
        detector_type="regex-pii",
        taxonomy_hit="PII.Contact.Email",
        confidence_score=0.45,  # Low confidence, would trigger fallback
        model_version="mapper-lora@v1.0.0",
        mapping_method="fallback",
        metadata={"request_id": "req-003", "source": "document-scan"}
    )
    print(f"✅ Created audit record: {audit_record3.event_id}")
    
    # Create lineage records
    print("\n3. Creating lineage records...")
    
    lineage1 = audit_manager.create_lineage_record(
        detector_name="deberta-toxicity",
        detector_version="v1",
        original_label="toxic",
        canonical_label="HARM.SPEECH.Toxicity",
        confidence_score=0.95,
        mapping_method="model",
        model_version="mapper-lora@v1.0.0",
        tenant_id="demo-tenant"
    )
    print(f"✅ Created lineage: toxic -> HARM.SPEECH.Toxicity")
    
    lineage2 = audit_manager.create_lineage_record(
        detector_name="openai-moderation",
        detector_version="v1",
        original_label="hate",
        canonical_label="HARM.SPEECH.Hate.Other",
        confidence_score=0.88,
        mapping_method="model",
        model_version="mapper-lora@v1.0.0",
        tenant_id="demo-tenant"
    )
    print(f"✅ Created lineage: hate -> HARM.SPEECH.Hate.Other")
    
    # Record and resolve incidents
    print("\n4. Recording and resolving incidents...")
    
    incident1 = audit_manager.record_incident(
        taxonomy_label="HARM.SPEECH.Toxicity",
        detector_type="deberta-toxicity",
        tenant_id="demo-tenant",
        severity="high"
    )
    print(f"✅ Recorded incident: {incident1.incident_id}")
    
    incident2 = audit_manager.record_incident(
        taxonomy_label="HARM.SPEECH.Hate.Other",
        detector_type="openai-moderation",
        tenant_id="demo-tenant",
        severity="medium"
    )
    print(f"✅ Recorded incident: {incident2.incident_id}")
    
    # Resolve the first incident after some time
    print("⏳ Simulating incident resolution...")
    success = audit_manager.resolve_incident(incident1.incident_id)
    if success:
        print(f"✅ Resolved incident: {incident1.incident_id}")
        print(f"   Resolution time: {incident1.resolution_time_hours:.2f} hours")
    
    # Generate compliance report
    print("\n5. Generating compliance report...")
    try:
        compliance_report = audit_manager.generate_compliance_report(
            tenant_id="demo-tenant"
        )
        
        print(f"✅ Compliance Report Generated:")
        print(f"   Report ID: {compliance_report.metadata.report_id}")
        print(f"   Total Incidents: {compliance_report.coverage_metrics.total_incidents}")
        print(f"   Resolved Incidents: {compliance_report.coverage_metrics.covered_incidents}")
        print(f"   Coverage: {compliance_report.coverage_metrics.coverage_percentage:.1f}%")
        if compliance_report.coverage_metrics.mttr_hours:
            print(f"   MTTR: {compliance_report.coverage_metrics.mttr_hours:.2f} hours")
        print(f"   Audit Records: {len(compliance_report.audit_records)}")
        print(f"   Lineage Records: {len(compliance_report.lineage_records)}")
        print(f"   Control Mappings: {len(compliance_report.control_mappings)}")
        
    except Exception as e:
        print(f"❌ Failed to generate compliance report: {e}")
    
    # Generate coverage report
    print("\n6. Generating coverage report...")
    try:
        coverage_report = audit_manager.generate_coverage_report(
            tenant_id="demo-tenant"
        )
        
        print(f"✅ Coverage Report Generated:")
        print(f"   Report ID: {coverage_report.metadata.report_id}")
        print(f"   Total Taxonomy Labels: {coverage_report.total_taxonomy_labels}")
        print(f"   Covered Labels: {coverage_report.covered_labels}")
        print(f"   Coverage: {coverage_report.coverage_percentage:.1f}%")
        print(f"   Uncovered Labels: {len(coverage_report.uncovered_labels)}")
        
        if coverage_report.detector_statistics:
            print(f"   Detector Statistics:")
            for detector, stats in coverage_report.detector_statistics.items():
                print(f"     - {detector}: {stats['total_mappings']} mappings, "
                      f"avg confidence: {stats['average_confidence']:.2f}")
        
    except Exception as e:
        print(f"❌ Failed to generate coverage report: {e}")
    
    # Demonstrate lineage tracking
    print("\n7. Demonstrating lineage tracking...")
    lineage_records = audit_manager.get_lineage_for_label(
        "HARM.SPEECH.Toxicity", 
        tenant_id="demo-tenant"
    )
    
    print(f"✅ Found {len(lineage_records)} lineage records for HARM.SPEECH.Toxicity:")
    for record in lineage_records:
        print(f"   - {record.detector_name}: {record.original_label} -> {record.canonical_label}")
        print(f"     Confidence: {record.confidence_score:.2f}, Method: {record.mapping_method}")
    
    # Validate compliance mappings
    print("\n8. Validating compliance mappings...")
    try:
        validation_results = audit_manager.validate_compliance_mappings()
        
        print(f"✅ Compliance Mapping Validation:")
        print(f"   Total Taxonomy Labels: {validation_results['total_taxonomy_labels']}")
        print(f"   Mapped Labels: {validation_results['total_mapped_labels']}")
        print(f"   Mapping Coverage: {validation_results['mapping_coverage_percentage']:.1f}%")
        
        if validation_results['unmapped_taxonomy_labels']:
            print(f"   Unmapped Labels: {len(validation_results['unmapped_taxonomy_labels'])}")
            for label in validation_results['unmapped_taxonomy_labels'][:5]:  # Show first 5
                print(f"     - {label}")
            if len(validation_results['unmapped_taxonomy_labels']) > 5:
                print(f"     ... and {len(validation_results['unmapped_taxonomy_labels']) - 5} more")
        
    except Exception as e:
        print(f"❌ Failed to validate compliance mappings: {e}")
    
    print("\n🎉 Audit trail demo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("✅ Audit record creation with framework mappings")
    print("✅ Lineage tracking from detector outputs to canonical labels")
    print("✅ Incident recording and resolution with MTTR calculation")
    print("✅ Compliance report generation with coverage metrics")
    print("✅ Coverage report with taxonomy and detector statistics")
    print("✅ Compliance mapping validation")


if __name__ == "__main__":
    main()