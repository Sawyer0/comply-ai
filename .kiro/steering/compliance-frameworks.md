---
inclusion: fileMatch
fileMatchPattern: '**/compliance/**'
---

# Compliance Framework Implementation

Reference the compliance schema definitions:
#[[file:schemas/compliance-frameworks.json]]

## Framework Mapping Architecture

### SOC 2 Type II Implementation
```python
# src/llama_mapper/compliance/soc2.py
class SOC2Mapper:
    """Maps canonical taxonomy to SOC 2 controls"""
    
    CONTROL_MAPPINGS = {
        "pii.person_name": ["CC6.1", "CC6.7"],
        "pii.ssn": ["CC6.1", "CC6.7", "CC7.1"],
        "security.credentials": ["CC6.1", "CC6.2"],
        "content.toxic": ["CC7.1", "CC7.2"]
    }
    
    def map_to_controls(self, canonical_result: CanonicalTaxonomy) -> List[SOC2Control]:
        """Map canonical taxonomy to SOC 2 controls"""
        controls = self.CONTROL_MAPPINGS.get(
            f"{canonical_result.category}.{canonical_result.subcategory}",
            []
        )
        
        return [
            SOC2Control(
                control_id=control_id,
                description=self.get_control_description(control_id),
                evidence_type=self.get_evidence_type(control_id),
                risk_level=self.calculate_risk_level(canonical_result.confidence)
            )
            for control_id in controls
        ]
```

### ISO 27001:2022 Implementation
```python
# src/llama_mapper/compliance/iso27001.py
class ISO27001Mapper:
    """Maps canonical taxonomy to ISO 27001 controls"""
    
    CONTROL_MAPPINGS = {
        "pii": {
            "controls": ["A.8.2.1", "A.8.2.2", "A.13.2.1"],
            "classification": "confidential",
            "handling_requirements": ["encryption", "access_control", "audit_trail"]
        },
        "security": {
            "controls": ["A.9.1.1", "A.9.2.1", "A.14.2.1"],
            "classification": "restricted",
            "handling_requirements": ["immediate_action", "incident_response"]
        }
    }
```

## Evidence Generation

### Automated Evidence Collection
```python
def generate_compliance_evidence(
    detection_results: List[DetectionResult],
    framework: ComplianceFramework
) -> ComplianceEvidence:
    """Generate compliance evidence from detection results"""
    
    evidence = ComplianceEvidence(
        framework=framework,
        timestamp=datetime.utcnow(),
        detection_summary=summarize_detections(detection_results),
        control_mappings=map_to_controls(detection_results, framework),
        risk_assessment=assess_compliance_risk(detection_results),
        remediation_actions=generate_remediation_plan(detection_results)
    )
    
    return evidence
```

### Audit Trail Generation
```python
class AuditTrailGenerator:
    """Generate comprehensive audit trails for compliance"""
    
    def create_audit_record(self, event: ComplianceEvent) -> AuditRecord:
        """Create detailed audit record"""
        return AuditRecord(
            event_id=generate_uuid(),
            timestamp=event.timestamp,
            event_type=event.type,
            user_id=event.user_id,
            resource_id=event.resource_id,
            action=event.action,
            outcome=event.outcome,
            metadata={
                "ip_address": event.ip_address,
                "user_agent": event.user_agent,
                "session_id": event.session_id,
                "compliance_framework": event.framework
            },
            hash=self.calculate_integrity_hash(event)
        )
```

## Risk Assessment Engine

### Risk Calculation
```python
class ComplianceRiskAssessor:
    """Assess compliance risk based on detection results"""
    
    RISK_WEIGHTS = {
        "pii.ssn": 1.0,           # Highest risk
        "pii.credit_card": 0.9,
        "pii.person_name": 0.6,
        "security.credentials": 0.8,
        "content.toxic": 0.4
    }
    
    def calculate_risk_score(self, detections: List[Detection]) -> RiskScore:
        """Calculate overall compliance risk score"""
        total_risk = 0.0
        max_individual_risk = 0.0
        
        for detection in detections:
            category_key = f"{detection.category}.{detection.subcategory}"
            weight = self.RISK_WEIGHTS.get(category_key, 0.3)
            
            # Risk = weight * confidence * severity_multiplier
            individual_risk = weight * detection.confidence * self.get_severity_multiplier(detection)
            total_risk += individual_risk
            max_individual_risk = max(max_individual_risk, individual_risk)
        
        return RiskScore(
            overall_score=min(total_risk, 1.0),
            max_individual_risk=max_individual_risk,
            risk_level=self.categorize_risk_level(total_risk),
            contributing_factors=self.identify_risk_factors(detections)
        )
```

### Remediation Planning
```python
def generate_remediation_plan(detections: List[Detection]) -> RemediationPlan:
    """Generate actionable remediation plan"""
    
    actions = []
    for detection in detections:
        if detection.category == "pii":
            actions.extend([
                RemediationAction(
                    type="data_masking",
                    priority="high",
                    description="Mask or redact PII data",
                    estimated_effort="2 hours"
                ),
                RemediationAction(
                    type="access_review",
                    priority="medium", 
                    description="Review access controls for PII data",
                    estimated_effort="4 hours"
                )
            ])
        elif detection.category == "security":
            actions.append(
                RemediationAction(
                    type="credential_rotation",
                    priority="critical",
                    description="Rotate exposed credentials immediately",
                    estimated_effort="1 hour"
                )
            )
    
    return RemediationPlan(
        actions=actions,
        total_estimated_effort=sum(action.estimated_effort for action in actions),
        priority_order=sorted(actions, key=lambda x: x.priority_score, reverse=True)
    )
```

## Framework-Specific Configurations

### HIPAA Configuration
```yaml
# config/compliance/hipaa.yaml
framework:
  name: "HIPAA"
  version: "2013"
  
safeguards:
  administrative:
    - security_officer_designation
    - workforce_training
    - access_management
    
  physical:
    - facility_access_controls
    - workstation_use_restrictions
    - device_media_controls
    
  technical:
    - access_control
    - audit_controls
    - integrity
    - person_authentication
    - transmission_security

phi_categories:
  - names
  - geographic_subdivisions
  - dates
  - telephone_numbers
  - fax_numbers
  - email_addresses
  - ssn
  - medical_record_numbers
  - health_plan_beneficiary_numbers
  - account_numbers
  - certificate_license_numbers
  - vehicle_identifiers
  - device_identifiers
  - web_urls
  - ip_addresses
  - biometric_identifiers
  - full_face_photos
  - other_unique_identifiers
```

### Custom Framework Support
```python
class CustomFrameworkMapper:
    """Support for custom compliance frameworks"""
    
    def __init__(self, framework_config: dict):
        self.framework_config = framework_config
        self.control_mappings = self.load_control_mappings()
        self.risk_weights = self.load_risk_weights()
    
    def load_framework_from_config(self, config_path: str) -> ComplianceFramework:
        """Load custom framework from configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return ComplianceFramework(
            name=config['framework']['name'],
            version=config['framework']['version'],
            controls=self.parse_controls(config['controls']),
            mappings=self.parse_mappings(config['mappings']),
            risk_matrix=self.parse_risk_matrix(config['risk_matrix'])
        )
```

## Reporting & Analytics

### Compliance Dashboard Data
```python
def generate_compliance_dashboard_data(
    tenant_id: str,
    framework: str,
    time_range: TimeRange
) -> DashboardData:
    """Generate data for compliance dashboard"""
    
    detections = get_detections_for_period(tenant_id, time_range)
    
    return DashboardData(
        summary_metrics=calculate_summary_metrics(detections),
        risk_trends=calculate_risk_trends(detections, time_range),
        control_coverage=calculate_control_coverage(detections, framework),
        top_risk_areas=identify_top_risk_areas(detections),
        remediation_status=get_remediation_status(tenant_id),
        compliance_score=calculate_compliance_score(detections, framework)
    )
```

### Audit Report Generation
```python
def generate_audit_report(
    tenant_id: str,
    framework: str,
    report_period: DateRange
) -> AuditReport:
    """Generate comprehensive audit report"""
    
    return AuditReport(
        executive_summary=generate_executive_summary(),
        control_assessment=assess_control_effectiveness(),
        risk_analysis=perform_risk_analysis(),
        findings_summary=summarize_findings(),
        remediation_tracking=track_remediation_progress(),
        recommendations=generate_recommendations(),
        appendices={
            "detailed_findings": get_detailed_findings(),
            "evidence_inventory": get_evidence_inventory(),
            "methodology": get_assessment_methodology()
        }
    )
```