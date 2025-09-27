package detector_orchestration.compliance

import future.keywords.if
import future.keywords.in

# Default compliance policy - strict enforcement
default allow := false
default compliant := false

# Compliance frameworks supported
compliance_frameworks := {
    "SOC2": "soc2",
    "ISO27001": "iso27001", 
    "ISO42001": "iso42001",
    "HIPAA": "hipaa",
    "GDPR": "gdpr",
    "CCPA": "ccpa",
    "PCI_DSS": "pci_dss",
    "EU_AI_ACT": "eu_ai_act"
}

# Allow processing if all compliance requirements are met
allow if {
    input.tenant_id
    input.policy_bundle
    input.compliance_framework
    
    # Get tenant compliance policy
    tenant_policy := data.tenant_policies[input.tenant_id][input.policy_bundle]
    
    # Verify required compliance framework is supported
    input.compliance_framework in compliance_frameworks
    
    # Check compliance requirements are met
    meets_compliance_requirements(input, tenant_policy)
    
    # Verify audit trail requirements
    meets_audit_requirements(input, tenant_policy)
    
    # Check data handling requirements
    meets_data_handling_requirements(input, tenant_policy)
}

# Check if compliance requirements are satisfied
meets_compliance_requirements(request, tenant_policy) if {
    framework := request.compliance_framework
    
    # SOC 2 compliance requirements
    framework == "soc2"
    validate_soc2_requirements(request, tenant_policy)
}

meets_compliance_requirements(request, tenant_policy) if {
    framework := request.compliance_framework
    
    # ISO 27001 compliance requirements
    framework == "iso27001"
    validate_iso27001_requirements(request, tenant_policy)
}

meets_compliance_requirements(request, tenant_policy) if {
    framework := request.compliance_framework
    
    # ISO 42001 (AI Management) compliance requirements
    framework == "iso42001"
    validate_iso42001_requirements(request, tenant_policy)
}

meets_compliance_requirements(request, tenant_policy) if {
    framework := request.compliance_framework
    
    # HIPAA compliance requirements
    framework == "hipaa"
    validate_hipaa_requirements(request, tenant_policy)
}

meets_compliance_requirements(request, tenant_policy) if {
    framework := request.compliance_framework
    
    # GDPR compliance requirements
    framework == "gdpr"
    validate_gdpr_requirements(request, tenant_policy)
}

meets_compliance_requirements(request, tenant_policy) if {
    framework := request.compliance_framework
    
    # EU AI Act compliance requirements
    framework == "eu_ai_act"
    validate_eu_ai_act_requirements(request, tenant_policy)
}

# SOC 2 Type II compliance validation
validate_soc2_requirements(request, tenant_policy) if {
    # Security principle: Access controls and encryption
    tenant_policy.encryption_enabled == true
    tenant_policy.access_control_enabled == true
    
    # Availability principle: Monitoring and redundancy required
    tenant_policy.monitoring_enabled == true
    tenant_policy.backup_enabled == true
    
    # Processing integrity: Data validation and error handling
    tenant_policy.data_validation_enabled == true
    tenant_policy.error_handling_enabled == true
    
    # Confidentiality: Sensitive data protection
    tenant_policy.pii_protection_enabled == true
    
    # Privacy: Data minimization and retention
    tenant_policy.data_retention_policy != null
    tenant_policy.data_minimization_enabled == true
}

# ISO 27001 compliance validation
validate_iso27001_requirements(request, tenant_policy) if {
    # Information security management requirements
    tenant_policy.security_controls_implemented == true
    tenant_policy.risk_assessment_completed == true
    tenant_policy.incident_response_plan == true
    
    # Access control requirements (A.9)
    tenant_policy.access_control_policy == true
    tenant_policy.user_access_management == true
    
    # Cryptography requirements (A.10)
    tenant_policy.encryption_policy == true
    tenant_policy.key_management == true
    
    # Operations security requirements (A.12)
    tenant_policy.change_management == true
    tenant_policy.malware_protection == true
}

# ISO 42001 AI Management compliance validation
validate_iso42001_requirements(request, tenant_policy) if {
    # AI system governance requirements
    tenant_policy.ai_governance_framework == true
    tenant_policy.ai_risk_management == true
    
    # AI system lifecycle management
    tenant_policy.ai_model_versioning == true
    tenant_policy.ai_testing_procedures == true
    
    # AI system monitoring and performance
    tenant_policy.ai_performance_monitoring == true
    tenant_policy.ai_bias_monitoring == true
    
    # AI system transparency and explainability
    tenant_policy.ai_explainability_enabled == true
    tenant_policy.ai_audit_trail_enabled == true
}

# HIPAA compliance validation
validate_hipaa_requirements(request, tenant_policy) if {
    # Administrative safeguards
    tenant_policy.hipaa_officer_assigned == true
    tenant_policy.workforce_training_completed == true
    
    # Physical safeguards
    tenant_policy.facility_access_controls == true
    tenant_policy.workstation_security == true
    
    # Technical safeguards
    tenant_policy.access_control_unique_user == true
    tenant_policy.automatic_logoff == true
    tenant_policy.encryption_decryption == true
    
    # PHI handling specific requirements
    tenant_policy.phi_minimum_necessary == true
    tenant_policy.business_associate_agreements == true
}

# GDPR compliance validation
validate_gdpr_requirements(request, tenant_policy) if {
    # Lawful basis for processing
    tenant_policy.lawful_basis_documented == true
    
    # Data subject rights
    tenant_policy.right_to_access == true
    tenant_policy.right_to_rectification == true
    tenant_policy.right_to_erasure == true
    tenant_policy.right_to_portability == true
    
    # Data protection principles
    tenant_policy.data_minimization == true
    tenant_policy.purpose_limitation == true
    tenant_policy.storage_limitation == true
    
    # Security and breach notification
    tenant_policy.data_protection_impact_assessment == true
    tenant_policy.breach_notification_procedures == true
}

# EU AI Act compliance validation
validate_eu_ai_act_requirements(request, tenant_policy) if {
    # Risk-based approach - High-risk AI systems
    tenant_policy.ai_system_risk_assessment == true
    tenant_policy.ai_system_conformity_assessment == true
    
    # Transparency and explainability requirements
    tenant_policy.ai_transparency_obligations == true
    tenant_policy.ai_explainability_enabled == true
    tenant_policy.ai_decision_logging == true
    
    # Human oversight requirements
    tenant_policy.human_oversight_enabled == true
    tenant_policy.human_intervention_capability == true
    tenant_policy.human_decision_override == true
    
    # Data governance and quality management
    tenant_policy.ai_data_governance == true
    tenant_policy.ai_data_quality_management == true
    tenant_policy.ai_training_data_validation == true
    
    # Technical documentation and record keeping
    tenant_policy.ai_technical_documentation == true
    tenant_policy.ai_system_records_keeping == true
    tenant_policy.ai_incident_reporting == true
    
    # Accuracy, robustness and cybersecurity
    tenant_policy.ai_accuracy_requirements == true
    tenant_policy.ai_robustness_testing == true
    tenant_policy.ai_cybersecurity_measures == true
    
    # Fundamental rights impact assessment
    tenant_policy.fundamental_rights_impact_assessment == true
    tenant_policy.bias_detection_monitoring == true
    tenant_policy.discrimination_prevention == true
}

# Audit requirements validation
meets_audit_requirements(request, tenant_policy) if {
    # Audit logging must be enabled
    tenant_policy.audit_logging_enabled == true
    
    # Retention period must meet compliance minimum
    audit_retention_days := tenant_policy.audit_retention_days
    min_retention := get_minimum_audit_retention(request.compliance_framework)
    audit_retention_days >= min_retention
    
    # Audit integrity protection
    tenant_policy.audit_integrity_protection == true
}

# Data handling requirements validation
meets_data_handling_requirements(request, tenant_policy) if {
    # Data classification is required
    tenant_policy.data_classification_enabled == true
    
    # Encryption requirements based on data sensitivity
    validate_encryption_requirements(request, tenant_policy)
    
    # Data lifecycle management
    tenant_policy.data_lifecycle_policy == true
    
    # Cross-border transfer restrictions
    validate_data_transfer_restrictions(request, tenant_policy)
}

# Get minimum audit retention based on compliance framework
get_minimum_audit_retention(framework) := days if {
    framework == "soc2"
    days := 365  # 1 year minimum
} else if {
    framework == "iso27001"
    days := 1095  # 3 years
} else if {
    framework == "hipaa"
    days := 2555  # 7 years
} else if {
    framework == "gdpr"
    days := 1095  # 3 years
} else if {
    framework == "eu_ai_act"
    days := 1095  # 3 years minimum for AI system records
} else {
    # Default conservative retention
    days := 2555  # 7 years
}

# Validate encryption requirements
validate_encryption_requirements(request, tenant_policy) if {
    # Encryption in transit required
    tenant_policy.encryption_in_transit == true
    
    # Encryption at rest for sensitive data
    data_sensitivity := get_data_sensitivity_level(request)
    
    data_sensitivity in {"sensitive", "highly_sensitive"}
    tenant_policy.encryption_at_rest == true
}

validate_encryption_requirements(request, tenant_policy) if {
    # Public/internal data may not require encryption at rest
    data_sensitivity := get_data_sensitivity_level(request)
    data_sensitivity in {"public", "internal"}
    
    # But encryption in transit is always required
    tenant_policy.encryption_in_transit == true
}

# Get data sensitivity level from request
get_data_sensitivity_level(request) := sensitivity if {
    request.data_classification
    sensitivity := request.data_classification
} else {
    # Default to sensitive for compliance safety
    sensitivity := "sensitive"
}

# Validate data transfer restrictions
validate_data_transfer_restrictions(request, tenant_policy) if {
    # If cross-border transfer is involved
    request.cross_border_transfer == true
    
    # Adequate protection must be in place
    tenant_policy.adequacy_decision == true
    tenant_policy.standard_contractual_clauses == true
}

validate_data_transfer_restrictions(request, tenant_policy) if {
    # No cross-border transfer
    request.cross_border_transfer != true
}

# Generate compliance report
generate_compliance_report(request, tenant_policy) := report if {
    framework := request.compliance_framework
    
    report := {
        "framework": framework,
        "status": determine_compliance_status(request, tenant_policy),
        "requirements_met": list_met_requirements(request, tenant_policy),
        "requirements_failed": list_failed_requirements(request, tenant_policy),
        "risk_score": calculate_compliance_risk_score(request, tenant_policy),
        "recommendations": generate_compliance_recommendations(request, tenant_policy),
        "next_review_date": calculate_next_review_date(framework)
    }
}

# Determine overall compliance status
determine_compliance_status(request, tenant_policy) := status if {
    meets_compliance_requirements(request, tenant_policy)
    meets_audit_requirements(request, tenant_policy)
    meets_data_handling_requirements(request, tenant_policy)
    status := "compliant"
} else {
    status := "non_compliant"
}

# Calculate compliance risk score (0-100, higher is riskier)
calculate_compliance_risk_score(request, tenant_policy) := score if {
    base_score := 0
    
    # Add points for missing critical controls
    missing_encryption := tenant_policy.encryption_enabled != true
    missing_audit := tenant_policy.audit_logging_enabled != true
    missing_access_control := tenant_policy.access_control_enabled != true
    
    encryption_penalty := 30 * to_number(missing_encryption)
    audit_penalty := 25 * to_number(missing_audit)
    access_penalty := 20 * to_number(missing_access_control)
    
    score := base_score + encryption_penalty + audit_penalty + access_penalty
}

# Convert boolean to number for scoring
to_number(true) := 1
to_number(false) := 0

# Calculate next compliance review date
calculate_next_review_date(framework) := next_date if {
    framework in {"soc2", "iso27001"}
    # Annual reviews required
    next_date := "365_days"
} else if {
    framework in {"hipaa", "gdpr"}
    # More frequent reviews for healthcare/privacy
    next_date := "180_days"
} else if {
    framework == "eu_ai_act"
    # EU AI Act requires regular monitoring and updates
    next_date := "90_days"
} else {
    # Default semi-annual review
    next_date := "180_days"
}
