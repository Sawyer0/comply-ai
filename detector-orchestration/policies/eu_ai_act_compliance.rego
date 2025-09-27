package detector_orchestration.eu_ai_act

import future.keywords.if
import future.keywords.in

# EU AI Act compliance policy following Single Responsibility Principle
# This module handles ONLY EU AI Act compliance requirements

# Default EU AI Act policy - strict enforcement for high-risk AI systems
default allow := false
default eu_ai_act_compliant := false

# EU AI Act risk categories
ai_risk_categories := {
    "unacceptable": "prohibited_ai_practices",
    "high": "high_risk_ai_systems", 
    "limited": "limited_risk_ai_systems",
    "minimal": "minimal_risk_ai_systems"
}

# Allow processing if EU AI Act requirements are met
allow if {
    input.tenant_id
    input.policy_bundle
    input.ai_risk_category
    
    # Get tenant EU AI Act policy
    tenant_policy := data.tenant_policies[input.tenant_id][input.policy_bundle]
    
    # Verify AI risk category is supported
    input.ai_risk_category in ai_risk_categories
    
    # Check EU AI Act compliance requirements
    meets_eu_ai_act_requirements(input, tenant_policy)
    
    # Verify transparency obligations
    meets_transparency_obligations(input, tenant_policy)
    
    # Check human oversight requirements
    meets_human_oversight_requirements(input, tenant_policy)
    
    # Validate data governance requirements
    meets_data_governance_requirements(input, tenant_policy)
    
    # Check technical documentation requirements
    meets_technical_documentation_requirements(input, tenant_policy)
    
    # Verify accuracy and robustness requirements
    meets_accuracy_robustness_requirements(input, tenant_policy)
    
    # Check fundamental rights protection
    meets_fundamental_rights_requirements(input, tenant_policy)
}

# Check if EU AI Act requirements are satisfied based on risk category
meets_eu_ai_act_requirements(request, tenant_policy) if {
    risk_category := request.ai_risk_category
    
    # High-risk AI systems - strictest requirements
    risk_category == "high"
    validate_high_risk_requirements(request, tenant_policy)
}

meets_eu_ai_act_requirements(request, tenant_policy) if {
    risk_category := request.ai_risk_category
    
    # Limited risk AI systems - transparency requirements
    risk_category == "limited"
    validate_limited_risk_requirements(request, tenant_policy)
}

meets_eu_ai_act_requirements(request, tenant_policy) if {
    risk_category := request.ai_risk_category
    
    # Minimal risk AI systems - basic requirements
    risk_category == "minimal"
    validate_minimal_risk_requirements(request, tenant_policy)
}

# High-risk AI systems validation (Article 6-15 EU AI Act)
validate_high_risk_requirements(request, tenant_policy) if {
    # Risk management system
    tenant_policy.ai_system_risk_assessment == true
    tenant_policy.ai_system_conformity_assessment == true
    tenant_policy.ai_risk_management_system == true
    
    # Data governance and quality management
    tenant_policy.ai_data_governance == true
    tenant_policy.ai_data_quality_management == true
    tenant_policy.ai_training_data_validation == true
    tenant_policy.ai_data_bias_monitoring == true
    
    # Technical documentation
    tenant_policy.ai_technical_documentation == true
    tenant_policy.ai_system_records_keeping == true
    tenant_policy.ai_incident_reporting == true
    
    # Record keeping
    tenant_policy.ai_system_logs_enabled == true
    tenant_policy.ai_decision_audit_trail == true
    tenant_policy.ai_performance_monitoring == true
    
    # Transparency and provision of information to users
    tenant_policy.ai_transparency_obligations == true
    tenant_policy.ai_user_information_provision == true
    tenant_policy.ai_system_capabilities_disclosure == true
    
    # Human oversight
    tenant_policy.human_oversight_enabled == true
    tenant_policy.human_intervention_capability == true
    tenant_policy.human_decision_override == true
    tenant_policy.human_monitoring_continuous == true
    
    # Accuracy, robustness and cybersecurity
    tenant_policy.ai_accuracy_requirements == true
    tenant_policy.ai_robustness_testing == true
    tenant_policy.ai_cybersecurity_measures == true
    tenant_policy.ai_adversarial_testing == true
    
    # Fundamental rights impact assessment
    tenant_policy.fundamental_rights_impact_assessment == true
    tenant_policy.bias_detection_monitoring == true
    tenant_policy.discrimination_prevention == true
    tenant_policy.fairness_assessment == true
}

# Limited risk AI systems validation (Article 52 EU AI Act)
validate_limited_risk_requirements(request, tenant_policy) if {
    # Transparency obligations
    tenant_policy.ai_transparency_obligations == true
    tenant_policy.ai_user_information_provision == true
    tenant_policy.ai_system_capabilities_disclosure == true
    
    # Basic human oversight
    tenant_policy.human_oversight_enabled == true
    tenant_policy.human_intervention_capability == true
    
    # Basic accuracy requirements
    tenant_policy.ai_accuracy_requirements == true
    tenant_policy.ai_robustness_testing == true
    
    # Basic bias monitoring
    tenant_policy.bias_detection_monitoring == true
    tenant_policy.discrimination_prevention == true
}

# Minimal risk AI systems validation (Article 69 EU AI Act)
validate_minimal_risk_requirements(request, tenant_policy) if {
    # Basic transparency
    tenant_policy.ai_transparency_obligations == true
    tenant_policy.ai_user_information_provision == true
    
    # Basic accuracy
    tenant_policy.ai_accuracy_requirements == true
    
    # Basic bias prevention
    tenant_policy.discrimination_prevention == true
}

# Transparency obligations validation
meets_transparency_obligations(request, tenant_policy) if {
    # AI system identification
    tenant_policy.ai_system_identification == true
    tenant_policy.ai_system_purpose_disclosure == true
    tenant_policy.ai_system_limitations_disclosure == true
    
    # User information requirements
    tenant_policy.ai_user_information_provision == true
    tenant_policy.ai_system_capabilities_disclosure == true
    tenant_policy.ai_decision_explainability == true
    
    # Automated decision-making disclosure
    tenant_policy.automated_decision_disclosure == true
    tenant_policy.human_intervention_right == true
}

# Human oversight requirements validation
meets_human_oversight_requirements(request, tenant_policy) if {
    # Human oversight capabilities
    tenant_policy.human_oversight_enabled == true
    tenant_policy.human_intervention_capability == true
    tenant_policy.human_decision_override == true
    
    # Continuous monitoring for high-risk systems
    risk_category := request.ai_risk_category
    risk_category == "high"
    tenant_policy.human_monitoring_continuous == true
}

meets_human_oversight_requirements(request, tenant_policy) if {
    # Basic human oversight for non-high-risk systems
    risk_category := request.ai_risk_category
    risk_category in {"limited", "minimal"}
    
    tenant_policy.human_oversight_enabled == true
    tenant_policy.human_intervention_capability == true
}

# Data governance requirements validation
meets_data_governance_requirements(request, tenant_policy) if {
    # Data governance framework
    tenant_policy.ai_data_governance == true
    tenant_policy.ai_data_quality_management == true
    tenant_policy.ai_training_data_validation == true
    
    # Data bias monitoring
    tenant_policy.ai_data_bias_monitoring == true
    tenant_policy.bias_detection_monitoring == true
    
    # Data protection compliance
    tenant_policy.data_protection_compliance == true
    tenant_policy.privacy_by_design == true
}

# Technical documentation requirements validation
meets_technical_documentation_requirements(request, tenant_policy) if {
    # Technical documentation
    tenant_policy.ai_technical_documentation == true
    tenant_policy.ai_system_records_keeping == true
    
    # Incident reporting
    tenant_policy.ai_incident_reporting == true
    tenant_policy.ai_system_logs_enabled == true
    
    # Audit trail
    tenant_policy.ai_decision_audit_trail == true
    tenant_policy.ai_performance_monitoring == true
}

# Accuracy and robustness requirements validation
meets_accuracy_robustness_requirements(request, tenant_policy) if {
    # Accuracy requirements
    tenant_policy.ai_accuracy_requirements == true
    tenant_policy.ai_robustness_testing == true
    
    # Cybersecurity measures
    tenant_policy.ai_cybersecurity_measures == true
    tenant_policy.ai_adversarial_testing == true
    
    # Performance monitoring
    tenant_policy.ai_performance_monitoring == true
    tenant_policy.ai_continuous_monitoring == true
}

# Fundamental rights requirements validation
meets_fundamental_rights_requirements(request, tenant_policy) if {
    # Fundamental rights impact assessment
    tenant_policy.fundamental_rights_impact_assessment == true
    
    # Bias and discrimination prevention
    tenant_policy.bias_detection_monitoring == true
    tenant_policy.discrimination_prevention == true
    tenant_policy.fairness_assessment == true
    
    # Equal treatment
    tenant_policy.equal_treatment_guarantee == true
    tenant_policy.non_discrimination_measures == true
}

# Generate EU AI Act compliance report
generate_eu_ai_act_report(request, tenant_policy) := report if {
    risk_category := request.ai_risk_category
    
    report := {
        "risk_category": risk_category,
        "compliance_status": determine_eu_ai_act_compliance_status(request, tenant_policy),
        "requirements_met": list_eu_ai_act_met_requirements(request, tenant_policy),
        "requirements_failed": list_eu_ai_act_failed_requirements(request, tenant_policy),
        "risk_score": calculate_eu_ai_act_risk_score(request, tenant_policy),
        "recommendations": generate_eu_ai_act_recommendations(request, tenant_policy),
        "next_assessment_date": calculate_next_eu_ai_act_assessment(risk_category)
    }
}

# Determine EU AI Act compliance status
determine_eu_ai_act_compliance_status(request, tenant_policy) := status if {
    meets_eu_ai_act_requirements(request, tenant_policy)
    meets_transparency_obligations(request, tenant_policy)
    meets_human_oversight_requirements(request, tenant_policy)
    meets_data_governance_requirements(request, tenant_policy)
    meets_technical_documentation_requirements(request, tenant_policy)
    meets_accuracy_robustness_requirements(request, tenant_policy)
    meets_fundamental_rights_requirements(request, tenant_policy)
    status := "compliant"
} else {
    status := "non_compliant"
}

# Calculate EU AI Act risk score (0-100, higher is riskier)
calculate_eu_ai_act_risk_score(request, tenant_policy) := score if {
    base_score := 0
    
    # Add points for missing critical AI Act requirements
    missing_risk_assessment := tenant_policy.ai_system_risk_assessment != true
    missing_transparency := tenant_policy.ai_transparency_obligations != true
    missing_human_oversight := tenant_policy.human_oversight_enabled != true
    missing_accuracy := tenant_policy.ai_accuracy_requirements != true
    missing_bias_monitoring := tenant_policy.bias_detection_monitoring != true
    
    risk_assessment_penalty := 25 * to_number(missing_risk_assessment)
    transparency_penalty := 20 * to_number(missing_transparency)
    human_oversight_penalty := 20 * to_number(missing_human_oversight)
    accuracy_penalty := 15 * to_number(missing_accuracy)
    bias_penalty := 20 * to_number(missing_bias_monitoring)
    
    score := base_score + risk_assessment_penalty + transparency_penalty + human_oversight_penalty + accuracy_penalty + bias_penalty
}

# Convert boolean to number for scoring
to_number(true) := 1
to_number(false) := 0

# Calculate next EU AI Act assessment date
calculate_next_eu_ai_act_assessment(risk_category) := next_date if {
    risk_category == "high"
    # High-risk systems require quarterly assessments
    next_date := "90_days"
} else if {
    risk_category == "limited"
    # Limited risk systems require semi-annual assessments
    next_date := "180_days"
} else {
    # Minimal risk systems require annual assessments
    next_date := "365_days"
}

# Generate EU AI Act compliance recommendations
generate_eu_ai_act_recommendations(request, tenant_policy) := recommendations if {
    risk_score := calculate_eu_ai_act_risk_score(request, tenant_policy)
    risk_category := request.ai_risk_category
    
    recommendations := get_eu_ai_act_recommendations_by_score(risk_score, risk_category, tenant_policy)
}

get_eu_ai_act_recommendations_by_score(score, risk_category, tenant_policy) := recommendations if {
    score >= 80  # Critical non-compliance
    recommendations := [
        "Immediate EU AI Act compliance review required",
        "Implement comprehensive AI risk management system",
        "Establish human oversight mechanisms",
        "Deploy bias detection and monitoring systems",
        "Create technical documentation and audit trails",
        "Conduct fundamental rights impact assessment"
    ]
} else if {
    score >= 60  # Significant compliance gaps
    recommendations := [
        "Enhance AI transparency and explainability",
        "Strengthen human oversight capabilities",
        "Improve data governance and quality management",
        "Implement continuous bias monitoring",
        "Update technical documentation"
    ]
} else if {
    score >= 40  # Moderate compliance gaps
    recommendations := [
        "Review and update AI system documentation",
        "Enhance monitoring and logging capabilities",
        "Strengthen accuracy and robustness testing",
        "Improve user information provision"
    ]
} else {
    # Good compliance posture
    recommendations := [
        "Maintain current EU AI Act compliance measures",
        "Regular compliance monitoring and updates",
        "Stay informed about EU AI Act developments"
    ]
}
