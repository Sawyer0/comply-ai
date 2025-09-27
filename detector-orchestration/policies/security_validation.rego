package detector_orchestration.security

import future.keywords.if
import future.keywords.in

# Default security policy - deny by default (fail-safe)
default allow := false
default secure := false

# Security threat categories
threat_categories := {
    "injection": "sql_injection_nosql_injection_command_injection",
    "malware": "virus_trojan_ransomware_spyware",
    "social_engineering": "phishing_pretexting_baiting",
    "data_exfiltration": "unauthorized_data_access_data_theft",
    "privilege_escalation": "vertical_horizontal_escalation",
    "denial_of_service": "ddos_resource_exhaustion"
}

# Allow processing if security validation passes
allow if {
    input.tenant_id
    input.policy_bundle
    
    # Get tenant security policy
    tenant_policy := data.tenant_policies[input.tenant_id][input.policy_bundle]
    
    # Validate security controls are in place
    validate_security_controls(input, tenant_policy)
    
    # Check threat detection capabilities
    validate_threat_detection(input, tenant_policy)
    
    # Verify access controls
    validate_access_controls(input, tenant_policy)
    
    # Ensure incident response readiness
    validate_incident_response(input, tenant_policy)
}

# Validate essential security controls
validate_security_controls(request, tenant_policy) if {
    # Authentication controls
    tenant_policy.multi_factor_authentication == true
    tenant_policy.strong_password_policy == true
    tenant_policy.session_management == true
    
    # Network security controls
    tenant_policy.network_segmentation == true
    tenant_policy.firewall_protection == true
    tenant_policy.intrusion_detection == true
    
    # Application security controls
    tenant_policy.input_validation == true
    tenant_policy.output_encoding == true
    tenant_policy.secure_communication == true
}

# Validate threat detection capabilities
validate_threat_detection(request, tenant_policy) if {
    # Malware detection
    tenant_policy.malware_scanning_enabled == true
    
    # Injection attack detection
    tenant_policy.injection_detection_enabled == true
    
    # Anomaly detection
    tenant_policy.behavioral_analytics == true
    tenant_policy.anomaly_detection_threshold <= 0.3  # Low threshold for sensitivity
    
    # Security monitoring
    tenant_policy.security_event_monitoring == true
    tenant_policy.threat_intelligence_feeds == true
}

# Validate access control mechanisms
validate_access_controls(request, tenant_policy) if {
    # Principle of least privilege
    tenant_policy.least_privilege_enabled == true
    
    # Role-based access control
    tenant_policy.rbac_enabled == true
    tenant_policy.role_segregation == true
    
    # Identity verification
    tenant_policy.identity_verification == true
    
    # Authorization mechanisms
    tenant_policy.authorization_policies == true
    tenant_policy.resource_access_controls == true
}

# Validate incident response capabilities
validate_incident_response(request, tenant_policy) if {
    # Incident detection
    tenant_policy.security_incident_detection == true
    
    # Response procedures
    tenant_policy.incident_response_plan == true
    tenant_policy.escalation_procedures == true
    
    # Recovery capabilities
    tenant_policy.disaster_recovery_plan == true
    tenant_policy.business_continuity_plan == true
    
    # Communication protocols
    tenant_policy.incident_communication_plan == true
}

# Assess security posture score (0-100, higher is more secure)
calculate_security_score(tenant_policy) := score if {
    # Core security controls (40 points max)
    auth_score := calculate_auth_score(tenant_policy) * 0.4
    
    # Network security (25 points max)
    network_score := calculate_network_score(tenant_policy) * 0.25
    
    # Application security (20 points max)
    app_score := calculate_app_score(tenant_policy) * 0.2
    
    # Monitoring and response (15 points max)
    monitoring_score := calculate_monitoring_score(tenant_policy) * 0.15
    
    score := auth_score + network_score + app_score + monitoring_score
}

# Calculate authentication security score
calculate_auth_score(tenant_policy) := score if {
    mfa_points := 25 * to_number(tenant_policy.multi_factor_authentication == true)
    password_points := 15 * to_number(tenant_policy.strong_password_policy == true)
    session_points := 10 * to_number(tenant_policy.session_management == true)
    
    score := mfa_points + password_points + session_points
}

# Calculate network security score
calculate_network_score(tenant_policy) := score if {
    segmentation_points := 10 * to_number(tenant_policy.network_segmentation == true)
    firewall_points := 8 * to_number(tenant_policy.firewall_protection == true)
    ids_points := 7 * to_number(tenant_policy.intrusion_detection == true)
    
    score := segmentation_points + firewall_points + ids_points
}

# Calculate application security score
calculate_app_score(tenant_policy) := score if {
    validation_points := 8 * to_number(tenant_policy.input_validation == true)
    encoding_points := 6 * to_number(tenant_policy.output_encoding == true)
    communication_points := 6 * to_number(tenant_policy.secure_communication == true)
    
    score := validation_points + encoding_points + communication_points
}

# Calculate monitoring and response score
calculate_monitoring_score(tenant_policy) := score if {
    monitoring_points := 6 * to_number(tenant_policy.security_event_monitoring == true)
    incident_points := 5 * to_number(tenant_policy.incident_response_plan == true)
    recovery_points := 4 * to_number(tenant_policy.disaster_recovery_plan == true)
    
    score := monitoring_points + incident_points + recovery_points
}

# Convert boolean to number for scoring
to_number(true) := 1
to_number(false) := 0

# Detect potential security threats in content
detect_security_threats(content) := threats if {
    threats := {
        "sql_injection": detect_sql_injection(content),
        "xss_attempt": detect_xss_attempt(content),
        "command_injection": detect_command_injection(content),
        "path_traversal": detect_path_traversal(content),
        "malicious_scripts": detect_malicious_scripts(content)
    }
}

# SQL injection detection patterns
detect_sql_injection(content) if {
    # Common SQL injection patterns
    regex.match(`(?i)(union|select|insert|update|delete|drop|create|alter)\s+`, content)
}

detect_sql_injection(content) if {
    # SQL injection with comments
    regex.match(`(?i)--|\*\/|\*\*|\/\*`, content)
}

detect_sql_injection(content) if {
    # Boolean-based SQL injection
    regex.match(`(?i)(and|or)\s+\d+\s*=\s*\d+`, content)
}

# Cross-site scripting (XSS) detection
detect_xss_attempt(content) if {
    # Script tag injection
    regex.match(`(?i)<script[^>]*>.*?</script>`, content)
}

detect_xss_attempt(content) if {
    # Event handler injection
    regex.match(`(?i)on(load|click|mouseover|error|focus)\s*=`, content)
}

detect_xss_attempt(content) if {
    # JavaScript protocol injection
    regex.match(`(?i)javascript\s*:`, content)
}

# Command injection detection
detect_command_injection(content) if {
    # Common command injection patterns
    regex.match(`(?i)(;|&&|\|\||\|)\s*(rm|del|format|shutdown|reboot)`, content)
}

detect_command_injection(content) if {
    # Command substitution attempts
    regex.match(`\$\(.*\)|`.*``, content)
}

# Path traversal detection
detect_path_traversal(content) if {
    # Directory traversal patterns
    regex.match(`\.\.\/|\.\.\\`, content)
}

detect_path_traversal(content) if {
    # Encoded traversal attempts
    regex.match(`%2e%2e%2f|%2e%2e%5c`, content)
}

# Malicious script detection
detect_malicious_scripts(content) if {
    # PowerShell execution attempts
    regex.match(`(?i)powershell|invoke-expression|iex`, content)
}

detect_malicious_scripts(content) if {
    # Bash/shell execution attempts
    regex.match(`(?i)(bash|sh|cmd)\s+-c`, content)
}

# Security policy enforcement levels
get_enforcement_level(tenant_policy, threat_level) := level if {
    threat_level == "critical"
    level := "block"
} else if {
    threat_level == "high"
    tenant_policy.security_tolerance == "strict"
    level := "block"
} else if {
    threat_level == "high"
    tenant_policy.security_tolerance == "moderate"
    level := "warn"
} else if {
    threat_level == "medium"
    tenant_policy.security_tolerance == "strict"
    level := "warn"
} else if {
    threat_level == "medium"
    level := "monitor"
} else {
    # Low threat level
    level := "allow"
}

# Generate security recommendations
generate_security_recommendations(request, tenant_policy) := recommendations if {
    security_score := calculate_security_score(tenant_policy)
    
    recommendations := get_recommendations_by_score(security_score, tenant_policy)
}

get_recommendations_by_score(score, tenant_policy) := recommendations if {
    score < 50  # Critical security gaps
    recommendations := [
        "Enable multi-factor authentication immediately",
        "Implement network segmentation",
        "Enable comprehensive audit logging",
        "Deploy intrusion detection system",
        "Establish incident response procedures"
    ]
} else if {
    score < 70  # Moderate security gaps
    recommendations := [
        "Enhance monitoring capabilities",
        "Implement behavioral analytics",
        "Review and update access controls",
        "Strengthen encryption policies"
    ]
} else if {
    score < 85  # Minor security improvements
    recommendations := [
        "Consider advanced threat detection",
        "Implement zero-trust architecture",
        "Enhance security automation"
    ]
} else {
    # Good security posture
    recommendations := [
        "Maintain current security controls",
        "Regular security assessments",
        "Keep security policies updated"
    ]
}

# Validate security detector requirements
validate_security_detectors(request, tenant_policy) := valid if {
    required_detectors := get_required_security_detectors(tenant_policy)
    available_detectors := request.available_detectors
    
    # Check if all required security detectors are available
    missing_detectors := [detector | 
        detector := required_detectors[_]
        not detector in available_detectors
    ]
    
    count(missing_detectors) == 0
    valid := true
}

# Get required security detectors based on threat level
get_required_security_detectors(tenant_policy) := detectors if {
    tenant_policy.threat_level == "critical"
    detectors := ["malware-scanner", "injection-detector", "anomaly-detector", "behavior-analyzer"]
} else if {
    tenant_policy.threat_level == "high"
    detectors := ["malware-scanner", "injection-detector", "anomaly-detector"]
} else if {
    tenant_policy.threat_level == "medium"
    detectors := ["malware-scanner", "injection-detector"]
} else {
    # Low threat level
    detectors := ["basic-security-scan"]
}
