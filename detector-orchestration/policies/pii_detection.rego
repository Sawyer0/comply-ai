package detector_orchestration.pii

import future.keywords.if
import future.keywords.in

# Default PII detection policy - deny by default for security
default allow := false
default pii_detected := false

# PII detection rules based on content analysis
pii_detected if {
    input.content
    contains_ssn(input.content)
}

pii_detected if {
    input.content
    contains_credit_card(input.content)
}

pii_detected if {
    input.content
    contains_phone_number(input.content)
}

pii_detected if {
    input.content
    contains_email_address(input.content)
}

pii_detected if {
    input.content
    contains_address(input.content)
}

# Allow processing if PII handling is configured and compliant
allow if {
    input.tenant_id
    input.policy_bundle
    
    # Get tenant PII policy
    tenant_policy := data.tenant_policies[input.tenant_id][input.policy_bundle]
    tenant_policy.pii_handling_enabled == true
    
    # Verify required detectors are configured
    required_pii_detectors := get_required_pii_detectors(tenant_policy)
    count(required_pii_detectors) > 0
    
    # Ensure all required detectors are available
    all_available := [detector | 
        detector := required_pii_detectors[_]
        detector in input.available_detectors
    ]
    count(all_available) == count(required_pii_detectors)
}

# Get required PII detectors based on tenant sensitivity level
get_required_pii_detectors(tenant_policy) := detectors if {
    tenant_policy.sensitivity_level == "critical"
    detectors := ["regex-pii", "bert-ner", "spacy-ner"]
} else if {
    tenant_policy.sensitivity_level == "high"
    detectors := ["regex-pii", "bert-ner"]
} else if {
    tenant_policy.sensitivity_level == "medium"
    detectors := ["regex-pii"]
} else {
    # Low sensitivity - basic detection
    detectors := ["echo"]
}

# SSN detection patterns
contains_ssn(content) if {
    # Pattern for XXX-XX-XXXX format
    regex.match(`\d{3}-\d{2}-\d{4}`, content)
}

contains_ssn(content) if {
    # Pattern for XXXXXXXXX format (9 consecutive digits)
    regex.match(`\b\d{9}\b`, content)
}

# Credit card detection patterns
contains_credit_card(content) if {
    # Visa pattern (starts with 4, 16 digits)
    regex.match(`4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}`, content)
}

contains_credit_card(content) if {
    # MasterCard pattern (starts with 5, 16 digits)
    regex.match(`5\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}`, content)
}

contains_credit_card(content) if {
    # American Express pattern (starts with 34 or 37, 15 digits)
    regex.match(`3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}`, content)
}

# Phone number detection patterns
contains_phone_number(content) if {
    # US phone number patterns
    regex.match(`\(\d{3}\)[\s-]?\d{3}[\s-]?\d{4}`, content)
}

contains_phone_number(content) if {
    # Alternative US phone format
    regex.match(`\d{3}[\s.-]\d{3}[\s.-]\d{4}`, content)
}

# Email address detection
contains_email_address(content) if {
    regex.match(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`, content)
}

# Address detection (basic patterns)
contains_address(content) if {
    # Street address pattern with numbers and common street suffixes
    regex.match(`\d+\s+[A-Za-z\s]+(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Ct|Court)\.?`, content)
}

# PII sensitivity scoring based on detection results
calculate_pii_risk_score(detected_types) := score if {
    # High risk PII types
    high_risk_types := {"ssn", "credit_card", "passport"}
    high_risk_count := count([t | t := detected_types[_]; t in high_risk_types])
    
    # Medium risk PII types  
    medium_risk_types := {"phone", "email", "address"}
    medium_risk_count := count([t | t := detected_types[_]; t in medium_risk_types])
    
    # Calculate weighted score
    score := (high_risk_count * 10) + (medium_risk_count * 5)
}

# Data classification based on PII detection
classify_data_sensitivity(pii_types, pii_count) := classification if {
    pii_count >= 5
    high_risk_present := count([t | t := pii_types[_]; t in {"ssn", "credit_card", "passport"}]) > 0
    high_risk_present
    classification := "highly_sensitive"
} else if {
    pii_count >= 3
    classification := "sensitive" 
} else if {
    pii_count >= 1
    classification := "protected"
} else {
    classification := "public"
}

# Retention policy based on PII sensitivity
get_retention_requirements(classification) := requirements if {
    classification == "highly_sensitive"
    requirements := {
        "max_retention_days": 30,
        "encryption_required": true,
        "audit_required": true,
        "anonymization_required": true
    }
} else if {
    classification == "sensitive"
    requirements := {
        "max_retention_days": 90,
        "encryption_required": true,
        "audit_required": true,
        "anonymization_required": false
    }
} else if {
    classification == "protected"
    requirements := {
        "max_retention_days": 365,
        "encryption_required": true,
        "audit_required": false,
        "anonymization_required": false
    }
} else {
    # Public data
    requirements := {
        "max_retention_days": 2555, # 7 years
        "encryption_required": false,
        "audit_required": false,
        "anonymization_required": false
    }
}
