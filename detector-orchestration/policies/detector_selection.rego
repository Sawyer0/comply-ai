package detector_orchestration

import future.keywords.if
import future.keywords.in

# Default detector selection policy
default allow := false

# Allow detector selection based on tenant policy and content type
allow if {
    input.tenant_id
    input.content_type
    input.policy_bundle

    # Get tenant policy for detector selection
    tenant_policy := data.tenant_policies[input.tenant_id][input.policy_bundle]

    # Check if detector is allowed for content type
    allowed_detectors := get_allowed_detectors(input.content_type, tenant_policy)

    # Check if detector is in allowed list
    input.detector_name in allowed_detectors
}

# Get allowed detectors based on content type and tenant policy
get_allowed_detectors(content_type, tenant_policy) := detectors if {
    content_type == "text"
    detectors := tenant_policy.allowed_text_detectors
} else if content_type == "image" {
    detectors := tenant_policy.allowed_image_detectors
} else {
    # Default to basic detectors for unknown content types
    detectors := ["echo", "regex-pii"]
}

# Prioritize detectors based on tenant requirements
prioritize_detectors(detectors, tenant_policy) := prioritized if {
    # Sort detectors by priority if specified
    prioritized := [detector |
        detector := detectors[_]
        priority := get_detector_priority(detector, tenant_policy)
    ]
    sort(prioritized, priority)
}

get_detector_priority(detector, tenant_policy) := priority if {
    priority := tenant_policy.detector_priorities[detector]
} else {
    # Default priority
    priority := 5
}

# Health-aware detector selection
select_healthy_detectors(detectors, health_status) := healthy_detectors if {
    healthy_detectors := [detector |
        detector := detectors[_]
        health_status[detector].healthy == true
    ]
}

# Circuit breaker aware selection
select_circuit_breaker_safe(detectors, circuit_breaker_status) := safe_detectors if {
    safe_detectors := [detector |
        detector := detectors[_]
        circuit_breaker_status[detector].state != "open"
    ]
}
