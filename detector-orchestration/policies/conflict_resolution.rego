package detector_orchestration

import future.keywords.if
import future.keywords.in

# Default conflict resolution strategy
default resolve := "majority_vote"

# Resolve conflicts based on tenant policy and content type
resolve := strategy if {
    input.tenant_id
    input.policy_bundle
    input.content_type

    # Get tenant policy for conflict resolution
    tenant_policy := data.tenant_policies[input.tenant_id][input.policy_bundle]

    # Select strategy based on content type and tenant preferences
    strategy := get_resolution_strategy(input.content_type, tenant_policy, input.detector_results)
}

# Get resolution strategy based on content type and tenant policy
get_resolution_strategy(content_type, tenant_policy, detector_results) := strategy if {
    # Use highest_confidence for critical content
    content_type == "critical"
    strategy := "highest_confidence"
} else if {
    # Use majority vote with required detectors for sensitive content
    content_type == "sensitive"
    strategy := "required_detector_consensus"
} else if {
    # Use weighted average for general content
    count(detector_results) > 2
    strategy := "weighted_average"
} else {
    # Use majority vote for simple cases
    strategy := "majority_vote"
}

# Apply majority vote strategy
apply_majority_vote(detector_results) := result if {
    # Group results by classification
    classifications := {result.classification | result := detector_results[_]}

    # Count votes for each classification
    votes := {classification: count([r | r := detector_results[_]; r.classification == classification]) | classification := classifications}

    # Find classification with most votes
    max_votes := max([votes[classification] | classification := classifications])
    winning_classifications := [classification | classification := classifications; votes[classification] == max_votes]

    # If tie, use highest confidence
    count(winning_classifications) == 1
    result := {
        "classification": winning_classifications[0],
        "confidence": max([r.confidence | r := detector_results[_]; r.classification == winning_classifications[0]]),
        "strategy": "majority_vote"
    }
} else {
    # Tie breaker: use highest confidence
    highest_confidence := max([r.confidence | r := detector_results[_]])
    winner := [r | r := detector_results[_]; r.confidence == highest_confidence][0]
    result := {
        "classification": winner.classification,
        "confidence": winner.confidence,
        "strategy": "highest_confidence_tie_breaker"
    }
}

# Apply highest confidence strategy
apply_highest_confidence(detector_results) := result if {
    highest_confidence := max([r.confidence | r := detector_results[_]])
    winner := [r | r := detector_results[_]; r.confidence == highest_confidence][0]
    result := {
        "classification": winner.classification,
        "confidence": winner.confidence,
        "strategy": "highest_confidence"
    }
}

# Apply weighted average strategy
apply_weighted_average(detector_results) := result if {
    # Calculate weighted average confidence per classification
    classifications := {r.classification | r := detector_results[_]}

    weighted_results := {classification: sum([r.confidence * r.weight | r := detector_results[_]; r.classification == classification]) | classification := classifications}

    # Find classification with highest weighted score
    max_weighted := max([weighted_results[classification] | classification := classifications])
    winner := [classification | classification := classifications; weighted_results[classification] == max_weighted][0]

    result := {
        "classification": winner,
        "confidence": weighted_results[winner] / sum([r.weight | r := detector_results[_]; r.classification == winner]),
        "strategy": "weighted_average"
    }
}

# Apply required detector consensus strategy
apply_required_detector_consensus(detector_results, tenant_policy) := result if {
    # Check if all required detectors agree
    required_detectors := tenant_policy.required_detectors

    # Get results from required detectors only
    required_results := [r | r := detector_results[_]; r.detector_name in required_detectors]

    # Check if all required detectors found the same classification
    classifications := {r.classification | r := required_results[_]}

    count(classifications) == 1
    result := {
        "classification": classifications[0],
        "confidence": min([r.confidence | r := required_results[_]]),
        "strategy": "required_detector_consensus"
    }
} else {
    # Fall back to highest confidence if consensus not reached
    result := apply_highest_confidence(required_results)
    result.strategy := "required_detector_consensus_fallback"
}
