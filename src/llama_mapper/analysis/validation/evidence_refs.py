"""
Evidence reference validation for analysis module.

This module provides validation for evidence references to ensure they only
reference allowed fields from the input metrics.
"""

from typing import List, Set

# Allowed evidence reference field names (Task 2.2)
ALLOWED_EVIDENCE_REFS: Set[str] = {
    "period",
    "tenant", 
    "app",
    "route",
    "required_detectors",
    "observed_coverage",
    "required_coverage",
    "detector_errors",
    "high_sev_hits",
    "false_positive_bands",
    "policy_bundle",
    "env"
}


def validate_evidence_refs(evidence_refs: List[str]) -> tuple[bool, List[str]]:
    """
    Validate evidence references against allowed field names.
    
    Args:
        evidence_refs: List of evidence reference strings
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    for ref in evidence_refs:
        if not isinstance(ref, str):
            errors.append(f"Evidence reference must be string, got {type(ref)}")
            continue
            
        # Check if reference is in allowed fields
        if ref not in ALLOWED_EVIDENCE_REFS:
            errors.append(f"Evidence reference '{ref}' not in allowed fields: {sorted(ALLOWED_EVIDENCE_REFS)}")
    
    return len(errors) == 0, errors


def extract_field_references(evidence_refs: List[str]) -> Set[str]:
    """
    Extract field names from evidence references.
    
    Args:
        evidence_refs: List of evidence reference strings
        
    Returns:
        Set of referenced field names
    """
    referenced_fields = set()
    
    for ref in evidence_refs:
        if isinstance(ref, str) and ref in ALLOWED_EVIDENCE_REFS:
            referenced_fields.add(ref)
    
    return referenced_fields


def validate_evidence_refs_against_input(evidence_refs: List[str], input_data: dict) -> tuple[bool, List[str]]:
    """
    Validate evidence references against actual input data.
    
    Args:
        evidence_refs: List of evidence reference strings
        input_data: Input data dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # First validate against allowed fields
    is_valid, field_errors = validate_evidence_refs(evidence_refs)
    errors.extend(field_errors)
    
    if not is_valid:
        return False, errors
    
    # Then validate that referenced fields exist in input data
    for ref in evidence_refs:
        if ref not in input_data:
            errors.append(f"Evidence reference '{ref}' not found in input data")
    
    return len(errors) == 0, errors
