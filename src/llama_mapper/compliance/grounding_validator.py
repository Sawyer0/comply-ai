"""
Mandatory Grounding Validator with Citation Checking
Ensures all compliance outputs are properly grounded with verified citations.
"""

import json
import re
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from difflib import SequenceMatcher
import logging

from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with metadata."""
    chunk_text: str
    citation: str
    pub_date: date
    source_id: str
    authority: str
    section_granularity: Optional[str] = None
    confidence_score: float = 0.0


@dataclass
class GroundingResult:
    """Result of grounding validation."""
    is_grounded: bool
    validated_citations: List[Dict[str, Any]]
    failed_citations: List[Dict[str, Any]]
    temporal_issues: List[str]
    validation_errors: List[str]
    grounding_score: float
    
    
class CitationChecker:
    """Validates citations against retrieved chunks."""
    
    def __init__(self, similarity_threshold: float = 0.85, staleness_threshold_days: int = 365):
        self.similarity_threshold = similarity_threshold
        self.staleness_threshold_days = staleness_threshold_days
        
    def verify_citation(self, quoted_text: str, retrieved_chunks: List[RetrievedChunk]) -> Tuple[bool, Optional[RetrievedChunk], float]:
        """
        Verify that quoted text appears verbatim in retrieved chunks.
        
        Args:
            quoted_text: Text that should appear in chunks
            retrieved_chunks: Available document chunks
            
        Returns:
            Tuple of (is_valid, matching_chunk, similarity_score)
        """
        best_match = None
        best_score = 0.0
        
        for chunk in retrieved_chunks:
            # Exact substring match first
            if quoted_text.strip() in chunk.chunk_text:
                return True, chunk, 1.0
                
            # Fuzzy matching for minor variations
            similarity = SequenceMatcher(None, quoted_text.lower().strip(), 
                                       chunk.chunk_text.lower()).ratio()
            
            if similarity > best_score:
                best_score = similarity
                best_match = chunk
                
        is_valid = best_score >= self.similarity_threshold
        return is_valid, best_match if is_valid else None, best_score
    
    def check_temporal_validity(self, chunk: RetrievedChunk) -> Tuple[bool, Optional[str]]:
        """
        Check if source is current and not superseded.
        
        Args:
            chunk: Retrieved chunk to validate
            
        Returns:
            Tuple of (is_current, issue_description)
        """
        if not chunk.pub_date:
            return False, f"No publication date for source {chunk.source_id}"
            
        days_old = (date.today() - chunk.pub_date).days
        
        if days_old > self.staleness_threshold_days:
            return False, f"Source {chunk.source_id} is {days_old} days old (threshold: {self.staleness_threshold_days})"
            
        return True, None
    
    def validate_citation_format(self, citation: str) -> Tuple[bool, Optional[str]]:
        """
        Validate proper regulatory citation format.
        
        Args:
            citation: Citation string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Common regulatory citation patterns
        patterns = [
            r'^[A-Z0-9\s]+\s+(Art\.|Article)\s+[0-9]+.*$',  # GDPR Art. 35
            r'^[0-9]+\s+CFR\s+[0-9]+\.[0-9]+.*$',          # 21 CFR 11.10
            r'^[A-Z]+\s+§\s*[0-9]+.*$',                     # SOX § 404
            r'^ISO\s+[0-9]+.*$',                            # ISO 27001
            r'^[A-Z]+\s+Section\s+[0-9]+.*$',               # HIPAA Section 164
        ]
        
        for pattern in patterns:
            if re.match(pattern, citation.strip()):
                return True, None
                
        return False, f"Citation format not recognized: {citation}"


class ComplianceOutputValidator:
    """Validates compliance outputs against schemas with mandatory grounding."""
    
    def __init__(self, schema_path: str = "schemas/compliance_output_contracts.json"):
        self.citation_checker = CitationChecker()
        self.schema = self._load_schema(schema_path)
        
    def _load_schema(self, schema_path: str) -> Dict[str, Any]:
        """Load JSON schema for validation."""
        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema from {schema_path}: {e}")
            raise
    
    def validate_output(self, 
                       output: Dict[str, Any], 
                       retrieved_chunks: List[RetrievedChunk]) -> GroundingResult:
        """
        Comprehensive validation of compliance output.
        
        Args:
            output: Compliance analysis output to validate
            retrieved_chunks: Available retrieved chunks for grounding
            
        Returns:
            GroundingResult with validation details
        """
        validation_errors = []
        validated_citations = []
        failed_citations = []
        temporal_issues = []
        
        # 1. Schema validation
        try:
            validate(instance=output, schema=self.schema)
        except ValidationError as e:
            validation_errors.append(f"Schema validation failed: {e.message}")
            
        # 2. Citation validation
        if 'citations' in output:
            for citation_data in output['citations']:
                result = self._validate_single_citation(citation_data, retrieved_chunks)
                
                if result['is_valid']:
                    validated_citations.append(result)
                else:
                    failed_citations.append(result)
                    validation_errors.extend(result['errors'])
                    
                # Check temporal validity
                if result.get('chunk'):
                    is_current, issue = self.citation_checker.check_temporal_validity(result['chunk'])
                    if not is_current:
                        temporal_issues.append(issue)
        
        # 3. Mandatory grounding check
        if not output.get('grounding_validated', False):
            validation_errors.append("Grounding validation flag not set")
            
        # 4. Calculate grounding score
        total_citations = len(output.get('citations', []))
        valid_citations = len(validated_citations)
        grounding_score = valid_citations / total_citations if total_citations > 0 else 0.0
        
        # 5. Check conservative approach for uncertainty
        if output.get('confidence', 1.0) < 0.7 and not output.get('conservative_approach_applied', False):
            validation_errors.append("Conservative approach not applied for low confidence output")
            
        is_grounded = (
            len(validation_errors) == 0 and 
            len(failed_citations) == 0 and
            grounding_score >= 0.8 and
            len(temporal_issues) == 0
        )
        
        return GroundingResult(
            is_grounded=is_grounded,
            validated_citations=validated_citations,
            failed_citations=failed_citations,
            temporal_issues=temporal_issues,
            validation_errors=validation_errors,
            grounding_score=grounding_score
        )
    
    def _validate_single_citation(self, 
                                citation_data: Dict[str, Any], 
                                retrieved_chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        """Validate a single citation entry."""
        errors = []
        chunk_text = citation_data.get('chunk_text', '')
        citation_format = citation_data.get('citation', '')
        
        # Verify chunk text appears in retrieved documents
        is_valid, matching_chunk, similarity = self.citation_checker.verify_citation(
            chunk_text, retrieved_chunks
        )
        
        if not is_valid:
            errors.append(f"Chunk text not found in retrieved documents (similarity: {similarity:.2f})")
            
        # Validate citation format
        format_valid, format_error = self.citation_checker.validate_citation_format(citation_format)
        if not format_valid:
            errors.append(format_error)
            
        # Check required fields
        required_fields = ['chunk_text', 'citation', 'pub_date', 'source_id']
        for field in required_fields:
            if not citation_data.get(field):
                errors.append(f"Missing required field: {field}")
                
        return {
            'citation_data': citation_data,
            'is_valid': len(errors) == 0,
            'errors': errors,
            'similarity_score': similarity,
            'chunk': matching_chunk
        }


class GroundingEnforcer:
    """Enforces mandatory grounding in the compliance pipeline."""
    
    def __init__(self, validator: ComplianceOutputValidator):
        self.validator = validator
        
    def enforce_grounding(self, 
                         output: Dict[str, Any], 
                         retrieved_chunks: List[RetrievedChunk]) -> Tuple[bool, Dict[str, Any], GroundingResult]:
        """
        Enforce grounding requirements and apply fallbacks if needed.
        
        Args:
            output: Raw compliance output
            retrieved_chunks: Available chunks for grounding
            
        Returns:
            Tuple of (passed_validation, final_output, grounding_result)
        """
        # Validate grounding
        grounding_result = self.validator.validate_output(output, retrieved_chunks)
        
        if grounding_result.is_grounded:
            # Mark as validated
            output['grounding_validated'] = True
            return True, output, grounding_result
        
        # Apply fallback if grounding fails
        logger.warning(f"Grounding validation failed: {grounding_result.validation_errors}")
        
        # Return with explicit grounding failure flag
        output['grounding_validated'] = False
        output['grounding_errors'] = grounding_result.validation_errors
        output['requires_fallback'] = True
        
        return False, output, grounding_result


def create_sample_retrieved_chunks() -> List[RetrievedChunk]:
    """Create sample retrieved chunks for testing."""
    return [
        RetrievedChunk(
            chunk_text="Personal data shall be processed lawfully, fairly and in a transparent manner in relation to the data subject (lawfulness, fairness and transparency)",
            citation="GDPR Art. 5(1)(a)",
            pub_date=date(2018, 5, 25),
            source_id="GDPR-2016/679",
            authority="European Commission",
            section_granularity="Article 5, paragraph 1, point (a)"
        ),
        RetrievedChunk(
            chunk_text="Each entity subject to the requirements of this subpart shall implement an electronic signature system that ensures the integrity and authenticity of electronic records",
            citation="21 CFR 11.10(a)",
            pub_date=date(1997, 8, 20),
            source_id="21CFR11",
            authority="FDA",
            section_granularity="Section 11.10, subsection (a)"
        )
    ]


if __name__ == "__main__":
    # Example usage
    validator = ComplianceOutputValidator()
    enforcer = GroundingEnforcer(validator)
    
    # Sample output
    sample_output = {
        "analysis_type": "gap_analysis",
        "jurisdictions": [{"code": "EU", "name": "European Union", "effective_date": "2018-05-25"}],
        "effective_dates": ["2018-05-25"],
        "citations": [
            {
                "chunk_text": "Personal data shall be processed lawfully, fairly and in a transparent manner",
                "citation": "GDPR Art. 5(1)(a)",
                "pub_date": "2018-05-25",
                "source_id": "GDPR-2016/679"
            }
        ],
        "risk_rationale": {
            "level": "high",
            "justification": "Non-compliance with data processing principles",
            "evidence_based": True,
            "confidence": 0.9
        },
        "next_actions": [
            {
                "action": "Review data processing activities",
                "owner": "Data Protection Officer",
                "due_date": "2024-01-15",
                "priority": "high"
            }
        ],
        "confidence": 0.85,
        "gaps_identified": [
            {
                "gap_description": "Missing lawfulness assessment for data processing",
                "regulatory_requirement": "GDPR Art. 5(1)(a)",
                "current_state": "No documented lawfulness assessment",
                "target_state": "Documented lawfulness assessment for all processing activities"
            }
        ],
        "compliance_percentage": 75.0,
        "frameworks_assessed": ["GDPR"]
    }
    
    chunks = create_sample_retrieved_chunks()
    passed, final_output, result = enforcer.enforce_grounding(sample_output, chunks)
    
    print(f"Grounding validation passed: {passed}")
    print(f"Grounding score: {result.grounding_score}")
    print(f"Validation errors: {result.validation_errors}")
