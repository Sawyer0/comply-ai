"""
Tool Hooks for Compliance AI Training and Inference
Simulates external tools that the compliance AI will use in production.
"""

import json
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for regulatory documents."""
    doc_id: str
    title: str
    authority: str
    jurisdiction: str
    effective_date: date
    superseded_date: Optional[date] = None
    version: str = "1.0"
    source_url: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result from document retrieval."""
    chunk_text: str
    citation: str
    pub_date: date
    source_id: str
    authority: str
    section_granularity: str
    confidence_score: float
    metadata: DocumentMetadata


class RegulatoryRetrieval:
    """Simulates retrieval system with date filters and document IDs."""
    
    def __init__(self):
        self.knowledge_base = self._initialize_knowledge_base()
        
    def _initialize_knowledge_base(self) -> List[RetrievalResult]:
        """Initialize simulated knowledge base."""
        
        return [
            # GDPR Examples
            RetrievalResult(
                chunk_text="Personal data shall be processed lawfully, fairly and in a transparent manner in relation to the data subject (lawfulness, fairness and transparency)",
                citation="GDPR Art. 5(1)(a)",
                pub_date=date(2018, 5, 25),
                source_id="GDPR-2016/679",
                authority="European Commission",
                section_granularity="Article 5, paragraph 1, point (a)",
                confidence_score=0.95,
                metadata=DocumentMetadata(
                    doc_id="GDPR-2016/679",
                    title="General Data Protection Regulation",
                    authority="European Commission",
                    jurisdiction="EU",
                    effective_date=date(2018, 5, 25),
                    version="2016/679"
                )
            ),
            
            RetrievalResult(
                chunk_text="Where processing is carried out for the purpose of the legitimate interests pursued by the controller or by a third party, a balancing test shall be carried out",
                citation="GDPR Art. 6(1)(f)",
                pub_date=date(2018, 5, 25),
                source_id="GDPR-2016/679",
                authority="European Commission",
                section_granularity="Article 6, paragraph 1, point (f)",
                confidence_score=0.92,
                metadata=DocumentMetadata(
                    doc_id="GDPR-2016/679",
                    title="General Data Protection Regulation",
                    authority="European Commission",
                    jurisdiction="EU",
                    effective_date=date(2018, 5, 25)
                )
            ),
            
            # HIPAA Examples
            RetrievalResult(
                chunk_text="A covered entity must implement technical safeguards to guard against unauthorized access to electronic protected health information",
                citation="HIPAA 45 CFR 164.312(a)(1)",
                pub_date=date(2003, 4, 14),
                source_id="HIPAA-164.312",
                authority="HHS",
                section_granularity="Section 164.312, subsection (a)(1)",
                confidence_score=0.93,
                metadata=DocumentMetadata(
                    doc_id="HIPAA-164.312",
                    title="HIPAA Security Rule",
                    authority="Department of Health and Human Services",
                    jurisdiction="US",
                    effective_date=date(2003, 4, 14)
                )
            ),
            
            # SOX Examples
            RetrievalResult(
                chunk_text="Management must assess the effectiveness of the company's internal control over financial reporting as of the end of each fiscal year",
                citation="SOX Section 404(a)",
                pub_date=date(2002, 7, 30),
                source_id="SOX-404",
                authority="SEC",
                section_granularity="Section 404, subsection (a)",
                confidence_score=0.91,
                metadata=DocumentMetadata(
                    doc_id="SOX-404",
                    title="Sarbanes-Oxley Act Section 404",
                    authority="Securities and Exchange Commission",
                    jurisdiction="US",
                    effective_date=date(2002, 7, 30)
                )
            ),
            
            # FDA 21 CFR Examples
            RetrievalResult(
                chunk_text="Electronic records that are required to be maintained under predicate rules or submitted to FDA, may be maintained in electronic format in lieu of paper format",
                citation="21 CFR 11.1(a)",
                pub_date=date(1997, 8, 20),
                source_id="21CFR11",
                authority="FDA",
                section_granularity="Section 11.1, subsection (a)",
                confidence_score=0.94,
                metadata=DocumentMetadata(
                    doc_id="21CFR11",
                    title="Electronic Records; Electronic Signatures",
                    authority="Food and Drug Administration",
                    jurisdiction="US",
                    effective_date=date(1997, 8, 20)
                )
            ),
            
            # ISO 27001 Examples
            RetrievalResult(
                chunk_text="The organization shall establish, implement, maintain and continually improve an information security management system",
                citation="ISO 27001:2013 Clause 4.4",
                pub_date=date(2013, 10, 1),
                source_id="ISO27001-2013",
                authority="ISO",
                section_granularity="Clause 4.4",
                confidence_score=0.89,
                metadata=DocumentMetadata(
                    doc_id="ISO27001-2013",
                    title="Information Security Management Systems",
                    authority="International Organization for Standardization",
                    jurisdiction="GLOBAL",
                    effective_date=date(2013, 10, 1)
                )
            )
        ]
    
    def retrieve_with_filters(self, 
                            query: str,
                            date_filter: Optional[Tuple[date, date]] = None,
                            jurisdiction: Optional[str] = None,
                            authority: Optional[str] = None,
                            top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve documents with date and metadata filters.
        
        Args:
            query: Search query
            date_filter: Tuple of (start_date, end_date) for filtering
            jurisdiction: Filter by jurisdiction (EU, US, etc.)
            authority: Filter by regulatory authority
            top_k: Maximum number of results
            
        Returns:
            List of retrieval results
        """
        results = []
        
        for doc in self.knowledge_base:
            # Date filter
            if date_filter:
                start_date, end_date = date_filter
                if not (start_date <= doc.pub_date <= end_date):
                    continue
                    
            # Jurisdiction filter
            if jurisdiction and doc.metadata.jurisdiction != jurisdiction:
                continue
                
            # Authority filter
            if authority and authority.lower() not in doc.authority.lower():
                continue
                
            # Simple text matching (in production this would be semantic search)
            query_lower = query.lower()
            if (query_lower in doc.chunk_text.lower() or 
                query_lower in doc.citation.lower()):
                results.append(doc)
                
        # Sort by confidence score and return top_k
        results.sort(key=lambda x: x.confidence_score, reverse=True)
        return results[:top_k]


class CitationStringMatcher:
    """Citation checker with string matching and fuzzy bounds."""
    
    def __init__(self, exact_threshold: float = 0.95, fuzzy_threshold: float = 0.85):
        self.exact_threshold = exact_threshold
        self.fuzzy_threshold = fuzzy_threshold
        
    def check_citation(self, 
                      quoted_text: str, 
                      retrieved_chunks: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Check if quoted text appears in retrieved chunks.
        
        Args:
            quoted_text: Text that should appear in chunks
            retrieved_chunks: Retrieved document chunks
            
        Returns:
            Dictionary with validation results
        """
        best_match = None
        best_score = 0.0
        exact_match = False
        
        for chunk in retrieved_chunks:
            # Check for exact substring match
            if quoted_text.strip() in chunk.chunk_text:
                exact_match = True
                best_match = chunk
                best_score = 1.0
                break
                
            # Fuzzy matching
            similarity = SequenceMatcher(
                None, 
                quoted_text.lower().strip(), 
                chunk.chunk_text.lower()
            ).ratio()
            
            if similarity > best_score:
                best_score = similarity
                best_match = chunk
                
        validation_result = {
            "is_valid": exact_match or best_score >= self.fuzzy_threshold,
            "exact_match": exact_match,
            "similarity_score": best_score,
            "matching_chunk": asdict(best_match) if best_match else None,
            "validation_method": "exact" if exact_match else "fuzzy" if best_score >= self.fuzzy_threshold else "failed"
        }
        
        return validation_result


class EffectiveDateCalculator:
    """Calculator for windowed applicability of regulations."""
    
    def __init__(self):
        self.regulatory_timelines = self._load_regulatory_timelines()
        
    def _load_regulatory_timelines(self) -> Dict[str, Dict[str, Any]]:
        """Load regulatory timeline information."""
        
        return {
            "GDPR": {
                "adoption_date": date(2016, 4, 27),
                "effective_date": date(2018, 5, 25),
                "grace_period_months": 24,
                "enforcement_delay_months": 0
            },
            "CCPA": {
                "adoption_date": date(2018, 6, 28),
                "effective_date": date(2020, 1, 1),
                "grace_period_months": 18,
                "enforcement_delay_months": 6
            },
            "HIPAA": {
                "adoption_date": date(1996, 8, 21),
                "effective_date": date(2003, 4, 14),
                "grace_period_months": 24,
                "enforcement_delay_months": 0
            },
            "SOX": {
                "adoption_date": date(2002, 7, 30),
                "effective_date": date(2002, 7, 30),
                "grace_period_months": 0,
                "enforcement_delay_months": 0
            }
        }
    
    def calculate_applicability_window(self, 
                                     regulation: str, 
                                     assessment_date: date) -> Dict[str, Any]:
        """
        Calculate if regulation is applicable on given date.
        
        Args:
            regulation: Regulation identifier (e.g., "GDPR")
            assessment_date: Date to assess applicability
            
        Returns:
            Dictionary with applicability information
        """
        if regulation not in self.regulatory_timelines:
            return {
                "is_applicable": None,
                "status": "unknown_regulation",
                "message": f"No timeline data for regulation: {regulation}"
            }
            
        timeline = self.regulatory_timelines[regulation]
        effective_date = timeline["effective_date"]
        
        # Calculate enforcement start date
        enforcement_delay = timedelta(days=timeline["enforcement_delay_months"] * 30)
        enforcement_start = effective_date + enforcement_delay
        
        if assessment_date < effective_date:
            return {
                "is_applicable": False,
                "status": "not_yet_effective",
                "effective_date": effective_date.isoformat(),
                "days_until_effective": (effective_date - assessment_date).days,
                "message": f"{regulation} not yet effective on {assessment_date}"
            }
        elif assessment_date < enforcement_start:
            return {
                "is_applicable": True,
                "status": "grace_period",
                "effective_date": effective_date.isoformat(),
                "enforcement_start": enforcement_start.isoformat(),
                "days_until_enforcement": (enforcement_start - assessment_date).days,
                "message": f"{regulation} effective but in grace period"
            }
        else:
            return {
                "is_applicable": True,
                "status": "fully_enforced",
                "effective_date": effective_date.isoformat(),
                "enforcement_start": enforcement_start.isoformat(),
                "message": f"{regulation} fully effective and enforced"
            }


class OPAPolicyGenerator:
    """Generates OPA/Rego policies from compliance analysis."""
    
    def __init__(self):
        self.policy_templates = self._load_policy_templates()
        
    def _load_policy_templates(self) -> Dict[str, str]:
        """Load OPA policy templates."""
        
        return {
            "data_processing_consent": """
package compliance.data_processing

import future.keywords.if

# Data processing requires valid consent
allow_processing if {
    input.has_consent == true
    input.consent_type in ["explicit", "informed"]
    input.purpose_limitation == true
}

# Deny processing without consent
deny_processing if {
    input.has_consent == false
}

# Require consent verification
consent_verification_required if {
    input.data_type in ["personal", "sensitive"]
    not input.consent_verified
}
""",
            
            "access_control": """
package compliance.access_control

import future.keywords.if

# Allow access with proper authorization
allow_access if {
    input.user.authenticated == true
    input.user.role in data.authorized_roles[input.resource]
    input.user.mfa_verified == true
}

# Deny access for unauthorized users
deny_access if {
    input.user.authenticated == false
}

# Require additional verification for sensitive data
additional_verification_required if {
    input.resource.classification == "sensitive"
    not input.user.additional_verification
}
""",
            
            "audit_logging": """
package compliance.audit_logging

import future.keywords.if

# Require audit logging for sensitive operations
audit_required if {
    input.operation in ["create", "update", "delete", "access"]
    input.resource.requires_audit == true
}

# Log access to personal data
log_personal_data_access if {
    input.data_type == "personal"
    input.operation in ["read", "process", "share"]
}

# Retention policy for audit logs
log_retention_days := 2555 if {  # 7 years
    input.compliance_framework == "SOX"
}

log_retention_days := 2190 if {  # 6 years
    input.compliance_framework == "GDPR"
}
"""
        }
    
    def generate_policy_from_analysis(self, 
                                    analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate OPA policy based on compliance analysis.
        
        Args:
            analysis: Compliance analysis output
            
        Returns:
            Dictionary with generated policies
        """
        policies = {}
        
        # Determine policy type based on analysis
        analysis_type = analysis.get("analysis_type", "")
        frameworks = analysis.get("frameworks_assessed", [])
        
        # Generate data processing policy for privacy frameworks
        if any(fw in ["GDPR", "CCPA", "HIPAA"] for fw in frameworks):
            policies["data_processing"] = self._generate_data_processing_policy(analysis)
            
        # Generate access control policy for security frameworks
        if any(fw in ["SOX", "ISO27001", "HIPAA"] for fw in frameworks):
            policies["access_control"] = self._generate_access_control_policy(analysis)
            
        # Generate audit policy for compliance frameworks
        if any(fw in ["SOX", "HIPAA", "21CFR11"] for fw in frameworks):
            policies["audit_logging"] = self._generate_audit_policy(analysis)
            
        return policies
    
    def _generate_data_processing_policy(self, analysis: Dict[str, Any]) -> str:
        """Generate data processing policy based on analysis."""
        
        # Extract requirements from analysis
        requirements = []
        if "consent" in str(analysis).lower():
            requirements.append("input.has_consent == true")
        if "purpose limitation" in str(analysis).lower():
            requirements.append("input.purpose_limitation == true")
        if "data minimization" in str(analysis).lower():
            requirements.append("input.data_minimization == true")
            
        policy = f"""
package compliance.data_processing

import future.keywords.if

# Generated from compliance analysis
allow_processing if {{
    {chr(10).join(f"    {req}" for req in requirements)}
}}

# Deny processing if requirements not met
deny_processing if {{
    not allow_processing
}}
"""
        return policy.strip()
    
    def _generate_access_control_policy(self, analysis: Dict[str, Any]) -> str:
        """Generate access control policy based on analysis."""
        
        return self.policy_templates["access_control"]
    
    def _generate_audit_policy(self, analysis: Dict[str, Any]) -> str:
        """Generate audit policy based on analysis."""
        
        return self.policy_templates["audit_logging"]


class TextRedactionTool:
    """Redacts sensitive information from text before logging."""
    
    def __init__(self):
        self.redaction_patterns = self._load_redaction_patterns()
        
    def _load_redaction_patterns(self) -> Dict[str, str]:
        """Load regex patterns for sensitive data detection."""
        
        return {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "api_key": r'\b[A-Za-z0-9]{32,}\b'
        }
    
    def redact_text(self, text: str) -> Tuple[str, List[str]]:
        """
        Redact sensitive information from text.
        
        Args:
            text: Text to redact
            
        Returns:
            Tuple of (redacted_text, redaction_types)
        """
        redacted_text = text
        redaction_types = []
        
        for redaction_type, pattern in self.redaction_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                redacted_text = re.sub(pattern, f"[REDACTED_{redaction_type.upper()}]", redacted_text)
                redaction_types.append(redaction_type)
                
        return redacted_text, redaction_types


# Example usage functions for notebooks
def simulate_retrieval_with_filters(query: str, 
                                  date_range: Optional[Tuple[str, str]] = None,
                                  jurisdiction: Optional[str] = None) -> List[Dict[str, Any]]:
    """Simulate retrieval for notebook training."""
    
    retrieval = RegulatoryRetrieval()
    
    # Convert string dates to date objects if provided
    date_filter = None
    if date_range:
        start_str, end_str = date_range
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
        date_filter = (start_date, end_date)
    
    results = retrieval.retrieve_with_filters(
        query=query,
        date_filter=date_filter,
        jurisdiction=jurisdiction,
        top_k=3
    )
    
    return [asdict(result) for result in results]


def simulate_citation_checking(quoted_text: str, 
                             retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simulate citation checking for notebook training."""
    
    checker = CitationStringMatcher()
    
    # Convert dict chunks back to RetrievalResult objects
    chunk_objects = []
    for chunk_dict in retrieved_chunks:
        if 'metadata' in chunk_dict:
            metadata = DocumentMetadata(**chunk_dict['metadata'])
        else:
            metadata = DocumentMetadata(
                doc_id=chunk_dict.get('source_id', 'unknown'),
                title="Unknown Document",
                authority=chunk_dict.get('authority', 'Unknown'),
                jurisdiction="Unknown",
                effective_date=datetime.strptime(chunk_dict.get('pub_date', '2020-01-01'), '%Y-%m-%d').date()
            )
            
        chunk_obj = RetrievalResult(
            chunk_text=chunk_dict['chunk_text'],
            citation=chunk_dict['citation'],
            pub_date=datetime.strptime(chunk_dict['pub_date'], '%Y-%m-%d').date(),
            source_id=chunk_dict['source_id'],
            authority=chunk_dict['authority'],
            section_granularity=chunk_dict.get('section_granularity', ''),
            confidence_score=chunk_dict.get('confidence_score', 0.0),
            metadata=metadata
        )
        chunk_objects.append(chunk_obj)
    
    return checker.check_citation(quoted_text, chunk_objects)


def simulate_policy_generation(analysis: Dict[str, Any]) -> Dict[str, str]:
    """Simulate OPA policy generation for notebook training."""
    
    generator = OPAPolicyGenerator()
    return generator.generate_policy_from_analysis(analysis)


if __name__ == "__main__":
    # Example usage
    print("=== Retrieval Simulation ===")
    results = simulate_retrieval_with_filters(
        query="data processing consent",
        date_range=("2018-01-01", "2024-01-01"),
        jurisdiction="EU"
    )
    print(f"Found {len(results)} results")
    
    print("\n=== Citation Checking ===")
    citation_result = simulate_citation_checking(
        quoted_text="Personal data shall be processed lawfully",
        retrieved_chunks=results
    )
    print(f"Citation valid: {citation_result['is_valid']}")
    
    print("\n=== Policy Generation ===")
    sample_analysis = {
        "analysis_type": "gap_analysis",
        "frameworks_assessed": ["GDPR", "CCPA"]
    }
    policies = simulate_policy_generation(sample_analysis)
    print(f"Generated {len(policies)} policies: {list(policies.keys())}")
