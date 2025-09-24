"""
Temporal Awareness and Drift Tests for Regulatory Changes
Handles effective dates, regulatory updates, and temporal validation of compliance analysis.
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class RegulationStatus(Enum):
    """Status of regulatory documents."""
    DRAFT = "draft"
    PROPOSED = "proposed"
    EFFECTIVE = "effective"
    SUPERSEDED = "superseded"
    REPEALED = "repealed"


@dataclass
class RegulatoryUpdate:
    """Represents a regulatory update or change."""
    regulation_id: str
    title: str
    update_type: str  # "amendment", "new", "repeal", "clarification"
    effective_date: date
    announcement_date: date
    supersedes: Optional[str] = None
    key_changes: List[str] = None
    impact_assessment: str = ""
    transition_period_end: Optional[date] = None
    

@dataclass
class TemporalValidationResult:
    """Result of temporal validation."""
    is_current: bool
    is_effective: bool
    has_updates_pending: bool
    staleness_warnings: List[str]
    update_recommendations: List[str]
    superseded_references: List[str]


class RegulatoryTimelineTracker:
    """Tracks regulatory timelines and effective dates."""
    
    def __init__(self):
        self.regulatory_timeline = self._initialize_regulatory_timeline()
        self.recent_updates = self._load_recent_updates()
        
    def _initialize_regulatory_timeline(self) -> Dict[str, Dict[str, Any]]:
        """Initialize regulatory timeline with key dates."""
        
        return {
            "GDPR": {
                "adoption_date": date(2016, 4, 27),
                "effective_date": date(2018, 5, 25),
                "current_version": "2016/679",
                "status": RegulationStatus.EFFECTIVE,
                "major_updates": [
                    {"date": date(2020, 7, 16), "description": "Schrems II decision invalidates Privacy Shield"},
                    {"date": date(2021, 6, 4), "description": "European Commission adequacy decisions updates"},
                    {"date": date(2022, 1, 1), "description": "Brexit transition period ends"}
                ],
                "upcoming_changes": []
            },
            
            "CCPA": {
                "adoption_date": date(2018, 6, 28),
                "effective_date": date(2020, 1, 1),
                "current_version": "1798.100-1798.199",
                "status": RegulationStatus.SUPERSEDED,
                "superseded_by": "CPRA",
                "superseded_date": date(2023, 1, 1),
                "major_updates": [
                    {"date": date(2020, 10, 10), "description": "CCPA Regulations finalized"},
                    {"date": date(2021, 3, 15), "description": "CCPA amended by CPRA"}
                ]
            },
            
            "CPRA": {
                "adoption_date": date(2020, 11, 3),
                "effective_date": date(2023, 1, 1),
                "current_version": "Proposition 24",
                "status": RegulationStatus.EFFECTIVE,
                "supersedes": "CCPA",
                "major_updates": [],
                "upcoming_changes": [
                    {"date": date(2024, 7, 1), "description": "CPCA enforcement begins"}
                ]
            },
            
            "HIPAA": {
                "adoption_date": date(1996, 8, 21),
                "effective_date": date(2003, 4, 14),
                "current_version": "45 CFR Parts 160, 162, 164",
                "status": RegulationStatus.EFFECTIVE,
                "major_updates": [
                    {"date": date(2009, 2, 17), "description": "HITECH Act enhances HIPAA"},
                    {"date": date(2013, 1, 25), "description": "Omnibus Rule updates"},
                    {"date": date(2016, 6, 30), "description": "Breach notification updates"}
                ],
                "upcoming_changes": []
            },
            
            "SOX": {
                "adoption_date": date(2002, 7, 30),
                "effective_date": date(2002, 7, 30),
                "current_version": "Public Law 107-204",
                "status": RegulationStatus.EFFECTIVE,
                "major_updates": [
                    {"date": date(2004, 11, 15), "description": "AS 2201 implementation"},
                    {"date": date(2007, 6, 20), "description": "AS 5 replaces AS 2"}
                ],
                "upcoming_changes": []
            },
            
            # New and emerging regulations
            "EU_AI_ACT": {
                "adoption_date": date(2024, 5, 21),
                "effective_date": date(2026, 8, 2),
                "current_version": "2024/1689",
                "status": RegulationStatus.PROPOSED,
                "major_updates": [],
                "upcoming_changes": [
                    {"date": date(2025, 8, 2), "description": "Governance provisions apply"},
                    {"date": date(2026, 2, 2), "description": "Prohibited AI systems ban"},
                    {"date": date(2026, 8, 2), "description": "Full regulation effective"}
                ]
            }
        }
    
    def _load_recent_updates(self) -> List[RegulatoryUpdate]:
        """Load recent regulatory updates."""
        
        return [
            RegulatoryUpdate(
                regulation_id="GDPR",
                title="Data Protection Board Guidelines on Dark Patterns",
                update_type="clarification",
                effective_date=date(2023, 2, 14),
                announcement_date=date(2022, 11, 14),
                key_changes=[
                    "Clarifies dark patterns in consent mechanisms",
                    "Provides specific examples of non-compliant UI patterns",
                    "Updates guidance on consent withdrawal"
                ],
                impact_assessment="Medium - affects consent interface design"
            ),
            
            RegulatoryUpdate(
                regulation_id="CPRA",
                title="California Privacy Protection Agency Regulations",
                update_type="new",
                effective_date=date(2023, 3, 29),
                announcement_date=date(2022, 12, 8),
                key_changes=[
                    "Defines sensitive personal information processing",
                    "Clarifies consumer rights procedures",
                    "Establishes audit and risk assessment requirements"
                ],
                impact_assessment="High - new compliance obligations"
            ),
            
            RegulatoryUpdate(
                regulation_id="EU_AI_ACT",
                title="Artificial Intelligence Act Adoption",
                update_type="new",
                effective_date=date(2026, 8, 2),
                announcement_date=date(2024, 5, 21),
                key_changes=[
                    "Prohibits certain AI systems",
                    "Establishes high-risk AI system requirements",
                    "Creates governance framework for AI compliance"
                ],
                impact_assessment="Critical - new regulatory framework for AI"
            )
        ]
    
    def validate_temporal_compliance(self, 
                                   citations: List[Dict[str, Any]],
                                   analysis_date: date = None) -> TemporalValidationResult:
        """
        Validate temporal compliance of citations and analysis.
        
        Args:
            citations: List of regulatory citations to validate
            analysis_date: Date of analysis (defaults to today)
            
        Returns:
            TemporalValidationResult with validation details
        """
        if analysis_date is None:
            analysis_date = date.today()
            
        staleness_warnings = []
        update_recommendations = []
        superseded_references = []
        
        for citation in citations:
            pub_date_str = citation.get('pub_date')
            source_id = citation.get('source_id', 'Unknown')
            
            if not pub_date_str:
                staleness_warnings.append(f"No publication date for {source_id}")
                continue
                
            try:
                pub_date = datetime.strptime(pub_date_str, '%Y-%m-%d').date()
            except ValueError:
                staleness_warnings.append(f"Invalid date format for {source_id}: {pub_date_str}")
                continue
                
            # Check age
            age_days = (analysis_date - pub_date).days
            if age_days > 1825:  # 5 years
                staleness_warnings.append(f"Source {source_id} is {age_days} days old (published {pub_date})")
                
            # Check for superseded regulations
            regulation = self._identify_regulation_from_citation(citation)
            if regulation:
                timeline = self.regulatory_timeline.get(regulation)
                if timeline and timeline.get('status') == RegulationStatus.SUPERSEDED:
                    superseded_by = timeline.get('superseded_by')
                    superseded_date = timeline.get('superseded_date')
                    superseded_references.append(
                        f"{source_id} references {regulation} which was superseded by {superseded_by} on {superseded_date}"
                    )
                    
                # Check for pending updates
                if timeline and timeline.get('upcoming_changes'):
                    upcoming = timeline['upcoming_changes']
                    for change in upcoming:
                        change_date = change['date']
                        if change_date > analysis_date:
                            update_recommendations.append(
                                f"Upcoming change to {regulation} on {change_date}: {change['description']}"
                            )
        
        # Check for recent regulatory updates that might affect analysis
        relevant_updates = self._find_relevant_updates(citations, analysis_date)
        for update in relevant_updates:
            update_recommendations.append(
                f"Recent update to {update.regulation_id} (effective {update.effective_date}): {update.title}"
            )
        
        is_current = len(staleness_warnings) == 0
        is_effective = len(superseded_references) == 0
        has_updates_pending = len(update_recommendations) > 0
        
        return TemporalValidationResult(
            is_current=is_current,
            is_effective=is_effective,
            has_updates_pending=has_updates_pending,
            staleness_warnings=staleness_warnings,
            update_recommendations=update_recommendations,
            superseded_references=superseded_references
        )
    
    def _identify_regulation_from_citation(self, citation: Dict[str, Any]) -> Optional[str]:
        """Identify regulation from citation."""
        
        citation_text = citation.get('citation', '').upper()
        source_id = citation.get('source_id', '').upper()
        
        if 'GDPR' in citation_text or 'GDPR' in source_id:
            return 'GDPR'
        elif 'CCPA' in citation_text or 'CCPA' in source_id:
            return 'CCPA'
        elif 'CPRA' in citation_text or 'CPRA' in source_id:
            return 'CPRA'
        elif 'HIPAA' in citation_text or 'HIPAA' in source_id:
            return 'HIPAA'
        elif 'SOX' in citation_text or 'SOX' in source_id:
            return 'SOX'
        elif 'AI ACT' in citation_text or 'AI' in source_id:
            return 'EU_AI_ACT'
            
        return None
    
    def _find_relevant_updates(self, 
                              citations: List[Dict[str, Any]], 
                              analysis_date: date) -> List[RegulatoryUpdate]:
        """Find recent updates relevant to the citations."""
        
        relevant_regulations = set()
        for citation in citations:
            regulation = self._identify_regulation_from_citation(citation)
            if regulation:
                relevant_regulations.add(regulation)
                
        relevant_updates = []
        cutoff_date = analysis_date - timedelta(days=365)  # Updates in last year
        
        for update in self.recent_updates:
            if (update.regulation_id in relevant_regulations and 
                update.effective_date >= cutoff_date):
                relevant_updates.append(update)
                
        return relevant_updates
    
    def generate_drift_test_cases(self) -> List[Dict[str, Any]]:
        """Generate test cases for temporal drift detection."""
        
        test_cases = []
        
        # Test case 1: Outdated regulation reference
        test_cases.append({
            "name": "outdated_ccpa_reference",
            "description": "Analysis references CCPA after CPRA supersession",
            "citations": [
                {
                    "citation": "CCPA Section 1798.100",
                    "pub_date": "2020-01-01",
                    "source_id": "CCPA-1798.100"
                }
            ],
            "analysis_date": date(2023, 6, 1),
            "expected_warnings": ["superseded"],
            "expected_recommendations": ["Update to CPRA references"]
        })
        
        # Test case 2: Recent update not reflected
        test_cases.append({
            "name": "missing_recent_update",
            "description": "Analysis doesn't reflect recent GDPR guidance updates",
            "citations": [
                {
                    "citation": "GDPR Art. 7",
                    "pub_date": "2018-05-25",
                    "source_id": "GDPR-2016/679"
                }
            ],
            "analysis_date": date(2023, 3, 1),
            "expected_warnings": [],
            "expected_recommendations": ["Consider recent dark patterns guidance"]
        })
        
        # Test case 3: Future regulation preparation
        test_cases.append({
            "name": "upcoming_ai_act",
            "description": "AI system analysis should consider upcoming EU AI Act",
            "citations": [
                {
                    "citation": "GDPR Art. 22",
                    "pub_date": "2018-05-25", 
                    "source_id": "GDPR-2016/679"
                }
            ],
            "analysis_date": date(2025, 1, 1),
            "expected_warnings": [],
            "expected_recommendations": ["Upcoming EU AI Act requirements"]
        })
        
        # Test case 4: Very old source
        test_cases.append({
            "name": "stale_source",
            "description": "Analysis uses very old regulatory source",
            "citations": [
                {
                    "citation": "Privacy Act 1974",
                    "pub_date": "1974-12-31",
                    "source_id": "PRIVACY-ACT-1974"
                }
            ],
            "analysis_date": date(2024, 1, 1),
            "expected_warnings": ["staleness"],
            "expected_recommendations": ["Verify current applicability"]
        })
        
        return test_cases
    
    def run_drift_tests(self, test_cases: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run temporal drift tests."""
        
        if test_cases is None:
            test_cases = self.generate_drift_test_cases()
            
        results = {
            "total_tests": len(test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": []
        }
        
        for test_case in test_cases:
            result = self.validate_temporal_compliance(
                test_case["citations"],
                test_case["analysis_date"]
            )
            
            # Check if expected warnings/recommendations are present
            test_passed = True
            test_details = {
                "name": test_case["name"],
                "description": test_case["description"],
                "passed": True,
                "issues": []
            }
            
            # Check for expected warnings
            for expected_warning in test_case.get("expected_warnings", []):
                found = False
                if expected_warning == "superseded" and result.superseded_references:
                    found = True
                elif expected_warning == "staleness" and result.staleness_warnings:
                    found = True
                    
                if not found:
                    test_passed = False
                    test_details["issues"].append(f"Expected warning '{expected_warning}' not found")
            
            # Check for expected recommendations  
            for expected_rec in test_case.get("expected_recommendations", []):
                found = any(expected_rec.lower() in rec.lower() 
                           for rec in result.update_recommendations)
                if not found:
                    test_passed = False
                    test_details["issues"].append(f"Expected recommendation '{expected_rec}' not found")
            
            test_details["passed"] = test_passed
            test_details["validation_result"] = {
                "is_current": result.is_current,
                "is_effective": result.is_effective,
                "warnings_count": len(result.staleness_warnings),
                "recommendations_count": len(result.update_recommendations),
                "superseded_count": len(result.superseded_references)
            }
            
            results["test_results"].append(test_details)
            
            if test_passed:
                results["passed_tests"] += 1
            else:
                results["failed_tests"] += 1
                
        return results


class TemporalAwarenessEvaluator:
    """Evaluates models for temporal awareness in compliance analysis."""
    
    def __init__(self):
        self.timeline_tracker = RegulatoryTimelineTracker()
        
    def evaluate_temporal_awareness(self, 
                                  model_output: Dict[str, Any],
                                  evaluation_date: date = None) -> Dict[str, Any]:
        """
        Evaluate model output for temporal awareness.
        
        Args:
            model_output: Compliance analysis output from model
            evaluation_date: Date to evaluate against
            
        Returns:
            Evaluation results with temporal awareness scores
        """
        if evaluation_date is None:
            evaluation_date = date.today()
            
        citations = model_output.get('citations', [])
        
        # Validate temporal compliance
        temporal_result = self.timeline_tracker.validate_temporal_compliance(
            citations, evaluation_date
        )
        
        # Score temporal awareness
        temporal_score = self._calculate_temporal_score(temporal_result, model_output)
        
        # Check if model requests updated sources
        update_awareness = self._check_update_awareness(model_output, temporal_result)
        
        return {
            "temporal_compliance": {
                "is_current": temporal_result.is_current,
                "is_effective": temporal_result.is_effective,
                "has_updates_pending": temporal_result.has_updates_pending
            },
            "temporal_awareness_score": temporal_score,
            "update_awareness": update_awareness,
            "warnings": temporal_result.staleness_warnings,
            "recommendations": temporal_result.update_recommendations,
            "superseded_references": temporal_result.superseded_references,
            "evaluation_date": evaluation_date.isoformat()
        }
    
    def _calculate_temporal_score(self, 
                                temporal_result: TemporalValidationResult,
                                model_output: Dict[str, Any]) -> float:
        """Calculate temporal awareness score (0.0 to 1.0)."""
        
        score = 1.0
        
        # Penalize for temporal issues
        if not temporal_result.is_current:
            score -= 0.3
        if not temporal_result.is_effective:
            score -= 0.4
        if temporal_result.has_updates_pending and not self._acknowledges_limitations(model_output):
            score -= 0.2
            
        # Bonus for acknowledging temporal limitations
        if self._acknowledges_temporal_context(model_output):
            score += 0.1
            
        return max(0.0, min(1.0, score))
    
    def _check_update_awareness(self, 
                              model_output: Dict[str, Any],
                              temporal_result: TemporalValidationResult) -> bool:
        """Check if model demonstrates awareness of need for updates."""
        
        # Look for indicators of temporal awareness
        analysis_text = str(model_output).lower()
        
        temporal_indicators = [
            "current regulations",
            "latest version",
            "recent updates",
            "effective date",
            "superseded",
            "amended",
            "verify current"
        ]
        
        has_temporal_language = any(indicator in analysis_text for indicator in temporal_indicators)
        
        # Check if model requests updates when needed
        requests_updates = (
            temporal_result.has_updates_pending and
            any(phrase in analysis_text for phrase in ["update", "current", "latest", "verify"])
        )
        
        return has_temporal_language or requests_updates
    
    def _acknowledges_limitations(self, model_output: Dict[str, Any]) -> bool:
        """Check if model acknowledges temporal limitations."""
        
        uncertainty_notice = model_output.get('uncertainty_notice', '').lower()
        notes = str(model_output.get('notes', '')).lower()
        
        limitation_phrases = [
            "as of",
            "current as of",
            "subject to regulatory updates",
            "verify current requirements",
            "consult latest regulations"
        ]
        
        return any(phrase in uncertainty_notice or phrase in notes 
                  for phrase in limitation_phrases)
    
    def _acknowledges_temporal_context(self, model_output: Dict[str, Any]) -> bool:
        """Check if model shows temporal context awareness."""
        
        citations = model_output.get('citations', [])
        
        # Check if citations include effective dates
        has_effective_dates = any('effective' in str(citation).lower() 
                                 for citation in citations)
        
        # Check if analysis mentions temporal context
        analysis_text = str(model_output).lower()
        temporal_context = any(phrase in analysis_text for phrase in [
            "effective date",
            "implementation date", 
            "enforcement date",
            "transition period",
            "phased implementation"
        ])
        
        return has_effective_dates or temporal_context


if __name__ == "__main__":
    # Demo temporal awareness
    tracker = RegulatoryTimelineTracker()
    evaluator = TemporalAwarenessEvaluator()
    
    # Run drift tests
    drift_results = tracker.run_drift_tests()
    print("=== Temporal Drift Test Results ===")
    print(f"Total tests: {drift_results['total_tests']}")
    print(f"Passed: {drift_results['passed_tests']}")
    print(f"Failed: {drift_results['failed_tests']}")
    
    for test in drift_results['test_results']:
        status = "✅" if test['passed'] else "❌"
        print(f"{status} {test['name']}: {test['description']}")
        if not test['passed']:
            print(f"  Issues: {test['issues']}")
    
    # Example temporal validation
    print("\n=== Example Temporal Validation ===")
    sample_citations = [
        {
            "citation": "CCPA Section 1798.100",
            "pub_date": "2020-01-01",
            "source_id": "CCPA-1798.100"
        }
    ]
    
    result = tracker.validate_temporal_compliance(sample_citations, date(2023, 6, 1))
    print(f"Is current: {result.is_current}")
    print(f"Is effective: {result.is_effective}")
    print(f"Superseded references: {result.superseded_references}")
    print(f"Update recommendations: {result.update_recommendations}")
