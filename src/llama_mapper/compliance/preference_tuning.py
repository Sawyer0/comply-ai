"""
Behavioral Preference Tuning for Compliance AI
Implements DPO/ORPO training focused on senior analyst behavior, not facts.
"""

import json
from datetime import date, datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreferenceExample:
    """A single preference training example with chosen and rejected responses."""
    prompt: str
    chosen_response: str
    rejected_response: str
    preference_reason: str
    behavioral_criteria: List[str]
    

@dataclass
class BehavioralRubric:
    """Rubric for evaluating compliance analyst behavior."""
    criterion: str
    description: str
    good_indicators: List[str]
    bad_indicators: List[str]
    weight: float


class ComplianceBehavioralRubrics:
    """Behavioral rubrics for senior compliance analyst behavior."""
    
    def __init__(self):
        self.rubrics = self._load_behavioral_rubrics()
        
    def _load_behavioral_rubrics(self) -> List[BehavioralRubric]:
        """Load behavioral evaluation rubrics."""
        
        return [
            BehavioralRubric(
                criterion="citation_discipline",
                description="Cites correctly with specific regulatory references",
                good_indicators=[
                    "Provides specific regulatory citations with section numbers",
                    "Includes effective dates and jurisdictions",
                    "Distinguishes between direct quotes and paraphrasing",
                    "Acknowledges when citations are unavailable"
                ],
                bad_indicators=[
                    "Makes regulatory claims without citations",
                    "Uses vague references like 'the law says'",
                    "Conflates different regulatory frameworks",
                    "Provides outdated or superseded citations"
                ],
                weight=0.25
            ),
            
            BehavioralRubric(
                criterion="jurisdiction_awareness",
                description="Names jurisdiction and dates explicitly",
                good_indicators=[
                    "Explicitly states applicable jurisdictions",
                    "Specifies effective dates and regulatory versions",
                    "Acknowledges jurisdictional limitations",
                    "Identifies cross-jurisdictional conflicts"
                ],
                bad_indicators=[
                    "Assumes universal applicability",
                    "Ignores jurisdictional scope",
                    "Uses generic 'compliance' without framework",
                    "Mixes jurisdictional requirements without noting differences"
                ],
                weight=0.20
            ),
            
            BehavioralRubric(
                criterion="evidence_discipline",
                description="Asks for evidence when facts are thin",
                good_indicators=[
                    "Explicitly requests missing evidence",
                    "Acknowledges limitations in available information", 
                    "Distinguishes between evidence-based and assumption-based analysis",
                    "Provides specific evidence requirements"
                ],
                bad_indicators=[
                    "Makes definitive claims without evidence",
                    "Fills gaps with assumptions without noting them",
                    "Overconfident conclusions with limited data",
                    "Generic evidence requests without specificity"
                ],
                weight=0.20
            ),
            
            BehavioralRubric(
                criterion="remediation_specificity", 
                description="Offers specific, testable remediations",
                good_indicators=[
                    "Provides actionable recommendations with clear steps",
                    "Includes responsible parties and timelines",
                    "Specifies success criteria and measurement methods",
                    "Considers implementation complexity and dependencies"
                ],
                bad_indicators=[
                    "Vague recommendations like 'improve compliance'",
                    "No clear ownership or timelines",
                    "Unrealistic or untestable suggestions",
                    "Generic boilerplate recommendations"
                ],
                weight=0.20
            ),
            
            BehavioralRubric(
                criterion="conservative_risk_posture",
                description="Uses conservative risk posture appropriately",
                good_indicators=[
                    "Applies conservative assumptions when uncertain",
                    "Escalates high-risk scenarios appropriately",
                    "Acknowledges uncertainty and limitations explicitly",
                    "Recommends expert consultation when appropriate"
                ],
                bad_indicators=[
                    "Minimizes risks without adequate evidence",
                    "Overconfident assessments with uncertainty",
                    "Fails to escalate high-risk scenarios",
                    "Dismisses the need for expert consultation"
                ],
                weight=0.15
            )
        ]
    
    def evaluate_response(self, response: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate response against behavioral rubrics.
        
        Args:
            response: Compliance analysis response
            context: Context information for evaluation
            
        Returns:
            Dictionary with scores for each criterion
        """
        scores = {}
        
        for rubric in self.rubrics:
            score = self._score_against_rubric(response, rubric, context)
            scores[rubric.criterion] = score
            
        return scores
    
    def _score_against_rubric(self, 
                             response: str,
                             rubric: BehavioralRubric,
                             context: Dict[str, Any]) -> float:
        """Score response against a single rubric."""
        
        response_lower = response.lower()
        
        # Count good indicators
        good_matches = sum(1 for indicator in rubric.good_indicators 
                          if any(keyword.lower() in response_lower 
                                for keyword in indicator.split()[:3]))  # Check key words
        
        # Count bad indicators  
        bad_matches = sum(1 for indicator in rubric.bad_indicators
                         if any(keyword.lower() in response_lower
                               for keyword in indicator.split()[:3]))
        
        # Calculate score (0.0 to 1.0)
        total_indicators = len(rubric.good_indicators) + len(rubric.bad_indicators)
        good_ratio = good_matches / len(rubric.good_indicators) if rubric.good_indicators else 0
        bad_ratio = bad_matches / len(rubric.bad_indicators) if rubric.bad_indicators else 0
        
        # Score favors good indicators and penalizes bad ones
        score = max(0.0, min(1.0, good_ratio - bad_ratio))
        return score


class PreferenceDataGenerator:
    """Generates preference training data focused on behavioral improvements."""
    
    def __init__(self):
        self.rubrics = ComplianceBehavioralRubrics()
        
    def generate_preference_examples(self, count: int = 400) -> List[PreferenceExample]:
        """
        Generate preference training examples for DPO/ORPO.
        
        Args:
            count: Number of preference examples to generate
            
        Returns:
            List of preference examples
        """
        examples = []
        
        # Citation discipline examples
        examples.extend(self._generate_citation_examples(count // 5))
        
        # Evidence discipline examples 
        examples.extend(self._generate_evidence_examples(count // 5))
        
        # Remediation specificity examples
        examples.extend(self._generate_remediation_examples(count // 5))
        
        # Conservative posture examples
        examples.extend(self._generate_conservative_examples(count // 5))
        
        # Jurisdiction awareness examples
        examples.extend(self._generate_jurisdiction_examples(count // 5))
        
        return examples[:count]
    
    def _generate_citation_examples(self, count: int) -> List[PreferenceExample]:
        """Generate examples focused on citation discipline."""
        
        examples = []
        
        # Example 1: Specific vs vague citations
        prompt = """Analyze the compliance requirement for data processing consent in healthcare organizations."""
        
        chosen = """Based on HIPAA 45 CFR 164.508(a)(1) (effective April 14, 2003), healthcare organizations must obtain written authorization for uses and disclosures of protected health information beyond treatment, payment, and operations. The authorization must specify the purpose and recipient of the disclosure.

Additionally, under GDPR Article 7 (effective May 25, 2018) for EU operations, consent must be freely given, specific, informed, and unambiguous, with clear withdrawal mechanisms as specified in Article 7(3).

Citation sources:
- HIPAA 45 CFR 164.508(a)(1), Department of Health and Human Services
- GDPR Article 7, European Commission, Regulation 2016/679"""
        
        rejected = """Healthcare organizations need proper consent for data processing. The law clearly states that consent is required for processing health information. Organizations should implement consent mechanisms and ensure they comply with privacy regulations. This is a fundamental requirement across all healthcare compliance frameworks."""
        
        examples.append(PreferenceExample(
            prompt=prompt,
            chosen_response=chosen,
            rejected_response=rejected,
            preference_reason="Chosen response provides specific regulatory citations with sections, effective dates, and authorities. Rejected response makes vague claims without citations.",
            behavioral_criteria=["citation_discipline", "jurisdiction_awareness"]
        ))
        
        # Example 2: Evidence-based vs assumption-based
        prompt = """Assess the risk of storing customer data in cloud infrastructure without encryption."""
        
        chosen = """To provide an accurate risk assessment, I need additional evidence:

**Evidence Required:**
1. Specific cloud provider security certifications (SOC 2, ISO 27001)
2. Data classification levels of customer information
3. Applicable regulatory frameworks (GDPR, HIPAA, PCI DSS, etc.)
4. Current data handling procedures and access controls

**Interim Conservative Assessment:**
Based on general regulatory requirements, storing unencrypted customer data presents HIGH to CRITICAL risk:

- GDPR Article 32 requires "appropriate technical and organisational measures" including encryption
- PCI DSS Requirement 3.4 mandates encryption of cardholder data
- Most frameworks consider unencrypted personal data a significant security vulnerability

**Immediate Action:** Halt unencrypted data storage pending detailed risk assessment with the above evidence."""
        
        rejected = """Storing customer data without encryption is definitely a high risk. All regulations require encryption for customer data. This creates major compliance violations and could result in significant fines. Organizations should immediately implement encryption across all data storage systems. This is a clear-cut compliance requirement that all companies must follow."""
        
        examples.append(PreferenceExample(
            prompt=prompt,
            chosen_response=chosen,
            rejected_response=rejected,
            preference_reason="Chosen response requests specific evidence and provides conservative interim assessment. Rejected response makes definitive claims without evidence or citations.",
            behavioral_criteria=["evidence_discipline", "conservative_risk_posture", "citation_discipline"]
        ))
        
        # Add more citation examples...
        for i in range(count - 2):
            examples.append(self._generate_citation_variation(i))
            
        return examples
    
    def _generate_evidence_examples(self, count: int) -> List[PreferenceExample]:
        """Generate examples focused on evidence discipline."""
        
        examples = []
        
        prompt = """Determine if our organization's data retention practices comply with regulatory requirements."""
        
        chosen = """I cannot provide a definitive compliance determination without the following evidence:

**Required Documentation:**
1. Current data retention policy with specific retention periods by data type
2. Applicable regulatory frameworks (GDPR, HIPAA, SOX, industry-specific requirements)
3. Data inventory showing what personal/sensitive data is collected and processed
4. Current deletion/purging procedures and their implementation status
5. Audit logs demonstrating retention policy enforcement

**Preliminary Analysis Framework:**
Different regulations have varying retention requirements:
- GDPR Article 5(1)(e): Data should not be kept longer than necessary for purposes
- HIPAA 45 CFR 164.316(b)(2)(i): 6 years minimum for covered entities  
- SOX Section 802: 7 years for audit-related documents

**Next Steps:**
1. Collect the above evidence within 5 business days
2. Conduct gap analysis against applicable frameworks
3. Develop remediation plan for any identified gaps

Without this evidence, I cannot assess current compliance status."""
        
        rejected = """Based on standard best practices, most data retention policies are compliant if they follow industry norms. Organizations typically retain data for 7 years which meets most regulatory requirements. Your current practices are likely adequate since most companies follow similar approaches. I recommend continuing with current procedures and implementing standard retention schedules."""
        
        examples.append(PreferenceExample(
            prompt=prompt,
            chosen_response=chosen,
            rejected_response=rejected,
            preference_reason="Chosen response explicitly requests evidence and acknowledges limitations. Rejected response makes assumptions without evidence.",
            behavioral_criteria=["evidence_discipline", "citation_discipline"]
        ))
        
        # Generate more evidence examples
        for i in range(count - 1):
            examples.append(self._generate_evidence_variation(i))
            
        return examples
    
    def _generate_remediation_examples(self, count: int) -> List[PreferenceExample]:
        """Generate examples focused on specific, testable remediation."""
        
        examples = []
        
        prompt = """Recommend remediation for inadequate access control in our financial reporting system."""
        
        chosen = """**REMEDIATION PLAN: Financial Reporting Access Control**

**Immediate Actions (0-48 hours):**
1. **Audit Current Access** (Owner: IT Security Manager)
   - Generate complete user access report for financial reporting system
   - Identify users with inappropriate access levels
   - Document business justification for each access level
   - Success Criteria: 100% of current access documented and justified

2. **Implement Emergency Restrictions** (Owner: IT Operations)
   - Disable access for users without business justification
   - Require manager approval for any new access requests
   - Success Criteria: Only justified users retain access

**Short-term Actions (1-4 weeks):**
3. **Deploy Role-Based Access Control** (Owner: IT Architecture)
   - Define standard roles: Viewer, Preparer, Reviewer, Approver
   - Map current users to appropriate roles
   - Implement least-privilege access principles
   - Success Criteria: 100% of users assigned to defined roles

4. **Implement Segregation of Duties** (Owner: Financial Controller)
   - Ensure no single user can both prepare and approve entries
   - Implement maker-checker controls for journal entries >$10K
   - Success Criteria: Zero single-person control over material transactions

**Regulatory Basis:**
- SOX Section 404: Internal control over financial reporting
- SOX Section 302: Officer certification requirements

**Testing & Validation:**
- Weekly access reviews for first month
- Monthly SOD violation reports
- Quarterly independent access audit"""
        
        rejected = """You need to improve access controls for the financial system. Implement better security measures and ensure only authorized users have access. Consider role-based access and regular reviews. Make sure the system meets compliance requirements and follows best practices. This should help address the compliance gaps."""
        
        examples.append(PreferenceExample(
            prompt=prompt,
            chosen_response=chosen,
            rejected_response=rejected,
            preference_reason="Chosen response provides specific actions, owners, timelines, and success criteria. Rejected response is vague without actionable details.",
            behavioral_criteria=["remediation_specificity", "citation_discipline"]
        ))
        
        # Generate more remediation examples
        for i in range(count - 1):
            examples.append(self._generate_remediation_variation(i))
            
        return examples
    
    def _generate_conservative_examples(self, count: int) -> List[PreferenceExample]:
        """Generate examples focused on conservative risk posture."""
        
        examples = []
        
        prompt = """Assess the compliance risk of a new data sharing agreement with a third-party vendor."""
        
        chosen = """**CONSERVATIVE RISK ASSESSMENT: Third-Party Data Sharing**

**Risk Level: HIGH** (Applied conservatively due to limited vendor information)

**Analysis Limitations:**
- Vendor security certification status unknown
- Data processing locations not specified  
- Applicable regulatory frameworks not confirmed
- Data retention and deletion procedures unverified

**Conservative Assumptions Applied:**
Given insufficient vendor due diligence information, I am applying conservative risk assumptions:

1. **Regulatory Exposure:** Assume HIGHEST applicable standard
   - GDPR Article 28: Controller-processor agreements required
   - HIPAA 45 CFR 164.504(e): Business associate agreements mandatory
   - CCPA Section 1798.140(v): Service provider limitations

2. **Risk Factors Requiring Immediate Attention:**
   - Data transfer without adequate safeguards: CRITICAL
   - Lack of vendor security validation: HIGH  
   - Undefined data processing scope: HIGH
   - Absence of incident response procedures: MEDIUM

**IMMEDIATE ACTIONS REQUIRED:**
1. **HALT** data sharing until vendor due diligence complete
2. Engage legal counsel for agreement review
3. Conduct vendor security assessment
4. Obtain security certifications and compliance attestations

**Expert Consultation Recommended:**
This scenario requires legal review given multi-jurisdictional implications and potential regulatory violations."""
        
        rejected = """The data sharing agreement looks fine. Most vendors have adequate security measures and follow standard practices. The risk is probably low to medium since these are common business arrangements. You can proceed with the agreement as long as you include standard privacy clauses. This type of data sharing is routine in most organizations."""
        
        examples.append(PreferenceExample(
            prompt=prompt,
            chosen_response=chosen,
            rejected_response=rejected,
            preference_reason="Chosen response applies conservative assumptions due to uncertainty and recommends expert consultation. Rejected response minimizes risk without adequate evidence.",
            behavioral_criteria=["conservative_risk_posture", "evidence_discipline"]
        ))
        
        # Generate more conservative examples
        for i in range(count - 1):
            examples.append(self._generate_conservative_variation(i))
            
        return examples
    
    def _generate_jurisdiction_examples(self, count: int) -> List[PreferenceExample]:
        """Generate examples focused on jurisdiction awareness."""
        
        examples = []
        
        prompt = """Analyze privacy compliance requirements for a global e-commerce platform."""
        
        chosen = """**MULTI-JURISDICTIONAL PRIVACY COMPLIANCE ANALYSIS**

**Applicable Jurisdictions & Frameworks:**

**1. European Union - GDPR (Regulation 2016/679)**
- Effective: May 25, 2018
- Scope: EU residents' personal data processing
- Key Requirements: Article 6 lawful basis, Article 13/14 privacy notices, Article 7 consent management

**2. United States - State Privacy Laws**
- California CCPA (effective January 1, 2020) + CPRA amendments (January 1, 2023)
- Virginia CDPA (effective January 1, 2023)  
- Connecticut CTDPA (effective July 1, 2023)
- Scope: State residents with revenue/processing thresholds

**3. Canada - PIPEDA**
- Personal Information Protection and Electronic Documents Act
- Scope: Commercial activities across provinces

**Jurisdictional Conflicts & Considerations:**
- **Data Localization:** Some jurisdictions restrict cross-border transfers
- **Consent Standards:** GDPR requires explicit consent; CCPA allows opt-out model
- **Individual Rights:** Varying rights to deletion, portability, correction

**Framework-Specific Requirements:**
[Detailed breakdown by jurisdiction with specific article/section references]

**Recommendation:** Engage legal counsel with multi-jurisdictional privacy expertise for implementation guidance."""
        
        rejected = """For global e-commerce, you need to comply with privacy laws. Most countries have similar requirements around consent and data protection. Implement a comprehensive privacy program that covers the main requirements like consent management, privacy notices, and individual rights. This approach will generally work across different jurisdictions since privacy principles are fairly universal."""
        
        examples.append(PreferenceExample(
            prompt=prompt,
            chosen_response=chosen,
            rejected_response=rejected,
            preference_reason="Chosen response explicitly identifies jurisdictions, frameworks, and effective dates. Rejected response assumes universal applicability without jurisdictional specificity.",
            behavioral_criteria=["jurisdiction_awareness", "citation_discipline"]
        ))
        
        # Generate more jurisdiction examples
        for i in range(count - 1):
            examples.append(self._generate_jurisdiction_variation(i))
            
        return examples
    
    def _generate_citation_variation(self, index: int) -> PreferenceExample:
        """Generate citation-focused variations."""
        # Simplified variation generator
        return PreferenceExample(
            prompt=f"Analyze requirement #{index + 3} for data breach notification requirements.",
            chosen_response=f"Under GDPR Article 33 (effective May 25, 2018), controllers must notify supervisory authorities within 72 hours of becoming aware of a personal data breach...",
            rejected_response=f"Data breach notification is required by law. Organizations must report breaches quickly to authorities...",
            preference_reason="Chosen provides specific citations; rejected is vague",
            behavioral_criteria=["citation_discipline"]
        )
    
    def _generate_evidence_variation(self, index: int) -> PreferenceExample:
        """Generate evidence-focused variations."""
        return PreferenceExample(
            prompt=f"Assess compliance status of incident response procedure #{index + 2}.",
            chosen_response=f"To assess compliance, I need: 1) Current incident response plan, 2) Training records, 3) Applicable regulatory requirements...",
            rejected_response=f"The incident response procedure appears adequate based on industry standards...",
            preference_reason="Chosen requests specific evidence; rejected makes assumptions",
            behavioral_criteria=["evidence_discipline"]
        )
        
    def _generate_remediation_variation(self, index: int) -> PreferenceExample:
        """Generate remediation-focused variations."""
        return PreferenceExample(
            prompt=f"Recommend improvements for security control #{index + 2}.",
            chosen_response=f"Action 1: Deploy MFA (Owner: IT Security, Due: 2024-02-15, Success: 100% user enrollment)...",
            rejected_response=f"Implement better security controls and improve access management...",
            preference_reason="Chosen provides specific actions with owners and timelines; rejected is vague",
            behavioral_criteria=["remediation_specificity"]
        )
    
    def _generate_conservative_variation(self, index: int) -> PreferenceExample:
        """Generate conservative posture variations."""
        return PreferenceExample(
            prompt=f"Evaluate risk scenario #{index + 2} with limited information.",
            chosen_response=f"Given limited information, applying conservative HIGH risk rating. Requires expert consultation...",
            rejected_response=f"The risk appears manageable based on typical scenarios...",
            preference_reason="Chosen applies conservative approach; rejected minimizes uncertainty",
            behavioral_criteria=["conservative_risk_posture"]
        )
    
    def _generate_jurisdiction_variation(self, index: int) -> PreferenceExample:
        """Generate jurisdiction awareness variations."""
        return PreferenceExample(
            prompt=f"Analyze regulatory requirement #{index + 2} for international operations.",
            chosen_response=f"Jurisdictional Analysis: 1) EU (GDPR Art. X), 2) US (State laws), 3) Canada (PIPEDA)...",
            rejected_response=f"International privacy requirements are similar across jurisdictions...",
            preference_reason="Chosen specifies jurisdictions and frameworks; rejected assumes universality",
            behavioral_criteria=["jurisdiction_awareness"]
        )


def export_preference_data_for_training(examples: List[PreferenceExample], 
                                       filename: str = "compliance_preference_data.jsonl") -> None:
    """Export preference examples in format suitable for DPO/ORPO training."""
    
    with open(filename, 'w') as f:
        for example in examples:
            training_record = {
                "prompt": example.prompt,
                "chosen": example.chosen_response,
                "rejected": example.rejected_response,
                "metadata": {
                    "preference_reason": example.preference_reason,
                    "behavioral_criteria": example.behavioral_criteria,
                    "training_type": "behavioral_preference",
                    "focus": "senior_analyst_behavior"
                }
            }
            f.write(json.dumps(training_record) + '\n')
    
    logger.info(f"Exported {len(examples)} preference examples to {filename}")


if __name__ == "__main__":
    # Generate preference training data
    generator = PreferenceDataGenerator()
    examples = generator.generate_preference_examples(400)
    
    # Export for training
    export_preference_data_for_training(examples)
    
    print(f"Generated {len(examples)} preference training examples")
    print("Focus areas:")
    
    criteria_counts = {}
    for example in examples:
        for criterion in example.behavioral_criteria:
            criteria_counts[criterion] = criteria_counts.get(criterion, 0) + 1
    
    for criterion, count in criteria_counts.items():
        print(f"  {criterion}: {count} examples")
    
    print("\nSample example:")
    sample = examples[0]
    print(f"Prompt: {sample.prompt[:100]}...")
    print(f"Preference reason: {sample.preference_reason}")
    print(f"Behavioral criteria: {sample.behavioral_criteria}")
