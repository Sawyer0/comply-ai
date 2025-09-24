"""
Compliance Intelligence Engine for automated regulatory mapping and gap analysis.

This engine provides sophisticated rule-based mapping of security findings to
compliance frameworks and generates actionable remediation plans.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from ..domain import (
    AnalysisConfiguration,
    AnalysisResult,
    BaseAnalysisEngine,
    ComplianceFramework,
    ComplianceGap,
    ComplianceMapping,
    IComplianceIntelligenceEngine,
    RemediationAction,
    RemediationPlan,
    RiskLevel,
    SecurityFinding,
)
from ..domain.entities import AnalysisRequest

logger = logging.getLogger(__name__)


class ComplianceIntelligenceEngine(BaseAnalysisEngine, IComplianceIntelligenceEngine):
    """
    Advanced compliance intelligence engine with regulatory expertise.
    
    Provides automated mapping to SOC 2, ISO 27001, HIPAA, GDPR, and PCI DSS
    frameworks using sophisticated rule-based logic and gap analysis.
    """
    
    def __init__(self, config: AnalysisConfiguration):
        super().__init__(config)
        self.compliance_config = config.parameters.get('compliance_intelligence', {})
        self.enabled_frameworks = self.compliance_config.get('enabled_frameworks', [
            'soc2', 'iso27001', 'hipaa', 'gdpr', 'pci_dss'
        ])
        self.control_mappings = self._initialize_control_mappings()
        self.gap_thresholds = self.compliance_config.get('gap_thresholds', {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        })
    
    def get_engine_name(self) -> str:
        """Get the name of this analysis engine."""
        return "compliance_intelligence"
    
    async def _perform_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform compliance intelligence analysis on the request data.
        
        Args:
            request: Analysis request containing security findings
            
        Returns:
            AnalysisResult with compliance mappings and gap analysis
        """
        # Extract security findings from request
        findings = self._extract_security_findings(request)
        
        # Map findings to compliance frameworks
        compliance_mappings = await self.map_to_frameworks(findings)
        
        # Analyze current compliance state
        compliance_state = self._analyze_compliance_state(request, compliance_mappings)
        
        # Identify compliance gaps
        compliance_gaps = await self.identify_compliance_gaps(compliance_state)
        
        # Generate remediation plan
        remediation_plan = await self.generate_remediation_plan(compliance_gaps)
        
        # Calculate confidence based on mapping quality
        confidence = self._calculate_compliance_confidence(compliance_mappings, compliance_gaps)
        
        # Create analysis result
        result = AnalysisResult(
            analysis_type="compliance_intelligence",
            confidence=confidence,
            compliance_mappings=compliance_mappings,
            recommendations=[action for action in remediation_plan.actions],
            evidence=[
                {
                    "type": "compliance_analysis",
                    "frameworks_analyzed": len(self.enabled_frameworks),
                    "mappings_created": len(compliance_mappings),
                    "gaps_identified": len(compliance_gaps),
                    "remediation_actions": len(remediation_plan.actions)
                }
            ],
            metadata={
                "compliance_state": compliance_state,
                "compliance_gaps": [gap.dict() for gap in compliance_gaps],
                "remediation_plan": remediation_plan.dict(),
                "frameworks_covered": self.enabled_frameworks
            }
        )
        
        return result
    
    async def map_to_frameworks(self, findings: List[SecurityFinding]) -> List[ComplianceMapping]:
        """
        Map security findings to compliance framework requirements.
        
        Args:
            findings: Security findings to map
            
        Returns:
            List of compliance mappings for different frameworks
        """
        mappings = []
        
        for framework_name in self.enabled_frameworks:
            try:
                framework = ComplianceFramework(framework_name)
                framework_mappings = await self._map_to_specific_framework(findings, framework)
                mappings.extend(framework_mappings)
            except ValueError:
                self.logger.warning(f"Unknown compliance framework: {framework_name}")
                continue
        
        return mappings
    
    async def identify_compliance_gaps(self, current_state: Dict[str, Any]) -> List[ComplianceGap]:
        """
        Identify gaps in compliance coverage.
        
        Args:
            current_state: Current compliance state to analyze
            
        Returns:
            List of identified compliance gaps with priorities
        """
        gaps = []
        
        for framework_name in self.enabled_frameworks:
            try:
                framework = ComplianceFramework(framework_name)
                framework_gaps = await self._identify_framework_gaps(current_state, framework)
                gaps.extend(framework_gaps)
            except ValueError:
                continue
        
        # Sort gaps by severity and priority
        gaps.sort(key=lambda g: (
            ['low', 'medium', 'high', 'critical'].index(g.severity.value),
            g.business_impact.total_risk_value
        ), reverse=True)
        
        return gaps
    
    async def generate_remediation_plan(self, gaps: List[ComplianceGap]) -> RemediationPlan:
        """
        Generate actionable remediation plan for compliance gaps.
        
        Args:
            gaps: Compliance gaps to address
            
        Returns:
            RemediationPlan with prioritized actions and timelines
        """
        if not gaps:
            return self._create_empty_remediation_plan()
        
        actions = []
        total_effort = {'hours': 0, 'cost': 0.0, 'resources': set()}
        
        # Group gaps by framework and control type
        gap_groups = self._group_gaps_by_priority(gaps)
        
        priority = 1
        for group_name, group_gaps in gap_groups.items():
            group_actions = await self._create_actions_for_gap_group(group_gaps, priority)
            actions.extend(group_actions)
            priority += len(group_actions)
        
        # Calculate total effort
        for action in actions:
            effort = action.estimated_effort
            total_effort['hours'] += effort.get('hours', 0)
            total_effort['cost'] += effort.get('cost', 0.0)
            if 'resources' in effort:
                total_effort['resources'].update(effort['resources'])
        
        total_effort['resources'] = list(total_effort['resources'])
        
        # Create timeline
        timeline = self._create_remediation_timeline(actions)
        
        # Define success metrics
        success_metrics = self._define_success_metrics(gaps)
        
        # Create risk mitigation strategies
        risk_mitigation = self._create_risk_mitigation_strategies(gaps)
        
        return RemediationPlan(
            title=f"Compliance Remediation Plan - {len(gaps)} gaps identified",
            description=f"Comprehensive plan to address compliance gaps across {len(self.enabled_frameworks)} frameworks",
            actions=actions,
            total_effort=total_effort,
            timeline=timeline,
            success_metrics=success_metrics,
            risk_mitigation=risk_mitigation
        )
    
    async def _map_to_specific_framework(self, findings: List[SecurityFinding], 
                                       framework: ComplianceFramework) -> List[ComplianceMapping]:
        """Map findings to a specific compliance framework."""
        mappings = []
        framework_controls = self.control_mappings.get(framework.value, {})
        
        for finding in findings:
            # Determine applicable controls for this finding
            applicable_controls = self._find_applicable_controls(finding, framework_controls)
            
            for control_id, control_info in applicable_controls.items():
                # Assess compliance status
                compliance_status = self._assess_compliance_status(finding, control_info)
                
                # Determine gap severity
                gap_severity = self._determine_gap_severity(finding, compliance_status)
                
                # Calculate remediation priority
                priority = self._calculate_remediation_priority(finding, gap_severity, framework)
                
                mapping = ComplianceMapping(
                    framework=framework,
                    control_id=control_id,
                    control_description=control_info['description'],
                    finding_ids=[finding.finding_id],
                    compliance_status=compliance_status,
                    gap_severity=gap_severity,
                    remediation_priority=priority
                )
                mappings.append(mapping)
        
        return mappings
    
    def _initialize_control_mappings(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Initialize comprehensive control mappings for all frameworks."""
        return {
            'soc2': {
                'CC6.1': {
                    'description': 'Logical and Physical Access Controls',
                    'categories': ['access_control', 'authentication', 'authorization'],
                    'detectors': ['presidio', 'access-control', 'auth-detector'],
                    'requirements': ['access_logging', 'user_management', 'privilege_control']
                },
                'CC6.7': {
                    'description': 'Data Transmission and Disposal',
                    'categories': ['data_transmission', 'encryption', 'data_disposal'],
                    'detectors': ['encryption-detector', 'data-transmission', 'pii-detector'],
                    'requirements': ['encryption_in_transit', 'secure_disposal', 'data_classification']
                },
                'CC7.1': {
                    'description': 'System Boundaries and Data Flow',
                    'categories': ['network_security', 'boundary_controls', 'data_flow'],
                    'detectors': ['network-scanner', 'boundary-detector', 'firewall-analyzer'],
                    'requirements': ['network_segmentation', 'boundary_monitoring', 'data_flow_controls']
                }
            },
            'iso27001': {
                'A.8.2.1': {
                    'description': 'Classification of Information',
                    'categories': ['data_classification', 'information_handling'],
                    'detectors': ['pii-detector', 'data-classifier', 'sensitivity-analyzer'],
                    'requirements': ['classification_scheme', 'labeling_procedures', 'handling_guidelines']
                },
                'A.8.2.2': {
                    'description': 'Labelling of Information',
                    'categories': ['data_labeling', 'metadata_management'],
                    'detectors': ['metadata-detector', 'labeling-validator'],
                    'requirements': ['labeling_standards', 'metadata_consistency', 'label_validation']
                },
                'A.13.2.1': {
                    'description': 'Information Transfer Policies and Procedures',
                    'categories': ['data_transfer', 'information_sharing'],
                    'detectors': ['transfer-monitor', 'sharing-detector', 'external-access'],
                    'requirements': ['transfer_policies', 'secure_channels', 'transfer_logging']
                }
            },
            'hipaa': {
                'Administrative Safeguards': {
                    'description': 'Administrative Safeguards for PHI',
                    'categories': ['administrative_controls', 'workforce_training', 'access_management'],
                    'detectors': ['hipaa-validator', 'phi-detector', 'access-control'],
                    'requirements': ['security_officer', 'workforce_training', 'access_procedures']
                },
                'Physical Safeguards': {
                    'description': 'Physical Safeguards for PHI',
                    'categories': ['physical_security', 'workstation_controls', 'media_controls'],
                    'detectors': ['physical-security', 'workstation-monitor', 'media-tracker'],
                    'requirements': ['facility_controls', 'workstation_security', 'media_disposal']
                },
                'Technical Safeguards': {
                    'description': 'Technical Safeguards for PHI',
                    'categories': ['technical_controls', 'encryption', 'audit_logging'],
                    'detectors': ['phi-detector', 'encryption-validator', 'audit-logger'],
                    'requirements': ['access_control', 'audit_controls', 'integrity_controls']
                }
            },
            'gdpr': {
                'Article 25': {
                    'description': 'Data Protection by Design and by Default',
                    'categories': ['privacy_by_design', 'data_minimization', 'purpose_limitation'],
                    'detectors': ['gdpr-scanner', 'privacy-analyzer', 'purpose-validator'],
                    'requirements': ['privacy_controls', 'data_minimization', 'purpose_binding']
                },
                'Article 32': {
                    'description': 'Security of Processing',
                    'categories': ['data_security', 'encryption', 'access_control'],
                    'detectors': ['security-scanner', 'encryption-detector', 'access-monitor'],
                    'requirements': ['security_measures', 'encryption_requirements', 'access_controls']
                },
                'Article 35': {
                    'description': 'Data Protection Impact Assessment',
                    'categories': ['impact_assessment', 'risk_evaluation', 'privacy_impact'],
                    'detectors': ['impact-assessor', 'risk-evaluator', 'privacy-impact'],
                    'requirements': ['impact_assessment', 'risk_mitigation', 'privacy_evaluation']
                }
            },
            'pci_dss': {
                'Requirement 3': {
                    'description': 'Protect Stored Cardholder Data',
                    'categories': ['data_protection', 'encryption', 'cardholder_data'],
                    'detectors': ['pci-scanner', 'card-detector', 'encryption-validator'],
                    'requirements': ['data_encryption', 'key_management', 'secure_storage']
                },
                'Requirement 4': {
                    'description': 'Encrypt Transmission of Cardholder Data',
                    'categories': ['transmission_security', 'network_encryption'],
                    'detectors': ['transmission-monitor', 'network-encryption', 'ssl-validator'],
                    'requirements': ['transmission_encryption', 'secure_protocols', 'key_exchange']
                },
                'Requirement 7': {
                    'description': 'Restrict Access to Cardholder Data by Business Need to Know',
                    'categories': ['access_control', 'need_to_know', 'role_based_access'],
                    'detectors': ['access-control', 'role-validator', 'privilege-monitor'],
                    'requirements': ['access_restrictions', 'role_definitions', 'privilege_management']
                }
            }
        }
    
    def _find_applicable_controls(self, finding: SecurityFinding, 
                                framework_controls: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Find applicable controls for a security finding."""
        applicable = {}
        
        for control_id, control_info in framework_controls.items():
            # Check if finding category matches control categories
            if finding.category.lower() in [cat.lower() for cat in control_info.get('categories', [])]:
                applicable[control_id] = control_info
                continue
            
            # Check if detector matches control detectors
            if finding.detector_id in control_info.get('detectors', []):
                applicable[control_id] = control_info
                continue
            
            # Check for keyword matches in detector name
            detector_keywords = finding.detector_id.lower().split('-')
            control_keywords = []
            for cat in control_info.get('categories', []):
                control_keywords.extend(cat.lower().split('_'))
            
            if any(keyword in control_keywords for keyword in detector_keywords):
                applicable[control_id] = control_info
        
        return applicable
    
    def _assess_compliance_status(self, finding: SecurityFinding, control_info: Dict[str, Any]) -> str:
        """Assess compliance status for a finding against a control."""
        # Determine compliance status based on finding severity and control requirements
        severity_impact = {
            'low': 'compliant_with_exceptions',
            'medium': 'partially_compliant',
            'high': 'non_compliant',
            'critical': 'major_non_compliance'
        }
        
        base_status = severity_impact.get(finding.severity.value.lower(), 'partially_compliant')
        
        # Adjust based on confidence
        if finding.confidence < 0.7:
            # Lower confidence findings are less likely to indicate non-compliance
            if base_status == 'major_non_compliance':
                base_status = 'non_compliant'
            elif base_status == 'non_compliant':
                base_status = 'partially_compliant'
        
        return base_status
    
    def _determine_gap_severity(self, finding: SecurityFinding, compliance_status: str) -> RiskLevel:
        """Determine gap severity based on finding and compliance status."""
        status_severity = {
            'compliant': RiskLevel.LOW,
            'compliant_with_exceptions': RiskLevel.LOW,
            'partially_compliant': RiskLevel.MEDIUM,
            'non_compliant': RiskLevel.HIGH,
            'major_non_compliance': RiskLevel.CRITICAL
        }
        
        base_severity = status_severity.get(compliance_status, RiskLevel.MEDIUM)
        
        # Escalate severity if finding is high confidence and high severity
        if finding.confidence > 0.8 and finding.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            if base_severity == RiskLevel.MEDIUM:
                base_severity = RiskLevel.HIGH
            elif base_severity == RiskLevel.HIGH:
                base_severity = RiskLevel.CRITICAL
        
        return base_severity
    
    def _calculate_remediation_priority(self, finding: SecurityFinding, gap_severity: RiskLevel, 
                                      framework: ComplianceFramework) -> int:
        """Calculate remediation priority (1-5, where 1 is highest priority)."""
        # Base priority from gap severity
        severity_priority = {
            RiskLevel.CRITICAL: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.MEDIUM: 3,
            RiskLevel.LOW: 4
        }
        
        base_priority = severity_priority.get(gap_severity, 3)
        
        # Adjust based on framework criticality
        framework_weights = {
            ComplianceFramework.HIPAA: 0,  # Highest priority
            ComplianceFramework.GDPR: 0,
            ComplianceFramework.PCI_DSS: 1,
            ComplianceFramework.SOC2: 1,
            ComplianceFramework.ISO27001: 2
        }
        
        framework_adjustment = framework_weights.get(framework, 1)
        
        # Adjust based on finding confidence
        confidence_adjustment = 0 if finding.confidence > 0.8 else 1
        
        final_priority = base_priority + framework_adjustment + confidence_adjustment
        return min(5, max(1, final_priority))
    
    def _analyze_compliance_state(self, request: AnalysisRequest, 
                                mappings: List[ComplianceMapping]) -> Dict[str, Any]:
        """Analyze current compliance state."""
        state = {
            'overall_status': 'unknown',
            'framework_status': {},
            'coverage_metrics': {},
            'risk_indicators': {}
        }
        
        # Analyze by framework
        for framework_name in self.enabled_frameworks:
            framework_mappings = [m for m in mappings if m.framework.value == framework_name]
            
            if framework_mappings:
                # Calculate compliance percentage
                compliant_count = sum(1 for m in framework_mappings 
                                    if m.compliance_status in ['compliant', 'compliant_with_exceptions'])
                compliance_percentage = compliant_count / len(framework_mappings)
                
                # Determine framework status
                if compliance_percentage >= 0.9:
                    framework_status = 'compliant'
                elif compliance_percentage >= 0.7:
                    framework_status = 'mostly_compliant'
                elif compliance_percentage >= 0.5:
                    framework_status = 'partially_compliant'
                else:
                    framework_status = 'non_compliant'
                
                state['framework_status'][framework_name] = {
                    'status': framework_status,
                    'compliance_percentage': compliance_percentage,
                    'total_controls': len(framework_mappings),
                    'compliant_controls': compliant_count
                }
        
        # Calculate coverage metrics
        state['coverage_metrics'] = self._calculate_coverage_metrics(request)
        
        # Identify risk indicators
        state['risk_indicators'] = self._identify_risk_indicators(mappings)
        
        # Determine overall status
        framework_statuses = [fs['status'] for fs in state['framework_status'].values()]
        if all(status == 'compliant' for status in framework_statuses):
            state['overall_status'] = 'compliant'
        elif any(status == 'non_compliant' for status in framework_statuses):
            state['overall_status'] = 'non_compliant'
        else:
            state['overall_status'] = 'partially_compliant'
        
        return state
    
    async def _identify_framework_gaps(self, current_state: Dict[str, Any], 
                                     framework: ComplianceFramework) -> List[ComplianceGap]:
        """Identify gaps for a specific framework."""
        gaps = []
        framework_status = current_state['framework_status'].get(framework.value, {})
        
        if framework_status.get('status') in ['non_compliant', 'partially_compliant']:
            # Create gap for overall framework compliance
            gap_severity = RiskLevel.HIGH if framework_status.get('status') == 'non_compliant' else RiskLevel.MEDIUM
            
            # Calculate business impact
            business_impact = await self._calculate_gap_business_impact(framework, gap_severity)
            
            # Estimate remediation effort
            remediation_effort = self._estimate_gap_remediation_effort(framework_status)
            
            # Determine deadline based on framework
            deadline = self._determine_compliance_deadline(framework)
            
            gap = ComplianceGap(
                framework=framework,
                control_id="overall_compliance",
                gap_description=f"Overall compliance gap in {framework.value.upper()} framework",
                severity=gap_severity,
                business_impact=business_impact,
                remediation_effort=remediation_effort,
                deadline=deadline
            )
            gaps.append(gap)
        
        return gaps
    
    def _extract_security_findings(self, request: AnalysisRequest) -> List[SecurityFinding]:
        """Extract security findings from analysis request."""
        findings = []
        
        # Extract from high severity hits
        for hit in request.high_sev_hits:
            severity = self._map_severity(hit.get('severity', 'medium'))
            
            finding = SecurityFinding(
                detector_id=hit.get('detector', 'unknown'),
                severity=severity,
                category=hit.get('category', 'security'),
                description=hit.get('description', 'Security finding detected'),
                timestamp=datetime.now(timezone.utc),
                metadata=hit,
                confidence=hit.get('confidence', 0.8)
            )
            findings.append(finding)
        
        # Extract from detector errors (treat as findings)
        for detector, error_info in request.detector_errors.items():
            if error_info.get('severity', 'low') in ['high', 'critical']:
                finding = SecurityFinding(
                    detector_id=detector,
                    severity=RiskLevel.HIGH,
                    category='detector_error',
                    description=f"Detector error: {error_info.get('message', 'Unknown error')}",
                    timestamp=datetime.now(timezone.utc),
                    metadata=error_info,
                    confidence=0.9
                )
                findings.append(finding)
        
        return findings
    
    def _map_severity(self, severity_str: str) -> RiskLevel:
        """Map string severity to RiskLevel enum."""
        severity_mapping = {
            'low': RiskLevel.LOW,
            'medium': RiskLevel.MEDIUM,
            'high': RiskLevel.HIGH,
            'critical': RiskLevel.CRITICAL
        }
        return severity_mapping.get(severity_str.lower(), RiskLevel.MEDIUM)
    
    def _calculate_compliance_confidence(self, mappings: List[ComplianceMapping], 
                                       gaps: List[ComplianceGap]) -> float:
        """Calculate confidence in compliance analysis."""
        if not mappings:
            return 0.5  # Medium confidence when no mappings
        
        # Base confidence from number of mappings
        mapping_confidence = min(1.0, len(mappings) / 10)  # Assume 10 mappings for full confidence
        
        # Adjust based on gap analysis quality
        gap_confidence = 1.0 if gaps else 0.8  # Higher confidence when gaps are identified
        
        return (mapping_confidence + gap_confidence) / 2
    
    def _create_empty_remediation_plan(self) -> RemediationPlan:
        """Create empty remediation plan when no gaps are found."""
        return RemediationPlan(
            title="No Compliance Gaps Identified",
            description="Current compliance state appears satisfactory",
            actions=[],
            total_effort={'hours': 0, 'cost': 0.0, 'resources': []},
            timeline={'status': 'no_action_required'},
            success_metrics=['maintain_current_compliance_level'],
            risk_mitigation={'strategy': 'continuous_monitoring'}
        )
    
    def _group_gaps_by_priority(self, gaps: List[ComplianceGap]) -> Dict[str, List[ComplianceGap]]:
        """Group gaps by priority for remediation planning."""
        groups = {
            'critical_immediate': [],
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        for gap in gaps:
            if gap.severity == RiskLevel.CRITICAL:
                groups['critical_immediate'].append(gap)
            elif gap.severity == RiskLevel.HIGH:
                groups['high_priority'].append(gap)
            elif gap.severity == RiskLevel.MEDIUM:
                groups['medium_priority'].append(gap)
            else:
                groups['low_priority'].append(gap)
        
        return {k: v for k, v in groups.items() if v}  # Remove empty groups
    
    async def _create_actions_for_gap_group(self, gaps: List[ComplianceGap], 
                                          start_priority: int) -> List[RemediationAction]:
        """Create remediation actions for a group of gaps."""
        actions = []
        
        for i, gap in enumerate(gaps):
            action = RemediationAction(
                title=f"Address {gap.framework.value.upper()} compliance gap",
                description=f"Remediate {gap.gap_description}",
                priority=start_priority + i,
                estimated_effort=gap.remediation_effort,
                timeline=self._create_action_timeline(gap),
                success_criteria=self._create_success_criteria(gap),
                dependencies=self._identify_action_dependencies(gap, gaps)
            )
            actions.append(action)
        
        return actions
    
    def _calculate_coverage_metrics(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Calculate coverage metrics from request data."""
        total_detectors = len(request.required_detectors)
        covered_detectors = len([d for d, coverage in request.observed_coverage.items() 
                               if coverage >= request.required_coverage.get(d, 0.0)])
        
        coverage_percentage = covered_detectors / total_detectors if total_detectors > 0 else 0.0
        
        return {
            'total_detectors': total_detectors,
            'covered_detectors': covered_detectors,
            'coverage_percentage': coverage_percentage,
            'gaps_count': total_detectors - covered_detectors
        }
    
    def _identify_risk_indicators(self, mappings: List[ComplianceMapping]) -> Dict[str, Any]:
        """Identify risk indicators from compliance mappings."""
        high_risk_mappings = [m for m in mappings if m.gap_severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        
        return {
            'high_risk_controls': len(high_risk_mappings),
            'total_controls': len(mappings),
            'risk_percentage': len(high_risk_mappings) / len(mappings) if mappings else 0.0,
            'frameworks_at_risk': len(set(m.framework.value for m in high_risk_mappings))
        }
    
    async def _calculate_gap_business_impact(self, framework: ComplianceFramework, 
                                           severity: RiskLevel) -> Any:
        """Calculate business impact of compliance gap."""
        from ..domain.analysis_models import BusinessImpact
        
        # Framework-specific impact multipliers
        framework_multipliers = {
            ComplianceFramework.HIPAA: 5.0,
            ComplianceFramework.GDPR: 4.5,
            ComplianceFramework.PCI_DSS: 4.0,
            ComplianceFramework.SOC2: 3.0,
            ComplianceFramework.ISO27001: 2.5
        }
        
        base_impact = framework_multipliers.get(framework, 3.0)
        
        # Severity multipliers
        severity_multipliers = {
            RiskLevel.CRITICAL: 10000,
            RiskLevel.HIGH: 5000,
            RiskLevel.MEDIUM: 2000,
            RiskLevel.LOW: 500
        }
        
        severity_impact = severity_multipliers.get(severity, 2000)
        total_impact = base_impact * severity_impact
        
        return BusinessImpact(
            financial_impact={
                'potential_fines': total_impact * 0.6,
                'audit_costs': total_impact * 0.2,
                'remediation_costs': total_impact * 0.2,
                'total_estimated_cost': total_impact
            },
            operational_impact={
                'audit_disruption': 'high' if severity in [RiskLevel.HIGH, RiskLevel.CRITICAL] else 'medium',
                'compliance_workload': 'increased'
            },
            reputational_impact={
                'regulatory_scrutiny': 'increased',
                'customer_confidence': 'at_risk' if severity == RiskLevel.CRITICAL else 'stable'
            },
            compliance_impact={
                'certification_risk': 'high' if severity == RiskLevel.CRITICAL else 'medium',
                'audit_findings': 'likely'
            },
            total_risk_value=total_impact,
            confidence_interval={'lower': total_impact * 0.7, 'upper': total_impact * 1.3},
            impact_timeline={
                'immediate': 'compliance_review_required',
                '30_days': 'audit_preparation',
                '90_days': 'certification_impact'
            }
        )
    
    def _estimate_gap_remediation_effort(self, framework_status: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate effort required to remediate compliance gap."""
        compliance_percentage = framework_status.get('compliance_percentage', 0.0)
        total_controls = framework_status.get('total_controls', 1)
        
        # Estimate hours based on gap size
        gap_size = 1.0 - compliance_percentage
        base_hours = gap_size * total_controls * 8  # 8 hours per control gap
        
        # Estimate cost (assuming $150/hour for compliance work)
        estimated_cost = base_hours * 150
        
        # Required resources
        resources = ['compliance_specialist', 'security_engineer']
        if gap_size > 0.5:
            resources.append('external_consultant')
        
        return {
            'hours': int(base_hours),
            'cost': estimated_cost,
            'resources': resources,
            'complexity': 'high' if gap_size > 0.5 else 'medium' if gap_size > 0.2 else 'low'
        }
    
    def _determine_compliance_deadline(self, framework: ComplianceFramework) -> Optional[datetime]:
        """Determine compliance deadline based on framework."""
        # Framework-specific deadline urgency
        deadline_days = {
            ComplianceFramework.HIPAA: 30,    # HIPAA violations need immediate attention
            ComplianceFramework.GDPR: 45,     # GDPR has strict enforcement
            ComplianceFramework.PCI_DSS: 60,  # PCI DSS has regular assessments
            ComplianceFramework.SOC2: 90,     # SOC 2 annual cycles
            ComplianceFramework.ISO27001: 120 # ISO 27001 longer cycles
        }
        
        days = deadline_days.get(framework, 90)
        return datetime.now(timezone.utc) + timedelta(days=days)
    
    def _create_remediation_timeline(self, actions: List[RemediationAction]) -> Dict[str, str]:
        """Create overall remediation timeline."""
        if not actions:
            return {'status': 'no_timeline_required'}
        
        # Group actions by priority
        critical_actions = [a for a in actions if a.priority <= 2]
        high_actions = [a for a in actions if 3 <= a.priority <= 4]
        medium_actions = [a for a in actions if a.priority >= 5]
        
        timeline = {}
        
        if critical_actions:
            timeline['immediate'] = f"Begin {len(critical_actions)} critical actions"
            timeline['week_1'] = "Complete critical compliance gaps"
        
        if high_actions:
            timeline['week_2-4'] = f"Address {len(high_actions)} high priority gaps"
        
        if medium_actions:
            timeline['month_2-3'] = f"Complete {len(medium_actions)} remaining actions"
        
        timeline['ongoing'] = "Continuous monitoring and maintenance"
        
        return timeline
    
    def _define_success_metrics(self, gaps: List[ComplianceGap]) -> List[str]:
        """Define success metrics for remediation plan."""
        metrics = [
            "All identified compliance gaps remediated",
            "Compliance audit findings reduced to zero",
            "Framework compliance percentage above 95%"
        ]
        
        # Add framework-specific metrics
        frameworks = set(gap.framework.value for gap in gaps)
        for framework in frameworks:
            metrics.append(f"{framework.upper()} compliance certification maintained")
        
        return metrics
    
    def _create_risk_mitigation_strategies(self, gaps: List[ComplianceGap]) -> Dict[str, Any]:
        """Create risk mitigation strategies."""
        critical_gaps = [g for g in gaps if g.severity == RiskLevel.CRITICAL]
        
        strategies = {
            'immediate_actions': [],
            'monitoring_plan': 'Continuous compliance monitoring',
            'escalation_procedures': 'Defined escalation for new gaps'
        }
        
        if critical_gaps:
            strategies['immediate_actions'] = [
                'Implement temporary controls',
                'Increase monitoring frequency',
                'Prepare incident response plan'
            ]
        
        return strategies
    
    def _create_action_timeline(self, gap: ComplianceGap) -> Dict[str, str]:
        """Create timeline for individual remediation action."""
        if gap.severity == RiskLevel.CRITICAL:
            return {
                'start': 'immediate',
                'milestone_1': '1 week - assessment complete',
                'milestone_2': '2 weeks - controls implemented',
                'completion': '3 weeks - validation complete'
            }
        elif gap.severity == RiskLevel.HIGH:
            return {
                'start': '1 week',
                'milestone_1': '3 weeks - assessment complete',
                'milestone_2': '6 weeks - controls implemented',
                'completion': '8 weeks - validation complete'
            }
        else:
            return {
                'start': '4 weeks',
                'milestone_1': '8 weeks - assessment complete',
                'milestone_2': '12 weeks - controls implemented',
                'completion': '16 weeks - validation complete'
            }
    
    def _create_success_criteria(self, gap: ComplianceGap) -> List[str]:
        """Create success criteria for remediation action."""
        return [
            f"Compliance gap in {gap.control_id} fully addressed",
            f"Control effectiveness validated through testing",
            f"Documentation updated and approved",
            f"Audit evidence collected and verified"
        ]
    
    def _identify_action_dependencies(self, gap: ComplianceGap, 
                                    all_gaps: List[ComplianceGap]) -> List[str]:
        """Identify dependencies between remediation actions."""
        dependencies = []
        
        # Check for framework dependencies
        same_framework_gaps = [g for g in all_gaps if g.framework == gap.framework and g != gap]
        
        # Higher severity gaps should be dependencies for lower severity ones
        higher_severity_gaps = [g for g in same_framework_gaps 
                              if ['low', 'medium', 'high', 'critical'].index(g.severity.value) > 
                                 ['low', 'medium', 'high', 'critical'].index(gap.severity.value)]
        
        for dep_gap in higher_severity_gaps:
            dependencies.append(f"Complete remediation of {dep_gap.control_id}")
        
        return dependencies