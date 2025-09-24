"""
Rollback Procedure Generator for safe threshold change management.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RollbackStep:
    """Single step in rollback procedure."""

    step_number: int
    step_name: str
    description: str
    estimated_duration_minutes: int
    required_permissions: List[str]
    verification_steps: List[str]
    automation_available: bool


@dataclass
class RollbackTrigger:
    """Condition that triggers rollback."""

    trigger_name: str
    condition: str
    severity: str
    auto_trigger: bool
    escalation_required: bool
    notification_list: List[str]


@dataclass
class RollbackProcedure:
    """Complete rollback procedure for threshold changes."""

    detector_id: str
    original_threshold: float
    current_threshold: float
    rollback_triggers: List[RollbackTrigger]
    rollback_steps: List[RollbackStep]
    total_rollback_time_minutes: int
    post_rollback_monitoring: Dict[str, Any]
    communication_plan: Dict[str, Any]
    lessons_learned_template: Dict[str, Any]


class RollbackProcedureGenerator:
    """
    Generates rollback procedures for safe threshold change management.

    Responsible for creating comprehensive rollback procedures with
    clear triggers, step-by-step instructions, and post-rollback actions.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rollback_templates = self._load_rollback_templates()

    def generate_rollback_procedure(
        self,
        detector_id: str,
        original_threshold: float,
        current_threshold: float,
        deployment_strategy: str,
        risk_assessment: Dict[str, Any],
        business_context: Dict[str, Any] = None,
    ) -> RollbackProcedure:
        """
        Generate comprehensive rollback procedure.

        Args:
            detector_id: ID of the detector
            original_threshold: Original threshold before changes
            current_threshold: Current threshold value
            deployment_strategy: Deployment strategy used
            risk_assessment: Risk assessment results
            business_context: Business context for rollback

        Returns:
            RollbackProcedure with detailed rollback instructions
        """
        try:
            logger.info(
                "Generating rollback procedure",
                detector_id=detector_id,
                original_threshold=original_threshold,
                current_threshold=current_threshold,
            )

            business_context = business_context or {}

            # Generate rollback triggers
            rollback_triggers = self._generate_rollback_triggers(
                risk_assessment, business_context, deployment_strategy
            )

            # Generate rollback steps
            rollback_steps = self._generate_rollback_steps(
                detector_id,
                original_threshold,
                current_threshold,
                deployment_strategy,
                risk_assessment,
            )

            # Calculate total rollback time
            total_time = sum(step.estimated_duration_minutes for step in rollback_steps)

            # Generate post-rollback monitoring plan
            post_rollback_monitoring = self._generate_post_rollback_monitoring(
                risk_assessment, business_context
            )

            # Generate communication plan
            communication_plan = self._generate_communication_plan(
                detector_id, risk_assessment, business_context
            )

            # Generate lessons learned template
            lessons_learned_template = self._generate_lessons_learned_template()

            procedure = RollbackProcedure(
                detector_id=detector_id,
                original_threshold=original_threshold,
                current_threshold=current_threshold,
                rollback_triggers=rollback_triggers,
                rollback_steps=rollback_steps,
                total_rollback_time_minutes=total_time,
                post_rollback_monitoring=post_rollback_monitoring,
                communication_plan=communication_plan,
                lessons_learned_template=lessons_learned_template,
            )

            logger.info(
                "Rollback procedure generated",
                detector_id=detector_id,
                total_rollback_time=total_time,
                trigger_count=len(rollback_triggers),
            )

            return procedure

        except Exception as e:
            logger.error(
                "Rollback procedure generation failed",
                error=str(e),
                detector_id=detector_id,
            )
            return self._create_error_procedure(
                detector_id, original_threshold, current_threshold, str(e)
            )

    def _generate_rollback_triggers(
        self,
        risk_assessment: Dict[str, Any],
        business_context: Dict[str, Any],
        deployment_strategy: str,
    ) -> List[RollbackTrigger]:
        """Generate rollback triggers based on risk assessment."""
        try:
            triggers = []

            # Performance-based triggers
            triggers.append(
                RollbackTrigger(
                    trigger_name="performance_degradation",
                    condition="F1 score drops below 80% of baseline for 15 minutes",
                    severity="high",
                    auto_trigger=True,
                    escalation_required=False,
                    notification_list=["ops-team", "security-team"],
                )
            )

            triggers.append(
                RollbackTrigger(
                    trigger_name="false_positive_spike",
                    condition="False positive rate increases by >50% for 10 minutes",
                    severity="medium",
                    auto_trigger=deployment_strategy
                    in ["canary_deployment", "blue_green_deployment"],
                    escalation_required=True,
                    notification_list=["ops-team", "security-lead"],
                )
            )

            triggers.append(
                RollbackTrigger(
                    trigger_name="false_negative_spike",
                    condition="False negative rate increases by >30% for 5 minutes",
                    severity="critical",
                    auto_trigger=True,
                    escalation_required=True,
                    notification_list=["ops-team", "security-team", "management"],
                )
            )

            # System-based triggers
            triggers.append(
                RollbackTrigger(
                    trigger_name="error_rate_increase",
                    condition="System error rate >5% for 5 minutes",
                    severity="high",
                    auto_trigger=True,
                    escalation_required=False,
                    notification_list=["ops-team"],
                )
            )

            # Business-based triggers
            if business_context.get("strict_sla", False):
                triggers.append(
                    RollbackTrigger(
                        trigger_name="sla_breach",
                        condition="SLA metrics breach defined thresholds",
                        severity="critical",
                        auto_trigger=True,
                        escalation_required=True,
                        notification_list=[
                            "ops-team",
                            "sla-manager",
                            "customer-success",
                        ],
                    )
                )

            # Manual triggers
            triggers.append(
                RollbackTrigger(
                    trigger_name="manual_escalation",
                    condition="Manual rollback requested by authorized personnel",
                    severity="variable",
                    auto_trigger=False,
                    escalation_required=False,
                    notification_list=["ops-team", "requester"],
                )
            )

            # Risk-specific triggers
            risk_factors = risk_assessment.get("risks", [])
            for risk in risk_factors:
                if risk.get("severity") == "critical":
                    triggers.append(
                        RollbackTrigger(
                            trigger_name=f"risk_{risk.get('type', 'unknown')}",
                            condition=f"Critical risk condition: {risk.get('description', 'Unknown')}",
                            severity="critical",
                            auto_trigger=True,
                            escalation_required=True,
                            notification_list=[
                                "ops-team",
                                "security-team",
                                "management",
                            ],
                        )
                    )

            return triggers

        except Exception as e:
            logger.error("Rollback triggers generation failed: %s", str(e))
            return [
                RollbackTrigger(
                    trigger_name="error",
                    condition="Error in trigger generation",
                    severity="unknown",
                    auto_trigger=False,
                    escalation_required=True,
                    notification_list=["ops-team"],
                )
            ]

    def _generate_rollback_steps(
        self,
        detector_id: str,
        original_threshold: float,
        current_threshold: float,
        deployment_strategy: str,
        risk_assessment: Dict[str, Any],
    ) -> List[RollbackStep]:
        """Generate rollback steps based on deployment strategy."""
        try:
            strategy_generators = {
                "immediate_deployment": self._generate_immediate_rollback_steps,
                "rolling_deployment": self._generate_rolling_rollback_steps,
                "blue_green_deployment": self._generate_blue_green_rollback_steps,
                "canary_deployment": self._generate_canary_rollback_steps,
            }

            generator = strategy_generators.get(
                deployment_strategy, self._generate_standard_rollback_steps
            )

            return generator(
                detector_id, original_threshold, current_threshold, risk_assessment
            )

        except Exception as e:
            logger.error("Rollback steps generation failed: %s", str(e))
            return self._generate_emergency_rollback_steps(
                detector_id, original_threshold
            )

    def _generate_immediate_rollback_steps(
        self,
        detector_id: str,
        original_threshold: float,
        current_threshold: float,
        risk_assessment: Dict[str, Any],
    ) -> List[RollbackStep]:
        """Generate rollback steps for immediate deployment."""
        return [
            RollbackStep(
                step_number=1,
                step_name="immediate_threshold_revert",
                description=f"Immediately revert threshold from {current_threshold} to {original_threshold}",
                estimated_duration_minutes=2,
                required_permissions=["threshold_update"],
                verification_steps=[
                    "Verify threshold value updated",
                    "Check system accepts new threshold",
                    "Confirm no immediate errors",
                ],
                automation_available=True,
            ),
            RollbackStep(
                step_number=2,
                step_name="performance_verification",
                description="Verify performance returns to baseline",
                estimated_duration_minutes=10,
                required_permissions=["monitoring_access"],
                verification_steps=[
                    "Check key performance metrics",
                    "Verify error rates return to normal",
                    "Confirm alert conditions cleared",
                ],
                automation_available=True,
            ),
            RollbackStep(
                step_number=3,
                step_name="stakeholder_notification",
                description="Notify stakeholders of rollback completion",
                estimated_duration_minutes=5,
                required_permissions=["notification_send"],
                verification_steps=[
                    "Send rollback completion notification",
                    "Update status dashboards",
                    "Log rollback in incident system",
                ],
                automation_available=True,
            ),
        ]

    def _generate_rolling_rollback_steps(
        self,
        detector_id: str,
        original_threshold: float,
        current_threshold: float,
        risk_assessment: Dict[str, Any],
    ) -> List[RollbackStep]:
        """Generate rollback steps for rolling deployment."""
        return [
            RollbackStep(
                step_number=1,
                step_name="halt_rollout",
                description="Immediately halt any ongoing rollout phases",
                estimated_duration_minutes=2,
                required_permissions=["deployment_control"],
                verification_steps=[
                    "Confirm rollout process stopped",
                    "Verify no pending changes",
                    "Check system state is stable",
                ],
                automation_available=True,
            ),
            RollbackStep(
                step_number=2,
                step_name="revert_to_original",
                description=f"Revert threshold to original value {original_threshold}",
                estimated_duration_minutes=3,
                required_permissions=["threshold_update"],
                verification_steps=[
                    "Verify threshold reverted",
                    "Check configuration consistency",
                    "Confirm system accepts change",
                ],
                automation_available=True,
            ),
            RollbackStep(
                step_number=3,
                step_name="performance_stabilization",
                description="Monitor performance stabilization",
                estimated_duration_minutes=15,
                required_permissions=["monitoring_access"],
                verification_steps=[
                    "Monitor key metrics for 15 minutes",
                    "Verify performance returns to baseline",
                    "Check for any residual issues",
                ],
                automation_available=True,
            ),
            RollbackStep(
                step_number=4,
                step_name="cleanup_and_notification",
                description="Clean up deployment artifacts and notify stakeholders",
                estimated_duration_minutes=10,
                required_permissions=["cleanup_access", "notification_send"],
                verification_steps=[
                    "Remove temporary configurations",
                    "Send rollback notifications",
                    "Update documentation",
                ],
                automation_available=False,
            ),
        ]

    def _generate_blue_green_rollback_steps(
        self,
        detector_id: str,
        original_threshold: float,
        current_threshold: float,
        risk_assessment: Dict[str, Any],
    ) -> List[RollbackStep]:
        """Generate rollback steps for blue-green deployment."""
        return [
            RollbackStep(
                step_number=1,
                step_name="traffic_cutback",
                description="Immediately route all traffic back to blue environment",
                estimated_duration_minutes=3,
                required_permissions=["traffic_routing"],
                verification_steps=[
                    "Verify 100% traffic on blue environment",
                    "Check green environment isolated",
                    "Confirm routing rules updated",
                ],
                automation_available=True,
            ),
            RollbackStep(
                step_number=2,
                step_name="green_environment_shutdown",
                description="Safely shutdown green environment",
                estimated_duration_minutes=5,
                required_permissions=["environment_control"],
                verification_steps=[
                    "Gracefully shutdown green services",
                    "Verify no active connections",
                    "Confirm resources released",
                ],
                automation_available=True,
            ),
            RollbackStep(
                step_number=3,
                step_name="blue_environment_verification",
                description="Verify blue environment performance",
                estimated_duration_minutes=10,
                required_permissions=["monitoring_access"],
                verification_steps=[
                    "Check blue environment health",
                    "Verify performance metrics",
                    "Confirm original threshold active",
                ],
                automation_available=True,
            ),
            RollbackStep(
                step_number=4,
                step_name="post_rollback_cleanup",
                description="Clean up green environment and notify stakeholders",
                estimated_duration_minutes=15,
                required_permissions=["cleanup_access", "notification_send"],
                verification_steps=[
                    "Remove green environment resources",
                    "Update load balancer configuration",
                    "Send completion notifications",
                ],
                automation_available=False,
            ),
        ]

    def _generate_canary_rollback_steps(
        self,
        detector_id: str,
        original_threshold: float,
        current_threshold: float,
        risk_assessment: Dict[str, Any],
    ) -> List[RollbackStep]:
        """Generate rollback steps for canary deployment."""
        return [
            RollbackStep(
                step_number=1,
                step_name="canary_traffic_halt",
                description="Immediately stop all traffic to canary",
                estimated_duration_minutes=2,
                required_permissions=["traffic_routing"],
                verification_steps=[
                    "Verify 0% traffic to canary",
                    "Check all traffic on production",
                    "Confirm canary isolated",
                ],
                automation_available=True,
            ),
            RollbackStep(
                step_number=2,
                step_name="canary_environment_analysis",
                description="Capture canary environment state for analysis",
                estimated_duration_minutes=10,
                required_permissions=["monitoring_access", "log_access"],
                verification_steps=[
                    "Export canary metrics",
                    "Capture relevant logs",
                    "Document observed issues",
                ],
                automation_available=False,
            ),
            RollbackStep(
                step_number=3,
                step_name="canary_shutdown",
                description="Safely shutdown canary environment",
                estimated_duration_minutes=5,
                required_permissions=["environment_control"],
                verification_steps=[
                    "Graceful canary shutdown",
                    "Verify resource cleanup",
                    "Confirm no residual processes",
                ],
                automation_available=True,
            ),
            RollbackStep(
                step_number=4,
                step_name="production_verification",
                description="Verify production environment stability",
                estimated_duration_minutes=15,
                required_permissions=["monitoring_access"],
                verification_steps=[
                    "Check production health metrics",
                    "Verify original threshold active",
                    "Monitor for 15 minutes",
                ],
                automation_available=True,
            ),
            RollbackStep(
                step_number=5,
                step_name="incident_documentation",
                description="Document rollback and initiate post-mortem",
                estimated_duration_minutes=20,
                required_permissions=["documentation_access"],
                verification_steps=[
                    "Create incident report",
                    "Schedule post-mortem meeting",
                    "Notify relevant stakeholders",
                ],
                automation_available=False,
            ),
        ]

    def _generate_standard_rollback_steps(
        self,
        detector_id: str,
        original_threshold: float,
        current_threshold: float,
        risk_assessment: Dict[str, Any],
    ) -> List[RollbackStep]:
        """Generate standard rollback steps."""
        return [
            RollbackStep(
                step_number=1,
                step_name="threshold_revert",
                description=f"Revert threshold from {current_threshold} to {original_threshold}",
                estimated_duration_minutes=5,
                required_permissions=["threshold_update"],
                verification_steps=[
                    "Update threshold configuration",
                    "Verify change applied",
                    "Check system stability",
                ],
                automation_available=True,
            ),
            RollbackStep(
                step_number=2,
                step_name="monitoring_and_notification",
                description="Monitor system and notify stakeholders",
                estimated_duration_minutes=15,
                required_permissions=["monitoring_access", "notification_send"],
                verification_steps=[
                    "Monitor key metrics",
                    "Send rollback notifications",
                    "Update status pages",
                ],
                automation_available=True,
            ),
        ]

    def _generate_emergency_rollback_steps(
        self, detector_id: str, original_threshold: float
    ) -> List[RollbackStep]:
        """Generate emergency rollback steps."""
        return [
            RollbackStep(
                step_number=1,
                step_name="emergency_revert",
                description=f"Emergency revert to threshold {original_threshold}",
                estimated_duration_minutes=1,
                required_permissions=["emergency_access"],
                verification_steps=["Verify revert completed"],
                automation_available=True,
            ),
        ]

    def _generate_post_rollback_monitoring(
        self, risk_assessment: Dict[str, Any], business_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate post-rollback monitoring plan."""
        return {
            "monitoring_duration_hours": 24,
            "key_metrics": [
                "False positive rate",
                "False negative rate",
                "System error rate",
                "Response time",
                "Throughput",
            ],
            "alert_thresholds": {
                "performance_degradation": "Any metric >10% worse than baseline",
                "error_rate": "Error rate >2%",
                "response_time": "Response time >150% of baseline",
            },
            "reporting_schedule": [
                "Immediate status update",
                "1-hour status report",
                "4-hour detailed report",
                "24-hour final report",
            ],
            "escalation_criteria": [
                "Any metric continues degrading after rollback",
                "New issues emerge post-rollback",
                "System instability persists",
            ],
        }

    def _generate_communication_plan(
        self,
        detector_id: str,
        risk_assessment: Dict[str, Any],
        business_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate communication plan for rollback."""
        return {
            "immediate_notifications": [
                "Operations team",
                "Security team",
                "On-call engineer",
            ],
            "escalation_notifications": [
                "Team leads",
                "Management",
                "Customer success (if customer-facing)",
            ],
            "communication_channels": [
                "Slack alerts",
                "Email notifications",
                "Status page updates",
                "Incident management system",
            ],
            "message_templates": {
                "rollback_initiated": f"Rollback initiated for detector {detector_id} due to performance issues",
                "rollback_completed": f"Rollback completed for detector {detector_id}, monitoring in progress",
                "rollback_verified": f"Rollback verified successful for detector {detector_id}, system stable",
            },
            "stakeholder_updates": {
                "frequency": "Every 30 minutes during rollback",
                "format": "Brief status with key metrics",
                "escalation_threshold": "If rollback takes >1 hour",
            },
        }

    def _generate_lessons_learned_template(self) -> Dict[str, Any]:
        """Generate lessons learned template."""
        return {
            "incident_summary": {
                "what_happened": "",
                "when_it_happened": "",
                "how_long_it_lasted": "",
                "impact_assessment": "",
            },
            "root_cause_analysis": {
                "immediate_cause": "",
                "contributing_factors": [],
                "systemic_issues": [],
            },
            "response_evaluation": {
                "what_went_well": [],
                "what_could_be_improved": [],
                "rollback_effectiveness": "",
            },
            "action_items": {
                "immediate_fixes": [],
                "process_improvements": [],
                "monitoring_enhancements": [],
                "training_needs": [],
            },
            "prevention_measures": {
                "detection_improvements": [],
                "testing_enhancements": [],
                "deployment_process_changes": [],
            },
        }

    def _load_rollback_templates(self) -> Dict[str, Any]:
        """Load rollback procedure templates."""
        return self.config.get(
            "rollback_templates",
            {
                "standard": {
                    "max_rollback_time_minutes": 30,
                    "required_approvals": ["ops_lead"],
                    "automation_level": "high",
                },
                "high_risk": {
                    "max_rollback_time_minutes": 15,
                    "required_approvals": ["ops_lead", "security_lead"],
                    "automation_level": "medium",
                },
                "critical": {
                    "max_rollback_time_minutes": 5,
                    "required_approvals": ["emergency_contact"],
                    "automation_level": "full",
                },
            },
        )

    def _create_error_procedure(
        self,
        detector_id: str,
        original_threshold: float,
        current_threshold: float,
        error_message: str,
    ) -> RollbackProcedure:
        """Create error rollback procedure."""
        return RollbackProcedure(
            detector_id=detector_id,
            original_threshold=original_threshold,
            current_threshold=current_threshold,
            rollback_triggers=[],
            rollback_steps=[],
            total_rollback_time_minutes=0,
            post_rollback_monitoring={"error": error_message},
            communication_plan={"error": error_message},
            lessons_learned_template={"error": error_message},
        )
