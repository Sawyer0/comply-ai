"""
Alert Generator

This module generates quality alerts from metrics violations.
Follows SRP by focusing solely on alert generation logic.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from .models import QualityAlert, AlertSeverity, ThresholdConfig, AlertRule

logger = logging.getLogger(__name__)


class AlertGenerator:
    """
    Generates quality alerts from metric violations.

    Single responsibility: Convert metric violations into alert objects.
    """

    def __init__(self, thresholds: List[ThresholdConfig], rules: List[AlertRule]):
        self.thresholds = {f"{t.component}.{t.metric_name}": t for t in thresholds}
        self.rules = {f"{r.component}.{r.rule_name}": r for r in rules}

        logger.info(
            "Alert Generator initialized",
            thresholds=len(self.thresholds),
            rules=len(self.rules),
        )

    def generate_alerts(
        self, component: str, metrics: Dict[str, Any]
    ) -> List[QualityAlert]:
        """
        Generate alerts from component metrics.

        Args:
            component: Source component name
            metrics: Quality metrics to check

        Returns:
            List of generated alerts
        """
        alerts = []

        # Check threshold violations
        threshold_alerts = self._check_threshold_violations(component, metrics)
        alerts.extend(threshold_alerts)

        # Check rule violations
        rule_alerts = self._check_rule_violations(component, metrics)
        alerts.extend(rule_alerts)

        return alerts

    def _check_threshold_violations(
        self, component: str, metrics: Dict[str, Any]
    ) -> List[QualityAlert]:
        """Check metrics against configured thresholds."""
        alerts = []

        for metric_name, metric_value in metrics.items():
            if not isinstance(metric_value, (int, float)):
                continue

            threshold_key = f"{component}.{metric_name}"
            threshold_config = self.thresholds.get(threshold_key)

            if threshold_config and self._is_threshold_violated(
                metric_value, threshold_config
            ):
                alert = self._create_threshold_alert(
                    component, metric_name, metric_value, threshold_config, metrics
                )
                alerts.append(alert)

        return alerts

    def _check_rule_violations(
        self, component: str, metrics: Dict[str, Any]
    ) -> List[QualityAlert]:
        """Check metrics against composite rules."""
        alerts = []

        for rule_key, rule in self.rules.items():
            if rule.component == component:
                if self._is_rule_violated(metrics, rule):
                    alert = self._create_rule_alert(component, rule, metrics)
                    alerts.append(alert)

        return alerts

    def _is_threshold_violated(self, value: float, threshold: ThresholdConfig) -> bool:
        """Check if value violates threshold."""
        if threshold.threshold_type == "greater_than":
            return value > threshold.threshold_value
        elif threshold.threshold_type == "less_than":
            return value < threshold.threshold_value
        elif threshold.threshold_type == "equals":
            return value == threshold.threshold_value
        elif threshold.threshold_type == "not_equals":
            return value != threshold.threshold_value
        elif threshold.threshold_type == "greater_equal":
            return value >= threshold.threshold_value
        elif threshold.threshold_type == "less_equal":
            return value <= threshold.threshold_value
        else:
            return False

    def _is_rule_violated(self, metrics: Dict[str, Any], rule: AlertRule) -> bool:
        """Check if composite rule is violated."""
        try:
            conditions = rule.conditions

            # Simple condition evaluation (can be extended)
            if "min_metrics" in conditions:
                required_metrics = conditions["min_metrics"]
                if not all(metric in metrics for metric in required_metrics):
                    return False

            if "thresholds" in conditions:
                for metric, threshold_info in conditions["thresholds"].items():
                    if metric not in metrics:
                        continue

                    value = metrics[metric]
                    threshold_type = threshold_info.get("type", "greater_than")
                    threshold_value = threshold_info.get("value")

                    if threshold_type == "greater_than" and value <= threshold_value:
                        return False
                    elif threshold_type == "less_than" and value >= threshold_value:
                        return False

            return True

        except Exception as e:
            logger.error("Rule evaluation failed", rule=rule.rule_name, error=str(e))
            return False

    def _create_threshold_alert(
        self,
        component: str,
        metric_name: str,
        metric_value: float,
        threshold: ThresholdConfig,
        all_metrics: Dict[str, Any],
    ) -> QualityAlert:
        """Create alert for threshold violation."""
        alert_id = f"threshold_{component}_{metric_name}_{int(datetime.now(timezone.utc).timestamp())}"

        return QualityAlert(
            id=alert_id,
            alert_type="threshold_violation",
            severity=threshold.severity,
            title=f"{component} {metric_name} threshold violation",
            description=f"Metric {metric_name} value {metric_value} violates {threshold.severity.value} threshold {threshold.threshold_value}",
            source_component=component,
            metrics=all_metrics,
            threshold_violated={
                "metric": metric_name,
                "value": metric_value,
                "threshold_type": threshold.threshold_type,
                "threshold_value": threshold.threshold_value,
                "severity": threshold.severity.value,
            },
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    def _create_rule_alert(
        self, component: str, rule: AlertRule, metrics: Dict[str, Any]
    ) -> QualityAlert:
        """Create alert for rule violation."""
        alert_id = f"rule_{component}_{rule.rule_name}_{int(datetime.now(timezone.utc).timestamp())}"

        return QualityAlert(
            id=alert_id,
            alert_type="rule_violation",
            severity=rule.severity,
            title=f"{component} rule violation: {rule.rule_name}",
            description=rule.description,
            source_component=component,
            metrics=metrics,
            threshold_violated={
                "rule_name": rule.rule_name,
                "conditions": rule.conditions,
                "severity": rule.severity.value,
            },
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
