"""
Metrics Dashboard for Compliance AI Quality Monitoring
Tracks grounding rate, schema validity, hallucination rate, and other key metrics.
"""

import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for a single analysis."""
    analysis_id: str
    timestamp: datetime
    analysis_type: str
    grounding_validated: bool
    schema_valid: bool
    citation_accuracy: float
    hallucination_detected: bool
    risk_rating_calibration: Optional[float]
    refusal_appropriate: bool
    confidence_score: float
    processing_time_ms: int


@dataclass
class AggregatedMetrics:
    """Aggregated quality metrics over a time period."""
    period_start: datetime
    period_end: datetime
    total_analyses: int
    grounding_rate: float
    schema_validity_rate: float
    citation_accuracy_avg: float
    hallucination_rate: float
    risk_calibration_score: float
    refusal_correctness: float
    avg_confidence: float
    avg_processing_time_ms: float
    

class ComplianceMetricsCollector:
    """Collects and stores quality metrics for compliance analyses."""
    
    def __init__(self):
        self.metrics_storage: List[QualityMetrics] = []
        self.hallucination_patterns = self._load_hallucination_patterns()
        
    def _load_hallucination_patterns(self) -> List[str]:
        """Load patterns that indicate potential hallucinations."""
        return [
            "I am certain that",
            "The law clearly states",
            "All experts agree",
            "Without question",
            "Definitely required",
            "Never permitted",
            "Always illegal",
            "100% compliance",
            "Zero risk",
            "Guaranteed safe"
        ]
    
    def record_analysis_metrics(self, 
                              analysis_output: Dict[str, Any],
                              grounding_result: Any,
                              processing_time_ms: int) -> QualityMetrics:
        """
        Record metrics for a single analysis.
        
        Args:
            analysis_output: The compliance analysis output
            grounding_result: Result from grounding validation
            processing_time_ms: Time taken to process
            
        Returns:
            QualityMetrics object
        """
        # Check for hallucination patterns
        analysis_text = json.dumps(analysis_output)
        hallucination_detected = any(
            pattern.lower() in analysis_text.lower() 
            for pattern in self.hallucination_patterns
        )
        
        # Check schema validity
        schema_valid = not bool(analysis_output.get('grounding_errors', []))
        
        # Calculate citation accuracy
        citation_accuracy = getattr(grounding_result, 'grounding_score', 0.0)
        
        # Check refusal appropriateness
        refusal_appropriate = self._assess_refusal_appropriateness(analysis_output)
        
        metrics = QualityMetrics(
            analysis_id=analysis_output.get('analysis_id', f"analysis_{len(self.metrics_storage)}"),
            timestamp=datetime.now(),
            analysis_type=analysis_output.get('analysis_type', 'unknown'),
            grounding_validated=analysis_output.get('grounding_validated', False),
            schema_valid=schema_valid,
            citation_accuracy=citation_accuracy,
            hallucination_detected=hallucination_detected,
            risk_rating_calibration=self._calculate_risk_calibration(analysis_output),
            refusal_appropriate=refusal_appropriate,
            confidence_score=analysis_output.get('confidence', 0.0),
            processing_time_ms=processing_time_ms
        )
        
        self.metrics_storage.append(metrics)
        return metrics
    
    def _assess_refusal_appropriateness(self, analysis_output: Dict[str, Any]) -> bool:
        """Assess if refusal behavior was appropriate."""
        
        # Check if analysis was refused
        if analysis_output.get('analysis_type') == 'constitutional_refusal':
            # Refusal should happen when:
            # 1. No citations available
            # 2. Cross-regulatory conflicts
            # 3. Insufficient evidence
            refusal_reason = analysis_output.get('refusal_reason', '')
            valid_refusal_reasons = [
                'citation', 'evidence', 'jurisdiction', 
                'conflict', 'legal interpretation'
            ]
            return any(reason in refusal_reason.lower() for reason in valid_refusal_reasons)
        
        # Check if refusal should have happened but didn't
        confidence = analysis_output.get('confidence', 1.0)
        has_citations = bool(analysis_output.get('citations', []))
        evidence_based = analysis_output.get('risk_rationale', {}).get('evidence_based', True)
        
        # Should refuse if very low confidence without citations
        should_refuse = confidence < 0.3 and not has_citations and not evidence_based
        if should_refuse:
            return False  # Should have refused but didn't
            
        return True  # Appropriate behavior
    
    def _calculate_risk_calibration(self, analysis_output: Dict[str, Any]) -> Optional[float]:
        """Calculate risk rating calibration score."""
        
        # This would normally compare against SME gold standard
        # For simulation, we use heuristics
        
        risk_level = analysis_output.get('risk_rationale', {}).get('level', '')
        confidence = analysis_output.get('confidence', 0.0)
        evidence_based = analysis_output.get('risk_rationale', {}).get('evidence_based', False)
        
        # Good calibration means:
        # - High confidence with evidence-based assessment
        # - Conservative approach when not evidence-based
        # - Risk level appropriate for confidence
        
        calibration_score = 0.0
        
        if evidence_based and confidence > 0.8:
            calibration_score += 0.4
        elif not evidence_based and analysis_output.get('conservative_approach_applied', False):
            calibration_score += 0.4
            
        # Risk level should match confidence
        if risk_level == 'critical' and confidence > 0.7:
            calibration_score += 0.3
        elif risk_level in ['low', 'negligible'] and confidence > 0.8:
            calibration_score += 0.3
        elif risk_level in ['medium', 'high'] and 0.5 <= confidence <= 0.9:
            calibration_score += 0.3
            
        # Conservative bias when uncertain
        if confidence < 0.7 and risk_level in ['high', 'critical']:
            calibration_score += 0.3
            
        return calibration_score
    
    def get_aggregated_metrics(self, 
                              period_hours: int = 24) -> AggregatedMetrics:
        """
        Get aggregated metrics for the specified period.
        
        Args:
            period_hours: Hours to look back for metrics
            
        Returns:
            AggregatedMetrics object
        """
        cutoff_time = datetime.now() - timedelta(hours=period_hours)
        recent_metrics = [m for m in self.metrics_storage if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return AggregatedMetrics(
                period_start=cutoff_time,
                period_end=datetime.now(),
                total_analyses=0,
                grounding_rate=0.0,
                schema_validity_rate=0.0,
                citation_accuracy_avg=0.0,
                hallucination_rate=0.0,
                risk_calibration_score=0.0,
                refusal_correctness=0.0,
                avg_confidence=0.0,
                avg_processing_time_ms=0.0
            )
        
        total = len(recent_metrics)
        
        return AggregatedMetrics(
            period_start=cutoff_time,
            period_end=datetime.now(),
            total_analyses=total,
            grounding_rate=sum(m.grounding_validated for m in recent_metrics) / total,
            schema_validity_rate=sum(m.schema_valid for m in recent_metrics) / total,
            citation_accuracy_avg=sum(m.citation_accuracy for m in recent_metrics) / total,
            hallucination_rate=sum(m.hallucination_detected for m in recent_metrics) / total,
            risk_calibration_score=sum(m.risk_rating_calibration or 0 for m in recent_metrics) / total,
            refusal_correctness=sum(m.refusal_appropriate for m in recent_metrics) / total,
            avg_confidence=sum(m.confidence_score for m in recent_metrics) / total,
            avg_processing_time_ms=sum(m.processing_time_ms for m in recent_metrics) / total
        )


class MetricsDashboard:
    """Dashboard for displaying compliance AI quality metrics."""
    
    def __init__(self, metrics_collector: ComplianceMetricsCollector):
        self.metrics_collector = metrics_collector
        
    def print_dashboard(self, period_hours: int = 24) -> None:
        """Print a formatted metrics dashboard."""
        
        metrics = self.metrics_collector.get_aggregated_metrics(period_hours)
        
        print("=" * 60)
        print("🏠 COMPLIANCE AI QUALITY DASHBOARD")
        print("=" * 60)
        print(f"Period: {metrics.period_start.strftime('%Y-%m-%d %H:%M')} to {metrics.period_end.strftime('%Y-%m-%d %H:%M')}")
        print(f"Total Analyses: {metrics.total_analyses}")
        print()
        
        print("📊 CORE QUALITY METRICS")
        print("-" * 30)
        self._print_metric("Grounding Rate", metrics.grounding_rate, target=0.95, unit="%")
        self._print_metric("Schema Validity Rate", metrics.schema_validity_rate, target=0.98, unit="%")
        self._print_metric("Citation Accuracy", metrics.citation_accuracy_avg, target=0.90, unit="%")
        self._print_metric("Hallucination Rate", metrics.hallucination_rate, target=0.02, unit="%", invert=True)
        self._print_metric("Risk Calibration Score", metrics.risk_calibration_score, target=0.85, unit="%")
        self._print_metric("Refusal Correctness", metrics.refusal_correctness, target=0.95, unit="%")
        print()
        
        print("⚡ PERFORMANCE METRICS")
        print("-" * 30)
        self._print_metric("Average Confidence", metrics.avg_confidence, target=0.75, unit="%")
        self._print_metric("Avg Processing Time", metrics.avg_processing_time_ms, target=2000, unit="ms", invert=True)
        print()
        
        # Analysis type breakdown
        self._print_analysis_breakdown(period_hours)
        
        # Failure exemplars
        self._print_failure_exemplars(period_hours)
        
    def _print_metric(self, name: str, value: float, target: float, unit: str, invert: bool = False) -> None:
        """Print a single metric with status indicator."""
        
        if unit == "%":
            display_value = value * 100
            display_target = target * 100
        else:
            display_value = value
            display_target = target
            
        # Determine status
        if invert:
            status = "✅" if value <= target else "❌" if value > target * 1.5 else "⚠️"
        else:
            status = "✅" if value >= target else "❌" if value < target * 0.8 else "⚠️"
            
        print(f"{status} {name:<20}: {display_value:6.1f}{unit} (target: {display_target:.1f}{unit})")
    
    def _print_analysis_breakdown(self, period_hours: int) -> None:
        """Print breakdown by analysis type."""
        
        cutoff_time = datetime.now() - timedelta(hours=period_hours)
        recent_metrics = [m for m in self.metrics_collector.metrics_storage if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return
            
        type_counts = Counter(m.analysis_type for m in recent_metrics)
        
        print("📈 ANALYSIS TYPE BREAKDOWN")
        print("-" * 30)
        for analysis_type, count in type_counts.most_common():
            percentage = (count / len(recent_metrics)) * 100
            print(f"  {analysis_type:<18}: {count:3d} ({percentage:4.1f}%)")
        print()
    
    def _print_failure_exemplars(self, period_hours: int) -> None:
        """Print examples of failures for learning."""
        
        cutoff_time = datetime.now() - timedelta(hours=period_hours)
        recent_metrics = [m for m in self.metrics_collector.metrics_storage if m.timestamp >= cutoff_time]
        
        failures = []
        
        for metric in recent_metrics:
            if not metric.grounding_validated:
                failures.append(f"❌ Grounding failure: {metric.analysis_id}")
            if not metric.schema_valid:
                failures.append(f"❌ Schema invalid: {metric.analysis_id}")
            if metric.hallucination_detected:
                failures.append(f"❌ Hallucination detected: {metric.analysis_id}")
            if not metric.refusal_appropriate:
                failures.append(f"❌ Inappropriate refusal: {metric.analysis_id}")
                
        if failures:
            print("🚨 FAILURE EXEMPLARS (for eval-to-train)")
            print("-" * 30)
            for failure in failures[:5]:  # Show top 5
                print(f"  {failure}")
            if len(failures) > 5:
                print(f"  ... and {len(failures) - 5} more")
            print()
        else:
            print("✅ No failures detected in this period!")
            print()
    
    def export_metrics_json(self, period_hours: int = 24) -> Dict[str, Any]:
        """Export metrics as JSON for external systems."""
        
        metrics = self.metrics_collector.get_aggregated_metrics(period_hours)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "period_hours": period_hours,
            "metrics": asdict(metrics),
            "status": self._calculate_overall_status(metrics),
            "alerts": self._generate_alerts(metrics)
        }
    
    def _calculate_overall_status(self, metrics: AggregatedMetrics) -> str:
        """Calculate overall system status."""
        
        if metrics.total_analyses == 0:
            return "no_data"
            
        critical_failures = (
            metrics.grounding_rate < 0.8 or
            metrics.schema_validity_rate < 0.9 or
            metrics.hallucination_rate > 0.1
        )
        
        if critical_failures:
            return "critical"
            
        warnings = (
            metrics.grounding_rate < 0.95 or
            metrics.citation_accuracy_avg < 0.85 or
            metrics.risk_calibration_score < 0.8
        )
        
        if warnings:
            return "warning"
            
        return "healthy"
    
    def _generate_alerts(self, metrics: AggregatedMetrics) -> List[str]:
        """Generate alerts for metric thresholds."""
        
        alerts = []
        
        if metrics.grounding_rate < 0.8:
            alerts.append("CRITICAL: Grounding rate below 80%")
        if metrics.schema_validity_rate < 0.9:
            alerts.append("CRITICAL: Schema validity below 90%")
        if metrics.hallucination_rate > 0.1:
            alerts.append("CRITICAL: Hallucination rate above 10%")
        if metrics.citation_accuracy_avg < 0.8:
            alerts.append("WARNING: Citation accuracy below 80%")
        if metrics.avg_processing_time_ms > 5000:
            alerts.append("WARNING: Processing time above 5 seconds")
            
        return alerts


def create_sample_metrics() -> ComplianceMetricsCollector:
    """Create sample metrics for demonstration."""
    
    collector = ComplianceMetricsCollector()
    
    # Sample outputs with varying quality
    sample_outputs = [
        {
            "analysis_id": "analysis_001",
            "analysis_type": "gap_analysis",
            "grounding_validated": True,
            "confidence": 0.92,
            "citations": [{"citation": "GDPR Art. 5(1)(a)", "chunk_text": "Sample text"}],
            "risk_rationale": {"level": "medium", "evidence_based": True}
        },
        {
            "analysis_id": "analysis_002", 
            "analysis_type": "risk_rating",
            "grounding_validated": False,
            "confidence": 0.45,
            "grounding_errors": ["Citation validation failed"],
            "risk_rationale": {"level": "high", "evidence_based": False}
        },
        {
            "analysis_id": "analysis_003",
            "analysis_type": "remediation_plan",
            "grounding_validated": True,
            "confidence": 0.88,
            "citations": [{"citation": "SOX Section 404", "chunk_text": "Sample compliance text"}],
            "risk_rationale": {"level": "high", "evidence_based": True}
        }
    ]
    
    # Mock grounding results
    class MockGroundingResult:
        def __init__(self, score):
            self.grounding_score = score
    
    # Record metrics
    for i, output in enumerate(sample_outputs):
        grounding_result = MockGroundingResult(0.9 if output["grounding_validated"] else 0.3)
        collector.record_analysis_metrics(output, grounding_result, 1200 + i * 100)
    
    return collector


if __name__ == "__main__":
    # Demo the dashboard
    collector = create_sample_metrics()
    dashboard = MetricsDashboard(collector)
    
    print("Sample Compliance AI Quality Dashboard:")
    dashboard.print_dashboard()
    
    # Export as JSON
    metrics_json = dashboard.export_metrics_json()
    print("\nJSON Export:")
    print(json.dumps(metrics_json, indent=2))
