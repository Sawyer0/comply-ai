"""
Quality metrics for RAG system.

Tracks and monitors the quality of RAG system performance.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class QualityMetric:
    """Represents a quality metric."""
    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QualityThreshold:
    """Represents a quality threshold."""
    metric_name: str
    threshold_value: float
    operator: str  # "gt", "lt", "eq", "gte", "lte"
    severity: str  # "warning", "error", "critical"
    description: str = ""


class RAGQualityMetrics:
    """Tracks and monitors RAG system quality metrics."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize quality metrics tracker.
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self.thresholds = {}
        self.logger = logging.getLogger(__name__)
    
    def track_retrieval_quality(self, query: str, retrieved_docs: List, 
                               relevance_scores: List[float]) -> None:
        """Track retrieval quality metrics."""
        try:
            # Calculate retrieval quality metrics
            avg_relevance = statistics.mean(relevance_scores) if relevance_scores else 0.0
            max_relevance = max(relevance_scores) if relevance_scores else 0.0
            min_relevance = min(relevance_scores) if relevance_scores else 0.0
            
            # Track metrics
            self._add_metric("retrieval_avg_relevance", avg_relevance)
            self._add_metric("retrieval_max_relevance", max_relevance)
            self._add_metric("retrieval_min_relevance", min_relevance)
            self._add_metric("retrieval_doc_count", len(retrieved_docs))
            
            # Calculate diversity metric
            diversity = self._calculate_diversity_metric(retrieved_docs)
            self._add_metric("retrieval_diversity", diversity)
            
            # Calculate coverage metric
            coverage = self._calculate_coverage_metric(retrieved_docs)
            self._add_metric("retrieval_coverage", coverage)
            
        except Exception as e:
            self.logger.error(f"Failed to track retrieval quality: {e}")
    
    def track_response_quality(self, query: str, response: str, 
                             user_feedback: Optional[Dict[str, Any]] = None) -> None:
        """Track response quality metrics."""
        try:
            # Calculate response quality metrics
            response_length = len(response)
            self._add_metric("response_length", response_length)
            
            # Calculate readability score (simplified)
            readability = self._calculate_readability_score(response)
            self._add_metric("response_readability", readability)
            
            # Calculate completeness score
            completeness = self._calculate_completeness_score(response)
            self._add_metric("response_completeness", completeness)
            
            # Track user feedback if available
            if user_feedback:
                self._track_user_feedback(user_feedback)
            
        except Exception as e:
            self.logger.error(f"Failed to track response quality: {e}")
    
    def track_knowledge_coverage(self, query: str, available_docs: List) -> None:
        """Track knowledge base coverage for queries."""
        try:
            # Calculate coverage metrics
            total_docs = len(available_docs)
            self._add_metric("knowledge_base_size", total_docs)
            
            # Calculate framework coverage
            frameworks = set(doc.regulatory_framework for doc in available_docs 
                          if hasattr(doc, 'regulatory_framework') and doc.regulatory_framework)
            framework_coverage = len(frameworks)
            self._add_metric("framework_coverage", framework_coverage)
            
            # Calculate industry coverage
            industries = set(doc.industry for doc in available_docs 
                          if hasattr(doc, 'industry') and doc.industry)
            industry_coverage = len(industries)
            self._add_metric("industry_coverage", industry_coverage)
            
            # Calculate document type coverage
            doc_types = set(doc.document_type for doc in available_docs 
                          if hasattr(doc, 'document_type') and doc.document_type)
            type_coverage = len(doc_types)
            self._add_metric("document_type_coverage", type_coverage)
            
        except Exception as e:
            self.logger.error(f"Failed to track knowledge coverage: {e}")
    
    def set_quality_threshold(self, metric_name: str, threshold_value: float, 
                            operator: str, severity: str, description: str = "") -> None:
        """Set quality threshold for a metric."""
        self.thresholds[metric_name] = QualityThreshold(
            metric_name=metric_name,
            threshold_value=threshold_value,
            operator=operator,
            severity=severity,
            description=description
        )
    
    def check_quality_thresholds(self) -> List[Dict[str, Any]]:
        """Check if any metrics exceed quality thresholds."""
        violations = []
        
        for metric_name, threshold in self.thresholds.items():
            if metric_name in self.metrics_history:
                recent_values = list(self.metrics_history[metric_name])
                if recent_values:
                    current_value = recent_values[-1].value
                    
                    if self._evaluate_threshold(current_value, threshold):
                        violations.append({
                            "metric_name": metric_name,
                            "current_value": current_value,
                            "threshold_value": threshold.threshold_value,
                            "operator": threshold.operator,
                            "severity": threshold.severity,
                            "description": threshold.description,
                            "timestamp": datetime.utcnow()
                        })
        
        return violations
    
    def get_quality_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get quality metrics summary."""
        try:
            summary = {}
            
            for metric_name, history in self.metrics_history.items():
                if not history:
                    continue
                
                # Filter by time window if specified
                if time_window:
                    cutoff_time = datetime.utcnow() - time_window
                    recent_metrics = [m for m in history if m.timestamp >= cutoff_time]
                else:
                    recent_metrics = list(history)
                
                if recent_metrics:
                    values = [m.value for m in recent_metrics]
                    summary[metric_name] = {
                        "count": len(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "latest": values[-1],
                        "latest_timestamp": recent_metrics[-1].timestamp
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get quality summary: {e}")
            return {}
    
    def _add_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a metric to the history."""
        metric = QualityMetric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        self.metrics_history[name].append(metric)
    
    def _calculate_diversity_metric(self, docs: List) -> float:
        """Calculate diversity metric for retrieved documents."""
        if not docs:
            return 0.0
        
        # Simple diversity calculation based on unique sources
        sources = set()
        for doc in docs:
            if hasattr(doc, 'source'):
                sources.add(doc.source)
        
        return len(sources) / len(docs) if docs else 0.0
    
    def _calculate_coverage_metric(self, docs: List) -> float:
        """Calculate coverage metric for retrieved documents."""
        if not docs:
            return 0.0
        
        # Simple coverage calculation based on unique frameworks
        frameworks = set()
        for doc in docs:
            if hasattr(doc, 'regulatory_framework') and doc.regulatory_framework:
                frameworks.add(doc.regulatory_framework)
        
        return len(frameworks) / 10.0  # Assuming 10 major frameworks
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate readability score for text."""
        if not text:
            return 0.0
        
        # Simplified readability calculation
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Simple readability score (lower is better)
        if avg_words_per_sentence <= 15:
            return 0.9
        elif avg_words_per_sentence <= 20:
            return 0.7
        elif avg_words_per_sentence <= 25:
            return 0.5
        else:
            return 0.3
    
    def _calculate_completeness_score(self, text: str) -> float:
        """Calculate completeness score for text."""
        if not text:
            return 0.0
        
        # Simple completeness indicators
        completeness_indicators = [
            "recommendation", "suggestion", "guidance", "implementation",
            "risk", "control", "compliance", "audit", "assessment"
        ]
        
        text_lower = text.lower()
        found_indicators = sum(1 for indicator in completeness_indicators 
                             if indicator in text_lower)
        
        return found_indicators / len(completeness_indicators)
    
    def _track_user_feedback(self, feedback: Dict[str, Any]) -> None:
        """Track user feedback metrics."""
        try:
            if "rating" in feedback:
                rating = feedback["rating"]
                self._add_metric("user_rating", rating)
            
            if "helpful" in feedback:
                helpful = 1.0 if feedback["helpful"] else 0.0
                self._add_metric("user_helpfulness", helpful)
            
            if "accuracy" in feedback:
                accuracy = feedback["accuracy"]
                self._add_metric("user_accuracy_rating", accuracy)
            
        except Exception as e:
            self.logger.error(f"Failed to track user feedback: {e}")
    
    def _evaluate_threshold(self, value: float, threshold: QualityThreshold) -> bool:
        """Evaluate if a value meets a threshold condition."""
        if threshold.operator == "gt":
            return value > threshold.threshold_value
        elif threshold.operator == "lt":
            return value < threshold.threshold_value
        elif threshold.operator == "eq":
            return value == threshold.threshold_value
        elif threshold.operator == "gte":
            return value >= threshold.threshold_value
        elif threshold.operator == "lte":
            return value <= threshold.threshold_value
        else:
            return False


class QualityTracker:
    """Advanced quality tracking with real-time monitoring."""
    
    def __init__(self, quality_metrics: RAGQualityMetrics):
        """Initialize quality tracker."""
        self.quality_metrics = quality_metrics
        self.logger = logging.getLogger(__name__)
    
    def start_quality_monitoring(self) -> None:
        """Start quality monitoring."""
        # Set up default quality thresholds
        self._setup_default_thresholds()
        
        # Start monitoring loop
        self._start_monitoring_loop()
    
    def _setup_default_thresholds(self) -> None:
        """Set up default quality thresholds."""
        self.quality_metrics.set_quality_threshold(
            "retrieval_avg_relevance", 0.7, "lt", "warning", 
            "Average retrieval relevance below threshold"
        )
        
        self.quality_metrics.set_quality_threshold(
            "response_completeness", 0.6, "lt", "warning",
            "Response completeness below threshold"
        )
        
        self.quality_metrics.set_quality_threshold(
            "user_rating", 3.0, "lt", "error",
            "User rating below acceptable level"
        )
    
    def _start_monitoring_loop(self) -> None:
        """Start monitoring loop."""
        # This would typically run in a separate thread or async task
        pass
    
    def get_quality_alerts(self) -> List[Dict[str, Any]]:
        """Get current quality alerts."""
        return self.quality_metrics.check_quality_thresholds()
    
    def get_quality_trends(self, metric_name: str, time_window: timedelta) -> Dict[str, Any]:
        """Get quality trends for a specific metric."""
        try:
            if metric_name not in self.quality_metrics.metrics_history:
                return {}
            
            history = self.quality_metrics.metrics_history[metric_name]
            cutoff_time = datetime.utcnow() - time_window
            recent_metrics = [m for m in history if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return {}
            
            values = [m.value for m in recent_metrics]
            timestamps = [m.timestamp for m in recent_metrics]
            
            # Calculate trend
            if len(values) >= 2:
                trend = "increasing" if values[-1] > values[0] else "decreasing"
            else:
                trend = "stable"
            
            return {
                "metric_name": metric_name,
                "trend": trend,
                "values": values,
                "timestamps": timestamps,
                "time_window": time_window.total_seconds(),
                "latest_value": values[-1],
                "latest_timestamp": timestamps[-1]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get quality trends: {e}")
            return {}
