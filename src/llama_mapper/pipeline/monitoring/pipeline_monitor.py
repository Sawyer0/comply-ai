"""
Pipeline Monitoring and Observability

Comprehensive monitoring system for pipeline execution.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from ..exceptions import MonitoringError

logger = structlog.get_logger(__name__)


class PipelineMonitor:
    """
    Comprehensive monitoring system for pipeline execution.
    
    Provides:
    - Execution metrics collection
    - Performance monitoring
    - Error tracking
    - Resource utilization monitoring
    """
    
    def __init__(self):
        self.logger = logger.bind(component="monitor")
        
        # Metrics storage (in production, this would be a time-series database)
        self._pipeline_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._stage_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._execution_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self._active_executions: Dict[str, Dict[str, Any]] = {}
        self._performance_baselines: Dict[str, Dict[str, float]] = {}
    
    async def record_pipeline_start(self, pipeline_name: str, pipeline_id: str) -> None:
        """Record pipeline execution start."""
        timestamp = datetime.now()
        
        execution_record = {
            "pipeline_name": pipeline_name,
            "pipeline_id": pipeline_id,
            "status": "running",
            "started_at": timestamp.isoformat(),
            "stages_completed": 0,
            "stages_failed": 0
        }
        
        self._active_executions[pipeline_id] = execution_record
        
        self.logger.info("Pipeline execution started",
                        pipeline_name=pipeline_name,
                        pipeline_id=pipeline_id)
    
    async def record_pipeline_completion(self, 
                                        pipeline_name: str, 
                                        pipeline_id: str,
                                        success: bool) -> None:
        """Record pipeline execution completion."""
        timestamp = datetime.now()
        
        if pipeline_id not in self._active_executions:
            self.logger.warning("Pipeline completion recorded for unknown execution",
                              pipeline_id=pipeline_id)
            return
        
        execution_record = self._active_executions[pipeline_id]
        execution_record.update({
            "status": "completed" if success else "failed",
            "completed_at": timestamp.isoformat(),
            "success": success,
            "duration_seconds": (timestamp - datetime.fromisoformat(
                execution_record["started_at"]
            )).total_seconds()
        })
        
        # Move to history
        self._execution_history.append(execution_record)
        del self._active_executions[pipeline_id]
        
        # Update pipeline metrics
        self._pipeline_metrics[pipeline_name].append({
            "timestamp": timestamp.isoformat(),
            "duration_seconds": execution_record["duration_seconds"],
            "success": success,
            "stages_completed": execution_record["stages_completed"],
            "stages_failed": execution_record["stages_failed"]
        })
        
        self.logger.info("Pipeline execution completed",
                        pipeline_name=pipeline_name,
                        pipeline_id=pipeline_id,
                        success=success,
                        duration_seconds=execution_record["duration_seconds"])
    
    async def record_stage_completion(self, 
                                     stage_name: str,
                                     duration_seconds: float,
                                     success: bool) -> None:
        """Record stage execution completion."""
        timestamp = datetime.now()
        
        stage_record = {
            "timestamp": timestamp.isoformat(),
            "stage_name": stage_name,
            "duration_seconds": duration_seconds,
            "success": success
        }
        
        self._stage_metrics[stage_name].append(stage_record)
        
        # Update active pipeline execution
        for execution in self._active_executions.values():
            if success:
                execution["stages_completed"] += 1
            else:
                execution["stages_failed"] += 1
        
        self.logger.debug("Stage execution recorded",
                         stage_name=stage_name,
                         duration_seconds=duration_seconds,
                         success=success)
    
    async def get_pipeline_metrics(self, pipeline_name: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a pipeline."""
        metrics = self._pipeline_metrics.get(pipeline_name, [])
        
        if not metrics:
            return {
                "pipeline_name": pipeline_name,
                "total_executions": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
                "last_execution": None
            }
        
        successful_executions = [m for m in metrics if m["success"]]
        total_executions = len(metrics)
        success_rate = len(successful_executions) / total_executions if total_executions > 0 else 0.0
        
        durations = [m["duration_seconds"] for m in metrics]
        average_duration = sum(durations) / len(durations) if durations else 0.0
        
        # Calculate percentiles
        sorted_durations = sorted(durations)
        p50_duration = self._calculate_percentile(sorted_durations, 50)
        p95_duration = self._calculate_percentile(sorted_durations, 95)
        p99_duration = self._calculate_percentile(sorted_durations, 99)
        
        return {
            "pipeline_name": pipeline_name,
            "total_executions": total_executions,
            "successful_executions": len(successful_executions),
            "failed_executions": total_executions - len(successful_executions),
            "success_rate": success_rate,
            "average_duration": average_duration,
            "p50_duration": p50_duration,
            "p95_duration": p95_duration,
            "p99_duration": p99_duration,
            "last_execution": metrics[-1]["timestamp"] if metrics else None,
            "recent_trend": self._calculate_recent_trend(pipeline_name)
        }
    
    async def get_stage_metrics(self, stage_name: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a stage."""
        metrics = self._stage_metrics.get(stage_name, [])
        
        if not metrics:
            return {
                "stage_name": stage_name,
                "total_executions": 0,
                "success_rate": 0.0,
                "average_duration": 0.0
            }
        
        successful_executions = [m for m in metrics if m["success"]]
        total_executions = len(metrics)
        success_rate = len(successful_executions) / total_executions if total_executions > 0 else 0.0
        
        durations = [m["duration_seconds"] for m in metrics]
        average_duration = sum(durations) / len(durations) if durations else 0.0
        
        return {
            "stage_name": stage_name,
            "total_executions": total_executions,
            "successful_executions": len(successful_executions),
            "failed_executions": total_executions - len(successful_executions),
            "success_rate": success_rate,
            "average_duration": average_duration,
            "p95_duration": self._calculate_percentile(sorted(durations), 95),
            "last_execution": metrics[-1]["timestamp"] if metrics else None
        }
    
    def _calculate_percentile(self, sorted_values: List[float], percentile: int) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        
        if lower_index == upper_index:
            return sorted_values[lower_index]
        
        # Linear interpolation
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight
    
    def _calculate_recent_trend(self, pipeline_name: str) -> str:
        """Calculate recent performance trend for pipeline."""
        metrics = self._pipeline_metrics.get(pipeline_name, [])
        
        if len(metrics) < 5:
            return "insufficient_data"
        
        # Compare last 5 executions with previous 5
        recent_metrics = metrics[-5:]
        previous_metrics = metrics[-10:-5] if len(metrics) >= 10 else metrics[:-5]
        
        recent_success_rate = sum(1 for m in recent_metrics if m["success"]) / len(recent_metrics)
        previous_success_rate = sum(1 for m in previous_metrics if m["success"]) / len(previous_metrics)
        
        recent_avg_duration = sum(m["duration_seconds"] for m in recent_metrics) / len(recent_metrics)
        previous_avg_duration = sum(m["duration_seconds"] for m in previous_metrics) / len(previous_metrics)
        
        # Determine trend
        success_improving = recent_success_rate > previous_success_rate + 0.1
        success_degrading = recent_success_rate < previous_success_rate - 0.1
        
        duration_improving = recent_avg_duration < previous_avg_duration * 0.9
        duration_degrading = recent_avg_duration > previous_avg_duration * 1.1
        
        if success_improving and duration_improving:
            return "improving"
        elif success_degrading or duration_degrading:
            return "degrading"
        else:
            return "stable"
    
    async def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get currently active pipeline executions."""
        return list(self._active_executions.values())
    
    async def get_execution_history(self, 
                                   pipeline_name: Optional[str] = None,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history with optional filtering."""
        history = self._execution_history
        
        if pipeline_name:
            history = [h for h in history if h["pipeline_name"] == pipeline_name]
        
        # Sort by completion time (most recent first)
        history = sorted(history, key=lambda x: x.get("completed_at", x["started_at"]), reverse=True)
        
        return history[:limit]
    
    async def detect_anomalies(self, pipeline_name: str) -> List[Dict[str, Any]]:
        """Detect performance anomalies in pipeline execution."""
        anomalies = []
        metrics = self._pipeline_metrics.get(pipeline_name, [])
        
        if len(metrics) < 10:
            return anomalies  # Need sufficient data for anomaly detection
        
        # Calculate baseline performance
        recent_metrics = metrics[-20:]  # Last 20 executions
        durations = [m["duration_seconds"] for m in recent_metrics]
        success_rates = [1.0 if m["success"] else 0.0 for m in recent_metrics]
        
        avg_duration = sum(durations) / len(durations)
        avg_success_rate = sum(success_rates) / len(success_rates)
        
        # Check for anomalies in recent executions
        for metric in metrics[-5:]:  # Check last 5 executions
            # Duration anomaly (>2x average)
            if metric["duration_seconds"] > avg_duration * 2:
                anomalies.append({
                    "type": "duration_anomaly",
                    "timestamp": metric["timestamp"],
                    "value": metric["duration_seconds"],
                    "baseline": avg_duration,
                    "severity": "high" if metric["duration_seconds"] > avg_duration * 3 else "medium"
                })
            
            # Success rate anomaly
            if not metric["success"] and avg_success_rate > 0.8:
                anomalies.append({
                    "type": "failure_anomaly",
                    "timestamp": metric["timestamp"],
                    "baseline_success_rate": avg_success_rate,
                    "severity": "high"
                })
        
        return anomalies
    
    async def generate_performance_report(self, 
                                         pipeline_name: str,
                                         days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        metrics = self._pipeline_metrics.get(pipeline_name, [])
        recent_metrics = [
            m for m in metrics 
            if datetime.fromisoformat(m["timestamp"]) >= cutoff_date
        ]
        
        if not recent_metrics:
            return {
                "pipeline_name": pipeline_name,
                "period_days": days,
                "no_data": True
            }
        
        # Calculate comprehensive statistics
        total_executions = len(recent_metrics)
        successful_executions = [m for m in recent_metrics if m["success"]]
        failed_executions = [m for m in recent_metrics if not m["success"]]
        
        durations = [m["duration_seconds"] for m in recent_metrics]
        success_durations = [m["duration_seconds"] for m in successful_executions]
        
        report = {
            "pipeline_name": pipeline_name,
            "period_days": days,
            "report_generated_at": datetime.now().isoformat(),
            
            # Execution statistics
            "total_executions": total_executions,
            "successful_executions": len(successful_executions),
            "failed_executions": len(failed_executions),
            "success_rate": len(successful_executions) / total_executions if total_executions > 0 else 0.0,
            
            # Duration statistics
            "average_duration": sum(durations) / len(durations) if durations else 0.0,
            "median_duration": self._calculate_percentile(sorted(durations), 50),
            "p95_duration": self._calculate_percentile(sorted(durations), 95),
            "p99_duration": self._calculate_percentile(sorted(durations), 99),
            
            # Success-only duration statistics
            "average_success_duration": sum(success_durations) / len(success_durations) if success_durations else 0.0,
            
            # Trend analysis
            "trend": self._calculate_recent_trend(pipeline_name),
            "anomalies": await self.detect_anomalies(pipeline_name),
            
            # Daily breakdown
            "daily_stats": self._calculate_daily_stats(recent_metrics)
        }
        
        return report
    
    def _calculate_daily_stats(self, metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate daily statistics from metrics."""
        daily_stats = defaultdict(lambda: {"executions": 0, "successes": 0, "total_duration": 0.0})
        
        for metric in metrics:
            date = datetime.fromisoformat(metric["timestamp"]).date().isoformat()
            daily_stats[date]["executions"] += 1
            if metric["success"]:
                daily_stats[date]["successes"] += 1
            daily_stats[date]["total_duration"] += metric["duration_seconds"]
        
        # Convert to list format
        result = []
        for date, stats in sorted(daily_stats.items()):
            result.append({
                "date": date,
                "executions": stats["executions"],
                "successes": stats["successes"],
                "failures": stats["executions"] - stats["successes"],
                "success_rate": stats["successes"] / stats["executions"] if stats["executions"] > 0 else 0.0,
                "average_duration": stats["total_duration"] / stats["executions"] if stats["executions"] > 0 else 0.0
            })
        
        return result
    
    async def set_performance_baseline(self, 
                                      pipeline_name: str,
                                      baseline_metrics: Dict[str, float]) -> None:
        """Set performance baseline for a pipeline."""
        self._performance_baselines[pipeline_name] = baseline_metrics
        
        self.logger.info("Performance baseline set",
                        pipeline_name=pipeline_name,
                        baseline_metrics=baseline_metrics)
    
    async def check_performance_against_baseline(self, 
                                                pipeline_name: str) -> Dict[str, Any]:
        """Check current performance against baseline."""
        baseline = self._performance_baselines.get(pipeline_name)
        if not baseline:
            return {"error": "No baseline set for pipeline"}
        
        current_metrics = await self.get_pipeline_metrics(pipeline_name)
        
        comparison = {
            "pipeline_name": pipeline_name,
            "baseline": baseline,
            "current": {
                "success_rate": current_metrics["success_rate"],
                "average_duration": current_metrics["average_duration"],
                "p95_duration": current_metrics["p95_duration"]
            },
            "comparison": {}
        }
        
        # Compare key metrics
        for metric in ["success_rate", "average_duration", "p95_duration"]:
            if metric in baseline:
                current_value = comparison["current"][metric]
                baseline_value = baseline[metric]
                
                if metric == "success_rate":
                    # Higher is better for success rate
                    improvement = current_value - baseline_value
                    status = "improved" if improvement > 0.05 else "degraded" if improvement < -0.05 else "stable"
                else:
                    # Lower is better for duration metrics
                    improvement = baseline_value - current_value
                    status = "improved" if improvement > baseline_value * 0.1 else "degraded" if improvement < -baseline_value * 0.1 else "stable"
                
                comparison["comparison"][metric] = {
                    "improvement": improvement,
                    "improvement_percentage": (improvement / baseline_value * 100) if baseline_value > 0 else 0,
                    "status": status
                }
        
        return comparison