#!/usr/bin/env python3
"""
Runbook Executor for Analysis Module

This script provides a command-line interface for executing operational runbooks
and automated incident response procedures.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.llama_mapper.analysis.monitoring.metrics_collector import AnalysisMetricsCollector
from src.llama_mapper.analysis.infrastructure.model_server import Phi3AnalysisModelServer
from src.llama_mapper.analysis.infrastructure.validator import AnalysisValidator
from src.llama_mapper.analysis.infrastructure.opa_generator import OPAPolicyGenerator
from src.llama_mapper.analysis.quality.quality_alerting_system import QualityAlertingSystem
from src.llama_mapper.cost_monitoring import CostMonitoringSystem
from src.llama_mapper.analysis.infrastructure.auth import APIKeyManager
from src.llama_mapper.analysis.security.waf.engine.rule_engine import WAFRuleEngine
from src.llama_mapper.analysis.resilience.circuit_breaker.implementation import CircuitBreaker
from src.llama_mapper.analysis.domain.services import WeeklyEvaluationService


class RunbookExecutor:
    """Executes operational runbooks for the analysis module."""
    
    def __init__(self):
        self.metrics_collector = AnalysisMetricsCollector("analysis-module")
        self.model_server = Phi3AnalysisModelServer("models/phi3-mini-3.8b")
        self.validator = AnalysisValidator()
        self.opa_generator = OPAPolicyGenerator()
        self.quality_system = QualityAlertingSystem()
        self.cost_system = CostMonitoringSystem()
        self.api_key_manager = APIKeyManager()
        self.waf_engine = WAFRuleEngine()
        self.evaluation_service = WeeklyEvaluationService()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components."""
        try:
            # Initialize cost monitoring system
            self.cost_system.start()
        except Exception as e:
            print(f"Warning: Could not initialize cost monitoring system: {e}")
    
    async def check_schema_validation_rate(self) -> Dict[str, Any]:
        """Check current schema validation rate."""
        try:
            rate = self.metrics_collector.get_metric("schema_valid_rate")
            return {
                "rate": rate,
                "threshold_warning": 0.95,
                "threshold_critical": 0.90,
                "status": "critical" if rate < 0.90 else "warning" if rate < 0.95 else "healthy"
            }
        except Exception as e:
            return {"error": str(e), "status": "unknown"}
    
    async def check_template_fallback_rate(self) -> Dict[str, Any]:
        """Check current template fallback rate."""
        try:
            rate = self.metrics_collector.get_metric("template_fallback_rate")
            return {
                "rate": rate,
                "threshold_warning": 0.20,
                "threshold_critical": 0.30,
                "status": "critical" if rate > 0.30 else "warning" if rate > 0.20 else "healthy"
            }
        except Exception as e:
            return {"error": str(e), "status": "unknown"}
    
    async def check_opa_compilation_rate(self) -> Dict[str, Any]:
        """Check current OPA compilation success rate."""
        try:
            rate = self.metrics_collector.get_metric("opa_compile_success_rate")
            return {
                "rate": rate,
                "threshold_warning": 0.95,
                "threshold_critical": 0.90,
                "status": "critical" if rate < 0.90 else "warning" if rate < 0.95 else "healthy"
            }
        except Exception as e:
            return {"error": str(e), "status": "unknown"}
    
    async def check_model_server_health(self) -> Dict[str, Any]:
        """Check model server health status."""
        try:
            health = self.model_server.get_health_status()
            return {
                "model_loaded": health.model_loaded,
                "memory_usage_mb": health.memory_usage_mb,
                "last_inference_time_ms": health.last_inference_time_ms,
                "error_rate": health.error_rate,
                "status": "healthy" if health.error_rate < 0.05 else "warning" if health.error_rate < 0.10 else "critical"
            }
        except Exception as e:
            return {"error": str(e), "status": "unknown"}
    
    async def check_quality_score(self) -> Dict[str, Any]:
        """Check current quality score."""
        try:
            score = self.metrics_collector.get_metric("quality_score")
            return {
                "score": score,
                "threshold_warning": 0.85,
                "threshold_critical": 0.80,
                "status": "critical" if score < 0.80 else "warning" if score < 0.85 else "healthy"
            }
        except Exception as e:
            return {"error": str(e), "status": "unknown"}
    
    async def check_cost_metrics(self) -> Dict[str, Any]:
        """Check current cost metrics."""
        try:
            costs = self.cost_system.get_current_costs()
            return {
                "daily_cost": costs.daily_cost,
                "daily_budget": costs.daily_budget,
                "usage_percentage": costs.usage_percentage,
                "status": "critical" if costs.usage_percentage > 100 else "warning" if costs.usage_percentage > 80 else "healthy"
            }
        except Exception as e:
            return {"error": str(e), "status": "unknown"}
    
    async def get_recent_validation_failures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent validation failures."""
        try:
            failures = self.validator.get_recent_validation_failures(limit=limit)
            return [
                {
                    "request_id": failure.request_id,
                    "error_message": failure.error_message,
                    "timestamp": failure.timestamp.isoformat(),
                    "input_snippet": failure.input_snippet[:100] + "..." if len(failure.input_snippet) > 100 else failure.input_snippet
                }
                for failure in failures
            ]
        except Exception as e:
            return [{"error": str(e)}]
    
    async def get_model_statistics(self) -> Dict[str, Any]:
        """Get model server statistics."""
        try:
            stats = self.model_server.get_model_statistics()
            return {
                "avg_confidence": stats.avg_confidence,
                "low_confidence_count": stats.low_confidence_count,
                "model_load_time": stats.model_load_time,
                "total_requests": stats.total_requests,
                "successful_requests": stats.successful_requests
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def get_opa_compilation_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent OPA compilation errors."""
        try:
            errors = self.opa_generator.get_recent_compilation_errors(limit=limit)
            return [
                {
                    "error_message": error.error_message,
                    "policy_snippet": error.policy_snippet[:100] + "..." if len(error.policy_snippet) > 100 else error.policy_snippet,
                    "timestamp": error.timestamp.isoformat()
                }
                for error in errors
            ]
        except Exception as e:
            return [{"error": str(e)}]
    
    async def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get cost breakdown by resource type."""
        try:
            breakdown = self.cost_system.get_cost_breakdown()
            return breakdown
        except Exception as e:
            return {"error": str(e)}
    
    async def get_cost_anomalies(self) -> List[Dict[str, Any]]:
        """Get detected cost anomalies."""
        try:
            anomalies = self.cost_system.detect_cost_anomalies()
            return [
                {
                    "description": anomaly.description,
                    "severity": anomaly.severity,
                    "cost_impact": anomaly.cost_impact,
                    "recommendations": anomaly.recommendations
                }
                for anomaly in anomalies
            ]
        except Exception as e:
            return [{"error": str(e)}]
    
    async def get_waf_statistics(self) -> Dict[str, Any]:
        """Get WAF statistics."""
        try:
            return {
                "total_rules": len(self.waf_engine.rules),
                "blocked_ips": len(self.waf_engine.blocked_ips),
                "suspicious_ips": len(self.waf_engine.suspicious_ips)
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        try:
            cb = CircuitBreaker("model_server")
            return {
                "state": cb.state.value,
                "failure_count": cb._failure_count,
                "success_count": cb._success_count,
                "last_failure_time": cb._last_failure_time.isoformat() if cb._last_failure_time else None
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def get_evaluation_status(self) -> Dict[str, Any]:
        """Get weekly evaluation status."""
        try:
            status = self.evaluation_service.get_system_status()
            return status
        except Exception as e:
            return {"error": str(e)}
    
    async def execute_health_check(self) -> Dict[str, Any]:
        """Execute comprehensive health check."""
        print("🔍 Executing comprehensive health check...")
        
        results = {}
        
        # Check all key metrics
        checks = [
            ("schema_validation", self.check_schema_validation_rate()),
            ("template_fallback", self.check_template_fallback_rate()),
            ("opa_compilation", self.check_opa_compilation_rate()),
            ("model_server", self.check_model_server_health()),
            ("quality_score", self.check_quality_score()),
            ("cost_metrics", self.check_cost_metrics()),
            ("waf_statistics", self.get_waf_statistics()),
            ("circuit_breaker", self.get_circuit_breaker_status()),
            ("evaluation_status", self.get_evaluation_status())
        ]
        
        for name, check_coro in checks:
            try:
                result = await check_coro
                results[name] = result
                status = result.get("status", "unknown")
                print(f"  {name}: {status}")
            except Exception as e:
                results[name] = {"error": str(e), "status": "error"}
                print(f"  {name}: error - {e}")
        
        return results
    
    async def execute_schema_validation_investigation(self) -> Dict[str, Any]:
        """Execute schema validation drop investigation."""
        print("🔍 Investigating schema validation drops...")
        
        results = {}
        
        # Check validation rate
        validation_rate = await self.check_schema_validation_rate()
        results["validation_rate"] = validation_rate
        
        if validation_rate.get("status") in ["warning", "critical"]:
            # Get recent failures
            failures = await self.get_recent_validation_failures(limit=10)
            results["recent_failures"] = failures
            
            # Get model statistics
            model_stats = await self.get_model_statistics()
            results["model_statistics"] = model_stats
            
            print(f"  Validation rate: {validation_rate.get('rate', 'unknown'):.2%}")
            print(f"  Recent failures: {len(failures)}")
            print(f"  Model avg confidence: {model_stats.get('avg_confidence', 'unknown')}")
        
        return results
    
    async def execute_template_fallback_investigation(self) -> Dict[str, Any]:
        """Execute template fallback spike investigation."""
        print("🔍 Investigating template fallback spikes...")
        
        results = {}
        
        # Check fallback rate
        fallback_rate = await self.check_template_fallback_rate()
        results["fallback_rate"] = fallback_rate
        
        if fallback_rate.get("status") in ["warning", "critical"]:
            # Get model health
            model_health = await self.check_model_server_health()
            results["model_health"] = model_health
            
            # Get model statistics
            model_stats = await self.get_model_statistics()
            results["model_statistics"] = model_stats
            
            print(f"  Fallback rate: {fallback_rate.get('rate', 'unknown'):.2%}")
            print(f"  Model loaded: {model_health.get('model_loaded', 'unknown')}")
            print(f"  Model error rate: {model_health.get('error_rate', 'unknown'):.2%}")
        
        return results
    
    async def execute_opa_compilation_investigation(self) -> Dict[str, Any]:
        """Execute OPA compilation failure investigation."""
        print("🔍 Investigating OPA compilation failures...")
        
        results = {}
        
        # Check compilation rate
        compilation_rate = await self.check_opa_compilation_rate()
        results["compilation_rate"] = compilation_rate
        
        if compilation_rate.get("status") in ["warning", "critical"]:
            # Get compilation errors
            errors = await self.get_opa_compilation_errors(limit=5)
            results["compilation_errors"] = errors
            
            print(f"  Compilation rate: {compilation_rate.get('rate', 'unknown'):.2%}")
            print(f"  Recent errors: {len(errors)}")
        
        return results
    
    async def execute_cost_anomaly_investigation(self) -> Dict[str, Any]:
        """Execute cost anomaly investigation."""
        print("🔍 Investigating cost anomalies...")
        
        results = {}
        
        # Check cost metrics
        cost_metrics = await self.check_cost_metrics()
        results["cost_metrics"] = cost_metrics
        
        if cost_metrics.get("status") in ["warning", "critical"]:
            # Get cost breakdown
            breakdown = await self.get_cost_breakdown()
            results["cost_breakdown"] = breakdown
            
            # Get anomalies
            anomalies = await self.get_cost_anomalies()
            results["anomalies"] = anomalies
            
            print(f"  Daily cost: ${cost_metrics.get('daily_cost', 'unknown'):.2f}")
            print(f"  Budget usage: {cost_metrics.get('usage_percentage', 'unknown'):.1f}%")
            print(f"  Anomalies detected: {len(anomalies)}")
        
        return results
    
    def generate_incident_report(self, investigation_type: str, results: Dict[str, Any]) -> str:
        """Generate incident report from investigation results."""
        timestamp = datetime.utcnow().isoformat()
        
        report = f"""# {investigation_type.replace('_', ' ').title()} Investigation Report

**Date**: {timestamp}
**Investigation Type**: {investigation_type}

## Summary
"""
        
        # Add summary based on investigation type
        if investigation_type == "schema_validation":
            rate = results.get("validation_rate", {}).get("rate", "unknown")
            status = results.get("validation_rate", {}).get("status", "unknown")
            report += f"- Schema validation rate: {rate:.2%} (status: {status})\n"
            report += f"- Recent failures: {len(results.get('recent_failures', []))}\n"
            
        elif investigation_type == "template_fallback":
            rate = results.get("fallback_rate", {}).get("rate", "unknown")
            status = results.get("fallback_rate", {}).get("status", "unknown")
            report += f"- Template fallback rate: {rate:.2%} (status: {status})\n"
            report += f"- Model loaded: {results.get('model_health', {}).get('model_loaded', 'unknown')}\n"
            
        elif investigation_type == "opa_compilation":
            rate = results.get("compilation_rate", {}).get("rate", "unknown")
            status = results.get("compilation_rate", {}).get("status", "unknown")
            report += f"- OPA compilation rate: {rate:.2%} (status: {status})\n"
            report += f"- Recent errors: {len(results.get('compilation_errors', []))}\n"
            
        elif investigation_type == "cost_anomaly":
            daily_cost = results.get("cost_metrics", {}).get("daily_cost", "unknown")
            usage_pct = results.get("cost_metrics", {}).get("usage_percentage", "unknown")
            report += f"- Daily cost: ${daily_cost:.2f}\n"
            report += f"- Budget usage: {usage_pct:.1f}%\n"
            report += f"- Anomalies detected: {len(results.get('anomalies', []))}\n"
        
        report += f"""
## Detailed Results
```json
{json.dumps(results, indent=2, default=str)}
```

## Recommendations
- Review the detailed results above
- Follow the appropriate runbook procedures
- Monitor recovery after applying fixes
- Document any additional actions taken

## Next Steps
1. Apply appropriate fixes based on root cause
2. Monitor metrics for recovery
3. Update documentation if needed
4. Schedule post-incident review
"""
        
        return report


async def main():
    """Main entry point for the runbook executor."""
    parser = argparse.ArgumentParser(description="Analysis Module Runbook Executor")
    parser.add_argument("command", choices=[
        "health-check",
        "investigate-schema-validation",
        "investigate-template-fallback", 
        "investigate-opa-compilation",
        "investigate-cost-anomaly",
        "check-metrics"
    ], help="Command to execute")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    executor = RunbookExecutor()
    
    try:
        if args.command == "health-check":
            results = await executor.execute_health_check()
        elif args.command == "investigate-schema-validation":
            results = await executor.execute_schema_validation_investigation()
        elif args.command == "investigate-template-fallback":
            results = await executor.execute_template_fallback_investigation()
        elif args.command == "investigate-opa-compilation":
            results = await executor.execute_opa_compilation_investigation()
        elif args.command == "investigate-cost-anomaly":
            results = await executor.execute_cost_anomaly_investigation()
        elif args.command == "check-metrics":
            results = await executor.execute_health_check()
        else:
            print(f"Unknown command: {args.command}")
            return 1
        
        # Output results
        if args.format == "json":
            output = json.dumps(results, indent=2, default=str)
        else:
            # Generate text report
            if args.command.startswith("investigate-"):
                investigation_type = args.command.replace("investigate-", "")
                output = executor.generate_incident_report(investigation_type, results)
            else:
                output = f"Health Check Results:\n{json.dumps(results, indent=2, default=str)}"
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Results written to {args.output}")
        else:
            print(output)
        
        return 0
        
    except Exception as e:
        print(f"Error executing runbook: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
