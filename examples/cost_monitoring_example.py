#!/usr/bin/env python3
"""Example of using the cost monitoring and autoscaling system."""

import asyncio
import json
from datetime import datetime, timezone, timedelta

from src.llama_mapper.cost_monitoring import (
    CostMonitoringSystem,
    CostMonitoringFactory,
    CostGuardrail,
    GuardrailAction,
    GuardrailSeverity,
    ScalingPolicy,
    ScalingTrigger,
    ResourceType,
)


async def main():
    """Main example function."""
    print("Cost Monitoring and Autoscaling Example")
    print("=" * 45)
    
    # Create cost monitoring system with production configuration
    config = CostMonitoringFactory.create_production_config()
    cost_system = CostMonitoringSystem(config)
    
    try:
        # Start the system
        print("Starting cost monitoring system...")
        await cost_system.start()
        print("✓ System started successfully")
        
        # Add some guardrails
        print("\nAdding cost guardrails...")
        
        # Daily budget guardrail
        daily_guardrail = CostGuardrail(
            guardrail_id="daily_budget",
            name="Daily Budget Limit",
            description="Prevent daily costs from exceeding budget",
            metric_type="daily_cost",
            threshold=config.daily_budget_limit,
            severity=GuardrailSeverity.HIGH,
            actions=[GuardrailAction.ALERT, GuardrailAction.NOTIFY_ADMIN],
            cooldown_minutes=60,
        )
        cost_system.add_guardrail(daily_guardrail)
        print("✓ Added daily budget guardrail")
        
        # API call cost guardrail
        api_guardrail = CostGuardrail(
            guardrail_id="api_cost",
            name="API Call Cost Limit",
            description="Prevent excessive API call costs",
            metric_type="api_calls",
            threshold=10000,  # 10k API calls
            severity=GuardrailSeverity.MEDIUM,
            actions=[GuardrailAction.ALERT, GuardrailAction.THROTTLE],
            cooldown_minutes=30,
        )
        cost_system.add_guardrail(api_guardrail)
        print("✓ Added API cost guardrail")
        
        # Add scaling policies
        print("\nAdding scaling policies...")
        
        # CPU scaling policy
        cpu_policy = ScalingPolicy(
            policy_id="cpu_scaling",
            name="CPU-based Scaling",
            description="Scale based on CPU usage and cost",
            resource_type=ResourceType.CPU,
            trigger=ScalingTrigger.COST_THRESHOLD,
            threshold=0.8,  # 80% of cost threshold
            min_instances=1,
            max_instances=10,
            scale_up_cooldown_minutes=5,
            scale_down_cooldown_minutes=15,
            cost_weight=0.6,
            performance_weight=0.4,
        )
        cost_system.add_scaling_policy(cpu_policy)
        print("✓ Added CPU scaling policy")
        
        # GPU scaling policy
        gpu_policy = ScalingPolicy(
            policy_id="gpu_scaling",
            name="GPU-based Scaling",
            description="Scale GPU resources based on demand and cost",
            resource_type=ResourceType.GPU,
            trigger=ScalingTrigger.PERFORMANCE_DEGRADATION,
            threshold=0.7,  # 70% performance threshold
            min_instances=0,
            max_instances=5,
            scale_up_cooldown_minutes=10,
            scale_down_cooldown_minutes=30,
            cost_weight=0.7,
            performance_weight=0.3,
        )
        cost_system.add_scaling_policy(gpu_policy)
        print("✓ Added GPU scaling policy")
        
        # Wait for some data collection
        print("\nCollecting data for 30 seconds...")
        await asyncio.sleep(30)
        
        # Get system status
        print("\nSystem Status:")
        status = cost_system.get_system_status()
        print(f"  Running: {status['running']}")
        print(f"  Uptime: {status['uptime_seconds']:.0f} seconds")
        
        # Get cost breakdown
        print("\nCost Breakdown (Last 7 days):")
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=7)
        breakdown = cost_system.get_cost_breakdown(start_time, end_time)
        print(f"  Total Cost: ${breakdown.total_cost:.2f}")
        print(f"  Compute Cost: ${breakdown.compute_cost:.2f}")
        print(f"  Memory Cost: ${breakdown.memory_cost:.2f}")
        print(f"  Storage Cost: ${breakdown.storage_cost:.2f}")
        print(f"  Network Cost: ${breakdown.network_cost:.2f}")
        print(f"  API Cost: ${breakdown.api_cost:.2f}")
        
        # Get cost trends
        print("\nCost Trends (Last 30 days):")
        trends = cost_system.get_cost_trends(30)
        print(f"  Total Cost: ${trends['total_cost']:.2f}")
        print(f"  Average Daily Cost: ${trends['average_daily_cost']:.2f}")
        print(f"  Data Points: {len(trends['costs'])}")
        
        # Get optimization recommendations
        print("\nOptimization Recommendations:")
        recommendations = cost_system.get_optimization_recommendations()
        if recommendations:
            total_savings = sum(rec.potential_savings for rec in recommendations)
            print(f"  Total Potential Savings: ${total_savings:.2f}")
            print(f"  Number of Recommendations: {len(recommendations)}")
            
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                print(f"  {i}. {rec.title}")
                print(f"     Category: {rec.category}")
                print(f"     Priority: {rec.priority}/10")
                print(f"     Potential Savings: ${rec.potential_savings:.2f}")
                print(f"     Confidence: {rec.confidence:.1%}")
        else:
            print("  No recommendations available")
        
        # Get cost anomalies
        print("\nCost Anomalies (Last 30 days):")
        anomalies = cost_system.get_cost_anomalies(days=30)
        if anomalies:
            print(f"  Total Anomalies: {len(anomalies)}")
            severity_counts = {}
            for anomaly in anomalies:
                severity_counts[anomaly.severity] = severity_counts.get(anomaly.severity, 0) + 1
            
            for severity, count in severity_counts.items():
                print(f"  {severity}: {count}")
        else:
            print("  No anomalies detected")
        
        # Get latest forecast
        print("\nCost Forecast:")
        forecast = cost_system.get_latest_forecast()
        if forecast:
            print(f"  Predicted Cost: ${forecast.predicted_cost:.2f}")
            print(f"  Confidence Level: {forecast.confidence_level:.1%}")
            print(f"  Confidence Interval: ${forecast.confidence_interval_lower:.2f} - ${forecast.confidence_interval_upper:.2f}")
            print(f"  Model Type: {forecast.model_type}")
        else:
            print("  No forecast available")
        
        # Get guardrail violations
        print("\nGuardrail Violations:")
        violations = cost_system.get_guardrail_violations()
        if violations:
            print(f"  Total Violations: {len(violations)}")
            for violation in violations[:3]:  # Show recent 3
                print(f"  • {violation.guardrail_id}: {violation.severity} - {violation.message}")
        else:
            print("  No violations detected")
        
        # Get scaling decisions
        print("\nScaling Decisions:")
        decisions = cost_system.get_scaling_decisions()
        if decisions:
            print(f"  Total Decisions: {len(decisions)}")
            for decision in decisions[:3]:  # Show recent 3
                print(f"  • {decision.action}: {decision.resource_type} "
                      f"({decision.current_instances} → {decision.target_instances})")
                print(f"    Reason: {decision.reason}")
        else:
            print("  No scaling decisions made")
        
        # Get comprehensive analytics summary
        print("\nAnalytics Summary:")
        summary = cost_system.get_analytics_summary(30)
        print(f"  Cost Trend Growth Rate: {summary['cost_trend']['cost_growth_rate']:.1f}%")
        print(f"  Total Recommendations: {summary['recommendations']['total']}")
        print(f"  Total Anomalies: {summary['anomalies']['total']}")
        print(f"  Critical Anomalies: {summary['anomalies']['critical']}")
        
        # Health check
        print("\nHealth Check:")
        health = await cost_system.health_check()
        print(f"  Overall Health: {health['overall_health']}")
        for component, info in health['components'].items():
            status_icon = "✓" if info['status'] == 'healthy' else "✗"
            print(f"  {status_icon} {component}: {info['status']}")
        
        print("\n✓ Example completed successfully!")
        
    except Exception as e:
        print(f"✗ Example failed: {e}")
        raise
    
    finally:
        # Stop the system
        print("\nStopping cost monitoring system...")
        await cost_system.stop()
        print("✓ System stopped successfully")


if __name__ == "__main__":
    asyncio.run(main())
