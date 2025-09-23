# Cost Monitoring and Autoscaling Guide

## Overview

The Llama Mapper cost monitoring and autoscaling system provides comprehensive cost control, budget management, and intelligent resource scaling to optimize both performance and costs in production environments.

## Key Features

### 🎯 **Cost Monitoring**
- Real-time cost tracking across all resources (CPU, GPU, memory, storage, network, API calls)
- Historical cost analysis and trending
- Cost breakdown by component and tenant
- Automated cost anomaly detection

### 🛡️ **Cost Guardrails**
- Configurable spending limits and budget controls
- Automated alerts and actions when thresholds are exceeded
- Emergency stop capabilities to prevent runaway costs
- Multi-tenant cost isolation and controls

### ⚡ **Cost-Aware Autoscaling**
- Intelligent scaling decisions based on both cost and performance
- Predictive scaling to optimize resource utilization
- Configurable scaling policies for different resource types
- Cost-performance trade-off optimization

### 📊 **Analytics & Reporting**
- Cost optimization recommendations
- Anomaly detection and alerting
- Cost forecasting and trend analysis
- Comprehensive reporting and dashboards

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Cost Monitoring System                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Metrics   │  │ Guardrails  │  │Autoscaling  │        │
│  │  Collector  │  │   System    │  │   System    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          │                                 │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Analytics & Reporting                      │
│  └─────────────────────────────────────────────────────────┤
│                          │                                 │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Configuration & Management                 │
│  └─────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Basic Setup

```python
from src.llama_mapper.cost_monitoring import (
    CostMonitoringSystem,
    CostMonitoringFactory,
)

# Create system with default configuration
config = CostMonitoringFactory.create_default_config()
cost_system = CostMonitoringSystem(config)

# Start the system
await cost_system.start()
```

### 2. Add Cost Guardrails

```python
from src.llama_mapper.cost_monitoring import (
    CostGuardrail,
    GuardrailAction,
    GuardrailSeverity,
)

# Daily budget guardrail
daily_guardrail = CostGuardrail(
    guardrail_id="daily_budget",
    name="Daily Budget Limit",
    description="Prevent daily costs from exceeding budget",
    metric_type="daily_cost",
    threshold=1000.0,  # $1000 daily limit
    severity=GuardrailSeverity.HIGH,
    actions=[GuardrailAction.ALERT, GuardrailAction.NOTIFY_ADMIN],
    cooldown_minutes=60,
)

cost_system.add_guardrail(daily_guardrail)
```

### 3. Configure Autoscaling

```python
from src.llama_mapper.cost_monitoring import (
    ScalingPolicy,
    ScalingTrigger,
    ResourceType,
)

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
    cost_weight=0.6,
    performance_weight=0.4,
)

cost_system.add_scaling_policy(cpu_policy)
```

## Configuration Options

### Environment-Specific Configurations

```python
# Development environment
dev_config = CostMonitoringFactory.create_development_config()

# Production environment
prod_config = CostMonitoringFactory.create_production_config()

# High-performance environment
perf_config = CostMonitoringFactory.create_high_performance_config()

# Cost-optimized environment
cost_opt_config = CostMonitoringFactory.create_cost_optimized_config()
```

### Custom Configuration

```python
from src.llama_mapper.cost_monitoring import CostMonitoringSystemConfig

config = CostMonitoringSystemConfig(
    # Budget limits
    daily_budget_limit=1000.0,
    monthly_budget_limit=30000.0,
    emergency_stop_threshold=5000.0,
    
    # Component settings
    metrics_collector=CostMonitoringConfig(
        collection_interval_seconds=60,
        retention_days=90,
    ),
    
    guardrails=CostGuardrailsConfig(
        max_violations_per_hour=10,
        escalation_delay_minutes=30,
    ),
    
    autoscaling=CostAwareScalingConfig(
        evaluation_interval_seconds=60,
        cost_threshold_percent=80.0,
        max_cost_increase_percent=50.0,
    ),
)
```

## CLI Commands

### System Status

```bash
# Check system status
mapper cost status

# Get cost breakdown
mapper cost breakdown --days 7 --format text

# View cost trends
mapper cost trends --days 30

# Get optimization recommendations
mapper cost recommendations --priority-min 7

# Check for anomalies
mapper cost anomalies --severity high

# Get cost forecast
mapper cost forecast
```

### Advanced Usage

```bash
# Tenant-specific analysis
mapper cost breakdown --tenant-id tenant-123 --days 14

# Category-specific recommendations
mapper cost recommendations --category compute --format json

# Export data
mapper cost trends --days 90 --format json > cost_trends.json
```

## Cost Guardrails

### Guardrail Types

1. **Budget Limits**
   - Daily, hourly, monthly spending limits
   - Emergency stop thresholds
   - Per-tenant budget controls

2. **Resource Limits**
   - API call limits
   - Compute resource limits
   - Storage usage limits

3. **Performance-Based**
   - Cost per performance ratio
   - Efficiency thresholds
   - Resource utilization limits

### Guardrail Actions

- `ALERT`: Send notifications
- `THROTTLE`: Reduce request rates
- `SCALE_DOWN`: Reduce resource allocation
- `PAUSE_SERVICE`: Temporarily pause services
- `BLOCK_REQUESTS`: Block new requests
- `NOTIFY_ADMIN`: Alert administrators

### Example Guardrail Configuration

```python
# Multi-tier guardrail system
guardrails = [
    # Warning level
    CostGuardrail(
        guardrail_id="daily_warning",
        name="Daily Budget Warning",
        metric_type="daily_cost",
        threshold=800.0,  # 80% of budget
        severity=GuardrailSeverity.MEDIUM,
        actions=[GuardrailAction.ALERT],
    ),
    
    # Critical level
    CostGuardrail(
        guardrail_id="daily_critical",
        name="Daily Budget Critical",
        metric_type="daily_cost",
        threshold=1000.0,  # 100% of budget
        severity=GuardrailSeverity.HIGH,
        actions=[GuardrailAction.ALERT, GuardrailAction.THROTTLE],
    ),
    
    # Emergency level
    CostGuardrail(
        guardrail_id="daily_emergency",
        name="Daily Budget Emergency",
        metric_type="daily_cost",
        threshold=1200.0,  # 120% of budget
        severity=GuardrailSeverity.CRITICAL,
        actions=[GuardrailAction.PAUSE_SERVICE, GuardrailAction.NOTIFY_ADMIN],
    ),
]
```

## Autoscaling Policies

### Scaling Triggers

- `COST_THRESHOLD`: Scale based on cost limits
- `PERFORMANCE_DEGRADATION`: Scale based on performance metrics
- `PREDICTED_COST`: Scale based on cost predictions
- `SCHEDULED`: Scale based on time schedules
- `MANUAL`: Manual scaling triggers

### Resource Types

- `CPU`: CPU cores and utilization
- `MEMORY`: Memory allocation and usage
- `GPU`: GPU instances and memory
- `REPLICAS`: Service replica count
- `NODES`: Infrastructure nodes

### Scaling Actions

- `SCALE_UP`: Increase resource allocation
- `SCALE_DOWN`: Decrease resource allocation
- `SCALE_OUT`: Add more instances
- `SCALE_IN`: Remove instances
- `NO_ACTION`: No scaling required

### Example Scaling Policy

```python
# Cost-optimized GPU scaling
gpu_policy = ScalingPolicy(
    policy_id="gpu_cost_optimized",
    name="Cost-Optimized GPU Scaling",
    description="Scale GPU resources with cost optimization",
    resource_type=ResourceType.GPU,
    trigger=ScalingTrigger.COST_THRESHOLD,
    threshold=0.7,  # 70% of cost threshold
    min_instances=0,  # Can scale to zero
    max_instances=5,
    scale_up_cooldown_minutes=10,
    scale_down_cooldown_minutes=30,
    cost_weight=0.8,  # Prioritize cost savings
    performance_weight=0.2,
)
```

## Cost Analytics

### Cost Breakdown

```python
# Get detailed cost breakdown
breakdown = cost_system.get_cost_breakdown(
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now()
)

print(f"Total Cost: ${breakdown.total_cost:.2f}")
print(f"Compute: ${breakdown.compute_cost:.2f}")
print(f"Memory: ${breakdown.memory_cost:.2f}")
print(f"Storage: ${breakdown.storage_cost:.2f}")
print(f"Network: ${breakdown.network_cost:.2f}")
print(f"API: ${breakdown.api_cost:.2f}")
```

### Cost Trends

```python
# Analyze cost trends
trends = cost_system.get_cost_trends(days=30)
print(f"Average Daily Cost: ${trends['average_daily_cost']:.2f}")
print(f"Cost Growth Rate: {trends.get('cost_growth_rate', 0):.1f}%")
print(f"Peak Daily Cost: ${max(trends['costs']):.2f}")
```

### Optimization Recommendations

```python
# Get optimization recommendations
recommendations = cost_system.get_optimization_recommendations(
    category="compute",
    priority_min=7
)

for rec in recommendations:
    print(f"{rec.title}: ${rec.potential_savings:.2f} savings")
    print(f"  Priority: {rec.priority}/10")
    print(f"  Confidence: {rec.confidence:.1%}")
    print(f"  Effort: {rec.effort_level}")
```

### Anomaly Detection

```python
# Get cost anomalies
anomalies = cost_system.get_cost_anomalies(
    severity="high",
    days=30
)

for anomaly in anomalies:
    print(f"Anomaly: {anomaly.anomaly_type}")
    print(f"  Cost: ${anomaly.cost_value:.2f}")
    print(f"  Expected: ${anomaly.expected_value:.2f}")
    print(f"  Deviation: {anomaly.deviation_percent:.1f}%")
```

## Integration Examples

### With FastAPI

```python
from fastapi import FastAPI
from src.llama_mapper.cost_monitoring import CostMonitoringSystem

app = FastAPI()
cost_system = CostMonitoringSystem(config)

@app.on_event("startup")
async def startup():
    await cost_system.start()

@app.on_event("shutdown")
async def shutdown():
    await cost_system.stop()

@app.get("/cost/status")
async def get_cost_status():
    return cost_system.get_system_status()

@app.get("/cost/breakdown")
async def get_cost_breakdown(days: int = 7):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    return cost_system.get_cost_breakdown(start_time, end_time)
```

### With Kubernetes

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cost-monitoring-config
data:
  config.yaml: |
    daily_budget_limit: 1000.0
    monthly_budget_limit: 30000.0
    emergency_stop_threshold: 5000.0
    metrics_collector:
      collection_interval_seconds: 60
      retention_days: 90
    guardrails:
      max_violations_per_hour: 10
    autoscaling:
      evaluation_interval_seconds: 60
      cost_threshold_percent: 80.0
```

### With Prometheus

```python
# Export metrics to Prometheus
from prometheus_client import Counter, Histogram, Gauge

cost_total = Counter('llama_mapper_cost_total', 'Total cost', ['component', 'tenant'])
cost_rate = Gauge('llama_mapper_cost_rate', 'Cost rate per hour')
anomaly_count = Counter('llama_mapper_anomalies_total', 'Total anomalies', ['severity'])
```

## Best Practices

### 1. **Gradual Implementation**
- Start with basic cost monitoring
- Add guardrails incrementally
- Implement autoscaling after establishing baselines

### 2. **Multi-Tenant Considerations**
- Set tenant-specific budgets
- Implement cost isolation
- Monitor cross-tenant resource usage

### 3. **Cost Optimization**
- Regular review of optimization recommendations
- Implement cost-aware development practices
- Monitor cost trends and adjust budgets

### 4. **Emergency Procedures**
- Test emergency stop procedures
- Have rollback plans for scaling decisions
- Maintain communication channels for alerts

### 5. **Monitoring and Alerting**
- Set up comprehensive monitoring
- Configure appropriate alert thresholds
- Regular review of cost analytics

## Troubleshooting

### Common Issues

1. **High False Positive Alerts**
   - Adjust anomaly detection thresholds
   - Review cost baselines
   - Implement alert cooldowns

2. **Aggressive Scaling**
   - Increase cooldown periods
   - Adjust cost/performance weights
   - Review scaling thresholds

3. **Missing Cost Data**
   - Check metrics collection configuration
   - Verify resource monitoring
   - Review data retention settings

### Debug Commands

```bash
# Check system health
mapper cost status

# View recent anomalies
mapper cost anomalies --days 1

# Check scaling decisions
mapper cost recommendations --category autoscaling

# Export detailed logs
mapper cost breakdown --days 7 --format json > debug.json
```

## Performance Considerations

### Resource Usage
- Metrics collection: ~1-2% CPU overhead
- Guardrail evaluation: ~0.5% CPU overhead
- Autoscaling decisions: ~0.1% CPU overhead
- Analytics processing: ~1% CPU overhead (periodic)

### Storage Requirements
- Metrics data: ~1MB per day per tenant
- Alert history: ~100KB per day
- Analytics data: ~10MB per month

### Network Impact
- Minimal network overhead for internal communication
- Optional external integrations (Prometheus, Grafana)

## Security Considerations

### Data Protection
- Cost data encryption at rest and in transit
- Tenant data isolation
- Access control and audit logging

### API Security
- Authentication for all cost monitoring APIs
- Rate limiting for cost queries
- Input validation and sanitization

### Compliance
- SOC 2 compliance for cost data handling
- GDPR compliance for EU customers
- Audit trail for all cost-related actions

## Future Enhancements

### Planned Features
- Machine learning-based cost prediction
- Advanced anomaly detection algorithms
- Integration with cloud provider cost APIs
- Real-time cost optimization recommendations
- Cost allocation and chargeback features

### Roadmap
- Q1: Enhanced ML-based forecasting
- Q2: Cloud provider integrations
- Q3: Advanced cost allocation
- Q4: Real-time optimization engine

## Support and Resources

### Documentation
- [API Reference](api_reference.md)
- [Configuration Guide](configuration_guide.md)
- [Troubleshooting Guide](troubleshooting_guide.md)

### Community
- GitHub Issues for bug reports
- Discussion forums for questions
- Slack channel for real-time support

### Professional Support
- Enterprise support available
- Custom configuration assistance
- Training and consulting services
