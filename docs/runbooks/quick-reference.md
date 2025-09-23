# Analysis Module Runbooks - Quick Reference

## 🚨 Emergency Response (0-5 minutes)

### Schema Validation Drops
```bash
# Check current rate
curl -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/analysis/metrics/schema-validation-rate"

# Quick investigation
python scripts/runbook-executor.py investigate-schema-validation
```

### Template Fallback Spikes
```bash
# Check current rate
curl -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/analysis/metrics/template-fallback-rate"

# Quick investigation
python scripts/runbook-executor.py investigate-template-fallback
```

### Model Server 5xx Errors
```bash
# Check health
curl "https://analysis-api.company.com/api/v1/analysis/health/model-server"

# Check logs
kubectl logs deployment/analysis-module --since=5m | grep -E "5[0-9][0-9]"
```

### OPA Compilation Failures
```bash
# Check compilation rate
curl -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/analysis/metrics/opa-compilation-rate"

# Quick investigation
python scripts/runbook-executor.py investigate-opa-compilation
```

### Cost Anomalies
```bash
# Check costs
curl -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/cost-monitoring/current-costs"

# Quick investigation
python scripts/runbook-executor.py investigate-cost-anomaly
```

## 🔧 Common Fixes

### Restart Analysis Module
```bash
kubectl rollout restart deployment/analysis-module
kubectl rollout status deployment/analysis-module
```

### Scale Up/Down
```bash
# Scale up
kubectl scale deployment analysis-module --replicas=3

# Scale down
kubectl scale deployment analysis-module --replicas=1
```

### Adjust Configuration
```bash
# Lower confidence cutoff
kubectl patch configmap analysis-config --patch '
{
  "data": {
    "ANALYSIS_CONFIDENCE_CUTOFF": "0.25"
  }
}'

# Increase timeout
kubectl patch configmap analysis-config --patch '
{
  "data": {
    "ANALYSIS_REQUEST_TIMEOUT_SECONDS": "60"
  }
}'
```

### Reset Circuit Breaker
```bash
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.resilience.circuit_breaker.implementation import CircuitBreaker
from src.llama_mapper.analysis.resilience.interfaces import CircuitState
cb = CircuitBreaker('model_server')
cb._state = CircuitState.CLOSED
cb._failure_count = 0
print('Circuit breaker reset')
"
```

## 📊 Monitoring Commands

### Health Check
```bash
# Comprehensive health check
python scripts/runbook-executor.py health-check

# Check specific metrics
python scripts/runbook-executor.py check-metrics --format json
```

### Resource Usage
```bash
# Check pod resources
kubectl top pods -l app=analysis-module

# Check node resources
kubectl top nodes
```

### Log Analysis
```bash
# Recent errors
kubectl logs deployment/analysis-module --since=10m | grep -i error

# Schema validation issues
kubectl logs deployment/analysis-module --since=10m | grep -i "schema.*invalid"

# Template fallbacks
kubectl logs deployment/analysis-module --since=10m | grep -i "template.*fallback"
```

## 🔑 API Key Management

### Create API Key
```bash
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.auth import APIKeyManager
manager = APIKeyManager()
key = manager.create_api_key(
    name='production-client',
    scopes=['analyze', 'batch_analyze'],
    rate_limit=1000,
    expires_in_days=90
)
print(f'API Key: {key.key_id}')
print(f'Secret: {key.secret}')
"
```

### List API Keys
```bash
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.auth import APIKeyManager
manager = APIKeyManager()
keys = manager.list_api_keys()
for key in keys:
    print(f'{key.name}: {key.key_id} ({key.status})')
"
```

### Rotate API Key
```bash
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.auth import APIKeyManager
manager = APIKeyManager()
new_key = manager.rotate_api_key('existing-key-id')
print(f'New Key: {new_key.key_id}')
print(f'New Secret: {new_key.secret}')
"
```

## 🛡️ WAF Management

### Block IP
```bash
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.security.waf.engine.rule_engine import WAFRuleEngine
engine = WAFRuleEngine()
engine.blocked_ips.add('192.168.1.100')
print('IP blocked')
"
```

### Check WAF Stats
```bash
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.security.waf.engine.rule_engine import WAFRuleEngine
engine = WAFRuleEngine()
print(f'Total rules: {len(engine.rules)}')
print(f'Blocked IPs: {len(engine.blocked_ips)}')
"
```

## 📈 Quality Monitoring

### Check Quality Score
```bash
curl -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/analysis/metrics/quality-score"
```

### Run Quality Evaluation
```bash
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.evaluator import QualityEvaluator
evaluator = QualityEvaluator()
result = evaluator.evaluate_recent_requests(days=1)
print(f'Quality score: {result.overall_score:.2%}')
"
```

## 💰 Cost Monitoring

### Check Current Costs
```bash
curl -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/cost-monitoring/current-costs"
```

### Get Cost Breakdown
```bash
kubectl exec -it deployment/cost-monitoring -- \
  python -c "
from src.llama_mapper.cost_monitoring import CostMonitoringSystem
system = CostMonitoringSystem()
breakdown = system.get_cost_breakdown()
for resource, cost in breakdown.items():
    print(f'{resource}: ${cost:.2f}')
"
```

## 📅 Weekly Evaluation

### Check Evaluation Status
```bash
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.domain.services import WeeklyEvaluationService
service = WeeklyEvaluationService()
status = service.get_system_status()
print(f'Monitoring active: {status[\"monitoring_active\"]}')
print(f'Last evaluation: {status[\"statistics\"][\"last_evaluation\"]}')
"
```

### Run Manual Evaluation
```bash
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.domain.services import WeeklyEvaluationService
service = WeeklyEvaluationService()
result = service.run_scheduled_evaluation('tenant-123')
print(f'Status: {result.status}')
"
```

## 🚨 Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|---------|
| Schema Validation Rate | < 95% | < 90% | Check model, restart if needed |
| Template Fallback Rate | > 20% | > 30% | Check model health, scale up |
| OPA Compilation Rate | < 95% | < 90% | Check Rego syntax, restart |
| Model Server Error Rate | > 5% | > 10% | Check resources, restart |
| Quality Score | < 85% | < 80% | Investigate components |
| Daily Cost | > 80% budget | > 100% budget | Scale down, optimize |

## 📞 Escalation

### Level 1 (0-15 min)
- On-call engineer
- Basic troubleshooting
- Restart services
- Scale resources

### Level 2 (15-30 min)
- Senior engineer
- Configuration changes
- Advanced debugging
- System optimization

### Level 3 (30+ min)
- Engineering manager
- Architecture changes
- External dependencies
- Business impact assessment

## 📱 Communication

- **Slack**: #analysis-module-alerts
- **PagerDuty**: Analysis Module
- **Email**: analysis-alerts@company.com
- **Status Page**: https://status.company.com

## 🔄 Maintenance Windows

- **Weekly**: Sunday 2-4 AM UTC
- **Monthly**: First Saturday 1-3 AM UTC
- **Quarterly**: First Sunday 12-6 AM UTC

---

*For detailed procedures, see [analysis-module-runbooks.md](./analysis-module-runbooks.md)*
