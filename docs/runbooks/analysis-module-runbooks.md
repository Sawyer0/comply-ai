# Analysis Module Operational Runbooks

This document contains operational runbooks for the Llama Mapper Analysis Module, providing step-by-step procedures for common operational scenarios, incident response, and maintenance tasks.

## Table of Contents

1. [Schema Validation Drops](#schema-validation-drops)
2. [Template Fallback Spikes](#template-fallback-spikes)
3. [OPA Compilation Failures](#opa-compilation-failures)
4. [Model Server 5xx Errors](#model-server-5xx-errors)
5. [Quality Degradation Alerts](#quality-degradation-alerts)
6. [Cost Anomaly Response](#cost-anomaly-response)
7. [API Key Management](#api-key-management)
8. [WAF Rule Management](#waf-rule-management)
9. [Circuit Breaker Recovery](#circuit-breaker-recovery)
10. [Weekly Evaluation Issues](#weekly-evaluation-issues)

---

## Schema Validation Drops

### Overview
Schema validation drops indicate that the analysis model is producing outputs that don't conform to the expected JSON schema, requiring fallback to template responses.

### Alert Thresholds
- **Warning**: Schema validation rate < 95%
- **Critical**: Schema validation rate < 90%

### Immediate Response (0-5 minutes)

#### 1. Acknowledge Alert
```bash
# Check current schema validation rate
curl -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/analysis/metrics/schema-validation-rate"

# Expected response: {"rate": 0.85, "threshold": 0.90, "status": "critical"}
```

#### 2. Check System Health
```bash
# Check analysis module health
curl "https://analysis-api.company.com/api/v1/analysis/health"

# Check model server status
curl "https://analysis-api.company.com/api/v1/analysis/health/model-server"

# Check recent errors
kubectl logs -f deployment/analysis-module --tail=100 | grep -i "schema"
```

#### 3. Identify Root Cause
```bash
# Check model server metrics
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.monitoring.metrics_collector import AnalysisMetricsCollector
collector = AnalysisMetricsCollector()
print('Schema validation rate:', collector.get_metric('schema_valid_rate'))
print('Template fallback rate:', collector.get_metric('template_fallback_rate'))
"

# Check recent analysis requests
kubectl logs deployment/analysis-module --since=10m | \
  grep -E "(schema.*invalid|template.*fallback)" | tail -20
```

### Investigation Steps (5-15 minutes)

#### 4. Analyze Recent Requests
```bash
# Get recent failed validations
kubectl exec -it deployment/analysis-module -- \
  python -c "
import json
from src.llama_mapper.analysis.infrastructure.validator import AnalysisValidator
validator = AnalysisValidator()

# Check last 10 validation failures
failures = validator.get_recent_validation_failures(limit=10)
for failure in failures:
    print(f'Request ID: {failure.request_id}')
    print(f'Error: {failure.error_message}')
    print(f'Timestamp: {failure.timestamp}')
    print('---')
"
```

#### 5. Check Model Performance
```bash
# Check model confidence scores
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.model_server import AnalysisModelServer
model_server = AnalysisModelServer()

# Get model statistics
stats = model_server.get_model_statistics()
print(f'Average confidence: {stats.avg_confidence}')
print(f'Low confidence count: {stats.low_confidence_count}')
print(f'Model load time: {stats.model_load_time}')
"
```

### Resolution Steps (15-30 minutes)

#### 6. Apply Immediate Fixes

**If model confidence is low:**
```bash
# Restart model server
kubectl rollout restart deployment/analysis-module

# Wait for rollout to complete
kubectl rollout status deployment/analysis-module

# Verify model is loaded
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.model_server import AnalysisModelServer
model_server = AnalysisModelServer()
print('Model loaded:', model_server.is_model_loaded())
"
```

**If schema validation logic has issues:**
```bash
# Check schema file integrity
kubectl exec -it deployment/analysis-module -- \
  python -c "
import json
from src.llama_mapper.analysis.schemas.AnalystInput import schema

# Validate schema
try:
    json.dumps(schema)
    print('Schema is valid JSON')
except Exception as e:
    print(f'Schema validation error: {e}')
"
```

#### 7. Update Configuration
```bash
# If needed, adjust confidence cutoff
kubectl patch configmap analysis-config --patch '
{
  "data": {
    "ANALYSIS_CONFIDENCE_CUTOFF": "0.25"
  }
}'

# Restart to apply changes
kubectl rollout restart deployment/analysis-module
```

### Verification (30-45 minutes)

#### 8. Monitor Recovery
```bash
# Watch schema validation rate recovery
watch -n 30 '
curl -s -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/analysis/metrics/schema-validation-rate" | \
  jq ".rate"
'

# Check for 5 minutes of stable >90% rate
for i in {1..10}; do
  rate=$(curl -s -H "Authorization: Bearer $API_KEY" \
    "https://analysis-api.company.com/api/v1/analysis/metrics/schema-validation-rate" | \
    jq -r ".rate")
  echo "Check $i: Schema validation rate = $rate"
  if (( $(echo "$rate < 0.90" | bc -l) )); then
    echo "Rate still below threshold"
    exit 1
  fi
  sleep 30
done
echo "Schema validation rate recovered!"
```

### Post-Incident Actions

#### 9. Document Incident
```bash
# Create incident report
cat > /tmp/schema-validation-incident-$(date +%Y%m%d-%H%M).md << EOF
# Schema Validation Drop Incident

**Date**: $(date)
**Duration**: [Duration of incident]
**Root Cause**: [Identified root cause]
**Resolution**: [Steps taken to resolve]
**Prevention**: [Steps to prevent recurrence]

## Timeline
- [Time]: Alert received
- [Time]: Investigation started
- [Time]: Root cause identified
- [Time]: Resolution applied
- [Time]: Service recovered

## Metrics
- Lowest schema validation rate: [Rate]
- Number of affected requests: [Count]
- Recovery time: [Duration]
EOF
```

#### 10. Update Monitoring
```bash
# Review and adjust alert thresholds if needed
kubectl patch configmap analysis-alerts --patch '
{
  "data": {
    "schema_validation_warning_threshold": "0.92",
    "schema_validation_critical_threshold": "0.88"
  }
}'
```

---

## Template Fallback Spikes

### Overview
Template fallback spikes occur when the analysis model cannot produce valid outputs and falls back to predefined templates, indicating potential model or validation issues.

### Alert Thresholds
- **Warning**: Template fallback rate > 20%
- **Critical**: Template fallback rate > 30%

### Immediate Response (0-5 minutes)

#### 1. Acknowledge Alert
```bash
# Check current fallback rate
curl -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/analysis/metrics/template-fallback-rate"

# Check fallback reasons
curl -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/analysis/metrics/fallback-reasons"
```

#### 2. Check System Load
```bash
# Check request volume
kubectl top pods -l app=analysis-module

# Check error rates
kubectl logs deployment/analysis-module --since=5m | \
  grep -E "(fallback|template)" | wc -l
```

### Investigation Steps (5-15 minutes)

#### 3. Analyze Fallback Patterns
```bash
# Get fallback reason breakdown
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.templates import AnalysisTemplates
templates = AnalysisTemplates()

# Get fallback statistics
stats = templates.get_fallback_statistics()
print('Fallback reasons:')
for reason, count in stats.reasons.items():
    print(f'  {reason}: {count}')
print(f'Total fallbacks: {stats.total_fallbacks}')
print(f'Fallback rate: {stats.fallback_rate:.2%}')
"
```

#### 4. Check Model Server Health
```bash
# Check model server metrics
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.model_server import AnalysisModelServer
model_server = AnalysisModelServer()

# Check model health
health = model_server.get_health_status()
print(f'Model loaded: {health.model_loaded}')
print(f'Memory usage: {health.memory_usage_mb}MB')
print(f'Last inference time: {health.last_inference_time_ms}ms')
print(f'Error rate: {health.error_rate:.2%}')
"
```

### Resolution Steps (15-30 minutes)

#### 5. Address Common Causes

**If model server is overloaded:**
```bash
# Scale up analysis module
kubectl scale deployment analysis-module --replicas=3

# Check resource limits
kubectl describe deployment analysis-module | grep -A 5 "Limits"
```

**If model confidence is consistently low:**
```bash
# Adjust confidence cutoff temporarily
kubectl patch configmap analysis-config --patch '
{
  "data": {
    "ANALYSIS_CONFIDENCE_CUTOFF": "0.20"
  }
}'

# Restart to apply changes
kubectl rollout restart deployment/analysis-module
```

**If schema validation is failing:**
```bash
# Check for schema changes
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.schemas.AnalystInput import schema
print('Schema version:', schema.get('$schema', 'unknown'))
print('Required fields:', list(schema.get('required', [])))
"
```

### Verification (30-45 minutes)

#### 6. Monitor Recovery
```bash
# Watch fallback rate recovery
watch -n 30 '
curl -s -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/analysis/metrics/template-fallback-rate" | \
  jq ".rate"
'

# Verify fallback rate drops below 20%
for i in {1..10}; do
  rate=$(curl -s -H "Authorization: Bearer $API_KEY" \
    "https://analysis-api.company.com/api/v1/analysis/metrics/template-fallback-rate" | \
    jq -r ".rate")
  echo "Check $i: Fallback rate = $rate"
  if (( $(echo "$rate > 0.20" | bc -l) )); then
    echo "Rate still above threshold"
    exit 1
  fi
  sleep 30
done
echo "Fallback rate recovered!"
```

---

## OPA Compilation Failures

### Overview
OPA compilation failures indicate that generated Rego policies have syntax errors or validation issues, preventing proper policy enforcement.

### Alert Thresholds
- **Warning**: OPA compilation success rate < 95%
- **Critical**: OPA compilation success rate < 90%

### Immediate Response (0-5 minutes)

#### 1. Check OPA Status
```bash
# Check OPA compilation metrics
curl -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/analysis/metrics/opa-compilation-rate"

# Check recent OPA errors
kubectl logs deployment/analysis-module --since=10m | \
  grep -i "opa.*error\|rego.*error" | tail -10
```

#### 2. Test OPA Compilation
```bash
# Test OPA compilation manually
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.opa_generator import OPAPolicyGenerator
generator = OPAPolicyGenerator()

# Test basic policy generation
try:
    policy = generator.generate_coverage_policy({'jailbreak': 0.95})
    print('Basic policy generation: SUCCESS')
except Exception as e:
    print(f'Basic policy generation: FAILED - {e}')

# Test policy compilation
try:
    result = generator.validate_rego(policy)
    print(f'Policy compilation: {result}')
except Exception as e:
    print(f'Policy compilation: FAILED - {e}')
"
```

### Investigation Steps (5-15 minutes)

#### 3. Analyze Compilation Errors
```bash
# Get detailed OPA error logs
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.opa_generator import OPAPolicyGenerator
generator = OPAPolicyGenerator()

# Get recent compilation errors
errors = generator.get_recent_compilation_errors(limit=5)
for error in errors:
    print(f'Error: {error.error_message}')
    print(f'Policy: {error.policy_snippet[:100]}...')
    print(f'Timestamp: {error.timestamp}')
    print('---')
"
```

#### 4. Check OPA Version and Configuration
```bash
# Check OPA version
kubectl exec -it deployment/analysis-module -- opa version

# Check OPA configuration
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.opa_generator import OPAPolicyGenerator
generator = OPAPolicyGenerator()
config = generator.get_configuration()
print('OPA Configuration:')
for key, value in config.items():
    print(f'  {key}: {value}')
"
```

### Resolution Steps (15-30 minutes)

#### 5. Fix Common Issues

**If Rego syntax errors:**
```bash
# Update OPA policy templates
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.opa_generator import OPAPolicyGenerator
generator = OPAPolicyGenerator()

# Validate all policy templates
templates = generator.get_policy_templates()
for name, template in templates.items():
    try:
        generator.validate_rego(template)
        print(f'Template {name}: VALID')
    except Exception as e:
        print(f'Template {name}: INVALID - {e}')
"
```

**If OPA binary issues:**
```bash
# Restart analysis module to reload OPA
kubectl rollout restart deployment/analysis-module

# Verify OPA is working
kubectl exec -it deployment/analysis-module -- \
  opa eval --data /dev/null 'true'
```

### Verification (30-45 minutes)

#### 6. Monitor Recovery
```bash
# Watch OPA compilation rate recovery
watch -n 30 '
curl -s -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/analysis/metrics/opa-compilation-rate" | \
  jq ".rate"
'

# Test policy generation
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.opa_generator import OPAPolicyGenerator
generator = OPAPolicyGenerator()

# Test multiple policy types
test_cases = [
    {'type': 'coverage', 'data': {'jailbreak': 0.95}},
    {'type': 'threshold', 'data': {'pii': 0.75}},
    {'type': 'violation', 'data': {'high_sev_hits': ['jailbreak']}}
]

for case in test_cases:
    try:
        if case['type'] == 'coverage':
            policy = generator.generate_coverage_policy(case['data'])
        elif case['type'] == 'threshold':
            policy = generator.generate_threshold_policy(case['data'])
        else:
            policy = generator.generate_violation_policy(case['data'])
        
        result = generator.validate_rego(policy)
        print(f'{case[\"type\"]} policy: {\"SUCCESS\" if result else \"FAILED\"}')
    except Exception as e:
        print(f'{case[\"type\"]} policy: FAILED - {e}')
"
```

---

## Model Server 5xx Errors

### Overview
Model server 5xx errors indicate internal server errors in the AI model inference pipeline, often related to model loading, memory issues, or inference failures.

### Alert Thresholds
- **Warning**: Model server error rate > 5%
- **Critical**: Model server error rate > 10%

### Immediate Response (0-5 minutes)

#### 1. Check Model Server Health
```bash
# Check model server status
curl "https://analysis-api.company.com/api/v1/analysis/health/model-server"

# Check recent 5xx errors
kubectl logs deployment/analysis-module --since=5m | \
  grep -E "5[0-9][0-9]" | tail -10
```

#### 2. Check Resource Usage
```bash
# Check pod resource usage
kubectl top pods -l app=analysis-module

# Check memory usage
kubectl exec -it deployment/analysis-module -- \
  python -c "
import psutil
import os
print(f'Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f}MB')
print(f'Available memory: {psutil.virtual_memory().available / 1024 / 1024:.1f}MB')
"
```

### Investigation Steps (5-15 minutes)

#### 3. Analyze Error Patterns
```bash
# Get detailed error logs
kubectl logs deployment/analysis-module --since=10m | \
  grep -A 5 -B 5 "5[0-9][0-9]" | tail -50

# Check model server metrics
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.model_server import AnalysisModelServer
model_server = AnalysisModelServer()

# Get error statistics
stats = model_server.get_error_statistics()
print('Error breakdown:')
for error_type, count in stats.error_types.items():
    print(f'  {error_type}: {count}')
print(f'Total errors: {stats.total_errors}')
print(f'Error rate: {stats.error_rate:.2%}')
"
```

#### 4. Check Model Status
```bash
# Check if model is loaded
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.model_server import AnalysisModelServer
model_server = AnalysisModelServer()

print(f'Model loaded: {model_server.is_model_loaded()}')
print(f'Model path: {model_server.model_path}')
print(f'Model size: {model_server.get_model_size_mb()}MB')
"
```

### Resolution Steps (15-30 minutes)

#### 5. Apply Fixes Based on Root Cause

**If memory issues:**
```bash
# Increase memory limits
kubectl patch deployment analysis-module --patch '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "analysis-module",
          "resources": {
            "limits": {
              "memory": "4Gi"
            },
            "requests": {
              "memory": "2Gi"
            }
          }
        }]
      }
    }
  }
}'
```

**If model loading issues:**
```bash
# Restart model server
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.model_server import AnalysisModelServer
model_server = AnalysisModelServer()
model_server.reload_model()
print('Model reloaded successfully')
"
```

**If inference timeout issues:**
```bash
# Increase timeout configuration
kubectl patch configmap analysis-config --patch '
{
  "data": {
    "ANALYSIS_REQUEST_TIMEOUT_SECONDS": "60"
  }
}'

# Restart to apply changes
kubectl rollout restart deployment/analysis-module
```

### Verification (30-45 minutes)

#### 6. Monitor Recovery
```bash
# Watch error rate recovery
watch -n 30 '
curl -s "https://analysis-api.company.com/api/v1/analysis/health/model-server" | \
  jq ".error_rate"
'

# Test model inference
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.model_server import AnalysisModelServer
model_server = AnalysisModelServer()

# Test inference
test_input = {
    'metrics': {'coverage': 0.85, 'accuracy': 0.92},
    'evidence_refs': ['detector_1', 'detector_2']
}

try:
    result = model_server.analyze(test_input)
    print('Model inference: SUCCESS')
    print(f'Confidence: {result.confidence}')
except Exception as e:
    print(f'Model inference: FAILED - {e}')
"
```

---

## Quality Degradation Alerts

### Overview
Quality degradation alerts indicate that analysis quality metrics are declining, potentially affecting the reliability of analysis outputs.

### Alert Thresholds
- **Warning**: Quality score < 85%
- **Critical**: Quality score < 80%

### Immediate Response (0-5 minutes)

#### 1. Check Quality Metrics
```bash
# Check current quality score
curl -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/analysis/metrics/quality-score"

# Check quality trends
curl -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/analysis/metrics/quality-trends"
```

#### 2. Check Related Metrics
```bash
# Check all quality-related metrics
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.quality import QualityAlertingSystem
system = QualityAlertingSystem()

# Get quality metrics
metrics = system.get_quality_metrics()
for metric_type, value in metrics.items():
    print(f'{metric_type}: {value}')
"
```

### Investigation Steps (5-15 minutes)

#### 3. Analyze Quality Components
```bash
# Get detailed quality breakdown
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.evaluator import QualityEvaluator
evaluator = QualityEvaluator()

# Get quality evaluation summary
summary = evaluator.get_evaluation_summary()
print('Quality Evaluation Summary:')
print(f'  Schema validation rate: {summary.schema_valid_rate:.2%}')
print(f'  Rubric score: {summary.rubric_score:.2f}')
print(f'  OPA compilation rate: {summary.opa_compile_success_rate:.2%}')
print(f'  Overall quality score: {summary.overall_quality_score:.2%}')
"
```

#### 4. Check Recent Analysis Quality
```bash
# Get recent quality evaluations
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.evaluator import QualityEvaluator
evaluator = QualityEvaluator()

# Get recent evaluations
recent = evaluator.get_recent_evaluations(limit=10)
print('Recent Quality Evaluations:')
for eval in recent:
    print(f'  {eval.timestamp}: {eval.quality_score:.2f} ({eval.evaluation_type})')
"
```

### Resolution Steps (15-30 minutes)

#### 5. Address Quality Issues

**If schema validation is low:**
```bash
# Follow schema validation drop runbook
# (See Schema Validation Drops section above)
```

**If rubric scores are low:**
```bash
# Check model performance
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.model_server import AnalysisModelServer
model_server = AnalysisModelServer()

# Check model confidence
stats = model_server.get_model_statistics()
print(f'Average confidence: {stats.avg_confidence:.2f}')
print(f'Low confidence requests: {stats.low_confidence_count}')

# If confidence is low, consider model retraining or parameter adjustment
"
```

**If OPA compilation is failing:**
```bash
# Follow OPA compilation failure runbook
# (See OPA Compilation Failures section above)
```

### Verification (30-45 minutes)

#### 6. Monitor Quality Recovery
```bash
# Watch quality score recovery
watch -n 60 '
curl -s -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/analysis/metrics/quality-score" | \
  jq ".score"
'

# Run quality evaluation
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.evaluator import QualityEvaluator
evaluator = QualityEvaluator()

# Run evaluation on recent data
result = evaluator.evaluate_recent_requests(days=1)
print(f'Quality evaluation result: {result.overall_score:.2%}')
print(f'Recommendations: {result.recommendations}')
"
```

---

## Cost Anomaly Response

### Overview
Cost anomalies indicate unexpected spikes in resource usage or costs, potentially due to increased load, inefficient operations, or resource leaks.

### Alert Thresholds
- **Warning**: Daily cost > 80% of budget
- **Critical**: Daily cost > 100% of budget

### Immediate Response (0-5 minutes)

#### 1. Check Cost Metrics
```bash
# Check current cost metrics
curl -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/cost-monitoring/current-costs"

# Check cost trends
curl -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/cost-monitoring/cost-trends"
```

#### 2. Check Resource Usage
```bash
# Check current resource usage
kubectl top nodes
kubectl top pods --all-namespaces | grep analysis

# Check cost monitoring system
kubectl exec -it deployment/cost-monitoring -- \
  python -c "
from src.llama_mapper.cost_monitoring import CostMonitoringSystem
system = CostMonitoringSystem()

# Get current costs
costs = system.get_current_costs()
print(f'Current daily cost: ${costs.daily_cost:.2f}')
print(f'Budget limit: ${costs.daily_budget:.2f}')
print(f'Usage percentage: {costs.usage_percentage:.1f}%')
"
```

### Investigation Steps (5-15 minutes)

#### 3. Analyze Cost Breakdown
```bash
# Get detailed cost breakdown
kubectl exec -it deployment/cost-monitoring -- \
  python -c "
from src.llama_mapper.cost_monitoring import CostMonitoringSystem
system = CostMonitoringSystem()

# Get cost breakdown
breakdown = system.get_cost_breakdown()
print('Cost Breakdown:')
for resource_type, cost in breakdown.items():
    print(f'  {resource_type}: ${cost:.2f}')
"
```

#### 4. Check for Resource Leaks
```bash
# Check for resource leaks
kubectl exec -it deployment/cost-monitoring -- \
  python -c "
from src.llama_mapper.cost_monitoring import CostMonitoringSystem
system = CostMonitoringSystem()

# Check for anomalies
anomalies = system.detect_cost_anomalies()
for anomaly in anomalies:
    print(f'Anomaly: {anomaly.description}')
    print(f'  Severity: {anomaly.severity}')
    print(f'  Cost impact: ${anomaly.cost_impact:.2f}')
    print(f'  Recommendations: {anomaly.recommendations}')
"
```

### Resolution Steps (15-30 minutes)

#### 5. Apply Cost Controls

**If CPU costs are high:**
```bash
# Scale down if possible
kubectl scale deployment analysis-module --replicas=1

# Check CPU usage per pod
kubectl top pods -l app=analysis-module --sort-by=cpu
```

**If memory costs are high:**
```bash
# Optimize memory limits
kubectl patch deployment analysis-module --patch '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "analysis-module",
          "resources": {
            "limits": {
              "memory": "2Gi"
            }
          }
        }]
      }
    }
  }
}'
```

**If storage costs are high:**
```bash
# Clean up old data
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.storage_backend import FileStorageBackend
storage = FileStorageBackend()

# Clean up old reports and logs
cleaned = storage.cleanup_old_data(days=30)
print(f'Cleaned up {cleaned} old files')
"
```

### Verification (30-45 minutes)

#### 6. Monitor Cost Recovery
```bash
# Watch cost trends
watch -n 300 '
curl -s -H "Authorization: Bearer $API_KEY" \
  "https://analysis-api.company.com/api/v1/cost-monitoring/current-costs" | \
  jq ".daily_cost"
'

# Check cost optimization recommendations
kubectl exec -it deployment/cost-monitoring -- \
  python -c "
from src.llama_mapper.cost_monitoring import CostMonitoringSystem
system = CostMonitoringSystem()

# Get optimization recommendations
recommendations = system.get_optimization_recommendations()
for rec in recommendations:
    print(f'Recommendation: {rec.description}')
    print(f'  Potential savings: ${rec.potential_savings:.2f}')
    print(f'  Implementation effort: {rec.effort}')
"
```

---

## API Key Management

### Overview
API key management involves creating, rotating, and revoking API keys for secure access to the analysis module.

### Key Operations

#### 1. Create New API Key
```bash
# Create new API key
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.auth import APIKeyManager
manager = APIKeyManager()

# Create key with specific scopes
key = manager.create_api_key(
    name='production-client',
    scopes=['analyze', 'batch_analyze'],
    rate_limit=1000,
    expires_in_days=90
)
print(f'API Key created: {key.key_id}')
print(f'Secret: {key.secret}')
print(f'Expires: {key.expires_at}')
"
```

#### 2. List API Keys
```bash
# List all API keys
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.auth import APIKeyManager
manager = APIKeyManager()

# List keys
keys = manager.list_api_keys()
for key in keys:
    print(f'Key: {key.name} ({key.key_id})')
    print(f'  Scopes: {key.scopes}')
    print(f'  Rate limit: {key.rate_limit}')
    print(f'  Status: {key.status}')
    print(f'  Expires: {key.expires_at}')
    print('---')
"
```

#### 3. Rotate API Key
```bash
# Rotate existing API key
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.auth import APIKeyManager
manager = APIKeyManager()

# Rotate key
new_key = manager.rotate_api_key('existing-key-id')
print(f'New API Key: {new_key.key_id}')
print(f'New Secret: {new_key.secret}')
print('Old key will be revoked in 24 hours')
"
```

#### 4. Revoke API Key
```bash
# Revoke API key
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.auth import APIKeyManager
manager = APIKeyManager()

# Revoke key
success = manager.revoke_api_key('key-id-to-revoke')
print(f'Key revoked: {success}')
"
```

---

## WAF Rule Management

### Overview
WAF (Web Application Firewall) rule management involves updating security rules, managing IP blocklists, and responding to security incidents.

### Rule Operations

#### 1. Add Custom WAF Rule
```bash
# Add custom WAF rule
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.security.waf.engine.rule_engine import WAFRuleEngine
from src.llama_mapper.analysis.security.waf.interfaces import AttackType, ViolationSeverity

engine = WAFRuleEngine()

# Add custom rule
success = engine.add_custom_rule(
    name='custom_malware_pattern',
    pattern=r'malware.*signature',
    attack_type=AttackType.MALICIOUS_PAYLOAD,
    severity=ViolationSeverity.CRITICAL,
    description='Custom malware pattern detection'
)
print(f'Custom rule added: {success}')
"
```

#### 2. Block Suspicious IP
```bash
# Block suspicious IP
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.security.waf.engine.rule_engine import WAFRuleEngine
engine = WAFRuleEngine()

# Block IP
engine.blocked_ips.add('192.168.1.100')
print('IP 192.168.1.100 blocked')

# Check blocked IPs
print(f'Blocked IPs: {list(engine.blocked_ips)}')
"
```

#### 3. Check WAF Statistics
```bash
# Check WAF statistics
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.security.waf.engine.rule_engine import WAFRuleEngine
engine = WAFRuleEngine()

# Get statistics
stats = {
    'total_rules': len(engine.rules),
    'blocked_ips': len(engine.blocked_ips),
    'suspicious_ips': len(engine.suspicious_ips)
}
print('WAF Statistics:')
for key, value in stats.items():
    print(f'  {key}: {value}')
"
```

---

## Circuit Breaker Recovery

### Overview
Circuit breaker recovery involves restoring service when circuit breakers are open due to upstream service failures.

### Recovery Operations

#### 1. Check Circuit Breaker Status
```bash
# Check circuit breaker status
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.resilience.circuit_breaker.implementation import CircuitBreaker
from src.llama_mapper.analysis.resilience.interfaces import CircuitState

# Check model server circuit breaker
cb = CircuitBreaker('model_server')
print(f'Circuit state: {cb.state}')
print(f'Failure count: {cb._failure_count}')
print(f'Success count: {cb._success_count}')
"
```

#### 2. Force Circuit Breaker Reset
```bash
# Force circuit breaker reset
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.resilience.circuit_breaker.implementation import CircuitBreaker
from src.llama_mapper.analysis.resilience.interfaces import CircuitState

# Reset circuit breaker
cb = CircuitBreaker('model_server')
cb._state = CircuitState.CLOSED
cb._failure_count = 0
cb._success_count = 0
print('Circuit breaker reset to CLOSED state')
"
```

#### 3. Test Circuit Breaker Recovery
```bash
# Test circuit breaker recovery
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.resilience.circuit_breaker.implementation import CircuitBreaker
from src.llama_mapper.analysis.resilience.retry.implementation import RetryManager

# Test with retry manager
retry_manager = RetryManager()
cb = CircuitBreaker('model_server')

# Test successful call
try:
    result = retry_manager.execute_with_retry(
        lambda: 'success',
        circuit_breaker=cb
    )
    print(f'Test call successful: {result}')
except Exception as e:
    print(f'Test call failed: {e}')
"
```

---

## Weekly Evaluation Issues

### Overview
Weekly evaluation issues involve problems with scheduled quality evaluations, report generation, or notification delivery.

### Common Issues and Solutions

#### 1. Check Evaluation Status
```bash
# Check evaluation status
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.domain.services import WeeklyEvaluationService
service = WeeklyEvaluationService()

# Get evaluation status
status = service.get_system_status()
print('Evaluation Status:')
print(f'  Monitoring active: {status[\"monitoring_active\"]}')
print(f'  Schedules created: {status[\"statistics\"][\"schedules_created\"]}')
print(f'  Evaluations run: {status[\"statistics\"][\"evaluations_run\"]}')
print(f'  Last evaluation: {status[\"statistics\"][\"last_evaluation\"]}')
"
```

#### 2. Run Manual Evaluation
```bash
# Run manual evaluation
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.domain.services import WeeklyEvaluationService
service = WeeklyEvaluationService()

# Run evaluation for specific tenant
result = service.run_scheduled_evaluation('tenant-123')
print(f'Evaluation result: {result.status}')
print(f'Report generated: {result.report_generated}')
print(f'Notifications sent: {result.notifications_sent}')
"
```

#### 3. Check Report Generation
```bash
# Check report generation
kubectl exec -it deployment/analysis-module -- \
  python -c "
from src.llama_mapper.analysis.infrastructure.report_generator import ReportGenerator
generator = ReportGenerator()

# Test report generation
try:
    report = generator.generate_quality_report('tenant-123', days=7)
    print(f'Report generated: {report.filename}')
    print(f'Report size: {report.size_bytes} bytes')
    print(f'Report format: {report.format}')
except Exception as e:
    print(f'Report generation failed: {e}')
"
```

---

## Emergency Contacts and Escalation

### On-Call Rotation
- **Primary**: [Primary on-call engineer]
- **Secondary**: [Secondary on-call engineer]
- **Manager**: [Engineering manager]

### Escalation Procedures
1. **Level 1**: On-call engineer (0-15 minutes)
2. **Level 2**: Senior engineer (15-30 minutes)
3. **Level 3**: Engineering manager (30+ minutes)

### Communication Channels
- **Slack**: #analysis-module-alerts
- **PagerDuty**: Analysis Module
- **Email**: analysis-alerts@company.com

### Post-Incident Review
All incidents should be followed by a post-incident review within 48 hours, including:
- Root cause analysis
- Timeline reconstruction
- Action items for prevention
- Documentation updates

---

## Maintenance Windows

### Scheduled Maintenance
- **Weekly**: Every Sunday 2-4 AM UTC
- **Monthly**: First Saturday 1-3 AM UTC
- **Quarterly**: First Sunday 12-6 AM UTC

### Maintenance Tasks
- **Weekly**: Log rotation, cache cleanup, health checks
- **Monthly**: Security updates, dependency updates, performance review
- **Quarterly**: Model retraining, capacity planning, disaster recovery testing

### Maintenance Notifications
- **24 hours before**: Slack notification
- **2 hours before**: Email notification
- **During maintenance**: Status page updates
- **After maintenance**: Completion notification

---

*This document should be reviewed and updated monthly or after any significant system changes.*
