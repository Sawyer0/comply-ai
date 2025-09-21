# Detector Orchestration Service Monitoring & Alerting

This document describes the comprehensive monitoring and alerting setup for the Detector Orchestration Service.

## Metrics

The service exposes Prometheus metrics at `/metrics` endpoint with the following locked metric names:

### Core Orchestration Metrics
- `orchestrate_requests_total` - Total orchestration requests (labeled by tenant, policy, status, processing_mode)
- `orchestrate_request_duration_ms` - Request duration histogram
- `orchestrate_request_duration_sync_ms` - Sync request duration histogram
- `orchestrate_request_duration_async_ms` - Async request duration histogram
- `detector_latency_ms` - Detector call latency (labeled by detector, status)
- `detector_health_status` - Detector health status (1=healthy, 0=unhealthy)
- `circuit_breaker_state` - Circuit breaker state (labeled by detector, state)
- `coverage_achieved` - Coverage achieved (labeled by tenant, policy)
- `policy_enforcement_total` - Policy enforcement results (labeled by tenant, policy, status, violation_type)

### Service Health Metrics
- `orchestrator_ready` - Service readiness status (1=ready, 0=not_ready)
- `orchestrator_service_dependencies_up` - Service dependency status (labeled by dependency)
- `orchestrator_async_jobs_active` - Number of active async jobs
- `orchestrator_async_jobs_queued` - Number of queued async jobs
- `orchestrator_async_jobs_completed_total` - Completed async jobs (labeled by status)
- `orchestrator_cache_hit_ratio` - Cache hit ratio (labeled by cache_type)

### Security & Infrastructure Metrics
- `orchestrator_rate_limit_requests_total` - Rate limit decisions
- `orchestrator_rbac_enforcement_total` - RBAC enforcement decisions
- `orchestrator_redis_backend_up` - Redis backend health
- `orchestrator_redis_backend_fallback_total` - Redis fallback occurrences

## Health Endpoints

### `/health`
Returns basic service health information:
```json
{
  "status": "healthy",
  "ts": "2025-01-20T10:30:00Z",
  "detectors_total": 5,
  "detectors_healthy": 4
}
```

### `/health/ready`
Kubernetes readiness probe that validates:
- All service dependencies are initialized
- Redis backend is healthy (if configured)
- At least one healthy detector is available
- Records readiness metrics

Returns 200 if ready, 503 if not ready.

## Prometheus Alerting Rules

The service includes comprehensive alerting rules covering:

### Critical Alerts
- `OrchestratorNotReady` - Service not ready for >5 minutes
- `AllDetectorsUnhealthy` - All detectors unhealthy for >5 minutes
- `OrchestrationErrorRateHigh` - Error rate >5% for >5 minutes

### Warning Alerts
- `OrchestratorServiceDependencyDown` - Service dependencies unavailable for >10 minutes
- `DetectorUnhealthy` - Individual detector unhealthy for >15 minutes
- `CircuitBreakerTripped` - Circuit breaker open for >5 minutes
- `OrchestrationLatencyHigh` - P95 latency >2s for >10 minutes
- `CoverageBelowThreshold` - Average coverage <80% for >15 minutes
- `AsyncJobsQueueHigh` - Queue >100 jobs for >10 minutes
- `CacheHitRatioLow` - Hit ratio <70% for >15 minutes
- `RedisBackendDown` - Redis backend unhealthy for >5 minutes

## Grafana Dashboards

The included dashboard (`grafana-dashboard.json`) provides comprehensive monitoring:

### Key Panels
- **Request Rate** - Requests per second by tenant/policy/status
- **Request Duration** - P95 latency for sync/async processing
- **Detector Health** - Real-time detector health status
- **Coverage Achieved** - Coverage metrics by tenant/policy
- **Circuit Breaker Status** - Open/closed breaker counts
- **Async Jobs** - Queue and active job counts
- **Cache Performance** - Hit ratio monitoring
- **Service Dependencies** - Dependency health table
- **Error Rate** - Error percentage over time

## Kubernetes Deployment

The deployment includes proper health probes:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 30
  periodSeconds: 15

readinessProbe:
  httpGet:
    path: /health/ready
    port: http
  initialDelaySeconds: 10
  periodSeconds: 5
```

## Service Monitor Configuration

The ServiceMonitor ensures proper metrics collection:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: detector-orchestration
spec:
  selector:
    matchLabels:
      app: detector-orchestration
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
    scrapeTimeout: 5s
```

## Deployment Instructions

1. **Deploy ServiceMonitor**:
   ```bash
   kubectl apply -f servicemonitor.yaml
   ```

2. **Deploy Prometheus Rules**:
   ```bash
   kubectl apply -f prometheus-rules.yaml
   ```

3. **Deploy Application**:
   ```bash
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   ```

4. **Import Grafana Dashboard**:
   - Go to Grafana → Dashboards → Import
   - Upload `grafana-dashboard.json`
   - Select Prometheus data source

## Alert Runbooks

All alerts include runbook URLs pointing to:
```
https://github.com/your-org/comply-ai/blob/main/docs/runbook/alert-runbooks.md
```

## Metric Validation

To validate metrics are working correctly:

```bash
# Check metrics endpoint
curl http://detector-orchestration:8000/metrics | grep orchestrator_ready

# Check health endpoints
curl http://detector-orchestration:8000/health
curl http://detector-orchestration:8000/health/ready

# Test alerting rules
kubectl port-forward svc/prometheus 9090:9090
# Visit http://localhost:9090/rules to see rule evaluation
```

## Troubleshooting

### Common Issues

1. **Metrics not appearing in Prometheus**
   - Check ServiceMonitor configuration
   - Verify service labels match selector
   - Ensure metrics endpoint is accessible

2. **Alerts not firing**
   - Validate metric names match exactly
   - Check rule evaluation in Prometheus UI
   - Verify for durations and thresholds

3. **Readiness probe failing**
   - Check service dependency initialization
   - Verify detector health status
   - Review Redis connectivity (if configured)

4. **Dashboard not loading**
   - Verify Prometheus data source
   - Check metric names in queries
   - Ensure time range is appropriate
