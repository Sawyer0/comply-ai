# Detector Orchestration Helm Chart

This Helm chart deploys the Detector Orchestration service with OPA policy integration and comprehensive configuration management.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- Optional: Redis for caching
- Optional: OPA for policy enforcement

## Installation

### Basic Installation

```bash
# Install with default values
helm install detector-orchestration ./charts/detector-orchestration

# Install with custom values
helm install detector-orchestration ./charts/detector-orchestration -f values-prod.yaml
```

### Environment-Specific Deployments

```bash
# Development
helm install detector-orchestration ./charts/detector-orchestration -f values-dev.yaml

# Staging
helm install detector-orchestration ./charts/detector-orchestration -f values-staging.yaml

# Production
helm install detector-orchestration ./charts/detector-orchestration -f values-prod.yaml
```

## Configuration

### Environment Variables

The service is configured through environment variables with the `ORCH_` prefix. Key configuration areas:

- **Core Settings**: `ORCH_ENVIRONMENT`, `ORCH_LOG_LEVEL`
- **SLA Configuration**: `ORCH_CONFIG__SLA__SYNC_REQUEST_SLA_MS`
- **Cache Configuration**: `ORCH_CONFIG__CACHE_BACKEND`, `ORCH_CONFIG__REDIS_URL`
- **OPA Configuration**: `ORCH_CONFIG__OPA_ENABLED`, `ORCH_CONFIG__OPA_URL`
- **Detector Configuration**: Configured through values.yaml

### Detector Configuration

Detectors are configured in the `detectors` section of values.yaml:

```yaml
detectors:
  toxicity:
    endpoint: "http://detector-toxicity:8000/detect"
    timeout_ms: 3000
    max_retries: 3
    auth:
      type: "api_key"
      header: "X-API-Key"
    weight: 1.0
    supported_content_types: ["text"]
```

### OPA Policy Integration

When OPA is enabled, policies are mounted from ConfigMaps and evaluated for:
- Detector selection based on tenant policies
- Conflict resolution strategies
- Coverage requirements

Example OPA query:
```bash
curl -X POST http://localhost:8181/v1/data/detector_orchestration/allow \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "tenant_id": "tenant-critical",
      "content_type": "text",
      "policy_bundle": "default",
      "detector_name": "toxicity"
    }
  }'
```

## Components

### Main Service
- **Image**: Configurable via `image.repository` and `image.tag`
- **Port**: 8000 (HTTP)
- **Health Checks**: `/health` endpoint
- **Metrics**: `/metrics` endpoint (Prometheus format)

### OPA Sidecar (Optional)
- **Purpose**: Policy evaluation for detector selection and conflict resolution
- **Port**: 8181
- **Policies**: Mounted from ConfigMap `detector-policies`

### Redis (Optional)
- **Purpose**: Caching and idempotency
- **Port**: 6379
- **Configuration**: External Redis or in-cluster deployment

## Monitoring

### Metrics
The service exposes Prometheus metrics at `/metrics`:
- Request latency and throughput
- Detector health and circuit breaker status
- Coverage achievement metrics
- Policy enforcement metrics

### Health Checks
- **Liveness Probe**: `/health` (30s delay, 10s interval)
- **Readiness Probe**: `/health` (5s delay, 5s interval)

### Service Monitor
When `metrics.serviceMonitor.enabled` is true, creates a ServiceMonitor for Prometheus Operator.

## Security

### RBAC
- ServiceAccount with minimal required permissions
- Role and RoleBinding for ConfigMap and Secret access

### Network Policies
When `networkPolicy.enabled` is true:
- Restricts ingress to service port
- Controls egress to detectors, Redis, and OPA
- DNS resolution allowed

### Security Context
- Non-root user execution
- Read-only root filesystem when possible
- Dropped capabilities

## Scaling

### Horizontal Pod Autoscaling
When `autoscaling.enabled` is true:
- Scales based on CPU and memory utilization
- Configurable min/max replicas per environment

### Resource Management
Resource limits and requests are environment-specific:
- **Dev**: 200m CPU, 256Mi memory limits
- **Staging**: 1000m CPU, 1Gi memory limits
- **Prod**: 2000m CPU, 2Gi memory limits

## Development

### Local Development
```bash
# Run with local configuration
helm install detector-orchestration ./charts/detector-orchestration -f values-dev.yaml

# Port forward for local access
kubectl port-forward svc/detector-orchestration 8000:80
```

### Building Custom Images
```bash
# Build and push custom image
docker build -t detector-orchestration:dev .
docker tag detector-orchestration:dev your-registry/detector-orchestration:dev
docker push your-registry/detector-orchestration:dev

# Update values.yaml
image:
  repository: your-registry/detector-orchestration
  tag: dev
```

## Troubleshooting

### Common Issues

1. **OPA Policy Errors**
   - Check ConfigMap `detector-policies` exists
   - Verify policy syntax with `opa test`
   - Check OPA sidecar logs

2. **Detector Communication Failures**
   - Verify detector endpoints are reachable
   - Check circuit breaker status at `/health`
   - Review network policies

3. **Redis Connection Issues**
   - Verify Redis service is running
   - Check Redis URL configuration
   - Review authentication settings

### Useful Commands

```bash
# Check service status
kubectl get pods -l app.kubernetes.io/name=detector-orchestration

# View logs
kubectl logs -l app.kubernetes.io/name=detector-orchestration

# Check OPA policies
kubectl exec -it <opa-pod> -- opa eval 'data.detector_orchestration'

# Test health endpoint
curl http://localhost:8000/health
```

## Upgrading

```bash
# Upgrade with new values
helm upgrade detector-orchestration ./charts/detector-orchestration -f values-prod.yaml

# Check upgrade status
helm status detector-orchestration

# Rollback if needed
helm rollback detector-orchestration 1
```

## Uninstallation

```bash
# Remove deployment
helm uninstall detector-orchestration

# Clean up persistent resources if any
kubectl delete configmap detector-policies
```
