---
inclusion: manual
---

# Microservice Deployment Guide & Infrastructure

## Overview

This guide covers deployment strategies for the 3-microservice llama-mapper system:
1. **Detector Orchestration Service**: Detector coordination, policy enforcement
2. **Analysis Service**: Advanced analysis, risk scoring, compliance intelligence  
3. **Mapper Service**: Core mapping, model serving, response generation

Reference the OpenAPI specification for detailed API contracts:
#[[file:docs/openapi.yaml]]

## Service Architecture

Each service is independently deployable with:
- Own database schema and connection pools
- Independent configuration management
- Service-specific monitoring and health checks
- Isolated scaling and resource management

## Environment Configuration

### Development Environment
```yaml
# config/development.yaml
environment: development
debug: true
log_level: DEBUG

# Service-specific database configurations
orchestration_database:
  host: localhost
  database: orchestration_dev
  
analysis_database:
  host: localhost
  database: analysis_dev
  
mapper_database:
  host: localhost
  database: mapper_dev
  port: 5432
  name: llama_mapper_dev
  
redis:
  host: localhost
  port: 6379
  
model:
  serving_backend: local
  model_path: ./checkpoints/llama-3-8b-lora
```

### Production Environment
```yaml
# config/production.yaml
environment: production
debug: false
log_level: INFO

database:
  host: ${DB_HOST}
  port: ${DB_PORT}
  name: ${DB_NAME}
  
redis:
  host: ${REDIS_HOST}
  port: ${REDIS_PORT}
  
model:
  serving_backend: vllm
  model_endpoint: ${MODEL_ENDPOINT}
```

## Docker Deployment

### Multi-Stage Dockerfile
```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8000
CMD ["uvicorn", "src.llama_mapper.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  mapper-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLAMA_MAPPER_DATABASE__HOST=postgres
      - LLAMA_MAPPER_REDIS__HOST=redis
    depends_on:
      - postgres
      - redis
      
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: llama_mapper
      POSTGRES_USER: mapper
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Kubernetes Deployment

### Deployment Manifest
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-mapper
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llama-mapper
  template:
    metadata:
      labels:
        app: llama-mapper
    spec:
      containers:
      - name: mapper
        image: llama-mapper:latest
        ports:
        - containerPort: 8000
        env:
        - name: LLAMA_MAPPER_DATABASE__HOST
          value: postgres-service
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### Service Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: llama-mapper-service
spec:
  selector:
    app: llama-mapper
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Infrastructure as Code

### Terraform Configuration
```hcl
# main.tf
resource "aws_ecs_cluster" "llama_mapper" {
  name = "llama-mapper"
}

resource "aws_ecs_service" "mapper_service" {
  name            = "mapper-service"
  cluster         = aws_ecs_cluster.llama_mapper.id
  task_definition = aws_ecs_task_definition.mapper.arn
  desired_count   = 3
  
  load_balancer {
    target_group_arn = aws_lb_target_group.mapper.arn
    container_name   = "mapper"
    container_port   = 8000
  }
}
```

## Monitoring & Observability

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'llama-mapper'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboards
- API Response Times
- Request Volume and Error Rates
- Model Inference Latency
- Database Connection Pool Status
- Redis Cache Hit Rates

### Alerting Rules
```yaml
# alerts.yml
groups:
  - name: llama-mapper
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        annotations:
          summary: "High error rate detected"
          
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        annotations:
          summary: "High latency detected"
```

## Security Configuration

### TLS/SSL Setup
```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.llamapper.com;
    
    ssl_certificate /etc/ssl/certs/llamapper.crt;
    ssl_certificate_key /etc/ssl/private/llamapper.key;
    
    location / {
        proxy_pass http://llama-mapper-service;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Secrets Management
```yaml
# Using Kubernetes secrets
apiVersion: v1
kind: Secret
metadata:
  name: llama-mapper-secrets
type: Opaque
data:
  database-password: <base64-encoded-password>
  api-key: <base64-encoded-api-key>
```

## Backup & Recovery

### Database Backup
```bash
#!/bin/bash
# backup.sh
pg_dump -h $DB_HOST -U $DB_USER -d llama_mapper > backup_$(date +%Y%m%d_%H%M%S).sql
aws s3 cp backup_*.sql s3://llama-mapper-backups/
```

### Model Checkpoint Backup
```bash
#!/bin/bash
# model-backup.sh
tar -czf model_checkpoint_$(date +%Y%m%d).tar.gz checkpoints/
aws s3 cp model_checkpoint_*.tar.gz s3://llama-mapper-models/
```

## Scaling Configuration

### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama-mapper-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-mapper
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Load Balancer Configuration
```yaml
# Application Load Balancer settings
health_check:
  path: /health
  interval: 30s
  timeout: 5s
  healthy_threshold: 2
  unhealthy_threshold: 3
```

## Deployment Pipeline

### CI/CD Workflow
1. **Build Stage**: Create Docker image, run tests
2. **Security Scan**: Vulnerability scanning, dependency audit
3. **Deploy to Staging**: Automated deployment to staging environment
4. **Integration Tests**: Run full test suite against staging
5. **Deploy to Production**: Blue-green deployment with rollback capability

### Rollback Strategy
- Maintain previous 3 versions for quick rollback
- Automated rollback on health check failures
- Database migration rollback procedures
- Model version rollback capabilities