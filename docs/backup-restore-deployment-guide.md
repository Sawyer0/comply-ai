# Backup and Restore Deployment Guide

This guide provides step-by-step instructions for deploying the database backup and restore system for the Comply-AI platform.

## Prerequisites

- Kubernetes cluster with Helm 3.x installed
- Access to S3-compatible storage (AWS S3 or Azure Blob Storage)
- Database credentials stored in Kubernetes secrets
- Storage credentials stored in Kubernetes secrets

## 1. Prepare Kubernetes Secrets

### Database Secrets

Create secrets for each database:

```bash
# PostgreSQL secret
kubectl create secret generic postgresql-backup-secret \
  --from-literal=password=your_postgresql_password

# ClickHouse secret
kubectl create secret generic clickhouse-backup-secret \
  --from-literal=password=your_clickhouse_password

# Redis secret
kubectl create secret generic redis-backup-secret \
  --from-literal=password=your_redis_password
```

### S3 Storage Secrets

For AWS S3:
```bash
kubectl create secret generic s3-backup-secret \
  --from-literal=access_key_id=your_aws_access_key \
  --from-literal=secret_access_key=your_aws_secret_key
```

For Azure Blob Storage:
```bash
kubectl create secret generic s3-backup-secret \
  --from-literal=access_key_id=your_azure_storage_account_key \
  --from-literal=secret_access_key=your_azure_storage_account_secret
```

## 2. Configure Helm Values

Update `charts/llama-mapper/values.yaml` with your backup configuration:

```yaml
backups:
  enabled: true
  s3:
    bucket: "comply-ai-backups"
    region: "us-east-1"
    endpointUrl: null  # Set to Azure Blob Storage URL if using Azure
    secretName: "s3-backup-secret"
  
  postgresql:
    enabled: true
    schedule: "0 2 * * *"  # Daily at 2 AM
    image: "python:3.11-slim"
    host: "postgresql-service.default.svc.cluster.local"
    port: 5432
    database: "llama_mapper"
    user: "complyai"
    secretName: "postgresql-backup-secret"
    resources:
      requests:
        cpu: "250m"
        memory: "512Mi"
      limits:
        cpu: "1"
        memory: "1Gi"
  
  clickhouse:
    enabled: true
    schedule: "0 3 * * *"  # Daily at 3 AM
    image: "python:3.11-slim"
    host: "clickhouse-service.default.svc.cluster.local"
    port: 9000
    database: "llama_mapper"
    user: "default"
    secretName: "clickhouse-backup-secret"
    resources:
      requests:
        cpu: "250m"
        memory: "512Mi"
      limits:
        cpu: "1"
        memory: "1Gi"
  
  redis:
    enabled: true
    schedule: "0 4 * * *"  # Daily at 4 AM
    image: "python:3.11-slim"
    host: "redis-service.default.svc.cluster.local"
    port: 6379
    secretName: "redis-backup-secret"
    resources:
      requests:
        cpu: "100m"
        memory: "256Mi"
      limits:
        cpu: "500m"
        memory: "512Mi"
  
  retention:
    days: 30
    complianceDays: 2555  # 7 years for compliance
```

## 3. Deploy Backup System

### Deploy with Helm

```bash
# Install or upgrade the chart
helm upgrade --install llama-mapper ./charts/llama-mapper \
  --namespace default \
  --values charts/llama-mapper/values.yaml
```

### Verify Deployment

```bash
# Check CronJobs
kubectl get cronjobs -n default

# Check ConfigMap
kubectl get configmap backup-scripts -n default

# Check secrets
kubectl get secrets -n default | grep backup
```

## 4. Test Backup System

### Manual Backup Test

```bash
# Test PostgreSQL backup
kubectl create job --from=cronjob/backup-postgresql test-postgresql-backup

# Test ClickHouse backup
kubectl create job --from=cronjob/backup-clickhouse test-clickhouse-backup

# Test Redis backup
kubectl create job --from=cronjob/backup-redis test-redis-backup
```

### Check Backup Logs

```bash
# View backup logs
kubectl logs job/test-postgresql-backup
kubectl logs job/test-clickhouse-backup
kubectl logs job/test-redis-backup
```

### Verify S3 Backups

```bash
# List backups in S3
aws s3 ls s3://comply-ai-backups/backups/llama-mapper/ --recursive

# For Azure Blob Storage
az storage blob list \
  --account-name complyaistorage \
  --container-name backups \
  --prefix backups/llama-mapper/ \
  --output table
```

## 5. Deploy Monitoring

### Backup Monitoring CronJob

Add to `values.yaml`:

```yaml
backups:
  monitoring:
    enabled: true
    schedule: "0 6 * * *"  # Daily at 6 AM
    image: "python:3.11-slim"
    resources:
      requests:
        cpu: "100m"
        memory: "256Mi"
      limits:
        cpu: "500m"
        memory: "512Mi"
```

### Deploy Monitoring

```bash
helm upgrade --install llama-mapper ./charts/llama-mapper \
  --namespace default \
  --values charts/llama-mapper/values.yaml
```

## 6. Test Restore System

### List Available Backups

```bash
# Create a test pod to run restore commands
kubectl run backup-test --image=python:3.11-slim --rm -it --restart=Never -- /bin/bash

# Inside the pod, install dependencies and list backups
pip install boto3 psycopg2-binary clickhouse-driver redis
python scripts/restore-databases.py --config config/backup-config.json --database postgresql --list
```

### Test Restore

```bash
# Restore PostgreSQL from latest backup
kubectl create job restore-postgresql-test \
  --from=cronjob/backup-postgresql \
  --dry-run=client -o yaml | \
  sed 's/backup-postgresql/restore-postgresql-test/g' | \
  kubectl apply -f -
```

## 7. Configure Alerting

### Email Alerts

Update `values.yaml`:

```yaml
backups:
  alerts:
    email:
      enabled: true
      smtp_server: "smtp.company.com"
      smtp_port: 587
      username: "backup-monitor@company.com"
      password: "password"
      from_email: "backup-monitor@company.com"
      recipients: ["admin@company.com"]
```

### Slack Alerts

```yaml
backups:
  alerts:
    slack:
      enabled: true
      webhook_url: "https://hooks.slack.com/services/..."
```

### Webhook Alerts

```yaml
backups:
  alerts:
    webhook:
      enabled: true
      url: "https://monitoring.company.com/webhook"
```

## 8. Production Considerations

### Resource Limits

Adjust resource limits based on your data size:

```yaml
backups:
  postgresql:
    resources:
      requests:
        cpu: "500m"
        memory: "1Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
```

### Backup Scheduling

Stagger backup schedules to avoid resource conflicts:

```yaml
backups:
  postgresql:
    schedule: "0 2 * * *"  # 2 AM
  clickhouse:
    schedule: "0 3 * * *"  # 3 AM
  redis:
    schedule: "0 4 * * *"  # 4 AM
```

### Retention Policies

Configure retention based on compliance requirements:

```yaml
backups:
  retention:
    days: 30          # Standard retention
    complianceDays: 2555  # 7 years for compliance
```

## 9. Troubleshooting

### Common Issues

1. **Backup fails with authentication error**:
   ```bash
   # Check secrets
   kubectl get secret postgresql-backup-secret -o yaml
   kubectl get secret s3-backup-secret -o yaml
   ```

2. **S3 upload fails**:
   ```bash
   # Check S3 credentials and permissions
   kubectl logs job/test-postgresql-backup
   ```

3. **CronJob not running**:
   ```bash
   # Check CronJob status
   kubectl describe cronjob backup-postgresql
   ```

### Log Locations

- **Backup logs**: Available in job logs
- **Application logs**: `/var/log/backup.log` inside containers
- **Kubernetes logs**: `kubectl logs -l app=backup-postgresql`

### Health Checks

```bash
# Check backup health
kubectl create job --from=cronjob/backup-monitor backup-health-check

# View health report
kubectl logs job/backup-health-check
```

## 10. Maintenance

### Regular Tasks

1. **Weekly**: Review backup health reports
2. **Monthly**: Test restore procedures
3. **Quarterly**: Update backup configurations
4. **Annually**: Review retention policies

### Updates

1. **Backup scripts**: Update as needed for new database versions
2. **Configuration**: Update for new database instances
3. **Monitoring**: Update alert thresholds based on usage patterns

## 11. Security Considerations

### Access Control

- Use Kubernetes RBAC to restrict backup access
- Store credentials in Kubernetes secrets
- Use IAM roles for S3 access when possible

### Encryption

- Enable encryption in transit for S3
- Use encrypted storage for backup files
- Encrypt sensitive data in backup files

### Audit Logging

- Enable audit logging for backup operations
- Monitor backup access and modifications
- Regular security reviews of backup procedures

## 12. Disaster Recovery

### Recovery Procedures

1. **Full platform restore**: Follow restore procedures for all databases
2. **Point-in-time recovery**: Restore to specific backup timestamp
3. **Cross-region recovery**: Restore from cross-region backups

### Testing

- Regular disaster recovery drills
- Test restore procedures monthly
- Document recovery time objectives (RTO)

## 13. Monitoring and Alerting

### Key Metrics

- Backup success/failure rates
- Backup duration and size
- Storage usage and costs
- Restore success rates

### Alerts

- Backup failures
- Backup delays
- Storage quota warnings
- Restore failures

### Dashboards

- Backup status dashboard
- Storage usage dashboard
- Recovery time dashboard
- Cost monitoring dashboard