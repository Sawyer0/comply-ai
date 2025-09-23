# Database Backup and Restore Procedures

This document provides comprehensive procedures for backing up and restoring all databases in the Comply-AI platform, including PostgreSQL, ClickHouse, and Redis.

## Overview

The backup and restore system provides:
- **Automated daily backups** for all databases
- **S3-compatible storage** (AWS S3 or Azure Blob Storage)
- **Point-in-time recovery** capabilities
- **Backup integrity verification**
- **Automated cleanup** of old backups
- **Monitoring and alerting** for backup failures

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   ClickHouse     │    │     Redis       │
│                 │    │                  │    │                 │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          └──────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────▼──────────────┐
                    │    Backup Scripts          │
                    │  (backup-databases.py)     │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │   S3 Storage              │
                    │ (AWS S3 or Azure Blob)    │
                    └────────────────────────────┘
```

## Database Configurations

### PostgreSQL
- **Host**: `postgresql-service.default.svc.cluster.local`
- **Port**: `5432`
- **Database**: `llama_mapper`
- **User**: `complyai`
- **Backup Format**: Custom format with compression

### ClickHouse
- **Host**: `clickhouse-service.default.svc.cluster.local`
- **Port**: `9000`
- **Database**: `llama_mapper`
- **User**: `default`
- **Backup Format**: SQL schema + CSV data

### Redis
- **Host**: `redis-service.default.svc.cluster.local`
- **Port**: `6379`
- **Password**: From Kubernetes secret
- **Backup Format**: RDB dump file

## Backup Procedures

### 1. Manual Backup

#### Single Database
```bash
# PostgreSQL
python scripts/backup-databases.py \
  --config config/backup-config.json \
  --database postgresql

# ClickHouse
python scripts/backup-databases.py \
  --config config/backup-config.json \
  --database clickhouse

# Redis
python scripts/backup-databases.py \
  --config config/backup-config.json \
  --database redis
```

#### All Databases
```bash
python scripts/backup-databases.py \
  --config config/backup-config.json \
  --database all
```

### 2. Automated Backup (Kubernetes CronJobs)

The platform includes automated backup CronJobs that run daily:

```bash
# Check CronJob status
kubectl get cronjobs -n default

# View CronJob logs
kubectl logs -l job-name=backup-postgresql -n default
kubectl logs -l job-name=backup-clickhouse -n default
kubectl logs -l job-name=backup-redis -n default
```

### 3. Backup Configuration

Create `config/backup-config.json`:

```json
{
  "s3": {
    "bucket": "comply-ai-backups",
    "region": "us-east-1",
    "endpoint_url": null
  },
  "postgresql": {
    "host": "postgresql-service.default.svc.cluster.local",
    "port": 5432,
    "database": "llama_mapper",
    "user": "complyai",
    "password": "from_kubernetes_secret"
  },
  "clickhouse": {
    "host": "clickhouse-service.default.svc.cluster.local",
    "port": 9000,
    "database": "llama_mapper",
    "user": "default",
    "password": "from_kubernetes_secret"
  },
  "redis": {
    "host": "redis-service.default.svc.cluster.local",
    "port": 6379,
    "password": "from_kubernetes_secret"
  },
  "retention": {
    "days": 30,
    "compliance_days": 2555
  },
  "alerts": {
    "email_recipients": ["admin@company.com"],
    "enabled": true
  }
}
```

### 4. Azure Blob Storage Configuration

For Azure Blob Storage (S3-compatible), update the S3 configuration:

```json
{
  "s3": {
    "bucket": "comply-ai-backups",
    "region": "eastus",
    "endpoint_url": "https://complyaistorage.blob.core.windows.net"
  }
}
```

Set environment variables:
```bash
export AWS_ACCESS_KEY_ID="your_azure_storage_account_key"
export AWS_SECRET_ACCESS_KEY="your_azure_storage_account_secret"
```

## Restore Procedures

### 1. List Available Backups

```bash
# List PostgreSQL backups
python scripts/restore-databases.py \
  --config config/backup-config.json \
  --database postgresql \
  --list

# List ClickHouse backups
python scripts/restore-databases.py \
  --config config/backup-config.json \
  --database clickhouse \
  --list

# List Redis backups
python scripts/restore-databases.py \
  --config config/backup-config.json \
  --database redis \
  --list
```

### 2. Restore from Specific Backup

#### PostgreSQL Restore
```bash
# Restore from timestamp
python scripts/restore-databases.py \
  --config config/backup-config.json \
  --database postgresql \
  --backup "2024-01-15-14-30-00"

# Restore from full S3 key
python scripts/restore-databases.py \
  --config config/backup-config.json \
  --database postgresql \
  --backup "backups/llama-mapper/postgresql/2024-01-15-14-30-00.dump"

# Restore with drop existing data
python scripts/restore-databases.py \
  --config config/backup-config.json \
  --database postgresql \
  --backup "2024-01-15-14-30-00" \
  --drop-existing
```

#### ClickHouse Restore
```bash
# Restore from timestamp
python scripts/restore-databases.py \
  --config config/backup-config.json \
  --database clickhouse \
  --backup "2024-01-15-14-30-00"

# Restore with drop existing data
python scripts/restore-databases.py \
  --config config/backup-config.json \
  --database clickhouse \
  --backup "2024-01-15-14-30-00" \
  --drop-existing
```

#### Redis Restore
```bash
# Restore from timestamp
python scripts/restore-databases.py \
  --config config/backup-config.json \
  --database redis \
  --backup "2024-01-15-14-30-00"

# Restore with flush existing data
python scripts/restore-databases.py \
  --config config/backup-config.json \
  --database redis \
  --backup "2024-01-15-14-30-00" \
  --drop-existing
```

### 3. Kubernetes Restore Jobs

For production restores, use Kubernetes Jobs:

```bash
# Create PostgreSQL restore job
kubectl apply -f charts/llama-mapper/templates/restore-postgresql-job.yaml

# Create ClickHouse restore job
kubectl apply -f charts/llama-mapper/templates/restore-clickhouse-job.yaml

# Create Redis restore job
kubectl apply -f charts/llama-mapper/templates/restore-redis-job.yaml
```

## Monitoring and Alerting

### 1. Backup Health Monitoring

```bash
# Check backup health
python scripts/backup-monitor.py \
  --config config/backup-config.json \
  --check

# Generate backup report
python scripts/backup-monitor.py \
  --config config/backup-config.json \
  --report

# Check and send alerts
python scripts/backup-monitor.py \
  --config config/backup-config.json \
  --check --alerts
```

### 2. Monitoring Configuration

Add monitoring configuration to `config/backup-config.json`:

```json
{
  "smtp": {
    "enabled": true,
    "smtp_server": "smtp.company.com",
    "smtp_port": 587,
    "username": "backup-monitor@company.com",
    "password": "password",
    "from_email": "backup-monitor@company.com"
  },
  "slack": {
    "enabled": true,
    "webhook_url": "https://hooks.slack.com/services/..."
  },
  "webhook": {
    "enabled": true,
    "url": "https://monitoring.company.com/webhook"
  },
  "databases": {
    "postgresql": {
      "expected_frequency_hours": 24
    },
    "clickhouse": {
      "expected_frequency_hours": 24
    },
    "redis": {
      "expected_frequency_hours": 24
    }
  }
}
```

### 3. Automated Monitoring (Kubernetes CronJob)

```bash
# Check monitoring CronJob
kubectl get cronjobs -n default | grep backup-monitor

# View monitoring logs
kubectl logs -l job-name=backup-monitor -n default
```

## Disaster Recovery Procedures

### 1. Full Platform Restore

In case of complete platform failure:

1. **Restore PostgreSQL**:
   ```bash
   python scripts/restore-databases.py \
     --config config/backup-config.json \
     --database postgresql \
     --backup "latest_timestamp" \
     --drop-existing
   ```

2. **Restore ClickHouse**:
   ```bash
   python scripts/restore-databases.py \
     --config config/backup-config.json \
     --database clickhouse \
     --backup "latest_timestamp" \
     --drop-existing
   ```

3. **Restore Redis**:
   ```bash
   python scripts/restore-databases.py \
     --config config/backup-config.json \
     --database redis \
     --backup "latest_timestamp" \
     --drop-existing
   ```

4. **Restart Services**:
   ```bash
   kubectl rollout restart deployment/llama-mapper-api
   kubectl rollout restart deployment/llama-mapper-worker
   ```

### 2. Point-in-Time Recovery

For point-in-time recovery:

1. **Identify target timestamp** from backup list
2. **Restore databases** to that timestamp
3. **Verify data integrity**
4. **Restart services**

### 3. Recovery Time Objectives (RTO)

- **PostgreSQL**: 15 minutes
- **ClickHouse**: 30 minutes
- **Redis**: 5 minutes
- **Full Platform**: 60 minutes

## Backup Cleanup

### 1. Manual Cleanup

```bash
# Clean up backups older than 30 days
python scripts/backup-databases.py \
  --config config/backup-config.json \
  --cleanup 30

# Dry run cleanup
python scripts/backup-databases.py \
  --config config/backup-config.json \
  --cleanup 30 \
  --dry-run
```

### 2. Automated Cleanup

The backup scripts automatically clean up old backups based on retention policies:
- **Standard retention**: 30 days
- **Compliance retention**: 7 years (2555 days)

## Troubleshooting

### Common Issues

1. **Backup fails with authentication error**:
   - Check Kubernetes secrets
   - Verify database credentials
   - Ensure network connectivity

2. **S3 upload fails**:
   - Check AWS/Azure credentials
   - Verify bucket permissions
   - Check network connectivity

3. **Restore fails**:
   - Verify backup file integrity
   - Check database connectivity
   - Ensure sufficient disk space

### Log Locations

- **Backup logs**: `/var/log/backup.log`
- **Restore logs**: `/var/log/restore.log`
- **Monitor logs**: `/var/log/backup-monitor.log`
- **Kubernetes logs**: `kubectl logs -l app=backup`

### Health Checks

```bash
# Check backup script health
python scripts/backup-databases.py \
  --config config/backup-config.json \
  --database postgresql \
  --verify

# Check restore script health
python scripts/restore-databases.py \
  --config config/backup-config.json \
  --database postgresql \
  --list
```

## Security Considerations

1. **Encryption**: All backups are encrypted in transit and at rest
2. **Access Control**: Backup access is restricted to authorized personnel
3. **Audit Logging**: All backup/restore operations are logged
4. **Key Management**: Database passwords are stored in Kubernetes secrets
5. **Network Security**: Backup operations use secure connections

## Compliance

The backup system supports compliance requirements:
- **SOC 2**: Audit trails and access controls
- **ISO 27001**: Information security management
- **HIPAA**: Healthcare data protection
- **GDPR**: Data protection and privacy

## Maintenance

### Regular Tasks

1. **Weekly**: Review backup health reports
2. **Monthly**: Test restore procedures
3. **Quarterly**: Update backup configurations
4. **Annually**: Review retention policies

### Updates

1. **Backup scripts**: Update as needed for new database versions
2. **Configuration**: Update for new database instances
3. **Monitoring**: Update alert thresholds based on usage patterns
