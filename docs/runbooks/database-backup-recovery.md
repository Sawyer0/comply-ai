# Database Backup and Recovery Procedures

This document provides comprehensive procedures for database backup and recovery operations for the Llama Mapper Analysis Module.

## Table of Contents

1. [Overview](#overview)
2. [Backup Procedures](#backup-procedures)
3. [Restore Procedures](#restore-procedures)
4. [Disaster Recovery](#disaster-recovery)
5. [Testing and Validation](#testing-and-validation)
6. [Monitoring and Alerting](#monitoring-and-alerting)
7. [RPO/RTO Targets](#rporto-targets)

---

## Overview

The Llama Mapper Analysis Module uses multiple data stores:

- **PostgreSQL**: Primary database for structured data (API keys, evaluation results, configuration)
- **ClickHouse**: Analytics database for metrics and time-series data
- **Redis**: Cache and session storage
- **S3**: Object storage for reports, logs, and model artifacts

### Backup Strategy

- **Full Backups**: Daily automated backups
- **Incremental Backups**: Every 4 hours during business hours
- **Point-in-Time Recovery**: Continuous WAL archiving for PostgreSQL
- **Cross-Region Replication**: S3 cross-region replication for critical data
- **Retention Policy**: 30 days for full backups, 7 days for incremental

---

## Backup Procedures

### PostgreSQL Backup

#### Automated Daily Backup
```bash
#!/bin/bash
# scripts/backup-postgresql.sh

set -e

# Configuration
BACKUP_DIR="/backups/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="llama_mapper_analysis"
DB_USER="backup_user"
DB_HOST="postgres-primary.company.com"
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Full backup
echo "Starting PostgreSQL full backup..."
pg_dump -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
    --verbose \
    --no-password \
    --format=custom \
    --compress=9 \
    --file="$BACKUP_DIR/postgresql_full_$DATE.dump"

# Verify backup
if [ $? -eq 0 ]; then
    echo "PostgreSQL backup completed successfully"
    
    # Upload to S3
    aws s3 cp "$BACKUP_DIR/postgresql_full_$DATE.dump" \
        "s3://llama-mapper-backups/postgresql/postgresql_full_$DATE.dump"
    
    # Clean up old local backups
    find "$BACKUP_DIR" -name "postgresql_full_*.dump" -mtime +7 -delete
    
    # Clean up old S3 backups
    aws s3 ls s3://llama-mapper-backups/postgresql/ | \
        grep "postgresql_full_" | \
        awk '{print $4}' | \
        sort | \
        head -n -$RETENTION_DAYS | \
        xargs -I {} aws s3 rm "s3://llama-mapper-backups/postgresql/{}"
else
    echo "PostgreSQL backup failed!"
    exit 1
fi
```

#### Incremental Backup (WAL Archiving)
```bash
#!/bin/bash
# scripts/backup-postgresql-wal.sh

set -e

# Configuration
WAL_ARCHIVE_DIR="/backups/postgresql/wal"
S3_WAL_BUCKET="s3://llama-mapper-backups/postgresql/wal"
RETENTION_DAYS=7

# Create WAL archive directory
mkdir -p "$WAL_ARCHIVE_DIR"

# Archive WAL files
echo "Archiving WAL files..."
rsync -av /var/lib/postgresql/data/pg_wal/ "$WAL_ARCHIVE_DIR/"

# Upload to S3
aws s3 sync "$WAL_ARCHIVE_DIR/" "$S3_WAL_BUCKET/"

# Clean up old WAL files
find "$WAL_ARCHIVE_DIR" -name "*.wal" -mtime +$RETENTION_DAYS -delete
aws s3 ls "$S3_WAL_BUCKET/" | \
    awk '{print $4}' | \
    sort | \
    head -n -$((RETENTION_DAYS * 24)) | \
    xargs -I {} aws s3 rm "$S3_WAL_BUCKET/{}"
```

#### PostgreSQL Configuration for WAL Archiving
```bash
# /etc/postgresql/15/main/postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'test ! -f /backups/postgresql/wal/%f && cp %p /backups/postgresql/wal/%f'
archive_timeout = 300
max_wal_senders = 3
wal_keep_segments = 32
```

### ClickHouse Backup

#### Automated ClickHouse Backup
```bash
#!/bin/bash
# scripts/backup-clickhouse.sh

set -e

# Configuration
BACKUP_DIR="/backups/clickhouse"
DATE=$(date +%Y%m%d_%H%M%S)
CLICKHOUSE_HOST="clickhouse-primary.company.com"
CLICKHOUSE_USER="backup_user"
CLICKHOUSE_PASSWORD="backup_password"
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Full backup
echo "Starting ClickHouse full backup..."
clickhouse-backup create \
    --config /etc/clickhouse-backup/config.yml \
    --remote-storage s3 \
    --s3-bucket llama-mapper-backups \
    --s3-path clickhouse/ \
    "clickhouse_full_$DATE"

# Verify backup
if [ $? -eq 0 ]; then
    echo "ClickHouse backup completed successfully"
    
    # Clean up old backups
    clickhouse-backup list --remote-storage s3 | \
        grep "clickhouse_full_" | \
        sort | \
        head -n -$RETENTION_DAYS | \
        xargs -I {} clickhouse-backup delete --remote-storage s3 {}
else
    echo "ClickHouse backup failed!"
    exit 1
fi
```

#### ClickHouse Backup Configuration
```yaml
# /etc/clickhouse-backup/config.yml
clickhouse:
  username: backup_user
  password: backup_password
  host: clickhouse-primary.company.com
  port: 9000
  databases:
    - llama_mapper_analysis
  tables:
    - llama_mapper_analysis.analysis_metrics
    - llama_mapper_analysis.quality_evaluations
    - llama_mapper_analysis.cost_metrics

s3:
  access_key: AWS_ACCESS_KEY_ID
  secret_key: AWS_SECRET_ACCESS_KEY
  bucket: llama-mapper-backups
  region: us-west-2
  path: clickhouse/
  compression_format: gzip
  compression_level: 1
  disable_ssl: false
  part_size: 104857600
  storage_class: STANDARD
```

### Redis Backup

#### Redis RDB Backup
```bash
#!/bin/bash
# scripts/backup-redis.sh

set -e

# Configuration
BACKUP_DIR="/backups/redis"
DATE=$(date +%Y%m%d_%H%M%S)
REDIS_HOST="redis-primary.company.com"
REDIS_PORT=6379
RETENTION_DAYS=7

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Trigger RDB save
echo "Triggering Redis RDB save..."
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" BGSAVE

# Wait for save to complete
while [ "$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LASTSAVE)" = "$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LASTSAVE)" ]; do
    sleep 1
done

# Copy RDB file
echo "Copying Redis RDB file..."
cp /var/lib/redis/dump.rdb "$BACKUP_DIR/redis_$DATE.rdb"

# Upload to S3
aws s3 cp "$BACKUP_DIR/redis_$DATE.rdb" \
    "s3://llama-mapper-backups/redis/redis_$DATE.rdb"

# Clean up old backups
find "$BACKUP_DIR" -name "redis_*.rdb" -mtime +$RETENTION_DAYS -delete
aws s3 ls s3://llama-mapper-backups/redis/ | \
    grep "redis_" | \
    awk '{print $4}' | \
    sort | \
    head -n -$RETENTION_DAYS | \
    xargs -I {} aws s3 rm "s3://llama-mapper-backups/redis/{}"
```

### S3 Object Storage Backup

#### S3 Cross-Region Replication
```json
{
  "Role": "arn:aws:iam::123456789012:role/replication-role",
  "Rules": [
    {
      "ID": "ReplicateAnalysisData",
      "Status": "Enabled",
      "Prefix": "analysis/",
      "Destination": {
        "Bucket": "arn:aws:s3:::llama-mapper-backups-dr",
        "StorageClass": "STANDARD_IA"
      }
    },
    {
      "ID": "ReplicateReports",
      "Status": "Enabled",
      "Prefix": "reports/",
      "Destination": {
        "Bucket": "arn:aws:s3:::llama-mapper-backups-dr",
        "StorageClass": "GLACIER"
      }
    }
  ]
}
```

#### S3 Lifecycle Policy
```json
{
  "Rules": [
    {
      "ID": "AnalysisDataLifecycle",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "analysis/"
      },
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        },
        {
          "Days": 365,
          "StorageClass": "DEEP_ARCHIVE"
        }
      ]
    }
  ]
}
```

---

## Restore Procedures

### PostgreSQL Restore

#### Full Database Restore
```bash
#!/bin/bash
# scripts/restore-postgresql.sh

set -e

# Configuration
BACKUP_FILE="$1"
DB_NAME="llama_mapper_analysis"
DB_USER="restore_user"
DB_HOST="postgres-restore.company.com"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    echo "Example: $0 postgresql_full_20240101_120000.dump"
    exit 1
fi

# Download backup from S3 if needed
if [[ "$BACKUP_FILE" == s3://* ]]; then
    echo "Downloading backup from S3..."
    aws s3 cp "$BACKUP_FILE" "/tmp/$(basename "$BACKUP_FILE")"
    BACKUP_FILE="/tmp/$(basename "$BACKUP_FILE")"
fi

# Drop and recreate database
echo "Dropping and recreating database..."
psql -h "$DB_HOST" -U "$DB_USER" -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;"
psql -h "$DB_HOST" -U "$DB_USER" -d postgres -c "CREATE DATABASE $DB_NAME;"

# Restore database
echo "Restoring database from backup..."
pg_restore -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
    --verbose \
    --no-password \
    --clean \
    --if-exists \
    "$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo "PostgreSQL restore completed successfully"
    
    # Verify restore
    echo "Verifying restore..."
    psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "SELECT COUNT(*) FROM information_schema.tables;"
else
    echo "PostgreSQL restore failed!"
    exit 1
fi
```

#### Point-in-Time Recovery
```bash
#!/bin/bash
# scripts/restore-postgresql-pitr.sh

set -e

# Configuration
TARGET_TIME="$1"  # Format: YYYY-MM-DD HH:MM:SS
DB_NAME="llama_mapper_analysis"
DB_USER="restore_user"
DB_HOST="postgres-restore.company.com"
WAL_ARCHIVE_DIR="/backups/postgresql/wal"

if [ -z "$TARGET_TIME" ]; then
    echo "Usage: $0 '<target_time>'"
    echo "Example: $0 '2024-01-01 12:00:00'"
    exit 1
fi

# Download WAL files from S3
echo "Downloading WAL files from S3..."
aws s3 sync s3://llama-mapper-backups/postgresql/wal/ "$WAL_ARCHIVE_DIR/"

# Create recovery configuration
cat > /tmp/recovery.conf << EOF
restore_command = 'cp $WAL_ARCHIVE_DIR/%f %p'
recovery_target_time = '$TARGET_TIME'
recovery_target_action = 'promote'
EOF

# Restore base backup
echo "Restoring base backup..."
# (Use the most recent full backup before target time)
LATEST_BACKUP=$(aws s3 ls s3://llama-mapper-backups/postgresql/ | \
    grep "postgresql_full_" | \
    awk '{print $4}' | \
    sort | \
    tail -1)

aws s3 cp "s3://llama-mapper-backups/postgresql/$LATEST_BACKUP" "/tmp/$LATEST_BACKUP"

# Extract backup
pg_restore -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" \
    --verbose \
    --no-password \
    --clean \
    --if-exists \
    "/tmp/$LATEST_BACKUP"

# Apply WAL files for point-in-time recovery
echo "Applying WAL files for point-in-time recovery..."
# (This would be done by PostgreSQL automatically with the recovery.conf file)

echo "Point-in-time recovery completed successfully"
```

### ClickHouse Restore

#### ClickHouse Full Restore
```bash
#!/bin/bash
# scripts/restore-clickhouse.sh

set -e

# Configuration
BACKUP_NAME="$1"
CLICKHOUSE_HOST="clickhouse-restore.company.com"
CLICKHOUSE_USER="restore_user"
CLICKHOUSE_PASSWORD="restore_password"

if [ -z "$BACKUP_NAME" ]; then
    echo "Usage: $0 <backup_name>"
    echo "Example: $0 clickhouse_full_20240101_120000"
    exit 1
fi

# List available backups
echo "Available backups:"
clickhouse-backup list --remote-storage s3

# Restore backup
echo "Restoring ClickHouse backup: $BACKUP_NAME"
clickhouse-backup restore \
    --config /etc/clickhouse-backup/config.yml \
    --remote-storage s3 \
    --s3-bucket llama-mapper-backups \
    --s3-path clickhouse/ \
    "$BACKUP_NAME"

if [ $? -eq 0 ]; then
    echo "ClickHouse restore completed successfully"
    
    # Verify restore
    echo "Verifying restore..."
    clickhouse-client -h "$CLICKHOUSE_HOST" -u "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" \
        --query "SELECT COUNT(*) FROM llama_mapper_analysis.analysis_metrics"
else
    echo "ClickHouse restore failed!"
    exit 1
fi
```

### Redis Restore

#### Redis RDB Restore
```bash
#!/bin/bash
# scripts/restore-redis.sh

set -e

# Configuration
BACKUP_FILE="$1"
REDIS_HOST="redis-restore.company.com"
REDIS_PORT=6379

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    echo "Example: $0 redis_20240101_120000.rdb"
    exit 1
fi

# Download backup from S3 if needed
if [[ "$BACKUP_FILE" == s3://* ]]; then
    echo "Downloading backup from S3..."
    aws s3 cp "$BACKUP_FILE" "/tmp/$(basename "$BACKUP_FILE")"
    BACKUP_FILE="/tmp/$(basename "$BACKUP_FILE")"
fi

# Stop Redis
echo "Stopping Redis..."
systemctl stop redis

# Backup current RDB file
echo "Backing up current RDB file..."
cp /var/lib/redis/dump.rdb /var/lib/redis/dump.rdb.backup.$(date +%Y%m%d_%H%M%S)

# Copy restored RDB file
echo "Copying restored RDB file..."
cp "$BACKUP_FILE" /var/lib/redis/dump.rdb

# Set proper permissions
chown redis:redis /var/lib/redis/dump.rdb
chmod 640 /var/lib/redis/dump.rdb

# Start Redis
echo "Starting Redis..."
systemctl start redis

# Verify restore
echo "Verifying restore..."
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" PING

if [ $? -eq 0 ]; then
    echo "Redis restore completed successfully"
else
    echo "Redis restore failed!"
    exit 1
fi
```

---

## Disaster Recovery

### Complete System Recovery

#### Recovery Playbook
```bash
#!/bin/bash
# scripts/disaster-recovery.sh

set -e

# Configuration
RECOVERY_TIME="$1"  # Format: YYYY-MM-DD HH:MM:SS
ENVIRONMENT="$2"    # staging or production

if [ -z "$RECOVERY_TIME" ] || [ -z "$ENVIRONMENT" ]; then
    echo "Usage: $0 '<recovery_time>' <environment>"
    echo "Example: $0 '2024-01-01 12:00:00' production"
    exit 1
fi

echo "🚨 Starting disaster recovery for $ENVIRONMENT at $RECOVERY_TIME"

# Step 1: Provision new infrastructure
echo "📦 Provisioning new infrastructure..."
if [ "$ENVIRONMENT" = "production" ]; then
    terraform apply -var="environment=production" -auto-approve
else
    terraform apply -var="environment=staging" -auto-approve
fi

# Step 2: Restore PostgreSQL
echo "🐘 Restoring PostgreSQL..."
./scripts/restore-postgresql-pitr.sh "$RECOVERY_TIME"

# Step 3: Restore ClickHouse
echo "📊 Restoring ClickHouse..."
LATEST_CLICKHOUSE_BACKUP=$(aws s3 ls s3://llama-mapper-backups/clickhouse/ | \
    grep "clickhouse_full_" | \
    awk '{print $4}' | \
    sort | \
    tail -1 | \
    sed 's/clickhouse_full_//' | \
    sed 's/\///')
./scripts/restore-clickhouse.sh "$LATEST_CLICKHOUSE_BACKUP"

# Step 4: Restore Redis
echo "🔴 Restoring Redis..."
LATEST_REDIS_BACKUP=$(aws s3 ls s3://llama-mapper-backups/redis/ | \
    grep "redis_" | \
    awk '{print $4}' | \
    sort | \
    tail -1)
./scripts/restore-redis.sh "s3://llama-mapper-backups/redis/$LATEST_REDIS_BACKUP"

# Step 5: Restore S3 data
echo "☁️ Restoring S3 data..."
aws s3 sync s3://llama-mapper-backups-dr/ s3://llama-mapper-analysis-restored/

# Step 6: Update DNS
echo "🌐 Updating DNS..."
if [ "$ENVIRONMENT" = "production" ]; then
    # Update production DNS to point to new infrastructure
    aws route53 change-resource-record-sets \
        --hosted-zone-id Z1234567890 \
        --change-batch file://dns-changes-production.json
else
    # Update staging DNS
    aws route53 change-resource-record-sets \
        --hosted-zone-id Z0987654321 \
        --change-batch file://dns-changes-staging.json
fi

# Step 7: Verify system health
echo "🔍 Verifying system health..."
./scripts/verify-system-health.sh

echo "✅ Disaster recovery completed successfully!"
```

### Cross-Region Failover

#### Failover Script
```bash
#!/bin/bash
# scripts/cross-region-failover.sh

set -e

# Configuration
PRIMARY_REGION="us-west-2"
DR_REGION="us-east-1"
ENVIRONMENT="$1"

if [ -z "$ENVIRONMENT" ]; then
    echo "Usage: $0 <environment>"
    echo "Example: $0 production"
    exit 1
fi

echo "🔄 Starting cross-region failover to $DR_REGION"

# Step 1: Update Route 53 health checks
echo "🏥 Updating health checks..."
aws route53 change-resource-record-sets \
    --hosted-zone-id Z1234567890 \
    --change-batch file://health-check-failover.json

# Step 2: Promote DR database
echo "🐘 Promoting DR database..."
aws rds promote-read-replica \
    --db-instance-identifier llama-mapper-analysis-dr

# Step 3: Update application configuration
echo "⚙️ Updating application configuration..."
kubectl set env deployment/analysis-module \
    DATABASE_URL="postgresql://user:pass@dr-db.company.com:5432/llama_mapper_analysis" \
    REDIS_URL="redis://dr-redis.company.com:6379" \
    CLICKHOUSE_URL="http://dr-clickhouse.company.com:8123"

# Step 4: Scale up DR infrastructure
echo "📈 Scaling up DR infrastructure..."
kubectl scale deployment analysis-module --replicas=3

# Step 5: Verify failover
echo "🔍 Verifying failover..."
./scripts/verify-failover.sh

echo "✅ Cross-region failover completed successfully!"
```

---

## Testing and Validation

### Backup Validation

#### Automated Backup Testing
```bash
#!/bin/bash
# scripts/test-backups.sh

set -e

echo "🧪 Testing backup integrity..."

# Test PostgreSQL backup
echo "Testing PostgreSQL backup..."
LATEST_PG_BACKUP=$(aws s3 ls s3://llama-mapper-backups/postgresql/ | \
    grep "postgresql_full_" | \
    awk '{print $4}' | \
    sort | \
    tail -1)

aws s3 cp "s3://llama-mapper-backups/postgresql/$LATEST_PG_BACKUP" "/tmp/test_$LATEST_PG_BACKUP"

# Test backup integrity
pg_restore --list "/tmp/test_$LATEST_PG_BACKUP" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ PostgreSQL backup is valid"
else
    echo "❌ PostgreSQL backup is corrupted"
    exit 1
fi

# Test ClickHouse backup
echo "Testing ClickHouse backup..."
LATEST_CH_BACKUP=$(clickhouse-backup list --remote-storage s3 | \
    grep "clickhouse_full_" | \
    sort | \
    tail -1)

clickhouse-backup list --remote-storage s3 | grep "$LATEST_CH_BACKUP" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ ClickHouse backup is valid"
else
    echo "❌ ClickHouse backup is corrupted"
    exit 1
fi

# Test Redis backup
echo "Testing Redis backup..."
LATEST_REDIS_BACKUP=$(aws s3 ls s3://llama-mapper-backups/redis/ | \
    grep "redis_" | \
    awk '{print $4}' | \
    sort | \
    tail -1)

aws s3 cp "s3://llama-mapper-backups/redis/$LATEST_REDIS_BACKUP" "/tmp/test_$LATEST_REDIS_BACKUP"

# Test RDB file integrity
redis-check-rdb "/tmp/test_$LATEST_REDIS_BACKUP" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Redis backup is valid"
else
    echo "❌ Redis backup is corrupted"
    exit 1
fi

echo "🎉 All backup tests passed!"
```

### Recovery Testing

#### Monthly Recovery Drill
```bash
#!/bin/bash
# scripts/monthly-recovery-drill.sh

set -e

echo "🎯 Starting monthly recovery drill..."

# Create test environment
echo "Creating test environment..."
kubectl create namespace recovery-test
kubectl apply -f k8s/recovery-test/ -n recovery-test

# Test PostgreSQL recovery
echo "Testing PostgreSQL recovery..."
./scripts/restore-postgresql.sh postgresql_full_$(date -d '1 day ago' +%Y%m%d)_120000.dump

# Test ClickHouse recovery
echo "Testing ClickHouse recovery..."
LATEST_CH_BACKUP=$(clickhouse-backup list --remote-storage s3 | \
    grep "clickhouse_full_" | \
    sort | \
    tail -1)
./scripts/restore-clickhouse.sh "$LATEST_CH_BACKUP"

# Test Redis recovery
echo "Testing Redis recovery..."
LATEST_REDIS_BACKUP=$(aws s3 ls s3://llama-mapper-backups/redis/ | \
    grep "redis_" | \
    awk '{print $4}' | \
    sort | \
    tail -1)
./scripts/restore-redis.sh "s3://llama-mapper-backups/redis/$LATEST_REDIS_BACKUP"

# Run application tests
echo "Running application tests..."
kubectl run recovery-test --image=analysis-module:test -n recovery-test -- \
    python -m pytest tests/integration/test_recovery.py

# Clean up test environment
echo "Cleaning up test environment..."
kubectl delete namespace recovery-test

echo "✅ Monthly recovery drill completed successfully!"
```

---

## Monitoring and Alerting

### Backup Monitoring

#### Backup Success Monitoring
```yaml
# monitoring/backup-alerts.yml
groups:
- name: backup-alerts
  rules:
  - alert: BackupFailed
    expr: backup_success == 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Backup failed for {{ $labels.database }}"
      description: "Backup for {{ $labels.database }} has been failing for more than 5 minutes"

  - alert: BackupTooOld
    expr: time() - backup_last_success > 86400
    for: 1h
    labels:
      severity: warning
    annotations:
      summary: "Backup is too old for {{ $labels.database }}"
      description: "Last successful backup for {{ $labels.database }} was more than 24 hours ago"

  - alert: BackupSizeAnomaly
    expr: (backup_size - backup_size offset 1d) / backup_size offset 1d > 0.5
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Backup size anomaly for {{ $labels.database }}"
      description: "Backup size for {{ $labels.database }} increased by more than 50% compared to yesterday"
```

#### Backup Metrics Collection
```python
# monitoring/backup-metrics.py
import boto3
import psycopg2
import redis
import time
from prometheus_client import Gauge, Counter, push_to_gateway

# Metrics
backup_success = Gauge('backup_success', 'Backup success status', ['database'])
backup_last_success = Gauge('backup_last_success', 'Last successful backup timestamp', ['database'])
backup_size = Gauge('backup_size_bytes', 'Backup size in bytes', ['database'])
backup_duration = Gauge('backup_duration_seconds', 'Backup duration in seconds', ['database'])

def collect_backup_metrics():
    """Collect backup metrics from all databases."""
    
    # PostgreSQL metrics
    try:
        conn = psycopg2.connect("postgresql://user:pass@postgres.company.com:5432/llama_mapper_analysis")
        cursor = conn.cursor()
        cursor.execute("SELECT pg_database_size('llama_mapper_analysis')")
        db_size = cursor.fetchone()[0]
        
        # Check if backup exists
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(
            Bucket='llama-mapper-backups',
            Prefix='postgresql/postgresql_full_',
            MaxKeys=1
        )
        
        if response.get('Contents'):
            latest_backup = response['Contents'][0]
            backup_success.labels(database='postgresql').set(1)
            backup_last_success.labels(database='postgresql').set(
                latest_backup['LastModified'].timestamp()
            )
            backup_size.labels(database='postgresql').set(latest_backup['Size'])
        else:
            backup_success.labels(database='postgresql').set(0)
            
    except Exception as e:
        print(f"Error collecting PostgreSQL metrics: {e}")
        backup_success.labels(database='postgresql').set(0)
    
    # ClickHouse metrics
    try:
        # Similar logic for ClickHouse
        pass
    except Exception as e:
        print(f"Error collecting ClickHouse metrics: {e}")
        backup_success.labels(database='clickhouse').set(0)
    
    # Redis metrics
    try:
        # Similar logic for Redis
        pass
    except Exception as e:
        print(f"Error collecting Redis metrics: {e}")
        backup_success.labels(database='redis').set(0)

if __name__ == "__main__":
    while True:
        collect_backup_metrics()
        push_to_gateway('localhost:9091', job='backup-metrics')
        time.sleep(300)  # Collect every 5 minutes
```

---

## RPO/RTO Targets

### Recovery Point Objective (RPO)
- **PostgreSQL**: 15 minutes (WAL archiving every 5 minutes)
- **ClickHouse**: 4 hours (incremental backups)
- **Redis**: 1 hour (RDB saves)
- **S3**: 0 minutes (continuous replication)

### Recovery Time Objective (RTO)
- **PostgreSQL**: 30 minutes (full restore)
- **ClickHouse**: 45 minutes (full restore)
- **Redis**: 15 minutes (RDB restore)
- **S3**: 5 minutes (cross-region sync)

### Service Level Objectives (SLO)
- **Backup Success Rate**: 99.9%
- **Backup Completion Time**: < 2 hours
- **Recovery Success Rate**: 99.5%
- **Recovery Completion Time**: < 1 hour

---

## Emergency Contacts

### On-Call Rotation
- **Primary**: Database Administrator
- **Secondary**: Site Reliability Engineer
- **Manager**: Engineering Manager

### Escalation Procedures
1. **Level 1**: Database Administrator (0-15 minutes)
2. **Level 2**: Site Reliability Engineer (15-30 minutes)
3. **Level 3**: Engineering Manager (30+ minutes)

### Communication Channels
- **Slack**: #database-alerts
- **PagerDuty**: Database Recovery
- **Email**: db-recovery@company.com

---

*This document should be reviewed and updated quarterly or after any significant infrastructure changes.*
