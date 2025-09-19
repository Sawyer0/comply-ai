# Backup, Restore, and Disaster Recovery Runbook

Scope
- Datastores: PostgreSQL or ClickHouse (hot data); S3 (immutable archive, WORM)
- Environments: dev, stage, prod

Goals
- Nightly backups of hot data
- Immutable S3 object storage with Object Lock (WORM)
- Documented restore drills and RTO goals

1) Backups (Kubernetes CronJobs)
- PostgreSQL
  - Enable in Helm values:
    - backups.enabled=true
    - backups.postgresql.enabled=true
    - backups.postgresql.passwordSecretRef=<k8s-secret-name>
    - backups.s3.bucket=<bucket>
    - backups.s3.awsAccessKeyIdSecretRef=<aws-secret>
    - backups.s3.awsSecretAccessKeySecretRef=<aws-secret>
  - Schedule: values.backups.postgresql.schedule (default 02:00 UTC)
  - Output layout: s3://<bucket>/<prefix>/postgresql/<release>/<timestamp>.sql.gz
- ClickHouse
  - Enable in Helm values:
    - backups.enabled=true
    - backups.clickhouse.enabled=true
    - backups.clickhouse.passwordSecretRef=<k8s-secret-name>
    - backups.s3.bucket=<bucket>
    - backups.s3.awsAccessKeyIdSecretRef=<aws-secret>
    - backups.s3.awsSecretAccessKeySecretRef=<aws-secret>
  - Schedule: values.backups.clickhouse.schedule (default 02:30 UTC)
  - Output layout: s3://<bucket>/<prefix>/clickhouse/<release>/<timestamp>/
  - Note: CronJob uses ClickHouse BACKUP TO S3; requires server-side permissions and version supporting BACKUP/RESTORE.

2) S3 WORM and lifecycle
- Use Terraform in infra/terraform/s3_worm_bucket to create bucket with Object Lock (governance-mode) and lifecycle rules.
- Default retention: 7 years (configurable)

3) Restore procedures (Kubernetes Jobs)
- PostgreSQL restore
  - Set values:
    - restore.enabled=true
    - restore.type=postgresql
    - restore.snapshot=<backups/llama-mapper/postgresql/<release>/<timestamp>.sql.gz>
    - restore.postgresql.passwordSecretRef=<secret>
  - Apply chart with restore.* values; Job runs once and completes.
- ClickHouse restore
  - Set values:
    - restore.enabled=true
    - restore.type=clickhouse
    - restore.snapshot=<backups/llama-mapper/clickhouse/<release>/<timestamp>>
    - restore.clickhouse.passwordSecretRef=<secret>

4) Integrity checks
- After backup: verify object presence and non-zero size on S3; optionally compute checksum and store alongside.
- After restore: run validation query checks (row counts, sample data integrity) and application smoke tests (/health, /map on known inputs).

5) RTO drills
- Quarterly drill: simulate loss of hot data, then restore from latest snapshot, measure RTO from incident start to API healthy.
- Goal: RTO <= 30 minutes for PostgreSQL, <= 45 minutes for ClickHouse (tune per environment).

6) Permissions
- Principle of least privilege:
  - K8s backup jobs: write-only to backups prefix, no delete.
  - Restore jobs: read-only from backups prefix.
  - S3 bucket: Object Lock enabled, versioning ON.

7) Verification commands (examples)
- S3 listing
  - aws s3 ls s3://<bucket>/<prefix>/postgresql/<release>/ --recursive --human-readable --summarize
- Postgres restore smoke test
  - psql -h <host> -U <user> -d <db> -c "SELECT COUNT(*) FROM storage_records;"
- ClickHouse restore smoke test
  - clickhouse-client --host <host> --query "SELECT COUNT() FROM storage_records"

8) Notes
- Ensure backup images include pg_dump/psql and aws-cli; for ClickHouse, include clickhouse-client.
- For production, add checksum/signing of backup artifacts and write integrity logs to audit trail.
