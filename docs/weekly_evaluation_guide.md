# Weekly Evaluation Scheduling and Reporting Guide

This guide covers the weekly evaluation scheduling and reporting system for the Llama Mapper Analysis Module. The system provides automated quality evaluation scheduling, report generation, and notification capabilities.

## Overview

The weekly evaluation system consists of several key components:

- **WeeklyEvaluationService**: Core service for managing evaluation schedules and execution
- **CLI Commands**: Command-line interface for managing evaluations
- **Kubernetes CronJob**: Automated scheduling via Kubernetes
- **Quality Alerting**: Integration with the existing quality alerting system
- **Report Generation**: PDF report generation with quality metrics

## Features

### 🔄 Automated Scheduling
- Cron-based scheduling with configurable intervals
- Support for multiple tenants with independent schedules
- Automatic execution via Kubernetes CronJob
- Schedule management (create, update, cancel)

### 📊 Quality Evaluation
- Integration with existing quality evaluation system
- Schema validation rate monitoring
- Rubric scoring and OPA compilation success tracking
- Evidence accuracy assessment
- Drift detection over time

### 📄 Report Generation
- PDF reports with comprehensive quality metrics
- Trend analysis and historical comparison
- Customizable report templates
- Automated report storage and archival

### 🔔 Notifications
- Email notifications for evaluation results
- Slack integration for team notifications
- Configurable alert thresholds
- Multi-channel notification support

## Configuration

### Environment Variables

```bash
# Weekly evaluation settings
LLAMA_MAPPER_WEEKLY_EVALUATIONS_ENABLED=true
LLAMA_MAPPER_DEFAULT_WEEKLY_SCHEDULE="0 9 * * 1"  # Monday 9 AM
LLAMA_MAPPER_WEEKLY_EVALUATION_RETENTION_DAYS=90
LLAMA_MAPPER_WEEKLY_EVALUATION_NOTIFICATIONS=true

# Storage configuration
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/db
S3_BUCKET=llama-mapper-reports
AWS_REGION=us-east-1

# Notification settings
NOTIFICATION_EMAIL=admin@example.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Kubernetes Configuration

The weekly evaluation system can be deployed as a Kubernetes CronJob:

```yaml
# values.yaml
weeklyEvaluations:
  enabled: true
  schedule: "0 9 * * 1"  # Every Monday at 9 AM UTC
  redisUrl: "redis://redis:6379"
  databaseUrl: "postgresql://user:pass@postgres:5432/db"
  s3Bucket: "llama-mapper-reports"
  awsRegion: "us-east-1"
  notificationEmail: "admin@example.com"
  resources:
    requests:
      cpu: "250m"
      memory: "512Mi"
    limits:
      cpu: "1"
      memory: "1Gi"
```

## CLI Usage

### Schedule a Weekly Evaluation

```bash
# Basic scheduling (Monday 9 AM)
mapper weekly-eval schedule --tenant-id "tenant-123"

# Custom schedule (Tuesday 10 AM)
mapper weekly-eval schedule \
  --tenant-id "tenant-123" \
  --cron-schedule "0 10 * * 2" \
  --recipients "admin@example.com,team@example.com"

# With configuration file
mapper weekly-eval schedule \
  --tenant-id "tenant-123" \
  --config-file "evaluation-config.json"
```

### Manage Schedules

```bash
# List all scheduled evaluations
mapper weekly-eval list

# List evaluations for specific tenant
mapper weekly-eval list --tenant-id "tenant-123"

# Show only active schedules
mapper weekly-eval list --active-only

# Get status of specific schedule
mapper weekly-eval status --schedule-id "schedule-123"

# Cancel a schedule
mapper weekly-eval cancel --schedule-id "schedule-123"
```

### Run Evaluations

```bash
# Run specific evaluation immediately
mapper weekly-eval run --schedule-id "schedule-123" --force

# Dry run to see what would be executed
mapper weekly-eval run --dry-run
```

## Configuration Files

### Evaluation Configuration

Create a JSON configuration file for custom evaluation settings:

```json
{
  "thresholds": {
    "schema_valid_rate": 0.98,
    "rubric_score": 0.8,
    "opa_compile_success_rate": 0.95,
    "evidence_accuracy": 0.85
  },
  "evaluation_period_days": 7,
  "include_detailed_metrics": true,
  "notification_channels": ["email", "slack"],
  "report_formats": ["pdf", "json"],
  "retention_days": 90
}
```

### Cron Schedule Examples

```bash
# Every Monday at 9 AM
"0 9 * * 1"

# Every weekday at 8 AM
"0 8 * * 1-5"

# Every Sunday at midnight
"0 0 * * 0"

# Every 6 hours
"0 */6 * * *"

# First day of every month at 10 AM
"0 10 1 * *"
```

## Report Structure

Weekly evaluation reports include:

### Executive Summary
- Overall quality score
- Key metrics overview
- Trend analysis
- Critical issues summary

### Detailed Metrics
- Schema validation rate
- Rubric scoring breakdown
- OPA compilation success rate
- Evidence accuracy analysis
- Individual example scores

### Historical Comparison
- Week-over-week trends
- Monthly averages
- Performance benchmarks
- Improvement recommendations

### Technical Details
- Evaluation configuration
- Data sources and coverage
- Model performance metrics
- System health indicators

## Monitoring and Alerting

### Quality Thresholds

The system monitors several quality metrics:

- **Schema Validation Rate**: Minimum 98%
- **Rubric Score**: Minimum 0.8
- **OPA Compilation Success Rate**: Minimum 95%
- **Evidence Accuracy**: Minimum 0.85

### Alert Types

- **Warning**: Metrics approaching thresholds
- **Critical**: Metrics below thresholds
- **Trend**: Significant degradation over time
- **System**: Evaluation execution failures

### Notification Channels

- **Email**: Detailed reports with attachments
- **Slack**: Summary notifications with links
- **Webhook**: Custom integrations
- **Logging**: Structured log events

## Troubleshooting

### Common Issues

#### Evaluation Not Running
```bash
# Check schedule status
mapper weekly-eval status --schedule-id "schedule-123"

# Verify cron expression
mapper weekly-eval list --tenant-id "tenant-123"

# Check Kubernetes CronJob
kubectl get cronjobs -n llama-mapper
kubectl describe cronjob llama-mapper-weekly-evaluation
```

#### Report Generation Failures
```bash
# Check storage configuration
kubectl get configmap -n llama-mapper
kubectl logs -n llama-mapper deployment/llama-mapper

# Verify S3 permissions
aws s3 ls s3://llama-mapper-reports/
```

#### Notification Issues
```bash
# Test email configuration
mapper weekly-eval run --schedule-id "schedule-123" --force

# Check Slack webhook
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Test notification"}' \
  $SLACK_WEBHOOK_URL
```

### Log Analysis

```bash
# View evaluation logs
kubectl logs -n llama-mapper cronjob/llama-mapper-weekly-evaluation

# Filter for specific tenant
kubectl logs -n llama-mapper cronjob/llama-mapper-weekly-evaluation | grep "tenant-123"

# Check for errors
kubectl logs -n llama-mapper cronjob/llama-mapper-weekly-evaluation | grep ERROR
```

## Best Practices

### Scheduling
- Schedule evaluations during low-traffic periods
- Avoid overlapping evaluations for the same tenant
- Use appropriate timeouts for long-running evaluations
- Monitor resource usage and adjust limits as needed

### Configuration
- Use environment-specific configurations
- Regularly review and update quality thresholds
- Maintain separate configurations for different tenants
- Document custom evaluation criteria

### Monitoring
- Set up alerts for evaluation failures
- Monitor report generation performance
- Track notification delivery success rates
- Regular review of quality trends

### Security
- Use secure storage for reports (S3 with encryption)
- Implement proper access controls for notifications
- Regularly rotate API keys and credentials
- Audit evaluation access and permissions

## API Reference

### WeeklyEvaluationService

```python
from llama_mapper.analysis.domain.services import WeeklyEvaluationService

# Schedule evaluation
schedule_id = await service.schedule_weekly_evaluation(
    tenant_id="tenant-123",
    cron_schedule="0 9 * * 1",
    report_recipients=["admin@example.com"],
    evaluation_config={"threshold": 0.8}
)

# Run evaluation
result = await service.run_scheduled_evaluation(schedule_id)

# List schedules
schedules = await service.list_scheduled_evaluations(tenant_id="tenant-123")

# Cancel schedule
success = await service.cancel_scheduled_evaluation(schedule_id)
```

### Quality Metrics

```python
from llama_mapper.analysis.domain.entities import QualityMetrics

metrics = QualityMetrics(
    total_examples=100,
    schema_valid_rate=0.98,
    rubric_score=0.85,
    opa_compile_success_rate=0.96,
    evidence_accuracy=0.82,
    individual_rubric_scores=[0.8, 0.9, 0.85, ...]
)
```

## Migration Guide

### From Manual Evaluations

1. **Export existing evaluation data**
   ```bash
   # Export current quality metrics
   mapper quality export --output quality-baseline.json
   ```

2. **Create initial schedules**
   ```bash
   # Schedule for each tenant
   mapper weekly-eval schedule --tenant-id "tenant-1"
   mapper weekly-eval schedule --tenant-id "tenant-2"
   ```

3. **Configure notifications**
   ```bash
   # Update notification settings
   kubectl patch configmap llama-mapper-config \
     --patch '{"data":{"NOTIFICATION_EMAIL":"admin@example.com"}}'
   ```

4. **Deploy CronJob**
   ```bash
   # Enable weekly evaluations
   helm upgrade llama-mapper ./charts/llama-mapper \
     --set weeklyEvaluations.enabled=true
   ```

### Upgrading Existing Deployments

1. **Backup current configuration**
   ```bash
   kubectl get configmap llama-mapper-config -o yaml > config-backup.yaml
   ```

2. **Update Helm values**
   ```bash
   helm upgrade llama-mapper ./charts/llama-mapper \
     --set weeklyEvaluations.enabled=true \
     --set weeklyEvaluations.schedule="0 9 * * 1"
   ```

3. **Verify deployment**
   ```bash
   kubectl get cronjobs -n llama-mapper
   mapper weekly-eval list
   ```

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review logs for error messages
3. Verify configuration settings
4. Test with dry-run mode
5. Contact the development team with specific error details

## Changelog

### v1.0.0
- Initial release of weekly evaluation system
- Basic scheduling and report generation
- CLI commands for management
- Kubernetes CronJob support
- Email and Slack notifications
