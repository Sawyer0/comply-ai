# Weekly Evaluation System

A comprehensive weekly evaluation scheduling and reporting system for the Llama Mapper Analysis Module. This system provides automated quality evaluation scheduling, report generation, and notification capabilities.

## Architecture

The weekly evaluation system follows a clean architecture pattern with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Weekly Evaluation System                 │
├─────────────────────────────────────────────────────────────┤
│  CLI Commands  │  Domain Services  │  Infrastructure       │
│  - schedule    │  - WeeklyEvalSvc  │  - Storage Backend    │
│  - run         │  - QualityService │  - Report Generator   │
│  - list        │  - HealthService  │  - Alerting System    │
│  - cancel      │                   │  - Configuration      │
│  - status      │                   │                       │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Domain Services

- **WeeklyEvaluationService**: Core service for managing evaluation schedules and execution
- **QualityService**: Handles quality evaluation logic and metrics calculation
- **HealthService**: Provides health checks and system status

### Infrastructure

- **Storage Backend**: Pluggable storage implementations (File, Database, S3)
- **Report Generator**: Multi-format report generation (PDF, CSV, JSON)
- **Alerting System**: Multi-channel notification system
- **Configuration**: Environment and file-based configuration management

### CLI Commands

- **schedule**: Create and manage evaluation schedules
- **run**: Execute scheduled evaluations
- **list**: View scheduled evaluations
- **cancel**: Remove evaluation schedules
- **status**: Check evaluation status and statistics

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
- CSV exports for data analysis
- JSON reports for API integration
- Trend analysis and historical comparison
- Customizable report templates

### 🔔 Notifications
- Email notifications for evaluation results
- Slack integration for team notifications
- Configurable alert thresholds
- Multi-channel notification support

### 💾 Storage
- Pluggable storage backends
- File-based storage for development
- Database storage for production
- S3 storage for cloud deployments
- Configurable retention policies

## Quick Start

### 1. Schedule an Evaluation

```bash
# Basic scheduling (Monday 9 AM)
mapper weekly-eval schedule --tenant-id "tenant-123"

# Custom schedule with notifications
mapper weekly-eval schedule \
  --tenant-id "tenant-123" \
  --cron-schedule "0 10 * * 2" \
  --recipients "admin@example.com,team@example.com"
```

### 2. Run Evaluations

```bash
# Run specific evaluation immediately
mapper weekly-eval run --schedule-id "schedule-123" --force

# List all scheduled evaluations
mapper weekly-eval list

# Check evaluation status
mapper weekly-eval status --schedule-id "schedule-123"
```

### 3. Manage Schedules

```bash
# Cancel a schedule
mapper weekly-eval cancel --schedule-id "schedule-123"

# List evaluations for specific tenant
mapper weekly-eval list --tenant-id "tenant-123"
```

## Configuration

### Environment Variables

```bash
# Basic settings
LLAMA_MAPPER_WEEKLY_EVALUATIONS_ENABLED=true
LLAMA_MAPPER_DEFAULT_WEEKLY_SCHEDULE="0 9 * * 1"
LLAMA_MAPPER_EVALUATION_PERIOD_DAYS=7

# Quality thresholds
LLAMA_MAPPER_SCHEMA_VALID_THRESHOLD=0.98
LLAMA_MAPPER_RUBRIC_SCORE_THRESHOLD=0.8
LLAMA_MAPPER_OPA_COMPILE_THRESHOLD=0.95
LLAMA_MAPPER_EVIDENCE_ACCURACY_THRESHOLD=0.85

# Notifications
LLAMA_MAPPER_NOTIFICATION_EMAIL="admin@example.com,team@example.com"
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

# Storage
LLAMA_MAPPER_STORAGE_BACKEND="file"
LLAMA_MAPPER_STORAGE_DIR="/tmp/evaluations"
DATABASE_URL="postgresql://user:pass@localhost/db"
S3_BUCKET="llama-mapper-reports"
```

### Configuration Files

Create a JSON or YAML configuration file:

```json
{
  "enabled": true,
  "default_schedule": "0 9 * * 1",
  "evaluation_period_days": 7,
  "thresholds": {
    "schema_valid_rate": 0.98,
    "rubric_score": 0.8,
    "opa_compile_success_rate": 0.95,
    "evidence_accuracy": 0.85
  },
  "notifications": {
    "enabled": true,
    "email_recipients": ["admin@example.com"],
    "slack_webhook_url": "https://hooks.slack.com/services/..."
  },
  "storage": {
    "backend_type": "file",
    "storage_dir": "/tmp/evaluations",
    "retention_days": 90
  }
}
```

## Deployment

### Kubernetes Deployment

```yaml
# values.yaml
weeklyEvaluations:
  enabled: true
  schedule: "0 9 * * 1"  # Every Monday at 9 AM UTC
  redisUrl: "redis://redis:6379"
  databaseUrl: "postgresql://user:pass@postgres:5432/db"
  s3Bucket: "llama-mapper-reports"
  notificationEmail: "admin@example.com"
  resources:
    requests:
      cpu: "250m"
      memory: "512Mi"
    limits:
      cpu: "1"
      memory: "1Gi"
```

Deploy with Helm:

```bash
helm upgrade llama-mapper ./charts/llama-mapper \
  --set weeklyEvaluations.enabled=true \
  --set weeklyEvaluations.schedule="0 9 * * 1" \
  --set weeklyEvaluations.notificationEmail="admin@example.com"
```

## Development

### Running Tests

```bash
# Unit tests
pytest tests/unit/test_weekly_evaluation_service.py -v
pytest tests/unit/test_weekly_evaluation_cli.py -v
pytest tests/unit/test_evaluation_config.py -v

# Integration tests
pytest tests/integration/test_weekly_evaluation_integration.py -v

# All tests
pytest tests/ -v
```

### Code Quality

```bash
# Linting
flake8 src/llama_mapper/analysis/
black src/llama_mapper/analysis/
isort src/llama_mapper/analysis/

# Type checking
mypy src/llama_mapper/analysis/
```

### Adding New Features

1. **Add new domain interfaces** in `domain/interfaces.py`
2. **Implement infrastructure components** in `infrastructure/`
3. **Add CLI commands** in `cli/commands/weekly_evaluation.py`
4. **Update configuration** in `config/evaluation_config.py`
5. **Add tests** in `tests/unit/` and `tests/integration/`

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

# Get statistics
stats = service.get_service_statistics()
```

### Configuration

```python
from llama_mapper.analysis.config.evaluation_config import (
    WeeklyEvaluationConfig,
    get_evaluation_config,
    validate_evaluation_config
)

# Load from environment
config = get_evaluation_config()

# Load from file
config = WeeklyEvaluationConfig.from_file("config.yaml")

# Validate configuration
validate_evaluation_config(config)

# Save configuration
config.to_file("output.json", format="json")
```

### Storage Backend

```python
from llama_mapper.analysis.infrastructure.storage_backend import create_storage_backend

# File storage
backend = create_storage_backend("file", storage_dir="/tmp/evaluations")

# Database storage
backend = create_storage_backend("database", database_url="postgresql://...")

# S3 storage
backend = create_storage_backend("s3", s3_bucket="my-bucket", database_url="postgresql://...")
```

## Monitoring

### Service Statistics

The service tracks comprehensive statistics:

```python
stats = service.get_service_statistics()
# {
#   "service_uptime_seconds": 3600.0,
#   "schedules_created": 5,
#   "evaluations_run": 10,
#   "evaluations_failed": 1,
#   "reports_generated": 10,
#   "notifications_sent": 20,
#   "last_evaluation": "2024-01-15T09:00:00Z",
#   "active_schedules": 4,
#   "success_rate": 0.909
# }
```

### Logging

The system provides structured logging with context:

```python
logger.info(
    "Completed scheduled evaluation for tenant tenant-123",
    extra={
        "tenant_id": "tenant-123",
        "schedule_id": "schedule-456",
        "status": "completed",
        "data_points": 100,
        "quality_score": 0.85,
        "report_path": "/tmp/report.pdf"
    }
)
```

### Health Checks

Monitor service health:

```python
# Check service statistics
stats = service.get_service_statistics()
if stats["success_rate"] < 0.9:
    # Alert on low success rate
    pass

# Check active schedules
schedules = await service.list_scheduled_evaluations()
if len(schedules) == 0:
    # Alert on no active schedules
    pass
```

## Troubleshooting

### Common Issues

1. **Evaluation not running**
   - Check schedule status: `mapper weekly-eval status --schedule-id <id>`
   - Verify cron expression: `mapper weekly-eval list`
   - Check Kubernetes CronJob: `kubectl get cronjobs -n llama-mapper`

2. **Report generation failures**
   - Check storage configuration
   - Verify file permissions
   - Check S3 credentials and permissions

3. **Notification issues**
   - Test email configuration
   - Verify Slack webhook URL
   - Check notification settings

### Debug Mode

Enable debug logging:

```bash
export LLAMA_MAPPER_LOG_LEVEL=DEBUG
mapper weekly-eval run --schedule-id <id> --force
```

### Performance Issues

- Monitor resource usage in Kubernetes
- Check evaluation timeout settings
- Review concurrent evaluation limits
- Analyze storage backend performance

## Contributing

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure all tests pass before submitting PR
5. Follow the established error handling patterns

## License

This code is part of the Llama Mapper project and follows the same license terms.
