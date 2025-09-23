# Quality Alerting System

A comprehensive quality monitoring and alerting system for the Llama Mapper Analysis Module. This system provides real-time quality monitoring, degradation detection, and multi-channel alerting capabilities.

## Features

### 🔍 Quality Monitoring
- **Real-time Metrics Tracking**: Monitor quality metrics with configurable retention
- **Statistical Analysis**: Calculate mean, standard deviation, percentiles, and trends
- **Thread-safe Storage**: Concurrent metric recording and retrieval
- **Performance Monitoring**: Internal performance metrics and error tracking

### 🚨 Degradation Detection
- **Threshold-based Detection**: Configurable warning and critical thresholds
- **Anomaly Detection**: Statistical outlier detection using z-scores
- **Trend Analysis**: Linear regression for trend detection
- **Multiple Detection Types**: Sudden drops, gradual decline, threshold breaches

### 📢 Alerting System
- **Multi-channel Notifications**: Email, Slack, webhooks, and logging
- **Alert Lifecycle Management**: Active → Acknowledged → Resolved workflow
- **Deduplication**: Prevent alert spam with configurable windows
- **Severity Levels**: Low, Medium, High, Critical alert classification

### ⚙️ Configuration Management
- **Flexible Thresholds**: Per-metric threshold configuration
- **Environment-based Settings**: Support for environment variables
- **Validation**: Comprehensive configuration validation
- **Hot Reloading**: Runtime configuration updates

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Quality Alerting System                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Quality Monitor │  │ Degradation     │  │ Alert        │ │
│  │                 │  │ Detector        │  │ Manager      │ │
│  │ • Metrics Store │  │ • Thresholds    │  │ • Lifecycle  │ │
│  │ • Statistics    │  │ • Anomalies     │  │ • Routing    │ │
│  │ • Trends        │  │ • Trends        │  │ • History    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Email Handler   │  │ Slack Handler   │  │ Webhook      │ │
│  │                 │  │                 │  │ Handler      │ │
│  │ • SMTP          │  │ • Webhooks      │  │ • HTTP       │ │
│  │ • Templates     │  │ • Formatting    │  │ • JSON       │ │
│  │ • Attachments   │  │ • Channels      │  │ • Headers    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from src.llama_mapper.analysis.quality import (
    QualityAlertingSystem, QualityMetric, QualityMetricType
)

# Initialize the system
system = QualityAlertingSystem()

# Start monitoring
system.start_monitoring()

# Record a quality metric
metric = QualityMetric(
    metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
    value=0.95,
    timestamp=datetime.now(),
    labels={"service": "analysis", "version": "1.0"}
)

system.process_metric(metric)

# Stop monitoring
system.stop_monitoring()
```

### Configuration

```python
from src.llama_mapper.analysis.quality import QualityAlertingSettings

# Create settings
settings = QualityAlertingSettings(
    monitoring_interval_seconds=60,
    max_metrics_per_type=10000,
    alert_retention_days=30,
    
    # Email configuration
    email_enabled=True,
    email_smtp_server="smtp.gmail.com",
    email_smtp_port=587,
    email_username="alerts@company.com",
    email_password="app-password",
    email_from="alerts@company.com",
    email_to=["admin@company.com", "devops@company.com"],
    
    # Slack configuration
    slack_enabled=True,
    slack_webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    slack_channel="#alerts",
    
    # Threshold configuration
    schema_validation_warning=0.95,
    schema_validation_critical=0.90,
    template_fallback_warning=0.20,
    template_fallback_critical=0.30
)

# Convert to system configuration
config = settings.to_config()
```

### Adding Alert Handlers

```python
# Add email alerts
system.add_email_handler(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    username="alerts@company.com",
    password="app-password",
    from_email="alerts@company.com",
    to_emails=["admin@company.com"]
)

# Add Slack alerts
system.add_slack_handler(
    webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    channel="#alerts"
)

# Add webhook alerts
system.add_webhook_handler(
    webhook_url="https://your-monitoring-system.com/webhook",
    headers={"Authorization": "Bearer your-token"}
)
```

## Quality Metrics

### Supported Metric Types

| Metric Type | Description | Range | Unit |
|-------------|-------------|-------|------|
| `SCHEMA_VALIDATION_RATE` | JSON schema validation success rate | 0.0-1.0 | Rate |
| `TEMPLATE_FALLBACK_RATE` | LLM template fallback usage rate | 0.0-1.0 | Rate |
| `OPA_COMPILATION_SUCCESS_RATE` | OPA policy compilation success rate | 0.0-1.0 | Rate |
| `CONFIDENCE_SCORE` | Analysis confidence score | 0.0-1.0 | Score |
| `RESPONSE_TIME` | API response time | >0 | Seconds |
| `ERROR_RATE` | Error occurrence rate | 0.0-1.0 | Rate |
| `THROUGHPUT` | Request processing rate | >0 | Requests/second |
| `CUSTOM_METRIC` | User-defined metrics | Any | Custom |

### Metric Recording

```python
# Record different types of metrics
metrics = [
    QualityMetric(
        metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
        value=0.98,
        timestamp=datetime.now(),
        labels={"service": "analysis", "endpoint": "/analyze"}
    ),
    QualityMetric(
        metric_type=QualityMetricType.RESPONSE_TIME,
        value=1.2,
        timestamp=datetime.now(),
        labels={"service": "analysis", "endpoint": "/analyze"}
    ),
    QualityMetric(
        metric_type=QualityMetricType.ERROR_RATE,
        value=0.02,
        timestamp=datetime.now(),
        labels={"service": "analysis", "error_type": "validation"}
    )
]

for metric in metrics:
    system.process_metric(metric)
```

## Degradation Detection

### Detection Algorithms

#### 1. Threshold-based Detection
```python
from src.llama_mapper.analysis.quality import QualityThreshold

threshold = QualityThreshold(
    metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
    warning_threshold=0.95,    # Warning below 95%
    critical_threshold=0.90,   # Critical below 90%
    min_samples=10,            # Need 10 samples
    time_window_minutes=60,    # Within 60 minutes
    enabled=True
)

system.add_threshold(threshold)
```

#### 2. Anomaly Detection
```python
# Configure anomaly sensitivity
system = QualityAlertingSystem(
    anomaly_sensitivity=2.0,  # 2 standard deviations
    trend_window_minutes=30
)

# Anomalies are automatically detected
# when metrics deviate significantly from normal patterns
```

#### 3. Trend Analysis
```python
# Get trend information
trends = system.quality_monitor.get_metric_trends(
    QualityMetricType.SCHEMA_VALIDATION_RATE,
    time_window_minutes=60
)

print(f"Trend: {trends['trend']}")        # increasing/decreasing/stable
print(f"Slope: {trends['slope']:.4f}")    # Trend strength
print(f"R²: {trends['r_squared']:.3f}")   # Correlation quality
```

### Degradation Types

- **`SUDDEN_DROP`**: Immediate quality decrease (>20%)
- **`GRADUAL_DECLINE`**: Long-term quality degradation
- **`THRESHOLD_BREACH`**: Crossing configured thresholds
- **`ANOMALY`**: Statistical outliers
- **`TREND_REVERSAL`**: Quality pattern changes

## Alert Management

### Alert Lifecycle

```
┌─────────┐    acknowledge    ┌──────────────┐    resolve    ┌──────────┐
│ ACTIVE  │ ────────────────► │ ACKNOWLEDGED │ ────────────► │ RESOLVED │
└─────────┘                   └──────────────┘               └──────────┘
     │                              │                              │
     │ suppress                     │ suppress                     │
     ▼                              ▼                              ▼
┌──────────┐                   ┌──────────┐                   ┌──────────┐
│SUPPRESSED│                   │SUPPRESSED│                   │SUPPRESSED│
└──────────┘                   └──────────┘                   └──────────┘
```

### Alert Operations

```python
# Get active alerts
active_alerts = system.alert_manager.get_active_alerts()

# Acknowledge an alert
system.alert_manager.acknowledge_alert(alert_id, "admin_user")

# Resolve an alert
system.alert_manager.resolve_alert(alert_id, "admin_user")

# Suppress an alert
system.alert_manager.suppress_alert(alert_id, "False positive")

# Get alert history
history = system.alert_manager.get_alert_history(
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now(),
    severity=AlertSeverity.HIGH
)
```

### Alert Severity Levels

- **`LOW`**: Minor quality issues, informational
- **`MEDIUM`**: Moderate quality degradation, attention needed
- **`HIGH`**: Significant quality problems, immediate action
- **`CRITICAL`**: Severe quality failures, urgent response

## Alert Handlers

### Email Handler

```python
from src.llama_mapper.analysis.quality import EmailAlertHandler

email_handler = EmailAlertHandler(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    username="alerts@company.com",
    password="app-password",
    from_email="alerts@company.com",
    to_emails=["admin@company.com", "devops@company.com"],
    use_tls=True
)

system.alert_handlers.append(email_handler)
```

### Slack Handler

```python
from src.llama_mapper.analysis.quality import SlackAlertHandler

slack_handler = SlackAlertHandler(
    webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    channel="#alerts",
    username="Quality Monitor",
    icon_emoji=":warning:"
)

system.alert_handlers.append(slack_handler)
```

### Webhook Handler

```python
from src.llama_mapper.analysis.quality import WebhookAlertHandler

webhook_handler = WebhookAlertHandler(
    webhook_url="https://your-monitoring-system.com/webhook",
    headers={"Authorization": "Bearer your-token"},
    timeout=10
)

system.alert_handlers.append(webhook_handler)
```

### Composite Handler

```python
from src.llama_mapper.analysis.quality import CompositeAlertHandler

# Route alerts to multiple handlers
composite_handler = CompositeAlertHandler([
    email_handler,
    slack_handler,
    webhook_handler
])

system.alert_handlers = [composite_handler]
```

## Dashboard and Monitoring

### System Status

```python
# Get comprehensive system status
status = system.get_system_status()

print(f"Monitoring Active: {status['monitoring_active']}")
print(f"Thresholds Configured: {status['thresholds_configured']}")
print(f"Alert Handlers: {status['alert_handlers']}")
print(f"Metrics Processed: {status['statistics']['metrics_processed']}")
print(f"Degradations Detected: {status['statistics']['degradations_detected']}")
print(f"Alerts Created: {status['statistics']['alerts_created']}")
```

### Quality Dashboard Data

```python
# Get data for quality dashboard
dashboard_data = system.get_quality_dashboard_data()

# Current metrics
current_metrics = dashboard_data["current_metrics"]
for metric_type, value in current_metrics.items():
    print(f"{metric_type}: {value:.3f}")

# Trends
trends = dashboard_data["trends"]
for metric_type, trend in trends.items():
    print(f"{metric_type}: {trend['trend']} (slope: {trend['slope']:.4f})")

# Active alerts
active_alerts = dashboard_data["active_alerts"]
for alert in active_alerts:
    print(f"Alert: {alert['title']} ({alert['severity']})")
```

### Performance Statistics

```python
# Get performance statistics
perf_stats = system.quality_monitor.get_performance_statistics()

print(f"Metrics Recorded: {perf_stats['metrics_recorded']}")
print(f"Metrics Retrieved: {perf_stats['metrics_retrieved']}")
print(f"Cleanup Operations: {perf_stats['cleanup_operations']}")
print(f"Errors: {perf_stats['errors']}")
if perf_stats['last_error']:
    print(f"Last Error: {perf_stats['last_error']}")
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest src/llama_mapper/analysis/tests/

# Run specific test types
python -m pytest src/llama_mapper/analysis/tests/unit/
python -m pytest src/llama_mapper/analysis/tests/integration/
python -m pytest src/llama_mapper/analysis/tests/performance/

# Run with coverage
python -m pytest --cov=src.llama_mapper.analysis.quality --cov-report=html

# Run performance tests
python -m pytest -m performance src/llama_mapper/analysis/tests/
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end system testing
- **Performance Tests**: Load and scalability testing
- **Smoke Tests**: Basic functionality verification

### Test Runner

```bash
# Use the comprehensive test runner
python src/llama_mapper/analysis/tests/run_quality_tests.py --type all --coverage --report

# Run specific test types
python src/llama_mapper/analysis/tests/run_quality_tests.py --type unit
python src/llama_mapper/analysis/tests/run_quality_tests.py --type integration
python src/llama_mapper/analysis/tests/run_quality_tests.py --type performance

# Run with linting
python src/llama_mapper/analysis/tests/run_quality_tests.py --linting --report
```

## Configuration

### Environment Variables

```bash
# Quality monitoring settings
export LLAMA_MAPPER_QUALITY_MONITORING_ENABLED=true
export LLAMA_MAPPER_QUALITY_MAX_METRICS=10000
export LLAMA_MAPPER_QUALITY_RETENTION_HOURS=24
export LLAMA_MAPPER_QUALITY_CLEANUP_INTERVAL=60

# Email settings
export LLAMA_MAPPER_QUALITY_EMAIL_ENABLED=true
export LLAMA_MAPPER_QUALITY_EMAIL_SMTP_SERVER=smtp.gmail.com
export LLAMA_MAPPER_QUALITY_EMAIL_SMTP_PORT=587
export LLAMA_MAPPER_QUALITY_EMAIL_USERNAME=alerts@company.com
export LLAMA_MAPPER_QUALITY_EMAIL_PASSWORD=app-password
export LLAMA_MAPPER_QUALITY_EMAIL_FROM=alerts@company.com
export LLAMA_MAPPER_QUALITY_EMAIL_TO=admin@company.com,devops@company.com

# Slack settings
export LLAMA_MAPPER_QUALITY_SLACK_ENABLED=true
export LLAMA_MAPPER_QUALITY_SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
export LLAMA_MAPPER_QUALITY_SLACK_CHANNEL=#alerts

# Threshold settings
export LLAMA_MAPPER_QUALITY_SCHEMA_VALIDATION_WARNING=0.95
export LLAMA_MAPPER_QUALITY_SCHEMA_VALIDATION_CRITICAL=0.90
export LLAMA_MAPPER_QUALITY_TEMPLATE_FALLBACK_WARNING=0.20
export LLAMA_MAPPER_QUALITY_TEMPLATE_FALLBACK_CRITICAL=0.30
```

### Configuration Files

```yaml
# quality_config.yaml
quality_alerting:
  monitoring_interval_seconds: 60
  max_metrics_per_type: 10000
  alert_retention_days: 30
  deduplication_window_minutes: 15
  
  email:
    enabled: true
    smtp_server: smtp.gmail.com
    smtp_port: 587
    username: alerts@company.com
    password: app-password
    from_email: alerts@company.com
    to_emails:
      - admin@company.com
      - devops@company.com
  
  slack:
    enabled: true
    webhook_url: https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
    channel: "#alerts"
  
  thresholds:
    schema_validation_rate:
      warning: 0.95
      critical: 0.90
    template_fallback_rate:
      warning: 0.20
      critical: 0.30
    confidence_score:
      warning: 0.70
      critical: 0.60
```

## Best Practices

### 1. Metric Recording
- Record metrics at appropriate intervals (not too frequent)
- Use meaningful labels for categorization
- Include relevant metadata for context
- Validate metric values before recording

### 2. Threshold Configuration
- Set thresholds based on historical data
- Use different thresholds for different environments
- Regularly review and adjust thresholds
- Test threshold changes in staging first

### 3. Alert Management
- Use appropriate severity levels
- Implement alert acknowledgment workflows
- Regularly review and resolve alerts
- Suppress false positives appropriately

### 4. Performance Optimization
- Monitor system performance metrics
- Use appropriate retention periods
- Clean up old data regularly
- Scale resources based on load

### 5. Security
- Secure alert handler credentials
- Use environment variables for sensitive data
- Implement proper access controls
- Monitor for security-related quality issues

## Troubleshooting

### Common Issues

#### 1. High Memory Usage
```python
# Check storage statistics
stats = system.quality_monitor.get_storage_statistics()
print(f"Total metrics: {stats['total_metrics']}")

# Reduce retention period
system = QualityAlertingSystem(
    max_metrics_per_type=5000,  # Reduce capacity
    default_retention_hours=12  # Reduce retention
)
```

#### 2. Alert Spam
```python
# Increase deduplication window
system = QualityAlertingSystem(
    deduplication_window_minutes=30  # Increase from default 15
)

# Adjust thresholds
threshold = QualityThreshold(
    metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
    warning_threshold=0.90,    # Lower threshold
    critical_threshold=0.85,   # Lower threshold
    min_samples=20,            # Require more samples
    time_window_minutes=120    # Longer time window
)
```

#### 3. Performance Issues
```python
# Check performance statistics
perf_stats = system.quality_monitor.get_performance_statistics()
print(f"Errors: {perf_stats['errors']}")
print(f"Last error: {perf_stats['last_error']}")

# Optimize monitoring interval
system = QualityAlertingSystem(
    monitoring_interval_seconds=120  # Less frequent monitoring
)
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create system with debug settings
system = QualityAlertingSystem(
    monitoring_interval_seconds=10,  # More frequent monitoring
    max_metrics_per_type=1000        # Smaller capacity for debugging
)

# Monitor system status
status = system.get_system_status()
print(f"System status: {status}")
```

## API Reference

### Core Classes

- **`QualityAlertingSystem`**: Main system class
- **`QualityMonitor`**: Metrics tracking and storage
- **`QualityDegradationDetector`**: Degradation detection algorithms
- **`AlertManager`**: Alert lifecycle management
- **`QualityMetric`**: Metric data structure
- **`QualityThreshold`**: Threshold configuration
- **`Alert`**: Alert data structure

### Alert Handlers

- **`LoggingAlertHandler`**: Log-based alerts
- **`EmailAlertHandler`**: Email notifications
- **`SlackAlertHandler`**: Slack notifications
- **`WebhookAlertHandler`**: HTTP webhook notifications
- **`CompositeAlertHandler`**: Multi-channel routing

### Configuration

- **`QualityAlertingConfig`**: System configuration
- **`QualityAlertingSettings`**: Environment-based settings
- **`EmailConfig`**: Email handler configuration
- **`SlackConfig`**: Slack handler configuration
- **`WebhookConfig`**: Webhook handler configuration

## Contributing

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd comply-ai

# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest src/llama_mapper/analysis/tests/

# Run linting
flake8 src/llama_mapper/analysis/quality/
mypy src/llama_mapper/analysis/quality/
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation for changes

### Testing

- Write unit tests for new functionality
- Add integration tests for complex features
- Include performance tests for critical paths
- Ensure all tests pass before submitting

## License

This quality alerting system is part of the Llama Mapper project and follows the same licensing terms.
