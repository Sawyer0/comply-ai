# Performance Testing Suite for Detector Orchestration

This directory contains comprehensive performance testing tools for the detector orchestration service. The suite includes load testing, fault tolerance testing, and scalability testing using both k6 and Locust frameworks.

## Overview

The performance testing suite provides:

- **Load Testing**: Test system behavior under various load conditions
- **Fault Tolerance Testing**: Verify graceful degradation under failures
- **Scalability Testing**: Test detector coordination at scale
- **Performance Validation**: Ensure <2s p95 response time requirement

## Quick Start

### Prerequisites

1. **Docker and Docker Compose** installed
2. **Python 3.11+** for running the test orchestrator
3. **k6** (via Docker) for load testing
4. **Locust** (via Docker) for distributed load testing

### Environment Setup

1. **Start the test environment**:
   ```bash
   cd perf
   docker-compose -f docker-compose.test.yml up -d
   ```

2. **Verify services are running**:
   ```bash
   # Check orchestration service
   curl http://localhost:8000/health

   # Check mock detectors
   curl http://localhost:8001/health
   curl http://localhost:8002/health
   curl http://localhost:8003/health
   curl http://localhost:8004/health
   curl http://localhost:8005/health
   ```

3. **Access monitoring tools**:
   - **Grafana**: http://localhost:3000 (admin/admin)
   - **Prometheus**: http://localhost:9090
   - **Locust UI**: http://localhost:8089

## Test Types

### 1. Smoke Tests
Light load tests to verify basic functionality.

```bash
# Run smoke tests
python run_performance_tests.py --test-type smoke

# Or with k6 only
docker run --rm --network host \
  -v $(pwd)/../k6:/scripts \
  -e K6_SMOKE_VUS=5 \
  -e K6_SMOKE_DURATION=2m \
  -e PERF_BASE_URL=http://localhost:8000 \
  grafana/k6:latest \
  run /scripts/orchestration_load.js
```

### 2. Load Tests
Normal operational load testing.

```bash
# Run load tests
python run_performance_tests.py --test-type load

# Or with Locust only
docker run --rm --network host \
  -v $(pwd)/..:/mnt/locust \
  -e LOCUST_USERS=50 \
  -e LOCUST_SPAWN_RATE=5 \
  -e LOCUST_RUN_TIME=600s \
  -e LOCUST_HOST=http://localhost:8000 \
  locustio/locust:latest \
  -f /mnt/locust/perf/locustfile_orchestration.py \
  --headless
```

### 3. Stress Tests
High load testing to find system limits.

```bash
# Run stress tests
python run_performance_tests.py --test-type stress
```

### 4. Fault Tolerance Tests
Test system behavior under various failure conditions.

```bash
# Run fault tolerance tests
python run_performance_tests.py --test-type fault-tolerance
```

### 5. Scalability Tests
Test detector coordination with different batch sizes and concurrency.

```bash
# Run scalability tests
python run_performance_tests.py --test-type scalability
```

### 6. All Tests
Run the complete test suite.

```bash
# Run all tests
python run_performance_tests.py --test-type all
```

## Configuration

### Test Configuration

Edit `test-config.yaml` to customize test parameters:

```yaml
# Service settings
service:
  host: "0.0.0.0"
  port: 8000
  workers: 4

# Detector configurations
detectors:
  mock-success-detector:
    endpoint: "http://mock-detector:8001/detect"
    timeout_ms: 5000
    max_retries: 3
    # ... more detector configs

# Performance targets
performance_targets:
  p95_response_time_ms: 2000  # <2s p95 requirement
  success_rate: 0.95  # 95% success rate
  throughput_rps: 100  # 100 requests per second minimum
```

### Environment Variables

Key environment variables for test configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `ORCHESTRATION_PERF_TENANT_ID` | Test tenant ID | perf-tenant |
| `ORCHESTRATION_PERF_API_KEY` | API key for authentication | test-key |
| `ORCHESTRATION_PERF_API_KEY_HEADER` | API key header name | X-API-Key |
| `ORCHESTRATION_POLICY_BUNDLE` | Policy bundle for tests | default |
| `ORCHESTRATION_PERF_BATCH_SIZE` | Batch size for testing | 5 |

## Mock Detectors

The test suite includes configurable mock detectors that simulate:

- **Variable response times** (200ms - 5s)
- **Configurable success rates** (70% - 100%)
- **Different failure modes** (timeouts, errors, slow responses)
- **Batch processing support**
- **Realistic detection results**

### Mock Detector Types

1. **Success Detector** (`mock-detector-success`)
   - 100% success rate
   - Fast response times (200-500ms)
   - All capabilities enabled

2. **Variable Detector** (`mock-detector-variable`)
   - 98% success rate
   - Variable response times (100ms - 2s)
   - Code analysis capabilities

3. **Failing Detector** (`mock-detector-failing`)
   - 70% success rate
   - Occasional failures for fault tolerance testing
   - Content moderation capabilities

4. **Timeout Detector** (`mock-detector-timeout`)
   - 95% success rate
   - Occasional timeouts (up to 30s)
   - Deep analysis capabilities

5. **Batch Detector** (`mock-detector-batch`)
   - 95% success rate
   - Batch processing mode
   - Bulk analysis capabilities

## Monitoring and Metrics

### Prometheus Metrics

The orchestration service exposes metrics at `/metrics`:

- `orchestrate_requests_total` - Total orchestration requests
- `detector_latency_ms` - Detector response times
- `coverage_achieved` - Detection coverage metrics
- `circuit_breaker_state` - Circuit breaker status

### Grafana Dashboards

Pre-configured dashboards are available at http://localhost:3000:

1. **Orchestration Performance** - Response times, throughput, error rates
2. **Detector Health** - Individual detector status and performance
3. **Load Test Results** - Real-time test metrics and thresholds

## Test Reports

Test results are automatically saved to JSON files:

```bash
# Generate report from latest results
python run_performance_tests.py --report-only

# List all result files
ls -la performance_test_results_*.json
```

### Report Format

```json
{
  "test_suite": "detector_orchestration_performance",
  "timestamp": "2024-01-15T10:30:00Z",
  "summary": {
    "total_tests": 5,
    "passed": 4,
    "failed": 1,
    "success_rate": 0.8,
    "duration_seconds": 1800.5
  },
  "results": {
    "smoke": {
      "status": "success",
      "k6_result": {...},
      "locust_result": {...}
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   # Check service logs
   docker-compose -f docker-compose.test.yml logs orchestration

   # Verify Redis is running
   docker-compose -f docker-compose.test.yml ps
   ```

2. **Mock detectors not responding**
   ```bash
   # Check detector health
   curl http://localhost:8001/health

   # Restart specific detector
   docker-compose -f docker-compose.test.yml restart mock-detector-success
   ```

3. **Performance tests failing**
   ```bash
   # Check service health first
   curl http://localhost:8000/health

   # Run smoke test to verify basic functionality
   python run_performance_tests.py --test-type smoke
   ```

### Performance Tuning

1. **Adjust load test parameters**:
   ```bash
   # Reduce load for slower systems
   export K6_LOAD_TARGET_VUS=25
   export LOCUST_USERS=25
   ```

2. **Increase timeouts for slower detectors**:
   ```yaml
   # In test-config.yaml
   orchestration:
     default_timeout_ms: 10000  # Increase from 5000
   ```

3. **Monitor system resources**:
   ```bash
   # Monitor Docker resource usage
   docker stats

   # Check system memory and CPU
   htop
   ```

## Advanced Usage

### Custom Test Scenarios

Create custom test scenarios by modifying `locustfile_orchestration.py`:

```python
class CustomLoadTest(HttpUser):
    @task
    def custom_scenario(self):
        # Your custom test logic here
        pass
```

### Distributed Load Testing

For large-scale testing, use Locust's distributed mode:

```bash
# Master node
docker run --rm -p 8089:8089 \
  -v $(pwd)/..:/mnt/locust \
  locustio/locust:latest \
  -f /mnt/locust/perf/locustfile_orchestration.py \
  --master --host=http://localhost:8000

# Worker nodes
docker run --rm \
  -v $(pwd)/..:/mnt/locust \
  locustio/locust:latest \
  -f /mnt/locust/perf/locustfile_orchestration.py \
  --worker --master-host=localhost
```

### CI/CD Integration

Integrate performance tests into your CI/CD pipeline:

```bash
# Run smoke tests in CI
python run_performance_tests.py --test-type smoke

# Generate performance report
python run_performance_tests.py --report-only > performance_report.json
```

## Requirements Validation

This test suite validates the following requirements:

- ✅ **<2s p95 response time** for orchestration requests
- ✅ **95% success rate** under normal load
- ✅ **Graceful degradation** under detector failures
- ✅ **Circuit breaker protection** for failing detectors
- ✅ **Proper error handling** for malformed requests
- ✅ **Cache performance** and invalidation
- ✅ **Batch processing** capabilities
- ✅ **Async job processing** for long-running requests

## Support

For issues with the performance testing suite:

1. Check the troubleshooting section above
2. Verify all services are running with `docker-compose ps`
3. Check service logs with `docker-compose logs`
4. Review the test configuration in `test-config.yaml`

## Contributing

When contributing to the performance testing suite:

1. **Update mock detectors** to reflect new detector capabilities
2. **Add new test scenarios** for new orchestration features
3. **Update performance targets** based on new requirements
4. **Document new test types** in this README
5. **Add monitoring dashboards** for new metrics
