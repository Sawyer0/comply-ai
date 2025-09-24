# Comprehensive Testing Framework for Llama Mapper

## Overview

This document describes the comprehensive testing framework implemented for the Llama Mapper multi-service architecture. The framework provides unified testing capabilities across:

- **Core Mapper Service** (main llama-mapper)
- **Detector Orchestration Service** (detector-orchestration/)
- **Analysis Service** (containerized analytics)

## Framework Components

### 1. Unified Testing Infrastructure ✅

#### Standardized pytest Configuration
- **File**: `pytest.ini`
- **Features**: Unified markers, coverage settings, async support
- **Markers**: unit, integration, performance, security, chaos, e2e, cross_service

#### Shared Test Fixtures
- **File**: `tests/conftest.py`
- **Capabilities**:
  - Service cluster management
  - Mock service registry
  - Test data factory
  - Database management
  - Environment provisioning

#### Test Utilities
- **Location**: `tests/utils/`
- **Components**:
  - `service_cluster.py` - Multi-service cluster management
  - `mock_services.py` - Mock implementations for unit testing
  - `test_data.py` - Synthetic data generation with privacy compliance
  - `environment.py` - Docker-based test environment management
  - `database.py` - Test database setup and cleanup

### 2. Multi-Service Coverage Strategy ✅

#### Coverage Aggregation
- **File**: `tests/frameworks/coverage.py`
- **Features**:
  - Service-specific coverage collection (85% threshold)
  - Critical path coverage validation (95% threshold)
  - Cross-service interaction coverage
  - Unified reporting with HTML output

#### Mutation Testing
- **File**: `tests/frameworks/mutation.py`
- **Capabilities**:
  - Multi-tool support (mutmut, cosmic-ray)
  - Service-specific mutation testing
  - Critical path mutation validation (90% threshold)
  - Quality assessment and recommendations

### 3. Cross-Service Integration Testing ✅

#### Contract Testing
- **File**: `tests/frameworks/contract_testing.py`
- **Features**:
  - API contract validation between services
  - JSON schema validation
  - Backward compatibility checking
  - Contract registry management

#### End-to-End Workflows
- **File**: `tests/frameworks/end_to_end.py`
- **Workflows**:
  - Detection → Mapping → Analysis
  - Batch processing across services
  - Failure recovery scenarios
  - Data consistency validation

### 4. Multi-Service Performance Testing ✅

#### Performance Framework
- **File**: `tests/frameworks/performance.py`
- **Capabilities**:
  - Service-specific performance targets:
    - Core Mapper: p95 < 100ms, 1000 RPS
    - Detector Orchestration: p95 < 200ms, 500 RPS
    - Analysis Service: p95 < 500ms, 100 RPS
  - Cross-service workflow performance
  - Load testing with multiple generators
  - Bottleneck identification

#### Load Testing Types
- Baseline testing
- Stress testing
- Spike testing
- Endurance testing
- Volume testing

### 5. Chaos Engineering Framework ✅

#### Chaos Testing
- **File**: `tests/frameworks/chaos.py`
- **Failure Types**:
  - Service crashes
  - Network partitions
  - High latency injection
  - Memory exhaustion
  - Database failures
  - Cascading failure prevention

#### Resilience Validation
- Recovery time measurement
- Blast radius containment
- Expected behavior validation
- System stability assessment

### 6. Test Environment Automation ✅

#### Docker-based Environments
- **File**: `tests/docker-compose.test.yml`
- **Services**: All three services + supporting infrastructure
- **Features**: Isolated environments, service mesh testing, parallel execution

#### Environment Management
- Automated provisioning and cleanup
- Service discovery testing
- Network isolation
- Resource conflict prevention

## Usage Guide

### Running Tests

#### Basic Test Execution
```bash
# Run all tests
pytest

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m performance
pytest -m security
pytest -m chaos

# Run service-specific tests
pytest -m core_mapper
pytest -m detector_orchestration
pytest -m analysis_service
```

#### Coverage Testing
```bash
# Run with coverage
pytest --cov=src/llama_mapper --cov-report=html

# Generate unified coverage report
python -c "
from tests.frameworks.coverage import CoverageAggregator
import asyncio
aggregator = CoverageAggregator()
report = asyncio.run(aggregator.generate_unified_report())
print(f'Overall coverage: {report.overall_coverage:.1f}%')
"
```

#### Performance Testing
```bash
# Run performance tests
pytest -m performance

# Custom performance test
python -c "
from tests.frameworks.performance import MultiServicePerformanceTester, LoadTestConfig, LoadTestType
import asyncio
tester = MultiServicePerformanceTester()
config = LoadTestConfig(
    test_name='custom_test',
    test_type=LoadTestType.BASELINE,
    duration_seconds=60,
    concurrent_users=50,
    ramp_up_seconds=10,
    ramp_down_seconds=10
)
# Run with actual service clients
"
```

#### Chaos Testing
```bash
# Run chaos tests
pytest -m chaos

# Custom chaos test
python -c "
from tests.frameworks.chaos import ChaosTestOrchestrator, FailureType
import asyncio
orchestrator = ChaosTestOrchestrator(service_clients)
result = asyncio.run(orchestrator.inject_service_failure(
    'core_mapper', FailureType.SERVICE_CRASH, 30
))
print(f'Test success: {result.success}')
"
```

### Test Data Management

#### Using Test Data Factory
```python
from tests.utils.test_data import TestDataFactory

# Create factory
factory = TestDataFactory(use_synthetic=True, privacy_scrubbing=True)

# Generate test data
detector_output = factory.create_detector_output("presidio")
mapping_payload = factory.create_mapping_payload()
golden_dataset = factory.create_golden_dataset("regression_test", size=100)
```

#### Mock Services
```python
from tests.utils.mock_services import MockServiceRegistry

# Create mock registry
registry = MockServiceRegistry()

# Get service mocks
mapper_mock = registry.create_mapper_mock()
orchestration_mock = registry.create_orchestration_mock()
analysis_mock = registry.create_analysis_mock()

# Configure failure scenarios
registry.configure_failure_scenario("mapper", "timeout")
```

### Multi-Service Environment Setup

#### Using Docker Compose
```bash
# Start test environment
docker-compose -f tests/docker-compose.test.yml up -d

# Run tests against environment
CORE_MAPPER_URL=http://localhost:8001 \
DETECTOR_ORCHESTRATION_URL=http://localhost:8002 \
ANALYSIS_SERVICE_URL=http://localhost:8003 \
pytest tests/integration/

# Cleanup
docker-compose -f tests/docker-compose.test.yml down
```

#### Programmatic Environment Management
```python
from tests.utils.environment import TestEnvironmentManager

# Create environment manager
env_manager = TestEnvironmentManager(test_config)
await env_manager.setup()

# Provision isolated environment
env_id = await env_manager.provision_isolated_environment("test_001")

# Use environment...

# Cleanup
await env_manager.cleanup_environment(env_id)
```

## Configuration

### Test Configuration
```python
# tests/conftest.py
@dataclass
class TestConfig:
    # Service ports
    core_mapper_port: int = 8000
    detector_orchestration_port: int = 8001
    analysis_service_port: int = 8002
    
    # Database settings
    postgres_test_db: str = "llama_mapper_test"
    redis_test_db: int = 1
    
    # Test data settings
    use_synthetic_data: bool = True
    privacy_scrubbing: bool = True
    tenant_isolation: bool = True
```

### Performance Targets
```python
# tests/frameworks/performance.py
SERVICE_TARGETS = {
    'core_mapper': {
        'latency_p95_ms': 100.0,
        'throughput_rps': 1000.0,
        'error_rate_threshold': 0.01
    },
    'detector_orchestration': {
        'latency_p95_ms': 200.0,
        'throughput_rps': 500.0,
        'error_rate_threshold': 0.01
    },
    'analysis_service': {
        'latency_p95_ms': 500.0,
        'throughput_rps': 100.0,
        'error_rate_threshold': 0.02
    }
}
```

## Continuous Integration

### CI/CD Integration
```yaml
# .github/workflows/testing.yml
name: Comprehensive Testing
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run unit tests
        run: pytest tests/unit/ -m "unit and not slow"
  
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Start test environment
        run: docker-compose -f tests/docker-compose.test.yml up -d
      - name: Run integration tests
        run: pytest tests/integration/ -m integration
  
  performance-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2
      - name: Run performance tests
        run: pytest tests/performance/ -m performance
```

### Quality Gates
```bash
# Quality gate script
#!/bin/bash
set -e

echo "Running quality gates..."

# Unit tests with coverage
pytest tests/unit/ --cov=src --cov-fail-under=85

# Integration tests
pytest tests/integration/ -m integration

# Contract tests
pytest tests/contract/ -m contract

# Security tests
pytest tests/security/ -m security

# Performance baseline
pytest tests/performance/ -m "performance and baseline"

echo "All quality gates passed!"
```

## Reports and Monitoring

### Test Reports
- **Coverage Reports**: `tests/coverage/unified_report.html`
- **Performance Reports**: `tests/performance/performance_report.html`
- **Mutation Reports**: `tests/coverage/mutation_report.html`
- **Chaos Reports**: `tests/chaos/resilience_report.html`

### Monitoring Integration
- Prometheus metrics for test execution
- Grafana dashboards for test trends
- Alert manager for test failures
- Correlation with service metrics

## Best Practices

### Test Organization
1. **Isolation**: Each test should be independent
2. **Determinism**: Tests should produce consistent results
3. **Speed**: Fast feedback loops with parallel execution
4. **Clarity**: Clear test names and documentation

### Cross-Service Testing
1. **Contract-First**: Define contracts before implementation
2. **Mock External**: Mock external services, test internal contracts
3. **Data Consistency**: Validate data flow across services
4. **Error Propagation**: Test error handling across boundaries

### Performance Testing
1. **Baseline First**: Establish performance baselines
2. **Realistic Load**: Use production-like traffic patterns
3. **Resource Monitoring**: Track CPU, memory, and network
4. **Gradual Increase**: Ramp up load gradually

### Chaos Engineering
1. **Hypotheses**: Form clear hypotheses about system behavior
2. **Small Blast Radius**: Start with limited failure scope
3. **Monitoring**: Comprehensive monitoring during experiments
4. **Rollback Plan**: Always have recovery procedures ready

## Troubleshooting

### Common Issues

#### Test Environment Setup
```bash
# Check Docker connectivity
docker ps

# Verify service health
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health

# Check logs
docker-compose -f tests/docker-compose.test.yml logs
```

#### Database Connectivity
```bash
# Test PostgreSQL connection
PGPASSWORD=test_password psql -h localhost -p 5433 -U test_user -d llama_mapper_test -c "SELECT 1;"

# Test Redis connection
redis-cli -h localhost -p 6380 ping
```

#### Performance Issues
- Check resource limits in Docker
- Verify network connectivity between services
- Monitor CPU and memory usage
- Check for DNS resolution delays

### Debugging Tests
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use pytest debugging
pytest --pdb tests/integration/test_workflow.py

# Verbose output
pytest -v -s tests/unit/test_coverage.py
```

## Next Steps

### Enhancements
1. **AI-Powered Test Generation**: Generate test cases using ML
2. **Visual Regression Testing**: UI/API response comparison
3. **Distributed Testing**: Multi-region test execution
4. **Canary Testing**: Automated canary deployment testing

### Integration
1. **Service Mesh**: Istio integration for advanced testing
2. **APM Integration**: Application performance monitoring
3. **Log Aggregation**: Centralized logging for test analysis
4. **Alerting**: Proactive test failure detection

This comprehensive testing framework provides enterprise-grade testing capabilities for the Llama Mapper multi-service architecture, ensuring reliability, performance, and security across all components.
