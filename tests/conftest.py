"""
Unified pytest configuration for Llama Mapper multi-service testing.

This module provides shared fixtures and test utilities that work across:
- Core Mapper Service (main llama-mapper)
- Detector Orchestration Service (detector-orchestration/)
- Analysis Service (containerized analytics)
"""

import asyncio
import os
import pytest
from typing import Dict, Any, List
from dataclasses import dataclass

try:
    import docker
except ImportError:
    docker = None

try:
    import asyncpg
except ImportError:
    asyncpg = None

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

# Import test utilities (with fallback handling)
try:
    from tests.utils.service_cluster import ServiceCluster, ServiceClusterConfig
    from tests.utils.mock_services import MockServiceRegistry
    from tests.utils.test_data import TestDataFactory
    from tests.utils.environment import TestEnvironmentManager
    from tests.utils.database import TestDatabaseManager
except ImportError as e:
    # Graceful fallback for when utils are not yet available
    print(f"Warning: Test utilities not available: {e}")
    ServiceCluster = None
    ServiceClusterConfig = None
    MockServiceRegistry = None
    TestDataFactory = None
    TestEnvironmentManager = None
    TestDatabaseManager = None


# Test Configuration
@dataclass
class TestConfig:
    """Unified test configuration for all services."""
    
    # Service configurations
    core_mapper_port: int = 8000
    detector_orchestration_port: int = 8001
    analysis_service_port: int = 8002
    
    # Database configurations
    postgres_test_db: str = "llama_mapper_test"
    clickhouse_test_db: str = "llama_mapper_analytics_test"
    redis_test_db: int = 1
    
    # Test data configurations
    use_synthetic_data: bool = True
    tenant_isolation: bool = True
    privacy_scrubbing: bool = True
    
    # Performance test configurations
    load_test_duration: int = 30  # seconds
    max_concurrent_requests: int = 100
    
    # Environment configurations
    docker_compose_file: str = "tests/docker-compose.test.yml"
    cleanup_after_tests: bool = True


@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """Provide test configuration for all test sessions."""
    return TestConfig()


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Service Cluster Fixtures
@pytest.fixture(scope="session")
async def service_cluster(test_config: TestConfig):
    """Provide a complete service cluster for testing all three services."""
    if ServiceCluster is None or ServiceClusterConfig is None:
        pytest.skip("Service cluster utilities not available")
    
    cluster_config = ServiceClusterConfig(
        core_mapper_port=test_config.core_mapper_port,
        detector_orchestration_port=test_config.detector_orchestration_port,
        analysis_service_port=test_config.analysis_service_port,
        postgres_db=test_config.postgres_test_db,
        clickhouse_db=test_config.clickhouse_test_db,
        redis_db=test_config.redis_test_db
    )
    
    cluster = ServiceCluster(cluster_config)
    await cluster.start()
    
    try:
        yield cluster
    finally:
        await cluster.stop()


@pytest.fixture(scope="function")
def isolated_service_mocks() -> Dict[str, Any]:
    """Provide isolated mocks for each service for unit testing."""
    if MockServiceRegistry is None:
        pytest.skip("Mock service registry not available")
    
    registry = MockServiceRegistry()
    
    return {
        'mapper': registry.create_mapper_mock(),
        'orchestration': registry.create_orchestration_mock(),
        'analysis': registry.create_analysis_mock(),
        'registry': registry
    }


# Database Fixtures
@pytest.fixture(scope="session")
async def test_db_manager(config: TestConfig):
    """Manage test databases for all services."""
    if TestDatabaseManager is None:
        pytest.skip("Database manager not available")
    
    db_manager = TestDatabaseManager(config)
    await db_manager.setup_databases()
    
    try:
        yield db_manager
    finally:
        await db_manager.cleanup_databases()


@pytest.fixture(scope="function")
async def postgres_test_pool(db_manager: TestDatabaseManager):
    """Provide PostgreSQL connection pool for testing."""
    if asyncpg is None:
        pytest.skip("asyncpg not available")
    
    pool = await db_manager.get_postgres_pool()
    
    # Clean state for each test
    await db_manager.clean_postgres_state()
    
    try:
        yield pool
    finally:
        await db_manager.clean_postgres_state()


@pytest.fixture(scope="function")
async def redis_test_client(db_manager: TestDatabaseManager):
    """Provide Redis client for testing."""
    if redis is None:
        pytest.skip("redis not available")
    
    client = await db_manager.get_redis_client()
    
    # Clean state for each test
    await client.flushdb()
    
    try:
        yield client
    finally:
        await client.flushdb()


# Test Data Fixtures
@pytest.fixture(scope="session")
def test_data_factory(config: TestConfig):
    """Provide test data factory for generating consistent test data."""
    if TestDataFactory is None:
        pytest.skip("Test data factory not available")
    
    return TestDataFactory(
        use_synthetic=config.use_synthetic_data,
        privacy_scrubbing=config.privacy_scrubbing
    )


@pytest.fixture(scope="function")
def sample_detector_output(data_factory: TestDataFactory) -> Dict[str, Any]:
    """Provide sample detector output for testing."""
    return data_factory.create_detector_output(
        detector_type="presidio",
        confidence=0.95,
        findings=[
            {
                "entity_type": "PERSON",
                "confidence": 0.95,
                "start": 0,
                "end": 10,
                "text": "[REDACTED]"
            }
        ]
    )


@pytest.fixture(scope="function")
def sample_mapping_payload(data_factory: TestDataFactory) -> Dict[str, Any]:
    """Provide sample mapping payload for testing."""
    return data_factory.create_mapping_payload(
        detector_outputs=[
            data_factory.create_detector_output("presidio"),
            data_factory.create_detector_output("deberta")
        ],
        framework="SOC2",
        tenant_id="test_tenant"
    )


@pytest.fixture(scope="function")
def tenant_test_data(data_factory: TestDataFactory) -> Dict[str, Any]:
    """Provide tenant-specific test data with isolation."""
    return data_factory.create_tenant_data("test_tenant_001")


# Environment Management Fixtures
@pytest.fixture(scope="session")
async def test_environment_manager(config: TestConfig):
    """Manage test environments for multi-service testing."""
    if TestEnvironmentManager is None:
        pytest.skip("Test environment manager not available")
    
    env_manager = TestEnvironmentManager(config)
    await env_manager.setup()
    
    try:
        yield env_manager
    finally:
        if config.cleanup_after_tests:
            await env_manager.cleanup()


@pytest.fixture(scope="function")
async def isolated_test_environment(
    env_manager: TestEnvironmentManager
) -> str:
    """Provide isolated test environment for individual tests."""
    import uuid
    test_id = f"test_{str(uuid.uuid4())[:8]}"
    env = await env_manager.provision_isolated_environment(test_id)
    
    try:
        yield env
    finally:
        await env_manager.cleanup_environment(test_id)


# Cross-Service Integration Fixtures
@pytest.fixture(scope="function")
async def cross_service_client(cluster: ServiceCluster) -> Dict[str, Any]:
    """Provide HTTP clients for all services in the cluster."""
    return {
        'mapper': cluster.get_mapper_client(),
        'orchestration': cluster.get_orchestration_client(),
        'analysis': cluster.get_analysis_client()
    }


@pytest.fixture(scope="function")
def correlation_id() -> str:
    """Provide correlation ID for tracing cross-service requests."""
    import uuid
    return str(uuid.uuid4())


# Performance Testing Fixtures
@pytest.fixture(scope="session")
def performance_test_config(config: TestConfig) -> Dict[str, Any]:
    """Provide configuration for performance testing."""
    return {
        'duration': config.load_test_duration,
        'max_concurrent': config.max_concurrent_requests,
        'sla_targets': {
            'core_mapper': {'p95_latency_ms': 100, 'throughput_rps': 1000},
            'detector_orchestration': {'p95_latency_ms': 200, 'throughput_rps': 500},
            'analysis_service': {'p95_latency_ms': 500, 'throughput_rps': 100}
        }
    }


# Security Testing Fixtures
@pytest.fixture(scope="function")
def security_test_payloads() -> Dict[str, List[str]]:
    """Provide malicious payloads for security testing."""
    return {
        'sql_injection': [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users--"
        ],
        'xss': [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src='javascript:alert(\"xss\")'></iframe>"
        ],
        'path_traversal': [
            "../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/hosts",
            "C:\\windows\\win.ini"
        ]
    }


# Chaos Testing Fixtures
@pytest.fixture(scope="function")
def chaos_scenarios() -> List[Dict[str, Any]]:
    """Provide chaos engineering test scenarios."""
    return [
        {
            'name': 'core_mapper_service_failure',
            'target_service': 'core_mapper',
            'failure_type': 'service_crash',
            'duration': 30,
            'expected_behavior': 'orchestration_fallback_to_rules'
        },
        {
            'name': 'detector_orchestration_network_partition',
            'target_service': 'detector_orchestration',
            'failure_type': 'network_partition',
            'duration': 60,
            'expected_behavior': 'direct_detector_calls'
        },
        {
            'name': 'analysis_service_resource_exhaustion',
            'target_service': 'analysis_service',
            'failure_type': 'memory_exhaustion',
            'duration': 45,
            'expected_behavior': 'core_functionality_preserved'
        }
    ]


# Custom pytest markers configuration
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", 
        "integration: Integration tests across components"
    )
    config.addinivalue_line(
        "markers",
        "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers",
        "security: Security and privacy tests"
    )
    config.addinivalue_line(
        "markers",
        "chaos: Chaos engineering and fault tolerance tests"
    )
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end workflow tests"
    )
    config.addinivalue_line(
        "markers",
        "slow: Slow running tests (> 30 seconds)"
    )
    config.addinivalue_line(
        "markers",
        "core_mapper: Tests specific to Core Mapper service"
    )
    config.addinivalue_line(
        "markers",
        "detector_orchestration: Tests specific to Detector Orchestration service"
    )
    config.addinivalue_line(
        "markers",
        "analysis_service: Tests specific to Analysis service"
    )
    config.addinivalue_line(
        "markers",
        "cross_service: Tests spanning multiple services"
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add appropriate markers."""
    for item in items:
        # Add slow marker for tests that might take longer
        if "performance" in item.keywords or "chaos" in item.keywords or "e2e" in item.keywords:
            item.add_marker(pytest.mark.slow)
        
        # Add service-specific markers based on file location
        test_path = str(item.fspath)
        if "core_mapper" in test_path or "mapper" in test_path:
            item.add_marker(pytest.mark.core_mapper)
        elif "detector_orchestration" in test_path or "orchestration" in test_path:
            item.add_marker(pytest.mark.detector_orchestration)
        elif "analysis" in test_path:
            item.add_marker(pytest.mark.analysis_service)
        
        # Add cross-service marker for integration tests
        if "integration" in item.keywords and any(
            service in test_path for service in ["cross_service", "e2e", "workflow"]
        ):
            item.add_marker(pytest.mark.cross_service)


# Async test configuration
@pytest.fixture(scope="session")
def anyio_backend():
    """Configure anyio backend for async tests."""
    return "asyncio"


# Test reporting hooks
def pytest_runtest_makereport(item, call):
    """Generate test reports with correlation tracking."""
    if "correlation_id" in item.fixturenames:
        # Add correlation ID to test reports for cross-service tracing
        if hasattr(item, 'correlation_id'):
            call.correlation_id = item.correlation_id


# Environment setup and teardown
@pytest.fixture(scope="session", autouse=True)
async def setup_test_environment():
    """Setup global test environment."""
    # Ensure test environment variables are set
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("LOG_LEVEL", "DEBUG")
    os.environ.setdefault("PRIVACY_MODE", "true")
    
    # Setup test directories
    test_dirs = [
        "tests/artifacts",
        "tests/logs", 
        "tests/coverage",
        "tests/performance",
        "tests/security"
    ]
    
    for test_dir in test_dirs:
        os.makedirs(test_dir, exist_ok=True)
    
    yield
    
    # Cleanup test artifacts if configured
    if os.getenv("CLEANUP_TEST_ARTIFACTS", "true").lower() == "true":
        import shutil
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir, ignore_errors=True)