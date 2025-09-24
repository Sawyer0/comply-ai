---
inclusion: fileMatch
fileMatchPattern: '**/test*/**'
---

# Testing Strategy & Guidelines

## Testing Pyramid

### Unit Tests (70%)
- Test individual functions and classes in isolation
- Mock external dependencies
- Fast execution (< 1ms per test)
- High coverage of business logic

### Integration Tests (20%)
- Test component interactions
- Use real database connections (test DB)
- Test API endpoints end-to-end
- Validate data flow between services

### End-to-End Tests (10%)
- Test complete user workflows
- Use production-like environment
- Validate system behavior from user perspective
- Include performance and reliability testing

## Test Organization

### Directory Structure
```
tests/
├── unit/
│   ├── test_mapper_service.py
│   ├── test_detector_registry.py
│   └── test_compliance_analyzer.py
├── integration/
│   ├── test_api_endpoints.py
│   ├── test_database_operations.py
│   └── test_service_communication.py
├── performance/
│   ├── test_load_handling.py
│   └── test_response_times.py
├── security/
│   ├── test_authentication.py
│   └── test_input_validation.py
└── fixtures/
    ├── detector_outputs.json
    └── compliance_mappings.json
```

### Test Naming Conventions
```python
def test_<method_name>_<scenario>_<expected_result>():
    """
    Test method_name when scenario occurs, expecting expected_result
    """
    pass

# Examples:
def test_map_detector_output_valid_input_returns_canonical_taxonomy():
def test_map_detector_output_invalid_confidence_raises_validation_error():
def test_get_compliance_mapping_nonexistent_framework_returns_404():
```

## Test Data Management

### Fixtures
```python
@pytest.fixture
def sample_detector_output():
    return {
        "detector_type": "presidio",
        "findings": [
            {
                "entity_type": "PERSON",
                "confidence": 0.95,
                "start": 0,
                "end": 10
            }
        ]
    }

@pytest.fixture
def mock_mapper_service():
    with patch('llama_mapper.services.MapperService') as mock:
        yield mock
```

### Factory Pattern
```python
class DetectorOutputFactory:
    @staticmethod
    def create_pii_detection(confidence=0.9, entity_type="PERSON"):
        return DetectorOutput(
            detector_type="presidio",
            confidence=confidence,
            entity_type=entity_type,
            # ... other fields
        )
```

## Mocking Guidelines

### External Services
- Mock all external API calls
- Use `responses` library for HTTP mocking
- Mock database connections in unit tests
- Use dependency injection for easier mocking

### Model Inference
```python
@patch('llama_mapper.models.LlamaMapper.predict')
def test_mapper_prediction(mock_predict):
    mock_predict.return_value = CanonicalTaxonomy(
        category="pii",
        subcategory="person_name",
        confidence=0.95
    )
    # Test logic here
```

## Performance Testing

### Load Testing
- Use `locust` for load testing API endpoints
- Test with realistic data volumes
- Measure response times under load
- Validate system behavior at capacity limits

### Benchmark Tests
```python
@pytest.mark.performance
def test_mapper_service_throughput():
    """Test mapper service can handle required throughput"""
    start_time = time.time()
    
    # Process batch of requests
    results = process_batch(test_inputs)
    
    elapsed = time.time() - start_time
    throughput = len(test_inputs) / elapsed
    
    assert throughput >= REQUIRED_THROUGHPUT
```

## Security Testing

### Input Validation Tests
```python
@pytest.mark.security
def test_api_rejects_malicious_input():
    """Test API properly validates and rejects malicious input"""
    malicious_inputs = [
        "'; DROP TABLE users; --",
        "<script>alert('xss')</script>",
        "../../etc/passwd"
    ]
    
    for malicious_input in malicious_inputs:
        response = client.post("/api/v1/map", json={
            "detector_output": malicious_input
        })
        assert response.status_code == 400
```

### Authentication Tests
```python
@pytest.mark.security
def test_endpoint_requires_authentication():
    """Test protected endpoints require valid authentication"""
    response = client.get("/api/v1/mappings")
    assert response.status_code == 401
    
    response = client.get("/api/v1/mappings", headers={
        "Authorization": "Bearer invalid_token"
    })
    assert response.status_code == 401
```

## Test Configuration

### Environment Setup
```python
# conftest.py
@pytest.fixture(scope="session")
def test_database():
    """Create test database for integration tests"""
    db = create_test_database()
    yield db
    cleanup_test_database(db)

@pytest.fixture
def client():
    """Create test client for API testing"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
```

### Test Markers
```python
# pytest.ini
[tool:pytest]
markers =
    unit: Unit tests
    integration: Integration tests  
    performance: Performance tests
    security: Security tests
    slow: Slow running tests
```

## Continuous Integration

### Test Execution
- Run unit tests on every commit
- Run integration tests on pull requests
- Run performance tests nightly
- Run security tests weekly

### Coverage Requirements
- Minimum 80% overall coverage
- Minimum 90% coverage for critical paths
- Block PRs that decrease coverage
- Generate coverage reports in CI

### Test Reporting
- Generate JUnit XML reports for CI
- Create coverage reports in HTML format
- Send notifications for test failures
- Track test execution trends over time