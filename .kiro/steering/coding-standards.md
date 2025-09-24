---
inclusion: always
---

# Coding Standards & Best Practices

## Code Quality Standards

### Type Safety
- Use type hints for all function parameters and return values
- Leverage Pydantic models for data validation and serialization
- Run `mypy` checks before committing code
- Prefer `Optional[T]` over `Union[T, None]`

### Error Handling
- Use custom exception classes that inherit from appropriate base exceptions
- Always include context in error messages for debugging
- Implement proper logging with structured data (JSON format)
- Use circuit breaker patterns for external service calls

### Performance Guidelines
- Use async/await for I/O operations
- Implement connection pooling for database operations
- Cache frequently accessed data with Redis
- Use batch operations when processing multiple items

### Security Practices
- Never log sensitive data (PII, API keys, raw content)
- Validate all inputs using Pydantic models
- Use parameterized queries for database operations
- Implement rate limiting on all public endpoints

## Testing Requirements

### Test Coverage
- Maintain minimum 80% test coverage
- Write unit tests for all business logic
- Include integration tests for API endpoints
- Add performance tests for critical paths

### Test Structure
```python
# Test file naming: test_<module_name>.py
# Test class naming: Test<ClassName>
# Test method naming: test_<method_name>_<scenario>

class TestMapperService:
    def test_map_detector_output_success(self):
        # Arrange
        # Act  
        # Assert
        pass
        
    def test_map_detector_output_invalid_input(self):
        # Test error cases
        pass
```

### Fixtures and Mocks
- Use pytest fixtures for common test data
- Mock external dependencies in unit tests
- Use factory patterns for test data generation
- Clean up resources in test teardown

## Documentation Standards

### Code Documentation
- Write docstrings for all public functions and classes
- Use Google-style docstrings
- Include examples in docstrings for complex functions
- Document API endpoints with OpenAPI annotations

### API Documentation
- All endpoints must have OpenAPI documentation
- Include request/response examples
- Document error responses and status codes
- Provide SDK usage examples

## Git Workflow

### Commit Messages
- Use conventional commit format: `type(scope): description`
- Types: feat, fix, docs, style, refactor, test, chore
- Keep first line under 50 characters
- Include issue numbers when applicable

### Branch Naming
- Feature branches: `feature/short-description`
- Bug fixes: `fix/short-description`
- Hotfixes: `hotfix/short-description`
- Use kebab-case for branch names

### Pull Request Requirements
- All tests must pass
- Code coverage must not decrease
- At least one reviewer approval required
- Include description of changes and testing performed