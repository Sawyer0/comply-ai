# Requirements Document

## Introduction

This specification outlines the requirements for refactoring the current monolithic llama-mapper codebase into a clean 3-microservice architecture. The goal is to improve maintainability, scalability, and developer experience by creating clear service boundaries and eliminating duplicate implementations.

The current codebase has grown organically and suffers from:
- Duplicate risk scoring implementations
- Inconsistent engine organization 
- Scattered configuration management
- Mixed concerns across directories
- Complex interdependencies

The refactoring will create three focused microservices with clear responsibilities and clean APIs.

## Requirements

### Requirement 1: Service Boundary Definition

**User Story:** As a system architect, I want clearly defined service boundaries so that each microservice has a single responsibility and minimal coupling.

#### Acceptance Criteria

1. WHEN the refactoring is complete THEN the system SHALL consist of exactly 3 microservices
2. WHEN examining service responsibilities THEN each service SHALL have a clearly defined domain boundary
3. WHEN services communicate THEN they SHALL use well-defined HTTP APIs with documented contracts
4. WHEN a service fails THEN other services SHALL continue operating with graceful degradation
5. IF a service needs data from another service THEN it SHALL use the service's public API, not direct database access

### Requirement 2: Detector Orchestration Service

**User Story:** As a compliance engineer, I want a dedicated orchestration service so that detector coordination is handled independently from analysis and mapping.

#### Acceptance Criteria

1. WHEN the orchestration service receives a request THEN it SHALL coordinate multiple detectors efficiently
2. WHEN detectors are unhealthy THEN the service SHALL implement circuit breaker patterns and failover
3. WHEN routing decisions are made THEN the service SHALL apply policy enforcement consistently
4. WHEN service discovery is needed THEN the service SHALL automatically discover and register detectors
5. IF detector health changes THEN the service SHALL update routing decisions accordingly

### Requirement 3: Analysis Service

**User Story:** As a data analyst, I want a dedicated analysis service so that pattern recognition, risk scoring, and compliance mapping are handled by specialized components.

#### Acceptance Criteria

1. WHEN detector results are received THEN the service SHALL perform pattern recognition analysis
2. WHEN risk scoring is requested THEN the service SHALL calculate comprehensive risk scores using ML models
3. WHEN compliance mapping is needed THEN the service SHALL map findings to appropriate frameworks (SOC2, ISO27001, HIPAA)
4. WHEN quality evaluation is performed THEN the service SHALL monitor analysis quality and detect degradation
5. IF ML models are unavailable THEN the service SHALL fall back to rule-based analysis
6. WHEN RAG capabilities are needed THEN the service SHALL provide regulatory knowledge enhancement

### Requirement 4: Mapper Service

**User Story:** As an API consumer, I want a dedicated mapping service so that core mapping functionality and model serving are optimized for performance.

#### Acceptance Criteria

1. WHEN mapping requests are received THEN the service SHALL generate structured responses using fine-tuned models
2. WHEN model inference is performed THEN the service SHALL use high-performance backends (vLLM/TGI)
3. WHEN models are unavailable THEN the service SHALL provide template-based fallback responses
4. WHEN response validation is needed THEN the service SHALL ensure JSON schema compliance
5. IF model serving fails THEN the service SHALL implement circuit breakers and graceful degradation
6. WHEN model training is required THEN the service SHALL support LoRA fine-tuning workflows

### Requirement 5: Clean Code Organization

**User Story:** As a developer, I want a well-organized codebase so that I can easily navigate, understand, and maintain the system.

#### Acceptance Criteria

1. WHEN examining the codebase THEN duplicate implementations SHALL be eliminated
2. WHEN looking for functionality THEN related components SHALL be grouped logically
3. WHEN adding new features THEN the structure SHALL follow consistent patterns
4. WHEN reviewing code THEN each service SHALL have clear separation of concerns
5. IF configuration is needed THEN it SHALL be organized by domain and purpose

### Requirement 6: Service Communication

**User Story:** As a system integrator, I want well-defined service contracts so that services can communicate reliably and be tested independently.

#### Acceptance Criteria

1. WHEN services communicate THEN they SHALL use HTTP APIs with OpenAPI specifications
2. WHEN request/response formats are defined THEN they SHALL be versioned and backward compatible
3. WHEN service calls are made THEN they SHALL include proper error handling and retries
4. WHEN authentication is required THEN services SHALL use consistent API key mechanisms
5. IF service contracts change THEN they SHALL maintain backward compatibility or provide migration paths

### Requirement 7: ML Component Distribution

**User Story:** As an ML engineer, I want ML components properly distributed across services so that each service has the appropriate ML capabilities for its domain.

#### Acceptance Criteria

1. WHEN ML models are deployed THEN they SHALL be placed in the service that uses them most
2. WHEN analysis is performed THEN ML-enhanced capabilities SHALL be in the Analysis Service
3. WHEN mapping is performed THEN core mapping models SHALL be in the Mapper Service
4. WHEN ML fails THEN each service SHALL have rule-based fallback mechanisms
5. IF model training is needed THEN it SHALL be centralized in the appropriate service

### Requirement 8: Configuration Management

**User Story:** As a DevOps engineer, I want centralized configuration management so that each service can be configured independently while maintaining consistency.

#### Acceptance Criteria

1. WHEN services start THEN they SHALL load configuration from environment variables and config files
2. WHEN configuration changes THEN services SHALL support hot-reload where appropriate
3. WHEN deploying services THEN each SHALL have independent configuration management
4. WHEN secrets are needed THEN they SHALL be managed securely per service
5. IF configuration is invalid THEN services SHALL fail fast with clear error messages

### Requirement 9: Testing Strategy

**User Story:** As a QA engineer, I want comprehensive testing capabilities so that each service can be tested independently and integration points are validated.

#### Acceptance Criteria

1. WHEN unit tests are run THEN each service SHALL have independent test suites
2. WHEN integration tests are performed THEN service contracts SHALL be validated
3. WHEN performance tests are needed THEN each service SHALL be testable independently
4. WHEN mocking is required THEN service dependencies SHALL be easily mockable
5. IF tests fail THEN they SHALL provide clear feedback about the failure location

### Requirement 10: Migration Strategy

**User Story:** As a project manager, I want a phased migration approach so that the refactoring can be done incrementally with minimal risk.

#### Acceptance Criteria

1. WHEN migration begins THEN it SHALL follow a clear phase-by-phase approach
2. WHEN each phase completes THEN the system SHALL remain functional
3. WHEN services are extracted THEN existing functionality SHALL be preserved
4. WHEN APIs are created THEN they SHALL maintain backward compatibility during transition
5. IF migration issues occur THEN there SHALL be clear rollback procedures