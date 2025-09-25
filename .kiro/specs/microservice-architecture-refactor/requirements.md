# Requirements Document

## Introduction

This specification outlines the requirements for refactoring the current production-grade llama-mapper codebase into a clean 3-microservice architecture following Single Responsibility Principle (SRP) and proper distributed system design patterns. This is a comprehensive consolidation and reorganization effort that must preserve all existing functionality while creating a maintainable, scalable architecture.

The current codebase is a sophisticated production system with:
- Multiple ML model serving backends (vLLM, TGI)
- Comprehensive CLI tooling and database management
- Advanced analysis engines with statistical components
- Production monitoring and observability
- Security features including WAF integration
- Quality assurance and evaluation systems
- Complex configuration management
- Extensive testing infrastructure
- Performance optimization components

Current architectural challenges:
- Duplicate risk scoring implementations (3+ versions)
- Inconsistent engine organization (files vs directories)
- Scattered configuration across 10+ files
- Mixed concerns in infrastructure layer
- Complex interdependencies between components
- Outdated documentation not reflecting current implementation
- CLI and database components spread across multiple locations

The refactoring will consolidate all functionality into three focused microservices with clear domain boundaries, proper separation of concerns, and distributed system resilience patterns.

## Requirements

### Requirement 1: Comprehensive Service Boundary Definition

**User Story:** As a system architect, I want clearly defined service boundaries following SRP and distributed system principles so that each microservice has a single responsibility, minimal coupling, and can be deployed independently.

#### Acceptance Criteria

1. WHEN the refactoring is complete THEN the system SHALL consist of exactly 3 microservices (Detector Orchestration, Analysis Service, Mapper Service) with no shared code dependencies
2. WHEN model serving is implemented THEN backends (vLLM, TGI, CPU fallback) SHALL be internal implementation details within services, not separate services
3. WHEN examining service responsibilities THEN each service SHALL follow Single Responsibility Principle with clear domain boundaries
4. WHEN services communicate THEN they SHALL use clean, well-defined HTTP APIs with OpenAPI 3.0 specifications
5. WHEN a service fails THEN other services SHALL continue operating with circuit breakers and graceful degradation
6. WHEN database access is needed THEN each service SHALL have its own database schema or use service APIs
7. WHEN configuration is loaded THEN each service SHALL have independent configuration management
8. WHEN CLI operations are performed THEN they SHALL be distributed appropriately across services
9. IF cross-cutting concerns exist THEN they SHALL be handled through shared libraries or service mesh patterns

### Requirement 2: Detector Orchestration Service (Consolidation)

**User Story:** As a compliance engineer, I want a consolidated orchestration service that preserves all existing detector coordination functionality while providing clean service boundaries.

#### Acceptance Criteria

1. WHEN the orchestration service is deployed THEN it SHALL preserve all existing detector coordination capabilities
2. WHEN detectors are managed THEN the service SHALL maintain the current registry, health monitoring, and circuit breaker implementations
3. WHEN policy enforcement occurs THEN the service SHALL consolidate all OPA policy management and conflict resolution logic
4. WHEN service discovery runs THEN it SHALL preserve the existing configuration reloader and detector discovery mechanisms
5. WHEN rate limiting is applied THEN the service SHALL maintain all current rate limiting and authentication features
6. WHEN caching is used THEN the service SHALL consolidate idempotency and response caching with Redis backend
7. WHEN metrics are collected THEN the service SHALL preserve all Prometheus metrics and monitoring capabilities
8. WHEN CLI operations are needed THEN the service SHALL provide CLI commands for policy management and detector operations
9. WHEN async job processing is required THEN the service SHALL maintain the job manager and async processing capabilities
10. WHEN multi-tenancy is enforced THEN the service SHALL provide tenant isolation, routing, and tenant-specific configurations
11. WHEN plugins are used THEN the service SHALL support detector plugins, policy plugins, and extension points
12. WHEN pipelines are executed THEN the service SHALL provide orchestration pipelines, validation pipelines, and audit pipelines

### Requirement 3: Analysis Service (Comprehensive Consolidation)

**User Story:** As a data analyst, I want a comprehensive analysis service that consolidates all analysis engines, statistical components, quality systems, and RAG capabilities while maintaining production-grade performance.

#### Acceptance Criteria

1. WHEN analysis is performed THEN the service SHALL consolidate all pattern recognition engines (temporal, frequency, correlation, anomaly detection)
2. WHEN risk scoring is calculated THEN the service SHALL merge all risk scoring implementations into a single, comprehensive system
3. WHEN compliance intelligence is needed THEN the service SHALL integrate all compliance framework mapping capabilities
4. WHEN statistical analysis occurs THEN the service SHALL provide all current statistical analyzers and pattern classifiers
5. WHEN quality evaluation runs THEN the service SHALL consolidate the quality alerting system, degradation detection, and monitoring
6. WHEN RAG processing is required THEN the service SHALL integrate the complete RAG system with knowledge base and retrieval
7. WHEN threshold optimization is needed THEN the service SHALL provide the statistical optimizer and impact simulator
8. WHEN business relevance is assessed THEN the service SHALL include pattern evolution tracking and multi-pattern analysis
9. WHEN model serving is required THEN the service SHALL internally use appropriate model backends (Phi-3 via vLLM/TGI/CPU as implementation details)
10. WHEN CLI operations are needed THEN the service SHALL provide analysis-specific CLI commands
11. WHEN configuration is managed THEN the service SHALL consolidate all analysis-related configuration files
12. WHEN fallback mechanisms activate THEN the service SHALL provide comprehensive rule-based alternatives
13. WHEN evaluation scheduling is required THEN the service SHALL maintain weekly evaluation services and alert management
14. WHEN privacy controls are enforced THEN the service SHALL consolidate all privacy-first architecture (metadata-only logging, content scrubbing, no raw content persistence)
15. WHEN ML models are deployed THEN the service SHALL provide specialized analysis models (Phi-3, embeddings) with proper fallback mechanisms
16. WHEN multi-tenancy is supported THEN the service SHALL provide tenant-specific analytics, data isolation, and resource quotas
17. WHEN plugins are integrated THEN the service SHALL support analysis engine plugins, ML plugins, and quality plugins
18. WHEN pipelines are managed THEN the service SHALL provide training pipelines, analysis pipelines, and batch processing pipelines

### Requirement 4: Mapper Service (Core Engine Consolidation)

**User Story:** As an API consumer, I want a high-performance mapping service that consolidates all core mapping functionality, model serving, training infrastructure, and production capabilities.

#### Acceptance Criteria

1. WHEN mapping requests are processed THEN the service SHALL consolidate all core mapping logic and response generation
2. WHEN model inference occurs THEN the service SHALL provide unified model serving with vLLM, TGI, and CPU fallback backends
3. WHEN training is performed THEN the service SHALL consolidate all training infrastructure (LoRA, Phi-3, model loading, checkpoints)
4. WHEN production deployment happens THEN the service SHALL include model versioning, A/B testing, and production utilities
5. WHEN validation is required THEN the service SHALL provide comprehensive input/output validation and JSON schema compliance
6. WHEN optimization is needed THEN the service SHALL include performance optimization and resource management
7. WHEN generation occurs THEN the service SHALL consolidate all content generation and context management capabilities
8. WHEN evaluation is performed THEN the service SHALL provide model evaluation and quality metrics
9. WHEN CLI operations are needed THEN the service SHALL provide mapping-specific CLI commands and utilities
10. WHEN configuration is managed THEN the service SHALL handle all mapping and model-related configuration
11. WHEN fallback mechanisms activate THEN the service SHALL provide template-based and rule-based alternatives
12. WHEN cost monitoring is required THEN the service SHALL include cost tracking and optimization features
13. WHEN security is enforced THEN the service SHALL maintain authentication, rate limiting, and security validation
14. WHEN batch processing is needed THEN the service SHALL support batch operations and pipeline orchestration
15. WHEN privacy controls are enforced THEN the service SHALL maintain privacy-first design (no raw content logging, metadata-only persistence)
16. WHEN ML models are served THEN the service SHALL internally use high-performance backends (vLLM/TGI) for Llama-3-8B and LoRA models
17. WHEN multi-tenancy is supported THEN the service SHALL provide tenant-specific models, data isolation, and cost tracking
18. WHEN plugins are integrated THEN the service SHALL support mapping plugins, model plugins, and validation plugins
19. WHEN pipelines are managed THEN the service SHALL provide training pipelines, deployment pipelines, and optimization pipelines

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

### Requirement 10: Database and Storage Consolidation

**User Story:** As a database administrator, I want proper database separation and consolidation so that each service has appropriate data access patterns and storage management.

#### Acceptance Criteria

1. WHEN database schemas are designed THEN each service SHALL have logical separation of data concerns
2. WHEN storage is managed THEN the system SHALL consolidate all storage backends (PostgreSQL, ClickHouse, Redis, S3/MinIO)
3. WHEN backup and restore operations occur THEN they SHALL be distributed appropriately across services
4. WHEN database migrations run THEN each service SHALL manage its own schema evolution
5. WHEN multi-database operations are needed THEN they SHALL use proper distributed transaction patterns
6. WHEN Azure integration is required THEN the service SHALL maintain all Azure storage and database capabilities
7. IF data consistency is required THEN services SHALL use appropriate consistency patterns (eventual consistency, SAGA, etc.)

### Requirement 11: CLI and Tooling Consolidation

**User Story:** As a DevOps engineer, I want consolidated CLI tooling so that each service provides appropriate command-line interfaces while maintaining unified user experience.

#### Acceptance Criteria

1. WHEN CLI commands are executed THEN they SHALL be logically distributed across the appropriate services
2. WHEN detector management is needed THEN the orchestration service SHALL provide detector CLI commands
3. WHEN analysis operations are performed THEN the analysis service SHALL provide analysis CLI commands
4. WHEN mapping operations are executed THEN the mapper service SHALL provide mapping CLI commands
5. WHEN cross-service operations are needed THEN there SHALL be a unified CLI wrapper or service coordination
6. WHEN configuration validation occurs THEN each service SHALL provide its own validation commands
7. WHEN backup/restore operations run THEN they SHALL be handled by the appropriate service
8. IF plugin systems are needed THEN they SHALL be maintained in the appropriate service context

### Requirement 12: Monitoring and Observability Consolidation

**User Story:** As a site reliability engineer, I want comprehensive monitoring and observability so that each service provides appropriate metrics, logging, and tracing capabilities.

#### Acceptance Criteria

1. WHEN metrics are collected THEN each service SHALL provide Prometheus metrics appropriate to its domain
2. WHEN logging occurs THEN all services SHALL use structured logging with correlation IDs
3. WHEN tracing is performed THEN distributed tracing SHALL work across all service boundaries
4. WHEN health checks run THEN each service SHALL provide comprehensive health endpoints
5. WHEN alerting is configured THEN it SHALL be appropriate to each service's responsibilities
6. WHEN dashboards are created THEN they SHALL provide service-specific and cross-service views
7. WHEN performance monitoring occurs THEN each service SHALL track its own performance characteristics
8. IF incident response is needed THEN each service SHALL provide appropriate debugging and diagnostic capabilities

### Requirement 13: Security and Compliance Consolidation

**User Story:** As a security engineer, I want comprehensive security measures so that each service maintains appropriate security controls and compliance capabilities.

#### Acceptance Criteria

1. WHEN authentication occurs THEN each service SHALL implement consistent API key and authentication mechanisms
2. WHEN authorization is enforced THEN role-based access control SHALL be maintained across services
3. WHEN WAF protection is active THEN it SHALL be appropriately distributed or centralized
4. WHEN secrets management occurs THEN each service SHALL have secure secret handling
5. WHEN audit logging happens THEN it SHALL be comprehensive and distributed appropriately
6. WHEN compliance frameworks are supported THEN they SHALL be maintained in the analysis service
7. WHEN encryption is required THEN it SHALL be implemented consistently across services
8. IF security incidents occur THEN each service SHALL provide appropriate security monitoring and response

### Requirement 14: Performance and Scalability Consolidation

**User Story:** As a performance engineer, I want optimized performance characteristics so that each service can scale independently while maintaining overall system performance.

#### Acceptance Criteria

1. WHEN load testing occurs THEN each service SHALL be independently scalable
2. WHEN caching is implemented THEN it SHALL be appropriate to each service's data patterns
3. WHEN connection pooling is used THEN each service SHALL manage its own database connections
4. WHEN batch processing occurs THEN it SHALL be optimized for each service's workload characteristics
5. WHEN resource optimization is performed THEN each service SHALL have appropriate resource management
6. WHEN performance benchmarking runs THEN each service SHALL have its own performance targets
7. WHEN auto-scaling is configured THEN it SHALL be based on service-specific metrics
8. IF performance degradation occurs THEN each service SHALL have appropriate performance monitoring and alerting

### Requirement 15: Documentation and Knowledge Management

**User Story:** As a technical writer, I want comprehensive documentation so that the refactored architecture is well-documented and maintainable.

#### Acceptance Criteria

1. WHEN documentation is created THEN it SHALL reflect the actual implementation and architecture
2. WHEN API documentation is generated THEN each service SHALL have complete OpenAPI specifications
3. WHEN architectural decisions are made THEN they SHALL be documented in Architecture Decision Records (ADRs)
4. WHEN deployment guides are written THEN they SHALL cover each service independently and together
5. WHEN runbooks are created THEN they SHALL be service-specific and comprehensive
6. WHEN developer onboarding occurs THEN documentation SHALL enable quick understanding of each service
7. WHEN troubleshooting is needed THEN documentation SHALL provide clear diagnostic procedures
8. IF knowledge transfer is required THEN documentation SHALL be comprehensive enough for new team members

### Requirement 16: ML and Privacy Architecture Consolidation

**User Story:** As an ML engineer and privacy officer, I want all ML components and privacy controls properly consolidated so that the system maintains its sophisticated ML capabilities while ensuring privacy-first design.

#### Acceptance Criteria

1. WHEN ML models are distributed THEN they SHALL be placed in the appropriate service (Analysis: Phi-3 for analysis, Mapper: Llama-3-8B for mapping)
2. WHEN model serving occurs THEN each service SHALL use the appropriate backend internally (vLLM/TGI/CPU fallback as implementation details, not separate services)
3. WHEN training infrastructure is used THEN LoRA fine-tuning, model loading, and checkpoint management SHALL be centralized in Mapper Service
4. WHEN privacy controls are enforced THEN metadata-only logging SHALL be maintained across all services
5. WHEN content processing occurs THEN raw content SHALL never be persisted, only metadata and results
6. WHEN embeddings are used THEN they SHALL be properly distributed (Analysis Service for analysis embeddings)
7. WHEN fallback mechanisms activate THEN rule-based alternatives SHALL be available for all ML components
8. WHEN model evaluation occurs THEN quality metrics and drift detection SHALL be maintained
9. WHEN cost monitoring is performed THEN ML resource usage SHALL be tracked and optimized
10. IF privacy audits occur THEN the system SHALL demonstrate no raw content persistence and proper data handling

### Requirement 17: Multi-Tenancy and Isolation

**User Story:** As a platform administrator, I want comprehensive multi-tenancy support so that multiple tenants can use the system with complete data isolation and tenant-specific customization.

#### Acceptance Criteria

1. WHEN tenant data is stored THEN each service SHALL implement row-level security and tenant data isolation
2. WHEN tenant routing occurs THEN requests SHALL be routed to tenant-specific resources and configurations
3. WHEN tenant analytics are generated THEN each tenant SHALL have isolated analytics and usage tracking
4. WHEN tenant customization is needed THEN each service SHALL support tenant-specific configurations and rules
5. WHEN resource quotas are enforced THEN each tenant SHALL have configurable resource limits and monitoring
6. WHEN tenant billing is calculated THEN cost allocation SHALL be tracked per tenant across all services
7. WHEN tenant access is controlled THEN fine-grained access controls SHALL be enforced per tenant
8. IF tenant data migration is required THEN secure tenant data export/import SHALL be supported

### Requirement 18: Plugin and Extension System

**User Story:** As a system integrator, I want a comprehensive plugin system so that the platform can be extended with custom functionality without modifying core services.

#### Acceptance Criteria

1. WHEN plugins are loaded THEN each service SHALL support dynamic plugin loading and lifecycle management
2. WHEN detector plugins are used THEN the orchestration service SHALL support custom detector implementations
3. WHEN analysis plugins are deployed THEN the analysis service SHALL support custom analysis engines and algorithms
4. WHEN mapping plugins are integrated THEN the mapper service SHALL support custom mapping logic and validation
5. WHEN extension points are defined THEN they SHALL provide clear interfaces for custom functionality
6. WHEN plugin configuration is managed THEN plugins SHALL have isolated configuration and dependency management
7. WHEN plugin security is enforced THEN plugins SHALL run in sandboxed environments with limited permissions
8. IF plugin conflicts occur THEN the system SHALL provide conflict resolution and plugin compatibility validation

### Requirement 19: Advanced Pipeline Management

**User Story:** As a DevOps engineer, I want sophisticated pipeline management so that training, deployment, and processing workflows are automated and reliable.

#### Acceptance Criteria

1. WHEN training pipelines execute THEN they SHALL support LoRA training, Phi-3 training, and checkpoint management
2. WHEN deployment pipelines run THEN they SHALL support multi-stage deployment with validation gates
3. WHEN data processing pipelines operate THEN they SHALL support both batch and streaming processing
4. WHEN quality pipelines execute THEN they SHALL provide automated quality evaluation and drift detection
5. WHEN pipeline orchestration occurs THEN it SHALL support dependency management and parallel execution
6. WHEN pipeline monitoring is active THEN it SHALL provide real-time pipeline status and performance metrics
7. WHEN pipeline failures occur THEN they SHALL provide automatic retry, rollback, and notification capabilities
8. IF pipeline customization is needed THEN it SHALL support custom pipeline stages and workflow definitions

### Requirement 20: Enterprise Integration and Advanced Features

**User Story:** As an enterprise architect, I want comprehensive enterprise integration so that the system integrates seamlessly with enterprise infrastructure and compliance requirements.

#### Acceptance Criteria

1. WHEN Azure integration is used THEN the system SHALL support Azure SQL, Azure Storage, Azure Key Vault, and Azure Monitor
2. WHEN PostgreSQL extensions are utilized THEN they SHALL provide advanced query optimization and performance monitoring
3. WHEN enterprise security is enforced THEN it SHALL include field-level encryption, secrets rotation, and comprehensive audit trails
4. WHEN taxonomy management occurs THEN it SHALL support versioned canonical taxonomies with migration tools
5. WHEN schema evolution happens THEN it SHALL provide backward-compatible schema evolution with validation
6. WHEN framework mappings are updated THEN they SHALL support versioned compliance framework mappings (SOC2, ISO27001, HIPAA, GDPR)
7. WHEN canary deployments are executed THEN they SHALL coordinate across all services with automated rollback
8. WHEN A/B testing is performed THEN it SHALL support model and algorithm A/B testing with statistical validation
9. WHEN feature flags are used THEN they SHALL support gradual feature rollout with runtime flag evaluation
10. IF enterprise compliance is audited THEN the system SHALL provide comprehensive audit trails and compliance evidence

### Requirement 21: Migration Strategy and Risk Management

**User Story:** As a project manager, I want a comprehensive migration strategy so that the refactoring can be done incrementally with minimal risk to production systems.

#### Acceptance Criteria

1. WHEN migration planning occurs THEN it SHALL include detailed risk assessment and mitigation strategies
2. WHEN each migration phase executes THEN the system SHALL remain fully functional with zero downtime
3. WHEN services are extracted THEN all existing functionality SHALL be preserved and tested
4. WHEN APIs are created THEN they SHALL be clean, well-designed interfaces optimized for maintainability
5. WHEN database migrations occur THEN they SHALL be reversible and tested
6. WHEN configuration changes happen THEN they SHALL be clean and well-organized for maximum maintainability
7. WHEN deployment occurs THEN it SHALL support blue-green or canary deployment patterns
8. WHEN rollback is needed THEN there SHALL be clear, tested rollback procedures for each phase
9. WHEN testing occurs THEN it SHALL include comprehensive integration testing between old and new systems
10. IF production issues arise THEN there SHALL be immediate rollback capabilities and incident response procedures