# Implementation Plan

- [ ] Contracts compliance with `.kiro/specs/service-contracts.md`
  - [ ] Produce `MapperPayload` exactly per locked schema; validate before /map [Sec 3]
  - [ ] Implement locked coverage semantics; map partial coverage to HTTP 206 [Sec 6]
  - [ ] Apply SLAs/timeouts; respect `mapper_timeout_budget_ms` on auto-map [Sec 7]
  - [ ] Implement canonical error codes and retry-safety guidance [Sec 8]
  - [ ] Honor idempotency and caching behaviors; bypass for CRITICAL [Sec 9]
  - [ ] Propagate `version_info` and provenance into `mapping_result` notes/provenance [Sec 10]
  - [ ] Emit locked metric names for dashboards and alerts [Sec 12]
  - [ ] Enforce boundaries: orchestrator never assigns canonical taxonomy [Sec 2]
  - [ ] Security/privacy: redact content in logs/metrics; tenant isolation [Sec 11]

- [ ] 1. Set up orchestration module structure and core data models
  - Create directory structure for orchestration module components
  - Define base data models (OrchestrationRequest, OrchestrationResponse, MapperPayload)
  - Implement locked schema validation for mapper handoff payload
  - _Requirements: 1.1, 3.5_

- [ ] 2. Implement core configuration and policy management
- [ ] 2.1 Create configuration models with SLA and timeout definitions
  - Write SLAConfig model with locked timeout semantics
  - Implement OrchestrationConfig with health monitoring thresholds
  - Create configuration loading and validation system
  - _Requirements: 6.1, 7.5_

- [ ] 2.2 Build policy store interface and tenant policy models
  - Implement TenantPolicy model with OPA/Rego integration support
  - Create PolicyStore interface with migration/versioning support
  - Write policy validation logic with conflict resolution defaults
  - _Requirements: 7.1, 7.2, 3.4_

- [ ] 2.3 Implement OPA policy engine integration
  - Create OPAPolicyEngine class for policy-as-code evaluation
  - Implement detector selection, coverage, and conflict resolution via Rego
  - Add OPA policy compilation and validation
  - _Requirements: 7.1, 7.2, 1.1_

- [ ] 2.4 Create policy validation CLI tool with OPA support
  - Implement PolicyValidationCLI for pre-rollout validation
  - Add OPA policy compilation validation and Rego syntax checking
  - Create validation reports with policy bundle and Rego rule testing
  - _Requirements: 7.2_

- [ ] 3. Build detector registry and client management
- [ ] 3.1 Implement detector client abstraction
  - Create DetectorClient class with auth and communication handling
  - Implement detector capability discovery and health checking
  - Add timeout and retry logic with exponential backoff
  - _Requirements: 2.1, 2.2, 6.3_

- [ ] 3.2 Create detector registry with service discovery
  - Implement DetectorRegistry for managing available detectors
  - Add dynamic detector registration and configuration updates
  - Create detector capability matching and selection logic
  - _Requirements: 7.1, 7.3_

- [ ] 4. Implement health monitoring and circuit breaker system
- [ ] 4.1 Create health monitor with continuous checking
  - Implement HealthMonitor with 30-second unhealthy removal threshold
  - Add background health checking with configurable intervals
  - Create health status tracking and recovery detection
  - _Requirements: 2.1, 2.2, 2.5_

- [ ] 4.2 Build circuit breaker with half-open testing
  - Implement CircuitBreaker with failure threshold and recovery logic
  - Add half-open state testing with limited request sampling
  - Create circuit breaker state tracking and metrics collection
  - _Requirements: 2.2, 6.1_

- [ ] 5. Create content router with policy-driven detector selection
- [ ] 5.1 Implement content analysis and type detection
  - Create content type analysis for optimal detector selection
  - Implement content-based routing rules and heuristics
  - Add content size and format validation
  - _Requirements: 1.1, 1.2_

- [ ] 5.2 Build policy-driven routing engine with OPA integration
  - Implement tenant policy application via OPA policy evaluation
  - Create OPA-based detector selection with fallback to hardcoded rules
  - Add routing decision logging with OPA decision audit trail
  - _Requirements: 1.3, 1.4, 5.4_

- [ ] 6. Implement detector coordinator with parallel execution
- [ ] 6.1 Create parallel detector execution engine
  - Implement DetectorCoordinator with concurrent detector calls
  - Add parallel group execution with timeout management
  - Create dependency handling for sequential detector chains
  - _Requirements: 4.1, 4.4, 6.2_

- [ ] 6.2 Build error handling and retry logic
  - Implement per-detector error handling with retry strategies
  - Add timeout handling and graceful degradation
  - Create fallback routing when detectors fail
  - _Requirements: 2.3, 2.4, 4.1_

- [ ] 7. Create response aggregator with conflict resolution
- [ ] 7.1 Implement multi-detector response aggregation
  - Create ResponseAggregator with locked MapperPayload generation
  - Implement score normalization to 0-1 range with provenance tracking
  - Add coverage calculation with weighted and required-set methods
  - _Requirements: 3.1, 3.2, 3.4_

- [ ] 7.2 Build conflict resolution engine with OPA policy support
  - Implement ConflictResolver with OPA-based strategy selection
  - Add default strategies per content type with OPA policy overrides
  - Create tie-breaker logic with OPA decision audit trail
  - _Requirements: 3.3, 3.4_

- [ ] 8. Implement caching system with policy-aware invalidation
- [ ] 8.1 Create response caching with composite key generation
  - Implement cache key generation from content hash, detector set, and policy bundle
  - Add TTL management with CRITICAL priority bypass
  - Create cache invalidation on policy and detector changes
  - _Requirements: 4.1, 6.1_

- [ ] 8.2 Build idempotency management
  - Implement idempotency key handling for all endpoints (not just batch)
  - Add 24-hour key retention with cached response return
  - Create idempotency validation and conflict detection
  - _Requirements: 4.1, 4.2_

- [ ] 9. Create async job management system
- [ ] 9.1 Implement async job processing
  - Create AsyncJob model with status tracking and progress updates
  - Implement job queue management with priority handling
  - Add job status endpoints and completion notifications
  - _Requirements: 4.2, 4.3_

- [ ] 9.2 Build sync-to-async conversion logic
  - Implement automatic conversion when SLA budgets are exceeded
  - Add sync request timeout handling with async fallback
  - Create job ID generation and status tracking
  - _Requirements: 4.1, 4.2_

- [ ] 10. Implement comprehensive metrics collection
- [ ] 10.1 Create orchestration metrics collector with locked metric names
  - Implement OrchestrationMetricsCollector with guaranteed metric names
  - Add orchestrate_requests_total, detector_latency_ms, coverage_achieved metrics
  - Create circuit_breaker_state and policy_enforcement_total tracking
  - _Requirements: 5.1, 5.2, 5.5_

- [ ] 10.2 Build security and tenancy metrics
  - Implement RBAC enforcement tracking and rate limit metrics
  - Add content redaction validation for logs and metrics
  - Create tenant-specific metric isolation and aggregation
  - _Requirements: 5.1, 5.5_

- [ ] 11. Create FastAPI orchestration service
- [ ] 11.1 Implement core orchestration endpoints
  - Create POST /orchestrate endpoint with request validation
  - Implement POST /orchestrate/batch with idempotency key handling
  - Add GET /orchestrate/status/{job_id} for async job tracking
  - _Requirements: 4.1, 4.2, 4.3_

- [ ] 11.2 Build authentication and authorization middleware
  - Implement RBAC scope validation for orchestrate vs status endpoints
  - Add per-tenant rate limiting with configurable thresholds
  - Create authentication middleware with multiple auth types
  - _Requirements: 6.1, 7.5_

- [ ] 11.3 Add error handling and canonical error responses
  - Implement canonical error code table with retry safety indicators
  - Create error response formatting with fallback_used tracking
  - Add HTTP status code mapping for coverage and timeout scenarios
  - _Requirements: 3.1, 4.1, 4.4_

- [ ] 12. Implement mapper integration and auto-mapping
- [ ] 12.1 Create mapper client for auto-mapping
  - Implement MapperClient for calling existing /map endpoint
  - Add mapper timeout budget management and error handling
  - Create mapper payload validation before handoff
  - _Requirements: 3.5, 4.1_

- [ ] 12.2 Build mapper integration pipeline
  - Implement automatic mapping when auto_map_results is enabled
  - Add mapper response integration into OrchestrationResponse
  - Create mapper failure handling with partial result return
  - _Requirements: 3.5, 4.1_

- [ ] 13. Create comprehensive unit tests
- [ ] 13.1 Write core component unit tests
  - Test content router policy application and detector selection
  - Test detector coordinator parallel execution and error handling
  - Test response aggregator conflict resolution and coverage calculation
  - _Requirements: 1.1, 2.1, 3.1_

- [ ] 13.2 Write configuration and policy tests
  - Test policy validation CLI with OPA policy compilation and validation
  - Test OPA policy evaluation for detector selection and conflict resolution
  - Test SLA configuration and timeout budget management
  - Test circuit breaker state transitions and recovery logic
  - _Requirements: 2.1, 6.1, 7.2_

- [ ] 14. Build integration tests
- [ ] 14.1 Create end-to-end orchestration tests
  - Test full orchestration pipeline from request to mapper handoff
  - Test multi-detector coordination with various failure scenarios
  - Test async job processing and status tracking
  - _Requirements: 4.1, 4.2, 6.2_

- [ ] 14.2 Create health monitoring and failover tests
  - Test detector health monitoring and automatic failover
  - Test circuit breaker activation and recovery behavior
  - Test policy enforcement and coverage requirement validation
  - _Requirements: 2.1, 2.2, 5.4_

- [ ] 15. Implement performance and load testing
- [ ] 15.1 Create load testing for concurrent orchestration
  - Implement k6/Locust tests for concurrent request handling
  - Test detector coordination scalability under load
  - Validate response time targets (<2s p95) under various loads
  - _Requirements: 6.2, 6.4_

- [ ] 15.2 Build fault tolerance testing
  - Test detector failure scenarios and graceful degradation
  - Test network partition handling and recovery behavior
  - Test cache performance and invalidation under load
  - _Requirements: 2.3, 2.4, 6.1_

- [ ] 16. Create deployment configuration
- [ ] 16.1 Build Docker container and Kubernetes manifests
  - Create Dockerfile with orchestration service configuration
  - Write Kubernetes deployment with proper resource allocation
  - Add service discovery configuration for detector registry
  - _Requirements: 6.1, 6.5_

- [ ] 16.2 Create configuration management with OPA integration
  - Implement environment-specific configuration loading
  - Add OPA server configuration and policy bundle management
  - Create detector registry configuration with service discovery
  - Create policy store configuration with OPA backend integration
  - _Requirements: 7.1, 7.4_

- [ ] 17. Build monitoring and alerting
- [ ] 17.1 Create Prometheus metrics and Grafana dashboards
  - Implement Prometheus metrics endpoint with locked metric names
  - Create Grafana dashboards for orchestration KPIs
  - Add alerting rules for critical thresholds and SLA violations
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 17.2 Implement health check and readiness probes
  - Create /health endpoint with detector availability checking
  - Implement /health/ready probe for Kubernetes readiness
  - Add service startup validation and dependency checking
  - _Requirements: 6.1, 6.5_

- [ ] 18. Create service integration and wiring
- [ ] 18.1 Implement service factory and dependency injection
  - Create create_orchestration_app factory function
  - Wire together all orchestration components with proper configuration
  - Add graceful startup and shutdown handling
  - _Requirements: 6.1, 6.5_

- [ ] 18.2 Build service discovery and registration
  - Implement automatic detector discovery and registration
  - Add configuration hot-reloading for detector and policy updates
  - Create service health reporting and status endpoints
  - _Requirements: 7.1, 7.3, 7.4_