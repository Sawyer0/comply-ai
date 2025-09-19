# Implementation Plan

- [ ] 1. Set up analysis module structure and core interfaces
  - Create directory structure for analysis module components
  - Define base interfaces and abstract classes for analysis pipeline
  - Set up module imports and package structure
  - _Requirements: 5.1, 5.2_

- [ ] 2. Implement input validation and schema management
- [ ] 2.1 Create input schema validation system
  - Write AnalysisRequest model with field validation and bounds checking
  - Implement custom validators for coverage values and evidence refs
  - Create JSON schema file (AnalystInput.schema.json) for request validation
  - _Requirements: 5.6, 4.4_

- [ ] 2.2 Implement evidence reference validation
  - Define ALLOWED_EVIDENCE_REFS constant with permitted field names
  - Create validator to ensure evidence_refs only reference allowed fields
  - Write unit tests for evidence reference validation edge cases
  - _Requirements: 4.5, 7.9_

- [ ] 3. Create core analysis data models
- [ ] 3.1 Implement response models with version tracking
  - Write VersionInfo model for taxonomy/frameworks/model versioning
  - Create AnalysisResponse model with all required fields and constraints
  - Implement BatchAnalysisRequest and BatchAnalysisResponse models
  - _Requirements: 5.9, 1.5_

- [ ] 3.2 Implement error response models
  - Create AnalysisErrorResponse with error types and fallback modes
  - Define error classification enums and response structures
  - Write validation for error response field constraints
  - _Requirements: 4.3, 6.6_

- [ ] 4. Build Phi-3 Mini model server integration
- [ ] 4.1 Implement model loading and initialization
  - Create AnalysisModelServer class extending existing ModelServer pattern
  - Implement Phi-3 Mini model loading with proper error handling
  - Set up model configuration with temperature and confidence cutoff
  - _Requirements: 6.1, 6.4_

- [ ] 4.2 Create deterministic prompt generation system
  - Implement _build_prompt method for structured metrics input
  - Create prompt templates for different analysis scenarios
  - Write unit tests for prompt consistency and determinism
  - _Requirements: 4.1, 6.1_

- [ ] 4.3 Implement confidence computation and governance
  - Create _compute_confidence method with calibrated scoring
  - Implement _apply_confidence_governance with config-driven cutoffs
  - Add confidence_cutoff_used tracking in responses
  - _Requirements: 6.6, 1.5_

- [ ] 5. Create template fallback system
- [ ] 5.1 Implement analysis templates for common scenarios
  - Create AnalysisTemplates class with coverage gap templates
  - Implement false positive tuning and incident summary templates
  - Add insufficient data template with proper error handling
  - _Requirements: 4.3, 6.5_

- [ ] 5.2 Build template selection and fallback logic
  - Implement automatic template selection based on input patterns
  - Create fallback trigger logic for low confidence and schema failures
  - Write unit tests for template selection accuracy
  - _Requirements: 4.3, 6.6_

- [ ] 6. Implement JSON schema validation with fallback
- [ ] 6.1 Create schema validator extending existing JSONValidator
  - Implement AnalysisValidator class with schema validation
  - Add validate_and_fallback method with automatic template fallback
  - Create schema validation metrics collection
  - _Requirements: 4.2, 4.3_

- [ ] 6.2 Build schema validation error handling
  - Implement graceful handling of schema validation failures
  - Create detailed error messages for validation failures
  - Add fallback response generation for invalid outputs
  - _Requirements: 4.3, 7.6_

- [ ] 7. Create OPA policy generation and validation
- [ ] 7.1 Implement OPA policy generator
  - Create OPAPolicyGenerator class with Rego snippet generation
  - Implement generate_coverage_policy and generate_threshold_policy methods
  - Add OPA compilation validation using opa check command
  - _Requirements: 3.3, 3.4, 7.8_

- [ ] 7.2 Build OPA validation pipeline
  - Implement validate_rego method with compilation checking
  - Create unit tests for policy generation accuracy
  - Add integration tests for OPA compilation success
  - _Requirements: 3.4, 7.8_

- [ ] 8. Implement PII redaction and security measures
- [ ] 8.1 Create PII detection and redaction system
  - Implement _redact_pii method with pattern matching
  - Create comprehensive PII pattern detection (SSN, email, phone, etc.)
  - Add unit tests for PII redaction on all free-text fields
  - _Requirements: 4.4, 7.9_

- [ ] 8.2 Build security validation pipeline
  - Implement automated PII scanning for all response fields
  - Create security validation tests for notes and reason fields
  - Add pre-response security checks with failure handling
  - _Requirements: 4.4, 7.9_

- [ ] 8.3 Implement security headers and log scrubbing
  - Enforce security headers (CORS, CSP, auth headers)
  - Implement request/response log scrubbing for sensitive data
  - Create CI test to prove PII redaction on all free-text fields
  - Add security header validation and compliance checking
  - _Requirements: 4.4, 7.9_

- [ ] 9. Create batch processing with idempotency
- [ ] 9.1 Implement batch request processing
  - Create analyze_batch method with concurrent processing
  - Implement per-item error handling and status tracking
  - Add batch size validation (≤100 items) and rate limiting
  - _Requirements: 5.7, 1.1_

- [ ] 9.2 Build idempotency key management
  - Implement IdempotencyCache for request deduplication
  - Create idempotency key validation and storage
  - Add cache expiration and cleanup mechanisms
  - _Requirements: 5.7_

- [ ] 10. Implement FastAPI endpoints and routing
- [ ] 10.1 Create single analysis endpoint
  - Implement POST /analyze endpoint with request/response handling
  - Add input validation, processing, and error response logic
  - Create endpoint-level metrics collection and logging
  - _Requirements: 5.1, 5.6_

- [ ] 10.2 Create batch analysis endpoint
  - Implement POST /analyze/batch with idempotency key handling
  - Add batch processing logic with per-item status tracking
  - Create batch-specific error handling and response formatting
  - _Requirements: 5.7, 5.8_

- [ ] 10.3 Generate OpenAPI specification and conformance testing
  - Generate OpenAPI spec for /analyze and /analyze/batch endpoints
  - Include error shapes, headers, and response models in specification
  - Create HTTP replay tests using golden cases for API conformance
  - Add OpenAPI validation in CI pipeline
  - _Requirements: 5.1, 5.7_

- [ ] 10.4 Implement authentication and abuse protection
  - Add API key authentication with rotation script
  - Implement per-tenant rate limiting with configurable thresholds
  - Create basic WAF rules for request filtering and abuse prevention
  - Add authentication middleware and security headers
  - _Requirements: 5.1, 7.9_

- [ ] 10.5 Build timeout, retry, and circuit breaker system
  - Implement client/server timeout configuration
  - Add model server retry logic with exponential backoff and jitter
  - Create circuit breaker to force template fallback when model unhealthy
  - Add timeout handling for batch processing
  - _Requirements: 6.5, 4.3_

- [ ] 11. Build quality evaluation and monitoring system
- [ ] 11.1 Implement quality evaluator
  - Create QualityEvaluator class with golden dataset loading
  - Implement evaluate_batch method with rubric scoring
  - Add drift detection with calculate_drift_score method
  - _Requirements: 7.1, 7.2, 7.5_

- [ ] 11.2 Create evaluation metrics and alerting
  - Implement evaluation metrics collection and storage
  - Create alerting logic for quality degradation detection
  - Add weekly evaluation scheduling and reporting
  - _Requirements: 7.5, 7.6, 7.7_

- [ ] 12. Implement comprehensive metrics collection
- [ ] 12.1 Create analysis-specific metrics collector
  - Extend MetricsCollector for analysis module metrics
  - Implement record_analysis_request and record_schema_validation methods
  - Add record_quality_score and performance metrics collection
  - _Requirements: 5.1, 7.6, 7.7_

- [ ] 12.2 Build alerting and monitoring pipeline
  - Implement alerting logic for schema validation rate < 98%
  - Create fallback rate monitoring with 10% threshold alerts
  - Add latency SLO monitoring and alerting for p95 targets
  - _Requirements: 7.6, 7.7, 5.8_

- [ ] 12.3 Create SLO documentation and load testing
  - Document target SLOs (p95 latency per CPU/GPU profile, availability targets)
  - Define expected req/sec/tenant capacity and scaling thresholds
  - Implement k6/Locust smoke tests in CI pipeline
  - Add staged load testing for performance validation
  - _Requirements: 5.8, 7.6_

- [ ] 12.4 Build cost monitoring and guardrails
  - Create instance sizing matrix with cost optimization recommendations
  - Implement autoscale thresholds based on load and cost efficiency
  - Add cost anomaly detection and alerting
  - Create toggle for CPU quantization vs GPU deployment modes
  - _Requirements: 5.5, 6.1_

- [ ] 12.5 Create runbooks and alert routing
  - Write playbooks for schema validation drops and fallback spikes
  - Create runbooks for OPA compile failures and model 5xx errors
  - Implement paging policy and ownership assignment
  - Add escalation procedures for critical service issues
  - _Requirements: 7.6, 7.7, 7.8_

- [ ] 13. Create configuration management integration
- [ ] 13.1 Implement analysis configuration system
  - Extend ConfigManager for analysis module settings
  - Add confidence cutoff, temperature, and model path configuration
  - Create environment-specific configuration loading
  - _Requirements: 6.6, 6.1_

- [ ] 13.2 Build version info management
  - Implement version tracking for taxonomy, frameworks, and model
  - Create version info loading and response integration
  - Add version change detection and logging
  - _Requirements: 5.9_

- [ ] 14. Write comprehensive unit tests
- [ ] 14.1 Create model server unit tests
  - Write tests for prompt generation consistency and determinism
  - Test confidence computation accuracy and calibration
  - Add tests for PII redaction and security validation
  - _Requirements: 4.1, 6.6, 7.9_

- [ ] 14.2 Create validation and template tests
  - Write tests for schema validation and fallback logic
  - Test template selection accuracy for different scenarios
  - Add tests for OPA policy generation and compilation
  - _Requirements: 4.3, 7.8_

- [ ] 15. Build integration tests
- [ ] 15.1 Create end-to-end API tests
  - Write integration tests for /analyze endpoint with full pipeline
  - Test /analyze/batch endpoint with idempotency and error handling
  - Add performance tests for latency SLO validation
  - _Requirements: 5.1, 5.7, 5.8_

- [ ] 15.2 Create OPA integration tests
  - Write tests for OPA compilation validation in CI pipeline
  - Test policy generation accuracy with real-world scenarios
  - Add regression tests for Rego syntax correctness
  - _Requirements: 7.8_

- [ ] 16. Implement CI/CD pipeline integration
- [ ] 16.1 Create automated OPA validation pipeline
  - Implement CI step for opa check on golden dataset outputs
  - Add pre-deployment OPA compilation validation
  - Create pipeline failure handling for compilation errors
  - _Requirements: 7.8_

- [ ] 16.2 Build quality gate validation
  - Implement automated quality evaluation in CI pipeline
  - Add schema validation rate checking (≥98% requirement)
  - Create deployment blocking for quality failures
  - _Requirements: 7.2, 7.6_

- [ ] 17. Create deployment configuration
- [ ] 17.1 Build Docker container configuration
  - Create Dockerfile extending llama-mapper-base image
  - Add Phi-3 Mini model and analysis dependencies
  - Configure environment variables and resource limits
  - _Requirements: 5.2, 6.1_

- [ ] 17.2 Create Kubernetes deployment manifests
  - Write deployment.yaml with proper resource allocation
  - Create service.yaml for endpoint exposure
  - Add canary deployment configuration with 10% traffic allocation
  - _Requirements: 5.5, 6.5_

- [ ] 17.3 Implement backup and disaster recovery
  - Create database backup and restore procedures
  - Implement S3 lifecycle policies and WORM compliance checks
  - Document RPO/RTO targets and recovery procedures
  - Add backup validation and restore drill automation
  - _Requirements: 5.2, 4.4_

- [ ] 18. Implement service integration and wiring
- [ ] 18.1 Create service factory and dependency injection
  - Implement create_analysis_app factory function
  - Wire together all analysis module components
  - Add proper dependency injection and configuration loading
  - _Requirements: 5.1, 5.2_

- [ ] 18.2 Build service startup and health checks
  - Implement service initialization and model loading
  - Create health check endpoints for readiness and liveness
  - Add graceful shutdown handling and resource cleanup
  - _Requirements: 5.2, 6.5_