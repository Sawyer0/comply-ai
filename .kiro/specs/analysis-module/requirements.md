# Requirements Document

## Introduction

The Analysis Module is a new component that will provide concise, auditable explanations and suggested remediations from structured metrics data. This module will analyze coverage gaps, incidents, anomalies, and provide threshold tuning suggestions to improve the overall security posture of the application. The module will optionally emit machine-actionable policy diffs (OPA/CEL) to fix coverage and threshold issues.

### Contracts

This spec inherits the cross-service constraints defined in `.kiro/specs/service-contracts.md`. This module uses locked coverage semantics, does not call detectors or assign taxonomy, includes `version_info` in responses, aligns error codes/SLOs, and enforces PII redaction.

## Requirements

### Requirement 1

**User Story:** As a security analyst, I want to receive automated analysis of coverage gaps per endpoint/model/tenant, so that I can quickly identify and address security monitoring blind spots.

#### Acceptance Criteria

1. WHEN structured metrics are provided for a time period and tenant THEN the system SHALL analyze required vs observed detector coverage
2. WHEN coverage gaps are detected THEN the system SHALL generate explanations limited to 120 characters
3. WHEN coverage gaps are detected THEN the system SHALL provide remediation suggestions limited to 120 characters
4. WHEN coverage analysis is performed THEN the system SHALL reference only provided metrics without inventing data
5. WHEN analysis is complete THEN the system SHALL return confidence scores between 0.0 and 1.0

### Requirement 2

**User Story:** As a security operations engineer, I want to receive incident and anomaly summaries for spikes and detector errors, so that I can understand system health issues and their impact.

#### Acceptance Criteria

1. WHEN detector errors occur THEN the system SHALL analyze 5xx error patterns and time buckets
2. WHEN high severity hits are detected THEN the system SHALL summarize taxonomy patterns and score distributions
3. WHEN anomalies are identified THEN the system SHALL provide concise explanations of the root cause
4. WHEN incident analysis is performed THEN the system SHALL include evidence references to source metrics
5. WHEN analysis output is generated THEN the system SHALL validate against the defined JSON schema

### Requirement 3

**User Story:** As a policy administrator, I want to receive threshold and quorum tuning suggestions based on false-positive analysis, so that I can optimize detector performance and reduce noise.

#### Acceptance Criteria

1. WHEN false-positive bands are provided THEN the system SHALL analyze score ranges and FP rates
2. WHEN tuning opportunities are identified THEN the system SHALL suggest specific threshold adjustments
3. WHEN policy changes are recommended THEN the system SHALL optionally generate OPA/Rego policy diffs
4. WHEN OPA diffs are generated THEN the system SHALL ensure they compile correctly
5. WHEN tuning suggestions are made THEN the system SHALL consider impact on overall coverage requirements

### Requirement 4

**User Story:** As a compliance officer, I want all analysis outputs to be deterministic and auditable, so that I can rely on consistent results for regulatory reporting.

#### Acceptance Criteria

1. WHEN analysis is performed THEN the system SHALL use deterministic processing with low temperature (0-0.2)
2. WHEN outputs are generated THEN the system SHALL validate against JSON schema with ≥98% success rate
3. WHEN schema validation fails THEN the system SHALL fall back to template-based responses
4. WHEN analysis includes notes THEN the system SHALL redact any potential PII
5. WHEN evidence is referenced THEN the system SHALL only cite provided metric IDs

### Requirement 5

**User Story:** As a system administrator, I want the analysis module to be deployed as a separate service endpoint, so that I can scale and manage it independently from the main mapping service.

#### Acceptance Criteria

1. WHEN the service is deployed THEN it SHALL expose an /analyze endpoint via FastAPI
2. WHEN requests are received THEN the system SHALL process structured input data according to the defined schema
3. WHEN the service starts THEN it SHALL share secrets and metrics infrastructure with existing services
4. WHEN deployment occurs THEN it SHALL follow the same Docker/Helm patterns as other services
5. WHEN the service is running THEN it SHALL support canary deployment with 10% traffic allocation
6. WHEN requests are received THEN they SHALL validate against AnalystInput.schema.json with invalid requests returning HTTP 400 with error detail
7. WHEN batch requests are made THEN POST /analyze/batch SHALL accept ≤100 items, return per-item status, and be idempotent via Idempotency-Key header
8. WHEN performance is measured THEN p95 latency SHALL meet target per deployment profile (CPU/GPU) in staging load tests
9. WHEN responses are generated THEN they SHALL include version_info {taxonomy, frameworks, model} for auditability

### Requirement 6

**User Story:** As a machine learning engineer, I want the system to start with zero-shot Phi-3 Mini model and support future fine-tuning, so that I can improve performance based on real-world usage patterns.

#### Acceptance Criteria

1. WHEN the system initializes THEN it SHALL use Phi-3 Mini (3.8B) model for zero-shot inference
2. WHEN quality issues are detected THEN the system SHALL support LoRA fine-tuning with 300-800 examples
3. WHEN model inference occurs THEN it SHALL process structured metrics inputs as defined in the specification
4. WHEN outputs are generated THEN they SHALL conform to the AnalystLLMOutput schema
5. WHEN the system operates THEN it SHALL maintain a kill-switch to fallback to template responses
6. WHEN confidence is below configured cutoff THEN service SHALL return template fallback with mode: 'fallback'

### Requirement 7

**User Story:** As a quality assurance engineer, I want the system to be continuously evaluated against golden test cases, so that I can ensure consistent performance and detect drift over time.

#### Acceptance Criteria

1. WHEN evaluation is performed THEN the system SHALL be tested against ≥150 golden examples
2. WHEN quality metrics are measured THEN JSON schema validity SHALL be ≥98%
3. WHEN human evaluation occurs THEN correctness/clarity rubric SHALL score ≥4/5
4. WHEN OPA snippets are generated THEN they SHALL compile successfully
5. WHEN drift monitoring runs THEN it SHALL alert if weekly rubric scores fall below 3.7/5
6. WHEN schema validation rate drops below 98% THEN system SHALL fire alert within 5 minutes
7. WHEN template fallback rate exceeds 10% over 15 minutes THEN system SHALL fire alert
8. WHEN OPA snippets are generated THEN they SHALL pass automated opa check compilation validation
9. WHEN free-text fields are processed THEN they SHALL pass PII-redaction check before returning