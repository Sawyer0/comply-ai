# Requirements Document

## Introduction

This feature involves fine-tuning a Llama-3-8B-Instruct model to create a "Mapper" that normalizes outputs from various AI safety detectors into a canonical taxonomy. The system will take raw detector outputs (like "toxic" from DeBERTa or "hate/threatening" from OpenAI Moderation) and map them to standardized labels (like HARM.SPEECH.Toxicity or HARM.SPEECH.Hate.Other) for consistent audit reporting and compliance monitoring.

## Requirements

### Requirement 1

**User Story:** As a compliance engineer, I want to normalize detector outputs into a canonical taxonomy, so that I can generate consistent audit reports across different AI safety tools.

#### Acceptance Criteria

1. WHEN the system receives a detector output THEN it SHALL map the output to exactly one canonical taxonomy label
2. WHEN the mapping confidence is below 0.6 THEN the system SHALL fallback to rule-based mapping or OTHER.Unknown
3. WHEN an unmapped detector label is encountered THEN the system SHALL assign OTHER.Unknown and log the unmapped label
4. IF a detector output is malformed or invalid THEN the system SHALL assign OTHER.ModelError

### Requirement 2

**User Story:** As a data scientist, I want to fine-tune Llama-3-8B-Instruct with LoRA, so that I can achieve high-quality mapping while maintaining model efficiency.

#### Acceptance Criteria

1. WHEN fine-tuning the model THEN the system SHALL use LoRA with r=16, α=32, learning rate=2e-4
2. WHEN training THEN the system SHALL limit to 1-2 epochs with sequence length 1-2k tokens
3. WHEN generating outputs THEN the system SHALL use temperature 0.0-0.2, top_p 0.9, max_new_tokens 200
4. WHEN the model generates output THEN it SHALL produce valid JSON conforming to the mapping schema

### Requirement 3

**User Story:** As a system administrator, I want to serve the fine-tuned model efficiently, so that I can handle mapping requests at scale.

#### Acceptance Criteria

1. WHEN deploying on GPU THEN the system SHALL use vLLM for serving
2. WHEN deploying on CPU THEN the system SHALL support TGI serving
3. WHEN memory is constrained THEN the system SHALL support 8-bit quantization (AWQ/GGUF)
4. WHEN serving requests THEN the system SHALL support batch processing for efficiency

### Requirement 4

**User Story:** As a compliance officer, I want all detector outputs mapped to the canonical taxonomy, so that I can ensure comprehensive coverage of safety categories.

#### Acceptance Criteria

1. WHEN processing DeBERTa outputs THEN the system SHALL map toxic→HARM.SPEECH.Toxicity, obscene→HARM.SPEECH.Obscenity, etc.
2. WHEN processing OpenAI Moderation outputs THEN the system SHALL map hate→HARM.SPEECH.Hate.Other, self-harm→HARM.VIOLENCE.SelfHarm, etc.
3. WHEN processing Llama Guard outputs THEN the system SHALL map violence→HARM.VIOLENCE.Physical, pii→PII.Other, etc.
4. WHEN processing regex PII outputs THEN the system SHALL map ssn→PII.Identifier.SSN, email→PII.Contact.Email, etc.

### Requirement 5

**User Story:** As a developer, I want a structured dataset preparation pipeline, so that I can generate high-quality training data from detector mappings.

#### Acceptance Criteria

1. WHEN preparing training data THEN the system SHALL load detector YAML configurations from pillars-detectors/ directory
2. WHEN generating training pairs THEN the system SHALL create input-output examples for each detector mapping
3. WHEN validating mappings THEN the system SHALL ensure all target labels exist in taxonomy.yaml
4. WHEN creating the dataset THEN the system SHALL format examples as instruction-following JSON tasks
5. WHEN generating training data THEN the system SHALL support synthetic examples (regex-generated PII, synthetic jailbreak prompts) to balance training sets

### Requirement 6

**User Story:** As a quality assurance engineer, I want JSON schema validation, so that I can ensure mapping outputs are structurally correct.

#### Acceptance Criteria

1. WHEN the model generates output THEN the system SHALL validate against pillars-detectors/schema.json
2. WHEN validation fails THEN the system SHALL retry with adjusted generation parameters
3. WHEN retries are exhausted THEN the system SHALL fallback to rule-based mapping
4. WHEN calculating confidence THEN the system SHALL use model logit softmax probability with calibrated thresholds
5. WHEN configuring confidence thresholds THEN the system SHALL support YAML-configurable thresholds with default 0.6 fallback
6. WHEN using fallback mapping THEN the system SHALL log the failure for model improvement

### Requirement 7

**User Story:** As a data governance specialist, I want versioned taxonomy and detector configurations, so that I can track changes and maintain audit trails.

#### Acceptance Criteria

1. WHEN updating taxonomy THEN the system SHALL maintain version numbers in taxonomy.yaml
2. WHEN modifying detector mappings THEN the system SHALL version individual detector YAML files
3. WHEN deploying changes THEN the system SHALL support migration between taxonomy versions with automated label remapping
4. WHEN auditing THEN the system SHALL provide clear lineage from detector output to canonical label
5. WHEN generating audit reports THEN the system SHALL include taxonomy version, model version, and framework mapping version tags

### Requirement 8

**User Story:** As a security officer, I want privacy-first data handling, so that I can ensure compliance with data protection regulations.

#### Acceptance Criteria

1. WHEN processing detector inputs THEN the system SHALL NOT persist raw detector inputs by default
2. WHEN logging THEN the system SHALL store only metadata (tenant ID, detector type, taxonomy hit)
3. WHEN isolating tenants THEN the system SHALL prevent cross-tenant data queries and ensure tenant-scoped access
4. WHEN encrypting data THEN the system SHALL use AES256 with KMS (BYOK supported)
5. WHEN managing secrets THEN the system SHALL use Hashicorp Vault or AWS Secrets Manager

### Requirement 9

**User Story:** As a compliance auditor, I want framework mapping and reporting capabilities, so that I can generate audit evidence for SOC 2, ISO 27001, and HIPAA.

#### Acceptance Criteria

1. WHEN mapping taxonomy labels THEN the system SHALL support SOC 2 CC7.2, ISO 27001 A.12.4.1, and HIPAA §164.308(a)
2. WHEN generating reports THEN the system SHALL support PDF, CSV, and JSON formats
3. WHEN creating audit reports THEN the system SHALL include coverage %, incidents, MTTR, and control mapping
4. WHEN adding new compliance frameworks THEN the system SHALL require approval process and version bump in frameworks.yaml
5. WHEN versioning reports THEN the system SHALL embed taxonomy version, model version, and frameworks version tags

### Requirement 10

**User Story:** As a platform engineer, I want containerized deployment with FastAPI, so that I can deploy the service reliably across environments.

#### Acceptance Criteria

1. WHEN building the service THEN the system SHALL use Python with FastAPI and Hugging Face stack
2. WHEN packaging THEN the system SHALL provide Docker image and Helm chart
3. WHEN serving requests THEN the system SHALL expose a FastAPI /map endpoint
4. WHEN scaling THEN the system SHALL start with 1-2 replicas and support autoscaling

### Requirement 11

**User Story:** As an SRE, I want comprehensive observability and performance gates, so that I can monitor service health and ensure SLA compliance.

#### Acceptance Criteria

1. WHEN monitoring THEN the system SHALL track request count, schema-valid %, fallback %, and latency P50/P95
2. WHEN evaluating quality THEN the system SHALL achieve ≥95% schema-valid outputs and ≥90% taxonomy F1
3. WHEN measuring performance THEN the system SHALL achieve P95 latency ≤250ms CPU / ≤120ms GPU
4. WHEN creating golden test cases THEN the system SHALL ensure ≥100 cases per detector covering ≥80% of taxonomy categories
5. WHEN deploying THEN GitHub Actions SHALL block merges if quality thresholds not met

### Requirement 12

**User Story:** As a data engineer, I want immutable storage with retention policies, so that I can maintain audit trails while managing storage costs.

#### Acceptance Criteria

1. WHEN storing outputs THEN the system SHALL use S3 immutable bucket with 90-day hot, 1-year cold WORM storage
2. WHEN storing normalized data THEN the system SHALL use ClickHouse/Postgres with 90-day retention
3. WHEN archiving THEN the system SHALL automatically transition data from hot to cold storage
4. WHEN querying THEN the system SHALL support both real-time and historical data access