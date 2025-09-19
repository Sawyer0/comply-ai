# Implementation Plan

- [x] 1. Set up project structure and core configuration management








  - Create Python package structure with FastAPI, Hugging Face, and PEFT dependencies
  - Implement ConfigManager class for YAML-configurable settings (confidence thresholds, model parameters)
  - Set up logging configuration with metadata-only privacy-first approach
  - _Requirements: 8.2, 6.5_
-

- [x] 2. Implement taxonomy and detector configuration loading




  - [x] 2.1 Create TaxonomyLoader for pillars-detectors/taxonomy.yaml


    - Load and validate taxonomy structure with version tracking
    - Implement taxonomy label validation and category grouping
    - _Requirements: 7.1, 7.4_

  - [x] 2.2 Create DetectorConfigLoader for pillars-detectors/*.yaml files


    - Load detector mapping configurations with version support
    - Validate mappings against taxonomy labels
    - Support for detector versioning and change tracking
    - _Requirements: 5.1, 5.3, 7.2_

  - [x] 2.3 Implement FrameworkMapper for pillars-detectors/frameworks.yaml


    - Load compliance framework mappings (SOC2, ISO27001, HIPAA)
    - Support framework expansion with approval workflow
    - Version tracking for framework mapping changes
    - _Requirements: 9.1, 9.4_

- [x] 3. Build dataset preparation pipeline







  - [x] 3.1 Create TrainingDataGenerator for instruction-following examples



    - Generate training pairs from detector mappings
    - Format examples to match pillars-detectors/schema.json output structure
    - Support for multi-label taxonomy assignments
    - _Requirements: 5.2, 5.4_

  - [x] 3.2 Implement SyntheticDataGenerator for balanced training sets


    - Generate synthetic PII examples using regex patterns
    - Create synthetic jailbreak and prompt injection examples
    - Balance training data across taxonomy categories
    - _Requirements: 5.5_


  - [x] 3.3 Create DatasetValidator for training data quality

    - Validate instruction-response format consistency
    - Ensure all target labels exist in taxonomy
    - Check coverage across detector types and taxonomy categories
    - _Requirements: 5.3, 11.4_

- [x] 4. Implement LoRA fine-tuning pipeline





  - [x] 4.1 Create ModelLoader for Llama-3-8B-Instruct


    - Load base model with proper tokenizer configuration
    - Set up model for instruction-following fine-tuning
    - Support for quantization options (FP16, 8-bit)
    - _Requirements: 2.3, 3.3_

  - [x] 4.2 Implement LoRATrainer with specified hyperparameters


    - Configure LoRA with r=16, α=32, learning rate=2e-4
    - Set up training loop for 1-2 epochs with sequence length 1-2k
    - Implement training metrics collection and validation
    - _Requirements: 2.1, 2.2_

  - [x] 4.3 Create CheckpointManager for model versioning


    - Save and load LoRA checkpoints with version tags
    - Support for model rollback and A/B testing
    - Integration with deployment pipeline
    - _Requirements: 7.1, 11.5_

- [x] 5. Build FastAPI service layer





  - [x] 5.1 Create MapperAPI with /map endpoint


    - Implement REST API following pillars-detectors/schema.json
    - Add request validation and error handling
    - Support for batch processing requests
    - _Requirements: 10.3, 1.1_

  - [x] 5.2 Implement ModelServer interface for vLLM/TGI


    - Abstract serving backend (vLLM for GPU, TGI for CPU)
    - Configure generation parameters (temperature 0.0-0.2, top_p 0.9)
    - Handle model loading and inference requests
    - _Requirements: 3.1, 3.2, 2.4_

  - [x] 5.3 Create JSONValidator for schema compliance


    - Validate model outputs against pillars-detectors/schema.json
    - Implement retry logic with adjusted generation parameters
    - Track validation success rates for monitoring
    - _Requirements: 6.1, 6.2, 11.1_

- [x] 6. Implement confidence evaluation and fallback system











  - [x] 6.1 Create ConfidenceEvaluator with calibrated thresholds







    - Calculate confidence using model logit softmax probability
    - Implement configurable confidence thresholds (default 0.6)
    - Support for confidence calibration and threshold tuning
    - _Requirements: 6.4, 6.5, 11.5_


  - [x] 6.2 Implement FallbackMapper for rule-based mapping


    - Use detector YAML configurations for direct mapping
    - Handle cases when model confidence is below threshold
    - Log fallback usage for model improvement tracking
    - _Requirements: 1.2, 6.3, 6.6_

- [x] 7. Build storage and persistence layer







  - [x] 7.1 Create StorageManager for S3 and database integration




    - Implement S3 immutable storage with WORM configuration
    - Set up ClickHouse/PostgreSQL for hot data with 90-day retention
    - Configure AES256-KMS encryption with BYOK support
    - _Requirements: 12.1, 12.2, 8.4_

  - [x] 7.2 Implement TenantIsolationManager for multi-tenancy




    - Ensure tenant-scoped data access and queries
    - Prevent cross-tenant data leakage
    - Support for tenant-specific configuration overrides
    - _Requirements: 8.3_

  - [x] 7.3 Create privacy-first logging system






    - Store only metadata (tenant ID, detector type, taxonomy hit)
    - Never persist raw detector inputs
    - Implement audit trail for compliance reporting
    - _Requirements: 8.1, 8.2_

- [x] 8. Implement monitoring and observability





  - [x] 8.1 Create MetricsCollector for Prometheus integration


    - Track request count, schema-valid %, fallback %, latency percentiles
    - Monitor model performance and confidence score distribution
    - Implement alerting for quality threshold violations
    - _Requirements: 11.1, 11.2_

  - [x] 8.2 Build quality gates for CI/CD pipeline



    - Implement automated testing with golden test cases (≥100 per detector)
    - Set up GitHub Actions to block merges if thresholds not met
    - Test schema validation, taxonomy F1, and latency requirements
    - _Requirements: 11.3, 11.4, 11.5_

- [x] 9. Create reporting and audit capabilities



  - [x] 9.1 Implement ReportGenerator for multiple formats


    - Generate PDF reports via WeasyPrint with embedded version tags
    - Create CSV exports via Pandas with version metadata
    - Support JSON API responses with version headers
    - _Requirements: 9.2, 9.5_

  - [x] 9.2 Build audit trail and compliance mapping





    - Map taxonomy labels to compliance controls using frameworks.yaml
    - Generate coverage reports with incidents, MTTR, and control mapping
    - Include lineage from detector output to canonical label
    - _Requirements: 9.3, 7.4_

- [x] 10. Implement versioning and migration system
  - [x] 10.1 Create TaxonomyMigrator for version transitions
    - Handle migration between taxonomy versions with automated label remapping (src/llama_mapper/versioning/taxonomy_migrator.py)
    - Support for backward compatibility and rollback scenarios (plan.invert())
    - Validate migration completeness and data integrity (validate_plan_completeness + MigrationReport)
    - _Requirements: 7.3_

  - [x] 10.2 Build version management for all components
    - Track taxonomy version (2025.09), model version (mapper-lora@vX.Y.Z), frameworks version (src/llama_mapper/versioning/version_manager.py; CLI: `mapper versions show`)
    - Embed version information in all outputs and reports (API enriches MappingResponse.provenance.model and notes with version tags; ReportGenerator already embeds)
    - Support for coordinated version updates across components (VersionSnapshot aggregates taxonomy/frameworks/detectors/model; CLI supports migration plan/apply)
    - _Requirements: 7.5, 9.5_

- [x] 11. Create deployment and containerization
- [x] 11.1 Build Docker image with multi-stage optimization
   - Create production-ready image with Python 3.11, FastAPI, Hugging Face stack (Dockerfile at project root)
   - Implement health checks and graceful shutdown handling (HEALTHCHECK → GET /health; FastAPI shutdown hook closes model_server sessions incl. TGI) (src/llama_mapper/api/mapper.py)
   - Optimize image size and security scanning (multi-stage build, non-root user, minimal runtime base, .dockerignore trims context)
   - Notes:
     - Dockerfile: multi-stage builder installs deps to /install, runtime copies to /usr/local; default CMD: `python -m src.llama_mapper.api.main --host 0.0.0.0 --port 8000`
     - .dockerignore excludes tests, .kiro/, local data, caches; README kept in build context
     - WARP.md updated with build/run instructions and notes on probes/HPA
   - _Requirements: 10.1, 10.2_

- [x] 11.2 Create Helm chart for Kubernetes deployment
   - Chart path: charts/llama-mapper (Chart.yaml, values.yaml, templates/*, .helmignore, NOTES.txt)
   - Profile switch: `.Values.profile` controls backend env and resources (tgi=CPU default, vllm=GPU); `LLAMA_MAPPER_SERVING__BACKEND` auto-follows profile
   - Resources: separate CPU/GPU requests/limits; GPU profile requests `nvidia.com/gpu: 1` and supports nodeSelector/tolerations
   - Probes & HPA: Liveness/Readiness → GET /health; autoscaling/v2 HPA on CPU utilization with configurable bounds
   - Configuration:
     - External config: `externalConfigMaps.enabled=true` and provide names; mounts to /app/pillars-detectors
     - Inline demo config: `inlineConfigs.enabled=true` (taxonomy/frameworks/schema + optional detectors)
     - Env vars: `.Values.env` controls LLAMA_MAPPER_* paths/serving options
   - Security: ServiceAccount created conditionally; container runs as non-root and drops all capabilities
   - Service/Ingress: ClusterIP by default; optional Ingress template with TLS
   - _Requirements: 10.2, 10.4_

- [x] 12. Define API contract and public interface
  - [x] 12.1 Create OpenAPI specification for /map endpoint
    - Define request/response schemas for single and batch operations
    - Specify error codes, rate limits, and idempotency key handling
    - Document negative cases (schema validation errors, fallback triggered, tenant misconfig)
    - _Requirements: 10.3, 1.1_

  - [x] 12.2 Implement API authentication and RBAC
    - Add API key authentication with per-tenant scopes
    - Create token rotation script and leave OIDC stub for future
    - Implement tenant-scoped access controls and request validation
    - _Requirements: 8.3_

- [x] 13. Establish SLOs and performance testing
  - [x] 13.1 Define service level objectives document
    - Set target SLOs (p95 ≤250ms CPU, ≥99.5% schema-valid, <10% fallback)
    - Document load model (events/sec per tenant) and capacity planning
    - Define alerting thresholds and escalation procedures
    - _Requirements: 11.2, 11.3_

  - [x] 13.2 Create performance testing suite
    - Implement Locust/k6 performance tests for CI smoke testing
    - Build separate load testing job for staging environment
    - Add conformance tests that replay golden cases via HTTP
    - _Requirements: 11.3, 11.4_

- [x] 14. Implement backup, disaster recovery, and migrations
  - [x] 14.1 Set up data backup and recovery systems
    - Create ClickHouse/PostgreSQL backup jobs with restore drill procedures
      - Helm CronJobs: charts/llama-mapper/templates/backup-postgresql-cronjob.yaml; charts/llama-mapper/templates/backup-clickhouse-cronjob.yaml
      - Restore Jobs: charts/llama-mapper/templates/restore-postgresql-job.yaml; charts/llama-mapper/templates/restore-clickhouse-job.yaml
      - Helm values: backups.* and restore.* in charts/llama-mapper/values.yaml
    - Configure S3 lifecycle rules (WORM + retention) with documented restore steps
      - Terraform module: infra/terraform/s3_worm_bucket (Object Lock, versioning, optional Glacier transitions)
      - Runbook: docs/runbook/backup-restore.md (procedures, verification, permissions)
    - Test backup integrity and recovery time objectives
      - RTO guidance and integrity checks documented in docs/runbook/backup-restore.md
    - _Requirements: 12.1, 12.2_

  - [x] 14.2 Create migration and rollback procedures
    - Document taxonomy/framework migration playbook with rollback steps
      - Runbook: docs/runbook/migrations.md (plan/apply/invert steps, VersionSnapshot usage)
    - Implement LoRA adapter rollback procedures and kill-switch for rule-only mapping
      - Kill-switch: runtime mode in ConfigManager (serving.mode = hybrid|rules_only) and API enforcement in MapperAPI
      - CLI controls: mapper runtime show-mode | set-mode | kill-switch on/off (src/llama_mapper/cli.py)
    - Create tenant configuration migration tools and validation
      - CLI: mapper tenant migrate-config | validate-config for tenant YAMLs
    - _Requirements: 7.3, 10.1_

- [x] 15. Build security and privacy safeguards
  - [x] 15.1 Create threat model and privacy checklist
    - Develop STRIDE threat analysis table for the service
    - Implement PII redaction in request payloads and logging do/don'ts
    - Add BYOK verification tests and pre-commit lints that ban raw content logs
    - Implementation notes:
      - PII redaction utilities at src/llama_mapper/security/redaction.py providing redact_text, redact_dict, and redact_request_model. Redacts emails, phones, IPs, SSNs, credit cards, JWTs, and API keys.
      - Pre-commit guardrail to block raw-content logging: scripts/check_no_raw_logs.py with configuration in .pre-commit-config.yaml. Install with "pre-commit install"; run manually with "pre-commit run --all-files". Use "# ok-to-log" to explicitly allow a sanitized line.
      - Documentation added at docs/security/privacy-checklist.md (do/don'ts) and docs/security/threat-model.md (STRIDE analysis).
      - Tests: tests/security/test_byok_and_redaction.py validates PII redaction behavior and SSE-KMS usage (BYOK) on S3 writes.
      - Notes: Complements existing privacy-first logging in src/llama_mapper/logging.py and src/llama_mapper/utils/logging.py. Continue to avoid logging request.output and any raw content; sanitize metadata with redact_dict before logging when needed.
    - _Requirements: 8.1, 8.2, 8.4_

  - [x] 15.2 Integrate SecretsManager with Vault/AWS Secrets Manager
    - Handle API keys, model weights, and encryption keys securely
    - Implement automatic secret rotation and least-privilege access
    - Add audit logging for secret access and usage
    - Implementation notes:
      - SecretsManager facade at src/llama_mapper/security/secrets_manager.py with backends: AWS (boto3), Vault KV v2 (hvac), and Env (dev fallback).
      - Configure backend via Settings(security__secrets_backend="aws"|"vault"|"env"). For Vault, use VAULT_ADDR/VAULT_TOKEN environment variables. For AWS, credentials flow via Settings.storage AWS fields or environment/instance profile.
      - Emits metadata-only "secret_access" audit logs (action, secret_name, version, tenant_id, success) via structlog; never logs secret values.
      - Rotation stubs provided per-backend (rotate/put), ready to integrate with rotation workflows.
      - Example usage: examples/secrets_demo.py.
      - Tests: tests/unit/test_secrets_manager.py (AWS/Vault/env via mocks); BYOK verification covered in tests/security/test_byok_and_redaction.py.
    - _Requirements: 8.5_

- [ ] 16. Create onboarding and configuration management
  - [ ] 16.1 Build configuration validation CLI
    - Create `mapper validate-config` CLI to lint taxonomy, frameworks, detector maps
    - Implement "add a detector" flow with required fields and file placement guidance
    - Add configuration validation and compatibility checking
    - _Requirements: 5.1, 5.3, 7.2_

  - [ ] 16.2 Implement tenant configuration override system
    - Define configuration precedence (global → tenant → environment)
    - Create tenant override precedence tests and validation
    - Document tenant-specific configuration management procedures
    - _Requirements: 8.3_

- [ ] 17. Establish operational runbooks and alerting
  - [ ] 17.1 Create operational runbooks and procedures
    - Document rollback procedures for LoRA adapters and configuration changes
    - Create kill-switch procedures to force rule-only mapping during incidents
    - Build troubleshooting guides for common failure scenarios
    - _Requirements: 11.5_

  - [ ] 17.2 Implement comprehensive alerting system
    - Set up alerts for schema-valid %, fallback %, latency, 5xx errors, queue lag
    - Configure pager escalation for critical service degradation
    - Create alert runbooks with investigation and resolution steps
    - _Requirements: 11.1, 11.2_

- [ ] 18. Define cost management and resource planning
  - [ ] 18.1 Establish cost guardrails and resource sizing
    - Document initial instance sizes and autoscaling rules with monthly cost bounds
    - Create guidance for switching between CPU↔GPU and quantization toggles
    - Implement cost monitoring and budget alerts
    - _Requirements: 3.1, 3.2, 10.4_

  - [ ] 18.2 Set up end-to-end testing and validation
    - Create integration tests for complete pipeline (detector input → canonical output)
    - Test fallback mechanisms under various failure conditions
    - Validate performance with different quantization settings and serving backends
    - _Requirements: 11.2, 11.3_