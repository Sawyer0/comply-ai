# Design Document

## Overview

This design document outlines the architecture for refactoring the llama-mapper codebase into exactly 3 microservices with clean separation of concerns, consolidated functionality, and optimized maintainability. The design follows Single Responsibility Principle and distributed system best practices while preserving all existing production capabilities.

## Architecture

### High-Level Service Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Requests                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│              Detector Orchestration Service                     │
│  • Detector coordination & health monitoring                    │
│  • Policy enforcement & conflict resolution                     │
│  • Service discovery & circuit breakers                         │
│  • Rate limiting & authentication                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │ OrchestrationResponse
┌─────────────────────▼───────────────────────────────────────────┐
│                   Analysis Service                              │
│  • Pattern recognition & statistical analysis                   │
│  • Risk scoring & compliance intelligence                       │
│  • Quality evaluation & RAG capabilities                        │
│  • Privacy controls & ML analysis models                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │ AnalysisResponse
┌─────────────────────▼───────────────────────────────────────────┐
│                    Mapper Service                               │
│  • Core mapping & response generation                           │
│  • Model serving (Llama-3-8B) & training                       │
│  • Validation & fallback mechanisms                             │
│  • Cost monitoring & performance optimization                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │ MappingResponse
┌─────────────────────▼───────────────────────────────────────────┐
│                        Client                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Taxonomy and Schema Management

#### Canonical Taxonomy System
```python
class CanonicalTaxonomy:
    """Centralized taxonomy management across all services"""
    
    # Core taxonomy categories
    SECURITY_CATEGORIES = [
        "pii.personal_identifiers",
        "pii.financial_data", 
        "pii.health_records",
        "security.credentials",
        "security.tokens",
        "content.toxic_language",
        "content.hate_speech",
        "content.violence"
    ]
    
    COMPLIANCE_FRAMEWORKS = {
        "soc2": ["CC6.1", "CC6.7", "CC7.1", "CC7.2"],
        "iso27001": ["A.8.2.1", "A.8.2.2", "A.13.2.1"],
        "hipaa": ["164.502", "164.504", "164.506"],
        "gdpr": ["Article.6", "Article.9", "Article.17"]
    }
    
    RISK_LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
```

#### Schema Evolution Management
```python
class SchemaEvolutionManager:
    """Manages schema evolution across service boundaries"""
    
    def validate_schema_compatibility(self, old_schema: dict, new_schema: dict) -> bool:
        """Validate backward compatibility of schema changes"""
        pass
    
    def generate_migration_plan(self, from_version: str, to_version: str) -> MigrationPlan:
        """Generate migration plan for schema evolution"""
        pass
    
    def execute_canary_schema_deployment(self, schema_version: str, traffic_percentage: float):
        """Deploy schema changes using canary deployment"""
        pass
```

#### Framework Mapping Versioning
```python
class FrameworkMappingRegistry:
    """Versioned framework mappings with canary deployment support"""
    
    def deploy_mapping_version(self, framework: str, version: str, canary_percentage: float):
        """Deploy new framework mapping with canary traffic"""
        pass
    
    def validate_mapping_accuracy(self, framework: str, test_cases: List[TestCase]) -> float:
        """Validate mapping accuracy before full deployment"""
        pass
```

### Canary Deployment Architecture

#### Multi-Service Canary Coordination
```
┌─────────────────────────────────────────────────────────────────┐
│                    Canary Deployment Flow                       │
├─────────────────────────────────────────────────────────────────┤
│  1. Schema/Taxonomy Changes                                     │
│     ├── Validate compatibility                                 │
│     ├── Deploy to canary environment                           │
│     └── Run integration tests                                  │
├─────────────────────────────────────────────────────────────────┤
│  2. Service-Level Canary                                       │
│     ├── Deploy new service version (5% traffic)                │
│     ├── Monitor key metrics (latency, errors, quality)         │
│     └── Gradual traffic increase (5% → 25% → 50% → 100%)      │
├─────────────────────────────────────────────────────────────────┤
│  3. Model/Algorithm Canary                                     │
│     ├── Deploy new model version                               │
│     ├── A/B test against current model                         │
│     ├── Validate quality metrics                               │
│     └── Gradual rollout based on performance                   │
├─────────────────────────────────────────────────────────────────┤
│  4. Cross-Service Validation                                   │
│     ├── End-to-end integration testing                         │
│     ├── Contract validation between services                   │
│     └── Rollback coordination if issues detected               │
└─────────────────────────────────────────────────────────────────┘
```

### Comprehensive Service Capabilities Matrix

| Component | Detector Orchestration | Analysis Service | Mapper Service |
|-----------|----------------------|------------------|----------------|
| **Primary Domain** | Detector coordination & policy | Advanced analysis & intelligence | Core mapping & generation |
| **ML Models** | None (rule-based only) | Phi-3 (analysis), embeddings | Llama-3-8B (mapping), LoRA |
| **Security Features** | WAF, RBAC, API keys, rate limiting | Privacy controls, content scrubbing | Field encryption, secure serving |
| **Database** | PostgreSQL (orchestration) | PostgreSQL + ClickHouse (analytics) | PostgreSQL (mapping) + S3 (models) |
| **Caching** | Redis (idempotency, responses) | Redis (analysis cache) | Redis (model cache) |
| **Monitoring** | Prometheus, health checks, alerts | Quality monitoring, drift detection | Cost monitoring, performance tracking |
| **Privacy** | Request sanitization | Metadata-only logging, scrubbing | No raw content persistence |
| **Resilience** | Circuit breakers, retries | Fallback analysis engines | Template fallbacks, model failover |
| **CLI Commands** | Detector/policy/health mgmt | Analysis/quality/RAG operations | Mapping/training/model operations |
| **Configuration** | Dynamic policy config | Analysis & ML config | Model & serving config |
| **Observability** | Distributed tracing, correlation IDs | Business metrics, anomaly detection | Cost analytics, usage tracking |
| **Scalability** | Horizontal scaling, load balancing | Batch processing, stream processing | Model serving optimization |
| **Compliance** | Policy enforcement, audit trails | Framework mapping, compliance intelligence | Evidence generation, audit support |
| **Taxonomy** | Detector taxonomy, routing taxonomy | Analysis taxonomy, risk taxonomy | Canonical taxonomy, framework mappings |
| **Schemas** | Orchestration schemas, policy schemas | Analysis schemas, quality schemas | Mapping schemas, response schemas |
| **Deployment** | Blue-green, canary deployments | Model canary, A/B testing | Model versioning, canary serving |
| **Multi-Tenancy** | Tenant routing, isolation | Tenant-specific analysis | Tenant data isolation |
| **Plugins** | Policy plugins, detector plugins | Analysis engine plugins | Model serving plugins |
| **Pipelines** | Orchestration pipelines | Training pipelines, quality pipelines | Deployment pipelines |
| **Enterprise** | Azure integration, audit trails | Advanced analytics, monitoring | Enterprise security, compliance |

### Advanced Feature Distribution

#### Security & Privacy Features
- **WAF Integration**: SQL injection, XSS, command injection, path traversal protection
- **Authentication**: Multi-factor auth, JWT security, API key rotation
- **Authorization**: RBAC with fine-grained permissions
- **Encryption**: Field-level encryption, HashiCorp Vault integration
- **Privacy**: Metadata-only logging, content scrubbing, data minimization
- **Audit**: Complete audit trails, compliance evidence generation

#### Monitoring & Observability
- **Metrics**: Prometheus metrics, business KPIs, security events
- **Tracing**: OpenTelemetry, Jaeger integration, correlation IDs
- **Logging**: Structured JSON logging, privacy-compliant scrubbing
- **Alerting**: Intelligent alerting, escalation policies, incident response
- **Quality**: Model drift detection, performance anomaly detection
- **Cost**: Real-time cost tracking, budget alerts, optimization recommendations

#### Resilience & Performance
- **Circuit Breakers**: Service protection with configurable thresholds
- **Retry Logic**: Exponential backoff with jitter
- **Bulkhead Isolation**: Resource isolation patterns
- **Timeout Management**: Configurable timeout policies
- **Load Balancing**: Intelligent load distribution
- **Auto-scaling**: Metrics-based horizontal scaling

#### ML & AI Capabilities
- **Model Serving**: vLLM, TGI, CPU fallback backends
- **Training**: LoRA fine-tuning, checkpoint management
- **Quality**: Model evaluation, drift detection, A/B testing
- **Optimization**: Performance optimization, resource management
- **Fallbacks**: Rule-based alternatives for all ML components
- **RAG**: Regulatory knowledge enhancement, compliance guidance

#### Taxonomy & Schema Management
- **Canonical Taxonomy**: Centralized taxonomy with versioning and evolution
- **Framework Mappings**: Versioned compliance framework mappings (SOC2, ISO27001, HIPAA, GDPR)
- **Schema Evolution**: Backward-compatible schema evolution with validation
- **Migration Tools**: Automated taxonomy and schema migration utilities

#### Deployment & Release Management
- **Canary Deployments**: Multi-service canary deployment coordination
- **Blue-Green**: Zero-downtime deployments with environment switching
- **Feature Flags**: Gradual feature rollout with runtime flag evaluation
- **A/B Testing**: Model and algorithm A/B testing framework
- **Version Management**: Semantic versioning with compatibility matrices

#### Multi-Tenancy & Isolation
- **Tenant Isolation**: Row-level security, tenant-specific schemas, data isolation
- **Multi-Tenant Configuration**: Tenant-specific settings, resource quotas, access controls
- **Tenant Analytics**: Per-tenant usage tracking, performance monitoring, cost allocation

#### Plugin & Extension System
- **CLI Plugin System**: Dynamic plugin loading, command registration, plugin management
- **Extension Points**: Configurable extension points for custom functionality
- **Hook System**: Event-driven hooks for custom processing logic

#### Advanced Pipeline Management
- **Training Pipelines**: LoRA training, Phi-3 training, checkpoint management
- **Deployment Pipelines**: Multi-stage deployment with validation gates
- **Data Pipelines**: Batch processing, streaming data processing
- **Quality Pipelines**: Automated quality evaluation, drift detection

#### Enterprise Integration
- **Azure Integration**: Azure SQL, Azure Storage, Azure Key Vault, Azure Monitor
- **PostgreSQL Extensions**: Advanced query optimization, performance monitoring
- **Enterprise Security**: Field-level encryption, secrets rotation, audit trails

## Components and Interfaces

### 1. Detector Orchestration Service

#### Core Components
```
detector-orchestration/
├── src/orchestration/
│   ├── api/                     # HTTP API endpoints
│   │   ├── orchestration.py     # Main orchestration endpoints
│   │   ├── policy.py           # Policy management endpoints
│   │   ├── registry.py         # Detector registry endpoints
│   │   └── health.py           # Health and monitoring endpoints
│   │
│   ├── core/                    # Core orchestration logic
│   │   ├── coordinator.py       # Detector coordination
│   │   ├── router.py           # Content routing
│   │   ├── aggregator.py       # Response aggregation
│   │   └── conflict_resolver.py # Conflict resolution
│   │
│   ├── policy/                  # Policy management
│   │   ├── manager.py          # Policy lifecycle management
│   │   ├── engine.py           # OPA policy engine
│   │   ├── store.py            # Policy storage
│   │   ├── validator.py        # Policy validation
│   │   ├── versioning.py       # Policy versioning
│   │   └── migration.py        # Policy migration tools
│   │
│   ├── schemas/                 # Schema management
│   │   ├── orchestration_schemas.py # Orchestration schemas
│   │   ├── policy_schemas.py    # Policy definition schemas
│   │   ├── detector_schemas.py  # Detector capability schemas
│   │   └── routing_schemas.py   # Routing decision schemas
│   │
│   ├── discovery/               # Service discovery
│   │   ├── detector_manager.py  # Detector discovery
│   │   ├── health_monitor.py    # Health monitoring
│   │   ├── registry.py         # Service registry
│   │   └── config_reloader.py  # Configuration hot-reload
│   │
│   ├── resilience/              # Resilience patterns
│   │   ├── circuit_breaker.py   # Circuit breaker implementation
│   │   ├── rate_limiter.py     # Rate limiting
│   │   └── retry_handler.py    # Retry logic
│   │
│   ├── cache/                   # Caching layer
│   │   ├── idempotency.py      # Idempotency cache
│   │   ├── response_cache.py   # Response caching
│   │   └── redis_backend.py    # Redis implementation
│   │
│   ├── security/                # Security components
│   │   ├── auth/               # Authentication system
│   │   │   ├── api_key_manager.py # API key management
│   │   │   ├── jwt_security.py    # JWT token security
│   │   │   └── secrets_rotation.py # Automated secrets rotation
│   │   ├── rbac/               # Role-based access control
│   │   │   ├── rbac_manager.py    # RBAC implementation
│   │   │   ├── permissions.py     # Permission definitions
│   │   │   └── roles.py          # Role definitions
│   │   ├── waf/                # Web Application Firewall
│   │   │   ├── attack_detector.py # Attack pattern detection
│   │   │   ├── sql_injection.py   # SQL injection protection
│   │   │   ├── xss_protection.py  # XSS prevention
│   │   │   ├── command_injection.py # Command injection prevention
│   │   │   └── path_traversal.py  # Path traversal protection
│   │   ├── encryption/         # Encryption services
│   │   │   ├── field_encryption.py # Field-level encryption
│   │   │   └── vault_integration.py # HashiCorp Vault integration
│   │   └── validation/         # Security validation
│   │       ├── input_sanitizer.py # Multi-layer input sanitization
│   │       └── security_validator.py # Security validation
│   │
│   ├── monitoring/              # Monitoring & observability
│   │   ├── metrics/            # Metrics collection
│   │   │   ├── prometheus_metrics.py # Prometheus metrics
│   │   │   ├── business_metrics.py   # Business-specific metrics
│   │   │   └── security_metrics.py   # Security event metrics
│   │   ├── tracing/            # Distributed tracing
│   │   │   ├── correlation_id.py     # Correlation ID management
│   │   │   ├── opentelemetry.py      # OpenTelemetry integration
│   │   │   └── jaeger_exporter.py    # Jaeger tracing
│   │   ├── logging/            # Structured logging
│   │   │   ├── structured_logger.py  # JSON structured logging
│   │   │   └── log_scrubber.py      # Privacy-compliant logging
│   │   ├── health/             # Health monitoring
│   │   │   ├── health_checker.py    # Comprehensive health checks
│   │   │   └── dependency_monitor.py # Dependency health monitoring
│   │   └── alerting/           # Alert management
│   │       ├── alert_manager.py     # Alert coordination
│   │       ├── escalation.py        # Alert escalation policies
│   │       └── incident_response.py # Incident response automation
│   │
│   ├── cli/                     # CLI commands
│   │   ├── detector_commands.py # Detector management
│   │   ├── policy_commands.py  # Policy management
│   │   └── health_commands.py  # Health operations
│   │
│   ├── tenancy/                 # Multi-tenancy support
│   │   ├── tenant_manager.py   # Tenant management
│   │   ├── tenant_isolation.py # Tenant data isolation
│   │   ├── tenant_routing.py   # Tenant-aware routing
│   │   └── tenant_config.py    # Tenant-specific configuration
│   │
│   ├── plugins/                 # Plugin system
│   │   ├── plugin_manager.py   # Plugin lifecycle management
│   │   ├── detector_plugins.py # Detector plugin interface
│   │   ├── policy_plugins.py   # Policy plugin interface
│   │   └── extension_points.py # Extension point definitions
│   │
│   ├── pipelines/               # Pipeline management
│   │   ├── orchestration_pipeline.py # Orchestration pipeline
│   │   ├── validation_pipeline.py    # Validation pipeline
│   │   └── audit_pipeline.py         # Audit trail pipeline
│   │
│   └── config/                  # Configuration
│       ├── settings.py         # Service settings
│       ├── detector_config.py  # Detector configuration
│       ├── policy_config.py    # Policy configuration
│       └── tenant_config.py    # Multi-tenant configuration
```

#### Key Interfaces
```python
# Service API Contract
class OrchestrationRequest(BaseModel):
    content: str
    tenant_id: str
    policy_bundle: str
    processing_mode: ProcessingMode
    metadata: Optional[Dict[str, Any]] = None

class OrchestrationResponse(BaseModel):
    request_id: str
    detector_results: List[DetectorResult]
    aggregation_summary: AggregationSummary
    coverage_achieved: float
    processing_time_ms: float
    provenance: List[ProvenanceEntry]
```

### 2. Analysis Service

#### Core Components
```
analysis-service/
├── src/analysis/
│   ├── api/                     # HTTP API endpoints
│   │   ├── analysis.py         # Main analysis endpoints
│   │   ├── quality.py          # Quality evaluation endpoints
│   │   ├── rag.py              # RAG endpoints
│   │   └── evaluation.py      # Evaluation endpoints
│   │
│   ├── engines/                 # Analysis engines
│   │   ├── core/               # Primary engines
│   │   │   ├── pattern_recognition/
│   │   │   │   ├── temporal_analyzer.py
│   │   │   │   ├── frequency_analyzer.py
│   │   │   │   ├── correlation_analyzer.py
│   │   │   │   └── anomaly_detector.py
│   │   │   ├── risk_scoring/   # Consolidated risk scoring
│   │   │   │   ├── risk_engine.py
│   │   │   │   ├── technical_scorer.py
│   │   │   │   ├── business_scorer.py
│   │   │   │   ├── regulatory_scorer.py
│   │   │   │   └── temporal_scorer.py
│   │   │   ├── compliance_intelligence/
│   │   │   │   ├── framework_mapper.py
│   │   │   │   ├── soc2_mapper.py
│   │   │   │   ├── iso27001_mapper.py
│   │   │   │   ├── hipaa_mapper.py
│   │   │   │   ├── gdpr_mapper.py
│   │   │   │   └── custom_framework_mapper.py
│   │   │   └── template_orchestrator/
│   │   │       ├── orchestrator.py
│   │   │       ├── template_manager.py
│   │   │       └── response_formatter.py
│   │
│   ├── taxonomy/                # Taxonomy management
│   │   ├── analysis_taxonomy.py # Analysis-specific taxonomy
│   │   ├── risk_taxonomy.py     # Risk classification taxonomy
│   │   ├── compliance_taxonomy.py # Compliance framework taxonomy
│   │   ├── pattern_taxonomy.py  # Pattern classification taxonomy
│   │   └── taxonomy_validator.py # Taxonomy validation
│   │
│   ├── schemas/                 # Schema management
│   │   ├── analysis_schemas.py  # Analysis request/response schemas
│   │   ├── pattern_schemas.py   # Pattern analysis schemas
│   │   ├── risk_schemas.py      # Risk scoring schemas
│   │   ├── compliance_schemas.py # Compliance mapping schemas
│   │   └── quality_schemas.py   # Quality metrics schemas
│   │   │
│   │   ├── statistical/         # Statistical components
│   │   │   ├── pattern_classifier.py
│   │   │   ├── business_relevance_assessor.py
│   │   │   ├── pattern_evolution_tracker.py
│   │   │   └── multi_pattern_analyzer.py
│   │   │
│   │   ├── optimization/        # Optimization engines
│   │   │   ├── threshold_optimizer.py
│   │   │   ├── statistical_optimizer.py
│   │   │   ├── impact_simulator.py
│   │   │   └── risk_factor_analyzer.py
│   │   │
│   │   └── interfaces.py        # Engine interfaces
│   │
│   ├── ml/                      # ML components
│   │   ├── model_server/        # Analysis model serving
│   │   │   ├── phi3_backend.py  # Phi-3 model backend
│   │   │   ├── vllm_backend.py  # vLLM implementation
│   │   │   ├── tgi_backend.py   # TGI implementation
│   │   │   └── cpu_fallback.py  # CPU fallback
│   │   ├── embeddings/          # Embedding models
│   │   │   ├── analysis_embeddings.py
│   │   │   └── compliance_embeddings.py
│   │   └── fallback/            # Rule-based fallbacks
│   │       ├── pattern_fallback.py
│   │       ├── risk_fallback.py
│   │       └── analysis_fallback.py
│   │
│   ├── rag/                     # RAG system
│   │   ├── knowledge_base/      # Regulatory documents
│   │   │   ├── document_processor.py
│   │   │   ├── knowledge_store.py
│   │   │   └── regulatory_db.py
│   │   ├── retrieval/           # Document retrieval
│   │   │   ├── retriever.py
│   │   │   ├── ranker.py
│   │   │   └── context_builder.py
│   │   ├── guardrails/          # Compliance guardrails
│   │   │   ├── compliance_checker.py
│   │   │   └── regulatory_validator.py
│   │   └── evaluation/          # RAG quality metrics
│   │       ├── rag_evaluator.py
│   │       └── quality_metrics.py
│   │
│   ├── quality/                 # Quality system
│   │   ├── alerting/           # Quality alerting
│   │   │   ├── alert_manager.py
│   │   │   ├── alert_handlers.py
│   │   │   └── quality_monitor.py
│   │   ├── evaluation/         # Quality evaluation
│   │   │   ├── evaluator.py
│   │   │   ├── weekly_service.py
│   │   │   └── degradation_detector.py
│   │   └── config/             # Quality configuration
│   │       ├── quality_config.py
│   │       └── evaluation_config.py
│   │
│   ├── privacy/                 # Privacy-first architecture
│   │   ├── content_scrubber.py  # Content scrubbing & sanitization
│   │   ├── metadata_logger.py   # Metadata-only logging
│   │   ├── privacy_validator.py # Privacy compliance validation
│   │   ├── data_minimization.py # Data minimization controls
│   │   └── retention_manager.py # Data retention policies
│   │
│   ├── security/                # Security components
│   │   ├── waf/                # Web Application Firewall
│   │   │   ├── attack_patterns.py   # Attack pattern detection
│   │   │   ├── injection_protection.py # Injection attack prevention
│   │   │   ├── xss_prevention.py    # XSS protection
│   │   │   └── security_headers.py  # Security header management
│   │   ├── auth/               # Authentication & authorization
│   │   │   ├── multi_factor.py     # Multi-factor authentication
│   │   │   ├── session_manager.py  # Session management
│   │   │   └── token_validator.py  # Token validation
│   │   └── encryption/         # Encryption services
│   │       ├── field_encryption.py # Field-level encryption
│   │       └── key_management.py   # Key management
│   │
│   ├── cli/                     # CLI commands
│   │   ├── analysis_commands.py # Analysis operations
│   │   ├── quality_commands.py  # Quality operations
│   │   └── rag_commands.py     # RAG operations
│   │
│   ├── infrastructure/          # Infrastructure components
│   │   ├── database/           # Database management
│   │   │   ├── connection_manager.py # Advanced connection pooling
│   │   │   ├── migration_manager.py  # Schema migrations
│   │   │   ├── backup_manager.py     # Database backup/restore
│   │   │   ├── multi_db_manager.py   # Multi-database coordination
│   │   │   └── schema_migration.py   # Database schema evolution
│   │
│   ├── deployment/              # Deployment management
│   │   ├── canary/             # Canary deployment system
│   │   │   ├── canary_controller.py  # Canary deployment controller
│   │   │   ├── traffic_splitter.py   # Traffic splitting logic
│   │   │   ├── health_validator.py   # Canary health validation
│   │   │   └── rollback_manager.py   # Automatic rollback
│   │   ├── blue_green/         # Blue-green deployments
│   │   │   ├── environment_manager.py # Environment switching
│   │   │   ├── validation_suite.py   # Pre-switch validation
│   │   │   └── dns_manager.py        # DNS switching
│   │   ├── feature_flags/      # Feature flag system
│   │   │   ├── flag_manager.py       # Feature flag management
│   │   │   ├── gradual_rollout.py    # Gradual feature rollout
│   │   │   └── flag_evaluation.py    # Runtime flag evaluation
│   │   └── versioning/         # Version management
│   │       ├── semantic_versioning.py # Semantic version management
│   │       ├── compatibility_matrix.py # Version compatibility
│   │       └── deprecation_manager.py # Deprecation lifecycle
│   │   ├── storage/            # Storage backends
│   │   │   ├── s3_manager.py        # S3/MinIO integration
│   │   │   ├── azure_storage.py     # Azure storage integration
│   │   │   └── storage_abstraction.py # Storage abstraction layer
│   │   ├── messaging/          # Event streaming
│   │   │   ├── kafka_producer.py    # Kafka event publishing
│   │   │   ├── kafka_consumer.py    # Kafka event consumption
│   │   │   └── event_sourcing.py    # Event sourcing implementation
│   │   └── service_mesh/       # Service mesh integration
│   │       ├── istio_config.py      # Istio configuration
│   │       ├── mtls_manager.py      # mTLS management
│   │       └── network_policies.py  # Network policy management
│   │
│   ├── resilience/              # Advanced resilience patterns
│   │   ├── circuit_breaker.py   # Circuit breaker implementation
│   │   ├── bulkhead.py         # Bulkhead isolation
│   │   ├── timeout_manager.py   # Timeout management
│   │   ├── retry_policies.py    # Advanced retry policies
│   │   └── chaos_engineering.py # Chaos engineering tools
│   │
│   ├── tenancy/                 # Multi-tenancy support
│   │   ├── tenant_analytics.py # Tenant-specific analytics
│   │   ├── tenant_isolation.py # Analysis data isolation
│   │   ├── tenant_quotas.py    # Resource quota management
│   │   └── tenant_customization.py # Tenant-specific customization
│   │
│   ├── plugins/                 # Plugin system
│   │   ├── analysis_plugins.py # Analysis engine plugins
│   │   ├── ml_plugins.py       # ML model plugins
│   │   ├── quality_plugins.py  # Quality evaluation plugins
│   │   └── rag_plugins.py      # RAG system plugins
│   │
│   ├── pipelines/               # Pipeline management
│   │   ├── training_pipeline.py    # ML training pipelines
│   │   ├── analysis_pipeline.py    # Analysis processing pipeline
│   │   ├── quality_pipeline.py     # Quality evaluation pipeline
│   │   └── batch_pipeline.py       # Batch processing pipeline
│   │
│   └── config/                  # Configuration management
│       ├── settings.py         # Service settings
│       ├── analysis_config.py  # Analysis configuration
│       ├── ml_config.py        # ML configuration
│       ├── security_config.py  # Security configuration
│       ├── monitoring_config.py # Monitoring configuration
│       ├── tenant_config.py    # Multi-tenant configuration
│       └── dynamic_config.py   # Hot-reloadable configuration
```

#### Key Interfaces
```python
# Service API Contract
class AnalysisRequest(BaseModel):
    orchestration_response: OrchestrationResponse
    analysis_type: AnalysisType
    frameworks: List[str]
    tenant_id: str

class AnalysisResponse(BaseModel):
    request_id: str
    pattern_analysis: PatternAnalysisResult
    risk_scores: RiskScoringResult
    compliance_mappings: List[ComplianceMappingResult]
    quality_metrics: QualityMetrics
    rag_insights: Optional[RAGInsights]
    processing_time_ms: float
```

### 3. Mapper Service

#### Core Components
```
mapper-service/
├── src/mapper/
│   ├── api/                     # HTTP API endpoints
│   │   ├── mapping.py          # Main mapping endpoints
│   │   ├── batch.py            # Batch processing endpoints
│   │   ├── training.py         # Training endpoints
│   │   └── validation.py       # Validation endpoints
│   │
│   ├── core/                    # Core mapping logic
│   │   ├── mapper.py           # Main mapping engine
│   │   ├── response_generator.py # Response generation
│   │   ├── taxonomy_mapper.py   # Taxonomy mapping
│   │   └── framework_adapter.py # Framework adaptation
│   │
│   ├── ml/                      # ML components
│   │   ├── model_server/        # Model serving
│   │   │   ├── llama_backend.py # Llama-3-8B backend
│   │   │   ├── vllm_backend.py  # vLLM implementation
│   │   │   ├── tgi_backend.py   # TGI implementation
│   │   │   └── cpu_fallback.py  # CPU fallback
│   │   ├── training/            # Training infrastructure
│   │   │   ├── lora_trainer.py  # LoRA fine-tuning
│   │   │   ├── model_loader.py  # Model management
│   │   │   ├── checkpoint_manager.py # Checkpoint management
│   │   │   └── training_pipeline.py # Training orchestration
│   │   ├── generation/          # Response generation
│   │   │   ├── content_generator.py
│   │   │   ├── template_engine.py
│   │   │   └── context_manager.py
│   │   └── optimization/        # Model optimization
│   │       ├── performance_optimizer.py
│   │       ├── resource_manager.py
│   │       └── cost_optimizer.py
│   │
│   ├── serving/                 # Model serving infrastructure
│   │   ├── model_registry.py    # Model versioning & registry
│   │   ├── deployment_manager.py # Model deployment orchestration
│   │   ├── canary_deployment.py # Canary deployment management
│   │   ├── ab_testing.py       # A/B testing framework
│   │   ├── blue_green.py       # Blue-green deployments
│   │   └── production_utils.py  # Production utilities
│   │
│   ├── taxonomy/                # Taxonomy management
│   │   ├── taxonomy_manager.py  # Taxonomy versioning & management
│   │   ├── canonical_taxonomy.py # Canonical taxonomy definitions
│   │   ├── framework_mappings.py # Framework-specific mappings
│   │   ├── taxonomy_validator.py # Taxonomy validation
│   │   └── migration_manager.py # Taxonomy migration tools
│   │
│   ├── schemas/                 # Schema management
│   │   ├── schema_registry.py   # Schema versioning & registry
│   │   ├── json_schemas/        # JSON schema definitions
│   │   │   ├── request_schemas.py
│   │   │   ├── response_schemas.py
│   │   │   └── internal_schemas.py
│   │   ├── schema_validator.py  # Runtime schema validation
│   │   ├── schema_evolution.py  # Schema evolution management
│   │   └── compatibility_checker.py # Schema compatibility validation
│   │
│   ├── validation/              # Validation components
│   │   ├── input_validator.py   # Input validation
│   │   ├── output_validator.py  # Output validation
│   │   ├── json_validator.py    # JSON schema validation
│   │   └── schema_manager.py    # Schema management
│   │
│   ├── fallback/                # Fallback mechanisms
│   │   ├── template_mapper.py   # Template-based mapping
│   │   ├── rule_engine.py      # Rule-based mapping
│   │   └── fallback_coordinator.py # Fallback coordination
│   │
│   ├── monitoring/              # Monitoring & observability
│   │   ├── cost/               # Cost monitoring
│   │   │   ├── cost_monitor.py     # Real-time cost tracking
│   │   │   ├── budget_alerts.py    # Budget alerting
│   │   │   └── cost_optimizer.py   # Cost optimization
│   │   ├── performance/        # Performance monitoring
│   │   │   ├── performance_tracker.py # Performance metrics
│   │   │   ├── latency_monitor.py    # Latency monitoring
│   │   │   └── throughput_analyzer.py # Throughput analysis
│   │   ├── quality/            # Model quality monitoring
│   │   │   ├── drift_detector.py    # Model drift detection
│   │   │   ├── quality_metrics.py   # Quality assessment
│   │   │   └── anomaly_detection.py # Performance anomaly detection
│   │   ├── observability/      # Advanced observability
│   │   │   ├── distributed_tracing.py # Cross-service tracing
│   │   │   ├── correlation_analysis.py # Event correlation
│   │   │   └── root_cause_analysis.py # Automated RCA
│   │   └── analytics/          # Usage analytics
│   │       ├── usage_analytics.py   # Usage pattern analysis
│   │       ├── tenant_analytics.py  # Tenant-specific analytics
│   │       └── capacity_planning.py # Capacity planning
│   │
│   ├── cli/                     # CLI commands
│   │   ├── mapping_commands.py  # Mapping operations
│   │   ├── training_commands.py # Training operations
│   │   └── model_commands.py   # Model management
│   │
│   ├── tenancy/                 # Multi-tenancy support
│   │   ├── tenant_models.py    # Tenant-specific models
│   │   ├── tenant_isolation.py # Mapping data isolation
│   │   ├── tenant_customization.py # Tenant-specific mapping rules
│   │   └── tenant_billing.py   # Tenant cost tracking
│   │
│   ├── plugins/                 # Plugin system
│   │   ├── mapping_plugins.py  # Mapping engine plugins
│   │   ├── model_plugins.py    # Model serving plugins
│   │   ├── validation_plugins.py # Validation plugins
│   │   └── fallback_plugins.py # Fallback mechanism plugins
│   │
│   ├── pipelines/               # Pipeline management
│   │   ├── training_pipeline.py    # Model training pipeline
│   │   ├── deployment_pipeline.py  # Model deployment pipeline
│   │   ├── validation_pipeline.py  # Response validation pipeline
│   │   └── optimization_pipeline.py # Performance optimization pipeline
│   │
│   └── config/                  # Configuration
│       ├── settings.py         # Service settings
│       ├── mapping_config.py   # Mapping configuration
│       ├── model_config.py     # Model configuration
│       └── tenant_config.py    # Multi-tenant configuration
```

#### Key Interfaces
```python
# Service API Contract
class MappingRequest(BaseModel):
    analysis_response: AnalysisResponse
    output_format: OutputFormat
    tenant_id: str
    idempotency_key: Optional[str] = None

class MappingResponse(BaseModel):
    request_id: str
    taxonomy: List[str]
    scores: Dict[str, float]
    confidence: float
    framework_mappings: Dict[str, Any]
    version_info: VersionInfo
    processing_time_ms: float
    cost_metrics: CostMetrics
```

## Database Architecture

### Multi-Database Strategy

#### Database Distribution
```
┌─────────────────────────────────────────────────────────────────┐
│                    Database Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  Detector Orchestration Service                                │
│  ├── PostgreSQL (Primary)                                      │
│  │   ├── detector_registry                                     │
│  │   ├── policy_store                                          │
│  │   ├── orchestration_audit                                   │
│  │   └── health_monitoring                                     │
│  └── Redis (Cache)                                             │
│      ├── idempotency_cache                                     │
│      ├── response_cache                                        │
│      └── rate_limit_store                                      │
├─────────────────────────────────────────────────────────────────┤
│  Analysis Service                                               │
│  ├── PostgreSQL (Primary)                                      │
│  │   ├── analysis_results                                      │
│  │   ├── quality_metrics                                       │
│  │   ├── pattern_analysis                                      │
│  │   └── compliance_mappings                                   │
│  ├── ClickHouse (Analytics)                                    │
│  │   ├── time_series_metrics                                   │
│  │   ├── performance_analytics                                 │
│  │   └── business_intelligence                                 │
│  └── Redis (Cache)                                             │
│      ├── analysis_cache                                        │
│      ├── rag_embeddings                                        │
│      └── quality_cache                                         │
├─────────────────────────────────────────────────────────────────┤
│  Mapper Service                                                 │
│  ├── PostgreSQL (Primary)                                      │
│  │   ├── mapping_results                                       │
│  │   ├── model_versions                                        │
│  │   ├── training_metadata                                     │
│  │   └── cost_tracking                                         │
│  ├── S3/MinIO (Object Storage)                                 │
│  │   ├── model_artifacts                                       │
│  │   ├── training_checkpoints                                  │
│  │   └── backup_data                                           │
│  └── Redis (Cache)                                             │
│      ├── model_cache                                           │
│      ├── response_cache                                        │
│      └── cost_cache                                            │
└─────────────────────────────────────────────────────────────────┘
```

#### Azure Integration Support
```python
# Azure-specific database configurations
class AzureDatabaseConfig:
    # Azure SQL Database integration
    azure_sql_connection: str
    # Azure Cosmos DB for global distribution
    cosmos_db_endpoint: str
    # Azure Redis Cache
    azure_redis_connection: str
    # Azure Blob Storage
    azure_storage_account: str
    # Azure Key Vault for secrets
    key_vault_url: str
```

### Backup and Disaster Recovery

#### Multi-Service Backup Strategy
```python
class BackupOrchestrator:
    """Coordinates backups across all services"""
    
    async def execute_full_backup(self):
        """Execute coordinated backup across all services"""
        backup_tasks = [
            self.backup_orchestration_data(),
            self.backup_analysis_data(),
            self.backup_mapper_data(),
            self.backup_shared_resources()
        ]
        
        results = await asyncio.gather(*backup_tasks, return_exceptions=True)
        return self.validate_backup_consistency(results)
    
    async def restore_from_backup(self, backup_timestamp: datetime):
        """Restore all services from coordinated backup"""
        # Implement point-in-time recovery across services
        pass
```

## Data Models

### Core Domain Models

#### Shared Models (via HTTP contracts)
```python
# Cross-service communication models
class DetectorResult(BaseModel):
    detector: str
    output: str
    confidence: float
    processing_time_ms: float
    metadata: Dict[str, Any]

class PatternAnalysisResult(BaseModel):
    patterns_detected: List[Pattern]
    temporal_analysis: TemporalAnalysis
    frequency_analysis: FrequencyAnalysis
    correlation_analysis: CorrelationAnalysis
    anomaly_score: float

class RiskScoringResult(BaseModel):
    overall_risk_score: float
    technical_risk: float
    business_risk: float
    regulatory_risk: float
    temporal_risk: float
    risk_factors: List[RiskFactor]

class ComplianceMappingResult(BaseModel):
    framework: str
    controls: List[ComplianceControl]
    evidence_requirements: List[EvidenceRequirement]
    compliance_score: float
```

#### Service-Specific Models

**Detector Orchestration Models:**
```python
class RoutingPlan(BaseModel):
    primary_detectors: List[str]
    secondary_detectors: List[str]
    coverage_requirements: Dict[str, float]
    timeout_ms: int

class PolicyDecision(BaseModel):
    selected_detectors: List[str]
    policy_applied: str
    routing_reason: str
    coverage_method: CoverageMethod
```

**Analysis Service Models:**
```python
class QualityMetrics(BaseModel):
    analysis_quality_score: float
    confidence_distribution: Dict[str, float]
    pattern_consistency: float
    risk_calibration: float

class RAGInsights(BaseModel):
    regulatory_context: List[RegulatoryContext]
    compliance_guidance: List[ComplianceGuidance]
    risk_mitigation_suggestions: List[RiskMitigation]
```

**Mapper Service Models:**
```python
class VersionInfo(BaseModel):
    taxonomy_version: str
    model_version: str
    framework_version: str
    training_date: datetime

class CostMetrics(BaseModel):
    inference_cost: float
    compute_time_ms: float
    token_usage: TokenUsage
    resource_utilization: ResourceUtilization
```

## Error Handling

### Error Hierarchy
```python
# Base service errors
class ServiceError(Exception):
    def __init__(self, message: str, error_code: str, retryable: bool = False):
        self.message = message
        self.error_code = error_code
        self.retryable = retryable

# Service-specific errors
class OrchestrationError(ServiceError):
    pass

class AnalysisError(ServiceError):
    pass

class MappingError(ServiceError):
    pass

# Specific error types
class ModelServingError(ServiceError):
    pass

class ValidationError(ServiceError):
    pass

class PolicyEnforcementError(ServiceError):
    pass
```

### Circuit Breaker Implementation
```python
class ServiceCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    async def call_with_circuit_breaker(self, func: Callable, *args, **kwargs):
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

## Testing Strategy

### Testing Pyramid

#### Unit Tests (70%)
- **Detector Orchestration**: Policy logic, routing algorithms, aggregation logic
- **Analysis Service**: Pattern recognition algorithms, risk scoring calculations, compliance mapping
- **Mapper Service**: Core mapping logic, validation rules, fallback mechanisms

#### Integration Tests (20%)
- **Service-to-Service**: HTTP API contracts, error handling, timeout behavior
- **Database Integration**: Schema validation, data consistency, migration testing
- **External Dependencies**: Model serving backends, Redis caching, monitoring systems

#### End-to-End Tests (10%)
- **Complete Workflows**: Full request flow through all three services
- **Performance Testing**: Load testing, latency validation, resource utilization
- **Failure Scenarios**: Service failures, network partitions, data corruption

### Test Infrastructure
```python
# Service test fixtures
@pytest.fixture
async def orchestration_service():
    """Fixture for orchestration service testing"""
    config = TestOrchestrationConfig()
    service = OrchestrationService(config)
    await service.start()
    yield service
    await service.stop()

@pytest.fixture
async def analysis_service():
    """Fixture for analysis service testing"""
    config = TestAnalysisConfig()
    service = AnalysisService(config)
    await service.start()
    yield service
    await service.stop()

@pytest.fixture
async def mapper_service():
    """Fixture for mapper service testing"""
    config = TestMapperConfig()
    service = MapperService(config)
    await service.start()
    yield service
    await service.stop()

# Contract testing
class TestServiceContracts:
    async def test_orchestration_to_analysis_contract(self):
        """Test the contract between orchestration and analysis services"""
        orchestration_response = create_test_orchestration_response()
        analysis_request = AnalysisRequest.from_orchestration_response(orchestration_response)
        assert analysis_request.is_valid()
    
    async def test_analysis_to_mapper_contract(self):
        """Test the contract between analysis and mapper services"""
        analysis_response = create_test_analysis_response()
        mapping_request = MappingRequest.from_analysis_response(analysis_response)
        assert mapping_request.is_valid()
```

This design provides a comprehensive architecture for the 3-microservice refactoring while preserving all existing functionality and ensuring clean, maintainable code organization.