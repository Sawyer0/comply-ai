# Product Overview

Llama Mapper is a privacy-first, audit-ready service that normalizes raw detector outputs into a canonical taxonomy for compliance evidence generation. The system uses LoRA fine-tuning on Llama-3-8B-Instruct to create deterministic mappings from various AI safety detectors to standardized labels.

## Core Features

- **Privacy-First Architecture**: Metadata-only logging, no raw detector inputs persisted
- **Compliance Ready**: Framework mapping for SOC 2, ISO 27001, and HIPAA compliance  
- **High Performance**: vLLM/TGI serving with confidence-based fallback
- **Audit Trail**: Complete versioning for taxonomy, models, and frameworks
- **Rate Limited**: Configurable rate limiting with Redis backend support
- **Idempotency**: Redis-backed caching for consistent responses

## Dual-Model Architecture

The system employs two specialized models:

1. **Llama-3-8B (Compliance Mapper)**: Raw detector output → Canonical taxonomy mapping
2. **Phi-3-Mini (Compliance Analyst)**: Context-aware risk assessments and remediation guidance

## Key Use Cases

- Normalizing outputs from diverse AI safety detectors
- Generating compliance evidence for audits
- Providing framework-specific risk assessments
- Mapping PII, security, and content moderation alerts to standardized taxonomies

## Business Value Proposition

### For Compliance Teams
- **Unified Evidence Generation**: Single source of truth for compliance evidence across multiple frameworks
- **Audit Readiness**: Complete audit trails with versioned taxonomies and deterministic outputs
- **Risk Assessment**: Context-aware analysis with remediation guidance
- **Framework Agnostic**: Support for SOC 2, ISO 27001, HIPAA, and custom compliance frameworks

### For Security Teams
- **Detector Normalization**: Standardize outputs from diverse AI safety tools (Presidio, DeBERTa, custom detectors)
- **False Positive Reduction**: Confidence-based filtering and intelligent aggregation
- **Scalable Processing**: High-throughput processing with horizontal scaling capabilities
- **Integration Ready**: RESTful APIs with comprehensive SDK support

### For Engineering Teams
- **Privacy by Design**: No raw content persistence, metadata-only logging
- **Production Ready**: Enterprise-grade reliability with circuit breakers and fallback mechanisms
- **Observability**: Comprehensive monitoring with Prometheus/Grafana integration
- **Cost Optimization**: Built-in cost monitoring and resource optimization

## Product Architecture

### Service Components

#### Core Mapper Service
- **Purpose**: Primary mapping service for detector output normalization
- **Technology**: FastAPI + vLLM/TGI serving
- **Scaling**: Horizontal scaling with load balancing
- **Storage**: PostgreSQL for metadata, Redis for caching

#### Detector Orchestration Service
- **Purpose**: Coordinates multiple detector executions and aggregates results
- **Technology**: Separate FastAPI service with service discovery
- **Features**: Circuit breakers, retry logic, policy enforcement
- **Integration**: OPA policy engine for governance

#### Analysis Service (Containerized)
- **Purpose**: Advanced analytics and reporting capabilities
- **Technology**: Containerized Python service
- **Features**: Batch processing, trend analysis, compliance reporting
- **Storage**: ClickHouse for analytics data

### Data Flow Architecture

```
Input → Detector Orchestration → Individual Detectors → Mapper Service → Canonical Output
  ↓                                                                           ↓
Audit Trail ← Analysis Service ← Compliance Framework Mapping ← Taxonomy Normalization
```

## Supported Compliance Frameworks

### SOC 2 Type II
- **Controls**: CC6.1 (Logical Access), CC6.7 (Data Transmission), CC7.1 (System Boundaries)
- **Evidence Types**: Access logs, data classification, boundary controls
- **Automation**: Automated evidence collection and control testing

### ISO 27001:2022
- **Controls**: A.8.2.1 (Data Classification), A.8.2.2 (Data Labeling), A.13.2.1 (Information Transfer)
- **Evidence Types**: Classification records, transfer logs, access controls
- **Reporting**: Automated compliance dashboards and audit reports

### HIPAA
- **Safeguards**: Administrative, Physical, Technical safeguards
- **PHI Protection**: Automated PHI detection and classification
- **Breach Prevention**: Real-time monitoring and alerting

### Custom Frameworks
- **Extensible Taxonomy**: Support for custom compliance taxonomies
- **Framework Mapping**: Configurable mapping between internal taxonomies and framework requirements
- **Policy Engine**: OPA-based policy enforcement for custom rules

## Integration Patterns

### API Integration
- **RESTful APIs**: OpenAPI 3.0 specification with comprehensive documentation
- **Authentication**: API key-based with tenant isolation
- **Rate Limiting**: Configurable limits per tenant/API key
- **Webhooks**: Event-driven notifications for compliance events

### SDK Support
- **Python SDK**: Full-featured SDK with async support
- **JavaScript SDK**: Browser and Node.js compatible
- **Go SDK**: High-performance SDK for microservices
- **Java SDK**: Enterprise integration support
- **C# SDK**: .NET ecosystem integration

### Batch Processing
- **Bulk Analysis**: Process large datasets efficiently
- **Scheduled Jobs**: Automated compliance checks and reporting
- **Data Pipeline**: Integration with ETL/ELT pipelines
- **Export Formats**: JSON, CSV, PDF report generation

## Quality Assurance

### Model Quality
- **Confidence Scoring**: All outputs include confidence metrics
- **Fallback Mechanisms**: Rule-based fallbacks for low-confidence predictions
- **Continuous Evaluation**: Automated model performance monitoring
- **A/B Testing**: Safe model deployment with gradual rollouts

### Data Quality
- **Input Validation**: Comprehensive input sanitization and validation
- **Output Verification**: Automated output quality checks
- **Golden Dataset**: Curated test cases for regression testing
- **Performance Benchmarks**: Latency and throughput monitoring

### Security & Privacy
- **Zero-Trust Architecture**: All communications encrypted and authenticated
- **Data Minimization**: Only necessary metadata stored
- **Retention Policies**: Configurable data retention and deletion
- **Compliance Auditing**: Regular security and compliance audits

## Operational Excellence

### Monitoring & Observability
- **Metrics**: Comprehensive Prometheus metrics for all components
- **Logging**: Structured logging with correlation IDs
- **Tracing**: Distributed tracing for request flow analysis
- **Alerting**: Intelligent alerting with escalation policies

### Reliability & Performance
- **SLA Targets**: 99.9% uptime with sub-100ms p95 latency
- **Disaster Recovery**: Multi-region deployment with automated failover
- **Capacity Planning**: Predictive scaling based on usage patterns
- **Performance Optimization**: Continuous performance tuning and optimization

### Cost Management
- **Resource Optimization**: Intelligent resource allocation and scaling
- **Cost Monitoring**: Real-time cost tracking and budgeting
- **Usage Analytics**: Detailed usage reporting and optimization recommendations
- **Billing Integration**: Transparent usage-based billing with cost controls