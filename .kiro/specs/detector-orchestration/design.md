# Design Document

## Overview

The Detector Orchestration Layer is a new microservice that sits above the existing Llama Mapper infrastructure to coordinate multiple detector services. It acts as an intelligent traffic router and response aggregator, handling the complexity of multi-detector workflows while leveraging the existing mapping capabilities.

The orchestration layer receives content analysis requests, determines which detectors should process the content based on tenant policies and content type, coordinates parallel detector execution, monitors detector health, and aggregates responses into a unified payload that can be consumed by the existing `/map` endpoint.

## Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        CLIENT[Client Applications]
        BATCH[Batch Processors]
    end
    
    subgraph "Orchestration Layer (NEW)"
        API[FastAPI /orchestrate Endpoint]
        ROUTER[Content Router]
        HEALTH[Health Monitor]
        COORD[Detector Coordinator]
        AGG[Response Aggregator]
        CACHE[Response Cache]
    end
    
    subgraph "Detector Services"
        DET1[deberta-toxicity]
        DET2[llama-guard]
        DET3[openai-moderation]
        DET4[regex-pii]
        DETN[Other Detectors...]
    end
    
    subgraph "Existing Mapper Layer"
        MAPPER[/map Endpoint]
        MODEL[Model Server]
        VALIDATOR[JSON Validator]
        FALLBACK[Fallback Mapper]
    end
    
    CLIENT --> API
    BATCH --> API
    API --> ROUTER
    ROUTER --> COORD
    COORD --> DET1
    COORD --> DET2
    COORD --> DET3
    COORD --> DET4
    COORD --> DETN
    DET1 --> AGG
    DET2 --> AGG
    DET3 --> AGG
    DET4 --> AGG
    DETN --> AGG
    AGG --> MAPPER
    HEALTH --> DET1
    HEALTH --> DET2
    HEALTH --> DET3
    HEALTH --> DET4
    HEALTH --> DETN
    CACHE --> API
```

### Service Integration

The orchestration layer integrates with the existing ecosystem by:

- **Upstream**: Accepting requests from clients that currently call detectors directly
- **Downstream**: Sending aggregated detector outputs to the existing `/map` endpoint
- **Lateral**: Coordinating with detector services via their native APIs
- **Infrastructure**: Sharing monitoring, configuration, and deployment patterns

## Components and Interfaces

### 1. Orchestration API Service (`src/llama_mapper/orchestration/`)

**Primary Interfaces:**

```python
@app.post("/orchestrate", response_model=OrchestrationResponse)
async def orchestrate_detection(request: OrchestrationRequest) -> OrchestrationResponse

@app.post("/orchestrate/batch", response_model=BatchOrchestrationResponse)
async def orchestrate_detection_batch(
    request: BatchOrchestrationRequest,
    idempotency_key: str = Header(...)
) -> BatchOrchestrationResponse

@app.get("/orchestrate/status/{job_id}")
async def get_job_status(job_id: str) -> JobStatusResponse
```

**Input Schema (OrchestrationRequest):**
```python
class OrchestrationRequest(BaseModel):
    content: str = Field(..., max_length=50000, description="Content to analyze")
    content_type: ContentType = Field(..., description="Type of content (text, image, etc.)")
    tenant_id: str = Field(..., min_length=1, max_length=64)
    policy_bundle: str = Field(..., description="Policy bundle defining detector requirements")
    environment: Literal["dev", "stage", "prod"]
    processing_mode: ProcessingMode = Field(default=ProcessingMode.SYNC)
    priority: Priority = Field(default=Priority.NORMAL)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
    # Override specific detector selection (optional)
    required_detectors: Optional[List[str]] = Field(None, description="Force specific detectors")
    excluded_detectors: Optional[List[str]] = Field(None, description="Exclude specific detectors")

class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"
    CODE = "code"

class ProcessingMode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"

class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
```

**Output Schema (OrchestrationResponse):**
```python
class DetectorResult(BaseModel):
    detector: str
    status: DetectorStatus
    output: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: int
    confidence: Optional[float] = None

class DetectorStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"

# MAPPER HANDOFF SCHEMA (MUST-LOCK)
class MapperPayload(BaseModel):
    """Exact schema for handoff to existing /map endpoint."""
    detector: str = Field(..., description="Aggregated detector identifier")
    output: str = Field(..., description="Normalized aggregated output")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Aggregation metadata")
    tenant_id: str = Field(..., description="Tenant identifier")
    
    # Aggregation-specific metadata
    class Config:
        schema_extra = {
            "example": {
                "detector": "orchestrated-multi",
                "output": "toxic|hate|pii_detected",
                "metadata": {
                    "contributing_detectors": ["deberta-toxicity", "llama-guard", "regex-pii"],
                    "normalized_scores": {
                        "HARM.SPEECH.Toxicity": 0.85,
                        "HARM.SPEECH.Hate.Other": 0.72,
                        "PII.Identifier.SSN": 0.95
                    },
                    "conflict_resolution_applied": True,
                    "coverage_achieved": 1.0,
                    "aggregation_method": "weighted_average",
                    "provenance": [
                        {"detector": "deberta-toxicity", "confidence": 0.85, "output": "toxic"},
                        {"detector": "llama-guard", "confidence": 0.72, "output": "hate"},
                        {"detector": "regex-pii", "confidence": 0.95, "output": "ssn_detected"}
                    ]
                },
                "tenant_id": "tenant-123"
            }
        }

class OrchestrationResponse(BaseModel):
    request_id: str
    job_id: Optional[str] = None  # For async requests
    processing_mode: ProcessingMode
    detector_results: List[DetectorResult]
    aggregated_payload: Optional[MapperPayload] = None  # Locked schema for mapper
    mapping_result: Optional[MappingResponse] = None  # If auto-mapping enabled
    
    # Orchestration metadata
    total_processing_time_ms: int
    detectors_attempted: int
    detectors_succeeded: int
    detectors_failed: int
    coverage_achieved: float  # 0.0-1.0, see coverage semantics below
    
    # Health and routing info
    routing_decision: RoutingDecision
    fallback_used: bool
    timestamp: datetime
    
    # Error handling
    error_code: Optional[str] = None  # See canonical error codes below
    idempotency_key: Optional[str] = None  # For retry safety

class RoutingDecision(BaseModel):
    selected_detectors: List[str]
    routing_reason: str
    policy_applied: str
    coverage_requirements: Dict[str, float]
    health_status: Dict[str, bool]
```

### 2. Content Router (`src/llama_mapper/orchestration/router.py`)

Determines which detectors should process content based on policies and content analysis:

```python
class ContentRouter:
    def __init__(self, policy_manager: PolicyManager, detector_registry: DetectorRegistry):
        self.policy_manager = policy_manager
        self.detector_registry = detector_registry
        
    async def route_request(self, request: OrchestrationRequest) -> RoutingPlan:
        """Determine which detectors should process the request."""
        
    def analyze_content_type(self, content: str, declared_type: ContentType) -> ContentAnalysis:
        """Analyze content to determine optimal detector selection."""
        
    def apply_tenant_policy(self, tenant_id: str, policy_bundle: str) -> PolicyRequirements:
        """Load and apply tenant-specific detector policies."""
        
    def calculate_coverage_requirements(self, policy: PolicyRequirements) -> Dict[str, float]:
        """Calculate minimum detector coverage requirements."""

class RoutingPlan(BaseModel):
    primary_detectors: List[str]  # Must-run detectors
    secondary_detectors: List[str]  # Optional/fallback detectors
    parallel_groups: List[List[str]]  # Detectors that can run in parallel
    sequential_dependencies: Dict[str, List[str]]  # Detector dependencies
    timeout_config: Dict[str, int]  # Per-detector timeouts
    retry_config: Dict[str, int]  # Per-detector retry counts
```

### 3. Detector Coordinator (`src/llama_mapper/orchestration/coordinator.py`)

Manages parallel execution of multiple detectors:

```python
class DetectorCoordinator:
    def __init__(self, detector_clients: Dict[str, DetectorClient], circuit_breaker: CircuitBreaker):
        self.detector_clients = detector_clients
        self.circuit_breaker = circuit_breaker
        
    async def execute_routing_plan(self, 
                                 content: str, 
                                 routing_plan: RoutingPlan,
                                 request_id: str) -> List[DetectorResult]:
        """Execute the routing plan with parallel coordination."""
        
    async def execute_detector_group(self, 
                                   detectors: List[str], 
                                   content: str,
                                   timeout: int) -> List[DetectorResult]:
        """Execute a group of detectors in parallel."""
        
    async def execute_single_detector(self, 
                                    detector: str, 
                                    content: str,
                                    timeout: int) -> DetectorResult:
        """Execute a single detector with error handling."""
        
    def handle_detector_failure(self, detector: str, error: Exception) -> DetectorResult:
        """Handle detector failures with appropriate fallback."""

class DetectorClient:
    """Client for communicating with individual detector services."""
    
    def __init__(self, detector_name: str, endpoint: str, auth_config: Dict[str, Any]):
        self.detector_name = detector_name
        self.endpoint = endpoint
        self.auth_config = auth_config
        self.session = aiohttp.ClientSession()
        
    async def analyze(self, content: str, metadata: Dict[str, Any] = None) -> DetectorResult:
        """Send content to detector for analysis."""
        
    async def health_check(self) -> bool:
        """Check if detector is healthy and available."""
        
    async def get_capabilities(self) -> DetectorCapabilities:
        """Get detector capabilities and supported content types."""

class DetectorCapabilities(BaseModel):
    supported_content_types: List[ContentType]
    max_content_length: int
    average_processing_time_ms: int
    confidence_calibrated: bool
    batch_supported: bool
```

### 4. Health Monitor (`src/llama_mapper/orchestration/health_monitor.py`)

Continuously monitors detector health and manages availability:

```python
class HealthMonitor:
    def __init__(self, detector_clients: Dict[str, DetectorClient], check_interval: int = 30):
        self.detector_clients = detector_clients
        self.check_interval = check_interval
        self.health_status = {}
        self.last_check = {}
        
    async def start_monitoring(self):
        """Start continuous health monitoring background task."""
        
    async def check_all_detectors(self) -> Dict[str, HealthStatus]:
        """Check health of all registered detectors."""
        
    async def check_detector_health(self, detector: str) -> HealthStatus:
        """Check health of a specific detector."""
        
    def get_healthy_detectors(self) -> List[str]:
        """Get list of currently healthy detectors."""
        
    def get_detector_health(self, detector: str) -> HealthStatus:
        """Get current health status of a specific detector."""
        
    def mark_detector_unhealthy(self, detector: str, reason: str):
        """Mark a detector as unhealthy."""
        
    def mark_detector_healthy(self, detector: str):
        """Mark a detector as healthy."""

class HealthStatus(BaseModel):
    detector: str
    is_healthy: bool
    last_check: datetime
    response_time_ms: Optional[int]
    error_message: Optional[str]
    consecutive_failures: int
    uptime_percentage: float
```

### 5. Response Aggregator (`src/llama_mapper/orchestration/aggregator.py`)

Combines multiple detector outputs into unified payloads:

```python
class ResponseAggregator:
    def __init__(self, conflict_resolver: ConflictResolver):
        self.conflict_resolver = conflict_resolver
        
    async def aggregate_results(self, 
                              detector_results: List[DetectorResult],
                              routing_plan: RoutingPlan) -> AggregatedResponse:
        """Aggregate multiple detector results into unified payload."""
        
    def create_mapper_payload(self, aggregated_response: AggregatedResponse) -> Dict[str, Any]:
        """Create payload suitable for existing /map endpoint."""
        
    def calculate_coverage_score(self, 
                               detector_results: List[DetectorResult],
                               requirements: Dict[str, float]) -> float:
        """Calculate achieved coverage percentage."""
        
    def resolve_conflicts(self, conflicting_results: List[DetectorResult]) -> DetectorResult:
        """Resolve conflicts between detector outputs."""

class AggregatedResponse(BaseModel):
    primary_results: List[DetectorResult]  # Results from required detectors
    secondary_results: List[DetectorResult]  # Results from optional detectors
    consensus_output: Optional[str]  # Consensus across detectors
    confidence_score: float  # Aggregated confidence
    coverage_achieved: float
    conflict_resolution_applied: bool
    provenance: List[str]  # List of contributing detectors

class ConflictResolver:
    """Handles conflicts between detector outputs."""
    
    def __init__(self, resolution_strategy: ConflictResolutionStrategy):
        self.strategy = resolution_strategy
        
    def resolve(self, conflicting_results: List[DetectorResult]) -> DetectorResult:
        """Resolve conflicts using configured strategy."""

class ConflictResolutionStrategy(str, Enum):
    HIGHEST_CONFIDENCE = "highest_confidence"
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    MOST_RESTRICTIVE = "most_restrictive"
    TENANT_PREFERENCE = "tenant_preference"

# CONFLICT RESOLUTION POLICY (LOCKED)
CONFLICT_RESOLUTION_DEFAULTS = {
    ContentType.TEXT: ConflictResolutionStrategy.WEIGHTED_AVERAGE,
    ContentType.IMAGE: ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
    ContentType.DOCUMENT: ConflictResolutionStrategy.MOST_RESTRICTIVE,
    ContentType.CODE: ConflictResolutionStrategy.MAJORITY_VOTE
}

class ConflictResolution(BaseModel):
    """Conflict resolution result with full audit trail."""
    strategy_used: ConflictResolutionStrategy
    conflicting_detectors: List[str]
    winning_result: DetectorResult
    tie_breaker_applied: Optional[str] = None  # e.g., "alphabetical", "detector_priority"
    confidence_delta: float  # Difference between top two results
    resolution_notes: str  # Logged in mapping_result.notes
```

### 6. Policy Manager (`src/llama_mapper/orchestration/policy_manager.py`)

Manages tenant-specific detector policies and requirements:

```python
class PolicyManager:
    def __init__(self, policy_store: PolicyStore):
        self.policy_store = policy_store
        
    async def get_tenant_policy(self, tenant_id: str, policy_bundle: str) -> TenantPolicy:
        """Get policy configuration for a tenant."""
        
    async def validate_policy(self, policy: TenantPolicy) -> PolicyValidationResult:
        """Validate policy configuration."""
        
    async def update_tenant_policy(self, tenant_id: str, policy: TenantPolicy):
        """Update tenant policy configuration."""

# COVERAGE SEMANTICS (LOCKED)
class CoverageCalculation(BaseModel):
    """Defines how coverage is calculated and what 100% means."""
    calculation_method: CoverageMethod
    required_detector_set: List[str]  # Must all succeed for 100%
    weighted_requirements: Dict[str, float]  # detector -> weight (sum to 1.0)
    partial_coverage_threshold: float = 0.8  # Below this = HTTP 206, fallback_used=True
    
class CoverageMethod(str, Enum):
    REQUIRED_SET = "required_set"  # 100% = all required detectors succeeded
    WEIGHTED_COVERAGE = "weighted_coverage"  # 100% = weighted sum of successes = 1.0
    TAXONOMY_COVERAGE = "taxonomy_coverage"  # 100% = all taxonomy categories covered

class TenantPolicy(BaseModel):
    tenant_id: str
    policy_bundle: str
    version: str = "1.0"  # For policy migration/versioning
    
    # OPA/Rego policy integration
    rego_policy_path: Optional[str] = None  # Path to Rego policy file
    policy_data: Dict[str, Any] = {}  # Data for Rego policy evaluation
    
    # Coverage requirements (can be overridden by Rego)
    coverage_calculation: CoverageCalculation
    required_detectors: List[str]
    optional_detectors: List[str]
    
    # Conflict resolution
    conflict_resolution: ConflictResolutionStrategy
    conflict_resolution_overrides: Dict[ContentType, ConflictResolutionStrategy] = {}
    
    # Timeouts and retries
    timeout_config: Dict[str, int]
    retry_config: Dict[str, int]
    priority_weights: Dict[str, float]  # detector -> priority weight
    content_type_routing: Dict[ContentType, List[str]]
    
    # Security and access
    rbac_scopes: List[str] = ["orchestrate:read", "orchestrate:write"]
    rate_limit_per_minute: int = 1000

class OPAPolicyEngine:
    """OPA integration for policy-as-code orchestration rules."""
    
    def __init__(self, opa_endpoint: str = "http://localhost:8181"):
        self.opa_endpoint = opa_endpoint
        self.session = aiohttp.ClientSession()
        
    async def evaluate_detector_selection(self, 
                                        tenant_policy: TenantPolicy,
                                        request: OrchestrationRequest) -> DetectorSelectionResult:
        """Use OPA to evaluate which detectors should be selected."""
        
    async def evaluate_coverage_requirements(self,
                                           tenant_policy: TenantPolicy,
                                           detector_results: List[DetectorResult]) -> CoverageEvaluationResult:
        """Use OPA to evaluate if coverage requirements are met."""
        
    async def evaluate_conflict_resolution(self,
                                         tenant_policy: TenantPolicy,
                                         conflicting_results: List[DetectorResult]) -> ConflictResolutionResult:
        """Use OPA to determine conflict resolution strategy."""
        
    async def compile_policy(self, rego_policy: str) -> PolicyCompilationResult:
        """Compile and validate Rego policy."""
        
    async def load_policy_bundle(self, policy_bundle_path: str):
        """Load policy bundle into OPA."""

class DetectorSelectionResult(BaseModel):
    selected_detectors: List[str]
    selection_reason: str
    policy_applied: str
    rego_decision: Dict[str, Any]  # Raw OPA decision

class CoverageEvaluationResult(BaseModel):
    coverage_met: bool
    coverage_score: float
    missing_requirements: List[str]
    policy_violations: List[str]
    
class ConflictResolutionResult(BaseModel):
    resolution_strategy: ConflictResolutionStrategy
    winning_detector: str
    resolution_reason: str
    
class PolicyStore:
    """Interface for policy storage backend with migration/versioning."""
    
    async def get_policy(self, tenant_id: str, policy_bundle: str) -> Optional[TenantPolicy]:
        """Retrieve policy from storage."""
        
    async def store_policy(self, policy: TenantPolicy):
        """Store policy to backend."""
        
    async def list_policies(self, tenant_id: str) -> List[str]:
        """List available policy bundles for tenant."""
        
    async def migrate_policy(self, tenant_id: str, policy_bundle: str, from_version: str, to_version: str):
        """Migrate policy between versions."""
        
    async def validate_policy(self, policy: TenantPolicy) -> PolicyValidationResult:
        """Validate policy before storage."""

class PolicyValidationCLI:
    """CLI tool for validating tenant policy bundles before rollout."""
    
    def validate_bundle(self, policy_file: str) -> ValidationReport:
        """Validate policy bundle file."""
        
    def validate_all_tenants(self, policy_dir: str) -> Dict[str, ValidationReport]:
        """Validate all tenant policies in directory."""
        
    def check_detector_availability(self, policy: TenantPolicy) -> List[str]:
        """Check if all required detectors are available."""
```

## Data Models

### Core Orchestration Types

```python
# Job management for async processing
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AsyncJob(BaseModel):
    job_id: str
    request_id: str
    tenant_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    progress: float  # 0.0 to 1.0
    result: Optional[OrchestrationResponse]
    error: Optional[str]

# Circuit breaker for detector health
class CircuitBreakerState(str, Enum):
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        
    async def call(self, detector: str, func: Callable) -> Any:
        """Execute function with circuit breaker protection."""
        
    def record_success(self, detector: str):
        """Record successful detector call."""
        
    def record_failure(self, detector: str):
        """Record failed detector call."""
```

### Configuration Models

```python
# SLA & TIMEOUTS (LOCKED)
class SLAConfig(BaseModel):
    """Service Level Agreement configuration."""
    sync_request_sla_ms: int = 2000  # Global SLA for sync requests
    async_request_sla_ms: int = 30000  # Global SLA for async requests
    mapper_timeout_budget_ms: int = 500  # Time reserved for mapper call
    sync_to_async_threshold_ms: int = 1500  # Convert to async if exceeded
    
    # Per-priority SLA overrides
    priority_sla_overrides: Dict[Priority, int] = {
        Priority.CRITICAL: 1000,  # Faster SLA for critical requests
        Priority.LOW: 5000  # Slower SLA for low priority
    }

class OrchestrationConfig(BaseModel):
    # Service configuration
    max_concurrent_detectors: int = 10
    default_timeout_ms: int = 5000
    max_retries: int = 2
    sla_config: SLAConfig = SLAConfig()
    
    # Health monitoring (LOCKED SEMANTICS)
    health_check_interval_seconds: int = 30
    unhealthy_threshold: int = 3  # Failures before marking unhealthy
    unhealthy_removal_timeout_seconds: int = 30  # Remove from pool ≤30s per requirement
    recovery_check_interval_seconds: int = 60
    half_open_test_requests: int = 3  # Test requests in half-open state
    
    # Circuit breaker
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout_seconds: int = 60
    
    # Caching policy (LOCKED)
    response_cache_ttl_seconds: int = 300
    cache_enabled: bool = True
    cache_key_components: List[str] = ["content_hash", "detector_set", "policy_bundle"]
    critical_priority_cache_bypass: bool = True  # CRITICAL requests bypass cache
    cache_invalidation_on_policy_change: bool = True
    
    # Security & tenancy
    rbac_enabled: bool = True
    default_rate_limit_per_minute: int = 1000
    log_content_redaction: bool = True  # Never log raw content
    metrics_content_redaction: bool = True  # Never include content in metrics
    
    # Auto-mapping
    auto_map_results: bool = True
    mapper_endpoint: str = "http://localhost:8000/map"

class DetectorConfig(BaseModel):
    name: str
    endpoint: str
    auth_type: AuthType
    auth_config: Dict[str, Any]
    timeout_ms: int
    max_retries: int
    health_check_path: str
    capabilities: DetectorCapabilities
    priority_weight: float = 1.0
    
class AuthType(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
```

## Error Handling

### Error Classification

**CANONICAL ERROR CODE TABLE:**

| HTTP Code | Error Code | Description | Retry Safe | Fallback Used |
|-----------|------------|-------------|------------|---------------|
| 400 | `INVALID_REQUEST` | Invalid content format, missing fields | No | No |
| 400 | `POLICY_NOT_FOUND` | Policy bundle not found for tenant | No | No |
| 401 | `UNAUTHORIZED` | Invalid or missing authentication | No | No |
| 403 | `INSUFFICIENT_RBAC` | Missing required RBAC scopes | No | No |
| 403 | `RATE_LIMITED` | Tenant rate limit exceeded | Yes | No |
| 206 | `PARTIAL_COVERAGE` | Coverage below threshold but partial results available | No | Yes |
| 408 | `REQUEST_TIMEOUT` | Global request SLA exceeded | Yes | Yes |
| 429 | `DETECTOR_OVERLOADED` | All detectors at capacity | Yes | Yes |
| 502 | `ALL_DETECTORS_UNAVAILABLE` | No healthy detectors for required set | Yes | Yes |
| 502 | `DETECTOR_COMMUNICATION_FAILED` | Network/protocol errors with detectors | Yes | Yes |
| 500 | `AGGREGATION_FAILED` | Response aggregation logic failed | No | Yes |
| 500 | `INTERNAL_ERROR` | Unexpected service error | No | Yes |

**IDEMPOTENCY BEHAVIOR:**
- All endpoints accept `Idempotency-Key` header (not just batch)
- Sync requests: 24-hour key retention, return cached response
- Async requests: Job ID becomes idempotency key
- Key format: UUID v4 or client-generated string (max 64 chars)

### Fallback Strategies

**Detector Unavailability:**
- Route to healthy alternatives
- Reduce coverage requirements if configured
- Return partial results with warnings
- Fall back to cached responses if available

**Timeout Handling:**
- Per-detector timeout configuration
- Graceful degradation for slow detectors
- Partial result aggregation
- Async job conversion for long-running requests

**Coverage Failures:**
- Attempt alternative detector combinations
- Apply tenant-specific fallback policies
- Return results with coverage warnings
- Trigger alerts for policy violations

## Testing Strategy

### Unit Testing

1. **Router Tests**
   - Policy application accuracy
   - Content type detection
   - Detector selection logic

2. **Coordinator Tests**
   - Parallel execution correctness
   - Error handling and retries
   - Timeout management

3. **Aggregator Tests**
   - Response combination accuracy
   - Conflict resolution
   - Coverage calculation

### Integration Testing

1. **End-to-End Orchestration**
   - Full request/response cycles
   - Multi-detector coordination
   - Mapper integration

2. **Health Monitoring**
   - Detector failure detection
   - Recovery handling
   - Circuit breaker behavior

3. **Policy Enforcement**
   - Tenant policy application
   - Coverage requirement validation
   - Conflict resolution strategies

### Performance Testing

1. **Load Testing**
   - Concurrent request handling
   - Detector coordination scalability
   - Response time targets (<2s p95)

2. **Fault Tolerance Testing**
   - Detector failure scenarios
   - Network partition handling
   - Recovery behavior validation

## Deployment Architecture

### Container Structure

```dockerfile
FROM python:3.11-slim

# Install orchestration service
COPY src/llama_mapper/orchestration/ /app/src/llama_mapper/orchestration/
COPY requirements-orchestration.txt /app/

# Configuration
ENV ORCHESTRATION_CONFIG_PATH=/app/config/orchestration.yaml
ENV DETECTOR_REGISTRY_PATH=/app/config/detectors.yaml
ENV POLICY_STORE_TYPE=file
ENV POLICY_STORE_PATH=/app/config/policies/

EXPOSE 8002
CMD ["uvicorn", "llama_mapper.orchestration.main:app", "--host", "0.0.0.0", "--port", "8002"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orchestration-service
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: orchestration
        image: llama-mapper-orchestration:latest
        ports:
        - containerPort: 8002
        env:
        - name: SERVICE_NAME
          value: "orchestration"
        - name: DETECTOR_REGISTRY_CONFIG
          valueFrom:
            configMapKeyRef:
              name: detector-registry
              key: detectors.yaml
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8002
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service Discovery

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: detector-registry
data:
  detectors.yaml: |
    detectors:
      deberta-toxicity:
        endpoint: "http://deberta-service:8080/analyze"
        auth_type: "api_key"
        timeout_ms: 3000
        capabilities:
          supported_content_types: ["text"]
          max_content_length: 10000
      
      llama-guard:
        endpoint: "http://llama-guard-service:8080/analyze"
        auth_type: "bearer_token"
        timeout_ms: 5000
        capabilities:
          supported_content_types: ["text", "code"]
          max_content_length: 50000
      
      openai-moderation:
        endpoint: "http://openai-proxy:8080/moderate"
        auth_type: "api_key"
        timeout_ms: 2000
        capabilities:
          supported_content_types: ["text", "image"]
          max_content_length: 25000
```

## Monitoring and Observability

### Metrics Collection

**OBSERVABILITY CONTRACT (LOCKED METRIC NAMES):**

```python
class OrchestrationMetricsCollector(MetricsCollector):
    """Metrics collector with guaranteed metric names for dashboards."""
    
    def record_orchestration_request(self, tenant_id: str, policy_bundle: str, detectors_count: int, processing_time: float):
        """orchestrate_requests_total{tenant, policy, status}"""
        self.increment_counter("orchestrate_requests_total", {
            "tenant": tenant_id,
            "policy": policy_bundle,
            "status": "success"
        })
        self.record_histogram("orchestrate_request_duration_ms", processing_time, {
            "tenant": tenant_id,
            "policy": policy_bundle
        })
        
    def record_detector_coordination(self, detector: str, success: bool, processing_time: float):
        """detector_latency_ms{detector, status}"""
        self.record_histogram("detector_latency_ms", processing_time, {
            "detector": detector,
            "status": "success" if success else "failed"
        })
        
    def record_coverage_achievement(self, tenant_id: str, policy_bundle: str, achieved_coverage: float):
        """coverage_achieved{tenant, policy}"""
        self.record_gauge("coverage_achieved", achieved_coverage, {
            "tenant": tenant_id,
            "policy": policy_bundle
        })
        
    def record_circuit_breaker_state(self, detector: str, state: CircuitBreakerState):
        """circuit_breaker_state{detector, state}"""
        self.record_gauge("circuit_breaker_state", 1.0, {
            "detector": detector,
            "state": state.value
        })
        
    def record_health_check(self, detector: str, is_healthy: bool, response_time: float):
        """detector_health_status{detector}, detector_health_check_duration_ms{detector}"""
        self.record_gauge("detector_health_status", 1.0 if is_healthy else 0.0, {
            "detector": detector
        })
        self.record_histogram("detector_health_check_duration_ms", response_time, {
            "detector": detector
        })
        
    def record_policy_enforcement(self, tenant_id: str, policy_bundle: str, enforced: bool, violation_type: str = None):
        """policy_enforcement_total{tenant, policy, status, violation_type}"""
        self.increment_counter("policy_enforcement_total", {
            "tenant": tenant_id,
            "policy": policy_bundle,
            "status": "enforced" if enforced else "violated",
            "violation_type": violation_type or "none"
        })
        
    def record_cache_operation(self, operation: str, hit: bool, tenant_id: str):
        """cache_operations_total{operation, result, tenant}"""
        self.increment_counter("cache_operations_total", {
            "operation": operation,  # get, set, invalidate
            "result": "hit" if hit else "miss",
            "tenant": tenant_id
        })
```

### Key Performance Indicators

1. **Orchestration Metrics**
   - Request processing time p95 (target: <2s)
   - Detector coordination success rate (target: >99%)
   - Coverage achievement rate (target: >95%)

2. **Detector Health Metrics**
   - Detector availability (target: >99.5%)
   - Health check response time (target: <500ms)
   - Circuit breaker activation rate (target: <1%)

3. **Policy Compliance Metrics**
   - Policy enforcement success rate (target: 100%)
   - Coverage requirement violations (target: <1%)
   - Conflict resolution accuracy (target: >95%)

### Alerting Strategy

1. **Critical Alerts**
   - All detectors unavailable for tenant
   - Coverage requirements not met
   - Service availability <99%

2. **Warning Alerts**
   - Individual detector unhealthy >5 minutes
   - Response time p95 >1.5s
   - Coverage achievement <90%

3. **Info Alerts**
   - New detector registered
   - Policy configuration updated
   - Daily orchestration summary