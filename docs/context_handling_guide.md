# Context Handling Guide for Compliance AI

## Overview

This guide explains how context is handled throughout the Compliance AI system and how to implement the enhanced context management system.

## Current Context Handling

### ✅ What's Working

#### 1. **Request-Level Context**
- **Tenant Context**: Isolated tenant data with access controls
- **Request Context**: Request ID, tenant ID, user ID, processing time
- **Mapping Context**: Request ID, mapping method, fallback reason

#### 2. **Service-Level Context**
- **MapperPayload**: Basic context with tenant_id, detector, output, metadata
- **AnalysisRequest**: Rich context with period, tenant, app, route, coverage metrics

#### 3. **Training Context**
- **Context Variation**: Business context prefixes in training examples
- **Structured Prompts**: Context included in instruction prompts

### ⚠️ Current Limitations

#### 1. **Limited Mapper Context**
```python
# Current MapperPayload is minimal
class MapperPayload(BaseModel):
    detector: str
    output: str  # Just "toxic|hate|pii_detected"
    metadata: Optional[HandoffMetadata] = None
    tenant_id: str
```

**Missing:**
- Business context (industry, compliance requirements)
- Application context (which app/route triggered detection)
- Policy context (specific compliance frameworks)
- Historical context (previous similar detections)

#### 2. **Basic Context Usage**
```python
# Context is just appended as string, not structured
if context:
    formatted_input = f"Context: {context}\n\nInput: {detector_output}"
```

#### 3. **No Context-Aware Caching**
- Cache keys only use detector output + basic context
- No consideration of tenant-specific or policy-specific variations

## Enhanced Context System

### 🚀 New Context Architecture

#### 1. **Rich Context Types**
```python
@dataclass
class EnhancedContext:
    business: BusinessContext      # Industry, compliance, risk tolerance
    application: ApplicationContext # App, route, environment
    policy: PolicyContext          # Frameworks, enforcement, audit
    historical: HistoricalContext  # Patterns, false positives, trends
    tenant: TenantContext          # Custom configs, contacts
    detector: DetectorContext      # Detector-specific info
```

#### 2. **Context-Aware Prompts**
```python
# Before: Basic context
prompt = f"Context: {context}\n\nInput: {detector_output}"

# After: Rich, structured context
prompt = f"""You are a compliance taxonomy mapper with rich context awareness.

CONTEXT SUMMARY:
Industry: healthcare, Environment: prod, Frameworks: HIPAA, GDPR

BUSINESS CONTEXT:
Industry: healthcare
Compliance Requirements: HIPAA, GDPR
Risk Tolerance: strict
Data Classification: confidential

POLICY CONTEXT:
Policy Bundle: healthcare
Applicable Frameworks: HIPAA, GDPR
Enforcement Level: strict
Apply strict compliance mapping with detailed taxonomy

DETECTOR OUTPUT:
{detector_output}

INSTRUCTIONS:
- Map to canonical taxonomy considering all context above
- Apply business-specific compliance requirements
- Consider historical patterns and false positive rates
"""
```

#### 3. **Context-Aware Caching**
```python
# Before: Simple cache key
cache_key = hash(detector_output + context)

# After: Context-sensitive cache key
cache_key = f"{detector_output}|{industry}|{policy_bundle}|{risk_tolerance}"
```

## Implementation Guide

### Step 1: Enhanced MapperPayload

Update the MapperPayload to include richer context:

```python
@dataclass
class EnhancedMapperPayload(BaseModel):
    # Existing fields
    detector: str
    output: str
    tenant_id: str
    
    # Enhanced context fields
    business_context: Optional[BusinessContext] = None
    application_context: Optional[ApplicationContext] = None
    policy_context: Optional[PolicyContext] = None
    historical_context: Optional[HistoricalContext] = None
    
    # Backward compatibility
    metadata: Optional[HandoffMetadata] = None
```

### Step 2: Context Manager Integration

Integrate the ContextManager into the mapping service:

```python
class EnhancedMappingService:
    def __init__(self):
        self.context_manager = ContextManager()
        self.prompt_builder = ContextAwarePromptBuilder(self.context_manager)
        self.context_cache = ContextAwareCache(self.base_cache)
    
    async def map_with_context(self, payload: MapperPayload) -> MappingResponse:
        # Build enhanced context
        enhanced_context = self.context_manager.build_mapper_context(
            payload.dict(),
            tenant_context=self.get_tenant_context(payload.tenant_id),
            historical_data=self.get_historical_data(payload.tenant_id, payload.detector)
        )
        
        # Check context-aware cache
        cached_result = self.context_cache.get(
            payload.output, 
            enhanced_context, 
            context_sensitivity="medium"
        )
        if cached_result:
            return cached_result
        
        # Build context-aware prompt
        prompt = self.prompt_builder.build_mapper_prompt(
            payload.output, 
            enhanced_context
        )
        
        # Generate with enhanced context
        result = await self.generate_mapping(prompt)
        
        # Cache with context awareness
        self.context_cache.put(
            payload.output, 
            enhanced_context, 
            result, 
            context_sensitivity="medium"
        )
        
        return result
```

### Step 3: Training Data Enhancement

Enhance training data with rich context:

```python
def create_context_aware_training_examples():
    """Create training examples with rich context."""
    
    examples = []
    
    # Healthcare context example
    healthcare_context = EnhancedContext(
        business=BusinessContext(
            industry="healthcare",
            compliance_requirements=["HIPAA", "GDPR"],
            risk_tolerance="strict",
            data_classification="confidential"
        ),
        application=ApplicationContext(
            app_name="patient-portal",
            route="/api/patient-data",
            environment="prod"
        ),
        policy=PolicyContext(
            policy_bundle="healthcare",
            applicable_frameworks=["HIPAA", "GDPR"],
            enforcement_level="strict"
        )
    )
    
    prompt = prompt_builder.build_mapper_prompt(
        "email address detected: patient@hospital.com",
        healthcare_context
    )
    
    examples.append({
        "instruction": prompt,
        "response": json.dumps({
            "taxonomy": ["PII.Contact.Email", "PII.Health.PatientData"],
            "scores": {
                "PII.Contact.Email": 0.95,
                "PII.Health.PatientData": 0.90
            },
            "confidence": 0.93,
            "context_notes": "Healthcare context requires HIPAA compliance"
        })
    })
    
    return examples
```

### Step 4: Context Configuration

Create tenant-specific context configurations:

```yaml
# config/tenant_contexts.yaml
tenants:
  healthcare_tenant:
    business_context:
      industry: "healthcare"
      compliance_requirements: ["HIPAA", "GDPR", "HITECH"]
      risk_tolerance: "strict"
      data_classification: "confidential"
      jurisdiction: ["US", "EU"]
    
    policy_context:
      policy_bundle: "healthcare"
      applicable_frameworks: ["HIPAA", "GDPR"]
      enforcement_level: "strict"
      audit_requirements: ["quarterly_audit", "incident_review"]
      reporting_obligations: ["breach_notification", "compliance_reporting"]
    
    custom_taxonomy:
      "PII.Health.PatientData": "Patient-specific health information"
      "PII.Health.MedicalRecord": "Medical record identifiers"
  
  financial_tenant:
    business_context:
      industry: "financial_services"
      compliance_requirements: ["SOX", "PCI-DSS", "GDPR"]
      risk_tolerance: "strict"
      data_classification: "restricted"
      jurisdiction: ["US", "EU"]
    
    policy_context:
      policy_bundle: "financial"
      applicable_frameworks: ["SOX", "PCI-DSS", "GDPR"]
      enforcement_level: "strict"
      audit_requirements: ["annual_audit", "quarterly_review"]
      reporting_obligations: ["regulatory_reporting", "incident_reporting"]
```

## Benefits of Enhanced Context

### 1. **Improved Accuracy**
- **Business-aware mapping**: Healthcare PII vs Financial PII handled differently
- **Policy-aware decisions**: Strict vs lenient enforcement based on context
- **Historical awareness**: Learn from past false positives and patterns

### 2. **Better Caching**
- **Context-sensitive cache keys**: Different results for different contexts
- **Higher hit rates**: More precise cache matching
- **Reduced latency**: Better cache utilization

### 3. **Enhanced Training**
- **Rich training examples**: Context-aware training data
- **Better generalization**: Models learn context-dependent patterns
- **Improved performance**: Higher accuracy on context-specific scenarios

### 4. **Compliance Benefits**
- **Framework-specific mapping**: HIPAA vs GDPR vs SOX considerations
- **Audit trail**: Rich context in provenance and notes
- **Risk-aware decisions**: Conservative vs aggressive based on risk tolerance

## Migration Strategy

### Phase 1: Backward Compatibility
- Keep existing MapperPayload structure
- Add optional enhanced context fields
- Gradual rollout to high-value tenants

### Phase 2: Enhanced Features
- Enable context-aware prompts for new tenants
- Implement context-aware caching
- Add historical context integration

### Phase 3: Full Migration
- Migrate all tenants to enhanced context
- Deprecate legacy context handling
- Optimize based on usage patterns

## Monitoring and Metrics

### Context Usage Metrics
- Context types used per tenant
- Context-aware cache hit rates
- Context-specific accuracy improvements

### Performance Metrics
- Latency impact of enhanced context
- Memory usage of context objects
- Training data quality improvements

### Business Metrics
- Compliance accuracy by industry
- False positive reduction by context
- Customer satisfaction improvements

## Conclusion

The enhanced context system provides:

1. **Rich, structured context** for both Mapper and Analyst models
2. **Context-aware prompts** that consider business, policy, and historical factors
3. **Intelligent caching** that respects context variations
4. **Improved accuracy** through context-dependent decision making
5. **Better compliance** with framework-specific requirements

This system transforms the Compliance AI platform from a basic detector-to-taxonomy mapper into an intelligent, context-aware compliance system that understands business requirements, regulatory frameworks, and historical patterns.
