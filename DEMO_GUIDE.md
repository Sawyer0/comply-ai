# 🎯 Llama Mapper - Complete Demo Guide

## 🚀 Quick Start (5 Minutes)

### Step 1: Start All Services
```bash
python start_all_services.py
```

This starts three microservices:
- **Detector Orchestration** (Port 8000) - Coordinates security detectors
- **Analysis Service** (Port 8001) - Risk assessment and compliance analysis
- **Mapper Service** (Port 8002) - Compliance mapping and taxonomy normalization

### Step 2: Open Interactive Documentation
Open these URLs in your browser:
- http://localhost:8000/docs - Orchestration Service
- http://localhost:8001/docs - Analysis Service
- http://localhost:8002/docs - Mapper Service

---

## 🌐 Browser-Only Demo (No Code Required!)

### Option 1: Quick Demo Endpoints (Just Click!)

Copy these URLs into your browser to see instant results:

**1. Orchestration Demo** - See how we detect security issues:
```
http://localhost:8000/api/v1/orchestrate/demo
```

**2. Analysis Demo** - See risk assessment and recommendations:
```
http://localhost:8001/api/v1/analyze/demo
```

**3. Mapping Demo** - See compliance framework mappings:
```
http://localhost:8002/api/v1/map/demo
```

**4. Explore Taxonomy** - See how we categorize everything:
```
http://localhost:8002/api/v1/taxonomy
```

**5. Check Detector Status** - See available security detectors:
```
http://localhost:8000/api/v1/detectors
```

### Option 2: Interactive Swagger UI

1. Go to http://localhost:8000/docs
2. Find the "🧪 Interactive Demo" section
3. Click "Try it out" on any endpoint
4. Click "Execute" to see results

---

## 🎬 Complete Workflow Demo

### Scenario: Detecting PII and Security Issues

**Sample Content:**
```
Hi Sarah, my email is sarah.johnson@company.com and my SSN is 123-45-6789. 
The database password is admin123! and the API key is sk-1234567890abcdef.
Please call me at (555) 123-4567.
```

### Step-by-Step Workflow:

#### 1️⃣ Detect Security Issues (Orchestration)
**Endpoint:** `POST http://localhost:8000/api/v1/orchestrate`

**Request:**
```json
{
  "content": "Hi Sarah, my email is sarah.johnson@company.com and my SSN is 123-45-6789. The database password is admin123!",
  "detector_types": ["presidio", "deberta"],
  "processing_mode": "standard"
}
```

**What You'll See:**
- Email addresses detected
- SSN detected
- Password/credentials detected
- Confidence scores for each finding
- Processing time and metadata

#### 2️⃣ Analyze Risk (Analysis Service)
**Endpoint:** `POST http://localhost:8001/api/v1/analyze`

**Request:** (Use the results from Step 1)
```json
{
  "findings": [
    {
      "type": "PII.Contact.Email",
      "confidence": 0.95,
      "text": "sarah.johnson@company.com"
    },
    {
      "type": "PII.Identification.SSN",
      "confidence": 0.98,
      "text": "123-45-6789"
    }
  ],
  "target_framework": "SOC2"
}
```

**What You'll See:**
- Overall risk score (Low/Medium/High)
- Compliance framework mappings
- Specific control requirements
- Actionable recommendations
- Remediation guidance

#### 3️⃣ Generate Compliance Evidence (Mapper)
**Endpoint:** `POST http://localhost:8002/api/v1/map`

**Request:**
```json
{
  "detector": "presidio",
  "output": "email|ssn|password",
  "tenant_id": "demo-tenant",
  "metadata": {
    "contributing_detectors": ["presidio", "deberta"],
    "aggregation_method": "weighted_average"
  }
}
```

**What You'll See:**
- Canonical taxonomy labels
- Framework-specific mappings (SOC2, ISO27001, HIPAA)
- Evidence requirements for audits
- Confidence scores and metadata

---

## 🧪 Testing Different Scenarios

### Scenario 1: PII Detection
**Content:** `"Contact John at john.doe@example.com or call (555) 123-4567"`

**Expected Results:**
- Email detection
- Phone number detection
- Low-to-medium risk assessment
- GDPR/CCPA compliance mappings

### Scenario 2: Credential Exposure
**Content:** `"The API key is sk-1234567890 and password is SuperSecret123!"`

**Expected Results:**
- API key detection
- Password detection
- HIGH risk assessment
- SOC2 CC6.1 control mapping
- Immediate remediation recommendations

### Scenario 3: Mixed Content
**Content:** `"Normal business email about project deadlines and meeting schedules"`

**Expected Results:**
- No security findings
- Low risk assessment
- No compliance issues
- Clean bill of health

---

## 🎯 Demo Features to Highlight

### 1. **Privacy-First Architecture**
- No raw content is logged
- Only metadata and confidence scores stored
- Compliant with GDPR, CCPA, HIPAA

### 2. **Multi-Detector Orchestration**
- Runs multiple detectors simultaneously
- Aggregates results intelligently
- Handles detector failures gracefully

### 3. **Intelligent Risk Assessment**
- Context-aware risk scoring
- Framework-specific compliance mapping
- Actionable remediation guidance

### 4. **Audit-Ready Evidence**
- Standardized taxonomy
- Framework mappings (SOC2, ISO27001, HIPAA)
- Complete audit trail
- Versioned configurations

### 5. **Enterprise Features**
- Multi-tenant isolation
- Rate limiting and authentication
- Distributed tracing with correlation IDs
- Comprehensive monitoring and metrics

---

## 🔧 Advanced Demo Features

### Register Your Own Detector
**Endpoint:** `POST http://localhost:8000/api/v1/detectors/register`

```json
{
  "name": "My Custom PII Scanner",
  "type": "pii",
  "version": "1.0.0",
  "endpoint": "https://my-detector.company.com/api/v1/detect",
  "capabilities": ["email", "phone", "ssn"],
  "confidence_threshold": 0.7
}
```

### Query Compliance Knowledge (RAG)
**Endpoint:** `GET http://localhost:8001/api/v1/rag/query?query=GDPR%20compliance`

### Validate Taxonomy Labels
**Endpoint:** `GET http://localhost:8002/api/v1/taxonomy/validate?label=PII.Contact.Email`

---

## 📊 Monitoring and Observability

### Health Checks
- http://localhost:8000/health
- http://localhost:8001/health
- http://localhost:8002/health

### Service Statistics
- http://localhost:8000/api/v1/stats
- http://localhost:8001/api/v1/quality/metrics

### Model Status
- http://localhost:8002/api/v1/models/status

---

## 💡 Demo Tips

### For Sales/Business Demos:
1. **Start with the problem** - Show messy, inconsistent detector outputs
2. **Demonstrate the solution** - Show clean, standardized results
3. **Highlight compliance** - Show framework mappings and audit evidence
4. **Emphasize privacy** - Show metadata-only logging
5. **Show scalability** - Demonstrate multi-detector orchestration

### For Technical Demos:
1. **Show the architecture** - Three microservices working together
2. **Demonstrate APIs** - RESTful, well-documented, easy to integrate
3. **Test error handling** - Show graceful degradation
4. **Highlight extensibility** - Register custom detectors
5. **Show monitoring** - Health checks, metrics, tracing

### For Compliance Demos:
1. **Focus on frameworks** - SOC2, ISO27001, HIPAA mappings
2. **Show audit trails** - Complete request tracking
3. **Demonstrate evidence** - Audit-ready documentation
4. **Highlight privacy** - No raw content storage
5. **Show versioning** - Taxonomy and model versioning

---

## 🚀 Next Steps After Demo

### For Developers:
1. Review OpenAPI specifications at `/docs`
2. Check out client SDK examples in `docs/clients/`
3. Explore the codebase structure
4. Run the test suite: `pytest tests/`

### For Integration:
1. Get API keys configured
2. Set up tenant isolation
3. Configure rate limiting
4. Integrate with your security tools

### For Production:
1. Review deployment guides in `docs/deployment/`
2. Set up monitoring and alerting
3. Configure secrets management (Vault)
4. Enable Redis for caching and rate limiting

---

## 🎉 What Makes This Special

### Unlike Other Tools:
✅ **No setup required** - Just start and demo
✅ **Real functionality** - Not just mock data
✅ **Complete workflow** - Detection → Analysis → Compliance
✅ **Browser-testable** - No code required for basic demo
✅ **Production-ready** - Same APIs you'll use in production

### Perfect For:
- **Prospects** evaluating the platform
- **Developers** planning integrations
- **Compliance teams** understanding capabilities
- **Security teams** testing detection accuracy
- **Anyone** curious about AI safety and compliance automation

---

## 📞 Support

- **Documentation**: Check the `/docs` endpoint on each service
- **Issues**: Review error messages in the console
- **Questions**: See the comprehensive API documentation

**Happy Demoing! 🎯**
