# ⚡ Quick Demo Reference Card

## 🚀 Start Services
```bash
python start_all_services.py
```

## 🌐 Instant Demo URLs (Just Click!)

### 1. See Detection in Action
```
http://localhost:8000/api/v1/orchestrate/demo
```
Shows: How we detect PII, credentials, and security issues

### 2. See Risk Analysis
```
http://localhost:8001/api/v1/analyze/demo
```
Shows: Risk scoring and compliance recommendations

### 3. See Compliance Mapping
```
http://localhost:8002/api/v1/map/demo
```
Shows: Framework mappings (SOC2, ISO27001, HIPAA)

### 4. Explore Everything
```
http://localhost:8000/docs  # Orchestration
http://localhost:8001/docs  # Analysis
http://localhost:8002/docs  # Mapper
```

## 🎯 5-Minute Demo Script

### Step 1: Show the Problem (30 seconds)
"Security tools give inconsistent outputs. Hard to use for compliance."

### Step 2: Show Detection (1 minute)
Open: http://localhost:8000/api/v1/orchestrate/demo
"We detect PII, credentials, and security issues across multiple detectors."

### Step 3: Show Analysis (1 minute)
Open: http://localhost:8001/api/v1/analyze/demo
"We assess risk and map to compliance frameworks automatically."

### Step 4: Show Compliance (1 minute)
Open: http://localhost:8002/api/v1/map/demo
"We generate audit-ready evidence for SOC2, ISO27001, HIPAA."

### Step 5: Show Integration (1.5 minutes)
Open: http://localhost:8000/docs
"Simple REST APIs. Register your own detectors. Production-ready."

## 🎬 Sample Test Content

```json
{
  "content": "Hi Sarah, email me at sarah@company.com. My SSN is 123-45-6789 and the password is admin123!"
}
```

**Expected:** Detects email, SSN, password → High risk → SOC2 controls

## 💡 Key Selling Points

✅ **Privacy-First** - No raw content logged
✅ **Multi-Detector** - Orchestrates multiple security tools
✅ **Compliance-Ready** - SOC2, ISO27001, HIPAA mappings
✅ **Easy Integration** - REST APIs, SDKs available
✅ **Production-Ready** - Enterprise features built-in

## 🔥 Wow Moments

1. **Register Custom Detector** - Show extensibility
2. **RAG Query** - Ask compliance questions: http://localhost:8001/api/v1/rag/query?query=GDPR
3. **Taxonomy Validation** - Real-time label validation
4. **Health Monitoring** - Built-in observability

## 📊 Quick Stats to Mention

- **3 Microservices** working together
- **Multiple Detectors** orchestrated simultaneously
- **3+ Compliance Frameworks** supported
- **Sub-second** response times
- **Zero** raw content storage (privacy-first)

## 🎯 Perfect For

- **Sales Demos** - Show value in 5 minutes
- **Technical Evaluations** - Explore APIs and integration
- **Compliance Reviews** - Demonstrate audit readiness
- **Security Assessments** - Test detection accuracy

---

**Need more details?** See `DEMO_GUIDE.md` for the complete guide!
