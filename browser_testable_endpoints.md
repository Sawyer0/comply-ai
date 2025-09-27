# 🌐 Browser-Testable Endpoints (GET Only)

All these endpoints can be tested directly in your browser by copying and pasting the URLs.

## 🔧 Detector Orchestration Service (Port 8000)

### Basic Endpoints
- **Health Check**: http://localhost:8000/health
- **Root**: http://localhost:8000/
- **List Detectors**: http://localhost:8000/api/v1/detectors
- **Service Stats**: http://localhost:8000/api/v1/stats

### Demo Endpoints (NEW!)
- **Demo Orchestration**: http://localhost:8000/api/v1/orchestrate/demo
- **Detector Health**: http://localhost:8000/api/v1/health/presidio

## 📊 Analysis Service (Port 8001)

### Basic Endpoints  
- **Health Check**: http://localhost:8001/health
- **Root**: http://localhost:8001/
- **Quality Metrics**: http://localhost:8001/api/v1/quality/metrics
- **RAG Query**: http://localhost:8001/api/v1/rag/query?query=GDPR%20compliance

### Demo Endpoints (NEW!)
- **Demo Analysis**: http://localhost:8001/api/v1/analyze/demo

## 🗺️ Mapper Service (Port 8002)

### Basic Endpoints
- **Health Check**: http://localhost:8002/health  
- **Root**: http://localhost:8002/
- **Taxonomy**: http://localhost:8002/api/v1/taxonomy
- **Frameworks**: http://localhost:8002/api/v1/frameworks
- **Model Status**: http://localhost:8002/api/v1/models/status

### Demo Endpoints (NEW!)
- **Demo Mapping**: http://localhost:8002/api/v1/map/demo
- **Validate Label**: http://localhost:8002/api/v1/taxonomy/validate?label=PII.Contact.Email

## 🎯 What Each Demo Shows

### **Orchestration Demo** (`/api/v1/orchestrate/demo`)
Shows how the orchestration service would:
- Process input content: "John Doe's email is john.doe@example.com and SSN is 123-45-6789"
- Run multiple detectors (presidio, deberta)
- Return structured findings with confidence scores
- Track processing time and status

### **Analysis Demo** (`/api/v1/analyze/demo`)  
Shows how the analysis service would:
- Take detector results as input
- Perform risk assessment (low/medium/high)
- Generate compliance mappings for SOC2, ISO27001, HIPAA
- Provide actionable recommendations

### **Mapping Demo** (`/api/v1/map/demo`)
Shows how the mapper service would:
- Map detector outputs to canonical taxonomy (PII.Contact.Email, etc.)
- Generate framework-specific compliance mappings
- Provide evidence requirements for audits
- Calculate overall confidence scores

## 🔄 Complete Workflow Demo

1. **Start**: http://localhost:8000/api/v1/orchestrate/demo
2. **Analyze**: http://localhost:8001/api/v1/analyze/demo  
3. **Map**: http://localhost:8002/api/v1/map/demo
4. **Validate**: http://localhost:8002/api/v1/taxonomy/validate?label=PII.Contact.Email

This demonstrates the complete microservice workflow from detection → analysis → mapping → validation!

## 💡 Pro Tips

- **Bookmark these URLs** for quick testing
- **Check the JSON responses** to understand the data flow
- **Compare demo responses** to see how data flows between services
- **Use browser dev tools** to see the actual HTTP requests/responses

All these endpoints return JSON that you can view directly in your browser! 🎉