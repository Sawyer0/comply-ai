# 🎯 Try It Yourself - Complete Workflow Guide

## 🚀 Test the Full Platform in 3 Steps

### Step 1: 🔧 Find Security Issues (Orchestration Service)
**URL:** http://localhost:8000/docs

1. **Start with the demo:** Try `/api/v1/orchestrate/demo` to see sample output
2. **Test your own content:** Use `/api/v1/orchestrate` with this sample:

```json
{
  "content": "Hi Sarah, my email is sarah.johnson@company.com and my SSN is 123-45-6789. The database password is admin123!",
  "detector_types": ["presidio", "deberta"]
}
```

**What you'll see:** Structured findings showing PII and security issues with confidence scores.

### Step 2: 📊 Understand the Risk (Analysis Service)  
**URL:** http://localhost:8001/docs

1. **Start with the demo:** Try `/api/v1/analyze/demo` to see risk assessment
2. **Analyze your findings:** Take the results from Step 1 and send them to `/api/v1/analyze`

**What you'll see:** Risk scores, compliance mappings, and actionable recommendations.

### Step 3: 🗺️ Generate Compliance Evidence (Mapper Service)
**URL:** http://localhost:8002/docs

1. **Start with the demo:** Try `/api/v1/map/demo` to see compliance mapping
2. **Map your findings:** Take detector outputs and send them to `/api/v1/map`
3. **Explore the taxonomy:** Check `/api/v1/taxonomy` to see how we categorize everything

**What you'll see:** Standardized compliance mappings with audit-ready evidence.

## 🧪 Quick Test Samples

### Sample Content to Try:
```
"Hello John, please send the report to john.doe@acme.com. 
The API key is sk-1234567890abcdef and my phone is (555) 123-4567.
Also, the admin password is SuperSecret123!"
```

### Expected Flow:
1. **Orchestration** finds: Email, API key, phone, password
2. **Analysis** says: "High risk - credentials exposed"  
3. **Mapper** shows: "SOC2 controls CC6.1, CC7.1 apply"

## 💡 Pro Tips

### For Sales/Demo Purposes:
- **Start with demos** - Show the output format first
- **Use realistic data** - Mix PII, credentials, and normal text
- **Show the progression** - Raw findings → Risk assessment → Compliance evidence

### For Technical Evaluation:
- **Test edge cases** - Empty content, malformed data
- **Check error handling** - See how the system responds to bad input
- **Explore all endpoints** - Each service has multiple capabilities

### For Compliance Teams:
- **Focus on frameworks** - Try different target_framework values
- **Check evidence quality** - See what documentation gets generated
- **Validate taxonomy** - Use `/api/v1/taxonomy/validate` to check labels

## 🎯 What Makes This Special

### Unlike Other Tools:
- **No setup required** - Just use the browser
- **Real responses** - Not just mock data
- **Complete workflow** - See the full compliance automation process
- **Immediate feedback** - Understand the value proposition instantly

### Perfect For:
- **Prospects evaluating the platform**
- **Developers planning integrations** 
- **Compliance teams understanding capabilities**
- **Anyone curious about security automation**

## 🚀 Ready to Integrate?

Once you've tested the APIs and seen the value:

1. **Check the OpenAPI specs** - Each service has complete documentation
2. **Review the data models** - Understand request/response formats  
3. **Test with your real data** - See how it works with your actual content
4. **Plan your integration** - Decide which endpoints you need

The APIs are designed to be simple, consistent, and powerful. What you see in the docs is exactly what you get in production! 🎉