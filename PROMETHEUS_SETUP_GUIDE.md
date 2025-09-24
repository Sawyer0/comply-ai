# 📊 Prometheus Monitoring Setup for Comply-AI Analysis Module

## 🚀 **Quick Start with Docker Compose**

### **1. Start the Monitoring Stack**
```bash
# Start everything (Analysis API + Prometheus + Grafana + AlertManager)
docker-compose -f docker-compose.prometheus.yml up -d

# Or start just Prometheus to scrape your existing API
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v ./config/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus:v2.45.0
```

### **2. Access the Dashboards**
- **Analysis API**: http://localhost:8001
- **Prometheus**: http://localhost:9090  
- **Grafana**: http://localhost:3000 (admin/admin)
- **AlertManager**: http://localhost:9093

---

## 📈 **Metrics Being Collected**

### **Request Metrics**
```
analysis_requests_total{endpoint, status, tenant}          # Request counter
analysis_request_duration_seconds{endpoint, analysis_type} # Response time histogram
```

### **Business Metrics**
```
analysis_confidence_score{analysis_type, env}              # Confidence distribution
coverage_gap_rate{tenant, env}                            # Coverage gap detection rate
analysis_active_tenants                                   # Number of active tenants
```

### **Error Metrics**
```
analysis_errors_total{error_type, endpoint}               # Error counter
analysis_low_confidence_total{tenant, threshold}          # Low confidence alerts
```

### **Performance Metrics**
```
opa_policy_generation_duration_seconds{policy_type}       # OPA policy generation time
```

---

## 🔍 **Key Prometheus Queries**

### **SLO Monitoring**
```promql
# 95th percentile response time
histogram_quantile(0.95, rate(analysis_request_duration_seconds_bucket[5m]))

# Error rate
rate(analysis_errors_total[5m]) / rate(analysis_requests_total[5m])

# Coverage gap rate by tenant
avg(coverage_gap_rate) by (tenant)

# Low confidence rate
rate(analysis_low_confidence_total[5m])
```

### **Business Intelligence**
```promql
# Requests per minute by tenant
rate(analysis_requests_total[1m]) * 60

# Confidence score distribution
histogram_quantile(0.50, rate(analysis_confidence_score_bucket[5m]))

# Active tenant growth
increase(analysis_active_tenants[1h])
```

---

## 🚨 **Alerting Rules Configured**

### **Critical Alerts**
- **Service Down**: Analysis module unreachable
- **High Coverage Gap Rate**: >30% coverage gaps detected
- **High Error Rate**: >10% error rate for 2+ minutes

### **Warning Alerts**  
- **Slow Response Time**: 95th percentile >2 seconds
- **Low Confidence Rate**: >50% low confidence results
- **OPA Policy Generation Slow**: >1 second generation time

### **Capacity Alerts**
- **Request Rate Increasing**: Trend analysis for capacity planning
- **Active Tenant Growth**: Rapid tenant onboarding detection

---

## 📊 **Grafana Dashboard Queries**

### **Executive Summary Panel**
```promql
# Total Requests Today
increase(analysis_requests_total[24h])

# Average Confidence Score
rate(analysis_confidence_score_sum[1h]) / rate(analysis_confidence_score_count[1h])

# Coverage Gaps Detected
sum(increase(coverage_gap_rate[24h])) by (tenant)

# System Uptime
up{job="analysis-module"}
```

### **Performance Dashboard**
```promql
# Response Time Trends
histogram_quantile(0.95, rate(analysis_request_duration_seconds_bucket[5m]))
histogram_quantile(0.50, rate(analysis_request_duration_seconds_bucket[5m]))

# Request Volume
rate(analysis_requests_total[1m]) * 60

# Error Rate
rate(analysis_errors_total[5m]) / rate(analysis_requests_total[5m]) * 100
```

---

## 🔧 **Prometheus Configuration**

### **Scrape Configuration** (`config/prometheus.yml`)
```yaml
scrape_configs:
  - job_name: 'analysis-module'
    static_configs:
      - targets: ['localhost:8001']
    scrape_interval: 10s
    metrics_path: '/metrics'
```

### **Kubernetes Service Discovery** (for production)
```yaml
scrape_configs:
  - job_name: 'analysis-module-k8s'
    kubernetes_sd_configs:
      - role: service
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

---

## 🎯 **Demo Metrics for Investors**

### **Real-Time Business Metrics**
1. **Compliance Intelligence**: Coverage gaps detected per hour
2. **AI Performance**: Confidence score distribution
3. **Scale Metrics**: Active tenants and request volume
4. **Reliability**: Uptime and error rates

### **Sample Investor Queries**
```bash
# Show real-time coverage gap detection
curl "http://localhost:9090/api/v1/query?query=coverage_gap_rate"

# Show AI confidence scores
curl "http://localhost:9090/api/v1/query?query=rate(analysis_confidence_score_sum[1h])/rate(analysis_confidence_score_count[1h])"

# Show request volume trends
curl "http://localhost:9090/api/v1/query?query=rate(analysis_requests_total[5m])*60"
```

---

## 🚀 **Production Considerations**

### **Scaling Prometheus**
- Use federation for multi-region deployments
- Configure retention policies (default: 200h)
- Set up remote storage (Thanos, Cortex)

### **Security**
- Enable authentication for Prometheus/Grafana
- Use TLS for metric scraping
- Network segmentation for monitoring stack

### **High Availability**
- Run multiple Prometheus instances
- Use AlertManager clustering
- Backup Grafana dashboards and configs

---

## 📝 **Quick Test Commands**

```bash
# Check if metrics endpoint is working
curl http://localhost:8001/metrics

# Generate test data
for i in {1..10}; do
  curl -X POST "http://localhost:8001/api/v1/analysis/analyze" \
    -H "Content-Type: application/json" \
    -d @examples/sample_metrics.json
done

# Query metrics in Prometheus
curl "http://localhost:9090/api/v1/query?query=analysis_requests_total"

# Check alerts
curl http://localhost:9090/api/v1/alerts
```

---

## 🎪 **Demo Script Integration**

**For live demos, show:**
1. **Metrics Dashboard**: Real-time request volume and confidence scores
2. **Alert Simulation**: Trigger a coverage gap alert
3. **Business Intelligence**: Show tenant growth and compliance trends
4. **SLO Compliance**: Demonstrate sub-second response times

**This monitoring setup provides enterprise-grade observability for your compliance intelligence platform!** 🚀
