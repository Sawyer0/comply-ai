# Cost management and resource planning

This document defines cost guardrails, resource sizing guidance, switching between CPU↔GPU profiles and quantization, and budget alerting for the Llama Mapper service.

Overview
- Profiles: CPU (TGI) and GPU (vLLM) via Helm `.Values.profile`
- Resource guardrails: Kubernetes ResourceQuota and LimitRange (optional)
- Autoscaling: HPA on CPU utilization with min/max bounds
- Quantization: 8-bit and 4-bit options for model loading to reduce memory/compute
- Budgets: AWS Budgets with email/SNS alerts via Terraform
- Validation: E2E integration tests, fallback behavior tests, and performance runs across profiles

1) Instance sizes and autoscaling guardrails
- CPU/TGI (default)
  - Suitable for low/medium throughput, small cost footprint
  - Recommended starting values (adjust per load):
    - replicas: 1-3
    - requests: cpu=250m, memory=512Mi
    - limits:   cpu=1,    memory=1Gi
  - HPA: targetCPUUtilizationPercentage=70, minReplicas=1, maxReplicas=5
- GPU/vLLM
  - Use when latency/throughput requirements exceed CPU path
  - Node requirements: NVIDIA GPU nodes + device plugin
  - Recommended starting values:
    - requests: cpu=2, memory=8Gi, nvidia.com/gpu=1
    - limits:   cpu=4, memory=16Gi, nvidia.com/gpu=1
  - HPA as above; tune based on throughput goals

Helm values examples
- CPU profile (default):
  ```yaml
  profile: tgi
  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 5
    targetCPUUtilizationPercentage: 70
  resources:
    tgi:
      requests:
        cpu: "250m"
        memory: "512Mi"
      limits:
        cpu: "1"
        memory: "1Gi"
  ```
- GPU profile:
  ```yaml
  profile: vllm
  nodeSelector:
    nvidia.com/gpu.present: "true"
  resources:
    vllm:
      requests:
        cpu: "2"
        memory: "8Gi"
        nvidia.com/gpu: 1
      limits:
        cpu: "4"
        memory: "16Gi"
        nvidia.com/gpu: 1
  ```

2) Resource guardrails (namespace)
Enable optional namespace-level guardrails with ResourceQuota + LimitRange. These prevent runaway costs from excessive pod counts or oversized pods.

- Enable via Helm values:
  ```yaml
  costManagement:
    resourceQuota:
      enabled: true
      hard:
        requests.cpu: "2"
        requests.memory: "4Gi"
        limits.cpu: "4"
        limits.memory: "8Gi"
        pods: "10"
        services: "5"
        configmaps: "20"
        persistentvolumeclaims: "0"
    limitRange:
      enabled: true
      defaultContainer:
        requests:
          cpu: "250m"
          memory: "512Mi"
        limits:
          cpu: "1"
          memory: "1Gi"
  ```

3) Switching CPU↔GPU and quantization
- CPU↔GPU switching
  - Helm `.Values.profile`: `tgi` (CPU) or `vllm` (GPU)
  - The serving backend env `LLAMA_MAPPER_SERVING__BACKEND` follows the profile automatically
  - GPU profile requires GPU nodes and tolerations/nodeSelector as appropriate
- Quantization toggles
  - Purpose: reduce memory footprint and cost by using 8-bit or 4-bit weights when appropriate
  - Env var:
    - `LLAMA_MAPPER_MODEL__QUANTIZATION=8bit` or `4bit`
  - Notes:
    - Training pipeline (ModelLoader) supports 8-bit and 4-bit when enabled
    - For inference:
      - CPU/TGI: deploy or point to a quantized model endpoint
      - GPU/vLLM: use a quantized model artifact; advanced per-backend quantization flags may be needed (consult vLLM docs). This repo exposes the env for coordination and documentation; backend-specific wiring may require additional ops steps.

4) Cost monitoring and budget alerts
- AWS Budgets via Terraform (module included)
  - Path: `infra/terraform/aws_budgets`
  - Example usage:
    ```hcl
    module "llama_mapper_budget" {
      source          = "./infra/terraform/aws_budgets"
      budget_name     = "llama-mapper-monthly"
      limit_amount    = 500
      time_unit       = "MONTHLY"
      budget_type     = "COST"
      threshold_pct   = 80
      email_addresses = ["alerts@example.com"]
      # sns_topic_arn = var.sns_topic_arn
      tags = { Project = "llama-mapper", Env = var.env }
    }
    ```
  - Apply alongside your existing infra. Combine with tags to attribute spend to project/tenant.
- Kubernetes-level guardrails
  - Enable ResourceQuota/LimitRange (see section 2)
  - Use HPA to cap scaling (min/max replicas)

5) Validation: E2E tests and performance checks
- End-to-end (detector input → canonical output)
  - Already covered by integration tests (tests/integration/test_api_service.py) using schema validation and mapping logic
  - Batch and error-path tests included
- Fallback scenarios
  - Covered by tests/integration/test_confidence_and_fallback.py and quality-gates tests
- Performance across profiles/quantization
  - Use the performance testing suite (Locust/k6) introduced earlier to run smoke tests across profiles:
    - CPU/TGI profile: `--set profile=tgi`
    - GPU/vLLM profile: `--set profile=vllm`
    - Quantization: ensure model artifacts or backends are configured accordingly; set `LLAMA_MAPPER_MODEL__QUANTIZATION`
  - Validate SLOs (docs/slo.md): p95 ≤ 250ms CPU baseline, schema-valid ≥99.5%, fallback <10%

Quick commands
- Switch to CPU profile:
  ```bash
  helm upgrade --install mapper charts/llama-mapper \
    --namespace comply \
    --set profile=tgi
  ```
- Switch to GPU profile:
  ```bash
  helm upgrade --install mapper charts/llama-mapper \
    --namespace comply \
    --set profile=vllm \
    --set nodeSelector."nvidia.com/gpu.present"=true
  ```
- Enable guardrails:
  ```bash
  helm upgrade --install mapper charts/llama-mapper \
    --namespace comply \
    --set costManagement.resourceQuota.enabled=true \
    --set costManagement.limitRange.enabled=true
  ```

Notes
- Cost figures vary by region/provider; validate with your cloud pricing and FinOps tooling
- Start conservative; increase quotas/limits/HPA bounds with observed load
- Keep budgets slightly below expected monthly max to enable early warnings
