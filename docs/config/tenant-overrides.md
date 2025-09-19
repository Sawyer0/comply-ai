# Tenant-Specific Configuration Overrides

This document describes how to manage configuration precedence for tenants and environments.

Precedence order (last wins):
1. Global (base) configuration
2. Tenant overrides
3. Environment overrides
4. Environment variables

Directory layout (relative to your base config path):
- config.yaml (or a custom path)
- tenants/
  - TENANT_ID.yaml
- environments/
  - development.yaml
  - staging.yaml
  - production.yaml

Example base config (config.yaml):
```yaml
model:
  name: meta-llama/Llama-2-7b-chat-hf
serving:
  port: 8000
```

Tenant override (tenants/acme.yaml):
```yaml
model:
  name: acme-custom-model
```

Environment override (environments/production.yaml):
```yaml
serving:
  port: 9001
```

Effective config for tenant=acme in production:
- model.name -> acme-custom-model (tenant override)
- serving.port -> 9001 (environment override)

Environment variables can override any of the above. For example:
- MAPPER_SERVING_PORT=7777 will set serving.port to 7777
- MAPPER_ENVIRONMENT=staging will set the active environment name

Usage
- Programmatically: ConfigManager(config_path, tenant_id="acme", environment="production")
- CLI: use your preferred runtime wrapper to pass tenant/environment context as needed

Validation
Run configuration linting to ensure taxonomy and mappings are valid:
- mapper validate-config

Notes
- Only metadata should be logged; avoid raw content in logs.
- Keep tenant overrides minimal—only override what’s necessary.
