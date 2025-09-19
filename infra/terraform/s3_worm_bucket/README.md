# S3 WORM Bucket (Terraform)

This module provisions an S3 bucket configured for immutable storage (Object Lock / WORM), versioning, and optional Glacier transitions.

Usage

```
module "worm_bucket" {
  source = "./infra/terraform/s3_worm_bucket"

  bucket_name              = "my-llama-mapper-archive"
  region                   = "us-east-1"
  retention_years          = 7
  object_lock_mode         = "GOVERNANCE" # or COMPLIANCE
  enable_glacier_transition = true
  glacier_transition_days   = 180

  # Optional SSE-KMS
  # kms_key_id = "arn:aws:kms:..."
}
```

Notes
- Object Lock must be enabled at bucket creation time. This module creates a new bucket with Object Lock enabled.
- Governance mode allows privileged deletions with additional permissions; Compliance mode does not.
- Ensure IAM policies grant the backup/restore jobs only the minimum required permissions.
