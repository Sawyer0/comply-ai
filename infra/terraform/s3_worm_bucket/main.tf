terraform {
  required_version = ">= 1.3.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

resource "aws_s3_bucket" "this" {
  bucket              = var.bucket_name
  object_lock_enabled = true
  force_destroy       = false
}

resource "aws_s3_bucket_versioning" "this" {
  bucket = aws_s3_bucket.this.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Default encryption (SSE-S3 or SSE-KMS)
resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  bucket = aws_s3_bucket.this.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.kms_key_id == null ? "AES256" : "aws:kms"
      kms_master_key_id = var.kms_key_id
    }
  }
}

# Object Lock (WORM) default retention
resource "aws_s3_bucket_object_lock_configuration" "this" {
  bucket = aws_s3_bucket.this.id
  rule {
    default_retention {
      mode  = var.object_lock_mode
      years = var.retention_years
    }
  }
}

# Optional lifecycle transitions (example: move to Glacier after 180 days)
resource "aws_s3_bucket_lifecycle_configuration" "this" {
  bucket = aws_s3_bucket.this.id

  rule {
    id     = "transition-to-glacier"
    status = var.enable_glacier_transition ? "Enabled" : "Disabled"

    transition {
      days          = var.glacier_transition_days
      storage_class = "GLACIER"
    }
    noncurrent_version_transition {
      noncurrent_days = var.glacier_transition_days
      storage_class   = "GLACIER"
    }
  }
}

output "bucket_arn" {
  value = aws_s3_bucket.this.arn
}
output "bucket_name" {
  value = aws_s3_bucket.this.bucket
}
