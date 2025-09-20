variable "bucket_name" {
  description = "Name of the S3 bucket"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "kms_key_id" {
  description = "Optional KMS key ID for SSE-KMS encryption"
  type        = string
  default     = null
}

variable "retention_years" {
  description = "Default WORM retention period in years"
  type        = number
  default     = 7
}

variable "object_lock_mode" {
  description = "Object Lock mode"
  type        = string
  default     = "GOVERNANCE"
  validation {
    condition     = contains(["GOVERNANCE", "COMPLIANCE"], var.object_lock_mode)
    error_message = "object_lock_mode must be GOVERNANCE or COMPLIANCE"
  }
}

variable "enable_glacier_transition" {
  description = "Enable lifecycle transition to Glacier"
  type        = bool
  default     = false
}

variable "glacier_transition_days" {
  description = "Days until transition to Glacier"
  type        = number
  default     = 180
}
