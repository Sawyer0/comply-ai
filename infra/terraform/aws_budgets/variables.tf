variable "budget_name" {
  type        = string
  description = "Name of the AWS budget"
}

variable "limit_amount" {
  type        = number
  description = "Budget amount in USD"
}

variable "time_unit" {
  type        = string
  description = "Time unit for the budget (e.g., MONTHLY, QUARTERLY, ANNUALLY)"
  default     = "MONTHLY"
}

variable "budget_type" {
  type        = string
  description = "Budget type (COST, USAGE, RI_COVERAGE, RI_UTILIZATION, SAVINGS_PLANS_*)"
  default     = "COST"
}

variable "threshold_pct" {
  type        = number
  description = "Alert threshold percentage"
  default     = 80
}

variable "email_addresses" {
  type        = list(string)
  description = "List of email addresses to notify"
  default     = []
}

variable "sns_topic_arn" {
  type        = string
  description = "Optional SNS topic ARN for notifications"
  default     = null
}

variable "tags" {
  type        = map(string)
  description = "Optional tags for the budget resource"
  default     = {}
}
