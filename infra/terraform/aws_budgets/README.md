# AWS Budgets module for Llama Mapper cost guardrails

This Terraform module creates an AWS Cost Budget with alert notifications to email and/or SNS.

Usage example:

```hcl
module "llama_mapper_budget" {
  source = "./infra/terraform/aws_budgets"

  budget_name     = "llama-mapper-monthly"
  limit_amount    = 500   # USD per month
  time_unit       = "MONTHLY"
  budget_type     = "COST"
  threshold_pct   = 80    # alert at 80% of budget

  # One of the following notification targets (or both)
  email_addresses = ["alerts@example.com"]
  sns_topic_arn   = var.sns_topic_arn # optional

  tags = {
    Project = "llama-mapper"
    Env     = var.env
  }
}
```

Inputs:
- budget_name: Name of the budget (string)
- limit_amount: Monthly limit in USD (number)
- time_unit: Time unit (e.g., MONTHLY) (string)
- budget_type: Budget type (COST | USAGE | RI_UTILIZATION etc.) (string)
- threshold_pct: Percentage threshold for alerts (number)
- email_addresses: List of email recipients (list(string))
- sns_topic_arn: Optional SNS topic ARN for alerts (string)
- tags: Optional map of tags (map(string))

Outputs:
- budget_arn: ARN of the created budget
