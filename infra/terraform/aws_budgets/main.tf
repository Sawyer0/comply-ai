terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

resource "aws_budgets_budget" "this" {
  name              = var.budget_name
  budget_type       = var.budget_type
  limit_amount      = var.limit_amount
  limit_unit        = "USD"
  time_unit         = var.time_unit
  cost_types {
    include_credit             = true
    include_discount           = true
    include_other_subscription = true
    include_recurring          = true
    include_refund             = true
    include_subscription       = true
    include_support            = true
    include_taxes              = true
    include_upfront            = true
    use_amortized              = true
    use_blended                = false
  }

  dynamic "notification" {
    for_each = length(var.email_addresses) > 0 ? [1] : []
    content {
      comparison_operator        = "GREATER_THAN"
      notification_type          = "ACTUAL"
      threshold                  = var.threshold_pct
      threshold_type             = "PERCENTAGE"
      subscriber_email_addresses = var.email_addresses
    }
  }

  dynamic "notification" {
    for_each = var.sns_topic_arn != null && var.sns_topic_arn != "" ? [1] : []
    content {
      comparison_operator  = "GREATER_THAN"
      notification_type    = "ACTUAL"
      threshold            = var.threshold_pct
      threshold_type       = "PERCENTAGE"
      subscriber_sns_topic_arns = [var.sns_topic_arn]
    }
  }

  tags = var.tags
}
