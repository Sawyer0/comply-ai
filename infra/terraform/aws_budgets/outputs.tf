output "budget_arn" {
  description = "ARN of the AWS Budget"
  value       = aws_budgets_budget.this.arn
}
