"""
Report generator implementation for weekly evaluation reports.

This module provides concrete implementations of report generators
for creating evaluation reports in various formats.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from ...reporting.models import ReportData, ReportFormat
from ..domain.interfaces import IReportGenerator

logger = logging.getLogger(__name__)


class WeeklyEvaluationReportGenerator(IReportGenerator):
    """
    Report generator for weekly evaluation reports.

    Generates reports in PDF, CSV, and JSON formats with
    comprehensive quality metrics and analysis.
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize the report generator.

        Args:
            template_dir: Directory containing report templates
        """
        self.template_dir = template_dir
        logger.info("Initialized weekly evaluation report generator")

    def generate_report(
        self,
        report_data: ReportData,
        format_type: str,
        tenant_id: Optional[str] = None,
        requested_by: Optional[str] = None,
        report_type: Optional[str] = None,
    ) -> Union[bytes, str, Dict[str, Any]]:
        """
        Generate a report in the specified format.

        Args:
            report_data: Data to include in the report
            format_type: Output format (PDF, CSV, or JSON)
            tenant_id: Optional tenant ID for multi-tenancy
            requested_by: Optional user who requested the report
            report_type: Optional report type identifier

        Returns:
            Report content as bytes (PDF), string (CSV), or dict (JSON)

        Raises:
            ValueError: If format_type is not supported
            RuntimeError: If report generation fails
        """
        try:
            # Extract evaluation data from custom_data
            evaluation_data = report_data.custom_data
            quality_metrics = evaluation_data.get("quality_metrics", {})

            if format_type.upper() == "PDF":
                return self._generate_pdf_report(
                    quality_metrics, tenant_id, requested_by, report_type
                )
            elif format_type.upper() == "CSV":
                return self._generate_csv_report(
                    quality_metrics, tenant_id, requested_by, report_type
                )
            elif format_type.upper() == "JSON":
                return self._generate_json_report(
                    quality_metrics, tenant_id, requested_by, report_type
                )
            else:
                raise ValueError(f"Unsupported report format: {format_type}")

        except Exception as e:
            logger.error("Failed to generate %s report: %s", format_type, e)
            raise RuntimeError(f"Failed to generate report: {e}")

    def _generate_pdf_report(
        self,
        quality_metrics: Dict[str, Any],
        tenant_id: Optional[str],
        requested_by: Optional[str],
        report_type: Optional[str],
    ) -> bytes:
        """Generate PDF report."""
        try:
            # Create a simple PDF report using reportlab
            # Create PDF in memory
            import io

            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title = Paragraph("Weekly Quality Evaluation Report", styles["Title"])
            story.append(title)
            story.append(Spacer(1, 12))

            # Metadata
            metadata = [
                f"Tenant ID: {tenant_id or 'N/A'}",
                f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
                f"Requested by: {requested_by or 'System'}",
                f"Report type: {report_type or 'weekly_evaluation'}",
            ]

            for line in metadata:
                story.append(Paragraph(line, styles["Normal"]))
                story.append(Spacer(1, 6))

            # Quality metrics section
            story.append(Paragraph("Quality Metrics", styles["Heading2"]))
            story.append(Spacer(1, 12))

            metrics_data = [
                ("Total Examples", quality_metrics.get("total_examples", 0)),
                (
                    "Schema Validation Rate",
                    f"{quality_metrics.get('schema_valid_rate', 0):.2%}",
                ),
                ("Rubric Score", f"{quality_metrics.get('rubric_score', 0):.2f}"),
                (
                    "OPA Compilation Success Rate",
                    f"{quality_metrics.get('opa_compile_success_rate', 0):.2%}",
                ),
                (
                    "Evidence Accuracy",
                    f"{quality_metrics.get('evidence_accuracy', 0):.2f}",
                ),
            ]

            for metric_name, metric_value in metrics_data:
                story.append(
                    Paragraph(f"<b>{metric_name}:</b> {metric_value}", styles["Normal"])
                )
                story.append(Spacer(1, 6))

            # Individual scores section
            individual_scores = quality_metrics.get("individual_rubric_scores", [])
            if individual_scores:
                story.append(Paragraph("Individual Rubric Scores", styles["Heading3"]))
                story.append(Spacer(1, 12))

                avg_score = sum(individual_scores) / len(individual_scores)
                story.append(
                    Paragraph(f"Average Score: {avg_score:.2f}", styles["Normal"])
                )
                story.append(
                    Paragraph(
                        f"Min Score: {min(individual_scores):.2f}", styles["Normal"]
                    )
                )
                story.append(
                    Paragraph(
                        f"Max Score: {max(individual_scores):.2f}", styles["Normal"]
                    )
                )
                story.append(Spacer(1, 12))

            # Build PDF
            doc.build(story)
            buffer.seek(0)

            logger.info("Generated PDF report for tenant %s", tenant_id)
            return buffer.getvalue()

        except ImportError:
            logger.error("reportlab not available for PDF generation")
            # Fallback to simple text report
            return self._generate_text_report(
                quality_metrics, tenant_id, requested_by, report_type
            ).encode("utf-8")
        except Exception as e:
            logger.error("Failed to generate PDF report: %s", e)
            raise

    def _generate_csv_report(
        self,
        quality_metrics: Dict[str, Any],
        tenant_id: Optional[str],
        requested_by: Optional[str],
        report_type: Optional[str],
    ) -> str:
        """Generate CSV report."""
        try:
            import csv
            import io

            buffer = io.StringIO()
            writer = csv.writer(buffer)

            # Write header
            writer.writerow(
                [
                    "Metric",
                    "Value",
                    "Tenant ID",
                    "Generated At",
                    "Requested By",
                    "Report Type",
                ]
            )

            # Write metrics
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            metrics_data = [
                ("total_examples", quality_metrics.get("total_examples", 0)),
                ("schema_valid_rate", quality_metrics.get("schema_valid_rate", 0)),
                ("rubric_score", quality_metrics.get("rubric_score", 0)),
                (
                    "opa_compile_success_rate",
                    quality_metrics.get("opa_compile_success_rate", 0),
                ),
                ("evidence_accuracy", quality_metrics.get("evidence_accuracy", 0)),
            ]

            for metric_name, metric_value in metrics_data:
                writer.writerow(
                    [
                        metric_name,
                        metric_value,
                        tenant_id or "N/A",
                        timestamp,
                        requested_by or "System",
                        report_type or "weekly_evaluation",
                    ]
                )

            # Write individual scores
            individual_scores = quality_metrics.get("individual_rubric_scores", [])
            for i, score in enumerate(individual_scores):
                writer.writerow(
                    [
                        f"individual_rubric_score_{i}",
                        score,
                        tenant_id or "N/A",
                        timestamp,
                        requested_by or "System",
                        report_type or "weekly_evaluation",
                    ]
                )

            logger.info("Generated CSV report for tenant %s", tenant_id)
            return buffer.getvalue()

        except Exception as e:
            logger.error("Failed to generate CSV report: %s", e)
            raise

    def _generate_json_report(
        self,
        quality_metrics: Dict[str, Any],
        tenant_id: Optional[str],
        requested_by: Optional[str],
        report_type: Optional[str],
    ) -> Dict[str, Any]:
        """Generate JSON report."""
        try:
            report = {
                "metadata": {
                    "tenant_id": tenant_id,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "requested_by": requested_by,
                    "report_type": report_type or "weekly_evaluation",
                    "version": "1.0",
                },
                "quality_metrics": quality_metrics,
                "summary": {
                    "total_examples": quality_metrics.get("total_examples", 0),
                    "average_rubric_score": sum(
                        quality_metrics.get("individual_rubric_scores", [])
                    )
                    / max(len(quality_metrics.get("individual_rubric_scores", [])), 1),
                    "schema_validation_rate": quality_metrics.get(
                        "schema_valid_rate", 0
                    ),
                    "opa_compilation_success_rate": quality_metrics.get(
                        "opa_compile_success_rate", 0
                    ),
                    "evidence_accuracy": quality_metrics.get("evidence_accuracy", 0),
                },
                "alerts": self._generate_alerts(quality_metrics),
                "recommendations": self._generate_recommendations(quality_metrics),
            }

            logger.info("Generated JSON report for tenant %s", tenant_id)
            return report

        except Exception as e:
            logger.error("Failed to generate JSON report: %s", e)
            raise

    def _generate_text_report(
        self,
        quality_metrics: Dict[str, Any],
        tenant_id: Optional[str],
        requested_by: Optional[str],
        report_type: Optional[str],
    ) -> str:
        """Generate simple text report as fallback."""
        lines = [
            "Weekly Quality Evaluation Report",
            "=" * 40,
            "",
            f"Tenant ID: {tenant_id or 'N/A'}",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Requested by: {requested_by or 'System'}",
            f"Report type: {report_type or 'weekly_evaluation'}",
            "",
            "Quality Metrics:",
            "-" * 20,
            f"Total Examples: {quality_metrics.get('total_examples', 0)}",
            f"Schema Validation Rate: {quality_metrics.get('schema_valid_rate', 0):.2%}",
            f"Rubric Score: {quality_metrics.get('rubric_score', 0):.2f}",
            f"OPA Compilation Success Rate: {quality_metrics.get('opa_compile_success_rate', 0):.2%}",
            f"Evidence Accuracy: {quality_metrics.get('evidence_accuracy', 0):.2f}",
            "",
        ]

        # Add individual scores
        individual_scores = quality_metrics.get("individual_rubric_scores", [])
        if individual_scores:
            lines.extend(
                [
                    "Individual Rubric Scores:",
                    "-" * 30,
                    f"Average: {sum(individual_scores) / len(individual_scores):.2f}",
                    f"Min: {min(individual_scores):.2f}",
                    f"Max: {max(individual_scores):.2f}",
                    f"Count: {len(individual_scores)}",
                    "",
                ]
            )

        # Add alerts
        alerts = self._generate_alerts(quality_metrics)
        if alerts:
            lines.extend(
                [
                    "Alerts:",
                    "-" * 10,
                    *[f"- {alert}" for alert in alerts],
                    "",
                ]
            )

        return "\n".join(lines)

    def _generate_alerts(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """Generate alerts based on quality metrics."""
        alerts = []

        schema_rate = quality_metrics.get("schema_valid_rate", 0)
        if schema_rate < 0.98:
            alerts.append(
                f"Schema validation rate ({schema_rate:.2%}) below threshold (98%)"
            )

        rubric_score = quality_metrics.get("rubric_score", 0)
        if rubric_score < 0.8:
            alerts.append(f"Rubric score ({rubric_score:.2f}) below threshold (0.8)")

        opa_rate = quality_metrics.get("opa_compile_success_rate", 0)
        if opa_rate < 0.95:
            alerts.append(
                f"OPA compilation success rate ({opa_rate:.2%}) below threshold (95%)"
            )

        evidence_accuracy = quality_metrics.get("evidence_accuracy", 0)
        if evidence_accuracy < 0.85:
            alerts.append(
                f"Evidence accuracy ({evidence_accuracy:.2f}) below threshold (0.85)"
            )

        return alerts

    def _generate_recommendations(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality metrics."""
        recommendations = []

        schema_rate = quality_metrics.get("schema_valid_rate", 0)
        if schema_rate < 0.98:
            recommendations.append(
                "Review schema validation logic and improve data quality"
            )

        rubric_score = quality_metrics.get("rubric_score", 0)
        if rubric_score < 0.8:
            recommendations.append(
                "Consider retraining the model or adjusting evaluation criteria"
            )

        opa_rate = quality_metrics.get("opa_compile_success_rate", 0)
        if opa_rate < 0.95:
            recommendations.append(
                "Review OPA policy generation and fix compilation issues"
            )

        evidence_accuracy = quality_metrics.get("evidence_accuracy", 0)
        if evidence_accuracy < 0.85:
            recommendations.append(
                "Improve evidence collection and validation processes"
            )

        if not recommendations:
            recommendations.append("Quality metrics are within acceptable ranges")

        return recommendations
