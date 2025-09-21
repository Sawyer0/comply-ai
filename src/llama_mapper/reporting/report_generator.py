"""
Report generator for multiple output formats.

Generates PDF reports via WeasyPrint with embedded version tags,
CSV exports via Pandas with version metadata, and JSON API responses
with version headers as specified in requirements 9.2 and 9.5.
"""

import json
import logging
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast
from uuid import uuid4

import pandas as pd  # type: ignore[import-not-found,import-untyped]
from jinja2 import Environment, FileSystemLoader

from .models import (
    ComplianceReport,
    CoverageReport,
    ReportData,
    ReportFormat,
    ReportMetadata,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates reports in multiple formats with embedded version information.

    Supports PDF (via WeasyPrint), CSV (via Pandas), and JSON formats
    with comprehensive version tracking and metadata embedding.
    """

    def __init__(
        self,
        template_dir: Optional[Union[str, Path]] = None,
        taxonomy_version: str = "2025.09",
        model_version: str = "mapper-lora@v1.0.0",
        frameworks_version: str = "v1.0",
    ):
        """
        Initialize ReportGenerator.

        Args:
            template_dir: Directory containing report templates
            taxonomy_version: Current taxonomy version
            model_version: Current model version
            frameworks_version: Current frameworks version
        """
        self.template_dir = (
            Path(template_dir) if template_dir else Path(__file__).parent / "templates"
        )
        self.taxonomy_version = taxonomy_version
        self.model_version = model_version
        self.frameworks_version = frameworks_version

        # Initialize Jinja2 environment for templates
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)), autoescape=True
        )

        # Add custom filters
        self.jinja_env.filters["datetime_format"] = self._datetime_format
        self.jinja_env.filters["percentage"] = self._percentage_format

        logger.info(
            f"ReportGenerator initialized with template_dir: {self.template_dir}"
        )

    def generate_report(
        self,
        report_data: ReportData,
        format_type: ReportFormat,
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
        """
        # Create report metadata
        metadata = ReportMetadata(
            report_id=str(uuid4()),
            generated_at=datetime.utcnow(),
            taxonomy_version=self.taxonomy_version,
            model_version=self.model_version,
            frameworks_version=self.frameworks_version,
            tenant_id=tenant_id,
            requested_by=requested_by,
            report_type=report_type,
        )

        logger.info(
            f"Generating {format_type.value} report",
            extra={
                "report_id": metadata.report_id,
                "tenant_id": tenant_id,
            },
        )

        try:
            if format_type == ReportFormat.PDF:
                return self._generate_pdf_report(report_data, metadata)
            elif format_type == ReportFormat.CSV:
                return self._generate_csv_report(report_data, metadata)
            elif format_type == ReportFormat.JSON:
                return self._generate_json_report(report_data, metadata)
            else:
                raise ValueError(f"Unsupported report format: {format_type}")

        except Exception as e:
            logger.error(
                f"Failed to generate {format_type.value} report",
                extra={
                    "report_id": metadata.report_id,
                    "error": str(e),
                },
            )
            raise

    def _generate_pdf_report(
        self, report_data: ReportData, metadata: ReportMetadata
    ) -> bytes:
        """Generate PDF report using WeasyPrint with embedded version tags."""
        try:
            # Import WeasyPrint (optional dependency)
            from weasyprint import CSS, HTML  # type: ignore[import-not-found]
            from weasyprint.text.fonts import (
                FontConfiguration,  # type: ignore[import-not-found]
            )
        except ImportError:
            raise ImportError(
                "WeasyPrint is required for PDF generation. Install with: pip install weasyprint"
            )

        # Determine template based on report data
        template_name = self._select_pdf_template(report_data)
        template = self.jinja_env.get_template(template_name)

        # Prepare template context
        context = {
            "metadata": metadata,
            "report_data": report_data,
            "generated_at": metadata.generated_at,
            "version_info": {
                "taxonomy": metadata.taxonomy_version,
                "model": metadata.model_version,
                "frameworks": metadata.frameworks_version,
            },
        }

        # Render HTML
        html_content = template.render(**context)

        # Generate PDF with embedded metadata
        font_config = FontConfiguration()
        html_doc = HTML(string=html_content)

        # Add CSS styling
        css_path = self.template_dir / "styles" / "report.css"
        css_content = None
        if css_path.exists():
            css_content = CSS(filename=str(css_path), font_config=font_config)

        # Generate PDF
        pdf_bytes = cast(
            bytes,
            html_doc.write_pdf(
                stylesheets=[css_content] if css_content else None,
                font_config=font_config,
            ),
        )

        logger.info(
            "PDF report generated successfully",
            extra={
                "report_id": metadata.report_id,
                "size_bytes": len(pdf_bytes),
            },
        )

        return pdf_bytes

    def _generate_csv_report(
        self, report_data: ReportData, metadata: ReportMetadata
    ) -> str:
        """Generate CSV report using Pandas with version metadata."""
        csv_buffer = StringIO()

        # Write metadata header as comments
        csv_buffer.write(f"# Report ID: {metadata.report_id}\n")
        csv_buffer.write(f"# Generated At: {metadata.generated_at.isoformat()}\n")
        csv_buffer.write(f"# Taxonomy Version: {metadata.taxonomy_version}\n")
        csv_buffer.write(f"# Model Version: {metadata.model_version}\n")
        csv_buffer.write(f"# Frameworks Version: {metadata.frameworks_version}\n")
        if metadata.tenant_id:
            csv_buffer.write(f"# Tenant ID: {metadata.tenant_id}\n")
        if metadata.requested_by:
            csv_buffer.write(f"# Requested By: {metadata.requested_by}\n")
        csv_buffer.write("#\n")

        # Generate CSV content based on report data type
        if report_data.has_compliance_data():
            assert report_data.compliance_report is not None
            self._write_compliance_csv(csv_buffer, report_data.compliance_report)
        elif report_data.has_coverage_data():
            assert report_data.coverage_report is not None
            self._write_coverage_csv(csv_buffer, report_data.coverage_report)
        else:
            # Generic CSV for custom data
            self._write_generic_csv(csv_buffer, report_data.custom_data)

        csv_content = csv_buffer.getvalue()
        csv_buffer.close()

        logger.info(
            "CSV report generated successfully",
            extra={
                "report_id": metadata.report_id,
                "size_chars": len(csv_content),
            },
        )

        return csv_content

    def _generate_json_report(
        self, report_data: ReportData, metadata: ReportMetadata
    ) -> Dict[str, Any]:
        """Generate JSON report with version headers."""
        json_report: Dict[str, Any] = {"metadata": metadata.to_dict(), "data": {}}

        # Add report data based on type
        if report_data.has_compliance_data():
            assert report_data.compliance_report is not None
            json_report["data"]["compliance"] = report_data.compliance_report.to_dict()

        if report_data.has_coverage_data():
            assert report_data.coverage_report is not None
            json_report["data"]["coverage"] = report_data.coverage_report.to_dict()

        # Add any custom data
        if report_data.custom_data:
            json_report["data"]["custom"] = report_data.custom_data

        logger.info(
            "JSON report generated successfully",
            extra={"report_id": metadata.report_id},
        )

        return json_report

    def _select_pdf_template(self, report_data: ReportData) -> str:
        """Select appropriate PDF template based on report data."""
        if report_data.has_compliance_data():
            return "compliance_report.html"
        elif report_data.has_coverage_data():
            return "coverage_report.html"
        else:
            return "generic_report.html"

    def _write_compliance_csv(
        self, buffer: StringIO, compliance_report: ComplianceReport
    ) -> None:
        """Write compliance report data to CSV buffer."""
        # Audit records section
        buffer.write("# AUDIT RECORDS\n")
        if compliance_report.audit_records:
            audit_df = pd.DataFrame(
                [record.to_dict() for record in compliance_report.audit_records]
            )
            audit_df.to_csv(buffer, index=False)
        else:
            buffer.write("No audit records available\n")

        buffer.write("\n# CONTROL MAPPINGS\n")
        if compliance_report.control_mappings:
            mappings_df = pd.DataFrame(
                [mapping.to_dict() for mapping in compliance_report.control_mappings]
            )
            mappings_df.to_csv(buffer, index=False)
        else:
            buffer.write("No control mappings available\n")

        buffer.write("\n# LINEAGE RECORDS\n")
        if compliance_report.lineage_records:
            lineage_df = pd.DataFrame(
                [record.to_dict() for record in compliance_report.lineage_records]
            )
            lineage_df.to_csv(buffer, index=False)
        else:
            buffer.write("No lineage records available\n")

        # Coverage metrics
        buffer.write("\n# COVERAGE METRICS\n")
        metrics_data = compliance_report.coverage_metrics.to_dict()
        metrics_df = pd.DataFrame([metrics_data])
        metrics_df.to_csv(buffer, index=False)

    def _write_coverage_csv(
        self, buffer: StringIO, coverage_report: CoverageReport
    ) -> None:
        """Write coverage report data to CSV buffer."""
        # Summary statistics
        buffer.write("# COVERAGE SUMMARY\n")
        summary_data = {
            "total_taxonomy_labels": coverage_report.total_taxonomy_labels,
            "covered_labels": coverage_report.covered_labels,
            "coverage_percentage": coverage_report.coverage_percentage,
        }
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_csv(buffer, index=False)

        # Uncovered labels
        buffer.write("\n# UNCOVERED LABELS\n")
        if coverage_report.uncovered_labels:
            uncovered_df = pd.DataFrame(
                {"uncovered_label": coverage_report.uncovered_labels}
            )
            uncovered_df.to_csv(buffer, index=False)
        else:
            buffer.write("All labels are covered\n")

        # Detector statistics
        buffer.write("\n# DETECTOR STATISTICS\n")
        if coverage_report.detector_statistics:
            detector_rows = []
            for detector, stats in coverage_report.detector_statistics.items():
                row = {"detector": detector}
                row.update(stats)
                detector_rows.append(row)
            detector_df = pd.DataFrame(detector_rows)
            detector_df.to_csv(buffer, index=False)
        else:
            buffer.write("No detector statistics available\n")

    def _write_generic_csv(self, buffer: StringIO, custom_data: Dict[str, Any]) -> None:
        """Write generic custom data to CSV buffer."""
        buffer.write("# CUSTOM DATA\n")

        if not custom_data:
            buffer.write("No custom data available\n")
            return

        # Try to convert custom data to DataFrame
        try:
            if isinstance(custom_data, dict):
                # If it's a simple dict, create a single-row DataFrame
                if all(not isinstance(v, (dict, list)) for v in custom_data.values()):
                    df = pd.DataFrame([custom_data])
                    df.to_csv(buffer, index=False)
                else:
                    # Complex nested data - serialize as JSON strings
                    flattened = {}
                    for key, value in custom_data.items():
                        if isinstance(value, (dict, list)):
                            flattened[key] = json.dumps(value)
                        else:
                            flattened[key] = value
                    df = pd.DataFrame([flattened])
                    df.to_csv(buffer, index=False)
            else:
                # Fallback to JSON representation
                buffer.write(f"Data: {json.dumps(custom_data, indent=2)}\n")
        except Exception as e:
            logger.warning(f"Failed to convert custom data to CSV: {e}")
            buffer.write(f"Data: {json.dumps(custom_data, indent=2)}\n")

    def generate_compliance_report(
        self,
        audit_records: List[Dict[str, Any]],
        control_mappings: List[Dict[str, Any]],
        lineage_records: List[Dict[str, Any]],
        coverage_metrics: Dict[str, Any],
        format_type: ReportFormat,
        tenant_id: Optional[str] = None,
    ) -> Union[bytes, str, Dict[str, Any]]:
        """
        Generate a compliance report with framework mappings and coverage.

        Args:
            audit_records: List of audit record dictionaries
            control_mappings: List of control mapping dictionaries
            lineage_records: List of lineage record dictionaries
            coverage_metrics: Coverage metrics dictionary
            format_type: Output format
            tenant_id: Optional tenant ID

        Returns:
            Generated report in specified format
        """
        from .models import (
            AuditRecord,
            ComplianceControlMapping,
            CoverageMetrics,
            LineageRecord,
        )

        # Convert dictionaries to model objects
        audit_objs = []
        for record_dict in audit_records:
            audit_objs.append(
                AuditRecord(
                    event_id=record_dict["event_id"],
                    tenant_id=record_dict["tenant_id"],
                    detector_type=record_dict["detector_type"],
                    taxonomy_hit=record_dict["taxonomy_hit"],
                    confidence_score=record_dict["confidence_score"],
                    timestamp=(
                        datetime.fromisoformat(record_dict["timestamp"])
                        if isinstance(record_dict["timestamp"], str)
                        else record_dict["timestamp"]
                    ),
                    model_version=record_dict["model_version"],
                    mapping_method=record_dict["mapping_method"],
                    framework_mappings=record_dict.get("framework_mappings", []),
                    metadata=record_dict.get("metadata", {}),
                )
            )

        control_objs = []
        for mapping_dict in control_mappings:
            control_objs.append(
                ComplianceControlMapping(
                    taxonomy_label=mapping_dict["taxonomy_label"],
                    framework=mapping_dict["framework"],
                    control_id=mapping_dict["control_id"],
                    control_description=mapping_dict["control_description"],
                )
            )

        lineage_objs = []
        for lineage_dict in lineage_records:
            lineage_objs.append(
                LineageRecord(
                    detector_name=lineage_dict["detector_name"],
                    detector_version=lineage_dict["detector_version"],
                    original_label=lineage_dict["original_label"],
                    canonical_label=lineage_dict["canonical_label"],
                    confidence_score=lineage_dict["confidence_score"],
                    mapping_method=lineage_dict["mapping_method"],
                    timestamp=(
                        datetime.fromisoformat(lineage_dict["timestamp"])
                        if isinstance(lineage_dict["timestamp"], str)
                        else lineage_dict["timestamp"]
                    ),
                    model_version=lineage_dict.get("model_version"),
                    tenant_id=lineage_dict.get("tenant_id"),
                )
            )

        coverage_obj = CoverageMetrics(
            total_incidents=coverage_metrics["total_incidents"],
            covered_incidents=coverage_metrics["covered_incidents"],
            coverage_percentage=coverage_metrics["coverage_percentage"],
            mttr_hours=coverage_metrics.get("mttr_hours"),
            framework_specific_metrics=coverage_metrics.get(
                "framework_specific_metrics", {}
            ),
        )

        # Create metadata
        metadata = ReportMetadata(
            report_id=str(uuid4()),
            generated_at=datetime.utcnow(),
            taxonomy_version=self.taxonomy_version,
            model_version=self.model_version,
            frameworks_version=self.frameworks_version,
            tenant_id=tenant_id,
            report_type="compliance",
        )

        # Create compliance report
        compliance_report = ComplianceReport(
            metadata=metadata,
            coverage_metrics=coverage_obj,
            control_mappings=control_objs,
            audit_records=audit_objs,
            lineage_records=lineage_objs,
        )

        # Create report data container
        report_data = ReportData(compliance_report=compliance_report)

        return self.generate_report(
            report_data, format_type, tenant_id=tenant_id, report_type="compliance"
        )

    def generate_coverage_report(
        self,
        detector_statistics: Dict[str, Dict[str, Any]],
        taxonomy_coverage: Dict[str, Any],
        format_type: ReportFormat,
        tenant_id: Optional[str] = None,
    ) -> Union[bytes, str, Dict[str, Any]]:
        """
        Generate a coverage report showing taxonomy label coverage.

        Args:
            detector_statistics: Statistics for each detector
            taxonomy_coverage: Overall taxonomy coverage data
            format_type: Output format
            tenant_id: Optional tenant ID

        Returns:
            Generated report in specified format
        """
        from .models import CoverageReport

        # Create metadata
        metadata = ReportMetadata(
            report_id=str(uuid4()),
            generated_at=datetime.utcnow(),
            taxonomy_version=self.taxonomy_version,
            model_version=self.model_version,
            frameworks_version=self.frameworks_version,
            tenant_id=tenant_id,
            report_type="coverage",
        )

        # Create coverage report
        coverage_report = CoverageReport(
            metadata=metadata,
            total_taxonomy_labels=taxonomy_coverage["total_taxonomy_labels"],
            covered_labels=taxonomy_coverage["covered_labels"],
            uncovered_labels=taxonomy_coverage["uncovered_labels"],
            coverage_percentage=taxonomy_coverage["coverage_percentage"],
            detector_statistics=detector_statistics,
            category_breakdown=taxonomy_coverage.get("category_breakdown", {}),
        )

        # Create report data container
        report_data = ReportData(coverage_report=coverage_report)

        return self.generate_report(
            report_data, format_type, tenant_id=tenant_id, report_type="coverage"
        )

    def _datetime_format(
        self, dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S UTC"
    ) -> str:
        """Jinja2 filter for datetime formatting."""
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        return dt.strftime(format_str)

    def _percentage_format(self, value: float, decimals: int = 1) -> str:
        """Jinja2 filter for percentage formatting."""
        return f"{value:.{decimals}f}%"

    def create_template_directory(self) -> None:
        """Create template directory structure with default templates."""
        self.template_dir.mkdir(parents=True, exist_ok=True)
        styles_dir = self.template_dir / "styles"
        styles_dir.mkdir(exist_ok=True)

        # Create default templates if they don't exist
        self._create_default_templates()

        logger.info(f"Template directory created at: {self.template_dir}")

    def _create_default_templates(self) -> None:
        """Create default HTML templates for PDF generation."""
        # Compliance report template
        compliance_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Compliance Report - {{ metadata.report_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }
        .version-info { background: #f5f5f5; padding: 10px; margin: 10px 0; }
        .section { margin: 20px 0; }
        .metrics { display: flex; justify-content: space-around; margin: 20px 0; }
        .metric { text-align: center; padding: 10px; background: #e9e9e9; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Compliance Report</h1>
        <p>Report ID: {{ metadata.report_id }}</p>
        <p>Generated: {{ metadata.generated_at | datetime_format }}</p>
        {% if metadata.tenant_id %}<p>Tenant: {{ metadata.tenant_id }}</p>{% endif %}
    </div>

    <div class="version-info">
        <h3>Version Information</h3>
        <p>Taxonomy Version: {{ metadata.taxonomy_version }}</p>
        <p>Model Version: {{ metadata.model_version }}</p>
        <p>Frameworks Version: {{ metadata.frameworks_version }}</p>
    </div>

    <div class="section">
        <h2>Coverage Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <h3>{{ report_data.compliance_report.coverage_metrics.total_incidents }}</h3>
                <p>Total Incidents</p>
            </div>
            <div class="metric">
                <h3>{{ report_data.compliance_report.coverage_metrics.covered_incidents }}</h3>
                <p>Covered Incidents</p>
            </div>
            <div class="metric">
                <h3>{{ report_data.compliance_report.coverage_metrics.coverage_percentage | percentage }}</h3>
                <p>Coverage</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Control Mappings</h2>
        <table>
            <thead>
                <tr>
                    <th>Taxonomy Label</th>
                    <th>Framework</th>
                    <th>Control ID</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                {% for mapping in report_data.compliance_report.control_mappings %}
                <tr>
                    <td>{{ mapping.taxonomy_label }}</td>
                    <td>{{ mapping.framework }}</td>
                    <td>{{ mapping.control_id }}</td>
                    <td>{{ mapping.control_description }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
        """

        compliance_path = self.template_dir / "compliance_report.html"
        if not compliance_path.exists():
            compliance_path.write_text(compliance_template.strip())

        # Coverage report template
        coverage_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Coverage Report - {{ metadata.report_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }
        .version-info { background: #f5f5f5; padding: 10px; margin: 10px 0; }
        .section { margin: 20px 0; }
        .metrics { display: flex; justify-content: space-around; margin: 20px 0; }
        .metric { text-align: center; padding: 10px; background: #e9e9e9; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .uncovered { color: #d32f2f; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Coverage Report</h1>
        <p>Report ID: {{ metadata.report_id }}</p>
        <p>Generated: {{ metadata.generated_at | datetime_format }}</p>
        {% if metadata.tenant_id %}<p>Tenant: {{ metadata.tenant_id }}</p>{% endif %}
    </div>

    <div class="version-info">
        <h3>Version Information</h3>
        <p>Taxonomy Version: {{ metadata.taxonomy_version }}</p>
        <p>Model Version: {{ metadata.model_version }}</p>
        <p>Frameworks Version: {{ metadata.frameworks_version }}</p>
    </div>

    <div class="section">
        <h2>Coverage Summary</h2>
        <div class="metrics">
            <div class="metric">
                <h3>{{ report_data.coverage_report.total_taxonomy_labels }}</h3>
                <p>Total Labels</p>
            </div>
            <div class="metric">
                <h3>{{ report_data.coverage_report.covered_labels }}</h3>
                <p>Covered Labels</p>
            </div>
            <div class="metric">
                <h3>{{ report_data.coverage_report.coverage_percentage | percentage }}</h3>
                <p>Coverage</p>
            </div>
        </div>
    </div>

    {% if report_data.coverage_report.uncovered_labels %}
    <div class="section">
        <h2>Uncovered Labels</h2>
        <ul class="uncovered">
            {% for label in report_data.coverage_report.uncovered_labels %}
            <li>{{ label }}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}
</body>
</html>
        """

        coverage_path = self.template_dir / "coverage_report.html"
        if not coverage_path.exists():
            coverage_path.write_text(coverage_template.strip())

        # Basic CSS
        css_content = """
body {
    font-family: 'Helvetica', 'Arial', sans-serif;
    line-height: 1.6;
    color: #333;
    margin: 0;
    padding: 20px;
}

.header {
    border-bottom: 3px solid #2196F3;
    padding-bottom: 15px;
    margin-bottom: 30px;
}

.header h1 {
    color: #2196F3;
    margin: 0 0 10px 0;
}

.version-info {
    background: #f8f9fa;
    border-left: 4px solid #2196F3;
    padding: 15px;
    margin: 20px 0;
}

.section {
    margin: 30px 0;
}

.metrics {
    display: flex;
    justify-content: space-around;
    margin: 20px 0;
    flex-wrap: wrap;
}

.metric {
    text-align: center;
    padding: 20px;
    background: #e3f2fd;
    border-radius: 8px;
    min-width: 120px;
    margin: 5px;
}

.metric h3 {
    font-size: 2em;
    margin: 0;
    color: #1976D2;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

th, td {
    border: 1px solid #ddd;
    padding: 12px;
    text-align: left;
}

th {
    background-color: #2196F3;
    color: white;
    font-weight: bold;
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}

.uncovered {
    color: #d32f2f;
    font-weight: bold;
}

@media print {
    body { margin: 0; }
    .header { page-break-after: avoid; }
    .section { page-break-inside: avoid; }
}
        """

        css_path = self.template_dir / "styles" / "report.css"
        if not css_path.exists():
            css_path.write_text(css_content.strip())
