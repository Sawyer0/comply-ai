"""
Data models for reporting and audit functionality.

Defines the core data structures used across the reporting system
for consistency and type safety.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ReportFormat(Enum):
    """Supported report output formats."""

    PDF = "pdf"
    CSV = "csv"
    JSON = "json"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""

    SOC2 = "SOC2"
    ISO27001 = "ISO27001"
    HIPAA = "HIPAA"


@dataclass
class ReportMetadata:
    """Metadata embedded in all reports for version tracking and audit trails."""

    report_id: str
    generated_at: datetime
    taxonomy_version: str
    model_version: str
    frameworks_version: str
    generator_version: str = "1.0.0"
    tenant_id: Optional[str] = None
    requested_by: Optional[str] = None
    report_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for embedding in reports."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "taxonomy_version": self.taxonomy_version,
            "model_version": self.model_version,
            "frameworks_version": self.frameworks_version,
            "generator_version": self.generator_version,
            "tenant_id": self.tenant_id,
            "requested_by": self.requested_by,
            "report_type": self.report_type,
        }


@dataclass
class LineageRecord:
    """Records the lineage from detector output to canonical label."""

    detector_name: str
    detector_version: str
    original_label: str
    canonical_label: str
    confidence_score: float
    mapping_method: str  # "model" or "fallback"
    timestamp: datetime
    model_version: Optional[str] = None
    tenant_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert lineage record to dictionary."""
        return {
            "detector_name": self.detector_name,
            "detector_version": self.detector_version,
            "original_label": self.original_label,
            "canonical_label": self.canonical_label,
            "confidence_score": self.confidence_score,
            "mapping_method": self.mapping_method,
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version,
            "tenant_id": self.tenant_id,
        }


@dataclass
class AuditRecord:
    """Audit record for compliance reporting."""

    event_id: str
    tenant_id: str
    detector_type: str
    taxonomy_hit: str
    confidence_score: float
    timestamp: datetime
    model_version: str
    mapping_method: str
    framework_mappings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit record to dictionary."""
        return {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "detector_type": self.detector_type,
            "taxonomy_hit": self.taxonomy_hit,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version,
            "mapping_method": self.mapping_method,
            "framework_mappings": self.framework_mappings,
            "metadata": self.metadata,
        }


@dataclass
class ComplianceControlMapping:
    """Mapping between taxonomy labels and compliance controls."""

    taxonomy_label: str
    framework: str
    control_id: str
    control_description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert control mapping to dictionary."""
        return {
            "taxonomy_label": self.taxonomy_label,
            "framework": self.framework,
            "control_id": self.control_id,
            "control_description": self.control_description,
        }


@dataclass
class CoverageMetrics:
    """Coverage metrics for compliance reporting."""

    total_incidents: int
    covered_incidents: int
    coverage_percentage: float
    mttr_hours: Optional[float] = None  # Mean Time To Resolution
    framework_specific_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert coverage metrics to dictionary."""
        return {
            "total_incidents": self.total_incidents,
            "covered_incidents": self.covered_incidents,
            "coverage_percentage": self.coverage_percentage,
            "mttr_hours": self.mttr_hours,
            "framework_specific_metrics": self.framework_specific_metrics,
        }


@dataclass
class ComplianceReport:
    """Complete compliance report with framework mappings and coverage."""

    metadata: ReportMetadata
    coverage_metrics: CoverageMetrics
    control_mappings: List[ComplianceControlMapping]
    audit_records: List[AuditRecord]
    lineage_records: List[LineageRecord]
    framework_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert compliance report to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "coverage_metrics": self.coverage_metrics.to_dict(),
            "control_mappings": [
                mapping.to_dict() for mapping in self.control_mappings
            ],
            "audit_records": [record.to_dict() for record in self.audit_records],
            "lineage_records": [record.to_dict() for record in self.lineage_records],
            "framework_summary": self.framework_summary,
        }


@dataclass
class CoverageReport:
    """Coverage report showing taxonomy label coverage and detector mapping statistics."""

    metadata: ReportMetadata
    total_taxonomy_labels: int
    covered_labels: int
    uncovered_labels: List[str]
    coverage_percentage: float
    detector_statistics: Dict[str, Dict[str, Any]]
    category_breakdown: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert coverage report to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "total_taxonomy_labels": self.total_taxonomy_labels,
            "covered_labels": self.covered_labels,
            "uncovered_labels": self.uncovered_labels,
            "coverage_percentage": self.coverage_percentage,
            "detector_statistics": self.detector_statistics,
            "category_breakdown": self.category_breakdown,
        }


@dataclass
class ReportData:
    """Container for all data needed to generate reports."""

    compliance_report: Optional[ComplianceReport] = None
    coverage_report: Optional[CoverageReport] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def has_compliance_data(self) -> bool:
        """Check if compliance report data is available."""
        return self.compliance_report is not None

    def has_coverage_data(self) -> bool:
        """Check if coverage report data is available."""
        return self.coverage_report is not None
