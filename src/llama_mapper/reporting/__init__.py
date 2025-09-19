"""
Reporting and audit capabilities for the Llama Mapper system.

This module provides comprehensive reporting functionality including:
- PDF report generation via WeasyPrint with embedded version tags
- CSV exports via Pandas with version metadata  
- JSON API responses with version headers
- Audit trail and compliance mapping capabilities
"""

from .audit_trail import AuditTrailManager, IncidentData
from .models import (
    AuditRecord,
    ComplianceReport,
    CoverageReport,
    LineageRecord,
    ReportMetadata,
)
from .report_generator import ReportData, ReportFormat, ReportGenerator

__all__ = [
    "ReportGenerator",
    "ReportFormat",
    "ReportData",
    "AuditTrailManager",
    "IncidentData",
    "ReportMetadata",
    "ComplianceReport",
    "CoverageReport",
    "AuditRecord",
    "LineageRecord",
]
