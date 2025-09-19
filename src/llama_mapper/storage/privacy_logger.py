"""
Privacy-first logging system for LLaMA Mapper.

This module provides a logging system that stores only metadata
(tenant ID, detector type, taxonomy hit) and never persists raw
detector inputs, ensuring compliance with data protection regulations.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from ..config.settings import Settings


logger = structlog.get_logger(__name__)


class LogLevel(Enum):
    """Privacy log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    AUDIT = "audit"


class EventType(Enum):
    """Types of events that can be logged."""
    MAPPING_SUCCESS = "mapping_success"
    MAPPING_FAILURE = "mapping_failure"
    CONFIDENCE_FALLBACK = "confidence_fallback"
    SCHEMA_VALIDATION_ERROR = "schema_validation_error"
    TENANT_ACCESS_DENIED = "tenant_access_denied"
    DETECTOR_ACCESS_DENIED = "detector_access_denied"
    ENCRYPTION_ERROR = "encryption_error"
    STORAGE_ERROR = "storage_error"
    AUDIT_TRAIL = "audit_trail"


@dataclass
class PrivacyLogEntry:
    """
    Privacy-first log entry structure.
    
    Contains only metadata - no raw detector inputs or sensitive data.
    """
    event_id: str
    timestamp: datetime
    tenant_id: str
    event_type: EventType
    log_level: LogLevel
    
    # Metadata only - no raw data
    detector_type: Optional[str] = None
    taxonomy_hit: Optional[str] = None
    confidence_score: Optional[float] = None
    model_version: Optional[str] = None
    
    # Error information (sanitized)
    error_code: Optional[str] = None
    error_category: Optional[str] = None
    
    # Audit trail information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Performance metrics
    processing_time_ms: Optional[int] = None
    
    # Additional metadata (sanitized)
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['log_level'] = self.log_level.value
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class PrivacyLogger:
    """
    Privacy-first logging system.
    
    Features:
    - Stores only metadata (tenant ID, detector type, taxonomy hit)
    - Never persists raw detector inputs
    - Implements audit trail for compliance reporting
    - Configurable log levels and retention
    - Structured logging with sanitization
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logger.bind(component="privacy_logger")
        
        # Log storage (in production, this would be a database or log aggregation system)
        self._log_entries: List[PrivacyLogEntry] = []
        
        # Privacy settings
        self.max_message_length = settings.logging.max_message_length
        self.privacy_mode = settings.logging.privacy_mode
        
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata to remove sensitive information.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Sanitized metadata dictionary
        """
        if not self.privacy_mode:
            return metadata
        
        sanitized = {}
        
        # Allow-list of safe metadata keys
        safe_keys = {
            'detector_type', 'detector_version', 'model_version',
            'taxonomy_category', 'confidence_threshold', 'processing_time_ms',
            'request_id', 'session_id', 'user_id', 'tenant_id',
            'framework_mappings', 'compliance_tags'
        }
        
        for key, value in metadata.items():
            if key in safe_keys:
                # Further sanitize string values
                if isinstance(value, str) and len(value) > self.max_message_length:
                    sanitized[key] = value[:self.max_message_length] + "..."
                else:
                    sanitized[key] = value
        
        return sanitized
    
    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        return str(uuid.uuid4())
    
    async def log_mapping_success(
        self,
        tenant_id: str,
        detector_type: str,
        taxonomy_hit: str,
        confidence_score: float,
        model_version: str,
        processing_time_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a successful mapping operation.
        
        Args:
            tenant_id: Tenant identifier
            detector_type: Type of detector used
            taxonomy_hit: Canonical taxonomy label that was mapped
            confidence_score: Model confidence score
            model_version: Version of the model used
            processing_time_ms: Processing time in milliseconds
            metadata: Additional metadata (will be sanitized)
            
        Returns:
            Event ID for the log entry
        """
        event_id = self._generate_event_id()
        
        entry = PrivacyLogEntry(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc),
            tenant_id=tenant_id,
            event_type=EventType.MAPPING_SUCCESS,
            log_level=LogLevel.INFO,
            detector_type=detector_type,
            taxonomy_hit=taxonomy_hit,
            confidence_score=confidence_score,
            model_version=model_version,
            processing_time_ms=processing_time_ms,
            metadata=self._sanitize_metadata(metadata or {})
        )
        
        await self._store_log_entry(entry)
        
        self.logger.info(
            "Mapping success logged",
            event_id=event_id,
            tenant_id=tenant_id,
            detector_type=detector_type,
            taxonomy_hit=taxonomy_hit,
            confidence_score=confidence_score
        )
        
        return event_id
    
    async def log_mapping_failure(
        self,
        tenant_id: str,
        detector_type: str,
        error_code: str,
        error_category: str,
        model_version: str,
        processing_time_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a failed mapping operation.
        
        Args:
            tenant_id: Tenant identifier
            detector_type: Type of detector used
            error_code: Error code (sanitized)
            error_category: Error category (e.g., "validation", "model", "timeout")
            model_version: Version of the model used
            processing_time_ms: Processing time in milliseconds
            metadata: Additional metadata (will be sanitized)
            
        Returns:
            Event ID for the log entry
        """
        event_id = self._generate_event_id()
        
        entry = PrivacyLogEntry(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc),
            tenant_id=tenant_id,
            event_type=EventType.MAPPING_FAILURE,
            log_level=LogLevel.ERROR,
            detector_type=detector_type,
            error_code=error_code,
            error_category=error_category,
            model_version=model_version,
            processing_time_ms=processing_time_ms,
            metadata=self._sanitize_metadata(metadata or {})
        )
        
        await self._store_log_entry(entry)
        
        self.logger.error(
            "Mapping failure logged",
            event_id=event_id,
            tenant_id=tenant_id,
            detector_type=detector_type,
            error_code=error_code,
            error_category=error_category
        )
        
        return event_id
    
    async def log_confidence_fallback(
        self,
        tenant_id: str,
        detector_type: str,
        original_confidence: float,
        threshold: float,
        fallback_taxonomy: str,
        model_version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log when confidence falls below threshold and fallback is used.
        
        Args:
            tenant_id: Tenant identifier
            detector_type: Type of detector used
            original_confidence: Original model confidence
            threshold: Confidence threshold that triggered fallback
            fallback_taxonomy: Taxonomy assigned by fallback mechanism
            model_version: Version of the model used
            metadata: Additional metadata (will be sanitized)
            
        Returns:
            Event ID for the log entry
        """
        event_id = self._generate_event_id()
        
        entry = PrivacyLogEntry(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc),
            tenant_id=tenant_id,
            event_type=EventType.CONFIDENCE_FALLBACK,
            log_level=LogLevel.WARNING,
            detector_type=detector_type,
            taxonomy_hit=fallback_taxonomy,
            confidence_score=original_confidence,
            model_version=model_version,
            metadata=self._sanitize_metadata({
                **(metadata or {}),
                'confidence_threshold': threshold,
                'fallback_used': True
            })
        )
        
        await self._store_log_entry(entry)
        
        self.logger.warning(
            "Confidence fallback logged",
            event_id=event_id,
            tenant_id=tenant_id,
            detector_type=detector_type,
            original_confidence=original_confidence,
            threshold=threshold,
            fallback_taxonomy=fallback_taxonomy
        )
        
        return event_id
    
    async def log_tenant_access_denied(
        self,
        tenant_id: str,
        requested_resource: str,
        access_level: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log when tenant access is denied.
        
        Args:
            tenant_id: Tenant identifier
            requested_resource: Resource that was requested
            access_level: Access level that was attempted
            user_id: User identifier (if available)
            session_id: Session identifier (if available)
            metadata: Additional metadata (will be sanitized)
            
        Returns:
            Event ID for the log entry
        """
        event_id = self._generate_event_id()
        
        entry = PrivacyLogEntry(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc),
            tenant_id=tenant_id,
            event_type=EventType.TENANT_ACCESS_DENIED,
            log_level=LogLevel.WARNING,
            user_id=user_id,
            session_id=session_id,
            metadata=self._sanitize_metadata({
                **(metadata or {}),
                'requested_resource': requested_resource,
                'access_level': access_level
            })
        )
        
        await self._store_log_entry(entry)
        
        self.logger.warning(
            "Tenant access denied",
            event_id=event_id,
            tenant_id=tenant_id,
            requested_resource=requested_resource,
            access_level=access_level,
            user_id=user_id
        )
        
        return event_id
    
    async def log_audit_trail(
        self,
        tenant_id: str,
        action: str,
        resource: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an audit trail entry for compliance reporting.
        
        Args:
            tenant_id: Tenant identifier
            action: Action performed (e.g., "create", "read", "update", "delete")
            resource: Resource affected
            user_id: User identifier (if available)
            session_id: Session identifier (if available)
            request_id: Request identifier (if available)
            metadata: Additional metadata (will be sanitized)
            
        Returns:
            Event ID for the log entry
        """
        event_id = self._generate_event_id()
        
        entry = PrivacyLogEntry(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc),
            tenant_id=tenant_id,
            event_type=EventType.AUDIT_TRAIL,
            log_level=LogLevel.AUDIT,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            metadata=self._sanitize_metadata({
                **(metadata or {}),
                'action': action,
                'resource': resource
            })
        )
        
        await self._store_log_entry(entry)
        
        self.logger.info(
            "Audit trail logged",
            event_id=event_id,
            tenant_id=tenant_id,
            action=action,
            resource=resource,
            user_id=user_id
        )
        
        return event_id
    
    async def _store_log_entry(self, entry: PrivacyLogEntry) -> None:
        """
        Store a log entry.
        
        In production, this would write to a database or log aggregation system.
        For now, we store in memory for testing.
        
        Args:
            entry: Log entry to store
        """
        self._log_entries.append(entry)
        
        # In production, you would:
        # 1. Write to a database (PostgreSQL/ClickHouse)
        # 2. Send to log aggregation system (ELK, Splunk, etc.)
        # 3. Write to structured log files
        # 4. Send to SIEM systems for security monitoring
    
    async def get_audit_trail(
        self,
        tenant_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[EventType]] = None,
        limit: int = 1000
    ) -> List[PrivacyLogEntry]:
        """
        Retrieve audit trail entries for compliance reporting.
        
        Args:
            tenant_id: Tenant identifier
            start_time: Start time for the query
            end_time: End time for the query
            event_types: Filter by event types
            limit: Maximum number of entries to return
            
        Returns:
            List of log entries matching the criteria
        """
        entries = []
        
        for entry in self._log_entries:
            # Filter by tenant
            if entry.tenant_id != tenant_id:
                continue
            
            # Filter by time range
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            
            # Filter by event types
            if event_types and entry.event_type not in event_types:
                continue
            
            entries.append(entry)
            
            if len(entries) >= limit:
                break
        
        # Sort by timestamp (most recent first)
        entries.sort(key=lambda x: x.timestamp, reverse=True)
        
        return entries
    
    async def get_compliance_report(
        self,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Generate a compliance report for the specified time period.
        
        Args:
            tenant_id: Tenant identifier
            start_time: Start time for the report
            end_time: End time for the report
            
        Returns:
            Compliance report with statistics and audit information
        """
        entries = await self.get_audit_trail(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time
        )
        
        # Calculate statistics
        total_events = len(entries)
        event_counts = {}
        detector_counts = {}
        taxonomy_counts = {}
        error_counts = {}
        
        for entry in entries:
            # Count by event type
            event_type = entry.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Count by detector type
            if entry.detector_type:
                detector_counts[entry.detector_type] = detector_counts.get(entry.detector_type, 0) + 1
            
            # Count by taxonomy hit
            if entry.taxonomy_hit:
                taxonomy_counts[entry.taxonomy_hit] = taxonomy_counts.get(entry.taxonomy_hit, 0) + 1
            
            # Count by error category
            if entry.error_category:
                error_counts[entry.error_category] = error_counts.get(entry.error_category, 0) + 1
        
        return {
            'tenant_id': tenant_id,
            'report_period': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            },
            'summary': {
                'total_events': total_events,
                'event_types': event_counts,
                'detector_usage': detector_counts,
                'taxonomy_distribution': taxonomy_counts,
                'error_distribution': error_counts
            },
            'audit_entries': [entry.to_dict() for entry in entries[:100]]  # Include first 100 entries
        }
    
    async def cleanup_old_entries(self, retention_days: int = 90) -> int:
        """
        Clean up log entries older than the retention period.
        
        Args:
            retention_days: Number of days to retain log entries
            
        Returns:
            Number of entries cleaned up
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=retention_days)
        
        original_count = len(self._log_entries)
        self._log_entries = [
            entry for entry in self._log_entries
            if entry.timestamp >= cutoff_time
        ]
        
        cleaned_count = original_count - len(self._log_entries)
        
        if cleaned_count > 0:
            self.logger.info(
                "Privacy log cleanup completed",
                entries_cleaned=cleaned_count,
                retention_days=retention_days
            )
        
        return cleaned_count