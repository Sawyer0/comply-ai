"""
Metadata Logger

This module provides metadata-only logging functionality.
Follows SRP by focusing solely on logging metadata without raw content.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict

from .content_scrubber import ContentScrubber

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Metadata-only log entry."""

    timestamp: datetime
    component: str
    operation: str
    metadata: Dict[str, Any]
    content_hash: Optional[str] = None
    content_length: Optional[int] = None
    content_type: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None


class MetadataLogger:
    """
    Logs only metadata, never raw content.

    Single responsibility: Provide privacy-compliant logging that captures
    metadata and statistics without storing sensitive content.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logging_config = config.get("metadata_logging", {})

        # Configuration
        self.enabled = self.logging_config.get("enabled", True)
        self.include_content_hash = self.logging_config.get(
            "include_content_hash", True
        )
        self.include_content_stats = self.logging_config.get(
            "include_content_stats", True
        )
        self.max_metadata_size = self.logging_config.get("max_metadata_size", 1024)

        # Content scrubber for metadata
        self.content_scrubber = ContentScrubber(config)

        # Log storage (in production, would use proper log aggregation)
        self.log_entries: List[LogEntry] = []
        self.max_entries = self.logging_config.get("max_entries", 10000)

        # Statistics
        self.log_stats = {
            "total_logs": 0,
            "logs_by_component": {},
            "logs_by_operation": {},
            "metadata_bytes_logged": 0,
            "content_bytes_processed": 0,
        }

        logger.info(
            "Metadata Logger initialized",
            enabled=self.enabled,
            include_hash=self.include_content_hash,
        )

    def log_analysis_request(
        self,
        component: str,
        request_data: Dict[str, Any],
        content: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Log analysis request with metadata only.

        Args:
            component: Component making the request
            request_data: Request metadata (will be scrubbed)
            content: Raw content (will NOT be logged, only stats)
            user_context: User context information
        """
        if not self.enabled:
            return

        try:
            # Extract metadata from request
            metadata = self._extract_request_metadata(request_data)

            # Add content statistics without content
            if content:
                metadata.update(self._get_content_statistics(content))

            # Add user context
            if user_context:
                metadata.update(self._extract_user_metadata(user_context))

            # Create log entry
            log_entry = LogEntry(
                timestamp=datetime.now(timezone.utc),
                component=component,
                operation="analysis_request",
                metadata=metadata,
                content_hash=(
                    self._generate_content_hash(content)
                    if content and self.include_content_hash
                    else None
                ),
                content_length=len(content) if content else None,
                content_type=self._detect_content_type(content) if content else None,
                user_id=user_context.get("user_id") if user_context else None,
                session_id=user_context.get("session_id") if user_context else None,
                correlation_id=(
                    user_context.get("correlation_id") if user_context else None
                ),
            )

            self._store_log_entry(log_entry)

        except Exception as e:
            logger.error(
                "Failed to log analysis request", component=component, error=str(e)
            )

    def log_analysis_response(
        self,
        component: str,
        response_data: Dict[str, Any],
        processing_time: float,
        user_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Log analysis response with metadata only.

        Args:
            component: Component generating the response
            response_data: Response metadata (will be scrubbed)
            processing_time: Time taken to process request
            user_context: User context information
        """
        if not self.enabled:
            return

        try:
            # Extract metadata from response
            metadata = self._extract_response_metadata(response_data)
            metadata["processing_time_seconds"] = processing_time

            # Add user context
            if user_context:
                metadata.update(self._extract_user_metadata(user_context))

            # Create log entry
            log_entry = LogEntry(
                timestamp=datetime.now(timezone.utc),
                component=component,
                operation="analysis_response",
                metadata=metadata,
                user_id=user_context.get("user_id") if user_context else None,
                session_id=user_context.get("session_id") if user_context else None,
                correlation_id=(
                    user_context.get("correlation_id") if user_context else None
                ),
            )

            self._store_log_entry(log_entry)

        except Exception as e:
            logger.error(
                "Failed to log analysis response", component=component, error=str(e)
            )

    def log_quality_event(
        self,
        component: str,
        event_type: str,
        event_data: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Log quality-related event with metadata only.

        Args:
            component: Component generating the event
            event_type: Type of quality event
            event_data: Event metadata (will be scrubbed)
            user_context: User context information
        """
        if not self.enabled:
            return

        try:
            # Scrub event data
            scrubbed_metadata = self.content_scrubber.scrub_dict(event_data)
            scrubbed_metadata["event_type"] = event_type

            # Add user context
            if user_context:
                scrubbed_metadata.update(self._extract_user_metadata(user_context))

            # Create log entry
            log_entry = LogEntry(
                timestamp=datetime.now(timezone.utc),
                component=component,
                operation="quality_event",
                metadata=scrubbed_metadata,
                user_id=user_context.get("user_id") if user_context else None,
                session_id=user_context.get("session_id") if user_context else None,
                correlation_id=(
                    user_context.get("correlation_id") if user_context else None
                ),
            )

            self._store_log_entry(log_entry)

        except Exception as e:
            logger.error(
                "Failed to log quality event",
                component=component,
                event_type=event_type,
                error=str(e),
            )

    def log_error_event(
        self,
        component: str,
        error_type: str,
        error_metadata: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Log error event with metadata only.

        Args:
            component: Component where error occurred
            error_type: Type of error
            error_metadata: Error metadata (will be scrubbed)
            user_context: User context information
        """
        if not self.enabled:
            return

        try:
            # Scrub error metadata
            scrubbed_metadata = self.content_scrubber.scrub_dict(error_metadata)
            scrubbed_metadata["error_type"] = error_type

            # Add user context
            if user_context:
                scrubbed_metadata.update(self._extract_user_metadata(user_context))

            # Create log entry
            log_entry = LogEntry(
                timestamp=datetime.now(timezone.utc),
                component=component,
                operation="error_event",
                metadata=scrubbed_metadata,
                user_id=user_context.get("user_id") if user_context else None,
                session_id=user_context.get("session_id") if user_context else None,
                correlation_id=(
                    user_context.get("correlation_id") if user_context else None
                ),
            )

            self._store_log_entry(log_entry)

        except Exception as e:
            logger.error(
                "Failed to log error event",
                component=component,
                error_type=error_type,
                error=str(e),
            )

    def get_log_entries(
        self,
        component: Optional[str] = None,
        operation: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get log entries with optional filtering.

        Args:
            component: Filter by component
            operation: Filter by operation
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of entries to return

        Returns:
            List of log entries as dictionaries
        """
        try:
            filtered_entries = []

            for entry in reversed(self.log_entries):  # Most recent first
                # Apply filters
                if component and entry.component != component:
                    continue
                if operation and entry.operation != operation:
                    continue
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue

                filtered_entries.append(asdict(entry))

                if len(filtered_entries) >= limit:
                    break

            return filtered_entries

        except Exception as e:
            logger.error("Failed to get log entries", error=str(e))
            return []

    def get_log_statistics(self) -> Dict[str, Any]:
        """Get metadata logging statistics."""
        return {
            "enabled": self.enabled,
            "total_logs": self.log_stats["total_logs"],
            "logs_by_component": self.log_stats["logs_by_component"].copy(),
            "logs_by_operation": self.log_stats["logs_by_operation"].copy(),
            "metadata_bytes_logged": self.log_stats["metadata_bytes_logged"],
            "content_bytes_processed": self.log_stats["content_bytes_processed"],
            "current_entries": len(self.log_entries),
            "max_entries": self.max_entries,
            "configuration": {
                "include_content_hash": self.include_content_hash,
                "include_content_stats": self.include_content_stats,
                "max_metadata_size": self.max_metadata_size,
            },
        }

    def _extract_request_metadata(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from request data."""
        # Scrub the request data
        scrubbed_data = self.content_scrubber.scrub_dict(request_data)

        # Extract key metadata fields
        metadata = {
            "request_type": scrubbed_data.get("request_type", "unknown"),
            "analysis_type": scrubbed_data.get("analysis_type", "unknown"),
            "parameters_count": len(scrubbed_data.get("parameters", {})),
            "has_context": "context" in scrubbed_data,
            "request_size_bytes": len(json.dumps(scrubbed_data)),
        }

        # Add specific fields if present
        if "confidence_threshold" in scrubbed_data:
            metadata["confidence_threshold"] = scrubbed_data["confidence_threshold"]

        if "frameworks" in scrubbed_data:
            metadata["frameworks_requested"] = scrubbed_data["frameworks"]

        return self._limit_metadata_size(metadata)

    def _extract_response_metadata(
        self, response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract metadata from response data."""
        # Scrub the response data
        scrubbed_data = self.content_scrubber.scrub_dict(response_data)

        # Extract key metadata fields
        metadata = {
            "response_type": scrubbed_data.get("response_type", "unknown"),
            "success": scrubbed_data.get("success", True),
            "confidence": scrubbed_data.get("confidence"),
            "patterns_found": len(scrubbed_data.get("patterns", [])),
            "recommendations_count": len(scrubbed_data.get("recommendations", [])),
            "response_size_bytes": len(json.dumps(scrubbed_data)),
        }

        # Add error information if present
        if "error" in scrubbed_data:
            metadata["has_error"] = True
            metadata["error_type"] = type(scrubbed_data["error"]).__name__

        return self._limit_metadata_size(metadata)

    def _extract_user_metadata(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user metadata from context."""
        # Scrub user context
        scrubbed_context = self.content_scrubber.scrub_dict(user_context)

        # Extract safe user metadata
        metadata = {}

        safe_fields = [
            "tenant_id",
            "organization_id",
            "user_role",
            "session_duration",
            "request_count",
        ]
        for field in safe_fields:
            if field in scrubbed_context:
                metadata[f"user_{field}"] = scrubbed_context[field]

        return metadata

    def _get_content_statistics(self, content: str) -> Dict[str, Any]:
        """Get statistics about content without storing the content."""
        if not self.include_content_stats:
            return {}

        try:
            stats = {
                "content_length": len(content),
                "content_lines": content.count("\n") + 1,
                "content_words": len(content.split()),
                "content_chars": len(content.strip()),
            }

            # Update global statistics
            self.log_stats["content_bytes_processed"] += len(content)

            return stats

        except Exception as e:
            logger.error("Failed to get content statistics", error=str(e))
            return {}

    def _generate_content_hash(self, content: str) -> str:
        """Generate hash of content for tracking without storing content."""
        import hashlib

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _detect_content_type(self, content: str) -> str:
        """Detect content type from content characteristics."""
        try:
            if content.strip().startswith("{") and content.strip().endswith("}"):
                return "json"
            elif content.strip().startswith("<") and content.strip().endswith(">"):
                return "xml"
            elif "\n" in content and any(
                line.strip().startswith("#") for line in content.split("\n")[:5]
            ):
                return "markdown"
            else:
                return "text"
        except Exception:
            return "unknown"

    def _limit_metadata_size(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Limit metadata size to prevent excessive logging."""
        try:
            metadata_json = json.dumps(metadata)

            if len(metadata_json) <= self.max_metadata_size:
                return metadata

            # Truncate large metadata
            truncated_metadata = {"_truncated": True}

            for key, value in metadata.items():
                value_json = json.dumps({key: value})

                if (
                    len(json.dumps(truncated_metadata)) + len(value_json)
                    <= self.max_metadata_size
                ):
                    truncated_metadata[key] = value
                else:
                    break

            return truncated_metadata

        except Exception as e:
            logger.error("Failed to limit metadata size", error=str(e))
            return {"error": "metadata_size_limit_failed"}

    def _store_log_entry(self, log_entry: LogEntry):
        """Store log entry in memory (in production, would use proper storage)."""
        try:
            # Add to storage
            self.log_entries.append(log_entry)

            # Maintain size limit
            if len(self.log_entries) > self.max_entries:
                self.log_entries = self.log_entries[-self.max_entries :]

            # Update statistics
            self.log_stats["total_logs"] += 1

            component_count = self.log_stats["logs_by_component"].get(
                log_entry.component, 0
            )
            self.log_stats["logs_by_component"][log_entry.component] = (
                component_count + 1
            )

            operation_count = self.log_stats["logs_by_operation"].get(
                log_entry.operation, 0
            )
            self.log_stats["logs_by_operation"][log_entry.operation] = (
                operation_count + 1
            )

            metadata_size = len(json.dumps(log_entry.metadata))
            self.log_stats["metadata_bytes_logged"] += metadata_size

        except Exception as e:
            logger.error("Failed to store log entry", error=str(e))

    def clear_old_entries(self, older_than_hours: int = 24):
        """Clear log entries older than specified hours."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)

            original_count = len(self.log_entries)
            self.log_entries = [
                entry for entry in self.log_entries if entry.timestamp > cutoff_time
            ]

            cleared_count = original_count - len(self.log_entries)

            logger.info(
                "Cleared old log entries",
                cleared_count=cleared_count,
                older_than_hours=older_than_hours,
            )

        except Exception as e:
            logger.error("Failed to clear old entries", error=str(e))
