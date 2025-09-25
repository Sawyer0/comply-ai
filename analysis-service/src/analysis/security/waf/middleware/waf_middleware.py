"""
WAF middleware implementation.

This module provides FastAPI middleware that integrates WAF rules
for request filtering and abuse prevention.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from ..interfaces import (
    IWAFMetricsCollector,
    IWAFMiddleware,
    IWAFRuleEngine,
    WAFViolation,
)

logger = logging.getLogger(__name__)


class WAFMiddleware(IWAFMiddleware):
    """
    WAF middleware for request filtering and abuse prevention.

    Integrates WAF rules to scan and block malicious requests
    before they reach the application endpoints.
    """

    def __init__(
        self,
        waf_engine: Optional[IWAFRuleEngine] = None,
        metrics_collector: Optional[IWAFMetricsCollector] = None,
        block_mode: bool = True,
        log_violations: bool = True,
    ):
        """
        Initialize WAF middleware.

        Args:
            waf_engine: WAF rule engine instance
            metrics_collector: Metrics collector for WAF events
            block_mode: Whether to block requests or just log violations
            log_violations: Whether to log WAF violations
        """
        self._waf_engine = waf_engine
        self._metrics_collector = metrics_collector
        self._block_mode = block_mode
        self._log_violations_enabled = log_violations

        # Statistics
        self._stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "violations_by_type": {},
            "violations_by_severity": {},
            "top_attacking_ips": {},
        }

    async def process_request(self, request: Any) -> Tuple[bool, List[WAFViolation]]:
        """Process incoming request."""
        client_ip = self._get_client_ip(request)

        # Update statistics
        self._stats["total_requests"] += 1

        # Skip WAF for health checks and metrics endpoints
        if self._should_skip_waf(request):
            return True, []

        # Extract request data for scanning
        request_data = await self._extract_request_data(request)

        # Scan request with WAF rules
        violations = []
        if self._waf_engine:
            violations = self._waf_engine.scan(
                text=request_data.get("body", ""), client_ip=client_ip
            )

        # Process violations
        if violations:
            await self._handle_violations(violations, client_ip, request)
            return not self._block_mode, violations

        # Record metrics
        if self._metrics_collector:
            self._metrics_collector.record_request(client_ip, False)

        return True, []

    async def process_response(self, response: Any) -> Tuple[bool, List[WAFViolation]]:
        """Process outgoing response."""
        # For now, just return safe response
        return True, []

    def _get_client_ip(self, request: Any) -> str:
        """Extract client IP from request."""
        # Try various headers for client IP
        headers = getattr(request, "headers", {})

        # Check for forwarded headers
        forwarded_for = headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check for real IP header
        real_ip = headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Check for remote address
        remote_addr = getattr(request, "client", {}).get("host", "unknown")
        if remote_addr and remote_addr != "unknown":
            return remote_addr

        return "unknown"

    def _should_skip_waf(self, request: Any) -> bool:
        """Check if request should skip WAF processing."""
        path = getattr(request, "path", "")

        # Skip health checks and metrics
        skip_paths = ["/health", "/metrics", "/status", "/ping"]
        return any(path.startswith(skip_path) for skip_path in skip_paths)

    async def _extract_request_data(self, request: Any) -> Dict[str, Any]:
        """Extract data from request for WAF scanning."""
        data = {
            "method": getattr(request, "method", ""),
            "path": getattr(request, "path", ""),
            "headers": dict(getattr(request, "headers", {})),
            "query_params": dict(getattr(request, "query_params", {})),
            "body": "",
        }

        # Try to get request body
        try:
            if hasattr(request, "body"):
                body = await request.body()
                if body:
                    data["body"] = body.decode("utf-8", errors="ignore")
        except Exception as e:
            logger.warning("Failed to extract request body: %s", e)

        return data

    async def _handle_violations(
        self, violations: List[WAFViolation], client_ip: str, request: Any
    ) -> None:
        """Handle WAF violations."""
        # Update statistics
        self._stats["blocked_requests"] += 1

        # Track attacking IPs
        self._stats["top_attacking_ips"][client_ip] = (
            self._stats["top_attacking_ips"].get(client_ip, 0) + 1
        )

        # Update violation statistics
        for violation in violations:
            # Count by type
            violation_type = violation.violation_type.value
            self._stats["violations_by_type"][violation_type] = (
                self._stats["violations_by_type"].get(violation_type, 0) + 1
            )

            # Count by severity
            severity = violation.severity.value
            self._stats["violations_by_severity"][severity] = (
                self._stats["violations_by_severity"].get(severity, 0) + 1
            )

            # Record metrics
            if self._metrics_collector:
                self._metrics_collector.record_violation(violation)

        # Log violations if enabled
        if self._log_violations_enabled:
            for violation in violations:
                logger.warning(
                    "WAF violation detected",
                    violation_type=violation.violation_type.value,
                    severity=violation.severity.value,
                    rule=violation.rule_name,
                    client_ip=client_ip,
                    target=violation.target,
                )

    def _create_blocked_response(
        self, violations: List[WAFViolation], client_ip: str
    ) -> Dict[str, Any]:
        """Create response for blocked requests."""
        return {
            "status_code": 403,
            "content": {
                "error": "Request blocked by WAF",
                "violations": [v.to_dict() for v in violations],
                "client_ip": client_ip,
                "timestamp": time.time(),
            },
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get WAF middleware statistics."""
        stats = self._stats.copy()

        # Calculate additional metrics
        if stats["total_requests"] > 0:
            stats["block_rate"] = stats["blocked_requests"] / stats["total_requests"]
        else:
            stats["block_rate"] = 0.0

        return stats

    def reset_statistics(self) -> None:
        """Reset WAF middleware statistics."""
        self._stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "violations_by_type": {},
            "violations_by_severity": {},
            "top_attacking_ips": {},
        }
        logger.info("WAF middleware statistics reset")
