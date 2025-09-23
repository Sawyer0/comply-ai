"""
WAF middleware implementation.

This module provides FastAPI middleware that integrates WAF rules
for request filtering and abuse prevention.
"""

import json
import logging
import time
from typing import Dict, Optional, Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..interfaces import IWAFRuleEngine, IWAFMiddleware, WAFViolation, IWAFMetricsCollector

logger = logging.getLogger(__name__)


class WAFMiddleware(BaseHTTPMiddleware, IWAFMiddleware):
    """
    WAF middleware for request filtering and abuse prevention.
    
    Integrates WAF rules to scan and block malicious requests
    before they reach the application endpoints.
    """
    
    def __init__(
        self,
        app,
        waf_engine: Optional[IWAFRuleEngine] = None,
        metrics_collector: Optional[IWAFMetricsCollector] = None,
        block_mode: bool = True,
        log_violations: bool = True
    ):
        """
        Initialize WAF middleware.
        
        Args:
            app: FastAPI application
            waf_engine: WAF rule engine instance
            metrics_collector: Metrics collector for WAF events
            block_mode: Whether to block requests or just log violations
            log_violations: Whether to log WAF violations
        """
        super().__init__(app)
        self._waf_engine = waf_engine
        self._metrics_collector = metrics_collector
        self._block_mode = block_mode
        self._log_violations = log_violations
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "violations_by_type": {},
            "violations_by_severity": {},
            "top_attacking_ips": {}
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through WAF rules."""
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        # Update statistics
        self._stats["total_requests"] += 1
        
        # Skip WAF for health checks and metrics endpoints
        if self._should_skip_waf(request):
            return await call_next(request)
        
        # Extract request data for scanning
        request_data = await self._extract_request_data(request)
        
        # Scan request with WAF rules
        is_safe, violations = self._waf_engine.scan_request(
            method=request.method,
            path=str(request.url.path),
            headers=dict(request.headers),
            query_params=dict(request.query_params),
            body=request_data.get("body"),
            client_ip=client_ip
        )
        
        # Process violations
        if violations:
            await self._handle_violations(violations, client_ip, request)
            
            if self._block_mode:
                return self._create_blocked_response(violations, client_ip)
        
        # Record metrics
        processing_time = (time.time() - start_time) * 1000
        if self._metrics_collector:
            self._metrics_collector.record_waf_scan(
                is_safe=is_safe,
                violations_count=len(violations),
                processing_time_ms=processing_time,
                client_ip=client_ip
            )
        
        # Forward safe requests
        return await call_next(request)
    
    async def process_request(self, request: Any) -> tuple[bool, list[WAFViolation]]:
        """Process request through WAF (interface method)."""
        client_ip = self._get_client_ip(request)
        request_data = await self._extract_request_data(request)
        
        return self._waf_engine.scan_request(
            method=request.method,
            path=str(request.url.path),
            headers=dict(request.headers),
            query_params=dict(request.query_params),
            body=request_data.get("body"),
            client_ip=client_ip
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded IP headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _should_skip_waf(self, request: Request) -> bool:
        """Check if request should skip WAF scanning."""
        skip_paths = {
            "/health",
            "/metrics",
            "/api/v1/analysis/health",
            "/api/v1/analysis/metrics",
            "/docs",
            "/openapi.json",
            "/redoc"
        }
        
        return str(request.url.path) in skip_paths
    
    async def _extract_request_data(self, request: Request) -> Dict[str, Optional[str]]:
        """Extract request data for WAF scanning."""
        request_data = {"body": None}
        
        # Extract body for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Read body without consuming it
                body = await request.body()
                if body:
                    # Try to parse as JSON
                    try:
                        json_data = json.loads(body.decode())
                        request_data["body"] = json.dumps(json_data, separators=(',', ':'))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # If not JSON, use as string (truncated for security)
                        body_str = body.decode('utf-8', errors='ignore')
                        request_data["body"] = body_str[:10000]  # Limit size
            except Exception as e:
                logger.warning(f"Failed to extract request body: {e}")
        
        return request_data
    
    async def _handle_violations(
        self,
        violations: list[WAFViolation],
        client_ip: str,
        request: Request
    ) -> None:
        """Handle WAF violations."""
        # Update statistics
        self._stats["blocked_requests"] += 1
        
        for violation in violations:
            # Update violation type statistics
            violation_type = violation.violation_type.value
            self._stats["violations_by_type"][violation_type] = \
                self._stats["violations_by_type"].get(violation_type, 0) + 1
            
            # Update severity statistics
            severity = violation.severity.value
            self._stats["violations_by_severity"][severity] = \
                self._stats["violations_by_severity"].get(severity, 0) + 1
            
            # Update attacking IP statistics
            self._stats["top_attacking_ips"][client_ip] = \
                self._stats["top_attacking_ips"].get(client_ip, 0) + 1
        
        # Log violations if enabled
        if self._log_violations:
            self._log_violations(violations, client_ip, request)
    
    def _log_violations(
        self,
        violations: list[WAFViolation],
        client_ip: str,
        request: Request
    ) -> None:
        """Log WAF violations with context."""
        for violation in violations:
            log_data = {
                "event": "waf_violation",
                "client_ip": client_ip,
                "method": request.method,
                "path": str(request.url.path),
                "user_agent": request.headers.get("User-Agent", "unknown"),
                "violation_type": violation.violation_type.value,
                "severity": violation.severity.value,
                "rule": violation.rule_name,
                "message": violation.message,
                "timestamp": time.time()
            }
            
            # Use appropriate log level based on severity
            if violation.severity.value == "critical":
                logger.critical(f"WAF Critical Violation: {log_data}")
            elif violation.severity.value == "high":
                logger.error(f"WAF High Severity Violation: {log_data}")
            elif violation.severity.value == "medium":
                logger.warning(f"WAF Medium Severity Violation: {log_data}")
            else:
                logger.info(f"WAF Low Severity Violation: {log_data}")
    
    def _create_blocked_response(self, violations: list[WAFViolation], client_ip: str) -> JSONResponse:
        """Create response for blocked requests."""
        # Determine response status based on violation severity
        max_severity = max(violation.severity.value for violation in violations)
        
        if max_severity == "critical":
            status_code = 403
            message = "Request blocked due to critical security violation"
        elif max_severity == "high":
            status_code = 403
            message = "Request blocked due to high severity security violation"
        else:
            status_code = 400
            message = "Request blocked due to security violation"
        
        response_data = {
            "error": "Request Blocked",
            "message": message,
            "blocked_at": time.time(),
            "violations": [
                {
                    "type": v.violation_type.value,
                    "severity": v.severity.value,
                    "message": v.message
                }
                for v in violations
            ]
        }
        
        # Add security headers
        headers = {
            "X-WAF-Blocked": "true",
            "X-WAF-Violations": str(len(violations)),
            "X-WAF-Severity": max_severity
        }
        
        return JSONResponse(
            status_code=status_code,
            content=response_data,
            headers=headers
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get WAF statistics."""
        return {
            "total_requests": self._stats["total_requests"],
            "blocked_requests": self._stats["blocked_requests"],
            "block_rate": (
                self._stats["blocked_requests"] / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
            "violations_by_type": self._stats["violations_by_type"].copy(),
            "violations_by_severity": self._stats["violations_by_severity"].copy(),
            "top_attacking_ips": dict(
                sorted(
                    self._stats["top_attacking_ips"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ),
            "blocked_ips": list(self._waf_engine.get_blocked_ips()),
            "suspicious_ips": self._waf_engine.get_suspicious_ips()
        }
    
    def unblock_ip(self, client_ip: str) -> bool:
        """Unblock an IP address."""
        return self._waf_engine.unblock_ip(client_ip)
