"""
WAF Middleware

This module provides FastAPI middleware that integrates WAF rules
for request filtering and abuse prevention.
"""

from .waf_middleware import WAFMiddleware

__all__ = [
    "WAFMiddleware",
]
