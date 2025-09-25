"""
FastAPI dependencies for the Analysis Service.

This module provides dependency injection for:
- Authentication and authorization
- Database connections
- Security managers
- Configuration
"""

from typing import Dict, Any
from fastapi import Depends, HTTPException, Header
from ..shared_integration import get_shared_database, get_shared_logger
from ..security import SecurityManager, SecurityConfig

logger = get_shared_logger(__name__)

# Global instances
_security_manager = None
_db_pool = None


def get_security_manager() -> SecurityManager:
    """Get security manager instance."""
    global _security_manager
    if _security_manager is None:
        config = SecurityConfig()
        _security_manager = SecurityManager(config)
    return _security_manager


def get_database():
    """Get database connection."""
    global _db_pool
    if _db_pool is None:
        _db_pool = get_shared_database()
    return _db_pool


async def authenticate_request(
    authorization: str = Header(None), x_api_key: str = Header(None)
) -> Dict[str, Any]:
    """Authenticate incoming request."""
    security_manager = get_security_manager()

    headers = {}
    if authorization:
        headers["Authorization"] = authorization
    if x_api_key:
        headers["X-API-Key"] = x_api_key

    try:
        return await security_manager.authenticate_request(headers)
    except Exception as e:
        logger.warning("Authentication failed", error=str(e))
        raise HTTPException(status_code=401, detail="Authentication failed")


async def authorize_action(
    action: str,
    resource: str,
    user_info: Dict[str, Any] = Depends(authenticate_request),
) -> Dict[str, Any]:
    """Authorize user action on resource."""
    security_manager = get_security_manager()

    try:
        authorized = await security_manager.authorize_action(
            user_info, action, resource
        )
        if not authorized:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authorization failed", error=str(e))
        raise HTTPException(status_code=500, detail="Authorization error")


async def shutdown_dependencies():
    """Cleanup dependencies on shutdown."""
    global _security_manager, _db_pool

    if _security_manager:
        await _security_manager.cleanup_audit_logs()

    # Database cleanup is handled by shared integration
    _security_manager = None
    _db_pool = None
