"""Database health checking functionality following SRP.

This module provides ONLY database health checking functionality.
Single Responsibility: Check health status of database connections.
"""

import logging
from typing import Dict, Any, List
from .connection_pool import ConnectionPool

logger = logging.getLogger(__name__)


class DatabaseHealthChecker:
    """Checks health of database connection pools.

    Single Responsibility: Monitor database connection health.
    Does NOT handle: connection management, migrations, service coordination.
    """

    def __init__(self, pools: Dict[str, ConnectionPool]):
        self.pools = pools

    async def check_pool_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific connection pool."""
        if service_name not in self.pools:
            return {
                "service": service_name,
                "status": "not_configured",
                "error": "No connection pool configured",
            }

        pool = self.pools[service_name]

        try:
            async with pool.get_connection() as conn:
                await conn.fetchval("SELECT 1")

            stats = pool.get_pool_stats()
            return {"service": service_name, "status": "healthy", **stats}

        except Exception as e:
            return {"service": service_name, "status": "unhealthy", "error": str(e)}

    async def check_all_pools(self) -> Dict[str, Any]:
        """Check health of all configured connection pools."""
        health_status = {}

        for service_name in self.pools.keys():
            health_status[service_name] = await self.check_pool_health(service_name)

        return health_status

    def get_healthy_services(self) -> List[str]:
        """Get list of services with healthy database connections."""
        # This would need to be implemented with cached health status
        # For now, return empty list as this requires async operation
        return []

    def get_unhealthy_services(self) -> List[str]:
        """Get list of services with unhealthy database connections."""
        # This would need to be implemented with cached health status
        # For now, return empty list as this requires async operation
        return []


# Export only the health checking functionality
__all__ = [
    "DatabaseHealthChecker",
]
