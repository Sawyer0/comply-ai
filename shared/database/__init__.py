"""Database utilities and connection management following SRP."""

# Main coordination layer
from .connection_manager import (
    DatabaseConnectionManager,
    db_manager,
    initialize_databases,
    get_service_connection,
    close_all_databases,
    ServiceDatabase,
    get_service_db,
)

# SRP components
from .connection_pool import DatabaseConfig, ConnectionPool
from .service_registry import DatabaseServiceRegistry
from .health_checker import DatabaseHealthChecker

__all__ = [
    # Main coordination
    "DatabaseConnectionManager",
    "db_manager",
    "initialize_databases",
    "get_service_connection",
    "close_all_databases",
    "ServiceDatabase",
    "get_service_db",
    # SRP components
    "DatabaseConfig",
    "ConnectionPool",
    "DatabaseServiceRegistry",
    "DatabaseHealthChecker",
]
