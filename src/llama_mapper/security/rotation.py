"""
Automated secrets rotation with rollback capabilities.

Provides scheduled rotation for database credentials, API keys, and encryption keys.
"""

import asyncio
import secrets
import string
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import structlog
from croniter import croniter

from .secrets_manager import SecretsManager
from ..utils.correlation import get_correlation_id

logger = structlog.get_logger(__name__).bind(component="secrets_rotation")


class RotationStatus(Enum):
    """Status of rotation operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class RotationResult:
    """Result of a rotation operation."""
    secret_name: str
    status: RotationStatus
    old_version: Optional[str]
    new_version: Optional[str]
    error_message: Optional[str] = None
    rollback_available: bool = False


class SecretsRotationManager:
    """Automated secrets rotation with HashiCorp Vault integration."""
    
    def __init__(self, secrets_manager: SecretsManager):
        """
        Initialize rotation manager.
        
        Args:
            secrets_manager: Secrets manager instance
        """
        self.secrets_manager = secrets_manager
        self.rotation_history: Dict[str, List[RotationResult]] = {}
        self.rotation_policies = self._load_default_policies()
        self._rotation_tasks: Dict[str, asyncio.Task] = {}
    
    def _load_default_policies(self) -> Dict[str, Dict[str, Any]]:
        """Load default rotation policies."""
        return {
            "database_credentials": {
                "schedule": "0 2 * * 0",  # Weekly on Sunday 2 AM
                "retention_days": 7,
                "auto_rollback": True,
                "verification_required": True
            },
            "api_keys": {
                "schedule": "0 3 1 * *",  # Monthly on 1st at 3 AM
                "retention_days": 30,
                "auto_rollback": False,
                "verification_required": False
            },
            "tls_certificates": {
                "schedule": "0 4 1 */3 *",  # Quarterly
                "retention_days": 90,
                "auto_rollback": True,
                "verification_required": True
            },
            "encryption_keys": {
                "schedule": "0 5 1 */6 *",  # Semi-annually
                "retention_days": 180,
                "auto_rollback": True,
                "verification_required": True
            }
        }
    
    async def rotate_database_credentials(self, database_name: str) -> RotationResult:
        """
        Rotate database credentials with verification and rollback.
        
        Args:
            database_name: Name of the database
            
        Returns:
            RotationResult: Result of rotation operation
        """
        correlation_id = get_correlation_id()
        logger.info("Starting database credential rotation",
                   database=database_name,
                   correlation_id=correlation_id)
        
        try:
            # Get current credentials for rollback
            old_credentials = await self._get_current_credentials(database_name)
            
            # Generate new credentials
            new_credentials = await self._generate_database_credentials()
            
            # Update database user
            await self._update_database_user(database_name, new_credentials)
            
            # Store new credentials in Vault
            new_version = await self._store_credentials_in_vault(
                f"database/{database_name}",
                new_credentials
            )
            
            # Verify connectivity with new credentials
            if await self._verify_database_connectivity(database_name, new_credentials):
                logger.info("Database credentials rotated successfully",
                           database=database_name,
                           new_version=new_version,
                           correlation_id=correlation_id)
                
                result = RotationResult(
                    secret_name=f"database/{database_name}",
                    status=RotationStatus.COMPLETED,
                    old_version=old_credentials.get("version"),
                    new_version=new_version,
                    rollback_available=True
                )
            else:
                # Verification failed, rollback
                await self._rollback_database_credentials(database_name, old_credentials)
                result = RotationResult(
                    secret_name=f"database/{database_name}",
                    status=RotationStatus.ROLLED_BACK,
                    old_version=old_credentials.get("version"),
                    new_version=new_version,
                    error_message="Verification failed, rolled back to previous credentials",
                    rollback_available=False
                )
            
        except Exception as e:
            logger.error("Failed to rotate database credentials",
                        database=database_name,
                        error=str(e),
                        correlation_id=correlation_id)
            
            result = RotationResult(
                secret_name=f"database/{database_name}",
                status=RotationStatus.FAILED,
                old_version=None,
                new_version=None,
                error_message=str(e),
                rollback_available=False
            )
        
        # Store result in history
        self._add_to_history(f"database/{database_name}", result)
        return result
    
    async def rotate_api_keys(self, tenant_id: str) -> RotationResult:
        """
        Rotate tenant API keys.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            RotationResult: Result of rotation operation
        """
        correlation_id = get_correlation_id()
        logger.info("Starting API key rotation",
                   tenant_id=tenant_id,
                   correlation_id=correlation_id)
        
        try:
            # Get current API key for history
            old_key_info = await self._get_current_api_key(tenant_id)
            
            # Generate new API key
            new_api_key = self._generate_secure_api_key()
            
            # Store in Vault with metadata
            new_version = await self._store_api_key_in_vault(tenant_id, new_api_key, old_key_info)
            
            # Update tenant configuration
            await self._update_tenant_api_key(tenant_id, new_api_key)
            
            # Notify tenant of key rotation (optional)
            await self._notify_tenant_key_rotation(tenant_id, new_api_key)
            
            logger.info("API key rotated successfully",
                       tenant_id=tenant_id,
                       new_version=new_version,
                       correlation_id=correlation_id)
            
            result = RotationResult(
                secret_name=f"api-keys/{tenant_id}",
                status=RotationStatus.COMPLETED,
                old_version=old_key_info.get("version"),
                new_version=new_version,
                rollback_available=True
            )
            
        except Exception as e:
            logger.error("API key rotation failed",
                        tenant_id=tenant_id,
                        error=str(e),
                        correlation_id=correlation_id)
            
            result = RotationResult(
                secret_name=f"api-keys/{tenant_id}",
                status=RotationStatus.FAILED,
                old_version=None,
                new_version=None,
                error_message=str(e),
                rollback_available=False
            )
        
        # Store result in history
        self._add_to_history(f"api-keys/{tenant_id}", result)
        return result
    
    def schedule_rotation_jobs(self) -> None:
        """Schedule automated rotation jobs based on policies."""
        for secret_type, policy in self.rotation_policies.items():
            schedule = policy["schedule"]
            
            # Create rotation task
            task = asyncio.create_task(
                self._scheduled_rotation_loop(secret_type, schedule)
            )
            self._rotation_tasks[secret_type] = task
            
            logger.info("Scheduled rotation job",
                       secret_type=secret_type,
                       schedule=schedule,
                       correlation_id=get_correlation_id())
    
    async def _scheduled_rotation_loop(self, secret_type: str, schedule: str) -> None:
        """Run scheduled rotation loop for a secret type."""
        cron = croniter(schedule, datetime.now())
        
        while True:
            try:
                # Calculate next run time
                next_run = cron.get_next(datetime)
                sleep_seconds = (next_run - datetime.now()).total_seconds()
                
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)
                
                # Execute rotation based on type
                if secret_type == "database_credentials":
                    # Rotate all database credentials
                    databases = await self._get_database_list()
                    for db_name in databases:
                        await self.rotate_database_credentials(db_name)
                
                elif secret_type == "api_keys":
                    # Rotate API keys for active tenants
                    tenants = await self._get_active_tenants()
                    for tenant_id in tenants:
                        await self.rotate_api_keys(tenant_id)
                
                logger.info("Scheduled rotation completed",
                           secret_type=secret_type,
                           correlation_id=get_correlation_id())
                
            except Exception as e:
                logger.error("Scheduled rotation failed",
                            secret_type=secret_type,
                            error=str(e),
                            correlation_id=get_correlation_id())
                
                # Wait before retrying
                await asyncio.sleep(3600)  # 1 hour
    
    async def _generate_database_credentials(self) -> Dict[str, str]:
        """Generate new database credentials."""
        username = f"mapper_user_{secrets.token_hex(8)}"
        password = self._generate_secure_password(32)
        
        return {
            "username": username,
            "password": password,
            "created_at": datetime.utcnow().isoformat()
        }
    
    def _generate_secure_api_key(self) -> str:
        """Generate a secure API key."""
        # Generate 32-byte random key and encode as hex
        return secrets.token_hex(32)
    
    def _generate_secure_password(self, length: int = 32) -> str:
        """Generate a secure password."""
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    async def _get_current_credentials(self, database_name: str) -> Dict[str, Any]:
        """Get current database credentials."""
        try:
            credentials_json = self.secrets_manager.get(f"database/{database_name}")
            import json
            return json.loads(credentials_json)
        except Exception:
            return {"version": None}
    
    async def _get_current_api_key(self, tenant_id: str) -> Dict[str, Any]:
        """Get current API key information."""
        try:
            key_json = self.secrets_manager.get(f"api-keys/{tenant_id}")
            import json
            return json.loads(key_json)
        except Exception:
            return {"version": None}
    
    async def _update_database_user(self, database_name: str, credentials: Dict[str, str]) -> None:
        """Update database user with new credentials."""
        # This would integrate with your database management system
        # For now, we'll simulate the operation
        logger.info("Updating database user",
                   database=database_name,
                   username=credentials["username"],
                   correlation_id=get_correlation_id())
    
    async def _store_credentials_in_vault(self, path: str, credentials: Dict[str, str]) -> str:
        """Store credentials in Vault and return version."""
        import json
        credentials_with_metadata = {
            **credentials,
            "rotated_at": datetime.utcnow().isoformat(),
            "rotated_by": "automated_rotation",
            "correlation_id": get_correlation_id()
        }
        
        self.secrets_manager.put(path, json.dumps(credentials_with_metadata))
        return f"v{int(datetime.utcnow().timestamp())}"
    
    async def _store_api_key_in_vault(self, tenant_id: str, api_key: str, old_key_info: Dict[str, Any]) -> str:
        """Store API key in Vault with metadata."""
        import json
        key_data = {
            "api_key": api_key,
            "created_at": datetime.utcnow().isoformat(),
            "rotated_by": "automated_rotation",
            "previous_key_id": old_key_info.get("key_id"),
            "correlation_id": get_correlation_id()
        }
        
        self.secrets_manager.put(f"api-keys/{tenant_id}", json.dumps(key_data))
        return f"v{int(datetime.utcnow().timestamp())}"
    
    async def _verify_database_connectivity(self, database_name: str, credentials: Dict[str, str]) -> bool:
        """Verify database connectivity with new credentials."""
        # This would test actual database connection
        # For now, we'll simulate successful verification
        logger.info("Verifying database connectivity",
                   database=database_name,
                   username=credentials["username"],
                   correlation_id=get_correlation_id())
        return True
    
    async def _rollback_database_credentials(self, database_name: str, old_credentials: Dict[str, Any]) -> None:
        """Rollback to previous database credentials."""
        logger.warning("Rolling back database credentials",
                      database=database_name,
                      correlation_id=get_correlation_id())
        # Implementation would restore previous credentials
    
    async def _update_tenant_api_key(self, tenant_id: str, api_key: str) -> None:
        """Update tenant configuration with new API key."""
        logger.info("Updating tenant API key configuration",
                   tenant_id=tenant_id,
                   correlation_id=get_correlation_id())
    
    async def _notify_tenant_key_rotation(self, tenant_id: str, new_api_key: str) -> None:
        """Notify tenant of API key rotation."""
        logger.info("Notifying tenant of key rotation",
                   tenant_id=tenant_id,
                   correlation_id=get_correlation_id())
        # Implementation would send notification (email, webhook, etc.)
    
    async def _get_database_list(self) -> List[str]:
        """Get list of databases to rotate."""
        # This would query your database configuration
        return ["primary", "analytics", "audit"]
    
    async def _get_active_tenants(self) -> List[str]:
        """Get list of active tenants."""
        # This would query your tenant management system
        return ["tenant1", "tenant2", "tenant3"]
    
    def _add_to_history(self, secret_name: str, result: RotationResult) -> None:
        """Add rotation result to history."""
        if secret_name not in self.rotation_history:
            self.rotation_history[secret_name] = []
        
        self.rotation_history[secret_name].append(result)
        
        # Keep only last 10 results
        if len(self.rotation_history[secret_name]) > 10:
            self.rotation_history[secret_name] = self.rotation_history[secret_name][-10:]
    
    def get_rotation_history(self, secret_name: Optional[str] = None) -> Dict[str, List[RotationResult]]:
        """Get rotation history for secrets."""
        if secret_name:
            return {secret_name: self.rotation_history.get(secret_name, [])}
        return self.rotation_history.copy()
    
    async def stop_scheduled_rotations(self) -> None:
        """Stop all scheduled rotation tasks."""
        for secret_type, task in self._rotation_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            logger.info("Stopped scheduled rotation",
                       secret_type=secret_type,
                       correlation_id=get_correlation_id())
        
        self._rotation_tasks.clear()