"""
Storage backend implementation for weekly evaluation data.

This module provides concrete implementations of storage backends
for persisting evaluation schedules, results, and reports.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..domain.interfaces import IStorageBackend

logger = logging.getLogger(__name__)


class FileStorageBackend(IStorageBackend):
    """
    File-based storage backend for development and testing.
    
    Stores evaluation data in JSON files on the local filesystem.
    Not suitable for production use.
    """
    
    def __init__(self, storage_dir: str = "/tmp/llama_mapper_evaluations"):
        """
        Initialize file storage backend.
        
        Args:
            storage_dir: Directory to store evaluation data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.schedules_file = self.storage_dir / "schedules.json"
        self.results_file = self.storage_dir / "results.json"
        self.reports_dir = self.storage_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized file storage backend at {storage_dir}")
    
    async def store_evaluation_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Store evaluation schedule."""
        try:
            schedules = await self._load_schedules()
            schedules[schedule_data["schedule_id"]] = schedule_data
            await self._save_schedules(schedules)
            
            logger.debug(f"Stored evaluation schedule {schedule_data['schedule_id']}")
            
        except Exception as e:
            logger.error(f"Failed to store evaluation schedule: {e}")
            raise
    
    async def update_evaluation_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Update evaluation schedule."""
        try:
            schedules = await self._load_schedules()
            schedule_id = schedule_data["schedule_id"]
            
            if schedule_id not in schedules:
                raise ValueError(f"Schedule {schedule_id} not found")
            
            schedules[schedule_id] = schedule_data
            await self._save_schedules(schedules)
            
            logger.debug(f"Updated evaluation schedule {schedule_id}")
            
        except Exception as e:
            logger.error(f"Failed to update evaluation schedule: {e}")
            raise
    
    async def store_evaluation_result(self, result_data: Dict[str, Any]) -> None:
        """Store evaluation result."""
        try:
            results = await self._load_results()
            result_id = f"{result_data['schedule_id']}_{result_data['evaluation_date']}"
            results[result_id] = result_data
            await self._save_results(results)
            
            logger.debug(f"Stored evaluation result {result_id}")
            
        except Exception as e:
            logger.error(f"Failed to store evaluation result: {e}")
            raise
    
    async def save_evaluation_report(
        self, tenant_id: str, content: bytes, report_type: str
    ) -> str:
        """Save evaluation report."""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{tenant_id}_{report_type}_{timestamp}.pdf"
            file_path = self.reports_dir / filename
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            logger.debug(f"Saved evaluation report to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {e}")
            raise
    
    async def get_evaluation_schedules(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get evaluation schedules."""
        try:
            schedules = await self._load_schedules()
            
            if tenant_id:
                return [
                    schedule for schedule in schedules.values()
                    if schedule["tenant_id"] == tenant_id
                ]
            
            return list(schedules.values())
            
        except Exception as e:
            logger.error(f"Failed to get evaluation schedules: {e}")
            raise
    
    async def _load_schedules(self) -> Dict[str, Dict[str, Any]]:
        """Load schedules from file."""
        if not self.schedules_file.exists():
            return {}
        
        try:
            with open(self.schedules_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load schedules file: {e}")
            return {}
    
    async def _save_schedules(self, schedules: Dict[str, Dict[str, Any]]) -> None:
        """Save schedules to file."""
        try:
            with open(self.schedules_file, "w") as f:
                json.dump(schedules, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save schedules file: {e}")
            raise
    
    async def _load_results(self) -> Dict[str, Dict[str, Any]]:
        """Load results from file."""
        if not self.results_file.exists():
            return {}
        
        try:
            with open(self.results_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load results file: {e}")
            return {}
    
    async def _save_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Save results to file."""
        try:
            with open(self.results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save results file: {e}")
            raise


class DatabaseStorageBackend(IStorageBackend):
    """
    Database storage backend for production use.
    
    Stores evaluation data in a relational database.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize database storage backend.
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        logger.info("Initialized database storage backend")
    
    async def store_evaluation_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Store evaluation schedule."""
        # TODO: Implement database storage
        # This would use asyncpg or similar to store in PostgreSQL
        logger.warning("Database storage not yet implemented")
        raise NotImplementedError("Database storage not yet implemented")
    
    async def update_evaluation_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Update evaluation schedule."""
        # TODO: Implement database storage
        raise NotImplementedError("Database storage not yet implemented")
    
    async def store_evaluation_result(self, result_data: Dict[str, Any]) -> None:
        """Store evaluation result."""
        # TODO: Implement database storage
        raise NotImplementedError("Database storage not yet implemented")
    
    async def save_evaluation_report(
        self, tenant_id: str, content: bytes, report_type: str
    ) -> str:
        """Save evaluation report."""
        # TODO: Implement database storage with S3 integration
        raise NotImplementedError("Database storage not yet implemented")
    
    async def get_evaluation_schedules(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get evaluation schedules."""
        # TODO: Implement database storage
        raise NotImplementedError("Database storage not yet implemented")


class S3StorageBackend(IStorageBackend):
    """
    S3 storage backend for cloud deployments.
    
    Stores evaluation data in S3 with metadata in a database.
    """
    
    def __init__(self, s3_bucket: str, database_url: str):
        """
        Initialize S3 storage backend.
        
        Args:
            s3_bucket: S3 bucket name for report storage
            database_url: Database URL for metadata storage
        """
        self.s3_bucket = s3_bucket
        self.database_url = database_url
        logger.info(f"Initialized S3 storage backend with bucket {s3_bucket}")
    
    async def store_evaluation_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Store evaluation schedule."""
        # TODO: Implement S3 + database storage
        raise NotImplementedError("S3 storage not yet implemented")
    
    async def update_evaluation_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Update evaluation schedule."""
        # TODO: Implement S3 + database storage
        raise NotImplementedError("S3 storage not yet implemented")
    
    async def store_evaluation_result(self, result_data: Dict[str, Any]) -> None:
        """Store evaluation result."""
        # TODO: Implement S3 + database storage
        raise NotImplementedError("S3 storage not yet implemented")
    
    async def save_evaluation_report(
        self, tenant_id: str, content: bytes, report_type: str
    ) -> str:
        """Save evaluation report."""
        # TODO: Implement S3 storage
        raise NotImplementedError("S3 storage not yet implemented")
    
    async def get_evaluation_schedules(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get evaluation schedules."""
        # TODO: Implement S3 + database storage
        raise NotImplementedError("S3 storage not yet implemented")


def create_storage_backend(backend_type: str, **kwargs) -> IStorageBackend:
    """
    Factory function to create storage backend instances.
    
    Args:
        backend_type: Type of storage backend ('file', 'database', 's3')
        **kwargs: Backend-specific configuration
        
    Returns:
        Storage backend instance
        
    Raises:
        ValueError: If backend_type is not supported
    """
    if backend_type == "file":
        return FileStorageBackend(kwargs.get("storage_dir", "/tmp/llama_mapper_evaluations"))
    elif backend_type == "database":
        database_url = kwargs.get("database_url")
        if not database_url:
            raise ValueError("database_url is required for database backend")
        return DatabaseStorageBackend(database_url)
    elif backend_type == "s3":
        s3_bucket = kwargs.get("s3_bucket")
        database_url = kwargs.get("database_url")
        if not s3_bucket or not database_url:
            raise ValueError("s3_bucket and database_url are required for S3 backend")
        return S3StorageBackend(s3_bucket, database_url)
    else:
        raise ValueError(f"Unsupported storage backend type: {backend_type}")
