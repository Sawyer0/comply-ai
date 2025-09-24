"""
Storage backend implementation for weekly evaluation data.

This module provides concrete implementations of storage backends
for persisting evaluation schedules, results, and reports.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import uuid

try:
    import asyncpg
    if TYPE_CHECKING:
        from asyncpg import Pool
    else:
        Pool = asyncpg.Pool
except ImportError:
    asyncpg = None
    Pool = None

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception

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

        logger.info("Initialized file storage backend at %s", storage_dir)

    async def store_evaluation_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Store evaluation schedule."""
        try:
            schedules = await self._load_schedules()
            schedules[schedule_data["schedule_id"]] = schedule_data
            await self._save_schedules(schedules)

            logger.debug("Stored evaluation schedule %s", schedule_data["schedule_id"])

        except Exception as e:
            logger.error("Failed to store evaluation schedule: %s", e)
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

            logger.debug("Updated evaluation schedule %s", schedule_id)

        except Exception as e:
            logger.error("Failed to update evaluation schedule: %s", e)
            raise

    async def store_evaluation_result(self, result_data: Dict[str, Any]) -> None:
        """Store evaluation result."""
        try:
            results = await self._load_results()
            result_id = f"{result_data['schedule_id']}_{result_data['evaluation_date']}"
            results[result_id] = result_data
            await self._save_results(results)

            logger.debug("Stored evaluation result %s", result_id)

        except Exception as e:
            logger.error("Failed to store evaluation result: %s", e)
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

            logger.debug("Saved evaluation report to %s", file_path)
            return str(file_path)

        except Exception as e:
            logger.error("Failed to save evaluation report: %s", e)
            raise

    async def get_evaluation_schedules(
        self, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get evaluation schedules."""
        try:
            schedules = await self._load_schedules()

            if tenant_id:
                return [
                    schedule
                    for schedule in schedules.values()
                    if schedule["tenant_id"] == tenant_id
                ]

            return list(schedules.values())

        except Exception as e:
            logger.error("Failed to get evaluation schedules: %s", e)
            raise

    async def _load_schedules(self) -> Dict[str, Dict[str, Any]]:
        """Load schedules from file."""
        if not self.schedules_file.exists():
            return {}

        try:
            with open(self.schedules_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load schedules file: %s", e)
            return {}

    async def _save_schedules(self, schedules: Dict[str, Dict[str, Any]]) -> None:
        """Save schedules to file."""
        try:
            with open(self.schedules_file, "w") as f:
                json.dump(schedules, f, indent=2, default=str)
        except IOError as e:
            logger.error("Failed to save schedules file: %s", e)
            raise

    async def _load_results(self) -> Dict[str, Dict[str, Any]]:
        """Load results from file."""
        if not self.results_file.exists():
            return {}

        try:
            with open(self.results_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load results file: %s", e)
            return {}

    async def _save_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Save results to file."""
        try:
            with open(self.results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
        except IOError as e:
            logger.error("Failed to save results file: %s", e)
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
        if asyncpg is None:
            raise ImportError("asyncpg is required for DatabaseStorageBackend")
        
        self.database_url = database_url
        self._pool: Optional[Any] = None
        logger.info("Initialized database storage backend")

    async def _get_db_pool(self) -> Any:
        """Get database connection pool."""
        if self._pool is None:
            if asyncpg:
                self._pool = await asyncpg.create_pool(self.database_url)
            else:
                raise ImportError("asyncpg is required but not available")
            await self._ensure_tables()
        return self._pool

    async def _ensure_tables(self) -> None:
        """Ensure required database tables exist."""
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS evaluation_schedules (
            schedule_id VARCHAR(255) PRIMARY KEY,
            tenant_id VARCHAR(255),
            schedule_data JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS evaluation_results (
            result_id VARCHAR(255) PRIMARY KEY,
            schedule_id VARCHAR(255) REFERENCES evaluation_schedules(schedule_id),
            evaluation_date DATE,
            result_data JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS evaluation_reports (
            report_id VARCHAR(255) PRIMARY KEY,
            tenant_id VARCHAR(255),
            report_type VARCHAR(100),
            file_path TEXT,
            content_size INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_schedules_tenant ON evaluation_schedules(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_results_schedule ON evaluation_results(schedule_id);
        CREATE INDEX IF NOT EXISTS idx_reports_tenant ON evaluation_reports(tenant_id);
        """
        
        if self._pool:
            async with self._pool.acquire() as conn:
                await conn.execute(create_tables_sql)

    async def store_evaluation_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Store evaluation schedule."""
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO evaluation_schedules (schedule_id, tenant_id, schedule_data)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (schedule_id) DO UPDATE SET
                        schedule_data = $3,
                        updated_at = NOW()
                    """,
                    schedule_data["schedule_id"],
                    schedule_data.get("tenant_id"),
                    json.dumps(schedule_data)
                )
            
            logger.debug("Stored evaluation schedule %s", schedule_data["schedule_id"])

        except Exception as e:
            logger.error("Failed to store evaluation schedule: %s", e)
            raise

    async def update_evaluation_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Update evaluation schedule."""
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE evaluation_schedules 
                    SET schedule_data = $2, updated_at = NOW()
                    WHERE schedule_id = $1
                    """,
                    schedule_data["schedule_id"],
                    json.dumps(schedule_data)
                )
                
                if result == "UPDATE 0":
                    raise ValueError(f"Schedule {schedule_data['schedule_id']} not found")
            
            logger.debug("Updated evaluation schedule %s", schedule_data["schedule_id"])

        except Exception as e:
            logger.error("Failed to update evaluation schedule: %s", e)
            raise

    async def store_evaluation_result(self, result_data: Dict[str, Any]) -> None:
        """Store evaluation result."""
        try:
            result_id = f"{result_data['schedule_id']}_{result_data['evaluation_date']}"
            
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO evaluation_results (result_id, schedule_id, evaluation_date, result_data)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (result_id) DO UPDATE SET
                        result_data = $4
                    """,
                    result_id,
                    result_data["schedule_id"],
                    result_data["evaluation_date"],
                    json.dumps(result_data)
                )
            
            logger.debug("Stored evaluation result %s", result_id)

        except Exception as e:
            logger.error("Failed to store evaluation result: %s", e)
            raise

    async def save_evaluation_report(
        self, tenant_id: str, content: bytes, report_type: str
    ) -> str:
        """Save evaluation report."""
        try:
            # Store content to local file for database backend
            # In production, this could be enhanced to use S3
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_id = str(uuid.uuid4())
            filename = f"{tenant_id}_{report_type}_{timestamp}_{report_id}.pdf"
            
            # Create reports directory
            reports_dir = Path("/tmp/llama_mapper_reports")
            reports_dir.mkdir(exist_ok=True)
            
            file_path = reports_dir / filename
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Store metadata in database
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO evaluation_reports (report_id, tenant_id, report_type, file_path, content_size)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    report_id,
                    tenant_id,
                    report_type,
                    str(file_path),
                    len(content)
                )
            
            logger.debug("Saved evaluation report %s", report_id)
            return str(file_path)

        except Exception as e:
            logger.error("Failed to save evaluation report: %s", e)
            raise

    async def get_evaluation_schedules(
        self, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get evaluation schedules."""
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                if tenant_id:
                    rows = await conn.fetch(
                        """
                        SELECT schedule_data FROM evaluation_schedules 
                        WHERE tenant_id = $1 OR tenant_id IS NULL
                        ORDER BY created_at DESC
                        """,
                        tenant_id
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT schedule_data FROM evaluation_schedules 
                        ORDER BY created_at DESC
                        """
                    )
                
                schedules = [json.loads(row["schedule_data"]) for row in rows]
                return schedules

        except Exception as e:
            logger.error("Failed to get evaluation schedules: %s", e)
            raise


class S3StorageBackend(IStorageBackend):
    """
    S3 storage backend for cloud deployments.

    Stores evaluation data in S3 with metadata in a database.
    """

    def __init__(self, s3_bucket: str, database_url: str, aws_access_key_id: Optional[str] = None, aws_secret_access_key: Optional[str] = None):
        """
        Initialize S3 storage backend.

        Args:
            s3_bucket: S3 bucket name for report storage
            database_url: Database URL for metadata storage
            aws_access_key_id: AWS access key (optional, can use IAM roles)
            aws_secret_access_key: AWS secret key (optional, can use IAM roles)
        """
        if boto3 is None:
            raise ImportError("boto3 is required for S3StorageBackend")
        if asyncpg is None:
            raise ImportError("asyncpg is required for S3StorageBackend")
        
        self.s3_bucket = s3_bucket
        self.database_url = database_url
        self._pool: Optional[Any] = None
        
        # Initialize S3 client
        s3_config = {}
        if aws_access_key_id and aws_secret_access_key:
            s3_config.update({
                'aws_access_key_id': aws_access_key_id,
                'aws_secret_access_key': aws_secret_access_key
            })
        
        self.s3_client = boto3.client('s3', **s3_config)
        logger.info("Initialized S3 storage backend with bucket %s", s3_bucket)

    async def _get_db_pool(self) -> Any:
        """Get database connection pool."""
        if self._pool is None:
            if asyncpg:
                self._pool = await asyncpg.create_pool(self.database_url)
            else:
                raise ImportError("asyncpg is required but not available")
            await self._ensure_tables()
        return self._pool

    async def _ensure_tables(self) -> None:
        """Ensure required database tables exist."""
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS evaluation_schedules (
            schedule_id VARCHAR(255) PRIMARY KEY,
            tenant_id VARCHAR(255),
            schedule_data JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS evaluation_results (
            result_id VARCHAR(255) PRIMARY KEY,
            schedule_id VARCHAR(255) REFERENCES evaluation_schedules(schedule_id),
            evaluation_date DATE,
            result_data JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS evaluation_reports (
            report_id VARCHAR(255) PRIMARY KEY,
            tenant_id VARCHAR(255),
            report_type VARCHAR(100),
            s3_key TEXT,
            content_size INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_schedules_tenant ON evaluation_schedules(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_results_schedule ON evaluation_results(schedule_id);
        CREATE INDEX IF NOT EXISTS idx_reports_tenant ON evaluation_reports(tenant_id);
        """
        
        if self._pool:
            async with self._pool.acquire() as conn:
                await conn.execute(create_tables_sql)
        else:
            raise RuntimeError("Database pool not initialized")

    async def store_evaluation_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Store evaluation schedule."""
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO evaluation_schedules (schedule_id, tenant_id, schedule_data)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (schedule_id) DO UPDATE SET
                        schedule_data = $3,
                        updated_at = NOW()
                    """,
                    schedule_data["schedule_id"],
                    schedule_data.get("tenant_id"),
                    json.dumps(schedule_data)
                )
            
            logger.debug("Stored evaluation schedule %s", schedule_data["schedule_id"])

        except Exception as e:
            logger.error("Failed to store evaluation schedule: %s", e)
            raise

    async def update_evaluation_schedule(self, schedule_data: Dict[str, Any]) -> None:
        """Update evaluation schedule."""
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE evaluation_schedules 
                    SET schedule_data = $2, updated_at = NOW()
                    WHERE schedule_id = $1
                    """,
                    schedule_data["schedule_id"],
                    json.dumps(schedule_data)
                )
                
                if result == "UPDATE 0":
                    raise ValueError(f"Schedule {schedule_data['schedule_id']} not found")
            
            logger.debug("Updated evaluation schedule %s", schedule_data["schedule_id"])

        except Exception as e:
            logger.error("Failed to update evaluation schedule: %s", e)
            raise

    async def store_evaluation_result(self, result_data: Dict[str, Any]) -> None:
        """Store evaluation result."""
        try:
            result_id = f"{result_data['schedule_id']}_{result_data['evaluation_date']}"
            
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO evaluation_results (result_id, schedule_id, evaluation_date, result_data)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (result_id) DO UPDATE SET
                        result_data = $4
                    """,
                    result_id,
                    result_data["schedule_id"],
                    result_data["evaluation_date"],
                    json.dumps(result_data)
                )
            
            logger.debug("Stored evaluation result %s", result_id)

        except Exception as e:
            logger.error("Failed to store evaluation result: %s", e)
            raise

    async def save_evaluation_report(
        self, tenant_id: str, content: bytes, report_type: str
    ) -> str:
        """Save evaluation report."""
        try:
            # Generate S3 key
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_id = str(uuid.uuid4())
            s3_key = f"evaluation-reports/{tenant_id}/{report_type}/{timestamp}_{report_id}.pdf"
            
            # Upload to S3
            try:
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=content,
                    ContentType='application/pdf',
                    Metadata={
                        'tenant_id': tenant_id,
                        'report_type': report_type,
                        'report_id': report_id
                    }
                )
            except ClientError as e:
                logger.error("Failed to upload report to S3: %s", e)
                raise
            
            # Store metadata in database
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO evaluation_reports (report_id, tenant_id, report_type, s3_key, content_size)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    report_id,
                    tenant_id,
                    report_type,
                    s3_key,
                    len(content)
                )
            
            logger.debug("Saved evaluation report %s to S3", report_id)
            return f"s3://{self.s3_bucket}/{s3_key}"

        except Exception as e:
            logger.error("Failed to save evaluation report: %s", e)
            raise

    async def get_evaluation_schedules(
        self, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get evaluation schedules."""
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                if tenant_id:
                    rows = await conn.fetch(
                        """
                        SELECT schedule_data FROM evaluation_schedules 
                        WHERE tenant_id = $1 OR tenant_id IS NULL
                        ORDER BY created_at DESC
                        """,
                        tenant_id
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT schedule_data FROM evaluation_schedules 
                        ORDER BY created_at DESC
                        """
                    )
                
                schedules = [json.loads(row["schedule_data"]) for row in rows]
                return schedules

        except Exception as e:
            logger.error("Failed to get evaluation schedules: %s", e)
            raise

    async def get_report_download_url(self, report_id: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for downloading a report.
        
        Args:
            report_id: Report identifier
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL for downloading the report
        """
        try:
            pool = await self._get_db_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT s3_key FROM evaluation_reports WHERE report_id = $1",
                    report_id
                )
                
                if not row:
                    raise ValueError(f"Report {report_id} not found")
                
                s3_key = row["s3_key"]
                
                # Generate presigned URL
                url = self.s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': self.s3_bucket, 'Key': s3_key},
                    ExpiresIn=expiration
                )
                
                return url

        except Exception as e:
            logger.error("Failed to generate download URL for report %s: %s", report_id, e)
            raise


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
        return FileStorageBackend(
            kwargs.get("storage_dir", "/tmp/llama_mapper_evaluations")
        )
    elif backend_type == "database":
        database_url = kwargs.get("database_url")
        if not database_url:
            raise ValueError("database_url is required for database backend")
        return DatabaseStorageBackend(database_url)
    elif backend_type == "s3":
        s3_bucket = kwargs.get("s3_bucket")
        database_url = kwargs.get("database_url")
        aws_access_key_id = kwargs.get("aws_access_key_id")
        aws_secret_access_key = kwargs.get("aws_secret_access_key")
        if not s3_bucket or not database_url:
            raise ValueError("s3_bucket and database_url are required for S3 backend")
        return S3StorageBackend(s3_bucket, database_url, aws_access_key_id, aws_secret_access_key)
    else:
        raise ValueError(f"Unsupported storage backend type: {backend_type}")
