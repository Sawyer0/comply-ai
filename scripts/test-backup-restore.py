#!/usr/bin/env python3
"""
Test script for database backup and restore functionality.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def test_backup_script():
    """Test the backup script functionality."""
    print("🧪 Testing Database Backup Script...")

    # Import the backup module first
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "backup_databases", project_root / "scripts" / "backup-databases.py"
    )
    backup_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backup_module)

    # Mock all external dependencies
    with (
        patch.object(backup_module, "boto3") as mock_boto3,
        patch.object(backup_module, "psycopg2") as mock_psycopg2,
        patch.object(backup_module, "redis") as mock_redis,
        patch.object(backup_module, "ClickHouseClient") as mock_clickhouse,
        patch.object(backup_module, "subprocess") as mock_subprocess,
        patch.object(backup_module, "os") as mock_os,
    ):

        # Mock S3 client
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        # Mock PostgreSQL connection
        mock_pg_conn = MagicMock()
        mock_pg_cursor = MagicMock()
        mock_pg_conn.cursor.return_value.__enter__.return_value = mock_pg_cursor
        mock_pg_cursor.fetchone.return_value = ["1.2 GB"]
        mock_pg_cursor.fetchall.return_value = [
            ("public", "storage_records", 1000, 500, 50),
            ("public", "audit_logs", 5000, 100, 10),
        ]
        mock_psycopg2.connect.return_value = mock_pg_conn

        # Mock Redis client
        mock_redis_client = MagicMock()
        mock_redis_client.info.return_value = {
            "redis_version": "6.2.0",
            "used_memory_human": "128.5M",
            "connected_clients": 5,
            "total_commands_processed": 10000,
            "keyspace": {"db0": {"keys": 100}},
            "uptime_in_seconds": 3600,
        }
        mock_redis_client.lastsave.return_value = 1234567890
        mock_redis.Redis.return_value = mock_redis_client

        # Mock ClickHouse client
        mock_ch_client = MagicMock()
        mock_ch_client.execute.side_effect = [
            [("analysis_metrics",), ("quality_evaluations",)],  # SHOW TABLES
            [("CREATE TABLE analysis_metrics...",)],  # SHOW CREATE TABLE
            [("timestamp", "String"), ("value", "Float64")],  # DESCRIBE TABLE
            [("2024-01-01", 0.95), ("2024-01-02", 0.92)],  # SELECT * FROM TABLE
        ]
        mock_clickhouse.return_value = mock_ch_client

        # Mock subprocess calls
        mock_subprocess.run.return_value = Mock(returncode=0)

        # Mock environment
        mock_os.environ.copy.return_value = {"PGPASSWORD": "test"}

        # Use the already imported backup module

        # Test S3BackupManager
        s3_manager = backup_module.S3BackupManager("test-bucket", "us-west-2")
        assert s3_manager.bucket == "test-bucket"
        assert s3_manager.region == "us-west-2"
        print("✅ S3BackupManager initialization works")

        # Test PostgreSQLBackup
        pg_backup = backup_module.PostgreSQLBackup(
            "localhost", 5432, "test_db", "test_user", "test_pass"
        )
        assert pg_backup.host == "localhost"
        assert pg_backup.database == "test_db"
        print("✅ PostgreSQLBackup initialization works")

        # Test ClickHouseBackup
        ch_backup = backup_module.ClickHouseBackup(
            "localhost", 9000, "test_db", "test_user", "test_pass"
        )
        assert ch_backup.host == "localhost"
        assert ch_backup.database == "test_db"
        print("✅ ClickHouseBackup initialization works")

        # Test RedisBackup
        redis_backup = backup_module.RedisBackup("localhost", 6379, "test_pass")
        assert redis_backup.host == "localhost"
        assert redis_backup.port == 6379
        print("✅ RedisBackup initialization works")

        # Test DatabaseBackupManager
        config = {
            "s3": {"bucket": "test-bucket", "region": "us-west-2"},
            "postgresql": {
                "host": "localhost",
                "port": 5432,
                "database": "test",
                "user": "test",
                "password": "test",
            },
            "clickhouse": {
                "host": "localhost",
                "port": 9000,
                "database": "test",
                "user": "test",
                "password": "test",
            },
            "redis": {"host": "localhost", "port": 6379, "password": "test"},
        }

        backup_manager = backup_module.DatabaseBackupManager(config)
        assert backup_manager.s3_manager.bucket == "test-bucket"
        print("✅ DatabaseBackupManager initialization works")

        print("🎉 All backup script tests passed!")


def test_restore_script():
    """Test the restore script functionality."""
    print("\n🧪 Testing Database Restore Script...")

    # Import the restore module first
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "restore_databases", project_root / "scripts" / "restore-databases.py"
    )
    restore_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(restore_module)

    # Mock all external dependencies
    with (
        patch.object(restore_module, "boto3") as mock_boto3,
        patch.object(restore_module, "psycopg2") as mock_psycopg2,
        patch.object(restore_module, "redis") as mock_redis,
        patch.object(restore_module, "ClickHouseClient") as mock_clickhouse,
        patch.object(restore_module, "subprocess") as mock_subprocess,
        patch.object(restore_module, "os") as mock_os,
        patch.object(restore_module, "shutil") as mock_shutil,
    ):

        # Mock S3 client
        mock_s3_client = MagicMock()
        mock_s3_client.download_file.return_value = None
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": "backups/llama-mapper/postgresql/2024-01-01-12-00-00.dump",
                    "Size": 1024000,
                    "LastModified": "2024-01-01T12:00:00Z",
                }
            ]
        }
        mock_boto3.client.return_value = mock_s3_client

        # Mock PostgreSQL connection
        mock_pg_conn = MagicMock()
        mock_pg_cursor = MagicMock()
        mock_pg_conn.cursor.return_value.__enter__.return_value = mock_pg_cursor
        mock_pg_cursor.fetchone.return_value = [1000]
        mock_psycopg2.connect.return_value = mock_pg_conn

        # Mock Redis client
        mock_redis_client = MagicMock()
        mock_redis_client.info.return_value = {
            "redis_version": "6.2.0",
            "used_memory_human": "128.5M",
            "connected_clients": 5,
            "keyspace": {"db0": {"keys": 100}},
        }
        mock_redis_client.ping.return_value = True
        mock_redis.Redis.return_value = mock_redis_client

        # Mock ClickHouse client
        mock_ch_client = MagicMock()
        mock_ch_client.execute.return_value = [
            ("analysis_metrics", "MergeTree", 1000, 1024000),
            ("quality_evaluations", "MergeTree", 500, 512000),
        ]
        mock_clickhouse.return_value = mock_ch_client

        # Mock subprocess calls
        mock_subprocess.run.return_value = Mock(returncode=0)

        # Mock environment
        mock_os.environ.copy.return_value = {"PGPASSWORD": "test"}

        # Use the already imported restore module

        # Test S3RestoreManager
        s3_manager = restore_module.S3RestoreManager("test-bucket", "us-west-2")
        assert s3_manager.bucket == "test-bucket"
        assert s3_manager.region == "us-west-2"
        print("✅ S3RestoreManager initialization works")

        # Test PostgreSQLRestore
        pg_restore = restore_module.PostgreSQLRestore(
            "localhost", 5432, "test_db", "test_user", "test_pass"
        )
        assert pg_restore.host == "localhost"
        assert pg_restore.database == "test_db"
        print("✅ PostgreSQLRestore initialization works")

        # Test ClickHouseRestore
        ch_restore = restore_module.ClickHouseRestore(
            "localhost", 9000, "test_db", "test_user", "test_pass"
        )
        assert ch_restore.host == "localhost"
        assert ch_restore.database == "test_db"
        print("✅ ClickHouseRestore initialization works")

        # Test RedisRestore
        redis_restore = restore_module.RedisRestore("localhost", 6379, "test_pass")
        assert redis_restore.host == "localhost"
        assert redis_restore.port == 6379
        print("✅ RedisRestore initialization works")

        # Test DatabaseRestoreManager
        config = {
            "s3": {"bucket": "test-bucket", "region": "us-west-2"},
            "postgresql": {
                "host": "localhost",
                "port": 5432,
                "database": "test",
                "user": "test",
                "password": "test",
            },
            "clickhouse": {
                "host": "localhost",
                "port": 9000,
                "database": "test",
                "user": "test",
                "password": "test",
            },
            "redis": {"host": "localhost", "port": 6379, "password": "test"},
        }

        restore_manager = restore_module.DatabaseRestoreManager(config)
        assert restore_manager.s3_manager.bucket == "test-bucket"
        print("✅ DatabaseRestoreManager initialization works")

        # Test listing backups
        backups = restore_manager.list_available_backups("postgresql")
        assert len(backups) > 0
        assert backups[0]["timestamp"] == "2024-01-01-12-00-00"
        print("✅ Backup listing works")

        print("🎉 All restore script tests passed!")


def test_config_loading():
    """Test configuration loading."""
    print("\n🧪 Testing Configuration Loading...")

    # Test config file exists
    config_path = project_root / "config" / "backup-config.json"
    assert config_path.exists(), "Backup config file should exist"
    print("✅ Backup config file exists")

    # Test config file is valid JSON
    with open(config_path, "r") as f:
        config = json.load(f)

    assert "s3" in config
    assert "postgresql" in config
    assert "clickhouse" in config
    assert "redis" in config
    print("✅ Backup config file is valid JSON")

    # Test required fields
    assert config["s3"]["bucket"] == "llama-mapper-backups"
    assert config["postgresql"]["database"] == "llama_mapper_analysis"
    assert config["clickhouse"]["database"] == "llama_mapper_analysis"
    print("✅ Backup config has required fields")

    print("🎉 All configuration tests passed!")


def test_script_help():
    """Test script help functionality."""
    print("\n🧪 Testing Script Help...")

    import subprocess

    # Test backup script help
    result = subprocess.run(
        [sys.executable, "scripts/backup-databases.py", "--help"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 0
    assert "Database Backup Script" in result.stdout
    print("✅ Backup script help works")

    # Test restore script help
    result = subprocess.run(
        [sys.executable, "scripts/restore-databases.py", "--help"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 0
    assert "Database Restore Script" in result.stdout
    print("✅ Restore script help works")

    print("🎉 All script help tests passed!")


def main():
    """Run all tests."""
    print("🚀 Starting Database Backup/Restore Tests...\n")

    try:
        # Test configuration
        test_config_loading()

        # Test backup script
        test_backup_script()

        # Test restore script
        test_restore_script()

        # Test script help
        test_script_help()

        print("\n🎉 All tests completed successfully!")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
