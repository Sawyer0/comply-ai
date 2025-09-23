#!/usr/bin/env python3
"""
Database Restore Script for Comply-AI Platform

This script provides comprehensive restore functionality for all platform databases
including PostgreSQL, ClickHouse, and Redis with proper error handling, logging,
and S3 integration. Can be used with Azure Blob Storage (S3-compatible) or AWS S3.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import psycopg2
import redis
from botocore.exceptions import ClientError, NoCredentialsError
from clickhouse_driver import Client as ClickHouseClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/restore.log')
    ]
)
logger = logging.getLogger(__name__)


class DatabaseRestoreError(Exception):
    """Custom exception for database restore errors."""
    pass


class S3RestoreManager:
    """Manages S3 restore operations (works with AWS S3 and Azure Blob Storage)."""
    
    def __init__(self, bucket: str, region: str = 'us-east-1', endpoint_url: Optional[str] = None):
        self.bucket = bucket
        self.region = region
        self.endpoint_url = endpoint_url
        
        # Initialize S3 client (works with AWS S3 or Azure Blob Storage)
        if endpoint_url:
            # Azure Blob Storage with S3-compatible API
            self.s3_client = boto3.client(
                's3',
                region_name=region,
                endpoint_url=endpoint_url,
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
        else:
            # AWS S3
            self.s3_client = boto3.client('s3', region_name=region)
    
    def download_backup(self, s3_key: str, local_path: str) -> bool:
        """Download backup file from S3 (AWS or Azure Blob Storage)."""
        try:
            self.s3_client.download_file(self.bucket, s3_key, local_path)
            logger.info(f"Successfully downloaded s3://{self.bucket}/{s3_key} to {local_path}")
            return True
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to download {s3_key} from S3: {e}")
            return False
    
    def download_backup_directory(self, s3_prefix: str, local_path: str) -> bool:
        """Download backup directory from S3."""
        try:
            # List all objects with the prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=s3_prefix
            )
            
            if 'Contents' not in response:
                logger.error(f"No objects found with prefix: {s3_prefix}")
                return False
            
            # Create local directory
            local_dir = Path(local_path)
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Download each object
            for obj in response['Contents']:
                s3_key = obj['Key']
                relative_path = s3_key[len(s3_prefix):].lstrip('/')
                local_file = local_dir / relative_path
                
                # Create subdirectories if needed
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                self.s3_client.download_file(self.bucket, s3_key, str(local_file))
                logger.info(f"Downloaded {s3_key} to {local_file}")
            
            return True
            
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to download directory {s3_prefix} from S3: {e}")
            return False
    
    def list_available_backups(self, prefix: str) -> List[Dict]:
        """List available backups in S3."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            return response.get('Contents', [])
        except ClientError as e:
            logger.error(f"Failed to list backups: {e}")
            return []


class PostgreSQLRestore:
    """PostgreSQL restore operations."""
    
    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
    
    def restore_backup(self, backup_path: str, drop_existing: bool = False) -> bool:
        """Restore PostgreSQL backup."""
        try:
            # Set password environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = self.password
            
            if drop_existing:
                # Drop and recreate database
                logger.info("Dropping existing database...")
                drop_cmd = [
                    'psql',
                    '-h', self.host,
                    '-p', str(self.port),
                    '-U', self.user,
                    '-d', 'postgres',
                    '-c', f'DROP DATABASE IF EXISTS {self.database};'
                ]
                subprocess.run(drop_cmd, env=env, check=True)
                
                create_cmd = [
                    'psql',
                    '-h', self.host,
                    '-p', str(self.port),
                    '-U', self.user,
                    '-d', 'postgres',
                    '-c', f'CREATE DATABASE {self.database};'
                ]
                subprocess.run(create_cmd, env=env, check=True)
            
            # Restore backup
            logger.info(f"Restoring PostgreSQL backup from {backup_path}")
            restore_cmd = [
                'pg_restore',
                '-h', self.host,
                '-p', str(self.port),
                '-U', self.user,
                '-d', self.database,
                '--verbose',
                '--no-password',
                '--clean',
                '--if-exists',
                backup_path
            ]
            
            result = subprocess.run(
                restore_cmd,
                stderr=subprocess.PIPE,
                env=env,
                check=True
            )
            
            logger.info("PostgreSQL restore completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"PostgreSQL restore failed: {e.stderr.decode()}")
            return False
        except Exception as e:
            logger.error(f"PostgreSQL restore error: {e}")
            return False


class ClickHouseRestore:
    """ClickHouse restore operations."""
    
    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.client = ClickHouseClient(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
    
    def restore_backup(self, backup_path: str, drop_existing: bool = False) -> bool:
        """Restore ClickHouse backup."""
        try:
            backup_dir = Path(backup_path)
            metadata_file = backup_dir / 'backup_metadata.json'
            
            if not metadata_file.exists():
                logger.error("Backup metadata file not found")
                return False
            
            # Load backup metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Restoring ClickHouse backup from {backup_path}")
            logger.info(f"Backup contains {len(metadata['tables'])} tables")
            
            # Restore each table
            for table_info in metadata['tables']:
                table_name = table_info['name']
                logger.info(f"Restoring table: {table_name}")
                
                if drop_existing:
                    # Drop existing table
                    try:
                        self.client.execute(f"DROP TABLE IF EXISTS {table_name}")
                    except Exception as e:
                        logger.warning(f"Could not drop table {table_name}: {e}")
                
                # Restore table schema
                schema_file = backup_dir / table_info['schema_file']
                if schema_file.exists():
                    with open(schema_file, 'r') as f:
                        create_sql = f.read()
                    
                    self.client.execute(create_sql)
                    logger.info(f"Restored schema for table {table_name}")
                
                # Restore table data
                data_file = backup_dir / table_info['data_file']
                if data_file.exists() and data_file.stat().st_size > 0:
                    # Read CSV data
                    with open(data_file, 'r') as f:
                        lines = f.readlines()
                    
                    if len(lines) > 1:  # Has header and data
                        # Get column names from header
                        columns = lines[0].strip().split(',')
                        
                        # Insert data
                        for line in lines[1:]:
                            values = line.strip().split(',')
                            if len(values) == len(columns):
                                placeholders = ', '.join(['%s'] * len(values))
                                insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                                self.client.execute(insert_sql, values)
                        
                        logger.info(f"Restored {len(lines)-1} rows to table {table_name}")
                    else:
                        logger.info(f"Table {table_name} has no data to restore")
            
            logger.info("ClickHouse restore completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"ClickHouse restore error: {e}")
            return False


class RedisRestore:
    """Redis restore operations."""
    
    def __init__(self, host: str, port: int, password: Optional[str] = None):
        self.host = host
        self.port = port
        self.password = password
        self.client = redis.Redis(
            host=host,
            port=port,
            password=password,
            decode_responses=False
        )
    
    def restore_backup(self, backup_path: str, flush_existing: bool = False) -> bool:
        """Restore Redis backup."""
        try:
            if flush_existing:
                logger.info("Flushing existing Redis data...")
                self.client.flushall()
            
            # Stop Redis to prevent data corruption
            logger.info("Stopping Redis...")
            self.client.shutdown()
            
            # Wait for Redis to stop
            import time
            time.sleep(5)
            
            # Copy backup file to Redis data directory
            # Note: This requires access to the Redis data directory
            # In a containerized environment, this would be handled differently
            redis_data_dir = "/var/lib/redis"
            dump_file = os.path.join(redis_data_dir, "dump.rdb")
            
            # Copy backup to Redis data directory
            import shutil
            shutil.copy2(backup_path, dump_file)
            
            # Start Redis (this would be handled by the container orchestration)
            logger.info("Redis backup file copied. Redis will be restarted by the container orchestration.")
            
            return True
            
        except Exception as e:
            logger.error(f"Redis restore error: {e}")
            return False


class DatabaseRestoreManager:
    """Main restore manager for all databases."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.s3_manager = S3RestoreManager(
            bucket=config['s3']['bucket'],
            region=config['s3']['region'],
            endpoint_url=config['s3'].get('endpoint_url')  # For Azure Blob Storage
        )
        
        # Initialize database connections
        self.postgres = PostgreSQLRestore(
            host=config['postgresql']['host'],
            port=config['postgresql']['port'],
            database=config['postgresql']['database'],
            user=config['postgresql']['user'],
            password=config['postgresql']['password']
        )
        
        self.clickhouse = ClickHouseRestore(
            host=config['clickhouse']['host'],
            port=config['clickhouse']['port'],
            database=config['clickhouse']['database'],
            user=config['clickhouse']['user'],
            password=config['clickhouse']['password']
        )
        
        self.redis = RedisRestore(
            host=config['redis']['host'],
            port=config['redis']['port'],
            password=config['redis'].get('password')
        )
    
    def list_available_backups(self, database_type: str) -> List[Dict]:
        """List available backups for a database type."""
        prefix = f"backups/llama-mapper/{database_type}/"
        backups = self.s3_manager.list_available_backups(prefix)
        
        # Filter and format backup information
        formatted_backups = []
        for backup in backups:
            key = backup['Key']
            if database_type == 'postgresql' and key.endswith('.dump'):
                timestamp = key.split('/')[-1].replace('.dump', '')
                formatted_backups.append({
                    'timestamp': timestamp,
                    's3_key': key,
                    'size': backup['Size'],
                    'last_modified': backup['LastModified']
                })
            elif database_type == 'clickhouse' and key.endswith('backup_metadata.json'):
                timestamp = key.split('/')[-2]
                formatted_backups.append({
                    'timestamp': timestamp,
                    's3_key': key.replace('/backup_metadata.json', '/'),
                    'size': backup['Size'],
                    'last_modified': backup['LastModified']
                })
            elif database_type == 'redis' and key.endswith('.rdb'):
                timestamp = key.split('/')[-1].replace('.rdb', '')
                formatted_backups.append({
                    'timestamp': timestamp,
                    's3_key': key,
                    'size': backup['Size'],
                    'last_modified': backup['LastModified']
                })
        
        # Sort by timestamp (newest first)
        formatted_backups.sort(key=lambda x: x['timestamp'], reverse=True)
        return formatted_backups
    
    def restore_backup(self, database_type: str, s3_key: str, drop_existing: bool = False) -> bool:
        """Restore backup for specified database type."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                if database_type == 'postgresql':
                    backup_file = temp_path / "restore.dump"
                    success = self.s3_manager.download_backup(s3_key, str(backup_file))
                    if not success:
                        return False
                    
                    return self.postgres.restore_backup(str(backup_file), drop_existing)
                    
                elif database_type == 'clickhouse':
                    backup_dir = temp_path / "restore"
                    success = self.s3_manager.download_backup_directory(s3_key, str(backup_dir))
                    if not success:
                        return False
                    
                    return self.clickhouse.restore_backup(str(backup_dir), drop_existing)
                    
                elif database_type == 'redis':
                    backup_file = temp_path / "restore.rdb"
                    success = self.s3_manager.download_backup(s3_key, str(backup_file))
                    if not success:
                        return False
                    
                    return self.redis.restore_backup(str(backup_file), drop_existing)
                    
                else:
                    raise ValueError(f"Unknown database type: {database_type}")
                
        except Exception as e:
            logger.error(f"Restore failed for {database_type}: {e}")
            return False


def load_config(config_path: str) -> Dict:
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Database Restore Script')
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--database', choices=['postgresql', 'clickhouse', 'redis'], 
                       required=True, help='Database to restore')
    parser.add_argument('--backup', help='Specific backup to restore (S3 key or timestamp)')
    parser.add_argument('--list', action='store_true', help='List available backups')
    parser.add_argument('--drop-existing', action='store_true', 
                       help='Drop existing data before restore')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create restore manager
    restore_manager = DatabaseRestoreManager(config)
    
    if args.list:
        # List available backups
        backups = restore_manager.list_available_backups(args.database)
        if not backups:
            logger.info(f"No backups found for {args.database}")
        else:
            logger.info(f"Available {args.database} backups:")
            for backup in backups:
                logger.info(f"  {backup['timestamp']} - {backup['size']} bytes - {backup['last_modified']}")
        return
    
    if not args.backup:
        logger.error("Backup specification required (use --backup or --list)")
        sys.exit(1)
    
    # Determine S3 key
    if args.backup.startswith('backups/'):
        # Full S3 key provided
        s3_key = args.backup
    else:
        # Timestamp provided, construct S3 key
        if args.database == 'postgresql':
            s3_key = f"backups/llama-mapper/postgresql/{args.backup}.dump"
        elif args.database == 'clickhouse':
            s3_key = f"backups/llama-mapper/clickhouse/{args.backup}/"
        elif args.database == 'redis':
            s3_key = f"backups/llama-mapper/redis/{args.backup}.rdb"
    
    if args.dry_run:
        logger.info(f"DRY RUN: Would restore {args.database} from {s3_key}")
        if args.drop_existing:
            logger.info("DRY RUN: Would drop existing data")
        return
    
    # Perform restore
    logger.info(f"Starting restore of {args.database} from {s3_key}")
    
    success = restore_manager.restore_backup(
        args.database,
        s3_key,
        args.drop_existing
    )
    
    if not success:
        logger.error(f"Restore failed for {args.database}")
        sys.exit(1)
    
    logger.info(f"Successfully restored {args.database}")


if __name__ == '__main__':
    main()
