#!/usr/bin/env python3
"""
Azure Database Backup Script for Comply-AI Platform

This script provides comprehensive backup functionality for Azure managed services
including Azure Database for PostgreSQL, Azure Cache for Redis, and Azure Blob Storage
with proper error handling, logging, and Azure integration.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from azure.identity import DefaultAzureCredential
from azure.mgmt.postgresql import PostgreSQLManagementClient
from azure.mgmt.redis import RedisManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.storage.blob import BlobServiceClient
from azure.keyvault.secrets import SecretClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/azure-backup.log')
    ]
)
logger = logging.getLogger(__name__)


class AzureBackupError(Exception):
    """Custom exception for Azure backup errors."""
    pass


class AzurePostgreSQLBackup:
    """Azure Database for PostgreSQL backup operations."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.credential = DefaultAzureCredential()
        self.client = PostgreSQLManagementClient(
            credential=self.credential,
            subscription_id=config['azure']['subscription_id']
        )
        self.resource_group = config['azure']['resource_group']
        self.server_name = config['postgresql']['server_name']
    
    def create_manual_backup(self) -> bool:
        """Create manual backup using Azure CLI."""
        try:
            # Get admin password from Key Vault
            admin_password = self._get_secret('postgres-admin-password')
            
            # Create manual backup
            cmd = [
                'az', 'postgres', 'flexible-server', 'backup', 'create',
                '--resource-group', self.resource_group,
                '--server-name', self.server_name,
                '--backup-name', f"manual-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Manual backup created successfully: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create manual backup: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Manual backup error: {e}")
            return False
    
    def export_database(self, database_name: str, storage_uri: str) -> bool:
        """Export database to Azure Blob Storage."""
        try:
            # Get admin password from Key Vault
            admin_password = self._get_secret('postgres-admin-password')
            
            # Export database
            cmd = [
                'az', 'postgres', 'flexible-server', 'export',
                '--resource-group', self.resource_group,
                '--server-name', self.server_name,
                '--database-name', database_name,
                '--storage-uri', storage_uri,
                '--admin-user', self.config['postgresql']['admin_user'],
                '--admin-password', admin_password
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Database export completed: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to export database: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Database export error: {e}")
            return False
    
    def get_backup_status(self) -> Dict:
        """Get backup status and configuration."""
        try:
            # Get server details
            server = self.client.flexible_servers.get(
                resource_group_name=self.resource_group,
                server_name=self.server_name
            )
            
            # Get backup configuration
            backup_config = {
                'retention_days': server.backup.retention_days,
                'geo_redundant_backup': server.backup.geo_redundant_backup,
                'earliest_restore_date': server.backup.earliest_restore_date.isoformat() if server.backup.earliest_restore_date else None,
                'latest_restore_point': server.backup.latest_restore_point_time.isoformat() if server.backup.latest_restore_point_time else None
            }
            
            # List recent backups
            backups = list(self.client.backups.list_by_server(
                resource_group_name=self.resource_group,
                server_name=self.server_name
            ))
            
            recent_backups = []
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
            
            for backup in backups:
                if backup.created_time.replace(tzinfo=timezone.utc) > cutoff_date:
                    recent_backups.append({
                        'name': backup.name,
                        'created_time': backup.created_time.isoformat(),
                        'size': getattr(backup, 'size', None)
                    })
            
            return {
                'status': 'healthy',
                'backup_config': backup_config,
                'recent_backups': recent_backups,
                'backup_count': len(recent_backups)
            }
            
        except Exception as e:
            logger.error(f"Failed to get backup status: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'backup_count': 0
            }
    
    def _get_secret(self, secret_name: str) -> str:
        """Get secret from Azure Key Vault."""
        try:
            key_vault_url = self.config['azure']['key_vault_url']
            client = SecretClient(vault_url=key_vault_url, credential=self.credential)
            secret = client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            logger.error(f"Failed to get secret {secret_name}: {e}")
            raise


class AzureRedisBackup:
    """Azure Cache for Redis backup operations."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.credential = DefaultAzureCredential()
        self.client = RedisManagementClient(
            credential=self.credential,
            subscription_id=config['azure']['subscription_id']
        )
        self.resource_group = config['azure']['resource_group']
        self.redis_name = config['redis']['name']
    
    def configure_persistence(self) -> bool:
        """Configure Redis persistence settings."""
        try:
            # Configure RDB persistence
            redis_config = {
                'save': '900 1 300 10 60 10000',
                'maxmemory-policy': 'allkeys-lru'
            }
            
            # Update Redis configuration
            cmd = [
                'az', 'redis', 'update',
                '--resource-group', self.resource_group,
                '--name', self.redis_name,
                '--redis-configuration', json.dumps(redis_config)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Redis persistence configured: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to configure Redis persistence: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Redis persistence configuration error: {e}")
            return False
    
    def get_persistence_status(self) -> Dict:
        """Get Redis persistence configuration status."""
        try:
            # Get Redis instance details
            redis = self.client.redis.get(
                resource_group_name=self.resource_group,
                name=self.redis_name
            )
            
            # Check persistence configuration
            redis_config = redis.redis_configuration or {}
            save_config = redis_config.get('save', '')
            
            persistence_status = {
                'rdb_enabled': bool(save_config),
                'save_config': save_config,
                'maxmemory_policy': redis_config.get('maxmemory-policy', ''),
                'aof_enabled': redis_config.get('appendonly', '') == 'yes'
            }
            
            return {
                'status': 'healthy' if persistence_status['rdb_enabled'] else 'warning',
                'persistence_config': persistence_status
            }
            
        except Exception as e:
            logger.error(f"Failed to get Redis persistence status: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def trigger_manual_backup(self) -> bool:
        """Trigger manual Redis backup."""
        try:
            # Get Redis connection details
            redis = self.client.redis.get(
                resource_group_name=self.resource_group,
                name=self.redis_name
            )
            
            # Get Redis keys
            keys = self.client.redis.list_keys(
                resource_group_name=self.resource_group,
                name=self.redis_name
            )
            
            # Connect to Redis and trigger backup
            import redis as redis_client
            
            redis_client_instance = redis_client.Redis(
                host=redis.host_name,
                port=redis.port,
                password=keys.primary_key,
                ssl=True
            )
            
            # Trigger background save
            redis_client_instance.bgsave()
            
            # Wait for save to complete
            last_save = redis_client_instance.lastsave()
            while redis_client_instance.lastsave() == last_save:
                import time
                time.sleep(1)
            
            logger.info("Redis manual backup triggered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to trigger Redis backup: {e}")
            return False


class AzureBlobStorageBackup:
    """Azure Blob Storage backup operations."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.credential = DefaultAzureCredential()
        self.storage_account = config['storage']['account_name']
        self.container_name = config['storage']['container_name']
        
        # Initialize blob service client
        account_url = f"https://{self.storage_account}.blob.core.windows.net"
        self.blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=self.credential
        )
    
    def configure_immutable_storage(self) -> bool:
        """Configure immutable storage for compliance."""
        try:
            # Enable versioning
            cmd = [
                'az', 'storage', 'account', 'blob-service-properties', 'update',
                '--account-name', self.storage_account,
                '--enable-versioning', 'true'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Blob versioning enabled: {result.stdout}")
            
            # Enable immutable storage
            cmd = [
                'az', 'storage', 'account', 'blob-service-properties', 'update',
                '--account-name', self.storage_account,
                '--enable-immutable-storage', 'true'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Immutable storage enabled: {result.stdout}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to configure immutable storage: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Immutable storage configuration error: {e}")
            return False
    
    def configure_lifecycle_management(self) -> bool:
        """Configure lifecycle management policy."""
        try:
            # Create lifecycle management policy
            policy = {
                "rules": [{
                    "name": "backup-lifecycle",
                    "type": "Lifecycle",
                    "definition": {
                        "filters": {
                            "blobTypes": ["blockBlob"],
                            "prefixMatch": ["backups/"]
                        },
                        "actions": {
                            "baseBlob": {
                                "tierToCool": {
                                    "daysAfterModificationGreaterThan": 30
                                },
                                "tierToArchive": {
                                    "daysAfterModificationGreaterThan": 90
                                },
                                "delete": {
                                    "daysAfterModificationGreaterThan": 2555
                                }
                            }
                        }
                    }
                }]
            }
            
            cmd = [
                'az', 'storage', 'account', 'management-policy', 'create',
                '--account-name', self.storage_account,
                '--policy', json.dumps(policy)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Lifecycle management policy created: {result.stdout}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to configure lifecycle management: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Lifecycle management configuration error: {e}")
            return False
    
    def get_storage_status(self) -> Dict:
        """Get storage account status and configuration."""
        try:
            # Get storage account details
            cmd = [
                'az', 'storage', 'account', 'show',
                '--name', self.storage_account,
                '--query', '{sku:sku.name,replication:sku.name,httpsOnly:enableHttpsTrafficOnly,versioning:enableBlobVersioning}',
                '--output', 'json'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            storage_info = json.loads(result.stdout)
            
            # Check if container exists
            try:
                container_client = self.blob_service_client.get_container_client(self.container_name)
                container_properties = container_client.get_container_properties()
                container_exists = True
            except Exception:
                container_exists = False
            
            return {
                'status': 'healthy',
                'storage_info': storage_info,
                'container_exists': container_exists,
                'container_name': self.container_name
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get storage status: {e.stderr}")
            return {
                'status': 'error',
                'message': e.stderr
            }
        except Exception as e:
            logger.error(f"Storage status error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }


class AzureKeyVaultBackup:
    """Azure Key Vault backup operations."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.credential = DefaultAzureCredential()
        self.key_vault_url = config['azure']['key_vault_url']
        self.client = SecretClient(vault_url=self.key_vault_url, credential=self.credential)
    
    def backup_secrets(self, storage_client: AzureBlobStorageBackup) -> bool:
        """Backup all secrets to blob storage."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            
            # List all secrets
            secrets = list(self.client.list_properties_of_secrets())
            
            for secret_properties in secrets:
                secret_name = secret_properties.name
                logger.info(f"Backing up secret: {secret_name}")
                
                # Get secret value
                secret = self.client.get_secret(secret_name)
                
                # Upload to blob storage
                blob_name = f"keyvault/secrets/{secret_name}-{timestamp}.txt"
                blob_client = storage_client.blob_service_client.get_blob_client(
                    container=storage_client.container_name,
                    blob=blob_name
                )
                
                blob_client.upload_blob(secret.value, overwrite=True)
                logger.info(f"Secret {secret_name} backed up to {blob_name}")
            
            logger.info(f"Backed up {len(secrets)} secrets")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup secrets: {e}")
            return False


class AzureBackupManager:
    """Main Azure backup manager."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.postgres_backup = AzurePostgreSQLBackup(config)
        self.redis_backup = AzureRedisBackup(config)
        self.storage_backup = AzureBlobStorageBackup(config)
        self.keyvault_backup = AzureKeyVaultBackup(config)
    
    def run_backup(self, backup_type: str = 'all') -> Dict:
        """Run backup for specified services."""
        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'backup_type': backup_type,
            'results': {}
        }
        
        if backup_type in ['all', 'postgresql']:
            logger.info("Starting PostgreSQL backup...")
            # Create manual backup
            postgres_result = self.postgres_backup.create_manual_backup()
            results['results']['postgresql'] = {
                'status': 'success' if postgres_result else 'failed',
                'manual_backup': postgres_result
            }
            
            # Export database to blob storage
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            storage_uri = f"https://{self.config['storage']['account_name']}.blob.core.windows.net/{self.config['storage']['container_name']}/postgresql/export-{timestamp}.sql"
            
            export_result = self.postgres_backup.export_database(
                self.config['postgresql']['database_name'],
                storage_uri
            )
            results['results']['postgresql']['export'] = export_result
        
        if backup_type in ['all', 'redis']:
            logger.info("Starting Redis backup...")
            # Configure persistence if not already configured
            persistence_result = self.redis_backup.configure_persistence()
            results['results']['redis'] = {
                'status': 'success' if persistence_result else 'failed',
                'persistence_configured': persistence_result
            }
            
            # Trigger manual backup
            manual_backup_result = self.redis_backup.trigger_manual_backup()
            results['results']['redis']['manual_backup'] = manual_backup_result
        
        if backup_type in ['all', 'keyvault']:
            logger.info("Starting Key Vault backup...")
            keyvault_result = self.keyvault_backup.backup_secrets(self.storage_backup)
            results['results']['keyvault'] = {
                'status': 'success' if keyvault_result else 'failed',
                'secrets_backed_up': keyvault_result
            }
        
        return results
    
    def get_backup_status(self) -> Dict:
        """Get comprehensive backup status."""
        status = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'services': {}
        }
        
        # PostgreSQL status
        status['services']['postgresql'] = self.postgres_backup.get_backup_status()
        
        # Redis status
        status['services']['redis'] = self.redis_backup.get_persistence_status()
        
        # Storage status
        status['services']['storage'] = self.storage_backup.get_storage_status()
        
        return status
    
    def configure_backup_infrastructure(self) -> bool:
        """Configure backup infrastructure (immutable storage, lifecycle management)."""
        try:
            logger.info("Configuring backup infrastructure...")
            
            # Configure immutable storage
            immutable_result = self.storage_backup.configure_immutable_storage()
            
            # Configure lifecycle management
            lifecycle_result = self.storage_backup.configure_lifecycle_management()
            
            # Configure Redis persistence
            redis_result = self.redis_backup.configure_persistence()
            
            return immutable_result and lifecycle_result and redis_result
            
        except Exception as e:
            logger.error(f"Failed to configure backup infrastructure: {e}")
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
    parser = argparse.ArgumentParser(description='Azure Database Backup Script')
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--backup-type', choices=['all', 'postgresql', 'redis', 'keyvault'], 
                       default='all', help='Type of backup to perform')
    parser.add_argument('--status', action='store_true', help='Get backup status')
    parser.add_argument('--configure', action='store_true', help='Configure backup infrastructure')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create backup manager
    backup_manager = AzureBackupManager(config)
    
    if args.configure:
        if args.dry_run:
            logger.info("DRY RUN: Would configure backup infrastructure")
        else:
            success = backup_manager.configure_backup_infrastructure()
            if success:
                logger.info("Backup infrastructure configured successfully")
            else:
                logger.error("Failed to configure backup infrastructure")
                sys.exit(1)
        return
    
    if args.status:
        # Get backup status
        status = backup_manager.get_backup_status()
        print(json.dumps(status, indent=2))
        return
    
    if args.dry_run:
        logger.info(f"DRY RUN: Would perform {args.backup_type} backup")
        return
    
    # Run backup
    results = backup_manager.run_backup(args.backup_type)
    
    # Print results
    print(json.dumps(results, indent=2))
    
    # Check for failures
    failed_services = []
    for service, result in results['results'].items():
        if isinstance(result, dict) and result.get('status') == 'failed':
            failed_services.append(service)
    
    if failed_services:
        logger.error(f"Backup failed for services: {failed_services}")
        sys.exit(1)
    else:
        logger.info("All backups completed successfully")


if __name__ == '__main__':
    main()
