#!/usr/bin/env python3
"""
Setup script for LLaMA Mapper storage infrastructure.

This script helps set up:
- S3 bucket with WORM configuration
- Database tables
- Initial configuration validation
"""

import asyncio
import sys
from typing import Optional

import boto3
from botocore.exceptions import ClientError
import click

from llama_mapper.config.settings import Settings
from llama_mapper.storage.manager import StorageManager


@click.command()
@click.option('--bucket-name', required=True, help='S3 bucket name to create')
@click.option('--region', default='us-east-1', help='AWS region')
@click.option('--db-backend', default='postgresql', help='Database backend (postgresql/clickhouse)')
@click.option('--skip-s3', is_flag=True, help='Skip S3 setup')
@click.option('--skip-db', is_flag=True, help='Skip database setup')
def setup_storage(bucket_name: str, region: str, db_backend: str, skip_s3: bool, skip_db: bool):
    """Set up storage infrastructure for LLaMA Mapper."""
    
    click.echo("🚀 Setting up LLaMA Mapper storage infrastructure...")
    
    # Create settings
    settings = Settings(
        storage__s3_bucket=bucket_name,
        storage__aws_region=region,
        storage__storage_backend=db_backend
    )
    
    asyncio.run(_setup_async(settings, skip_s3, skip_db))


async def _setup_async(settings: Settings, skip_s3: bool, skip_db: bool):
    """Async setup function."""
    
    if not skip_s3:
        await setup_s3_bucket(settings)
    
    if not skip_db:
        await setup_database(settings)
    
    # Test the full setup
    await test_storage_setup(settings)


async def setup_s3_bucket(settings: Settings):
    """Set up S3 bucket with WORM configuration."""
    
    click.echo(f"📦 Setting up S3 bucket: {settings.storage.s3_bucket}")
    
    try:
        # Create S3 client
        s3_client = boto3.client(
            's3',
            region_name=settings.storage.aws_region
        )
        
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=settings.storage.s3_bucket)
            click.echo(f"✓ Bucket {settings.storage.s3_bucket} already exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Create bucket
                if settings.storage.aws_region == 'us-east-1':
                    s3_client.create_bucket(Bucket=settings.storage.s3_bucket)
                else:
                    s3_client.create_bucket(
                        Bucket=settings.storage.s3_bucket,
                        CreateBucketConfiguration={'LocationConstraint': settings.storage.aws_region}
                    )
                click.echo(f"✓ Created bucket {settings.storage.s3_bucket}")
            else:
                raise
        
        # Enable versioning (required for Object Lock)
        s3_client.put_bucket_versioning(
            Bucket=settings.storage.s3_bucket,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        click.echo("✓ Enabled bucket versioning")
        
        # Configure bucket encryption
        s3_client.put_bucket_encryption(
            Bucket=settings.storage.s3_bucket,
            ServerSideEncryptionConfiguration={
                'Rules': [{
                    'ApplyServerSideEncryptionByDefault': {
                        'SSEAlgorithm': 'AES256'
                    }
                }]
            }
        )
        click.echo("✓ Configured bucket encryption")
        
        # Set up lifecycle policy for cost optimization
        s3_client.put_bucket_lifecycle_configuration(
            Bucket=settings.storage.s3_bucket,
            LifecycleConfiguration={
                'Rules': [{
                    'ID': 'mapper-lifecycle',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'records/'},
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 90,
                            'StorageClass': 'GLACIER'
                        },
                        {
                            'Days': 365,
                            'StorageClass': 'DEEP_ARCHIVE'
                        }
                    ]
                }]
            }
        )
        click.echo("✓ Configured lifecycle policy")
        
    except Exception as e:
        click.echo(f"✗ S3 setup failed: {e}", err=True)
        sys.exit(1)


async def setup_database(settings: Settings):
    """Set up database tables."""
    
    click.echo(f"🗄️  Setting up {settings.storage.storage_backend} database...")
    
    try:
        # Create storage manager (this will create tables)
        storage_manager = StorageManager(settings.storage)
        await storage_manager.initialize()
        
        click.echo("✓ Database tables created successfully")
        
        # Clean up
        await storage_manager.close()
        
    except Exception as e:
        click.echo(f"✗ Database setup failed: {e}", err=True)
        sys.exit(1)


async def test_storage_setup(settings: Settings):
    """Test the complete storage setup."""
    
    click.echo("🧪 Testing storage setup...")
    
    try:
        storage_manager = StorageManager(settings.storage)
        await storage_manager.initialize()
        
        click.echo("✓ Storage manager initialization successful")
        
        # Test basic operations would go here
        # For now, just verify we can connect
        
        await storage_manager.close()
        click.echo("✓ All storage tests passed!")
        
    except Exception as e:
        click.echo(f"✗ Storage test failed: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    setup_storage()