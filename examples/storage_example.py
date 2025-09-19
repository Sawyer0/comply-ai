#!/usr/bin/env python3
"""
Example usage of the StorageManager for LLaMA Mapper.

This script demonstrates:
- Initializing the storage manager
- Storing mapping results
- Retrieving records
- Cleanup operations
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any

from llama_mapper.config.settings import Settings
from llama_mapper.storage.manager import StorageManager, StorageRecord


async def main():
    """Main example function."""
    
    # Initialize settings (you can override with environment variables)
    settings = Settings(
        storage__s3_bucket="my-llama-mapper-bucket",
        storage__aws_region="us-west-2",
        storage__storage_backend="postgresql",
        storage__db_host="localhost",
        storage__db_port=5432,
        storage__db_name="llama_mapper",
        storage__db_user="postgres",
        storage__db_password="password"
    )
    
    # Create storage manager
    storage_manager = StorageManager(settings.storage)
    
    try:
        # Initialize the storage manager
        print("Initializing storage manager...")
        await storage_manager.initialize()
        print("✓ Storage manager initialized successfully")
        
        # Create a sample record
        record = StorageRecord(
            id=str(uuid.uuid4()),
            source_data="Sample detector output: {'threat_level': 'high', 'category': 'malware'}",
            mapped_data="{'pillar': 'security', 'taxonomy_id': 'SEC-001', 'confidence': 0.95}",
            model_version="llama-2-7b-mapper-v1.0",
            timestamp=datetime.utcnow(),
            metadata={
                "detector_name": "example_detector",
                "processing_time_ms": 150,
                "input_tokens": 45,
                "output_tokens": 23
            }
        )
        
        # Store the record
        print(f"Storing record {record.id}...")
        stored_id = await storage_manager.store_record(record)
        print(f"✓ Record stored successfully with ID: {stored_id}")
        
        # Retrieve the record
        print(f"Retrieving record {stored_id}...")
        retrieved_record = await storage_manager.retrieve_record(stored_id)
        
        if retrieved_record:
            print("✓ Record retrieved successfully:")
            print(f"  - ID: {retrieved_record.id}")
            print(f"  - Model Version: {retrieved_record.model_version}")
            print(f"  - Timestamp: {retrieved_record.timestamp}")
            print(f"  - S3 Key: {retrieved_record.s3_key}")
            print(f"  - Encrypted: {retrieved_record.encrypted}")
        else:
            print("✗ Failed to retrieve record")
        
        # Demonstrate cleanup (this would normally be run as a scheduled job)
        print("Running cleanup for expired records...")
        cleaned_count = await storage_manager.cleanup_expired_records()
        print(f"✓ Cleaned up {cleaned_count} expired records")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        raise
    
    finally:
        # Clean up connections
        await storage_manager.close()
        print("✓ Storage manager closed")


if __name__ == "__main__":
    asyncio.run(main())