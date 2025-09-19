# Storage Manager

The Storage Manager provides a comprehensive solution for persisting LLaMA Mapper results with enterprise-grade features including immutable storage, encryption, and automatic lifecycle management.

## Features

### 🔒 Immutable Storage (WORM)
- S3 Object Lock configuration for Write Once Read Many compliance
- Configurable retention periods (default: 7 years)
- Automatic lifecycle transitions to reduce costs

### 🔥 Hot Data Access
- PostgreSQL or ClickHouse for fast querying
- 90-day retention in hot storage
- Automatic cleanup of expired records

### 🔐 Security & Encryption
- AES256-KMS encryption with BYOK support
- Data encryption at rest and in transit
- Configurable encryption keys

### 📊 Multi-Backend Support
- **PostgreSQL**: Full ACID compliance, complex queries
- **ClickHouse**: High-performance analytics, time-series data

## Quick Start

### 1. Install Dependencies

```bash
pip install -e .
```

### 2. Configure Settings

Set environment variables or update your `.env` file:

```bash
# S3 Configuration
STORAGE__S3_BUCKET=my-llama-mapper-bucket
STORAGE__AWS_REGION=us-west-2
STORAGE__S3_RETENTION_YEARS=7

# Database Configuration
STORAGE__STORAGE_BACKEND=postgresql
STORAGE__DB_HOST=localhost
STORAGE__DB_PORT=5432
STORAGE__DB_NAME=llama_mapper
STORAGE__DB_USER=postgres
STORAGE__DB_PASSWORD=your_password

# Encryption (optional)
STORAGE__KMS_KEY_ID=arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012
```

### 3. Set Up Infrastructure

```bash
python scripts/setup_storage.py --bucket-name my-llama-mapper-bucket --region us-west-2
```

### 4. Use in Your Code

```python
import asyncio
from datetime import datetime
from llama_mapper.config.settings import Settings
from llama_mapper.storage.manager import StorageManager, StorageRecord

async def store_mapping_result():
    settings = Settings()
    storage_manager = StorageManager(settings.storage)
    
    await storage_manager.initialize()
    
    # Create a record
    record = StorageRecord(
        id="unique-id",
        source_data="detector output",
        mapped_data="canonical taxonomy mapping",
        model_version="llama-2-7b-v1.0",
        timestamp=datetime.utcnow(),
        metadata={"detector": "example"}
    )
    
    # Store it
    record_id = await storage_manager.store_record(record)
    
    # Retrieve it
    retrieved = await storage_manager.retrieve_record(record_id)
    
    await storage_manager.close()

asyncio.run(store_mapping_result())
```

## Configuration Options

### Storage Backend

Choose between PostgreSQL and ClickHouse:

```python
# PostgreSQL (default) - Best for transactional workloads
STORAGE__STORAGE_BACKEND=postgresql

# ClickHouse - Best for analytics and high-throughput
STORAGE__STORAGE_BACKEND=clickhouse
STORAGE__DB_PORT=9000  # Default ClickHouse port
```

### S3 Configuration

```python
# Required
STORAGE__S3_BUCKET=your-bucket-name
STORAGE__AWS_REGION=us-west-2

# Optional
STORAGE__S3_RETENTION_YEARS=7  # WORM retention period
STORAGE__AWS_ACCESS_KEY_ID=your-key-id
STORAGE__AWS_SECRET_ACCESS_KEY=your-secret-key
```

### Encryption

```python
# Use AWS KMS (recommended for production)
STORAGE__KMS_KEY_ID=arn:aws:kms:region:account:key/key-id

# Or use local encryption key
STORAGE__ENCRYPTION_KEY=your-32-character-encryption-key
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Application   │───▶│  StorageManager  │───▶│   Hot Storage   │
└─────────────────┘    └──────────────────┘    │ (PostgreSQL/CH) │
                                │               └─────────────────┘
                                │               
                                ▼               
                        ┌─────────────────┐    
                        │   Cold Storage  │    
                        │      (S3)       │    
                        └─────────────────┘    
```

### Data Flow

1. **Write Path**: Records are stored simultaneously in hot database and S3
2. **Read Path**: Hot storage checked first, fallback to S3 if needed
3. **Lifecycle**: Records older than 90 days are removed from hot storage
4. **Immutability**: S3 Object Lock prevents accidental deletion

## Monitoring & Maintenance

### Cleanup Job

Run periodic cleanup to remove expired hot data:

```python
# In your scheduled job
cleaned_count = await storage_manager.cleanup_expired_records()
print(f"Cleaned up {cleaned_count} records")
```

### Health Checks

```python
# Test storage connectivity
try:
    await storage_manager.initialize()
    print("Storage healthy")
except Exception as e:
    print(f"Storage unhealthy: {e}")
```

## Best Practices

### 1. Connection Management
- Always call `initialize()` before use
- Always call `close()` when done
- Use connection pooling for high-throughput scenarios

### 2. Error Handling
- Implement retry logic for transient failures
- Monitor S3 and database connectivity
- Handle encryption key rotation gracefully

### 3. Performance
- Use ClickHouse for analytics workloads
- Batch operations when possible
- Monitor hot storage size and cleanup frequency

### 4. Security
- Use KMS for encryption key management
- Rotate encryption keys regularly
- Implement proper IAM policies for S3 access
- Use database connection encryption

## Troubleshooting

### Common Issues

**S3 Object Lock Error**
```
InvalidBucketState: Cannot enable Object Lock on existing bucket
```
Solution: Create a new bucket with versioning enabled from the start.

**Database Connection Failed**
```
asyncpg.exceptions.InvalidCatalogNameError: database "llama_mapper" does not exist
```
Solution: Create the database first or update the connection string.

**Encryption Key Error**
```
cryptography.fernet.InvalidToken
```
Solution: Ensure the encryption key is consistent across deployments.

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check storage manager logs:

```python
import structlog
logger = structlog.get_logger("llama_mapper.storage")
```