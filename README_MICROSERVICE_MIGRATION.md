# Microservice Database Migration Guide

This document provides a comprehensive guide for migrating from the monolithic database structure to separate databases for each microservice.

## Overview

The llama-mapper system has been refactored into 3 microservices, each with its own dedicated database:

1. **Detector Orchestration Service** (`orchestration_db`) - Port 5432
2. **Analysis Service** (`analysis_db`) - Port 5433  
3. **Mapper Service** (`mapper_db`) - Port 5434

## Database Schema Separation

### Detector Orchestration Service Schema
- **Tables**: `detectors`, `detector_executions`, `orchestration_requests`, `policies`, `policy_enforcements`, `rate_limits`, `service_registry`, `async_jobs`, `orchestration_audit`
- **Purpose**: Manages detector coordination, policy enforcement, and service discovery
- **Key Features**: Circuit breakers, health monitoring, rate limiting, OPA integration

### Analysis Service Schema  
- **Tables**: `analysis_requests`, `canonical_results`, `pattern_analysis`, `risk_scores`, `compliance_mappings`, `knowledge_base`, `rag_insights`, `quality_metrics`, `quality_alerts`, `weekly_evaluations`, `ml_models`, `analysis_pipelines`, `tenant_analytics`
- **Purpose**: Handles advanced analysis, risk scoring, compliance intelligence, and RAG system
- **Key Features**: Vector embeddings (pgvector), ML model tracking, quality monitoring

### Mapper Service Schema
- **Tables**: `mapping_requests`, `mapping_results`, `model_inferences`, `cost_metrics`, `training_jobs`, `model_versions`, `deployment_experiments`, `taxonomies`, `framework_configs`, `validation_schemas`, `feature_flags`, `storage_artifacts`
- **Purpose**: Core mapping, model serving, response generation, and training
- **Key Features**: Model versioning, A/B testing, cost tracking, taxonomy management

## Migration Process

### Prerequisites

1. **Database Setup**: Ensure all 3 PostgreSQL databases are running:
   ```bash
   # Using Docker Compose
   docker-compose -f docker-compose.microservices.yml up postgres-orchestration postgres-analysis postgres-mapper
   ```

2. **Environment Variables**: Set database URLs for each service:
   ```bash
   export ORCHESTRATION_DATABASE_URL="postgresql://orchestration:password@localhost:5432/orchestration_db"
   export ANALYSIS_DATABASE_URL="postgresql://analysis:password@localhost:5433/analysis_db"  
   export MAPPER_DATABASE_URL="postgresql://mapper:password@localhost:5434/mapper_db"
   export SOURCE_DATABASE_URL="postgresql://llama_mapper:password@localhost:5432/llama_mapper_db"
   ```

3. **Dependencies**: Install required Python packages:
   ```bash
   pip install asyncpg psycopg2-binary
   ```

### Running the Migration

Execute the master migration script:

```bash
python scripts/migrate_microservice_databases.py
```

The migration process includes:

1. **Source Database Check**: Validates the monolithic database is accessible and contains data
2. **Backup Creation**: Creates a full backup of the source database using `pg_dump`
3. **Schema Creation**: Creates tables, indexes, and RLS policies in each service database
4. **Data Migration**: Migrates existing data to appropriate service databases
5. **Validation**: Verifies migration success and data integrity

### Migration Steps Detail

#### 1. Orchestration Service Migration
- Migrates detector execution data from `storage_records`
- Creates default detector entries for existing detector types
- Sets up orchestration requests from grouped detector executions
- Initializes rate limiting and policy enforcement tables

#### 2. Analysis Service Migration  
- Creates analysis requests from existing storage records
- Generates canonical results from mapped data
- Calculates risk scores based on confidence metrics
- Sets up RAG knowledge base with compliance framework documents
- Migrates quality monitoring data and creates weekly evaluations

#### 3. Mapper Service Migration
- Creates mapping requests and results from storage records
- Sets up model inference tracking and cost metrics
- Initializes model versions and training job records
- Creates taxonomy and framework configuration data
- Sets up validation schemas and feature flags

## Post-Migration Validation

### Health Checks

Verify each service database is healthy:

```bash
# Check orchestration service
curl http://localhost:8000/health

# Check analysis service  
curl http://localhost:8001/health

# Check mapper service
curl http://localhost:8002/health
```

### Data Validation

The migration script automatically validates:

- All required tables exist
- Row Level Security (RLS) is enabled
- Indexes are created properly
- Data has been migrated correctly
- Service-specific features are working (e.g., vector extension for analysis service)

### Manual Verification

You can manually verify the migration by checking record counts:

```sql
-- Orchestration Service
SELECT 'detectors' as table_name, COUNT(*) FROM detectors
UNION ALL
SELECT 'detector_executions', COUNT(*) FROM detector_executions
UNION ALL  
SELECT 'orchestration_requests', COUNT(*) FROM orchestration_requests;

-- Analysis Service
SELECT 'analysis_requests' as table_name, COUNT(*) FROM analysis_requests
UNION ALL
SELECT 'canonical_results', COUNT(*) FROM canonical_results
UNION ALL
SELECT 'knowledge_base', COUNT(*) FROM knowledge_base;

-- Mapper Service  
SELECT 'mapping_requests' as table_name, COUNT(*) FROM mapping_requests
UNION ALL
SELECT 'mapping_results', COUNT(*) FROM mapping_results
UNION ALL
SELECT 'model_versions', COUNT(*) FROM model_versions;
```

## Rollback Procedure

If migration fails or issues are discovered:

1. **Stop all microservices**
2. **Restore from backup**:
   ```bash
   psql $SOURCE_DATABASE_URL < backups/pre_migration_backup_YYYYMMDD_HHMMSS.sql
   ```
3. **Clear service databases**:
   ```bash
   # Drop and recreate each service database
   dropdb orchestration_db && createdb orchestration_db
   dropdb analysis_db && createdb analysis_db  
   dropdb mapper_db && createdb mapper_db
   ```

## Troubleshooting

### Common Issues

1. **Connection Errors**: Verify database URLs and credentials
2. **Permission Errors**: Ensure database users have proper permissions
3. **Missing Extensions**: Install required PostgreSQL extensions (uuid-ossp, pgcrypto, vector)
4. **Data Conflicts**: Check for duplicate data or constraint violations

### Debug Mode

Run migration with debug logging:

```bash
PYTHONPATH=. python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
import asyncio
from scripts.migrate_microservice_databases import main
asyncio.run(main())
"
```

### Individual Service Migration

You can run migrations for individual services:

```bash
# Orchestration service only
python detector-orchestration/database/migrations.py

# Analysis service only  
python analysis-service/database/migrations.py

# Mapper service only
python mapper-service/database/migrations.py
```

## Configuration Updates

After successful migration, update service configurations:

### Docker Compose
Update `docker-compose.microservices.yml` with correct database URLs.

### Environment Variables
Update production environment variables to point to the new service-specific databases.

### Application Code
Update any remaining monolithic database references to use the new service-specific connections.

## Performance Considerations

### Connection Pooling
Each service uses independent connection pools:
- **Min Pool Size**: 5 connections per service
- **Max Pool Size**: 20 connections per service  
- **Pool Timeout**: 30 seconds
- **Pool Recycle**: 1 hour

### Indexing Strategy
All services include optimized indexes for:
- Tenant isolation queries
- Time-based queries  
- Foreign key relationships
- Performance-critical lookups

### Row Level Security
All tables use RLS policies for tenant isolation:
```sql
-- Example RLS policy
CREATE POLICY tenant_isolation_policy ON table_name
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id', true));
```

## Monitoring

### Migration Metrics
The migration script generates a comprehensive report including:
- Migration success/failure status per service
- Record counts before and after migration
- Performance metrics and timing
- Validation results

### Ongoing Monitoring
Set up monitoring for:
- Database connection health
- Query performance
- Storage usage
- Replication lag (if using read replicas)

## Security

### Database Security
- Each service database has isolated credentials
- Row Level Security enforces tenant isolation
- All connections use SSL/TLS encryption
- Database users have minimal required permissions

### Data Privacy
- No raw content is stored in databases (only hashes)
- Audit trails track all data access
- Automatic data retention and cleanup
- Field-level encryption for sensitive data

## Next Steps

After successful migration:

1. **Update CI/CD pipelines** to deploy services independently
2. **Configure monitoring and alerting** for each service
3. **Set up database backups** for each service database
4. **Update documentation** to reflect new architecture
5. **Train team members** on new service boundaries and operations

## Support

For migration issues or questions:
- Check the migration report: `migration_report.json`
- Review service logs for detailed error information
- Consult service-specific README files for additional guidance
- Use the troubleshooting section above for common issues