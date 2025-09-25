#!/usr/bin/env python3
"""
Master migration script for microservice database separation.

This script coordinates the migration of data from the monolithic database
structure to separate databases for each microservice.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.database.connection_manager import DatabaseConnectionManager, DatabaseConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MicroserviceMigrationOrchestrator:
    """Orchestrates the migration of all microservice databases."""

    def __init__(self):
        self.db_manager = DatabaseConnectionManager()
        self.migration_log = []

    async def initialize_connections(self) -> None:
        """Initialize database connections for all services."""
        configs = [
            DatabaseConfig(
                service_name="orchestration",
                database_url=os.getenv(
                    "ORCHESTRATION_DATABASE_URL",
                    "postgresql://orchestration:password@localhost:5432/orchestration_db",
                ),
            ),
            DatabaseConfig(
                service_name="analysis",
                database_url=os.getenv(
                    "ANALYSIS_DATABASE_URL",
                    "postgresql://analysis:password@localhost:5432/analysis_db",
                ),
            ),
            DatabaseConfig(
                service_name="mapper",
                database_url=os.getenv(
                    "MAPPER_DATABASE_URL",
                    "postgresql://mapper:password@localhost:5432/mapper_db",
                ),
            ),
        ]

        await self.db_manager.initialize(configs)
        logger.info("✅ Database connections initialized for all services")

    async def check_source_database(self) -> Dict[str, Any]:
        """Check the source monolithic database for migration readiness."""
        source_url = os.getenv(
            "SOURCE_DATABASE_URL",
            "postgresql://llama_mapper:password@localhost:5432/llama_mapper_db",
        )

        try:
            import asyncpg

            conn = await asyncpg.connect(source_url)

            # Check if source tables exist
            tables = await conn.fetch(
                """
                SELECT table_name, 
                       (SELECT COUNT(*) FROM information_schema.columns 
                        WHERE table_name = t.table_name AND table_schema = 'public') as column_count
                FROM information_schema.tables t
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """
            )

            # Check record counts
            record_counts = {}
            for table in tables:
                table_name = table["table_name"]
                try:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                    record_counts[table_name] = count
                except Exception as e:
                    record_counts[table_name] = f"Error: {e}"

            await conn.close()

            return {
                "status": "ready",
                "tables": [dict(row) for row in tables],
                "record_counts": record_counts,
                "total_tables": len(tables),
            }

        except Exception as e:
            logger.error(f"Failed to check source database: {e}")
            return {"status": "error", "error": str(e)}

    async def backup_source_database(self) -> Dict[str, Any]:
        """Create a backup of the source database before migration."""
        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"pre_migration_backup_{timestamp}.sql"

        source_url = os.getenv(
            "SOURCE_DATABASE_URL",
            "postgresql://llama_mapper:password@localhost:5432/llama_mapper_db",
        )

        try:
            # Use pg_dump to create backup
            import subprocess

            cmd = [
                "pg_dump",
                "--no-password",
                "--verbose",
                "--clean",
                "--no-acl",
                "--no-owner",
                source_url,
                "-f",
                str(backup_file),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return {
                    "status": "success",
                    "backup_file": str(backup_file),
                    "size_bytes": backup_file.stat().st_size,
                }
            else:
                return {"status": "error", "error": result.stderr}

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return {"status": "error", "error": str(e)}

    async def run_service_migrations(self) -> Dict[str, Any]:
        """Run migrations for all microservices."""
        migration_results = {}

        # Import migration managers
        sys.path.append(str(project_root / "detector-orchestration"))
        sys.path.append(str(project_root / "analysis-service"))
        sys.path.append(str(project_root / "mapper-service"))

        # Orchestration Service Migration
        logger.info("🔄 Starting Orchestration Service migration...")
        try:
            from detector_orchestration.database.migrations import MigrationManager

            orchestration_url = os.getenv(
                "ORCHESTRATION_DATABASE_URL",
                "postgresql://orchestration:password@localhost:5432/orchestration_db",
            )

            manager = MigrationManager(orchestration_url)
            await manager.migrate_from_monolith()
            validation = await manager.validate_schema()

            migration_results["orchestration"] = {
                "status": "success" if validation["valid"] else "failed",
                "validation": validation,
                "timestamp": datetime.now().isoformat(),
            }

            if validation["valid"]:
                logger.info("✅ Orchestration Service migration completed")
            else:
                logger.error(f"❌ Orchestration Service migration failed: {validation}")

        except Exception as e:
            logger.error(f"❌ Orchestration Service migration error: {e}")
            migration_results["orchestration"] = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

        # Analysis Service Migration
        logger.info("🔄 Starting Analysis Service migration...")
        try:
            from analysis_service.database.migrations import AnalysisMigrationManager

            analysis_url = os.getenv(
                "ANALYSIS_DATABASE_URL",
                "postgresql://analysis:password@localhost:5432/analysis_db",
            )

            manager = AnalysisMigrationManager(analysis_url)
            await manager.setup_vector_extension()
            await manager.migrate_from_monolith()
            validation = await manager.validate_schema()

            migration_results["analysis"] = {
                "status": "success" if validation["valid"] else "failed",
                "validation": validation,
                "timestamp": datetime.now().isoformat(),
            }

            if validation["valid"]:
                logger.info("✅ Analysis Service migration completed")
            else:
                logger.error(f"❌ Analysis Service migration failed: {validation}")

        except Exception as e:
            logger.error(f"❌ Analysis Service migration error: {e}")
            migration_results["analysis"] = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

        # Mapper Service Migration
        logger.info("🔄 Starting Mapper Service migration...")
        try:
            from mapper_service.database.migrations import MapperMigrationManager

            mapper_url = os.getenv(
                "MAPPER_DATABASE_URL",
                "postgresql://mapper:password@localhost:5432/mapper_db",
            )

            manager = MapperMigrationManager(mapper_url)
            await manager.migrate_from_monolith()
            validation = await manager.validate_schema()

            migration_results["mapper"] = {
                "status": "success" if validation["valid"] else "failed",
                "validation": validation,
                "timestamp": datetime.now().isoformat(),
            }

            if validation["valid"]:
                logger.info("✅ Mapper Service migration completed")
            else:
                logger.error(f"❌ Mapper Service migration failed: {validation}")

        except Exception as e:
            logger.error(f"❌ Mapper Service migration error: {e}")
            migration_results["mapper"] = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

        return migration_results

    async def validate_migrations(self) -> Dict[str, Any]:
        """Validate that all migrations completed successfully."""
        validation_results = {}

        # Check database health
        health_status = await self.db_manager.health_check()

        # Count records in each service
        for service_name in ["orchestration", "analysis", "mapper"]:
            try:
                async with self.db_manager.get_connection(service_name) as conn:
                    # Get table counts
                    tables = await conn.fetch(
                        """
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_type = 'BASE TABLE'
                        AND table_name != 'schema_migrations'
                    """
                    )

                    table_counts = {}
                    for table in tables:
                        table_name = table["table_name"]
                        count = await conn.fetchval(
                            f"SELECT COUNT(*) FROM {table_name}"
                        )
                        table_counts[table_name] = count

                    validation_results[service_name] = {
                        "health": health_status.get(service_name, {}),
                        "table_counts": table_counts,
                        "total_records": sum(table_counts.values()),
                    }

            except Exception as e:
                validation_results[service_name] = {"error": str(e)}

        return validation_results

    async def generate_migration_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive migration report."""
        report_data = {
            "migration_timestamp": datetime.now().isoformat(),
            "source_database_check": results.get("source_check", {}),
            "backup_status": results.get("backup", {}),
            "migration_results": results.get("migrations", {}),
            "validation_results": results.get("validation", {}),
            "summary": {
                "total_services": 3,
                "successful_migrations": sum(
                    1
                    for r in results.get("migrations", {}).values()
                    if r.get("status") == "success"
                ),
                "failed_migrations": sum(
                    1
                    for r in results.get("migrations", {}).values()
                    if r.get("status") in ["failed", "error"]
                ),
            },
        }

        # Save report
        report_file = Path("migration_report.json")
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        # Generate human-readable summary
        summary = f"""
🔄 Microservice Database Migration Report
========================================

Migration Timestamp: {report_data['migration_timestamp']}

📊 Summary:
- Total Services: {report_data['summary']['total_services']}
- Successful Migrations: {report_data['summary']['successful_migrations']}
- Failed Migrations: {report_data['summary']['failed_migrations']}

📋 Service Results:
"""

        for service, result in results.get("migrations", {}).items():
            status_emoji = "✅" if result.get("status") == "success" else "❌"
            summary += f"  {status_emoji} {service.title()} Service: {result.get('status', 'unknown')}\n"

            if "validation" in result:
                validation = result["validation"]
                if isinstance(validation, dict):
                    summary += (
                        f"    - Tables: {len(validation.get('existing_tables', []))}\n"
                    )
                    if "missing_tables" in validation and validation["missing_tables"]:
                        summary += (
                            f"    - Missing Tables: {validation['missing_tables']}\n"
                        )

        summary += f"\n📄 Full report saved to: {report_file}\n"

        return summary

    async def run_complete_migration(self) -> str:
        """Run the complete migration process."""
        results = {}

        try:
            # Initialize connections
            await self.initialize_connections()

            # Check source database
            logger.info("🔍 Checking source database...")
            results["source_check"] = await self.check_source_database()

            if results["source_check"]["status"] != "ready":
                raise Exception(f"Source database not ready: {results['source_check']}")

            # Create backup
            logger.info("💾 Creating backup of source database...")
            results["backup"] = await self.backup_source_database()

            if results["backup"]["status"] != "success":
                logger.warning(f"Backup failed: {results['backup']}")

            # Run migrations
            logger.info("🚀 Starting microservice migrations...")
            results["migrations"] = await self.run_service_migrations()

            # Validate results
            logger.info("✅ Validating migration results...")
            results["validation"] = await self.validate_migrations()

            # Generate report
            report = await self.generate_migration_report(results)

            return report

        except Exception as e:
            logger.error(f"Migration process failed: {e}")
            results["error"] = str(e)
            report = await self.generate_migration_report(results)
            return f"❌ Migration failed: {e}\n\n{report}"

        finally:
            await self.db_manager.close_all()


async def main():
    """Main migration entry point."""
    print("🔄 Starting Microservice Database Migration")
    print("=" * 50)

    orchestrator = MicroserviceMigrationOrchestrator()

    try:
        report = await orchestrator.run_complete_migration()
        print(report)

        # Exit with appropriate code
        if "❌ Migration failed" in report:
            sys.exit(1)
        else:
            print("🎉 Migration completed successfully!")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n⚠️  Migration interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
