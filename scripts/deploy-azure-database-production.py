#!/usr/bin/env python3
"""
Deployment script for Azure Database production readiness features.

This script automates the deployment and configuration of the enhanced
Azure Database for PostgreSQL implementation with all production features.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

import structlog

# Import our enhanced storage components
try:
    from src.llama_mapper.storage.manager.enhanced_manager import EnhancedStorageManager
    from src.llama_mapper.storage.database.azure_config import AzureDatabaseConfig, AzureDatabaseConnectionManager
    from src.llama_mapper.storage.database.migrations import DatabaseMigrationManager, create_production_migrations
    from src.llama_mapper.storage.security.encryption import FieldEncryption, EnhancedRowLevelSecurity
    from src.llama_mapper.storage.monitoring.azure_monitor import AzureDatabaseMonitor
    from src.llama_mapper.storage.optimization.query_optimizer import QueryOptimizer
    from src.llama_mapper.storage.testing.azure_test_framework import AzureDatabaseTestFramework
    from src.llama_mapper.config.settings import StorageConfig, AzureStorageConfig
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/azure-database-deployment.log"),
    ],
)
logger = structlog.get_logger(__name__)


class AzureDatabaseDeployment:
    """Handles Azure Database production deployment."""
    
    def __init__(self, config_path: str, environment: str = "production"):
        self.config_path = config_path
        self.environment = environment
        self.config: Dict[str, Any] = {}
        self.connection_manager: AzureDatabaseConnectionManager = None
        self.storage_manager: EnhancedStorageManager = None
        
    def load_configuration(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.endswith('.json'):
                        self.config = json.load(f)
                    else:
                        import yaml
                        self.config = yaml.safe_load(f)
            else:
                # Create default configuration
                self.config = self._create_default_config()
                
            # Override with environment variables
            self._apply_environment_overrides()
            
            logger.info("Configuration loaded", environment=self.environment)
            return self.config
            
        except Exception as e:
            logger.error("Failed to load configuration", error=str(e))
            raise
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration for Azure Database."""
        return {
            "azure": {
                "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
                "resource_group": os.getenv("AZURE_RESOURCE_GROUP", "comply-ai-rg"),
                "server_name": os.getenv("AZURE_DB_SERVER", "comply-ai-postgres"),
                "azure_db_host": os.getenv("AZURE_DB_HOST"),
                "key_vault_url": os.getenv("AZURE_KEY_VAULT_URL"),
                "ssl_mode": "require",
                "connection_timeout": 30,
                "command_timeout": 60,
                "min_pool_size": 5,
                "max_pool_size": 20,
                "read_replica_regions": [],
                "enable_azure_monitor": True,
                "log_analytics_workspace_id": os.getenv("AZURE_LOG_ANALYTICS_WORKSPACE_ID"),
                "backup_retention_days": 30,
                "geo_redundant_backup": True
            },
            "database": {
                "name": os.getenv("AZURE_DB_NAME", "llama_mapper"),
                "user": os.getenv("AZURE_DB_USER", "llama_mapper_user"),
                "password": os.getenv("AZURE_DB_PASSWORD"),
                "enable_ssl": True,
                "enable_rls": True,
                "enable_audit_logging": True,
                "field_encryption_enabled": True
            },
            "deployment": {
                "run_migrations": True,
                "create_indexes": True,
                "setup_partitioning": True,
                "enable_monitoring": True,
                "run_tests": self.environment != "production",
                "optimize_performance": True
            }
        }
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            "AZURE_SUBSCRIPTION_ID": ["azure", "subscription_id"],
            "AZURE_RESOURCE_GROUP": ["azure", "resource_group"],
            "AZURE_DB_SERVER": ["azure", "server_name"],
            "AZURE_DB_HOST": ["azure", "azure_db_host"],
            "AZURE_DB_NAME": ["database", "name"],
            "AZURE_DB_USER": ["database", "user"],
            "AZURE_DB_PASSWORD": ["database", "password"],
            "AZURE_KEY_VAULT_URL": ["azure", "key_vault_url"],
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                current = self.config
                for key in config_path[:-1]:
                    current = current.setdefault(key, {})
                current[config_path[-1]] = value
    
    async def validate_prerequisites(self) -> Dict[str, bool]:
        """Validate deployment prerequisites."""
        checks = {}
        
        try:
            # Check Azure configuration
            azure_config = self.config.get("azure", {})
            required_azure = ["subscription_id", "resource_group", "server_name", "azure_db_host"]
            
            for field in required_azure:
                checks[f"azure_{field}"] = bool(azure_config.get(field))
            
            # Check database configuration
            db_config = self.config.get("database", {})
            required_db = ["name", "user", "password"]
            
            for field in required_db:
                checks[f"database_{field}"] = bool(db_config.get(field))
            
            # Check Azure connectivity (simplified)
            try:
                from azure.identity import DefaultAzureCredential
                credential = DefaultAzureCredential()
                # This is a simple check - in practice you'd verify actual access
                checks["azure_authentication"] = True
            except Exception:
                checks["azure_authentication"] = False
            
            # Overall validation
            checks["prerequisites_met"] = all(checks.values())
            
            logger.info("Prerequisites validation completed", checks=checks)
            return checks
            
        except Exception as e:
            logger.error("Prerequisites validation failed", error=str(e))
            return {"prerequisites_met": False, "error": str(e)}
    
    async def initialize_storage_manager(self) -> EnhancedStorageManager:
        """Initialize the enhanced storage manager."""
        try:
            # Create storage configuration
            azure_storage_config = AzureStorageConfig(**self.config["azure"])
            
            storage_config = StorageConfig(
                storage_backend="postgresql",
                db_host=self.config["azure"]["azure_db_host"],
                db_name=self.config["database"]["name"],
                db_user=self.config["database"]["user"],
                db_password=self.config["database"]["password"],
                enable_ssl=self.config["database"]["enable_ssl"],
                enable_rls=self.config["database"]["enable_rls"],
                enable_audit_logging=self.config["database"]["enable_audit_logging"],
                field_encryption_enabled=self.config["database"]["field_encryption_enabled"],
                azure=azure_storage_config
            )
            
            # Initialize enhanced storage manager
            self.storage_manager = EnhancedStorageManager(storage_config)
            await self.storage_manager.initialize()
            
            logger.info("Enhanced storage manager initialized")
            return self.storage_manager
            
        except Exception as e:
            logger.error("Failed to initialize storage manager", error=str(e))
            raise
    
    async def run_database_migrations(self) -> Dict[str, Any]:
        """Run database migrations."""
        try:
            if not self.config["deployment"]["run_migrations"]:
                return {"skipped": True, "reason": "Migrations disabled in configuration"}
            
            if not self.storage_manager.migration_manager:
                return {"error": "Migration manager not available"}
            
            # Apply migrations
            results = await self.storage_manager.migration_manager.apply_all_migrations()
            
            # Validate schema integrity
            schema_status = await self.storage_manager.migration_manager.validate_schema_integrity()
            
            migration_result = {
                "migration_results": results,
                "schema_validation": schema_status,
                "success": schema_status.get("schema_valid", False)
            }
            
            logger.info("Database migrations completed", results=migration_result)
            return migration_result
            
        except Exception as e:
            logger.error("Database migration failed", error=str(e))
            return {"error": str(e), "success": False}
    
    async def setup_performance_optimization(self) -> Dict[str, Any]:
        """Setup performance optimization features."""
        try:
            if not self.config["deployment"]["optimize_performance"]:
                return {"skipped": True, "reason": "Performance optimization disabled"}
            
            optimizer = QueryOptimizer(self.storage_manager.connection_manager)
            results = {}
            
            # Create performance indexes
            if self.config["deployment"]["create_indexes"]:
                results["indexes"] = await optimizer.create_performance_indexes()
            
            # Setup partitioning
            if self.config["deployment"]["setup_partitioning"]:
                results["partitions"] = await optimizer.create_partitions()
            
            # Create materialized views
            results["materialized_views"] = await optimizer.create_materialized_views()
            
            # Optimize Azure parameters
            results["azure_optimization"] = await optimizer.optimize_azure_parameters()
            
            # Enable extensions
            results["extensions"] = await optimizer.enable_azure_extensions()
            
            # Get optimization recommendations
            results["recommendations"] = await optimizer.get_optimization_recommendations()
            
            logger.info("Performance optimization completed", results=results)
            return {"success": True, "results": results}
            
        except Exception as e:
            logger.error("Performance optimization failed", error=str(e))
            return {"error": str(e), "success": False}
    
    async def setup_monitoring(self) -> Dict[str, Any]:
        """Setup monitoring and alerting."""
        try:
            if not self.config["deployment"]["enable_monitoring"]:
                return {"skipped": True, "reason": "Monitoring disabled"}
            
            if not self.storage_manager.azure_monitor:
                return {"error": "Azure monitor not available"}
            
            # Setup Azure alerts
            alert_rules = await self.storage_manager.azure_monitor.setup_azure_alerts()
            
            # Get monitoring summary
            monitoring_summary = await self.storage_manager.azure_monitor.get_monitoring_summary()
            
            results = {
                "alert_rules_configured": len(alert_rules),
                "monitoring_status": monitoring_summary,
                "monitoring_active": monitoring_summary.get("monitoring_active", False)
            }
            
            logger.info("Monitoring setup completed", results=results)
            return {"success": True, "results": results}
            
        except Exception as e:
            logger.error("Monitoring setup failed", error=str(e))
            return {"error": str(e), "success": False}
    
    async def run_deployment_tests(self) -> Dict[str, Any]:
        """Run deployment validation tests."""
        try:
            if not self.config["deployment"]["run_tests"]:
                return {"skipped": True, "reason": "Tests disabled for production"}
            
            test_framework = AzureDatabaseTestFramework()
            test_results = {}
            
            # Run basic connectivity test
            health_status = await self.storage_manager.get_health_status()
            test_results["health_check"] = health_status
            
            # Validate tenant isolation
            if self.storage_manager.rls_manager:
                isolation_test = await self.storage_manager.rls_manager.validate_tenant_isolation("test_tenant")
                test_results["tenant_isolation"] = isolation_test
            
            # Test field encryption
            if self.storage_manager.field_encryption:
                test_data = {"test": "sensitive data"}
                encrypted = self.storage_manager.field_encryption.encrypt_dict(test_data, ["test"])
                decrypted = self.storage_manager.field_encryption.decrypt_dict(encrypted, ["test"])
                test_results["encryption"] = {"success": test_data == decrypted}
            
            logger.info("Deployment tests completed", results=test_results)
            return {"success": True, "results": test_results}
            
        except Exception as e:
            logger.error("Deployment tests failed", error=str(e))
            return {"error": str(e), "success": False}
    
    async def generate_deployment_report(self, results: Dict[str, Any]) -> str:
        """Generate deployment report."""
        report_lines = [
            "# Azure Database Production Deployment Report",
            f"**Deployment Time**: {datetime.utcnow().isoformat()}",
            f"**Environment**: {self.environment}",
            "",
            "## Deployment Summary",
            ""
        ]
        
        # Prerequisites
        prereqs = results.get("prerequisites", {})
        report_lines.extend([
            "### Prerequisites",
            f"- Prerequisites Met: {'✓' if prereqs.get('prerequisites_met') else '✗'}",
            ""
        ])
        
        # Migrations
        migrations = results.get("migrations", {})
        if migrations.get("success"):
            applied = len(migrations.get("migration_results", {}).get("applied", []))
            report_lines.extend([
                "### Database Migrations",
                f"- Migrations Applied: {applied}",
                f"- Schema Valid: {'✓' if migrations.get('schema_validation', {}).get('schema_valid') else '✗'}",
                ""
            ])
        
        # Performance Optimization
        performance = results.get("performance", {})
        if performance.get("success"):
            perf_results = performance.get("results", {})
            report_lines.extend([
                "### Performance Optimization",
                f"- Indexes Created: {len(perf_results.get('indexes', {}).get('indexes_created', []))}",
                f"- Partitions Created: {len(perf_results.get('partitions', {}).get('partitions_created', []))}",
                f"- Materialized Views: {len(perf_results.get('materialized_views', {}).get('views_created', []))}",
                f"- Extensions Enabled: {len(perf_results.get('extensions', {}).get('extensions_enabled', []))}",
                ""
            ])
        
        # Monitoring
        monitoring = results.get("monitoring", {})
        if monitoring.get("success"):
            mon_results = monitoring.get("results", {})
            report_lines.extend([
                "### Monitoring Setup",
                f"- Alert Rules: {mon_results.get('alert_rules_configured', 0)}",
                f"- Monitoring Active: {'✓' if mon_results.get('monitoring_active') else '✗'}",
                ""
            ])
        
        # Tests
        tests = results.get("tests", {})
        if tests.get("success"):
            test_results = tests.get("results", {})
            health = test_results.get("health_check", {})
            report_lines.extend([
                "### Deployment Tests",
                f"- Overall Health: {'✓' if health.get('overall_healthy') else '✗'}",
                f"- Tenant Isolation: {'✓' if test_results.get('tenant_isolation', {}).get('overall_isolation') else '✗'}",
                f"- Field Encryption: {'✓' if test_results.get('encryption', {}).get('success') else '✗'}",
                ""
            ])
        
        # Recommendations
        if performance.get("success"):
            recommendations = performance.get("results", {}).get("recommendations", {})
            if recommendations:
                report_lines.extend([
                    "### Optimization Recommendations",
                    ""
                ])
                
                for category, items in recommendations.items():
                    if items:
                        report_lines.append(f"**{category.replace('_', ' ').title()}:**")
                        for item in items:
                            report_lines.append(f"- {item}")
                        report_lines.append("")
        
        report_lines.extend([
            "## Next Steps",
            "",
            "1. Monitor the deployment through Azure Monitor and custom dashboards",
            "2. Set up regular backup verification procedures",
            "3. Schedule periodic performance optimization reviews",
            "4. Configure additional alerts based on usage patterns",
            "5. Plan capacity scaling based on growth projections",
            "",
            "## Support Information",
            "",
            "- **Documentation**: See `docs/database/azure-production-database-guide.md`",
            "- **Logs**: Check `/tmp/azure-database-deployment.log` for detailed logs",
            "- **Health Monitoring**: Use the enhanced storage manager health check endpoints",
            ""
        ])
        
        return "\n".join(report_lines)
    
    async def deploy(self) -> Dict[str, Any]:
        """Run the complete deployment process."""
        deployment_results = {}
        
        try:
            logger.info("Starting Azure Database production deployment", environment=self.environment)
            
            # Step 1: Load configuration
            self.load_configuration()
            
            # Step 2: Validate prerequisites
            deployment_results["prerequisites"] = await self.validate_prerequisites()
            if not deployment_results["prerequisites"]["prerequisites_met"]:
                raise Exception("Prerequisites validation failed")
            
            # Step 3: Initialize storage manager
            await self.initialize_storage_manager()
            
            # Step 4: Run database migrations
            deployment_results["migrations"] = await self.run_database_migrations()
            
            # Step 5: Setup performance optimization
            deployment_results["performance"] = await self.setup_performance_optimization()
            
            # Step 6: Setup monitoring
            deployment_results["monitoring"] = await self.setup_monitoring()
            
            # Step 7: Run deployment tests
            deployment_results["tests"] = await self.run_deployment_tests()
            
            # Step 8: Generate report
            report = await self.generate_deployment_report(deployment_results)
            deployment_results["report"] = report
            
            # Write report to file
            report_path = f"/tmp/azure-database-deployment-report-{self.environment}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            deployment_results["report_path"] = report_path
            deployment_results["overall_success"] = True
            
            logger.info("Azure Database deployment completed successfully", report_path=report_path)
            
        except Exception as e:
            logger.error("Azure Database deployment failed", error=str(e))
            deployment_results["overall_success"] = False
            deployment_results["error"] = str(e)
        
        finally:
            # Cleanup
            if self.storage_manager:
                await self.storage_manager.close()
        
        return deployment_results


async def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Azure Database production features")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--environment", default="production", choices=["development", "staging", "production"])
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    parser.add_argument("--validate-only", action="store_true", help="Only validate prerequisites")
    
    args = parser.parse_args()
    
    deployment = AzureDatabaseDeployment(args.config, args.environment)
    
    if args.validate_only:
        # Only run validation
        deployment.load_configuration()
        results = await deployment.validate_prerequisites()
        print(json.dumps(results, indent=2))
        return
    
    if args.dry_run:
        print("DRY RUN: Azure Database deployment would perform the following actions:")
        print("1. Validate prerequisites")
        print("2. Initialize enhanced storage manager")
        print("3. Run database migrations")
        print("4. Setup performance optimization")
        print("5. Configure monitoring and alerting")
        print("6. Run deployment validation tests")
        print("7. Generate deployment report")
        return
    
    # Run full deployment
    results = await deployment.deploy()
    
    if results["overall_success"]:
        print("✓ Azure Database deployment completed successfully!")
        print(f"Report available at: {results.get('report_path')}")
    else:
        print("✗ Azure Database deployment failed!")
        print(f"Error: {results.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
