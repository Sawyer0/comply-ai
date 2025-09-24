"""
Test environment management for multi-service testing.

This module provides utilities for managing test environments that include:
- Core Mapper Service
- Detector Orchestration Service
- Analysis Service
- Supporting infrastructure (databases, message queues, etc.)
"""

import asyncio
import os
import tempfile
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
import docker
from docker.models.networks import Network
from docker.models.volumes import Volume
import yaml
import structlog

logger = structlog.get_logger(__name__)


class TestEnvironmentManager:
    """Manages test environments for multi-service testing."""
    
    def __init__(self, test_config):
        self.config = test_config
        self.docker_client = docker.from_env()
        self.environments: Dict[str, Dict[str, Any]] = {}
        self.networks: Dict[str, Network] = {}
        self.volumes: Dict[str, Volume] = {}
        self.temp_dirs: List[Path] = []
        
        # Test environment templates
        self.service_templates = {
            "core_mapper": self._get_core_mapper_template(),
            "detector_orchestration": self._get_detector_orchestration_template(),
            "analysis_service": self._get_analysis_service_template(),
            "postgres": self._get_postgres_template(),
            "redis": self._get_redis_template(),
            "clickhouse": self._get_clickhouse_template()
        }
    
    async def setup(self) -> None:
        """Setup test environment manager."""
        logger.info("Setting up test environment manager")
        
        # Create shared test network
        await self._create_test_network()
        
        # Create shared volumes
        await self._create_test_volumes()
        
        logger.info("Test environment manager setup complete")
    
    async def cleanup(self) -> None:
        """Cleanup all test environments and resources."""
        logger.info("Cleaning up test environments")
        
        # Cleanup all environments
        for env_id in list(self.environments.keys()):
            await self.cleanup_environment(env_id)
        
        # Cleanup shared resources
        await self._cleanup_test_network()
        await self._cleanup_test_volumes()
        
        # Cleanup temporary directories
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        logger.info("Test environment cleanup complete")
    
    async def provision_isolated_environment(self, test_id: str) -> str:
        """Provision isolated test environment."""
        logger.info(f"Provisioning isolated environment", test_id=test_id)
        
        # Create isolated network for this test
        network_name = f"test-network-{test_id}"
        network = self.docker_client.networks.create(
            name=network_name,
            driver="bridge",
            internal=False
        )
        
        # Create environment configuration
        env_config = {
            "test_id": test_id,
            "network": network,
            "services": {},
            "databases": {},
            "ports": self._allocate_ports(test_id),
            "temp_dir": self._create_temp_dir(test_id)
        }
        
        # Start infrastructure services first
        await self._start_infrastructure_services(env_config)
        
        # Start application services
        await self._start_application_services(env_config)
        
        # Wait for all services to be ready
        await self._wait_for_environment_ready(env_config)
        
        self.environments[test_id] = env_config
        
        logger.info(f"Environment provisioned successfully", test_id=test_id)
        return test_id
    
    async def cleanup_environment(self, test_id: str) -> None:
        """Cleanup specific test environment."""
        if test_id not in self.environments:
            return
        
        logger.info(f"Cleaning up environment", test_id=test_id)
        
        env_config = self.environments[test_id]
        
        # Stop and remove services
        for service_name, container in env_config["services"].items():
            try:
                container.stop()
                container.remove()
                logger.debug(f"Removed service", service=service_name, test_id=test_id)
            except Exception as e:
                logger.warning(f"Error removing service", 
                             service=service_name, 
                             test_id=test_id, 
                             error=str(e))
        
        # Remove network
        try:
            env_config["network"].remove()
            logger.debug(f"Removed network", test_id=test_id)
        except Exception as e:
            logger.warning(f"Error removing network", test_id=test_id, error=str(e))
        
        # Cleanup temp directory
        temp_dir = env_config.get("temp_dir")
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        del self.environments[test_id]
        logger.info(f"Environment cleanup complete", test_id=test_id)
    
    async def setup_service_mesh(self, services: List[str]) -> Dict[str, Any]:
        """Setup service mesh for integration testing."""
        mesh_config = {
            "services": services,
            "network": self._create_mesh_network(),
            "service_discovery": {},
            "load_balancer": None
        }
        
        # Configure service discovery
        for service in services:
            mesh_config["service_discovery"][service] = {
                "name": service,
                "health_check": f"http://{service}:8000/health",
                "tags": ["test", "mesh"]
            }
        
        return mesh_config
    
    def get_environment_info(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a test environment."""
        if test_id not in self.environments:
            return None
        
        env_config = self.environments[test_id]
        
        return {
            "test_id": test_id,
            "status": "running",
            "services": list(env_config["services"].keys()),
            "ports": env_config["ports"],
            "network": env_config["network"].name,
            "endpoints": self._get_service_endpoints(env_config)
        }
    
    async def _create_test_network(self) -> None:
        """Create shared test network."""
        try:
            network = self.docker_client.networks.get("test-shared-network")
            self.networks["shared"] = network
        except docker.errors.NotFound:
            network = self.docker_client.networks.create(
                name="test-shared-network",
                driver="bridge"
            )
            self.networks["shared"] = network
    
    async def _create_test_volumes(self) -> None:
        """Create shared test volumes."""
        volume_names = ["test-postgres-data", "test-clickhouse-data", "test-redis-data"]
        
        for volume_name in volume_names:
            try:
                volume = self.docker_client.volumes.get(volume_name)
                self.volumes[volume_name] = volume
            except docker.errors.NotFound:
                volume = self.docker_client.volumes.create(name=volume_name)
                self.volumes[volume_name] = volume
    
    async def _cleanup_test_network(self) -> None:
        """Cleanup shared test network."""
        if "shared" in self.networks:
            try:
                self.networks["shared"].remove()
            except Exception as e:
                logger.warning(f"Error removing shared network", error=str(e))
    
    async def _cleanup_test_volumes(self) -> None:
        """Cleanup shared test volumes."""
        for volume_name, volume in self.volumes.items():
            try:
                volume.remove()
            except Exception as e:
                logger.warning(f"Error removing volume", volume=volume_name, error=str(e))
    
    async def _start_infrastructure_services(self, env_config: Dict[str, Any]) -> None:
        """Start infrastructure services (databases, etc.)."""
        test_id = env_config["test_id"]
        network = env_config["network"]
        ports = env_config["ports"]
        
        # Start PostgreSQL
        postgres_container = self.docker_client.containers.run(
            image="postgres:15",
            name=f"test-postgres-{test_id}",
            environment={
                "POSTGRES_DB": f"test_db_{test_id}",
                "POSTGRES_USER": "test_user",
                "POSTGRES_PASSWORD": "test_password"
            },
            ports={5432: ports["postgres"]},
            network=network.name,
            detach=True,
            remove=True
        )
        env_config["services"]["postgres"] = postgres_container
        
        # Start Redis
        redis_container = self.docker_client.containers.run(
            image="redis:7-alpine",
            name=f"test-redis-{test_id}",
            ports={6379: ports["redis"]},
            network=network.name,
            detach=True,
            remove=True
        )
        env_config["services"]["redis"] = redis_container
        
        # Start ClickHouse
        clickhouse_container = self.docker_client.containers.run(
            image="yandex/clickhouse-server:latest",
            name=f"test-clickhouse-{test_id}",
            environment={
                "CLICKHOUSE_DB": f"test_analytics_{test_id}",
                "CLICKHOUSE_USER": "test_user",
                "CLICKHOUSE_PASSWORD": "test_password"
            },
            ports={9000: ports["clickhouse"]},
            network=network.name,
            detach=True,
            remove=True
        )
        env_config["services"]["clickhouse"] = clickhouse_container
    
    async def _start_application_services(self, env_config: Dict[str, Any]) -> None:
        """Start application services."""
        test_id = env_config["test_id"]
        network = env_config["network"]
        ports = env_config["ports"]
        
        # Start Core Mapper Service
        mapper_container = self.docker_client.containers.run(
            image="llama-mapper:test",
            name=f"test-mapper-{test_id}",
            environment={
                "LLAMA_MAPPER_DATABASE__HOST": f"test-postgres-{test_id}",
                "LLAMA_MAPPER_DATABASE__PORT": "5432",
                "LLAMA_MAPPER_DATABASE__NAME": f"test_db_{test_id}",
                "LLAMA_MAPPER_REDIS__HOST": f"test-redis-{test_id}",
                "LLAMA_MAPPER_REDIS__PORT": "6379",
                "TESTING": "true"
            },
            ports={8000: ports["core_mapper"]},
            network=network.name,
            detach=True,
            remove=True
        )
        env_config["services"]["core_mapper"] = mapper_container
        
        # Start Detector Orchestration Service
        orch_container = self.docker_client.containers.run(
            image="detector-orchestration:test",
            name=f"test-orchestration-{test_id}",
            environment={
                "ORCH_DATABASE__HOST": f"test-postgres-{test_id}",
                "ORCH_DATABASE__PORT": "5432",
                "ORCH_DATABASE__NAME": f"test_db_{test_id}",
                "ORCH_REDIS__HOST": f"test-redis-{test_id}",
                "ORCH_REDIS__PORT": "6379",
                "ORCH_MAPPER_SERVICE_URL": f"http://test-mapper-{test_id}:8000",
                "TESTING": "true"
            },
            ports={8000: ports["detector_orchestration"]},
            network=network.name,
            detach=True,
            remove=True
        )
        env_config["services"]["detector_orchestration"] = orch_container
        
        # Start Analysis Service
        analysis_container = self.docker_client.containers.run(
            image="analysis-service:test",
            name=f"test-analysis-{test_id}",
            environment={
                "ANALYSIS_POSTGRES_HOST": f"test-postgres-{test_id}",
                "ANALYSIS_POSTGRES_PORT": "5432",
                "ANALYSIS_POSTGRES_DB": f"test_db_{test_id}",
                "ANALYSIS_CLICKHOUSE_HOST": f"test-clickhouse-{test_id}",
                "ANALYSIS_CLICKHOUSE_PORT": "9000",
                "ANALYSIS_CLICKHOUSE_DB": f"test_analytics_{test_id}",
                "TESTING": "true"
            },
            ports={8000: ports["analysis_service"]},
            network=network.name,
            detach=True,
            remove=True
        )
        env_config["services"]["analysis_service"] = analysis_container
    
    async def _wait_for_environment_ready(self, env_config: Dict[str, Any]) -> None:
        """Wait for all services in environment to be ready."""
        max_wait_time = 120  # 2 minutes
        check_interval = 5   # 5 seconds
        
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < max_wait_time:
            all_healthy = True
            
            for service_name, container in env_config["services"].items():
                try:
                    container.reload()
                    if container.status != "running":
                        all_healthy = False
                        break
                    
                    # Additional health checks for application services
                    if service_name in ["core_mapper", "detector_orchestration", "analysis_service"]:
                        # Would typically do HTTP health check here
                        pass
                        
                except Exception:
                    all_healthy = False
                    break
            
            if all_healthy:
                logger.info(f"Environment ready", test_id=env_config["test_id"])
                return
            
            await asyncio.sleep(check_interval)
        
        raise RuntimeError(f"Environment failed to become ready within {max_wait_time}s")
    
    def _allocate_ports(self, test_id: str) -> Dict[str, int]:
        """Allocate unique ports for test environment."""
        # Use hash of test_id to get consistent port allocation
        base_port = 20000 + (hash(test_id) % 10000)
        
        return {
            "core_mapper": base_port,
            "detector_orchestration": base_port + 1,
            "analysis_service": base_port + 2,
            "postgres": base_port + 10,
            "redis": base_port + 11,
            "clickhouse": base_port + 12
        }
    
    def _create_temp_dir(self, test_id: str) -> Path:
        """Create temporary directory for test environment."""
        temp_dir = Path(tempfile.mkdtemp(prefix=f"test-env-{test_id}-"))
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def _create_mesh_network(self) -> Network:
        """Create network for service mesh testing."""
        return self.docker_client.networks.create(
            name="test-service-mesh",
            driver="bridge",
            options={
                "com.docker.network.bridge.enable_icc": "true",
                "com.docker.network.driver.mtu": "1500"
            }
        )
    
    def _get_service_endpoints(self, env_config: Dict[str, Any]) -> Dict[str, str]:
        """Get service endpoints for environment."""
        ports = env_config["ports"]
        
        return {
            "core_mapper": f"http://localhost:{ports['core_mapper']}",
            "detector_orchestration": f"http://localhost:{ports['detector_orchestration']}",
            "analysis_service": f"http://localhost:{ports['analysis_service']}",
            "postgres": f"postgresql://test_user:test_password@localhost:{ports['postgres']}/test_db_{env_config['test_id']}",
            "redis": f"redis://localhost:{ports['redis']}/0",
            "clickhouse": f"http://localhost:{ports['clickhouse']}"
        }
    
    # Service template methods
    def _get_core_mapper_template(self) -> Dict[str, Any]:
        """Get Core Mapper service template."""
        return {
            "image": "llama-mapper:test",
            "environment": {
                "TESTING": "true",
                "LOG_LEVEL": "DEBUG"
            },
            "health_check": "/health",
            "dependencies": ["postgres", "redis"]
        }
    
    def _get_detector_orchestration_template(self) -> Dict[str, Any]:
        """Get Detector Orchestration service template."""
        return {
            "image": "detector-orchestration:test",
            "environment": {
                "TESTING": "true",
                "LOG_LEVEL": "DEBUG"
            },
            "health_check": "/health",
            "dependencies": ["postgres", "redis", "core_mapper"]
        }
    
    def _get_analysis_service_template(self) -> Dict[str, Any]:
        """Get Analysis service template."""
        return {
            "image": "analysis-service:test",
            "environment": {
                "TESTING": "true",
                "LOG_LEVEL": "DEBUG"
            },
            "health_check": "/health",
            "dependencies": ["postgres", "clickhouse"]
        }
    
    def _get_postgres_template(self) -> Dict[str, Any]:
        """Get PostgreSQL service template."""
        return {
            "image": "postgres:15",
            "environment": {
                "POSTGRES_USER": "test_user",
                "POSTGRES_PASSWORD": "test_password"
            },
            "health_check": "pg_isready -U test_user"
        }
    
    def _get_redis_template(self) -> Dict[str, Any]:
        """Get Redis service template."""
        return {
            "image": "redis:7-alpine",
            "health_check": "redis-cli ping"
        }
    
    def _get_clickhouse_template(self) -> Dict[str, Any]:
        """Get ClickHouse service template."""
        return {
            "image": "yandex/clickhouse-server:latest",
            "environment": {
                "CLICKHOUSE_USER": "test_user",
                "CLICKHOUSE_PASSWORD": "test_password"
            },
            "health_check": "clickhouse-client --query='SELECT 1'"
        }
