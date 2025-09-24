"""
Service cluster management for multi-service testing.

This module provides utilities to manage a cluster of all three services:
- Core Mapper Service
- Detector Orchestration Service
- Analysis Service
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import httpx
import docker
from docker.models.containers import Container
import asyncpg
import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ServiceClusterConfig:
    """Configuration for service cluster."""
    
    # Service ports
    core_mapper_port: int = 8000
    detector_orchestration_port: int = 8001
    analysis_service_port: int = 8002
    
    # Database configurations
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "llama_mapper_test"
    postgres_user: str = "test_user"
    postgres_password: str = "test_password"
    
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 9000
    clickhouse_db: str = "llama_mapper_analytics_test"
    
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1
    
    # Service configurations
    use_docker: bool = True
    health_check_timeout: int = 30
    health_check_interval: int = 1
    
    # Service URLs (computed)
    @property
    def core_mapper_url(self) -> str:
        return f"http://localhost:{self.core_mapper_port}"
    
    @property
    def detector_orchestration_url(self) -> str:
        return f"http://localhost:{self.detector_orchestration_port}"
    
    @property
    def analysis_service_url(self) -> str:
        return f"http://localhost:{self.analysis_service_port}"


class ServiceContainer:
    """Manages a single service container."""
    
    def __init__(self, name: str, image: str, port: int, environment: Dict[str, str] = None):
        self.name = name
        self.image = image
        self.port = port
        self.environment = environment or {}
        self.container: Optional[Container] = None
        self.docker_client = docker.from_env()
    
    async def start(self) -> None:
        """Start the service container."""
        try:
            # Stop existing container if running
            await self.stop()
            
            # Start new container
            self.container = self.docker_client.containers.run(
                image=self.image,
                name=self.name,
                ports={f"{self.port}/tcp": self.port},
                environment=self.environment,
                detach=True,
                remove=True,
                network_mode="host"
            )
            
            logger.info(f"Started service container", service=self.name, port=self.port)
            
        except Exception as e:
            logger.error(f"Failed to start service container", service=self.name, error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop the service container."""
        try:
            # Find and stop existing container
            containers = self.docker_client.containers.list(
                filters={"name": self.name}
            )
            
            for container in containers:
                container.stop()
                logger.info(f"Stopped service container", service=self.name)
                
        except Exception as e:
            logger.warning(f"Error stopping service container", service=self.name, error=str(e))
    
    async def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        if not self.container:
            return False
        
        try:
            self.container.reload()
            return self.container.status == "running"
        except Exception:
            return False
    
    async def wait_for_health(self, timeout: int = 30) -> bool:
        """Wait for service to become healthy."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await self.is_healthy():
                # Additional HTTP health check
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"http://localhost:{self.port}/health",
                            timeout=5.0
                        )
                        if response.status_code == 200:
                            logger.info(f"Service is healthy", service=self.name)
                            return True
                except Exception:
                    pass
            
            await asyncio.sleep(1)
        
        logger.error(f"Service health check timeout", service=self.name)
        return False


class ServiceCluster:
    """Manages a cluster of all three services for testing."""
    
    def __init__(self, config: ServiceClusterConfig):
        self.config = config
        self.services: Dict[str, ServiceContainer] = {}
        self.clients: Dict[str, httpx.AsyncClient] = {}
        self._setup_services()
    
    def _setup_services(self) -> None:
        """Setup service containers."""
        # Core Mapper Service
        self.services["core_mapper"] = ServiceContainer(
            name="test-core-mapper",
            image="llama-mapper:test",
            port=self.config.core_mapper_port,
            environment={
                "LLAMA_MAPPER_DATABASE__HOST": self.config.postgres_host,
                "LLAMA_MAPPER_DATABASE__PORT": str(self.config.postgres_port),
                "LLAMA_MAPPER_DATABASE__NAME": self.config.postgres_db,
                "LLAMA_MAPPER_REDIS__HOST": self.config.redis_host,
                "LLAMA_MAPPER_REDIS__PORT": str(self.config.redis_port),
                "LLAMA_MAPPER_REDIS__DB": str(self.config.redis_db),
                "TESTING": "true",
                "LOG_LEVEL": "DEBUG"
            }
        )
        
        # Detector Orchestration Service
        self.services["detector_orchestration"] = ServiceContainer(
            name="test-detector-orchestration",
            image="detector-orchestration:test",
            port=self.config.detector_orchestration_port,
            environment={
                "ORCH_DATABASE__HOST": self.config.postgres_host,
                "ORCH_DATABASE__PORT": str(self.config.postgres_port),
                "ORCH_DATABASE__NAME": self.config.postgres_db,
                "ORCH_REDIS__HOST": self.config.redis_host,
                "ORCH_REDIS__PORT": str(self.config.redis_port),
                "ORCH_REDIS__DB": str(self.config.redis_db),
                "ORCH_MAPPER_SERVICE_URL": self.config.core_mapper_url,
                "TESTING": "true",
                "LOG_LEVEL": "DEBUG"
            }
        )
        
        # Analysis Service
        self.services["analysis_service"] = ServiceContainer(
            name="test-analysis-service",
            image="analysis-service:test",
            port=self.config.analysis_service_port,
            environment={
                "ANALYSIS_POSTGRES_HOST": self.config.postgres_host,
                "ANALYSIS_POSTGRES_PORT": str(self.config.postgres_port),
                "ANALYSIS_POSTGRES_DB": self.config.postgres_db,
                "ANALYSIS_CLICKHOUSE_HOST": self.config.clickhouse_host,
                "ANALYSIS_CLICKHOUSE_PORT": str(self.config.clickhouse_port),
                "ANALYSIS_CLICKHOUSE_DB": self.config.clickhouse_db,
                "TESTING": "true",
                "LOG_LEVEL": "DEBUG"
            }
        )
    
    async def start(self) -> None:
        """Start all services in the cluster."""
        logger.info("Starting service cluster")
        
        # Start services in dependency order
        startup_order = ["core_mapper", "detector_orchestration", "analysis_service"]
        
        for service_name in startup_order:
            service = self.services[service_name]
            await service.start()
            
            # Wait for service to be healthy
            if not await service.wait_for_health(self.config.health_check_timeout):
                raise RuntimeError(f"Service {service_name} failed to start properly")
        
        # Setup HTTP clients
        await self._setup_clients()
        
        logger.info("Service cluster started successfully")
    
    async def stop(self) -> None:
        """Stop all services in the cluster."""
        logger.info("Stopping service cluster")
        
        # Close HTTP clients
        await self._cleanup_clients()
        
        # Stop services in reverse order
        stop_order = ["analysis_service", "detector_orchestration", "core_mapper"]
        
        for service_name in stop_order:
            service = self.services[service_name]
            await service.stop()
        
        logger.info("Service cluster stopped")
    
    async def _setup_clients(self) -> None:
        """Setup HTTP clients for all services."""
        self.clients = {
            "core_mapper": httpx.AsyncClient(
                base_url=self.config.core_mapper_url,
                timeout=30.0,
                headers={"Content-Type": "application/json"}
            ),
            "detector_orchestration": httpx.AsyncClient(
                base_url=self.config.detector_orchestration_url,
                timeout=30.0,
                headers={"Content-Type": "application/json"}
            ),
            "analysis_service": httpx.AsyncClient(
                base_url=self.config.analysis_service_url,
                timeout=30.0,
                headers={"Content-Type": "application/json"}
            )
        }
    
    async def _cleanup_clients(self) -> None:
        """Cleanup HTTP clients."""
        for client in self.clients.values():
            await client.aclose()
        self.clients.clear()
    
    def get_mapper_client(self) -> httpx.AsyncClient:
        """Get HTTP client for Core Mapper service."""
        return self.clients["core_mapper"]
    
    def get_orchestration_client(self) -> httpx.AsyncClient:
        """Get HTTP client for Detector Orchestration service."""
        return self.clients["detector_orchestration"]
    
    def get_analysis_client(self) -> httpx.AsyncClient:
        """Get HTTP client for Analysis service."""
        return self.clients["analysis_service"]
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all services."""
        health_status = {}
        
        for service_name, service in self.services.items():
            health_status[service_name] = await service.is_healthy()
        
        return health_status
    
    async def wait_for_all_healthy(self, timeout: int = 60) -> bool:
        """Wait for all services to be healthy."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            health_status = await self.health_check()
            
            if all(health_status.values()):
                logger.info("All services are healthy")
                return True
            
            logger.debug("Waiting for services to be healthy", status=health_status)
            await asyncio.sleep(2)
        
        logger.error("Timeout waiting for all services to be healthy")
        return False
    
    async def execute_cross_service_request(
        self, 
        workflow_type: str, 
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a cross-service request workflow."""
        correlation_id = payload.get("correlation_id", "test-correlation-id")
        
        if workflow_type == "detection_to_analysis":
            # Orchestration -> Mapper -> Analysis workflow
            
            # Step 1: Submit to orchestration
            orch_response = await self.clients["detector_orchestration"].post(
                "/api/v1/orchestrate",
                json=payload,
                headers={"X-Correlation-ID": correlation_id}
            )
            
            if orch_response.status_code != 200:
                raise RuntimeError(f"Orchestration failed: {orch_response.text}")
            
            orch_result = orch_response.json()
            
            # Step 2: Map results (may be automatic if auto_map is enabled)
            if "mapping_result" not in orch_result:
                map_response = await self.clients["core_mapper"].post(
                    "/api/v1/map",
                    json=orch_result["mapper_payload"],
                    headers={"X-Correlation-ID": correlation_id}
                )
                
                if map_response.status_code != 200:
                    raise RuntimeError(f"Mapping failed: {map_response.text}")
                
                map_result = map_response.json()
            else:
                map_result = orch_result["mapping_result"]
            
            # Step 3: Submit to analysis
            analysis_response = await self.clients["analysis_service"].post(
                "/api/v1/analyze",
                json={
                    "mapping_result": map_result,
                    "correlation_id": correlation_id
                },
                headers={"X-Correlation-ID": correlation_id}
            )
            
            if analysis_response.status_code != 200:
                raise RuntimeError(f"Analysis failed: {analysis_response.text}")
            
            return {
                "orchestration_result": orch_result,
                "mapping_result": map_result,
                "analysis_result": analysis_response.json(),
                "correlation_id": correlation_id
            }
        
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    async def inject_failure(self, service_name: str, failure_type: str) -> None:
        """Inject failure into a specific service for chaos testing."""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        
        service = self.services[service_name]
        
        if failure_type == "service_crash":
            await service.stop()
            logger.info(f"Injected service crash", service=service_name)
        
        elif failure_type == "network_partition":
            # Simulate network partition by blocking traffic
            # This would typically involve iptables rules or container networking
            logger.info(f"Injected network partition", service=service_name)
            
        elif failure_type == "resource_exhaustion":
            # Simulate resource exhaustion
            # This would typically involve stress testing or resource limits
            logger.info(f"Injected resource exhaustion", service=service_name)
        
        else:
            raise ValueError(f"Unknown failure type: {failure_type}")
    
    async def recover_from_failure(self, service_name: str, failure_type: str) -> None:
        """Recover from injected failure."""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        
        service = self.services[service_name]
        
        if failure_type == "service_crash":
            await service.start()
            await service.wait_for_health(self.config.health_check_timeout)
            logger.info(f"Recovered from service crash", service=service_name)
        
        elif failure_type in ["network_partition", "resource_exhaustion"]:
            # Recovery logic for other failure types
            logger.info(f"Recovered from {failure_type}", service=service_name)
        
        else:
            raise ValueError(f"Unknown failure type: {failure_type}")
