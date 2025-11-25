"""
Service registry for inter-service communication and discovery.

This module provides service discovery capabilities for microservices
to find and communicate with each other.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import redis.asyncio as redis

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ServiceRegistry:
    """Service registry for microservice discovery."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize service registry.
        
        Args:
            redis_url: Redis connection URL for service registry storage
        """
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.service_ttl = 30  # Service registration TTL in seconds
        
    async def _get_redis(self) -> redis.Redis:
        """Get Redis client connection."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(self.redis_url)
        return self.redis_client
    
    def _get_service_key(self, service_name: str) -> str:
        """Get Redis key for service registration."""
        return f"service_registry:{service_name}"
    
    async def register_service(
        self, 
        service_name: str, 
        service_url: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a service in the registry.
        
        Args:
            service_name: Name of the service
            service_url: URL/endpoint of the service
            metadata: Additional service metadata
        """
        try:
            client = await self._get_redis()
            
            service_data = {
                "name": service_name,
                "url": service_url,
                "metadata": metadata or {},
                "registered_at": datetime.utcnow().isoformat(),
                "last_heartbeat": datetime.utcnow().isoformat()
            }
            
            key = self._get_service_key(service_name)
            await client.setex(key, self.service_ttl, json.dumps(service_data))
            
            logger.info(
                "Service registered successfully",
                service_name=service_name,
                service_url=service_url,
                ttl=self.service_ttl
            )
            
        except Exception as e:
            logger.error(
                "Failed to register service",
                service_name=service_name,
                error=str(e)
            )
            raise
    
    async def unregister_service(self, service_name: str) -> None:
        """Unregister a service from the registry.
        
        Args:
            service_name: Name of the service to unregister
        """
        try:
            client = await self._get_redis()
            key = self._get_service_key(service_name)
            await client.delete(key)
            
            logger.info("Service unregistered", service_name=service_name)
            
        except Exception as e:
            logger.error(
                "Failed to unregister service",
                service_name=service_name,
                error=str(e)
            )
            raise
    
    async def discover_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Discover a service by name.
        
        Args:
            service_name: Name of the service to discover
            
        Returns:
            Service information if found, None otherwise
        """
        try:
            client = await self._get_redis()
            key = self._get_service_key(service_name)
            
            service_data = await client.get(key)
            if service_data:
                return json.loads(service_data.decode())
            
            return None
            
        except Exception as e:
            logger.error(
                "Failed to discover service",
                service_name=service_name,
                error=str(e)
            )
            return None
    
    async def list_services(self) -> List[Dict[str, Any]]:
        """List all registered services.
        
        Returns:
            List of registered services
        """
        try:
            client = await self._get_redis()
            keys = await client.keys("service_registry:*")
            
            services = []
            for key in keys:
                service_data = await client.get(key)
                if service_data:
                    services.append(json.loads(service_data.decode()))
            
            return services
            
        except Exception as e:
            logger.error("Failed to list services", error=str(e))
            return []
    
    async def health_check_service(self, service_name: str) -> bool:
        """Check if a service is healthy.
        
        Args:
            service_name: Name of the service
            
        Returns:
            True if service is healthy, False otherwise
        """
        service_info = await self.discover_service(service_name)
        if not service_info:
            return False
        
        # Check if service registration is recent
        try:
            last_heartbeat = datetime.fromisoformat(service_info["last_heartbeat"])
            if datetime.utcnow() - last_heartbeat > timedelta(seconds=self.service_ttl * 2):
                return False
            return True
        except Exception:
            return False
    
    async def heartbeat(self, service_name: str) -> None:
        """Send heartbeat for a service to keep it alive.
        
        Args:
            service_name: Name of the service
        """
        try:
            service_info = await self.discover_service(service_name)
            if service_info:
                service_info["last_heartbeat"] = datetime.utcnow().isoformat()
                await self.register_service(
                    service_name,
                    service_info["url"],
                    service_info["metadata"]
                )
        except Exception as e:
            logger.error(
                "Failed to send heartbeat",
                service_name=service_name,
                error=str(e)
            )
    
    async def start_heartbeat_task(self, service_name: str, interval: int = 10):
        """Start periodic heartbeat task for a service.
        
        Args:
            service_name: Name of the service
            interval: Heartbeat interval in seconds
        """
        async def heartbeat_loop():
            while True:
                try:
                    await self.heartbeat(service_name)
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        "Heartbeat task error",
                        service_name=service_name,
                        error=str(e)
                    )
                    await asyncio.sleep(interval)
        
        return asyncio.create_task(heartbeat_loop())
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


DatabaseServiceRegistry = ServiceRegistry
