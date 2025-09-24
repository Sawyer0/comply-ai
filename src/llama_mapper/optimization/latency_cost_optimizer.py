"""
Latency & Cost Optimization for Compliance AI Models

Consider stepping Mapper LoRA rank down (256 → 64/128) if latency matters;
Cache (detector, text) → mapping for 1–24h. Huge real-world hit-rates.
"""

import hashlib
import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass

# import redis  # External dependency - would need to be installed
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class LatencyConfig:
    """Configuration for latency optimization."""

    # LoRA rank optimization
    mapper_lora_rank: int = 128  # Reduced from 256 for latency
    mapper_lora_alpha: int = 256  # 2x rank

    # Caching configuration
    cache_ttl: int = 3600  # 1 hour default
    max_cache_size: int = 10000
    cache_hit_threshold: float = 0.8  # 80% hit rate target

    # Performance targets
    target_latency_p95: float = 100.0  # 100ms p95 latency
    target_throughput: int = 1000  # 1000 requests/second

    # Cost optimization
    enable_quantization: bool = True
    enable_gradient_checkpointing: bool = True
    enable_dynamic_batching: bool = True


@dataclass
class CacheEntry:
    """Cache entry for mapping results."""

    key: str
    mapping_result: Dict[str, Any]
    timestamp: float
    ttl: int
    hit_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""

    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    cache_hit_rate: float
    memory_usage: float
    gpu_utilization: float
    timestamp: float


class MappingCache:
    """High-performance cache for (detector, text) → mapping results."""

    def __init__(self, config: LatencyConfig, cache_backend: str = "memory"):
        self.config = config
        self.cache_backend = cache_backend
        self.redis_client: Optional[Any] = None  # Will be None if Redis not available
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0, "total_requests": 0}
        self.lock = threading.Lock()

        # Initialize cache backend
        if cache_backend == "redis":
            # self.redis_client = redis.Redis(host='localhost', port=6379, db=0)  # External dependency
            self.redis_client = None  # Simplified for now
        else:
            self.redis_client = None

    def get_cache_key(self, detector_output: str, context: str = "") -> str:
        """Generate cache key for detector output and context."""
        content = f"{detector_output}|{context}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached mapping result."""
        with self.lock:
            self.cache_stats["total_requests"] += 1

            if self.cache_backend == "redis":
                return self._get_from_redis(cache_key)
            else:
                return self._get_from_memory(cache_key)

    def _get_from_memory(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get from memory cache."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]

            # Check TTL
            if time.time() - entry.timestamp < entry.ttl:
                entry.hit_count += 1
                self.cache_stats["hits"] += 1
                return entry.mapping_result
            else:
                # Expired, remove
                del self.cache[cache_key]

        self.cache_stats["misses"] += 1
        return None

    def _get_from_redis(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get from Redis cache."""
        try:
            cached_data = (
                self.redis_client.get(cache_key)
                if self.redis_client and hasattr(self.redis_client, "get")
                else None
            )
            if cached_data:
                entry_data = json.loads(cached_data)
                entry = CacheEntry(**entry_data)

                # Check TTL
                if time.time() - entry.timestamp < entry.ttl:
                    entry.hit_count += 1
                    self.cache_stats["hits"] += 1
                    return entry.mapping_result
                else:
                    # Expired, remove
                    if self.redis_client and hasattr(self.redis_client, "delete"):
                        self.redis_client.delete(cache_key)

            self.cache_stats["misses"] += 1
            return None

        except Exception as e:
            logging.error("Redis cache error: %s", e)
            self.cache_stats["misses"] += 1
            return None

    def put(
        self, cache_key: str, mapping_result: Dict[str, Any], ttl: Optional[int] = None
    ) -> None:
        """Put mapping result in cache."""
        if ttl is None:
            ttl = self.config.cache_ttl

        entry = CacheEntry(
            key=cache_key, mapping_result=mapping_result, timestamp=time.time(), ttl=ttl
        )

        with self.lock:
            if self.cache_backend == "redis":
                self._put_to_redis(cache_key, entry)
            else:
                self._put_to_memory(cache_key, entry)

    def _put_to_memory(self, cache_key: str, entry: CacheEntry) -> None:
        """Put to memory cache with LRU eviction."""
        # Check cache size
        if len(self.cache) >= self.config.max_cache_size:
            # Evict least recently used entry
            lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            del self.cache[lru_key]
            self.cache_stats["evictions"] += 1

        self.cache[cache_key] = entry

    def _put_to_redis(self, cache_key: str, entry: CacheEntry) -> None:
        """Put to Redis cache."""
        try:
            entry_data = json.dumps(asdict(entry))
            if self.redis_client and hasattr(self.redis_client, "setex"):
                self.redis_client.setex(cache_key, entry.ttl, entry_data)
        except Exception as e:
            logging.error("Redis cache put error: %s", e)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats["total_requests"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        )

        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "cache_size": (
                len(self.cache) if self.cache_backend == "memory" else "unknown"
            ),
            "backend": self.cache_backend,
        }

    def clear_cache(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            if self.cache_backend == "redis":
                try:
                    if self.redis_client and hasattr(self.redis_client, "flushdb"):
                        self.redis_client.flushdb()
                except Exception as e:
                    logging.error("Redis cache clear error: %s", e)
            else:
                self.cache.clear()

            # Reset stats
            self.cache_stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "total_requests": 0,
            }


class LatencyOptimizer:
    """Optimizes model latency and cost."""

    def __init__(self, config: LatencyConfig):
        self.config = config
        self.cache = MappingCache(config)
        self.performance_history = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)

    def optimize_mapper_config(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize Mapper configuration for latency."""
        optimized_config = current_config.copy()

        # Reduce LoRA rank for latency
        optimized_config["lora_r"] = self.config.mapper_lora_rank
        optimized_config["lora_alpha"] = self.config.mapper_lora_alpha

        # Enable optimizations
        if self.config.enable_quantization:
            optimized_config["quantization_config"] = {
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
            }

        if self.config.enable_gradient_checkpointing:
            optimized_config["gradient_checkpointing"] = True

        # Optimize generation parameters
        optimized_config["generation_config"] = {
            "max_new_tokens": 50,  # Very small for Mapper
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "early_stopping": True,
        }

        return optimized_config

    def optimize_analyst_config(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize Analyst configuration for latency."""
        optimized_config = current_config.copy()

        # Keep Analyst LoRA small for efficiency
        optimized_config["lora_r"] = 16  # Small rank
        optimized_config["lora_alpha"] = 32  # 2x rank

        # Enable optimizations
        if self.config.enable_quantization:
            optimized_config["quantization_config"] = {
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
            }

        # Optimize generation parameters
        optimized_config["generation_config"] = {
            "max_new_tokens": 200,  # Concise for Analyst
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "early_stopping": True,
        }

        return optimized_config

    def get_cached_mapping(
        self, detector_output: str, context: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Get cached mapping result."""
        cache_key = self.cache.get_cache_key(detector_output, context)
        return self.cache.get(cache_key)

    def cache_mapping_result(
        self,
        detector_output: str,
        mapping_result: Dict[str, Any],
        context: str = "",
        ttl: Optional[int] = None,
    ) -> None:
        """Cache mapping result."""
        cache_key = self.cache.get_cache_key(detector_output, context)
        self.cache.put(cache_key, mapping_result, ttl)

    def record_performance(
        self,
        latency: float,
        throughput: float,
        memory_usage: float,
        gpu_utilization: float,
    ) -> None:
        """Record performance metrics."""
        metrics = PerformanceMetrics(
            latency_p50=latency,
            latency_p95=latency * 1.5,  # Approximate p95
            latency_p99=latency * 2.0,  # Approximate p99
            throughput=throughput,
            cache_hit_rate=self.cache.get_cache_stats()["hit_rate"],
            memory_usage=memory_usage,
            gpu_utilization=gpu_utilization,
            timestamp=time.time(),
        )

        self.performance_history.append(metrics)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.performance_history:
            return {"error": "No performance data available"}

        latencies = [m.latency_p50 for m in self.performance_history]
        throughputs = [m.throughput for m in self.performance_history]
        cache_hit_rates = [m.cache_hit_rate for m in self.performance_history]

        return {
            "latency_p50": float(np.mean(latencies)),
            "latency_p95": float(np.percentile(latencies, 95)),
            "latency_p99": float(np.percentile(latencies, 99)),
            "throughput_avg": float(np.mean(throughputs)),
            "cache_hit_rate_avg": float(np.mean(cache_hit_rates)),
            "total_requests": len(self.performance_history),
            "cache_stats": self.cache.get_cache_stats(),
        }

    def should_optimize_further(self) -> bool:
        """Determine if further optimization is needed."""
        summary = self.get_performance_summary()

        if "error" in summary:
            return False

        # Check if we're meeting targets
        latency_p95 = summary["latency_p95"]
        throughput = summary["throughput_avg"]
        cache_hit_rate = summary["cache_hit_rate_avg"]

        return (
            latency_p95 > self.config.target_latency_p95
            or throughput < self.config.target_throughput
            or cache_hit_rate < self.config.cache_hit_threshold
        )

    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations."""
        recommendations = []
        summary = self.get_performance_summary()

        if "error" in summary:
            return ["No performance data available for recommendations"]

        latency_p95 = summary["latency_p95"]
        throughput = summary["throughput_avg"]
        cache_hit_rate = summary["cache_hit_rate_avg"]

        if latency_p95 > self.config.target_latency_p95:
            recommendations.append(
                f"Latency p95 ({latency_p95:.1f}ms) exceeds target ({self.config.target_latency_p95}ms) - consider reducing LoRA rank further"
            )

        if throughput < self.config.target_throughput:
            recommendations.append(
                f"Throughput ({throughput:.1f} req/s) below target ({self.config.target_throughput} req/s) - consider dynamic batching"
            )

        if cache_hit_rate < self.config.cache_hit_threshold:
            recommendations.append(
                f"Cache hit rate ({cache_hit_rate:.1%}) below target ({self.config.cache_hit_threshold:.1%}) - consider increasing cache TTL"
            )

        if not recommendations:
            recommendations.append(
                "Performance targets are being met - no optimization needed"
            )

        return recommendations


class DynamicBatching:
    """Dynamic batching for improved throughput."""

    def __init__(self, max_batch_size: int = 8, max_wait_time: float = 0.01):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.batch_queue = []
        self.lock = threading.Lock()
        self.batch_stats = {
            "total_batches": 0,
            "total_requests": 0,
            "avg_batch_size": 0.0,
        }

    def add_request(self, request: Dict[str, Any]) -> None:
        """Add request to batch queue."""
        with self.lock:
            self.batch_queue.append(request)
            self.batch_stats["total_requests"] += 1

    def get_batch(self) -> List[Dict[str, Any]]:
        """Get batch of requests for processing."""
        with self.lock:
            if not self.batch_queue:
                return []

            # Take up to max_batch_size requests
            batch_size = min(len(self.batch_queue), self.max_batch_size)
            batch = self.batch_queue[:batch_size]
            self.batch_queue = self.batch_queue[batch_size:]

            self.batch_stats["total_batches"] += 1
            self.batch_stats["avg_batch_size"] = (
                self.batch_stats["avg_batch_size"]
                * (self.batch_stats["total_batches"] - 1)
                + batch_size
            ) / self.batch_stats["total_batches"]

            return batch

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch statistics."""
        return self.batch_stats.copy()


class CostOptimizer:
    """Optimizes costs through various strategies."""

    def __init__(self, config: LatencyConfig):
        self.config = config
        self.cost_tracking = {
            "total_requests": 0,
            "cache_hits": 0,
            "model_inferences": 0,
            "estimated_cost": 0.0,
        }

    def calculate_cost_savings(self) -> Dict[str, Any]:
        """Calculate cost savings from optimizations."""
        cache_stats = self.cost_tracking
        total_requests = cache_stats["total_requests"]

        if total_requests == 0:
            return {"error": "No requests processed"}

        cache_hit_rate = cache_stats["cache_hits"] / total_requests
        model_inferences = cache_stats["model_inferences"]

        # Estimate costs (these would be real costs in production)
        cost_per_inference = 0.001  # $0.001 per inference
        total_cost = model_inferences * cost_per_inference

        # Calculate savings from caching
        requests_without_cache = total_requests
        requests_with_cache = model_inferences
        cache_savings = (
            requests_without_cache - requests_with_cache
        ) * cost_per_inference

        return {
            "total_requests": total_requests,
            "cache_hit_rate": cache_hit_rate,
            "model_inferences": model_inferences,
            "total_cost": total_cost,
            "cache_savings": cache_savings,
            "savings_percentage": (
                (cache_savings / (total_cost + cache_savings)) * 100
                if (total_cost + cache_savings) > 0
                else 0
            ),
        }

    def record_request(self, used_cache: bool = False) -> None:
        """Record a request for cost tracking."""
        self.cost_tracking["total_requests"] += 1

        if used_cache:
            self.cost_tracking["cache_hits"] += 1
        else:
            self.cost_tracking["model_inferences"] += 1


# Example usage and testing
if __name__ == "__main__":
    # Create latency configuration
    config = LatencyConfig(
        mapper_lora_rank=128,  # Reduced from 256
        cache_ttl=3600,  # 1 hour
        target_latency_p95=100.0,  # 100ms
        target_throughput=1000,  # 1000 req/s
    )

    # Create optimizer
    optimizer = LatencyOptimizer(config)

    # Test caching
    detector_output = "email address detected: john@company.com"
    mapping_result = {
        "taxonomy": ["PII.Contact.Email"],
        "scores": {"PII.Contact.Email": 0.95},
        "confidence": 0.95,
    }

    # Cache the result
    optimizer.cache_mapping_result(detector_output, mapping_result)

    # Try to get from cache
    cached_result = optimizer.get_cached_mapping(detector_output)
    print(f"Cached result: {cached_result}")

    # Get performance summary
    performance_summary = optimizer.get_performance_summary()
    print(f"Performance summary: {performance_summary}")

    # Get optimization recommendations
    recommendations = optimizer.get_optimization_recommendations()
    print(f"Recommendations: {recommendations}")

    print(f"\n🎉 Latency & Cost Optimization System Ready!")
    print(f"  - Mapper LoRA rank: {config.mapper_lora_rank} (reduced from 256)")
    print(f"  - Cache TTL: {config.cache_ttl} seconds")
    print(f"  - Target latency p95: {config.target_latency_p95}ms")
    print(f"  - Target throughput: {config.target_throughput} req/s")
