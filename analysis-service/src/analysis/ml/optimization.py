"""
ML Model Optimization

This module provides optimization capabilities for ML models in the Analysis Service,
including performance optimization, caching, and resource management.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import json
import hashlib

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    ML model optimization system for Analysis Service.

    Provides:
    - Model performance optimization
    - Inference caching
    - Resource management
    - Batch processing optimization
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimization_config = config.get("optimization", {})

        # Caching configuration
        self.cache_enabled = self.optimization_config.get("cache_enabled", True)
        self.cache_ttl = self.optimization_config.get(
            "cache_ttl_seconds", 3600
        )  # 1 hour
        self.max_cache_size = self.optimization_config.get("max_cache_size", 1000)

        # Batch processing configuration
        self.batch_size = self.optimization_config.get("batch_size", 8)
        self.batch_timeout = self.optimization_config.get("batch_timeout_seconds", 1.0)

        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        self.total_inference_time = 0.0

        # Cache storage
        self.inference_cache = {}
        self.cache_timestamps = {}

        # Batch processing
        self.pending_requests = []
        self.batch_processor_running = False

        logger.info(
            "Model Optimizer initialized",
            cache_enabled=self.cache_enabled,
            batch_size=self.batch_size,
        )

    async def optimize_inference(
        self, model_callable, input_data: Any, cache_key: Optional[str] = None, **kwargs
    ) -> Any:
        """
        Optimize model inference with caching and batching.

        Args:
            model_callable: Model function to call
            input_data: Input data for the model
            cache_key: Optional cache key (auto-generated if not provided)
            **kwargs: Additional arguments for the model

        Returns:
            Model output
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Generate cache key if not provided
            if cache_key is None:
                cache_key = self._generate_cache_key(input_data, kwargs)

            # Check cache first
            if self.cache_enabled:
                cached_result = self._get_cached_result(cache_key)
                if cached_result is not None:
                    self.cache_hits += 1
                    self.total_requests += 1

                    logger.debug("Cache hit for inference", cache_key=cache_key[:16])
                    return cached_result

            self.cache_misses += 1

            # Perform inference
            result = await self._execute_inference(model_callable, input_data, **kwargs)

            # Cache result
            if self.cache_enabled:
                self._cache_result(cache_key, result)

            # Track performance
            inference_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.total_requests += 1
            self.total_inference_time += inference_time

            logger.debug(
                "Inference completed",
                cache_key=cache_key[:16],
                inference_time=inference_time,
            )

            return result

        except Exception as e:
            logger.error("Optimized inference failed", error=str(e))
            raise

    async def batch_optimize_inference(
        self, model_callable, input_batch: List[Any], **kwargs
    ) -> List[Any]:
        """
        Optimize batch inference with intelligent batching.

        Args:
            model_callable: Model function to call
            input_batch: List of input data
            **kwargs: Additional arguments for the model

        Returns:
            List of model outputs
        """
        try:
            # Check cache for each input
            results = []
            uncached_inputs = []
            uncached_indices = []

            for i, input_data in enumerate(input_batch):
                cache_key = self._generate_cache_key(input_data, kwargs)

                if self.cache_enabled:
                    cached_result = self._get_cached_result(cache_key)
                    if cached_result is not None:
                        results.append(cached_result)
                        self.cache_hits += 1
                        continue

                # Input not in cache
                results.append(None)  # Placeholder
                uncached_inputs.append(input_data)
                uncached_indices.append(i)
                self.cache_misses += 1

            # Process uncached inputs in optimized batches
            if uncached_inputs:
                uncached_results = await self._batch_process_inputs(
                    model_callable, uncached_inputs, **kwargs
                )

                # Insert results and cache them
                for idx, result in zip(uncached_indices, uncached_results):
                    results[idx] = result

                    if self.cache_enabled:
                        cache_key = self._generate_cache_key(input_batch[idx], kwargs)
                        self._cache_result(cache_key, result)

            self.total_requests += len(input_batch)

            return results

        except Exception as e:
            logger.error("Batch optimized inference failed", error=str(e))
            raise

    async def _execute_inference(
        self, model_callable, input_data: Any, **kwargs
    ) -> Any:
        """Execute single inference with optimization."""
        try:
            # Add to batch queue if batch processing is enabled
            if self.optimization_config.get("enable_batching", False):
                return await self._add_to_batch_queue(
                    model_callable, input_data, **kwargs
                )
            else:
                # Direct inference
                return await model_callable(input_data, **kwargs)

        except Exception as e:
            logger.error("Inference execution failed", error=str(e))
            raise

    async def _batch_process_inputs(
        self, model_callable, inputs: List[Any], **kwargs
    ) -> List[Any]:
        """Process inputs in optimized batches."""
        try:
            results = []

            # Process in batches
            for i in range(0, len(inputs), self.batch_size):
                batch = inputs[i : i + self.batch_size]

                # Process batch
                if hasattr(model_callable, "batch_process"):
                    # Model supports batch processing
                    batch_results = await model_callable.batch_process(batch, **kwargs)
                else:
                    # Process individually but concurrently
                    tasks = [
                        model_callable(input_data, **kwargs) for input_data in batch
                    ]
                    batch_results = await asyncio.gather(*tasks)

                results.extend(batch_results)

            return results

        except Exception as e:
            logger.error("Batch processing failed", error=str(e))
            raise

    async def _add_to_batch_queue(
        self, model_callable, input_data: Any, **kwargs
    ) -> Any:
        """Add request to batch processing queue."""
        try:
            # Create request future
            request_future = asyncio.Future()

            # Add to pending requests
            self.pending_requests.append(
                {
                    "callable": model_callable,
                    "input": input_data,
                    "kwargs": kwargs,
                    "future": request_future,
                    "timestamp": datetime.now(timezone.utc),
                }
            )

            # Start batch processor if not running
            if not self.batch_processor_running:
                asyncio.create_task(self._batch_processor())

            # Wait for result
            return await request_future

        except Exception as e:
            logger.error("Failed to add to batch queue", error=str(e))
            raise

    async def _batch_processor(self):
        """Background batch processor."""
        self.batch_processor_running = True

        try:
            while self.pending_requests:
                # Wait for batch to fill or timeout
                await asyncio.sleep(self.batch_timeout)

                if not self.pending_requests:
                    break

                # Extract batch
                batch_size = min(len(self.pending_requests), self.batch_size)
                batch_requests = self.pending_requests[:batch_size]
                self.pending_requests = self.pending_requests[batch_size:]

                # Group by callable (assuming same callable for simplicity)
                if batch_requests:
                    try:
                        # Process batch
                        inputs = [req["input"] for req in batch_requests]
                        kwargs = batch_requests[0]["kwargs"]  # Assume same kwargs
                        callable_func = batch_requests[0]["callable"]

                        results = await self._batch_process_inputs(
                            callable_func, inputs, **kwargs
                        )

                        # Set results
                        for request, result in zip(batch_requests, results):
                            request["future"].set_result(result)

                    except Exception as e:
                        # Set exception for all requests in batch
                        for request in batch_requests:
                            request["future"].set_exception(e)

        finally:
            self.batch_processor_running = False

    def _generate_cache_key(self, input_data: Any, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for input data and parameters."""
        try:
            # Create deterministic string representation
            input_str = json.dumps(input_data, sort_keys=True, default=str)
            kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)

            # Generate hash
            combined_str = f"{input_str}:{kwargs_str}"
            cache_key = hashlib.sha256(combined_str.encode()).hexdigest()

            return cache_key

        except Exception as e:
            logger.error("Failed to generate cache key", error=str(e))
            # Fallback to simple hash
            return hashlib.sha256(str(input_data).encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get result from cache if valid."""
        try:
            if cache_key not in self.inference_cache:
                return None

            # Check if cache entry is still valid
            timestamp = self.cache_timestamps.get(cache_key)
            if timestamp:
                age = (datetime.now(timezone.utc) - timestamp).total_seconds()
                if age > self.cache_ttl:
                    # Cache expired
                    del self.inference_cache[cache_key]
                    del self.cache_timestamps[cache_key]
                    return None

            return self.inference_cache[cache_key]

        except Exception as e:
            logger.error("Failed to get cached result", error=str(e))
            return None

    def _cache_result(self, cache_key: str, result: Any):
        """Cache inference result."""
        try:
            # Check cache size limit
            if len(self.inference_cache) >= self.max_cache_size:
                self._evict_oldest_cache_entries()

            # Store result and timestamp
            self.inference_cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now(timezone.utc)

        except Exception as e:
            logger.error("Failed to cache result", error=str(e))

    def _evict_oldest_cache_entries(self):
        """Evict oldest cache entries to make space."""
        try:
            # Sort by timestamp and remove oldest 20%
            sorted_entries = sorted(self.cache_timestamps.items(), key=lambda x: x[1])

            entries_to_remove = len(sorted_entries) // 5  # Remove 20%

            for cache_key, _ in sorted_entries[:entries_to_remove]:
                del self.inference_cache[cache_key]
                del self.cache_timestamps[cache_key]

            logger.debug("Evicted old cache entries", count=entries_to_remove)

        except Exception as e:
            logger.error("Failed to evict cache entries", error=str(e))

    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization performance metrics."""
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0
        )

        avg_inference_time = (
            self.total_inference_time / self.total_requests
            if self.total_requests > 0
            else 0
        )

        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.inference_cache),
            "average_inference_time": avg_inference_time,
            "pending_batch_requests": len(self.pending_requests),
            "batch_processor_running": self.batch_processor_running,
            "configuration": {
                "cache_enabled": self.cache_enabled,
                "cache_ttl": self.cache_ttl,
                "max_cache_size": self.max_cache_size,
                "batch_size": self.batch_size,
                "batch_timeout": self.batch_timeout,
            },
        }

    async def clear_cache(self):
        """Clear optimization cache."""
        self.inference_cache.clear()
        self.cache_timestamps.clear()
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info("Optimization cache cleared")

    async def shutdown(self):
        """Gracefully shutdown optimizer."""
        try:
            logger.info("Shutting down Model Optimizer...")

            # Wait for pending batch requests to complete
            if self.pending_requests:
                logger.info(
                    "Waiting for pending batch requests to complete",
                    count=len(self.pending_requests),
                )

                # Process remaining requests
                while self.pending_requests and self.batch_processor_running:
                    await asyncio.sleep(0.1)

            # Clear cache
            await self.clear_cache()

            logger.info("Model Optimizer shutdown complete")

        except Exception as e:
            logger.error("Error during Model Optimizer shutdown", error=str(e))
