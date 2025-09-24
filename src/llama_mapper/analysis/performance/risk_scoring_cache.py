"""
High-performance caching and optimization for Risk Scoring Framework.

This module provides intelligent caching, rate limiting, and performance
optimizations for production-grade risk scoring operations.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import threading
from contextlib import asynccontextmanager
from collections import defaultdict, OrderedDict

from ..domain.analysis_models import SecurityFinding, RiskScore, BusinessImpact
from ..config.risk_scoring_config import RiskScoringConfiguration

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: datetime
    ttl_seconds: int
    access_count: int = 0
    last_access: Optional[datetime] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now(timezone.utc) > self.timestamp + timedelta(seconds=self.ttl_seconds)
    
    def touch(self):
        """Update access metadata."""
        self.access_count += 1
        self.last_access = datetime.now(timezone.utc)


class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired_removals': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                self._stats['expired_removals'] += 1
                self._stats['misses'] += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats['hits'] += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value into cache."""
        with self._lock:
            ttl = ttl or self.default_ttl
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now(timezone.utc),
                ttl_seconds=ttl
            )
            
            # Remove if already exists
            if key in self._cache:
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = entry
            
            # Evict oldest if over capacity
            while len(self._cache) > self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self._stats['evictions'] += 1
                logger.debug(f"Evicted cache entry: {oldest_key}")
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self._cache[key]
                self._stats['expired_removals'] += 1
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'expired_removals': self._stats['expired_removals']
            }


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, max_tokens: int = 100, refill_rate: float = 10.0):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = max_tokens
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire tokens."""
        with self._lock:
            now = time.time()
            
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            tokens_to_add = int(elapsed * self.refill_rate)
            self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def acquire_async(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Async version of acquire with timeout."""
        start_time = time.time()
        
        while True:
            if self.acquire(tokens):
                return True
            
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            await asyncio.sleep(0.01)  # Small delay before retry


class FindingsHasher:
    """Utility for creating consistent hashes of security findings."""
    
    @staticmethod
    def hash_findings(findings: List[SecurityFinding]) -> str:
        """Create a consistent hash for a list of findings."""
        # Create deterministic representation
        findings_data = []
        for finding in sorted(findings, key=lambda f: f.finding_id):
            finding_dict = {
                'finding_id': finding.finding_id,
                'detector_id': finding.detector_id,
                'severity': finding.severity.value,
                'category': finding.category,
                'confidence': finding.confidence,
                # Only include stable metadata fields
                'metadata_hash': FindingsHasher._hash_metadata(finding.metadata)
            }
            findings_data.append(finding_dict)
        
        # Create JSON string and hash
        json_str = json.dumps(findings_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:16]
    
    @staticmethod
    def _hash_metadata(metadata: Dict[str, Any]) -> str:
        """Create hash of metadata focusing on stable fields."""
        if not metadata:
            return ""
        
        # Only include fields that affect risk calculation
        stable_fields = [
            'confidentiality_impact', 'integrity_impact', 'availability_impact',
            'attack_vector', 'attack_complexity', 'privileges_required',
            'user_interaction', 'scope', 'data_classification'
        ]
        
        stable_metadata = {
            key: value for key, value in metadata.items()
            if key in stable_fields
        }
        
        if not stable_metadata:
            return ""
        
        json_str = json.dumps(stable_metadata, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:8]


class PerformanceTracker:
    """Track performance metrics for risk scoring operations."""
    
    def __init__(self):
        self._metrics = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_operation(self, operation_type: str, duration_ms: float, success: bool = True):
        """Record an operation."""
        with self._lock:
            self._metrics[operation_type].append({
                'duration_ms': duration_ms,
                'success': success,
                'timestamp': datetime.now(timezone.utc)
            })
            
            # Keep only recent metrics (last 1000 operations per type)
            if len(self._metrics[operation_type]) > 1000:
                self._metrics[operation_type] = self._metrics[operation_type][-1000:]
    
    def get_stats(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if operation_type:
                return self._calculate_stats(operation_type, self._metrics[operation_type])
            
            return {
                op_type: self._calculate_stats(op_type, metrics)
                for op_type, metrics in self._metrics.items()
            }
    
    def _calculate_stats(self, operation_type: str, metrics: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics for a specific operation type."""
        if not metrics:
            return {
                'operation_type': operation_type,
                'count': 0,
                'success_rate': 0.0,
                'avg_duration_ms': 0.0,
                'p95_duration_ms': 0.0,
                'p99_duration_ms': 0.0
            }
        
        durations = [m['duration_ms'] for m in metrics]
        successes = [m for m in metrics if m['success']]
        
        durations.sort()
        
        return {
            'operation_type': operation_type,
            'count': len(metrics),
            'success_rate': len(successes) / len(metrics),
            'avg_duration_ms': sum(durations) / len(durations),
            'p95_duration_ms': durations[int(len(durations) * 0.95)] if durations else 0,
            'p99_duration_ms': durations[int(len(durations) * 0.99)] if durations else 0
        }


class RiskScoringCache:
    """High-performance cache for risk scoring operations."""
    
    def __init__(self, config: RiskScoringConfiguration):
        self.config = config
        self.enabled = config.performance_config.enable_caching
        self.ttl = config.performance_config.cache_ttl_seconds
        
        # Initialize cache components
        self._risk_score_cache = LRUCache(max_size=1000, default_ttl=self.ttl)
        self._business_impact_cache = LRUCache(max_size=500, default_ttl=self.ttl)
        self._component_cache = LRUCache(max_size=2000, default_ttl=self.ttl // 2)
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            max_tokens=config.performance_config.max_concurrent_calculations,
            refill_rate=config.performance_config.max_concurrent_calculations / 60.0  # per minute
        )
        
        # Background cleanup task
        self._cleanup_task = None
        if self.enabled:
            self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background task for cache cleanup."""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Cleanup every 5 minutes
                    self._cleanup_expired()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cache cleanup: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    def _cleanup_expired(self):
        """Clean up expired cache entries."""
        if not self.enabled:
            return
        
        risk_removed = self._risk_score_cache.cleanup_expired()
        impact_removed = self._business_impact_cache.cleanup_expired()
        component_removed = self._component_cache.cleanup_expired()
        
        total_removed = risk_removed + impact_removed + component_removed
        if total_removed > 0:
            logger.debug(f"Cleaned up {total_removed} expired cache entries")
    
    async def get_risk_score(self, findings: List[SecurityFinding]) -> Optional[RiskScore]:
        """Get cached risk score."""
        if not self.enabled:
            return None
        
        cache_key = self._generate_risk_score_key(findings)
        cached_score = self._risk_score_cache.get(cache_key)
        
        if cached_score:
            logger.debug(f"Cache hit for risk score: {cache_key}")
            return cached_score
        
        return None
    
    async def cache_risk_score(self, findings: List[SecurityFinding], risk_score: RiskScore) -> None:
        """Cache a risk score."""
        if not self.enabled:
            return
        
        cache_key = self._generate_risk_score_key(findings)
        self._risk_score_cache.put(cache_key, risk_score, self.ttl)
        logger.debug(f"Cached risk score: {cache_key}")
    
    async def get_business_impact(self, findings: List[SecurityFinding]) -> Optional[BusinessImpact]:
        """Get cached business impact."""
        if not self.enabled:
            return None
        
        cache_key = self._generate_business_impact_key(findings)
        cached_impact = self._business_impact_cache.get(cache_key)
        
        if cached_impact:
            logger.debug(f"Cache hit for business impact: {cache_key}")
            return cached_impact
        
        return None
    
    async def cache_business_impact(self, findings: List[SecurityFinding], impact: BusinessImpact) -> None:
        """Cache a business impact."""
        if not self.enabled:
            return
        
        cache_key = self._generate_business_impact_key(findings)
        self._business_impact_cache.put(cache_key, impact, self.ttl)
        logger.debug(f"Cached business impact: {cache_key}")
    
    async def get_component_result(self, component: str, findings: List[SecurityFinding]) -> Optional[Any]:
        """Get cached component calculation result."""
        if not self.enabled:
            return None
        
        cache_key = self._generate_component_key(component, findings)
        return self._component_cache.get(cache_key)
    
    async def cache_component_result(self, component: str, findings: List[SecurityFinding], result: Any) -> None:
        """Cache a component calculation result."""
        if not self.enabled:
            return
        
        cache_key = self._generate_component_key(component, findings)
        self._component_cache.put(cache_key, result, self.ttl // 2)
    
    def _generate_risk_score_key(self, findings: List[SecurityFinding]) -> str:
        """Generate cache key for risk score."""
        findings_hash = FindingsHasher.hash_findings(findings)
        config_hash = self._get_config_hash()
        return f"risk_score:{findings_hash}:{config_hash}"
    
    def _generate_business_impact_key(self, findings: List[SecurityFinding]) -> str:
        """Generate cache key for business impact."""
        findings_hash = FindingsHasher.hash_findings(findings)
        config_hash = self._get_config_hash()
        return f"business_impact:{findings_hash}:{config_hash}"
    
    def _generate_component_key(self, component: str, findings: List[SecurityFinding]) -> str:
        """Generate cache key for component result."""
        findings_hash = FindingsHasher.hash_findings(findings)
        config_hash = self._get_config_hash()
        return f"component:{component}:{findings_hash}:{config_hash}"
    
    def _get_config_hash(self) -> str:
        """Get hash of relevant configuration."""
        config_dict = self.config.to_dict()
        # Only include fields that affect calculations
        relevant_config = {
            'risk_weights': config_dict['risk_weights'],
            'calculation_method': config_dict['calculation_method']
        }
        json_str = json.dumps(relevant_config, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:8]
    
    def invalidate_all(self) -> None:
        """Invalidate all cache entries."""
        if not self.enabled:
            return
        
        self._risk_score_cache.clear()
        self._business_impact_cache.clear()
        self._component_cache.clear()
        logger.info("All cache entries invalidated")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self.enabled:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'risk_score_cache': self._risk_score_cache.get_stats(),
            'business_impact_cache': self._business_impact_cache.get_stats(),
            'component_cache': self._component_cache.get_stats(),
            'performance_stats': self.performance_tracker.get_stats()
        }
    
    @asynccontextmanager
    async def rate_limited_operation(self, operation_name: str, tokens: int = 1):
        """Context manager for rate-limited operations."""
        # Acquire rate limit
        acquired = await self.rate_limiter.acquire_async(
            tokens, 
            timeout=self.config.performance_config.calculation_timeout_seconds
        )
        
        if not acquired:
            raise Exception(f"Rate limit exceeded for operation: {operation_name}")
        
        start_time = time.time()
        success = False
        
        try:
            yield
            success = True
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.performance_tracker.record_operation(operation_name, duration_ms, success)
    
    def shutdown(self):
        """Shutdown cache and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        self.invalidate_all()
        logger.info("Risk scoring cache shutdown complete")
