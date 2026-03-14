"""
Intelligent Caching System for API Call Optimization

This module provides comprehensive caching capabilities to reduce expensive API calls
and improve performance across the entire system.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from functools import wraps

from config import get_settings


class CacheType(str, Enum):
    """Types of cached data."""
    AI_RESPONSE = "ai_response"
    REASONING_RESULT = "reasoning_result"
    DOCUMENT_ANALYSIS = "document_analysis"
    IMAGE_PROCESSING = "image_processing"
    MODEL_OUTPUT = "model_output"
    API_RESPONSE = "api_response"


class CacheLevel(str, Enum):
    """Cache storage levels."""
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"


@dataclass
class CacheEntry:
    """Represents a cached item."""
    key: str
    data: Any
    cache_type: CacheType
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: int = 3600
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > (self.created_at + timedelta(seconds=self.ttl_seconds))
    
    @property
    def size_estimate(self) -> int:
        """Estimate the size of the cached data in bytes."""
        try:
            return len(pickle.dumps(self.data))
        except Exception:
            return len(str(self.data))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'key': self.key,
            'cache_type': self.cache_type.value,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'ttl_seconds': self.ttl_seconds,
            'metadata': self.metadata,
        }


class CacheStats:
    """Cache performance statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_size_bytes = 0
        self.api_calls_saved = 0
        self.estimated_cost_saved = 0.0
        self.by_type: Dict[CacheType, Dict[str, int]] = {}
        self.start_time = datetime.now()
        
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def uptime_hours(self) -> float:
        """Calculate uptime in hours."""
        return (datetime.now() - self.start_time).total_seconds() / 3600
    
    def record_hit(self, cache_type: CacheType):
        """Record a cache hit."""
        self.hits += 1
        if cache_type not in self.by_type:
            self.by_type[cache_type] = {'hits': 0, 'misses': 0}
        self.by_type[cache_type]['hits'] += 1
    
    def record_miss(self, cache_type: CacheType):
        """Record a cache miss."""
        self.misses += 1
        if cache_type not in self.by_type:
            self.by_type[cache_type] = {'hits': 0, 'misses': 0}
        self.by_type[cache_type]['misses'] += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate_percent': self.hit_rate,
            'evictions': self.evictions,
            'total_size_bytes': self.total_size_bytes,
            'api_calls_saved': self.api_calls_saved,
            'estimated_cost_saved_usd': self.estimated_cost_saved,
            'uptime_hours': self.uptime_hours,
            'by_type': {
                cache_type.value: stats for cache_type, stats in self.by_type.items()
            }
        }


class CacheManager:
    """
    Intelligent caching system for optimizing API calls and expensive operations.
    
    Features:
    - Multi-level caching (memory + disk)
    - TTL-based expiration
    - LRU eviction policy
    - Content-based cache keys
    - Cache warming and prefetching
    - Performance analytics
    """
    
    def __init__(self, max_memory_size: int = 100 * 1024 * 1024):  # 100MB default
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.max_memory_size = max_memory_size
        
        # Storage
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.disk_cache_dir = self.settings.output.output_dir / "cache"
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStats()
        
        # Configuration
        self.default_ttl = {
            CacheType.AI_RESPONSE: 24 * 3600,  # 24 hours
            CacheType.REASONING_RESULT: 12 * 3600,  # 12 hours
            CacheType.DOCUMENT_ANALYSIS: 6 * 3600,  # 6 hours
            CacheType.IMAGE_PROCESSING: 24 * 3600,  # 24 hours
            CacheType.MODEL_OUTPUT: 12 * 3600,  # 12 hours
            CacheType.API_RESPONSE: 3600,  # 1 hour
        }
        
        # Load existing disk cache
        self._load_disk_cache_index()
        
        # Start cleanup task
        self._start_cleanup_task()
        
        self.logger.info(f"CacheManager initialized with {max_memory_size / (1024*1024):.1f}MB memory limit")
    
    def _generate_cache_key(self, data: Any, prefix: str = "") -> str:
        """Generate a deterministic cache key from data."""
        # Convert data to a stable string representation
        if isinstance(data, dict):
            # Sort keys for consistent hashing
            stable_str = json.dumps(data, sort_keys=True, default=str)
        elif isinstance(data, (list, tuple)):
            stable_str = json.dumps(list(data), default=str)
        else:
            stable_str = str(data)
        
        # Generate hash
        hash_obj = hashlib.sha256(stable_str.encode('utf-8'))
        cache_key = hash_obj.hexdigest()[:32]
        
        if prefix:
            return f"{prefix}:{cache_key}"
        return cache_key
    
    def _get_ttl(self, cache_type: CacheType, custom_ttl: Optional[int] = None) -> int:
        """Get TTL for cache type."""
        if custom_ttl:
            return custom_ttl
        return self.default_ttl.get(cache_type, 3600)
    
    def _evict_if_needed(self):
        """Evict entries if memory usage is too high."""
        current_size = sum(entry.size_estimate for entry in self.memory_cache.values())
        
        if current_size <= self.max_memory_size:
            return
        
        # Sort by last accessed time (LRU)
        entries_by_access = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Evict oldest entries
        for key, entry in entries_by_access:
            if current_size <= self.max_memory_size * 0.8:  # Leave some headroom
                break
                
            # Move to disk cache if valuable
            if entry.access_count > 1:
                self._save_to_disk(key, entry)
            
            del self.memory_cache[key]
            current_size -= entry.size_estimate
            self.stats.evictions += 1
    
    def _save_to_disk(self, key: str, entry: CacheEntry):
        """Save cache entry to disk."""
        try:
            cache_file = self.disk_cache_dir / f"{key}.cache"
            metadata_file = self.disk_cache_dir / f"{key}.meta"
            
            # Save data
            with open(cache_file, 'wb') as f:
                pickle.dump(entry.data, f)
            
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(entry.to_dict(), f)
                
        except Exception as e:
            self.logger.error(f"Failed to save cache entry to disk: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Load cache entry from disk."""
        try:
            cache_file = self.disk_cache_dir / f"{key}.cache"
            metadata_file = self.disk_cache_dir / f"{key}.meta"
            
            if not cache_file.exists() or not metadata_file.exists():
                return None
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                meta_dict = json.load(f)
            
            # Load data
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Reconstruct cache entry
            entry = CacheEntry(
                key=meta_dict['key'],
                data=data,
                cache_type=CacheType(meta_dict['cache_type']),
                created_at=datetime.fromisoformat(meta_dict['created_at']),
                last_accessed=datetime.fromisoformat(meta_dict['last_accessed']),
                access_count=meta_dict['access_count'],
                ttl_seconds=meta_dict['ttl_seconds'],
                metadata=meta_dict['metadata']
            )
            
            # Check if expired
            if entry.is_expired:
                self._remove_from_disk(key)
                return None
            
            return entry
            
        except Exception as e:
            self.logger.error(f"Failed to load cache entry from disk: {e}")
            return None
    
    def _remove_from_disk(self, key: str):
        """Remove cache entry from disk."""
        try:
            cache_file = self.disk_cache_dir / f"{key}.cache"
            metadata_file = self.disk_cache_dir / f"{key}.meta"
            
            cache_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            
        except Exception as e:
            self.logger.error(f"Failed to remove cache entry from disk: {e}")
    
    def _load_disk_cache_index(self):
        """Load disk cache index on startup."""
        try:
            for meta_file in self.disk_cache_dir.glob("*.meta"):
                key = meta_file.stem
                entry = self._load_from_disk(key)
                if entry and not entry.is_expired:
                    # Keep frequently accessed items in memory
                    if entry.access_count > 2:
                        self.memory_cache[key] = entry
                        
        except Exception as e:
            self.logger.error(f"Failed to load disk cache index: {e}")
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(300)  # Run every 5 minutes
                    self.cleanup_expired()
                except Exception as e:
                    self.logger.error(f"Cache cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def get(self, key: str, cache_type: CacheType) -> Optional[Any]:
        """Retrieve item from cache."""
        with self.lock:
            # Try memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                if entry.is_expired:
                    del self.memory_cache[key]
                    self.stats.record_miss(cache_type)
                    return None
                
                # Update access stats
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                self.stats.record_hit(cache_type)
                self.stats.api_calls_saved += 1
                
                return entry.data
            
            # Try disk cache
            entry = self._load_from_disk(key)
            if entry:
                # Move back to memory cache
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self.memory_cache[key] = entry
                
                self.stats.record_hit(cache_type)
                self.stats.api_calls_saved += 1
                
                return entry.data
            
            self.stats.record_miss(cache_type)
            return None
    
    def put(
        self, 
        key: str, 
        data: Any, 
        cache_type: CacheType, 
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store item in cache."""
        with self.lock:
            try:
                entry = CacheEntry(
                    key=key,
                    data=data,
                    cache_type=cache_type,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    ttl_seconds=self._get_ttl(cache_type, ttl),
                    metadata=metadata or {}
                )
                
                self.memory_cache[key] = entry
                self._evict_if_needed()
                
                # Update stats
                self.stats.total_size_bytes += entry.size_estimate
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to cache data: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Remove item from cache."""
        with self.lock:
            deleted = False
            
            # Remove from memory
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                self.stats.total_size_bytes -= entry.size_estimate
                del self.memory_cache[key]
                deleted = True
            
            # Remove from disk
            self._remove_from_disk(key)
            
            return deleted
    
    def clear(self, cache_type: Optional[CacheType] = None):
        """Clear cache entries."""
        with self.lock:
            if cache_type:
                # Clear specific cache type
                to_remove = [
                    key for key, entry in self.memory_cache.items()
                    if entry.cache_type == cache_type
                ]
                for key in to_remove:
                    self.delete(key)
            else:
                # Clear all
                self.memory_cache.clear()
                for cache_file in self.disk_cache_dir.glob("*.cache"):
                    cache_file.unlink(missing_ok=True)
                for meta_file in self.disk_cache_dir.glob("*.meta"):
                    meta_file.unlink(missing_ok=True)
                
                self.stats = CacheStats()
    
    def cleanup_expired(self):
        """Remove expired entries."""
        with self.lock:
            # Memory cache
            expired_keys = [
                key for key, entry in self.memory_cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                entry = self.memory_cache[key]
                self.stats.total_size_bytes -= entry.size_estimate
                del self.memory_cache[key]
            
            # Disk cache
            for meta_file in self.disk_cache_dir.glob("*.meta"):
                key = meta_file.stem
                entry = self._load_from_disk(key)
                if entry is None:  # Will be None if expired
                    continue
    
    def cache_decorator(
        self, 
        cache_type: CacheType, 
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None
    ):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key((args, kwargs), func.__name__)
                
                # Try to get from cache
                cached_result = self.get(cache_key, cache_type)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.put(cache_key, result, cache_type, ttl)
                
                return result
            return wrapper
        return decorator
    
    def async_cache_decorator(
        self,
        cache_type: CacheType,
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None
    ):
        """Async decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key((args, kwargs), func.__name__)
                
                # Try to get from cache
                cached_result = self.get(cache_key, cache_type)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                self.put(cache_key, result, cache_type, ttl)
                
                return result
            return wrapper
        return decorator
    
    def warm_cache(self, keys_and_functions: List[Dict[str, Any]]):
        """Pre-warm cache with commonly accessed data."""
        for item in keys_and_functions:
            try:
                key = item['key']
                func = item['function']
                cache_type = item['cache_type']
                args = item.get('args', [])
                kwargs = item.get('kwargs', {})
                
                # Check if already cached
                if self.get(key, cache_type) is not None:
                    continue
                
                # Execute and cache
                if asyncio.iscoroutinefunction(func):
                    # Skip async functions for now in warm_cache
                    continue
                else:
                    result = func(*args, **kwargs)
                    self.put(key, result, cache_type)
                    
            except Exception as e:
                self.logger.error(f"Failed to warm cache for {item.get('key', 'unknown')}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            stats_dict = self.stats.to_dict()
            stats_dict.update({
                'memory_cache_entries': len(self.memory_cache),
                'disk_cache_entries': len(list(self.disk_cache_dir.glob("*.cache"))),
                'memory_usage_bytes': sum(entry.size_estimate for entry in self.memory_cache.values()),
                'memory_usage_mb': sum(entry.size_estimate for entry in self.memory_cache.values()) / (1024 * 1024),
                'memory_limit_mb': self.max_memory_size / (1024 * 1024),
            })
            return stats_dict
    
    def get_cache_info(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get information about cached entries."""
        with self.lock:
            entries = []
            for key, entry in list(self.memory_cache.items())[:limit]:
                entries.append({
                    'key': key[:20] + '...' if len(key) > 20 else key,
                    'cache_type': entry.cache_type.value,
                    'size_bytes': entry.size_estimate,
                    'created_at': entry.created_at.isoformat(),
                    'last_accessed': entry.last_accessed.isoformat(),
                    'access_count': entry.access_count,
                    'ttl_seconds': entry.ttl_seconds,
                    'expired': entry.is_expired,
                    'metadata': entry.metadata
                })
            return entries


# Global cache manager instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        settings = get_settings()
        max_memory = getattr(settings.performance, 'max_memory_usage', 100 * 1024 * 1024)
        _cache_manager = CacheManager(max_memory_size=max_memory)
    return _cache_manager


# Convenience decorators
def cache_ai_response(ttl: int = 24 * 3600):
    """Decorator for caching AI API responses."""
    return get_cache_manager().cache_decorator(CacheType.AI_RESPONSE, ttl)


def cache_reasoning_result(ttl: int = 12 * 3600):
    """Decorator for caching reasoning results."""
    return get_cache_manager().cache_decorator(CacheType.REASONING_RESULT, ttl)


def cache_document_analysis(ttl: int = 6 * 3600):
    """Decorator for caching document analysis results."""
    return get_cache_manager().cache_decorator(CacheType.DOCUMENT_ANALYSIS, ttl)


def async_cache_ai_response(ttl: int = 24 * 3600):
    """Async decorator for caching AI API responses."""
    return get_cache_manager().async_cache_decorator(CacheType.AI_RESPONSE, ttl)


def async_cache_reasoning_result(ttl: int = 12 * 3600):
    """Async decorator for caching reasoning results."""
    return get_cache_manager().async_cache_decorator(CacheType.REASONING_RESULT, ttl)