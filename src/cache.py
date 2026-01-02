import hashlib
import time
from typing import Dict, Any, Optional
from collections import OrderedDict


class QueryCache:
    """
    LRU cache с TTL для кэширования результатов RAG pipeline.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }

    def _hash_query(self, query: str, **kwargs) -> str:
        """Создать хэш ключ из запроса и параметров"""
        key_str = f"{query.lower().strip()}_{sorted(kwargs.items())}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Получить результат из кэша"""
        key = self._hash_query(query, **kwargs)

        if key in self.cache:
            entry = self.cache[key]

            # Проверить TTL
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                self.stats["hits"] += 1
                # LRU: переместить в конец
                self.cache.move_to_end(key)
                return entry["result"]
            else:
                # Истек TTL
                del self.cache[key]

        self.stats["misses"] += 1
        return None

    def set(self, query: str, result: Dict[str, Any], **kwargs):
        """Сохранить результат в кэш"""
        key = self._hash_query(query, **kwargs)

        # LRU eviction
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
            self.stats["evictions"] += 1

        self.cache[key] = {
            "result": result,
            "timestamp": time.time()
        }

    def clear(self):
        """Очистить кэш"""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику кэша"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0

        return {
            **self.stats,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds
        }
