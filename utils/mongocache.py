import json

import threading

from functools import wraps
from inspect import signature

from typing import Callable
from enum import Enum


class Policy(Enum):
    RR   = 1
    FIFO = 2
    LIFO = 3
    LRU  = 4
    MRU  = 5


def _format(value):
    return json.dumps(value)


class MongoCache:
    """
    A cache implementation using MongoDB as backend with various eviction policies.
    
    Policies:
        RR   - Random replacement
        FIFO - First In, First Out
        LIFO - Last In, First Out
        LRU  - Least Recently Used
        MRU  - Most Recently Used
    
    The 'order' field tracks insertion/access order depending on policy:
        - FIFO/LIFO: based on insertion time (_id correlates with order)
        - LRU/MRU: updated on each cache access (higher = more recently used)
        - RR: order is not meaningful
    """
    
    def __init__(self, collection, max_size: int = 1024, policy: Policy = Policy.FIFO):
        """
        Initialize the cache.
        
        Args:
            collection: MongoDB collection to use for storage
            max_size: Maximum number of documents in cache (0 for unlimited)
            policy: Eviction policy to use when cache is full
        """
        self.collection = collection
        self.max_size = max(0, max_size)
        self.policy = policy
        self.lock = threading.Lock()

        # Ensure that collection is properly indexed
        with self.lock:
            collection.create_index('order', unique=True)
            collection.create_index('key', unique=True)

    def get_min_order(self) -> int:
        """Get the minimum order value in the cache. Returns 0 if cache is empty."""
        doc = self.collection.find_one({}, sort=[('order', 1)])
        return doc['order'] if doc and 'order' in doc else 0

    def get_max_order(self) -> int:
        """Get the maximum order value in the cache. Returns 0 if cache is empty."""
        doc = self.collection.find_one({}, sort=[('order', -1)])
        return doc['order'] if doc and 'order' in doc else 0

    def get(self, key: str) -> dict | None:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
        Returns:
            Document dict with 'key', 'order', 'value' fields, or None if not found
        """
        with self.lock:
            doc = self.collection.find_one({ 'key': key })
            if doc:
                order = self.get_max_order()
                self.collection.update_one({ '_id': doc['_id'] }, { '$set': { 'order': order + 1 } })
                return doc
            else:
                return None

    def set(self, key: str, value, policy: Policy | None = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            policy: Eviction policy to use (defaults to instance policy)
        """
        if policy is None:
            policy = self.policy

        with self.lock:
            # Evict if cache is full
            if self.max_size > 0 and self.collection.count_documents({}) >= self.max_size:
                match policy:
                    case Policy.RR:
                        # Random replacement - use aggregation to sample
                        doc = next(self.collection.aggregate([{'$sample': {'size': 1}}]), None)
                    case Policy.FIFO:
                        # First In, First Out
                        doc = self.collection.find_one(sort=[('_id', 1)])
                    case Policy.LIFO:
                        # Last In, First Out
                        doc = self.collection.find_one(sort=[('_id', -1)])
                    case Policy.LRU:
                        # Least Recently Used - smallest order
                        doc = self.collection.find_one(sort=[('order', 1)])
                    case Policy.MRU:
                        # Most Recently Used - largest order
                        doc = self.collection.find_one(sort=[('order', -1)])
                    case _:
                        doc = None

                if doc:
                    self.collection.delete_one({ '_id': doc['_id'] })

            self.collection.insert_one({ 'key': key, 'order': self.get_max_order() + 1, 'value': value })

    def cache(self, using: list[str] | None = None, policy: Policy | None = None):
        """
        Decorator to cache function results.

        Args:
            using: List of argument names to include in cache key.
                   If None, all arguments are included.
            policy: Eviction policy for this cached function (defaults to instance policy)

        Returns:
            Decorated function with caching behavior
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                sign = signature(func)
                bound_args = sign.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Build cache key from function arguments
                key_parts = []
                for name, value in bound_args.arguments.items():
                    if using is None or name in using:
                        key_parts.append(f'{name}={_format(value)}')
                key = f"{func.__name__}({', '.join(key_parts)})"

                # Try to get from cache
                result = self.get(key)
                if result is not None:
                    return result['value']

                # Call function and cache result
                result = func(*args, **kwargs)
                self.set(key, result, policy)

                return result

            return wrapper

        return decorator
