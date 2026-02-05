from functools import wraps
from typing import Callable

from pymongo.collection import Collection


class MongoCache:
    def __init__(self, collection, max_size: int=1024):
        assert isinstance(collection, Collection)
        assert isinstance(max_size, int)
        assert max_size >= 0
        self.collection = collection
        self.max_size = max_size

    def get(self, key: str):
        return self.collection.find_one({ 'key': key })

    def set(self, key: str, value):
        if self.max_size and self.collection.count_documents({}) >= self.max_size:
            self._remove_oldest()
        self.collection.insert_one({ 'key': key, 'value': value })

    def delete(self, key: str):
        self.collection.delete_one({ 'key': key })

    def _remove_oldest(self):
        doc = self.collection.find_one(sort=[('_id', 1)])
        if doc:
            self.collection.delete_one({ '_id': doc['_id'] })

    def fifo_cache(self):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = f"{func.__name__}-{str(args) if args else ''}{str(kwargs) if kwargs else ''}"

                result = self.get(key)
                if result is not None:
                    return result['value']
                
                result = func(*args, **kwargs)
                self.set(key, result)
                
                return result

            return wrapper

        return decorator
