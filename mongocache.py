from inspect import signature
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

    def fifo_cache(self, using=None):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                sign = signature(func)
                bound_args = sign.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                key = []
                for name, value in bound_args.arguments.items():
                    if not using or name in using:
                        key.append(f'{name}={value}')
                key = f"{func.__name__}({', '.join(key)})"

                result = self.get(key)
                if result is not None:
                    return result['value']
                
                result = func(*args, **kwargs)
                self.set(key, result)
                
                return result

            return wrapper

        return decorator
