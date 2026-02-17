import json

import threading

from functools import wraps
from inspect import signature

from typing import Callable, Any
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
    def __init__(self, collection, max_size: int=1024, policy=Policy.FIFO):
        assert isinstance(max_size, int)

        self.collection = collection
        self.max_size = max(0, max_size)
        self.policy = policy
        self.lock = threading.Lock()

        # enshure that collection is properly indexed
        with self.lock:
            collection.create_index('order', unique=True)

    def get_min_order(self):
        doc = self.collection.find_one({}, sort={ 'order': 1 })
        return doc['order']  if doc and 'order' in doc else  1

    def get_max_order(self):
        doc = self.collection.find_one({}, sort={ 'order': -1 })
        return doc['order']  if doc and 'order' in doc else  1

    def get(self, key: str):
        with self.lock:
            doc = self.collection.find_one({ 'key': key })
            if doc:
                order = self.get_max_order()
                self.collection.update_one({ '_id': doc['_id'] }, { '$set': { 'order': order + 1 } })
                return doc
            else:
                return None

    def set(self, key: str, value, policy = None):
        if not policy:
            policy = self.policy

        with self.lock:
            if self.max_size and self.collection.count_documents({}) >= self.max_size:
                match policy:
                    case Policy.RR:
                        # there is always at least one document
                        doc = next(self.collection.aggregate([ { '$sample': { 'size': 1 } } ]))
                    case Policy.FIFO:
                        doc = self.collection.find_one(sort=[('_id', 1)])
                    case Policy.LIFO:
                        doc = self.collection.find_one(sort=[('_id', -1)])
                    case Policy.LRU:
                        doc = self.collection.find_one(sort=[('order', 1)])
                    case Policy.MRU:
                        doc = self.collection.find_one(sort=[('order', -1)])
                    case _:
                        doc = None

                if doc:
                    self.collection.delete_one({ '_id': doc['_id'] })

            self.collection.insert_one({ 'key': key, 'order': self.get_max_order() + 1, 'value': value })

    def cache(self, using=None, policy=None):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                sign = signature(func)
                bound_args = sign.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                key = []
                for name, value in bound_args.arguments.items():
                    if (not using) or (name in using):
                        key.append(f'{name}={_format(value)}')
                key = f"{func.__name__}({', '.join(key)})"

                result = self.get(key)
                if result is not None:
                    return result['value']
                
                result = func(*args, **kwargs)
                self.set(key, result, policy)
                
                return result

            return wrapper

        return decorator
