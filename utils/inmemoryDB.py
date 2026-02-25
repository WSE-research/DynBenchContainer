import threading
import time
import random
from collections import defaultdict as dd
from typing import Any, Dict, List, Optional, Tuple


def _generate_objectid() -> str:
    """
    Generate a MongoDB-style ObjectId as a 12-byte integer.
    Consists of: timestamp (4 bytes), random (4 bytes), counter (4 bytes)
    """
    # Timestamp (4 bytes) - seconds since epoch
    timestamp = int(time.time())
    
    # Random value (5 bytes) - consistent per process
    with _generate_objectid._lock:
        random_value = getattr(_generate_objectid, '_random_value', None)
        if random_value is None:
            random_value = random.randint(0, 0xFFFFFFFF)
            _generate_objectid._random_value = random_value
        
        # Counter (4 bytes) - incrementing per call
        counter = getattr(_generate_objectid, '_counter', 0)
        counter = (counter + 1) % 0xFFFFFFFF
        _generate_objectid._counter = counter
    
    # Combine into a 12-byte integer (as Python int)
    object_id = (timestamp << 64) | (random_value << 32) | counter
    return f'{object_id:x}'

_generate_objectid._lock = threading.Lock()


class InMemoryCollection:
    """
    An in-memory collection that mimics MongoDB collection operations.
    Supports ordered storage for FIFO/LIFO/LRU/MRU policies.
    """
    
    def __init__(self):
        self._data: list = []
        self._indexes: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._indexes['_id'] = {} # _id index is always 1:1
    
    def create_index(self, field: str) -> None:
        """Creates an index on a field."""
        with self._lock:
            if field not in self._indexes:
                self._indexes[field] = dd(list)
                for doc in self._data:
                    if field in doc:
                        value = doc[field]
                        self._indexes[field][value].append(doc)
    
    def _filter(self, filter: dict):
        """Find set of _id's of documents matching filter"""
        found = set(self._indexes['_id'])

        if filter:
            # For any filter field that has an index
            for field, value in [(k, v) for k, v in filter.items() if k in self._indexes]:
                if field == '_id':
                    found = {value} if value in self._indexes['_id'] else set()
                else:
                    found = found & {doc['_id'] for doc in self._indexes[field][value]}

                del filter[field]

            found = {_id for _id in found if all(self._indexes['_id'][_id].get(k, not v) == v for k, v in filter.items())}
        
        return found        

    def find(self, 
                 filter: Optional[Dict[str, Any]] = None, 
                 sort: Optional[List[Tuple[str, int]]] = None) -> List[Dict[str, Any]]:
        """Find a single document matching the filter, optionally sorted."""
        with self._lock:
            found = self._filter(filter)

            if found:
                found_docs = [doc for doc in self._data if doc['_id'] in found]

                if sort:
                    for field, direction in sort:
                        reverse = direction == -1
                        found_docs.sort(key=lambda x: x.get(field, 0), reverse=reverse)

                return [i.copy() for i in found_docs]
            
            return []

    def find_one(self, 
                 filter: Optional[Dict[str, Any]] = None, 
                 sort: Optional[List[Tuple[str, int]]] = None) -> Optional[Dict[str, Any]]:
        """Find a single document matching the filter, optionally sorted."""
        found = self.find(filter, sort)

        if found:
            return found[0].copy()
        
        return None
    
    def insert_one(self, document: Dict[str, Any]) -> None:
        """Insert a document into the collection."""
        with self._lock:
            doc = document.copy()
            # Generate an _id if not present
            if '_id' not in doc:
                doc['_id'] = _generate_objectid()
            
            self._data.append(doc)
            
            for key in self._indexes.keys():
                if key in doc:
                    if key == '_id':
                        self._indexes[key][doc[key]] = doc
                    else:
                        self._indexes[key][doc[key]].append(doc)
    
    def update_one(self, filter: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Update a single document matching the filter with $set operator support."""
        with self._lock:
            found = self._filter(filter)
            
            if not found:
                return {'matched_count': 0, 'modified_count': 0}
            
            # Get the first matching document
            for doc in self._data:
                if doc['_id'] in found:
                    matched_doc = doc
                    break
            
            # Process $set operator
            modified = False
            if '$set' in update:
                for field, value in update['$set'].items():
                    if field not in matched_doc or matched_doc[field] != value:
                        matched_doc[field] = value
                        modified = True
            
            if modified:
                # Update indexes
                for key in self._indexes.keys():
                    if key in matched_doc:
                        if key == '_id':
                            self._indexes[key][matched_doc[key]] = matched_doc
                        else:
                            # Rebuild index for this field
                            self._indexes[key] = dd(list)
                            for d in self._data:
                                if key in d:
                                    self._indexes[key][d[key]].append(d)
            
            return {'matched_count': 1, 'modified_count': 1 if modified else 0}
    
    def count_documents(self, filter: Dict[str, Any] = None) -> int:
        """Count documents matching the filter."""
        with self._lock:
            found = self._filter(filter or {})
            return len(found)
    
    def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute aggregation pipeline with $sample operator support."""
        with self._lock:
            # Start with all documents
            result = [doc.copy() for doc in self._data]
            
            for stage in pipeline:
                if '$sample' in stage:
                    size = stage['$sample'].get('size', 1)
                    if size >= len(result):
                        continue
                    # Randomly sample documents
                    result = random.sample(result, min(size, len(result)))
            
            return result
    
    def delete_one(self, filter: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a single document matching the filter."""
        with self._lock:
            found = self._filter(filter)
            
            if not found:
                return {'deleted_count': 0}
            
            # Find and remove the first matching document
            for i, doc in enumerate(self._data):
                if doc['_id'] in found:
                    removed_doc = self._data.pop(i)
                    break
            
            # Update indexes
            for key, value in self._indexes.items():
                if key in removed_doc:
                    if key == '_id':
                        if removed_doc[key] in self._indexes['_id']:
                            del self._indexes['_id'][removed_doc[key]]
                    else:
                        # Remove from list index
                        # if removed_doc in self._indexes[key][removed_doc.get(key, None)]:
                        del self._indexes[key][removed_doc.get(key, None)]
            
            return {'deleted_count': 1}
    
    def drop(self) -> None:
        """Drop the collection, removing all data and indexes."""
        with self._lock:
            self._data = []
            self._indexes = {'_id': {}}
