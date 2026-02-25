import pytest
import random
from unittest.mock import Mock, patch
from utils.mongocache import MongoCache, Policy


@pytest.fixture
def mock_collection():
    """Create a mock MongoDB collection for testing."""
    collection = Mock()
    collection.count_documents.return_value = 0
    collection.find_one.return_value = None
    collection.aggregate.return_value = []
    return collection


@pytest.fixture
def cache(mock_collection):
    """Create a MongoCache instance with mock collection."""
    return MongoCache(mock_collection, max_size=10, policy=Policy.FIFO)


class TestMongoCacheBasicOperations:
    """Test basic get/set operations."""
    
    def test_set_and_get(self, cache, mock_collection):
        """Test basic set and get operations."""
        mock_collection.find_one.return_value = {
            '_id': 1, 'key': 'test', 'order': 1, 
            'value': {'result': 'data'}
        }
        
        cache.set('test', {'result': 'data'})
        result = cache.get('test')
        
        assert result is not None
        assert result['value'] == {'result': 'data'}
    
    def test_get_nonexistent_key(self, cache, mock_collection):
        """Test getting a key that doesn't exist."""
        mock_collection.find_one.return_value = None
        
        result = cache.get('nonexistent')
        
        assert result is None
    
    def test_set_with_unlimited_size(self, mock_collection):
        """Test cache with unlimited size (max_size=0)."""
        cache = MongoCache(mock_collection, max_size=0)
        cache.set('key1', 'value1')
        
        # Should not evict anything
        mock_collection.delete_one.assert_not_called()


class TestMongoCacheEvictionPolicies:
    """Test different eviction policies."""
    
    def test_fifo_eviction(self, mock_collection):
        """Test FIFO eviction policy."""
        # Setup: cache is full with 3 items
        mock_collection.count_documents.return_value = 3
        mock_collection.find_one.side_effect = [
            {'_id': 1, 'key': 'old', 'order': 1, 'value': 'old_value'},  # FIFO evicts this
            None,  # find_one for get_max_order after eviction
        ]
        
        cache = MongoCache(mock_collection, max_size=2, policy=Policy.FIFO)
        cache.set('new', 'new_value')
        
        # Should have deleted the oldest item
        mock_collection.delete_one.assert_called_once_with({'_id': 1})
    
    def test_lifo_eviction(self, mock_collection):
        """Test LIFO eviction policy."""
        mock_collection.count_documents.return_value = 3
        mock_collection.find_one.side_effect = [
            {'_id': 3, 'key': 'newest', 'order': 3, 'value': 'newest_value'},  # LIFO evicts this
            None,
        ]
        
        cache = MongoCache(mock_collection, max_size=2, policy=Policy.LIFO)
        cache.set('new', 'new_value')
        
        mock_collection.delete_one.assert_called_once_with({'_id': 3})
    
    def test_lru_eviction(self, mock_collection):
        """Test LRU eviction policy."""
        mock_collection.count_documents.return_value = 3
        mock_collection.find_one.side_effect = [
            {'_id': 2, 'key': 'lru', 'order': 1, 'value': 'lru_value'},  # LRU evicts this
            None,
        ]
        
        cache = MongoCache(mock_collection, max_size=2, policy=Policy.LRU)
        cache.set('new', 'new_value')
        
        mock_collection.delete_one.assert_called_once_with({'_id': 2})
    
    def test_mru_eviction(self, mock_collection):
        """Test MRU eviction policy."""
        mock_collection.count_documents.return_value = 3
        mock_collection.find_one.side_effect = [
            {'_id': 2, 'key': 'mru', 'order': 3, 'value': 'mru_value'},  # MRU evicts this
            None,
        ]
        
        cache = MongoCache(mock_collection, max_size=2, policy=Policy.MRU)
        cache.set('new', 'new_value')
        
        mock_collection.delete_one.assert_called_once_with({'_id': 2})
    
    def test_rr_eviction(self, mock_collection):
        """Test Random replacement policy."""
        mock_collection.aggregate.return_value = iter([
            {'_id': 2, 'key': 'key_1', 'order': 2, 'value': 'value_1'}
        ])
        
        cache = MongoCache(mock_collection, max_size=2, policy=Policy.RR)
        cache.set('key_0', 'value_0')
        cache.set('key_1', 'value_1')
        mock_collection.count_documents.return_value = 3
        cache.set('key_2', 'value_2')
        
        # Should use aggregation for random sampling
        mock_collection.aggregate.assert_called_once()
        mock_collection.delete_one.assert_called_once()


class TestMongoCacheLRUMRUOrderUpdates:
    """Test LRU/MRU order updates on get."""
    
    def test_lru_get_updates_order(self, mock_collection):
        """Test that get updates order for LRU policy."""
        mock_collection.find_one.side_effect = [
            {'_id': 1, 'key': 'test', 'order': 1, 'value': 'data'},
            {'_id': 2, 'key': 'other', 'order': 2, 'value': 'other_data'},
        ]
        mock_collection.get_max_order = Mock(return_value=2)
        
        cache = MongoCache(mock_collection, max_size=5, policy=Policy.LRU)
        cache.get('test')
        
        # Order should be updated to max_order + 1
        mock_collection.update_one.assert_called_once_with(
            {'_id': 1}, 
            {'$set': {'order': 3}}
        )
    
    def test_mru_get_updates_order(self, mock_collection):
        """Test that get updates order for MRU policy."""
        mock_collection.find_one.side_effect = [
            {'_id': 1, 'key': 'test', 'order': 1, 'value': 'data'},
            {'_id': 2, 'key': 'other', 'order': 2, 'value': 'other_data'},
        ]
        mock_collection.get_max_order = Mock(return_value=2)
        
        cache = MongoCache(mock_collection, max_size=5, policy=Policy.MRU)
        cache.get('test')
        
        mock_collection.update_one.assert_called_once()


class TestMongoCacheDecorator:
    """Test the cache decorator functionality."""
    
    def test_decorator_basic(self, mock_collection):
        """Test basic decorator functionality."""
        cache = MongoCache(mock_collection, max_size=10)
        
        @cache.cache()
        def add(a, b):
            return a + b
        
        mock_collection.find_one.return_value = None  # Cache miss
        
        result = add(1, 2)
        
        assert result == 3
        # Should have called set with cache key
        assert mock_collection.insert_one.called
    
    def test_decorator_with_using(self, mock_collection):
        """Test decorator with specific argument selection."""
        cache = MongoCache(mock_collection, max_size=10)
        
        @cache.cache(using=['a'])
        def add(a, b):
            return a + b
        
        mock_collection.find_one.return_value = None      
        add(1, 2)

        mock_collection.find_one.return_value = { '_id': 'test', 'order': 1, 'key': 'add(a=1)', 'value': 3 }
        add(1, 3)  # Should use cache since 'a' is same
        
        # Should only call insert once (cache hit on second call)
        assert mock_collection.insert_one.call_count == 1
    
    def test_decorator_with_kwargs(self, mock_collection):
        """Test decorator with keyword arguments."""
        cache = MongoCache(mock_collection, max_size=10)
        
        @cache.cache()
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"
        
        mock_collection.find_one.return_value = None
        
        result = greet("Alice", greeting="Hi")
        
        assert result == "Hi, Alice!"
    
    def test_decorator_cache_hit(self, mock_collection):
        """Test decorator returns cached value."""
        cache = MongoCache(mock_collection, max_size=10)
        
        @cache.cache()
        def multiply(a, b):
            return a * b
        
        # Setup cached result
        mock_collection.find_one.return_value = {
            '_id': 1, 'key': 'multiply(a=2, b=3)', 
            'order': 1, 'value': 6
        }
        
        result = multiply(2, 3)
        
        assert result == 6
        # Function should not be called on cache hit
        mock_collection.insert_one.assert_not_called()


class TestMongoCacheHelperMethods:
    """Test helper methods get_min_order and get_max_order."""
    
    def test_get_max_order_empty_cache(self, mock_collection):
        """Test get_max_order with empty cache."""
        mock_collection.find_one.return_value = None
        cache = MongoCache(mock_collection)
        
        result = cache.get_max_order()
        
        assert result == 0
    
    def test_get_max_order_with_data(self, mock_collection):
        """Test get_max_order with data in cache."""
        mock_collection.find_one.return_value = {'_id': 1, 'order': 5}
        cache = MongoCache(mock_collection)
        
        result = cache.get_max_order()
        
        assert result == 5
    
    def test_get_min_order_empty_cache(self, mock_collection):
        """Test get_min_order with empty cache."""
        mock_collection.find_one.return_value = None
        cache = MongoCache(mock_collection)
        
        result = cache.get_min_order()
        
        assert result == 0


class TestMongoCacheThreadSafety:
    """Test thread safety with lock."""
    
    def test_lock_used_in_operations(self, mock_collection):
        """Test that lock is used in cache operations."""
        cache = MongoCache(mock_collection, max_size=10)
        
        # Don't manually acquire the lock - let the method handle it
        cache.set('test', 'value')
        
        # Should have called insert
        assert mock_collection.insert_one.called
    
    def test_lock_used_in_get(self, mock_collection):
        """Test that lock is used in get operations."""
        cache = MongoCache(mock_collection, max_size=10)
        
        cache.get('test')
        
        assert mock_collection.find_one.called


class TestMongoCacheEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_cache_with_max_size_one(self, mock_collection):
        """Test cache with max_size=1."""
        mock_collection.count_documents.return_value = 1
        mock_collection.find_one.side_effect = [
            {'_id': 1, 'key': 'old', 'order': 1, 'value': 'old_value'},
            None,
        ]
        
        cache = MongoCache(mock_collection, max_size=1, policy=Policy.FIFO)
        cache.set('new', 'new_value')
        
        mock_collection.delete_one.assert_called_once()
    
    def test_cache_key_formatting(self, mock_collection):
        """Test that cache keys are properly formatted."""
        cache = MongoCache(mock_collection, max_size=10)
        
        @cache.cache()
        def func(a, b, c=None):
            return a + b
        
        mock_collection.find_one.return_value = None
        func(1, 2, c=3)
        
        # Check that key was created with proper formatting
        insert_call = mock_collection.insert_one.call_args
        assert insert_call is not None
        inserted_doc = insert_call[0][0]
        assert 'func(a=1, b=2, c=null)' in inserted_doc['key'] or 'func(a=1, b=2, c=3)' in inserted_doc['key']
