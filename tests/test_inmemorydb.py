import time
import threading

from utils.inmemoryDB import _generate_objectid, InMemoryCollection


class TestGenerateObjectId:
    """Tests for the _generate_objectid function."""
    
    def test_generates_unique_ids(self):
        """Test that generated IDs are unique."""
        ids = [_generate_objectid() for _ in range(100)]
        assert len(ids) == len(set(ids))
    
    def test_returns_hex_string(self):
        """Test that the function returns a hexadecimal string."""
        object_id = _generate_objectid()
        assert isinstance(object_id, str)
        assert all(c in '0123456789abcdef' for c in object_id)
    
    def test_id_length(self):
        """Test that the ID has appropriate length (12 bytes = 24 hex chars)."""
        object_id = _generate_objectid()
        # 12 bytes should be 24 hex characters
        assert len(object_id) == 24
    
    def test_timestamp_component(self):
        """Test that the ID contains timestamp information."""
        # Generate two IDs with a small delay
        id1 = _generate_objectid()
        time.sleep(0.1)
        id2 = _generate_objectid()
        
        # The second ID should be greater than the first (due to timestamp)
        assert int(id2, 16) >= int(id1, 16)


class TestInMemoryCollection:
    """Tests for the InMemoryCollection class."""
    
    def setup_method(self):
        """Create a new collection for each test."""
        self.collection = InMemoryCollection()
    
    def test_initialization(self):
        """Test that collection initializes correctly."""
        assert self.collection._data == []
        assert '_id' in self.collection._indexes
        assert isinstance(self.collection._lock, type(threading.Lock()))
    
    def test_create_index(self):
        """Test creating an index on a field."""
        self.collection.create_index('name')
        assert 'name' in self.collection._indexes
    
    def test_create_index_duplicate(self):
        """Test that creating duplicate indexes doesn't cause issues."""
        self.collection.create_index('name')
        initial_indexes = self.collection._indexes.copy()
        self.collection.create_index('name')
        assert self.collection._indexes == initial_indexes
    
    def test_insert_one_without_id(self):
        """Test inserting a document without an _id."""
        doc = {'name': 'Alice', 'age': 30}
        self.collection.insert_one(doc)
        
        assert len(self.collection._data) == 1
        assert '_id' in self.collection._data[0]
        assert self.collection._data[0]['name'] == 'Alice'
        assert self.collection._data[0]['age'] == 30
    
    def test_insert_one_with_id(self):
        """Test inserting a document with an _id."""
        doc = {'_id': 'custom_id', 'name': 'Bob', 'age': 25}
        self.collection.insert_one(doc)
        
        assert len(self.collection._data) == 1
        assert self.collection._data[0]['_id'] == 'custom_id'
    
    def test_insert_multiple_documents(self):
        """Test inserting multiple documents."""
        for i in range(5):
            self.collection.insert_one({'index': i})
        
        assert len(self.collection._data) == 5
    
    def test_insert_updates_indexes(self):
        """Test that insert updates indexes correctly."""
        self.collection.create_index('name')
        self.collection.insert_one({'name': 'Alice'})
        
        assert len(self.collection._indexes['name']['Alice']) == 1
    
    def test_find_no_match(self):
        """Test find with no matching documents."""
        result = self.collection.find({'name': 'NonExistent'})
        assert result == []
    
    def test_find_with_match(self):
        """Test find with matching documents."""
        self.collection.insert_one({'name': 'Alice', 'age': 30})
        self.collection.insert_one({'name': 'Bob', 'age': 25})
        
        result = self.collection.find({'name': 'Alice'})
        assert result is not None
        assert len(result) == 1
        assert result[0]['name'] == 'Alice'
    
    def test_find_returns_copies(self):
        """Test that find returns copies, not references."""
        self.collection.insert_one({'name': 'Alice', 'age': 30})
        
        result = self.collection.find()
        result[0]['name'] = 'Modified'
        
        # Original should be unchanged
        assert self.collection._data[0]['name'] == 'Alice'
    
    def test_find_with_sort(self):
        """Test find with sorting."""
        self.collection.insert_one({'name': 'Alice', 'age': 30})
        self.collection.insert_one({'name': 'Bob', 'age': 25})
        self.collection.insert_one({'name': 'Charlie', 'age': 35})
        
        # Sort by age ascending
        result = self.collection.find(sort=[('age', 1)])
        assert [doc['name'] for doc in result] == ['Bob', 'Alice', 'Charlie']
        
        # Sort by age descending
        result = self.collection.find(sort=[('age', -1)])
        assert [doc['name'] for doc in result] == ['Charlie', 'Alice', 'Bob']
    
    def test_find_one(self):
        """Test find_one returns a single document."""
        self.collection.insert_one({'name': 'Alice', 'age': 30})
        self.collection.insert_one({'name': 'Bob', 'age': 25})
        
        result = self.collection.find_one({'age': 25})
        assert result is not None
        assert isinstance(result, dict)
    
    def test_find_one_no_match(self):
        """Test find_one with no matching documents."""
        result = self.collection.find_one({'name': 'NonExistent'})
        assert result is None
    
    def test_update_one_no_match(self):
        """Test update_one with no matching documents."""
        result = self.collection.update_one({'name': 'NonExistent'}, {'$set': {'age': 30}})
        assert result['matched_count'] == 0
        assert result['modified_count'] == 0
    
    def test_update_one_with_set(self):
        """Test update_one with $set operator."""
        self.collection.insert_one({'name': 'Alice', 'age': 30})
        
        result = self.collection.update_one({'name': 'Alice'}, {'$set': {'age': 35}})
        assert result['matched_count'] == 1
        assert result['modified_count'] == 1
        assert self.collection._data[0]['age'] == 35
    
    def test_update_one_no_modification(self):
        """Test update_one when no actual modification occurs."""
        self.collection.insert_one({'name': 'Alice', 'age': 30})
        
        result = self.collection.update_one({'name': 'Alice'}, {'$set': {'age': 30}})
        assert result['matched_count'] == 1
        assert result['modified_count'] == 0
    
    def test_update_one_updates_indexes(self):
        """Test that update_one updates indexes correctly."""
        self.collection.create_index('name')
        self.collection.insert_one({'name': 'Alice', 'age': 30})
        
        self.collection.update_one({'name': 'Alice'}, {'$set': {'name': 'Alicia'}})
        
        assert 'Alice' not in self.collection._indexes['name']
        assert len(self.collection._indexes['name']['Alicia']) == 1
    
    def test_count_documents_no_filter(self):
        """Test count_documents with no filter."""
        for i in range(5):
            self.collection.insert_one({'index': i})
        
        assert self.collection.count_documents() == 5
    
    def test_count_documents_with_filter(self):
        """Test count_documents with filter."""
        self.collection.insert_one({'name': 'Alice', 'age': 30})
        self.collection.insert_one({'name': 'Bob', 'age': 25})
        self.collection.insert_one({'name': 'Charlie', 'age': 30})
        
        assert self.collection.count_documents({'age': 30}) == 2
    
    def test_aggregate_empty_pipeline(self):
        """Test aggregate with empty pipeline."""
        self.collection.insert_one({'name': 'Alice'})
        result = self.collection.aggregate([])
        assert len(result) == 1
    
    def test_aggregate_with_sample(self):
        """Test aggregate with $sample operator."""
        for i in range(10):
            self.collection.insert_one({'index': i})
        
        result = self.collection.aggregate([{'$sample': {'size': 5}}])
        assert len(result) == 5
    
    def test_aggregate_with_sample_larger_than_collection(self):
        """Test aggregate with $sample when size > collection size."""
        for i in range(3):
            self.collection.insert_one({'index': i})
        
        result = self.collection.aggregate([{'$sample': {'size': 10}}])
        assert len(result) == 3
    
    def test_delete_one_no_match(self):
        """Test delete_one with no matching documents."""
        result = self.collection.delete_one({'name': 'NonExistent'})
        assert result['deleted_count'] == 0
        assert len(self.collection._data) == 0
    
    def test_delete_one_with_match(self):
        """Test delete_one with matching documents."""
        self.collection.insert_one({'name': 'Alice'})
        self.collection.insert_one({'name': 'Bob'})
        
        result = self.collection.delete_one({'name': 'Alice'})
        assert result['deleted_count'] == 1
        assert len(self.collection._data) == 1
        assert self.collection._data[0]['name'] == 'Bob'
    
    def test_delete_one_updates_indexes(self):
        """Test that delete_one updates indexes correctly."""
        self.collection.create_index('name')
        self.collection.insert_one({'name': 'Alice'})
        self.collection.insert_one({'name': 'Bob'})
        
        self.collection.delete_one({'name': 'Alice'})
        
        # assert 'Alice' not in self.collection._indexes['name']
        assert len(self.collection._indexes['name']) == 1
    
    def test_drop(self):
        """Test dropping the collection."""
        self.collection.insert_one({'name': 'Alice'})
        self.collection.create_index('name')
        
        self.collection.drop()
        
        assert self.collection._data == []
        assert self.collection._indexes == {'_id': {}}
    
    def test_thread_safety(self):
        """Test that operations are thread-safe."""
        import threading
        
        def insert_docs():
            for i in range(100):
                self.collection.insert_one({'index': i})
        
        threads = [threading.Thread(target=insert_docs) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(self.collection._data) == 500
    
    def test_filter_with_indexed_field(self):
        """Test _filter method with indexed field."""
        self.collection.create_index('name')
        self.collection.insert_one({'name': 'Alice'})
        self.collection.insert_one({'name': 'Bob'})
        
        found = self.collection._filter({'name': 'Alice'})
        assert len(found) == 1
    
    def test_filter_with_non_indexed_field(self):
        """Test _filter method with non-indexed field."""
        self.collection.insert_one({'name': 'Alice', 'age': 30})
        self.collection.insert_one({'name': 'Bob', 'age': 25})
        
        found = self.collection._filter({'age': 30})
        assert len(found) == 1
    
    def test_filter_with_multiple_conditions(self):
        """Test _filter method with multiple conditions."""
        self.collection.insert_one({'name': 'Alice', 'age': 30})
        self.collection.insert_one({'name': 'Bob', 'age': 30})
        self.collection.insert_one({'name': 'Alice', 'age': 25})
        
        found = self.collection._filter({'name': 'Alice', 'age': 30})
        assert len(found) == 1
