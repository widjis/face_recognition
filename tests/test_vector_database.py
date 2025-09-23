"""Tests for vector database module."""

import pytest
import numpy as np
import tempfile
import shutil
import os
from datetime import datetime
from face_recognition.vector_db import VectorDatabase
from face_recognition.models import FaceEmbedding, SearchResult
from face_recognition.exceptions import VectorDatabaseError


class TestVectorDatabase:
    """Test cases for VectorDatabase class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        self.db = VectorDatabase(dimension=512, index_type="flat", db_path=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_embedding(self, seed: int = 42) -> FaceEmbedding:
        """Create a test embedding with deterministic random data."""
        np.random.seed(seed)
        vector = np.random.rand(512).astype(np.float32)
        vector = vector / np.linalg.norm(vector)  # Normalize
        
        return FaceEmbedding(
            vector=vector,
            dimension=512,
            model_version="test_v1.0",
            extraction_timestamp=datetime.now()
        )
    
    def test_database_initialization(self):
        """Test database initialization."""
        assert self.db.dimension == 512
        assert self.db.index_type == "flat"
        assert self.db.index is not None
        assert len(self.db) == 0
    
    def test_invalid_index_type(self):
        """Test initialization with invalid index type."""
        with pytest.raises(VectorDatabaseError, match="Unsupported index type"):
            VectorDatabase(dimension=512, index_type="invalid", db_path=self.temp_dir)
    
    def test_store_embedding(self):
        """Test storing a single embedding."""
        embedding = self.create_test_embedding()
        metadata = {"name": "John Doe", "age": 30}
        
        embedding_id = self.db.store_embedding(embedding, metadata)
        
        assert isinstance(embedding_id, str)
        assert embedding_id.startswith("emb_")
        assert len(self.db) == 1
        assert embedding_id in self.db
    
    def test_store_embedding_wrong_dimension(self):
        """Test storing embedding with wrong dimension."""
        # Create embedding with wrong dimension
        vector = np.random.rand(256).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        
        embedding = FaceEmbedding(
            vector=vector,
            dimension=256,
            model_version="test_v1.0",
            extraction_timestamp=datetime.now()
        )
        
        metadata = {"name": "John Doe"}
        
        with pytest.raises(VectorDatabaseError, match="Embedding dimension 256 doesn't match database dimension 512"):
            self.db.store_embedding(embedding, metadata)
    
    def test_search_similar_empty_database(self):
        """Test searching in empty database."""
        query_embedding = self.create_test_embedding()
        
        results = self.db.search_similar(query_embedding, top_k=5)
        
        assert results == []
    
    def test_search_similar_single_embedding(self):
        """Test searching with single embedding in database."""
        # Store an embedding
        embedding = self.create_test_embedding(seed=42)
        metadata = {"name": "John Doe", "id": "001"}
        embedding_id = self.db.store_embedding(embedding, metadata)
        
        # Search with same embedding (should find itself)
        results = self.db.search_similar(embedding, top_k=5)
        
        assert len(results) == 1
        assert results[0].embedding_id == embedding_id
        assert results[0].similarity_score > 0.99  # Should be very similar to itself
        assert results[0].metadata["name"] == "John Doe"
    
    def test_search_similar_multiple_embeddings(self):
        """Test searching with multiple embeddings in database."""
        # Store multiple embeddings
        embeddings_data = [
            (self.create_test_embedding(seed=1), {"name": "Alice", "id": "001"}),
            (self.create_test_embedding(seed=2), {"name": "Bob", "id": "002"}),
            (self.create_test_embedding(seed=3), {"name": "Charlie", "id": "003"}),
        ]
        
        stored_ids = []
        for embedding, metadata in embeddings_data:
            embedding_id = self.db.store_embedding(embedding, metadata)
            stored_ids.append(embedding_id)
        
        # Search with first embedding
        query_embedding = embeddings_data[0][0]
        results = self.db.search_similar(query_embedding, top_k=3)
        
        assert len(results) <= 3
        assert len(results) > 0
        
        # First result should be the same embedding (highest similarity)
        assert results[0].embedding_id == stored_ids[0]
        assert results[0].similarity_score > 0.99
    
    def test_search_similar_with_threshold(self):
        """Test searching with similarity threshold."""
        # Store embeddings
        embedding1 = self.create_test_embedding(seed=1)
        embedding2 = self.create_test_embedding(seed=2)
        
        self.db.store_embedding(embedding1, {"name": "Alice"})
        self.db.store_embedding(embedding2, {"name": "Bob"})
        
        # Search with high threshold
        results = self.db.search_similar(embedding1, top_k=5, threshold=0.99)
        
        # Should only return very similar results (likely just the same embedding)
        assert len(results) >= 1
        assert all(result.similarity_score >= 0.99 for result in results)
    
    def test_search_similar_wrong_dimension(self):
        """Test searching with wrong dimension embedding."""
        # Store an embedding
        embedding = self.create_test_embedding()
        self.db.store_embedding(embedding, {"name": "John"})
        
        # Create query with wrong dimension
        vector = np.random.rand(256).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        
        query_embedding = FaceEmbedding(
            vector=vector,
            dimension=256,
            model_version="test_v1.0",
            extraction_timestamp=datetime.now()
        )
        
        with pytest.raises(VectorDatabaseError, match="Query embedding dimension 256 doesn't match database dimension 512"):
            self.db.search_similar(query_embedding)
    
    def test_delete_embedding(self):
        """Test deleting an embedding."""
        embedding = self.create_test_embedding()
        metadata = {"name": "John Doe"}
        
        embedding_id = self.db.store_embedding(embedding, metadata)
        assert embedding_id in self.db
        
        # Delete the embedding
        success = self.db.delete_embedding(embedding_id)
        
        assert success is True
        assert embedding_id not in self.db
    
    def test_delete_nonexistent_embedding(self):
        """Test deleting non-existent embedding."""
        success = self.db.delete_embedding("nonexistent_id")
        assert success is False
    
    def test_get_embedding_info(self):
        """Test getting embedding information."""
        embedding = self.create_test_embedding()
        metadata = {"name": "John Doe", "age": 30}
        
        embedding_id = self.db.store_embedding(embedding, metadata)
        
        info = self.db.get_embedding_info(embedding_id)
        
        assert info is not None
        assert info["id"] == embedding_id
        assert info["metadata"]["name"] == "John Doe"
        assert info["metadata"]["age"] == 30
        assert info["model_version"] == "test_v1.0"
        assert info["dimension"] == 512
    
    def test_get_nonexistent_embedding_info(self):
        """Test getting info for non-existent embedding."""
        info = self.db.get_embedding_info("nonexistent_id")
        assert info is None
    
    def test_get_database_stats(self):
        """Test getting database statistics."""
        # Initially empty
        stats = self.db.get_database_stats()
        
        assert stats["total_embeddings"] == 0
        assert stats["dimension"] == 512
        assert stats["index_type"] == "flat"
        assert stats["metadata_entries"] == 0
        
        # Add some embeddings
        for i in range(3):
            embedding = self.create_test_embedding(seed=i)
            self.db.store_embedding(embedding, {"name": f"Person_{i}"})
        
        stats = self.db.get_database_stats()
        assert stats["total_embeddings"] == 3
        assert stats["metadata_entries"] == 3
    
    def test_list_all_embeddings(self):
        """Test listing all embeddings."""
        # Initially empty
        embeddings = self.db.list_all_embeddings()
        assert embeddings == []
        
        # Add embeddings
        names = ["Alice", "Bob", "Charlie"]
        for i, name in enumerate(names):
            embedding = self.create_test_embedding(seed=i)
            self.db.store_embedding(embedding, {"name": name})
        
        embeddings = self.db.list_all_embeddings()
        assert len(embeddings) == 3
        
        stored_names = [emb["metadata"]["name"] for emb in embeddings]
        assert set(stored_names) == set(names)
    
    def test_clear_database(self):
        """Test clearing the database."""
        # Add some embeddings
        for i in range(3):
            embedding = self.create_test_embedding(seed=i)
            self.db.store_embedding(embedding, {"name": f"Person_{i}"})
        
        assert len(self.db) == 3
        
        # Clear database
        self.db.clear_database()
        
        assert len(self.db) == 0
        assert self.db.list_all_embeddings() == []
    
    def test_batch_store_embeddings(self):
        """Test batch storing embeddings."""
        # Prepare batch data
        batch_data = []
        for i in range(5):
            embedding = self.create_test_embedding(seed=i)
            metadata = {"name": f"Person_{i}", "id": f"00{i}"}
            batch_data.append((embedding, metadata))
        
        # Batch store
        embedding_ids = self.db.batch_store_embeddings(batch_data)
        
        assert len(embedding_ids) == 5
        assert len(self.db) == 5
        assert all(emb_id in self.db for emb_id in embedding_ids)
    
    def test_batch_store_empty_list(self):
        """Test batch storing empty list."""
        embedding_ids = self.db.batch_store_embeddings([])
        assert embedding_ids == []
        assert len(self.db) == 0
    
    def test_batch_store_wrong_dimension(self):
        """Test batch storing with wrong dimension."""
        # Create embedding with wrong dimension
        vector = np.random.rand(256).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        
        embedding = FaceEmbedding(
            vector=vector,
            dimension=256,
            model_version="test_v1.0",
            extraction_timestamp=datetime.now()
        )
        
        batch_data = [(embedding, {"name": "John"})]
        
        with pytest.raises(VectorDatabaseError, match="Embedding dimension 256 doesn't match database dimension 512"):
            self.db.batch_store_embeddings(batch_data)
    
    def test_database_persistence(self):
        """Test that database persists across instances."""
        # Store embedding in first instance
        embedding = self.create_test_embedding()
        metadata = {"name": "John Doe"}
        embedding_id = self.db.store_embedding(embedding, metadata)
        
        # Create new database instance with same path
        db2 = VectorDatabase(dimension=512, index_type="flat", db_path=self.temp_dir)
        
        # Should load existing data
        assert len(db2) == 1
        assert embedding_id in db2
        
        info = db2.get_embedding_info(embedding_id)
        assert info["metadata"]["name"] == "John Doe"


class TestVectorDatabaseIntegration:
    """Integration tests for vector database."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_different_index_types(self):
        """Test different FAISS index types."""
        index_types = ["flat", "hnsw"]  # Skip IVF for simplicity in tests
        
        for index_type in index_types:
            db_path = os.path.join(self.temp_dir, f"db_{index_type}")
            db = VectorDatabase(dimension=128, index_type=index_type, db_path=db_path)
            
            # Store and search
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            
            embedding = FaceEmbedding(
                vector=vector,
                dimension=128,
                model_version="test_v1.0",
                extraction_timestamp=datetime.now()
            )
            
            embedding_id = db.store_embedding(embedding, {"name": "Test"})
            results = db.search_similar(embedding, top_k=1)
            
            assert len(results) == 1
            assert results[0].embedding_id == embedding_id
    
    def test_large_batch_operations(self):
        """Test operations with larger batches."""
        db = VectorDatabase(dimension=256, index_type="flat", db_path=self.temp_dir)
        
        # Create large batch
        batch_size = 100
        batch_data = []
        
        for i in range(batch_size):
            vector = np.random.rand(256).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            
            embedding = FaceEmbedding(
                vector=vector,
                dimension=256,
                model_version="test_v1.0",
                extraction_timestamp=datetime.now()
            )
            
            metadata = {"name": f"Person_{i:03d}", "batch": "test"}
            batch_data.append((embedding, metadata))
        
        # Batch store
        embedding_ids = db.batch_store_embeddings(batch_data)
        
        assert len(embedding_ids) == batch_size
        assert len(db) == batch_size
        
        # Test search
        query_embedding = batch_data[0][0]
        results = db.search_similar(query_embedding, top_k=10)
        
        assert len(results) <= 10
        assert len(results) > 0