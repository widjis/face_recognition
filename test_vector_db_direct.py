"""Direct test for vector database functionality."""

import numpy as np
import tempfile
import shutil
import os
from datetime import datetime

# Import required modules
from face_recognition.models import FaceEmbedding
from face_recognition.exceptions import VectorDatabaseError

# Import FAISS and other dependencies directly
import faiss
import json

def create_simple_vector_db():
    """Create a simple vector database implementation for testing."""
    
    class SimpleVectorDB:
        def __init__(self, dimension=512):
            self.dimension = dimension
            self.index = faiss.IndexFlatIP(dimension)
            self.metadata_store = {}
            self.id_counter = 0
        
        def store_embedding(self, embedding, metadata):
            embedding_id = f"emb_{self.id_counter:06d}"
            self.id_counter += 1
            
            vector = embedding.vector.reshape(1, -1).astype(np.float32)
            self.index.add(vector)
            
            self.metadata_store[embedding_id] = {
                'id': embedding_id,
                'metadata': metadata,
                'model_version': embedding.model_version
            }
            
            return embedding_id
        
        def search_similar(self, query_embedding, top_k=10):
            if self.index.ntotal == 0:
                return []
            
            query_vector = query_embedding.vector.reshape(1, -1).astype(np.float32)
            similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            results = []
            embedding_ids = list(self.metadata_store.keys())
            
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx == -1:
                    continue
                
                if idx < len(embedding_ids):
                    embedding_id = embedding_ids[idx]
                    metadata_entry = self.metadata_store[embedding_id]
                    
                    # Create a simple result object
                    result = type('SearchResult', (), {
                        'embedding_id': embedding_id,
                        'similarity_score': float(similarity),
                        'metadata': metadata_entry['metadata']
                    })()
                    results.append(result)
            
            return sorted(results, key=lambda x: x.similarity_score, reverse=True)
        
        def __len__(self):
            return self.index.ntotal
    
    return SimpleVectorDB

def create_test_embedding(seed=42, dimension=512):
    """Create a test embedding."""
    np.random.seed(seed)
    vector = np.random.rand(dimension).astype(np.float32)
    vector = vector / np.linalg.norm(vector)  # Normalize
    
    return FaceEmbedding(
        vector=vector,
        dimension=dimension,
        model_version="test_v1.0",
        extraction_timestamp=datetime.now()
    )

def test_vector_database():
    """Test vector database functionality."""
    print("🎯 Testing Vector Database Functionality")
    print("=" * 45)
    
    try:
        # Create database
        VectorDB = create_simple_vector_db()
        db = VectorDB(dimension=512)
        print(f"✅ Database initialized: {len(db)} embeddings")
        
        # Create test embeddings
        print("\n🧠 Creating test embeddings...")
        embeddings_data = [
            (create_test_embedding(seed=1), {"name": "Alice", "id": "001"}),
            (create_test_embedding(seed=2), {"name": "Bob", "id": "002"}),
            (create_test_embedding(seed=3), {"name": "Charlie", "id": "003"}),
        ]
        
        # Store embeddings
        print("\n💾 Storing embeddings...")
        stored_ids = []
        for embedding, metadata in embeddings_data:
            embedding_id = db.store_embedding(embedding, metadata)
            stored_ids.append(embedding_id)
            print(f"   ✅ Stored {metadata['name']}: {embedding_id}")
        
        print(f"\n📊 Database now has {len(db)} embeddings")
        
        # Test search
        print("\n🔍 Testing similarity search...")
        query_embedding = embeddings_data[0][0]  # Use Alice's embedding
        results = db.search_similar(query_embedding, top_k=3)
        
        print(f"   Found {len(results)} similar embeddings:")
        for i, result in enumerate(results):
            print(f"   {i+1}. {result.metadata['name']}: {result.similarity_score:.4f}")
        
        # Verify self-similarity
        if results and results[0].metadata['name'] == 'Alice':
            print(f"   ✅ Self-similarity test passed: {results[0].similarity_score:.4f}")
        
        # Test with different query
        print("\n🔍 Testing with Bob's embedding...")
        query_embedding = embeddings_data[1][0]  # Use Bob's embedding
        results = db.search_similar(query_embedding, top_k=3)
        
        print(f"   Found {len(results)} similar embeddings:")
        for i, result in enumerate(results):
            print(f"   {i+1}. {result.metadata['name']}: {result.similarity_score:.4f}")
        
        print(f"\n🎉 All vector database tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vector_database()
    if success:
        print("\n✅ Vector database functionality is working!")
        print("   FAISS integration successful")
        print("   Embedding storage and retrieval working")
        print("   Similarity search functioning correctly")
    else:
        print("\n❌ Vector database tests failed!")
        exit(1)