"""Simple test for vector database without pytest."""

import sys
import os
import tempfile
import shutil
import numpy as np
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, '.')

# Import directly from the module file
from face_recognition.models import FaceEmbedding
from face_recognition.exceptions import VectorDatabaseError

# Import the VectorDatabase class directly
exec(open('face_recognition/vector_db/database.py').read())

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
    """Test basic vector database functionality."""
    print("🎯 Testing Vector Database")
    print("=" * 30)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize database
        print("📁 Initializing database...")
        db = VectorDatabase(dimension=512, index_type="flat", db_path=temp_dir)
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
        for i, (embedding, metadata) in enumerate(embeddings_data):
            embedding_id = db.store_embedding(embedding, metadata)
            stored_ids.append(embedding_id)
            print(f"   ✅ Stored {metadata['name']}: {embedding_id}")
        
        print(f"\n📊 Database stats: {len(db)} embeddings")
        
        # Test search
        print("\n🔍 Testing similarity search...")
        query_embedding = embeddings_data[0][0]  # Use Alice's embedding
        results = db.search_similar(query_embedding, top_k=3)
        
        print(f"   Found {len(results)} similar embeddings:")
        for i, result in enumerate(results):
            print(f"   {i+1}. {result.metadata['name']}: {result.similarity_score:.4f}")
        
        # Test batch storage
        print("\n📦 Testing batch storage...")
        batch_data = [
            (create_test_embedding(seed=10), {"name": "David", "id": "004"}),
            (create_test_embedding(seed=11), {"name": "Eve", "id": "005"}),
        ]
        
        batch_ids = db.batch_store_embeddings(batch_data)
        print(f"   ✅ Batch stored {len(batch_ids)} embeddings")
        print(f"   Total embeddings: {len(db)}")
        
        # Test database stats
        print("\n📈 Database statistics:")
        stats = db.get_database_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test persistence
        print("\n💿 Testing persistence...")
        db2 = VectorDatabase(dimension=512, index_type="flat", db_path=temp_dir)
        print(f"   ✅ Reloaded database: {len(db2)} embeddings")
        
        # Verify data persisted
        all_embeddings = db2.list_all_embeddings()
        names = [emb['metadata']['name'] for emb in all_embeddings]
        print(f"   Stored names: {names}")
        
        print(f"\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    success = test_vector_database()
    if success:
        print("\n✅ Vector database is working correctly!")
    else:
        print("\n❌ Vector database tests failed!")
        sys.exit(1)