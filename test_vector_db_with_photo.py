"""Test vector database with your real photo."""

import cv2
import numpy as np
import tempfile
import shutil
from datetime import datetime

# Import our modules
from face_recognition.face_detection import FaceDetector
from face_recognition.embedding import EmbeddingExtractor
from face_recognition.models import FaceEmbedding

# Import FAISS for vector database
import faiss
import json
import os

class SimpleVectorDB:
    """Simple vector database for testing."""
    
    def __init__(self, dimension=512, db_path=None):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata_store = {}
        self.id_counter = 0
        self.db_path = db_path or tempfile.mkdtemp()
        
        # Ensure directory exists
        os.makedirs(self.db_path, exist_ok=True)
        
        # Try to load existing database
        self._load_database()
    
    def _load_database(self):
        """Load existing database if it exists."""
        index_path = os.path.join(self.db_path, "index.faiss")
        metadata_path = os.path.join(self.db_path, "metadata.json")
        
        try:
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.index = faiss.read_index(index_path)
                
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self.metadata_store = data.get('metadata', {})
                    self.id_counter = data.get('id_counter', 0)
                
                print(f"✅ Loaded existing database with {self.index.ntotal} embeddings")
        except Exception as e:
            print(f"⚠️ Could not load existing database: {e}")
    
    def _save_database(self):
        """Save database to disk."""
        try:
            index_path = os.path.join(self.db_path, "index.faiss")
            metadata_path = os.path.join(self.db_path, "metadata.json")
            
            faiss.write_index(self.index, index_path)
            
            data = {
                'metadata': self.metadata_store,
                'id_counter': self.id_counter,
                'dimension': self.dimension,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save database: {e}")
    
    def store_embedding(self, embedding, metadata):
        """Store an embedding with metadata."""
        embedding_id = f"emb_{self.id_counter:06d}"
        self.id_counter += 1
        
        vector = embedding.vector.reshape(1, -1).astype(np.float32)
        self.index.add(vector)
        
        self.metadata_store[embedding_id] = {
            'id': embedding_id,
            'metadata': metadata,
            'model_version': embedding.model_version,
            'extraction_timestamp': embedding.extraction_timestamp.isoformat(),
            'storage_timestamp': datetime.now().isoformat()
        }
        
        self._save_database()
        return embedding_id
    
    def search_similar(self, query_embedding, top_k=10, threshold=0.0):
        """Search for similar embeddings."""
        if self.index.ntotal == 0:
            return []
        
        query_vector = query_embedding.vector.reshape(1, -1).astype(np.float32)
        similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        results = []
        embedding_ids = list(self.metadata_store.keys())
        
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1 or similarity < threshold:
                continue
            
            if idx < len(embedding_ids):
                embedding_id = embedding_ids[idx]
                metadata_entry = self.metadata_store[embedding_id]
                
                result = type('SearchResult', (), {
                    'embedding_id': embedding_id,
                    'similarity_score': float(similarity),
                    'metadata': metadata_entry['metadata']
                })()
                results.append(result)
        
        return sorted(results, key=lambda x: x.similarity_score, reverse=True)
    
    def list_all_embeddings(self):
        """List all stored embeddings."""
        return list(self.metadata_store.values())
    
    def __len__(self):
        return self.index.ntotal

def test_face_recognition_with_vector_db():
    """Test complete face recognition pipeline with vector database."""
    print("🎯 Testing Complete Face Recognition Pipeline")
    print("=" * 50)
    
    # Load your photo
    image = cv2.imread("MTI230279.jpg")
    if image is None:
        print("❌ Could not load MTI230279.jpg")
        return False
    
    print(f"📸 Photo loaded: {image.shape}")
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize components
        detector = FaceDetector(method="haar", min_face_size=(20, 20))
        extractor = EmbeddingExtractor(model_name="simple", embedding_dim=512)
        db = SimpleVectorDB(dimension=512, db_path=temp_dir)
        
        print(f"✅ Components initialized")
        print(f"   Database path: {temp_dir}")
        print(f"   Initial database size: {len(db)}")
        
        # Step 1: Detect faces
        print(f"\n🔍 Step 1: Face Detection")
        faces = detector.detect_faces(image)
        print(f"   Found {len(faces)} face(s)")
        
        if not faces:
            print("   No faces detected - cannot proceed")
            return False
        
        # Step 2: Extract embeddings and store in database
        print(f"\n🧠 Step 2: Extract Embeddings and Store in Database")
        
        stored_embeddings = []
        for i, face in enumerate(faces):
            print(f"\n   Processing Face {i+1}:")
            print(f"   Position: ({face.x}, {face.y}), Size: {face.width}x{face.height}")
            
            # Preprocess face
            processed_face = detector.preprocess_face(image, face)
            
            # Extract embedding
            embedding = extractor.extract_embedding(processed_face)
            
            # Store in database with metadata
            metadata = {
                "name": f"Person_from_MTI230279_{i+1}",
                "source_image": "MTI230279.jpg",
                "face_position": {"x": face.x, "y": face.y, "width": face.width, "height": face.height},
                "confidence": face.confidence,
                "processing_date": datetime.now().isoformat()
            }
            
            embedding_id = db.store_embedding(embedding, metadata)
            stored_embeddings.append((embedding_id, embedding, metadata))
            
            print(f"   ✅ Stored embedding: {embedding_id}")
            print(f"      Dimension: {embedding.dimension}")
            print(f"      Vector norm: {np.linalg.norm(embedding.vector):.6f}")
        
        print(f"\n📊 Database now contains {len(db)} embeddings")
        
        # Step 3: Test similarity search
        print(f"\n🔍 Step 3: Test Similarity Search")
        
        if stored_embeddings:
            # Search with the first stored embedding
            query_embedding = stored_embeddings[0][1]
            query_metadata = stored_embeddings[0][2]
            
            print(f"   Searching for similar faces to: {query_metadata['name']}")
            
            results = db.search_similar(query_embedding, top_k=5, threshold=0.5)
            
            print(f"   Found {len(results)} similar faces:")
            for i, result in enumerate(results):
                print(f"   {i+1}. {result.metadata['name']}")
                print(f"      Similarity: {result.similarity_score:.4f}")
                print(f"      Source: {result.metadata['source_image']}")
        
        # Step 4: Test with different models
        print(f"\n🔬 Step 4: Test Different Embedding Models")
        
        if faces:
            face = faces[0]
            processed_face = detector.preprocess_face(image, face)
            
            models = ["simple", "facenet", "arcface"]
            model_embeddings = {}
            
            for model_name in models:
                model_extractor = EmbeddingExtractor(model_name=model_name, embedding_dim=512)
                embedding = model_extractor.extract_embedding(processed_face)
                
                # Store in database
                metadata = {
                    "name": f"Person_MTI230279_{model_name}",
                    "model": model_name,
                    "source_image": "MTI230279.jpg"
                }
                
                embedding_id = db.store_embedding(embedding, metadata)
                model_embeddings[model_name] = (embedding_id, embedding)
                
                print(f"   ✅ Stored {model_name} embedding: {embedding_id}")
        
        # Step 5: Cross-model similarity test
        print(f"\n🔄 Step 5: Cross-Model Similarity Test")
        
        if len(model_embeddings) >= 2:
            model_names = list(model_embeddings.keys())
            
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    
                    # Search for model1 embedding using model2 as query
                    query_embedding = model_embeddings[model2][1]
                    results = db.search_similar(query_embedding, top_k=10)
                    
                    # Find the model1 result
                    model1_result = None
                    for result in results:
                        if result.metadata.get('model') == model1:
                            model1_result = result
                            break
                    
                    if model1_result:
                        print(f"   {model2} → {model1}: {model1_result.similarity_score:.4f}")
        
        # Step 6: Database statistics
        print(f"\n📈 Step 6: Final Database Statistics")
        print(f"   Total embeddings: {len(db)}")
        
        all_embeddings = db.list_all_embeddings()
        print(f"   Stored faces:")
        for emb in all_embeddings:
            name = emb['metadata']['name']
            model = emb['metadata'].get('model', 'N/A')
            print(f"   - {name} (model: {model})")
        
        print(f"\n🎉 Complete pipeline test successful!")
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
    success = test_face_recognition_with_vector_db()
    if success:
        print(f"\n✅ Complete Face Recognition System Working!")
        print(f"   1. ✅ Face Detection")
        print(f"   2. ✅ Embedding Extraction") 
        print(f"   3. ✅ Vector Database Storage")
        print(f"   4. ✅ Similarity Search")
        print(f"   5. ✅ Multi-Model Support")
        print(f"   ")
        print(f"   Your photo has been processed through the complete system!")
    else:
        print(f"\n❌ Face recognition pipeline test failed!")
        exit(1)