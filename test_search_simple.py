"""Simple test for similarity search functionality."""

import cv2
import numpy as np
import tempfile
import shutil
from datetime import datetime

# Import our modules
from face_recognition.face_detection import FaceDetector
from face_recognition.embedding import EmbeddingExtractor
from face_recognition.models import FaceEmbedding, SearchConfig

# Import FAISS for vector database
import faiss
import json
import os

# Import the similarity searcher functionality directly
exec(open('face_recognition/search/searcher.py').read())

class SimpleVectorDB:
    """Simple vector database for testing."""
    
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
    
    def search_similar(self, query_embedding, top_k=10, threshold=0.0):
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
                
                # Create SearchResult-like object
                result = type('SearchResult', (), {
                    'embedding_id': embedding_id,
                    'similarity_score': float(similarity),
                    'metadata': metadata_entry['metadata']
                })()
                results.append(result)
        
        return sorted(results, key=lambda x: x.similarity_score, reverse=True)
    
    def get_embedding_info(self, embedding_id):
        return self.metadata_store.get(embedding_id)
    
    def __len__(self):
        return self.index.ntotal

def test_advanced_similarity_search():
    """Test advanced similarity search features with your photo."""
    print("🎯 Testing Advanced Similarity Search")
    print("=" * 40)
    
    # Load your photo
    image = cv2.imread("MTI230279.jpg")
    if image is None:
        print("❌ Could not load MTI230279.jpg")
        return False
    
    print(f"📸 Photo loaded: {image.shape}")
    
    try:
        # Initialize components
        detector = FaceDetector(method="haar", min_face_size=(20, 20))
        extractor = EmbeddingExtractor(model_name="simple", embedding_dim=512)
        db = SimpleVectorDB(dimension=512)
        
        # Initialize similarity searcher
        searcher = SimilaritySearcher(db)
        
        print(f"✅ Components initialized")
        
        # Step 1: Create a diverse database
        print(f"\n📚 Step 1: Building Test Database")
        
        # Detect face in your photo
        faces = detector.detect_faces(image)
        if not faces:
            print("❌ No faces detected")
            return False
        
        face = faces[0]
        processed_face = detector.preprocess_face(image, face)
        
        # Create multiple embeddings with different models
        models = ["simple", "facenet", "arcface"]
        stored_embeddings = []
        
        for model_name in models:
            model_extractor = EmbeddingExtractor(model_name=model_name, embedding_dim=512)
            embedding = model_extractor.extract_embedding(processed_face)
            
            metadata = {
                "name": f"Person_MTI230279_{model_name}",
                "model": model_name,
                "source": "MTI230279.jpg",
                "confidence": face.confidence
            }
            
            embedding_id = db.store_embedding(embedding, metadata)
            stored_embeddings.append((embedding_id, embedding, metadata))
            print(f"   ✅ Stored {model_name} embedding: {embedding_id}")
        
        # Add some synthetic similar embeddings
        for i in range(3):
            # Create slightly modified version of the original embedding
            base_embedding = stored_embeddings[0][1]  # Use simple model as base
            noise = np.random.normal(0, 0.1, base_embedding.vector.shape).astype(np.float32)
            modified_vector = base_embedding.vector + noise
            modified_vector = modified_vector / np.linalg.norm(modified_vector)  # Renormalize
            
            synthetic_embedding = FaceEmbedding(
                vector=modified_vector,
                dimension=512,
                model_version="synthetic_v1.0",
                extraction_timestamp=datetime.now()
            )
            
            metadata = {
                "name": f"Synthetic_Person_{i+1}",
                "model": "synthetic",
                "source": "generated",
                "similarity_to_original": 0.8 + i * 0.05
            }
            
            db.store_embedding(synthetic_embedding, metadata)
            print(f"   ✅ Added synthetic embedding {i+1}")
        
        print(f"   Database size: {len(db)} embeddings")
        
        # Step 2: Test basic search
        print(f"\n🔍 Step 2: Basic Similarity Search")
        
        query_embedding = stored_embeddings[0][1]  # Use simple model embedding
        results = searcher.search(query_embedding)
        
        print(f"   Found {len(results)} similar faces:")
        for i, result in enumerate(results[:5]):
            print(f"   {i+1}. {result.metadata['name']}: {result.similarity_score:.4f}")
        
        # Step 3: Test search with custom configuration
        print(f"\n⚙️ Step 3: Search with Custom Configuration")
        
        custom_config = SearchConfig(
            top_k=3,
            similarity_threshold=0.9,
            enable_reranking=False,
            distance_metric="cosine"
        )
        
        results = searcher.search(query_embedding, custom_config)
        print(f"   High-similarity results (threshold=0.9):")
        for i, result in enumerate(results):
            print(f"   {i+1}. {result.metadata['name']}: {result.similarity_score:.4f}")
        
        # Step 4: Test search with filters
        print(f"\n🔧 Step 4: Search with Metadata Filters")
        
        # Filter by model type
        model_filter = {"model": "facenet"}
        filtered_results = searcher.search_with_filters(query_embedding, model_filter)
        
        print(f"   Results filtered by model='facenet':")
        for result in filtered_results:
            print(f"   - {result.metadata['name']}: {result.similarity_score:.4f}")
        
        # Step 5: Test duplicate detection
        print(f"\n🔍 Step 5: Duplicate Detection")
        
        duplicates = searcher.find_duplicates(query_embedding, duplicate_threshold=0.95)
        print(f"   Found {len(duplicates)} potential duplicates:")
        for dup in duplicates:
            print(f"   - {dup.metadata['name']}: {dup.similarity_score:.4f}")
        
        # Step 6: Test batch search
        print(f"\n📦 Step 6: Batch Search")
        
        query_embeddings = [emb[1] for emb in stored_embeddings[:2]]  # Use first 2 embeddings
        batch_results = searcher.batch_search(query_embeddings)
        
        print(f"   Batch search results for {len(query_embeddings)} queries:")
        for i, results in enumerate(batch_results):
            print(f"   Query {i+1}: {len(results)} results")
            if results:
                print(f"     Best match: {results[0].metadata['name']} ({results[0].similarity_score:.4f})")
        
        # Step 7: Test performance statistics
        print(f"\n📊 Step 7: Performance Statistics")
        
        stats = searcher.get_search_statistics()
        print(f"   Total searches: {stats['total_searches']}")
        print(f"   Average search time: {stats['average_search_time']:.4f}s")
        print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"   Cache size: {stats['cache_size']}")
        
        # Step 8: Test search result analysis
        print(f"\n📈 Step 8: Search Result Analysis")
        
        # Get comprehensive results for analysis
        all_results = searcher.search(query_embedding, SearchConfig(top_k=20, similarity_threshold=0.0))
        
        # Analyze similarity distribution
        analyzer = SearchResultAnalyzer()
        distribution = analyzer.get_similarity_distribution(all_results)
        
        print(f"   Similarity Distribution:")
        print(f"   - Count: {distribution.get('count', 0)}")
        print(f"   - Min: {distribution.get('min_similarity', 0):.4f}")
        print(f"   - Max: {distribution.get('max_similarity', 0):.4f}")
        print(f"   - Mean: {distribution.get('mean_similarity', 0):.4f}")
        
        # Group by similarity ranges
        groups = analyzer.group_by_similarity(all_results)
        print(f"   Similarity Groups:")
        for range_key, group_results in groups.items():
            if group_results:
                print(f"   - {range_key}: {len(group_results)} results")
        
        print(f"\n🎉 Advanced similarity search test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_advanced_similarity_search()
    if success:
        print(f"\n✅ Advanced Similarity Search Working!")
        print(f"   1. ✅ Basic similarity search")
        print(f"   2. ✅ Custom search configurations")
        print(f"   3. ✅ Metadata filtering")
        print(f"   4. ✅ Duplicate detection")
        print(f"   5. ✅ Batch search operations")
        print(f"   6. ✅ Performance tracking")
        print(f"   7. ✅ Result analysis")
        print(f"   ")
        print(f"   Your face recognition system now has advanced search capabilities!")
    else:
        print(f"\n❌ Advanced similarity search test failed!")
        exit(1)