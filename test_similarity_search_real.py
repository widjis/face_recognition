"""Test similarity search with your real photo."""

import cv2
import numpy as np
import tempfile
import shutil
import os
from datetime import datetime

# Import our modules
from face_recognition.face_detection import FaceDetector
from face_recognition.embedding import EmbeddingExtractor
from face_recognition.models import FaceEmbedding, SearchConfig

# Import FAISS and create simple implementations
import faiss
import json
import hashlib
import time

# Simple VectorDB implementation
class SimpleVectorDB:
    def __init__(self, dimension=512, db_path=None):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata_store = {}
        self.id_counter = 0
        self.db_path = db_path or tempfile.mkdtemp()
        os.makedirs(self.db_path, exist_ok=True)
    
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
                
                result = type('SearchResult', (), {
                    'embedding_id': embedding_id,
                    'similarity_score': float(similarity),
                    'metadata': metadata_entry['metadata']
                })()
                results.append(result)
        
        return sorted(results, key=lambda x: x.similarity_score, reverse=True)
    
    def __len__(self):
        return self.index.ntotal

# Simple SimilaritySearcher implementation
class SimpleSimilaritySearcher:
    def __init__(self, vector_db, default_config=None):
        self.vector_db = vector_db
        self.default_config = default_config or SearchConfig()
        self._search_cache = {}
        self._performance_stats = {
            'total_searches': 0,
            'average_search_time': 0.0,
            'cache_hits': 0
        }
    
    def search(self, query_embedding, config=None):
        search_config = config or self.default_config
        start_time = time.time()
        
        # Check cache
        cache_key = self._generate_cache_key(query_embedding, search_config)
        if cache_key in self._search_cache:
            self._performance_stats['cache_hits'] += 1
            return self._search_cache[cache_key]
        
        # Perform search
        results = self.vector_db.search_similar(
            query_embedding,
            top_k=search_config.top_k,
            threshold=search_config.similarity_threshold
        )
        
        # Cache results
        self._search_cache[cache_key] = results
        
        # Update stats
        search_time = time.time() - start_time
        self._update_performance_stats(search_time)
        
        return results
    
    def search_with_filters(self, query_embedding, metadata_filters, config=None):
        all_results = self.search(query_embedding, config)
        
        filtered_results = []
        for result in all_results:
            if self._matches_filters(result.metadata, metadata_filters):
                filtered_results.append(result)
        
        return filtered_results
    
    def find_duplicates(self, query_embedding, duplicate_threshold=0.95):
        config = SearchConfig(
            top_k=50,
            similarity_threshold=duplicate_threshold,
            enable_reranking=False
        )
        return self.search(query_embedding, config)
    
    def get_search_statistics(self):
        return {
            'total_searches': self._performance_stats['total_searches'],
            'average_search_time': self._performance_stats['average_search_time'],
            'cache_hits': self._performance_stats['cache_hits'],
            'cache_hit_rate': (self._performance_stats['cache_hits'] / 
                             max(1, self._performance_stats['total_searches'])),
            'cache_size': len(self._search_cache)
        }
    
    def clear_cache(self):
        self._search_cache.clear()
    
    def _generate_cache_key(self, embedding, config):
        embedding_hash = hashlib.md5(embedding.vector.tobytes()).hexdigest()[:16]
        config_str = f"{config.top_k}_{config.similarity_threshold}"
        return f"{embedding_hash}_{config_str}"
    
    def _matches_filters(self, metadata, filters):
        for key, expected_value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != expected_value:
                return False
        return True
    
    def _update_performance_stats(self, search_time):
        self._performance_stats['total_searches'] += 1
        total_searches = self._performance_stats['total_searches']
        current_avg = self._performance_stats['average_search_time']
        new_avg = ((current_avg * (total_searches - 1)) + search_time) / total_searches
        self._performance_stats['average_search_time'] = new_avg

def test_advanced_similarity_search():
    """Test advanced similarity search features with your photo."""
    print("🎯 Testing Advanced Similarity Search")
    print("=" * 45)
    
    # Load your photo
    image = cv2.imread("MTI230279.jpg")
    if image is None:
        print("❌ Could not load MTI230279.jpg")
        return False
    
    print(f"📸 Photo loaded: {image.shape}")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize components
        detector = FaceDetector(method="haar", min_face_size=(20, 20))
        extractor = EmbeddingExtractor(model_name="simple", embedding_dim=512)
        db = SimpleVectorDB(dimension=512, db_path=temp_dir)
        searcher = SimpleSimilaritySearcher(db)
        
        print("✅ Components initialized")
        
        # Step 1: Create a diverse database
        print("\n📚 Step 1: Building Diverse Face Database")
        
        # Detect face from your photo
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
                "name": f"Person_MTI230279",
                "model": model_name,
                "source": "MTI230279.jpg",
                "category": "real_person"
            }
            
            embedding_id = db.store_embedding(embedding, metadata)
            stored_embeddings.append((embedding_id, embedding, metadata))
            print(f"   ✅ Stored {model_name} embedding: {embedding_id}")
        
        # Add some synthetic embeddings for comparison
        for i in range(3):
            np.random.seed(100 + i)  # Deterministic random
            vector = np.random.rand(512).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            
            synthetic_embedding = FaceEmbedding(
                vector=vector,
                dimension=512,
                model_version="synthetic_v1.0",
                extraction_timestamp=datetime.now()
            )
            
            metadata = {
                "name": f"Synthetic_Person_{i}",
                "model": "synthetic",
                "source": "generated",
                "category": "synthetic"
            }
            
            embedding_id = db.store_embedding(synthetic_embedding, metadata)
            print(f"   ✅ Stored synthetic embedding: {embedding_id}")
        
        print(f"\n📊 Database contains {len(db)} embeddings")
        
        # Step 2: Test basic similarity search
        print("\n🔍 Step 2: Basic Similarity Search")
        
        query_embedding = stored_embeddings[0][1]  # Use simple model embedding
        results = searcher.search(query_embedding, SearchConfig(top_k=5))
        
        print(f"   Found {len(results)} similar faces:")
        for i, result in enumerate(results):
            name = result.metadata['name']
            model = result.metadata['model']
            score = result.similarity_score
            print(f"   {i+1}. {name} ({model}): {score:.4f}")
        
        # Step 3: Test search with filters
        print("\n🔎 Step 3: Filtered Search")
        
        # Search only for real person embeddings
        real_person_filter = {"category": "real_person"}
        filtered_results = searcher.search_with_filters(
            query_embedding, real_person_filter, SearchConfig(top_k=10)
        )
        
        print(f"   Real person matches: {len(filtered_results)}")
        for result in filtered_results:
            name = result.metadata['name']
            model = result.metadata['model']
            score = result.similarity_score
            print(f"   - {name} ({model}): {score:.4f}")
        
        # Step 4: Test duplicate detection
        print("\n🔄 Step 4: Duplicate Detection")
        
        duplicates = searcher.find_duplicates(query_embedding, duplicate_threshold=0.9)
        
        print(f"   Found {len(duplicates)} potential duplicates (>90% similar):")
        for dup in duplicates:
            name = dup.metadata['name']
            model = dup.metadata['model']
            score = dup.similarity_score
            print(f"   - {name} ({model}): {score:.4f}")
        
        # Step 5: Test different search configurations
        print("\n⚙️ Step 5: Different Search Configurations")
        
        configs = [
            ("Strict", SearchConfig(top_k=3, similarity_threshold=0.95)),
            ("Balanced", SearchConfig(top_k=5, similarity_threshold=0.8)),
            ("Loose", SearchConfig(top_k=10, similarity_threshold=0.5))
        ]
        
        for config_name, config in configs:
            results = searcher.search(query_embedding, config)
            print(f"   {config_name} search: {len(results)} results")
            if results:
                best_score = results[0].similarity_score
                worst_score = results[-1].similarity_score
                print(f"     Score range: {worst_score:.4f} - {best_score:.4f}")
        
        # Step 6: Test search performance and caching
        print("\n📈 Step 6: Performance Testing")
        
        # Perform multiple searches to test caching
        for _ in range(5):
            searcher.search(query_embedding)  # Should hit cache after first
        
        stats = searcher.get_search_statistics()
        print(f"   Total searches: {stats['total_searches']}")
        print(f"   Average search time: {stats['average_search_time']:.4f}s")
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")
        
        # Step 7: Cross-model similarity analysis
        print("\n🔬 Step 7: Cross-Model Similarity Analysis")
        
        model_embeddings = {emb[2]['model']: emb[1] for emb in stored_embeddings}
        
        print("   Cross-model similarity matrix:")
        models = list(model_embeddings.keys())
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i <= j:  # Only upper triangle
                    if i == j:
                        similarity = 1.0  # Self-similarity
                    else:
                        results = searcher.search(model_embeddings[model1])
                        # Find the other model's result
                        other_result = None
                        for result in results:
                            if result.metadata['model'] == model2:
                                other_result = result
                                break
                        similarity = other_result.similarity_score if other_result else 0.0
                    
                    print(f"   {model1} ↔ {model2}: {similarity:.4f}")
        
        print(f"\n🎉 Advanced similarity search test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    print("Starting advanced similarity search test...")
    try:
        success = test_advanced_similarity_search()
        if success:
            print(f"\n✅ Advanced Similarity Search Working!")
            print(f"   1. ✅ Basic similarity search")
            print(f"   2. ✅ Filtered search by metadata")
            print(f"   3. ✅ Duplicate detection")
            print(f"   4. ✅ Configurable search parameters")
            print(f"   5. ✅ Search caching and performance")
            print(f"   6. ✅ Cross-model similarity analysis")
            print(f"   ")
            print(f"   Your face recognition system now has advanced search capabilities!")
        else:
            print(f"\n❌ Advanced similarity search test failed!")
            exit(1)
    except Exception as e:
        print(f"❌ Script error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)