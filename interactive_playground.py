"""Interactive playground for experimenting with face recognition models."""

import numpy as np
from datetime import datetime
from face_recognition.models import *
from face_recognition.exceptions import *

print("🎮 Face Recognition Interactive Playground")
print("=" * 50)
print("Available models:")
print("- FaceEmbedding")
print("- FaceRegion") 
print("- SearchResult")
print("- SearchConfig")
print("- RerankingFeatures")
print("- RecognitionRequest")
print("- RecognitionResponse")
print("\nAvailable exceptions:")
print("- FaceRecognitionError")
print("- FaceDetectionError")
print("- EmbeddingExtractionError")
print("- VectorDatabaseError")
print("- InvalidImageError")
print("- SimilaritySearchError")
print("- RerankingError")
print("- ConfigurationError")
print("\nExample usage:")
print(">>> vector = np.random.rand(512)")
print(">>> embedding = FaceEmbedding(vector, 512, 'facenet_v1', datetime.now())")
print(">>> config = SearchConfig(top_k=5, similarity_threshold=0.8)")
print(">>> print(f'Config: {config.top_k} results, threshold {config.similarity_threshold}')")
print("\nTry creating your own objects!")
print("=" * 50)

# Pre-create some sample data for easy experimentation
sample_vector = np.random.rand(512)
sample_embedding = FaceEmbedding(
    vector=sample_vector,
    dimension=512,
    model_version="facenet_v1",
    extraction_timestamp=datetime.now()
)

sample_face_region = FaceRegion(x=100, y=150, width=200, height=250, confidence=0.95)

sample_config = SearchConfig(top_k=5, similarity_threshold=0.8)

sample_result = SearchResult(
    embedding_id="sample_001",
    similarity_score=0.92,
    metadata={"name": "Sample Person", "id": "001"}
)

print(f"\n📦 Pre-created sample objects:")
print(f"- sample_embedding: {sample_embedding.dimension}D vector")
print(f"- sample_face_region: {sample_face_region.width}x{sample_face_region.height} at ({sample_face_region.x}, {sample_face_region.y})")
print(f"- sample_config: top_k={sample_config.top_k}, threshold={sample_config.similarity_threshold}")
print(f"- sample_result: {sample_result.metadata['name']} (score: {sample_result.similarity_score})")

print("\n🔧 Try these experiments:")
print("1. Create different embedding dimensions:")
print("   >>> small_vector = np.random.rand(128)")
print("   >>> small_embedding = FaceEmbedding(small_vector, 128, 'mobilenet', datetime.now())")

print("\n2. Test validation errors:")
print("   >>> try:")
print("   >>>     bad_region = FaceRegion(10, 20, -100, 50, 0.9)  # negative width")
print("   >>> except ValueError as e:")
print("   >>>     print(f'Error: {e}')")

print("\n3. Create search scenarios:")
print("   >>> results = [")
print("   >>>     SearchResult('emb1', 0.95, {'name': 'Alice'}),")
print("   >>>     SearchResult('emb2', 0.87, {'name': 'Bob'})")
print("   >>> ]")
print("   >>> best_match = max(results, key=lambda r: r.similarity_score)")

print("\n4. Experiment with configurations:")
print("   >>> strict_config = SearchConfig(top_k=3, similarity_threshold=0.9)")
print("   >>> loose_config = SearchConfig(top_k=20, similarity_threshold=0.5)")

print("\n🚀 Ready to experiment! The Python environment is loaded with all models.")