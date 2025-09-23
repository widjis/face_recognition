"""Interactive demo of the face recognition data models."""

import numpy as np
from datetime import datetime
from face_recognition.models import (
    FaceEmbedding, FaceRegion, SearchResult, SearchConfig,
    RerankingFeatures, RecognitionRequest, RecognitionResponse
)
from face_recognition.exceptions import FaceRecognitionError, InvalidImageError

def demo_face_embedding():
    """Demonstrate FaceEmbedding creation and validation."""
    print("=== FaceEmbedding Demo ===")
    
    # Create a valid face embedding
    vector = np.random.rand(512)
    embedding = FaceEmbedding(
        vector=vector,
        dimension=512,
        model_version="facenet_v1",
        extraction_timestamp=datetime.now()
    )
    
    print(f"✅ Created embedding with dimension: {embedding.dimension}")
    print(f"   Model version: {embedding.model_version}")
    print(f"   Vector shape: {embedding.vector.shape}")
    print(f"   Timestamp: {embedding.extraction_timestamp}")
    
    # Try to create an invalid embedding
    try:
        invalid_embedding = FaceEmbedding(
            vector=np.random.rand(256),  # Wrong size
            dimension=512,
            model_version="facenet_v1",
            extraction_timestamp=datetime.now()
        )
    except ValueError as e:
        print(f"❌ Validation caught error: {e}")
    
    print()

def demo_face_region():
    """Demonstrate FaceRegion creation and validation."""
    print("=== FaceRegion Demo ===")
    
    # Create a valid face region
    region = FaceRegion(x=100, y=150, width=200, height=250, confidence=0.95)
    
    print(f"✅ Created face region at ({region.x}, {region.y})")
    print(f"   Size: {region.width}x{region.height}")
    print(f"   Confidence: {region.confidence}")
    
    # Try invalid confidence
    try:
        invalid_region = FaceRegion(x=10, y=20, width=100, height=120, confidence=1.5)
    except ValueError as e:
        print(f"❌ Validation caught error: {e}")
    
    print()

def demo_search_config():
    """Demonstrate SearchConfig creation and validation."""
    print("=== SearchConfig Demo ===")
    
    # Create default config
    default_config = SearchConfig()
    print(f"✅ Default config - top_k: {default_config.top_k}, threshold: {default_config.similarity_threshold}")
    
    # Create custom config
    custom_config = SearchConfig(
        top_k=5,
        similarity_threshold=0.8,
        enable_reranking=False,
        distance_metric="euclidean"
    )
    print(f"✅ Custom config - top_k: {custom_config.top_k}, metric: {custom_config.distance_metric}")
    
    # Try invalid metric
    try:
        invalid_config = SearchConfig(distance_metric="manhattan")
    except ValueError as e:
        print(f"❌ Validation caught error: {e}")
    
    print()

def demo_search_result():
    """Demonstrate SearchResult creation."""
    print("=== SearchResult Demo ===")
    
    # Create search results
    result1 = SearchResult(
        embedding_id="emb_001",
        similarity_score=0.92,
        metadata={"name": "John Doe", "department": "Engineering"}
    )
    
    result2 = SearchResult(
        embedding_id="emb_002",
        similarity_score=0.87,
        metadata={"name": "Jane Smith", "department": "Marketing"},
        rerank_score=0.91
    )
    
    print(f"✅ Result 1: {result1.metadata['name']} (similarity: {result1.similarity_score})")
    print(f"✅ Result 2: {result2.metadata['name']} (similarity: {result2.similarity_score}, rerank: {result2.rerank_score})")
    
    print()

def demo_reranking_features():
    """Demonstrate RerankingFeatures creation."""
    print("=== RerankingFeatures Demo ===")
    
    features = RerankingFeatures(
        face_quality_score=0.85,
        landmark_confidence=0.92,
        pose_angle=12.5,
        illumination_score=0.78
    )
    
    print(f"✅ Reranking features:")
    print(f"   Quality: {features.face_quality_score}")
    print(f"   Landmarks: {features.landmark_confidence}")
    print(f"   Pose angle: {features.pose_angle}°")
    print(f"   Illumination: {features.illumination_score}")
    
    print()

def demo_recognition_request():
    """Demonstrate RecognitionRequest creation."""
    print("=== RecognitionRequest Demo ===")
    
    # Create fake image data (normally would be from cv2.imread)
    fake_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    config = SearchConfig(top_k=3, similarity_threshold=0.75)
    
    request = RecognitionRequest(
        image_data=fake_image,
        search_config=config,
        extract_features=True
    )
    
    print(f"✅ Created recognition request:")
    print(f"   Image shape: {request.image_data.shape}")
    print(f"   Search top_k: {request.search_config.top_k}")
    print(f"   Extract features: {request.extract_features}")
    
    print()

def demo_recognition_response():
    """Demonstrate RecognitionResponse creation."""
    print("=== RecognitionResponse Demo ===")
    
    # Create sample response data
    face_region = FaceRegion(x=50, y=75, width=150, height=180, confidence=0.96)
    search_result = SearchResult(
        embedding_id="emb_123",
        similarity_score=0.89,
        metadata={"name": "Alice Johnson"}
    )
    
    response = RecognitionResponse(
        detected_faces=[face_region],
        search_results=[search_result],
        processing_time_ms=245.7,
        success=True
    )
    
    print(f"✅ Recognition response:")
    print(f"   Faces detected: {len(response.detected_faces)}")
    print(f"   Search results: {len(response.search_results)}")
    print(f"   Processing time: {response.processing_time_ms}ms")
    print(f"   Success: {response.success}")
    
    if response.search_results:
        best_match = response.search_results[0]
        print(f"   Best match: {best_match.metadata['name']} ({best_match.similarity_score:.2f})")
    
    print()

def demo_exceptions():
    """Demonstrate custom exceptions."""
    print("=== Exception Demo ===")
    
    try:
        raise InvalidImageError("Image file is corrupted or unreadable")
    except FaceRecognitionError as e:
        print(f"✅ Caught exception: {e.error_code}")
        print(f"   Message: {e.message}")
    
    print()

def main():
    """Run all demos."""
    print("🎯 Face Recognition Models Interactive Demo\n")
    
    demo_face_embedding()
    demo_face_region()
    demo_search_config()
    demo_search_result()
    demo_reranking_features()
    demo_recognition_request()
    demo_recognition_response()
    demo_exceptions()
    
    print("🎉 Demo completed! All models are working correctly.")

if __name__ == "__main__":
    main()