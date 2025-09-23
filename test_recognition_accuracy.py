#!/usr/bin/env python3
"""Test recognition accuracy and reranking effectiveness."""

import os
import sys
import cv2
from pathlib import Path

# Add the face_recognition package to the path
sys.path.append(str(Path(__file__).parent))

from face_recognition.config.manager import ConfigurationManager
from face_recognition.pipeline import FaceRecognitionPipeline
from face_recognition.models import RecognitionRequest, SearchConfig


def test_recognition_with_known_image():
    """Test recognition with a known image from the database."""
    
    # Initialize pipeline
    config_manager = ConfigurationManager()
    config_manager.load_config(profile="development")
    
    pipeline = FaceRecognitionPipeline(
        config_manager=config_manager,
        db_path="test_pipeline_db"
    )
    
    # Get database info
    db_info = pipeline.get_database_info()
    print(f"📊 Database Info: {db_info['total_faces']} faces loaded")
    
    # Test with MTI230279.jpg (the image you mentioned)
    test_image_path = "data/MTI230279.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    # Load the test image
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"❌ Could not load image: {test_image_path}")
        return
    
    print(f"\n🔍 Testing recognition with: {test_image_path}")
    print(f"   Image shape: {image.shape}")
    
    # Test with different similarity thresholds
    thresholds = [0.3, 0.5, 0.7, 0.8]
    
    for threshold in thresholds:
        print(f"\n📏 Testing with similarity threshold: {threshold}")
        
        # Create recognition request
        request = RecognitionRequest(
            image_data=image,
            search_config=SearchConfig(
                top_k=5,
                similarity_threshold=threshold
            )
        )
        
        # Perform recognition
        response = pipeline.recognize_face(request)
        
        if not response.success:
            print(f"   ❌ Recognition failed: {response.error_message}")
            continue
        
        print(f"   🔍 Detected faces: {len(response.detected_faces)}")
        print(f"   📊 Search results: {len(response.search_results)}")
        
        # Show top results
        for i, result in enumerate(response.search_results[:3]):
            metadata = pipeline.metadata_store.get(result.embedding_id, {})
            person_id = metadata.get('person_id', 'Unknown')
            filename = metadata.get('filename', 'Unknown')
            
            print(f"   #{i+1}: {person_id} (from {filename})")
            print(f"        Similarity: {result.similarity_score:.3f}")
            print(f"        Embedding ID: {result.embedding_id}")
            
            # Check if this is the correct match
            if person_id == "MTI230279":
                print(f"        ✅ CORRECT MATCH!")
            else:
                print(f"        ❌ Incorrect match (expected MTI230279)")
    
    # Test reranking effectiveness
    print(f"\n🔄 Testing reranking effectiveness...")
    
    # Temporarily disable reranking
    original_reranker = pipeline.reranker
    pipeline.reranker = None
    
    request = RecognitionRequest(
        image_data=image,
        search_config=SearchConfig(top_k=5, similarity_threshold=0.5)
    )
    
    response_no_rerank = pipeline.recognize_face(request)
    print(f"   📊 Results WITHOUT reranking:")
    for i, result in enumerate(response_no_rerank.search_results[:3]):
        metadata = pipeline.metadata_store.get(result.embedding_id, {})
        person_id = metadata.get('person_id', 'Unknown')
        print(f"   #{i+1}: {person_id} - Similarity: {result.similarity_score:.3f}")
    
    # Re-enable reranking
    pipeline.reranker = original_reranker
    
    response_with_rerank = pipeline.recognize_face(request)
    print(f"   📊 Results WITH reranking:")
    for i, result in enumerate(response_with_rerank.search_results[:3]):
        metadata = pipeline.metadata_store.get(result.embedding_id, {})
        person_id = metadata.get('person_id', 'Unknown')
        print(f"   #{i+1}: {person_id} - Similarity: {result.similarity_score:.3f}")
    
    # Compare results
    if (response_no_rerank.search_results and response_with_rerank.search_results):
        no_rerank_top = response_no_rerank.search_results[0]
        with_rerank_top = response_with_rerank.search_results[0]
        
        no_rerank_metadata = pipeline.metadata_store.get(no_rerank_top.embedding_id, {})
        with_rerank_metadata = pipeline.metadata_store.get(with_rerank_top.embedding_id, {})
        
        print(f"\n📈 Reranking Impact:")
        print(f"   Without reranking: {no_rerank_metadata.get('person_id', 'Unknown')} ({no_rerank_top.similarity_score:.3f})")
        print(f"   With reranking: {with_rerank_metadata.get('person_id', 'Unknown')} ({with_rerank_top.similarity_score:.3f})")
        
        if no_rerank_metadata.get('person_id') != with_rerank_metadata.get('person_id'):
            print(f"   🔄 Reranking changed the top result!")
        else:
            print(f"   ➡️  Reranking kept the same top result")


if __name__ == "__main__":
    test_recognition_with_known_image()