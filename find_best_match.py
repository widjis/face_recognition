#!/usr/bin/env python3
"""Find the best match for a given photo using the face recognition system."""

import os
import sys
import cv2
from pathlib import Path

# Add the face_recognition package to the path
sys.path.append(str(Path(__file__).parent))

from face_recognition.config.manager import ConfigurationManager
from face_recognition.pipeline import FaceRecognitionPipeline
from face_recognition.models import RecognitionRequest, SearchConfig


def find_best_match(image_path: str, top_k: int = 5, similarity_threshold: float = 0.5):
    """Find the best matches for a given image."""
    
    print(f"🔍 Finding best match for: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Initialize pipeline
    print("🚀 Initializing face recognition pipeline...")
    config_manager = ConfigurationManager()
    config_manager.load_config(profile="development")
    
    pipeline = FaceRecognitionPipeline(
        config_manager=config_manager,
        db_path="test_pipeline_db"
    )
    
    # Get database info
    db_info = pipeline.get_database_info()
    print(f"📊 Database loaded: {db_info['total_faces']} faces available")
    
    # Load the image
    print(f"📷 Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"   Image dimensions: {image.shape}")
    
    # Create recognition request
    request = RecognitionRequest(
        image_data=image,
        search_config=SearchConfig(
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            enable_reranking=True
        )
    )
    
    # Perform recognition
    print(f"🔍 Searching for matches (threshold: {similarity_threshold})...")
    response = pipeline.recognize_face(request)
    
    if not response.success:
        print(f"❌ Recognition failed: {response.error_message}")
        return
    
    print(f"✅ Recognition completed successfully!")
    print(f"   🔍 Detected faces: {len(response.detected_faces)}")
    print(f"   📊 Search results: {len(response.search_results)}")
    
    if not response.search_results:
        print(f"❌ No matches found above similarity threshold {similarity_threshold}")
        print(f"💡 Try lowering the threshold (e.g., 0.3) to see more potential matches")
        return
    
    # Display results
    print(f"\n🎯 TOP {len(response.search_results)} MATCHES:")
    print("=" * 60)
    
    for i, result in enumerate(response.search_results):
        metadata = pipeline.metadata_store.get(result.embedding_id, {})
        person_id = metadata.get('person_id', 'Unknown')
        filename = metadata.get('filename', 'Unknown')
        
        print(f"#{i+1}: {person_id}")
        print(f"     📁 Source file: {filename}")
        print(f"     📊 Similarity: {result.similarity_score:.3f}")
        print(f"     🆔 Embedding ID: {result.embedding_id}")
        
        if result.rerank_score is not None:
            print(f"     🔄 Rerank score: {result.rerank_score:.3f}")
        
        # Confidence level
        if result.similarity_score >= 0.9:
            confidence = "🟢 VERY HIGH"
        elif result.similarity_score >= 0.8:
            confidence = "🟡 HIGH"
        elif result.similarity_score >= 0.7:
            confidence = "🟠 MEDIUM"
        else:
            confidence = "🔴 LOW"
        
        print(f"     🎯 Confidence: {confidence}")
        print("-" * 40)
    
    # Best match summary
    best_match = response.search_results[0]
    best_metadata = pipeline.metadata_store.get(best_match.embedding_id, {})
    best_person_id = best_metadata.get('person_id', 'Unknown')
    
    print(f"\n🏆 BEST MATCH: {best_person_id}")
    print(f"   📊 Similarity Score: {best_match.similarity_score:.3f}")
    print(f"   📁 Original File: {best_metadata.get('filename', 'Unknown')}")
    
    if best_match.similarity_score >= 0.8:
        print(f"   ✅ High confidence match!")
    elif best_match.similarity_score >= 0.6:
        print(f"   ⚠️  Medium confidence - manual verification recommended")
    else:
        print(f"   ❌ Low confidence - may not be a reliable match")


def main():
    """Main function to run the best match finder."""
    
    # Default image path
    image_path = "WIN_20250923_21_58_29_Pro.jpg"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    # Test with different thresholds if no matches found
    thresholds = [0.7, 0.5, 0.3, 0.1]
    
    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"🔍 SEARCHING WITH THRESHOLD: {threshold}")
        print(f"{'='*60}")
        
        find_best_match(image_path, top_k=5, similarity_threshold=threshold)
        
        # If we found matches, stop here
        config_manager = ConfigurationManager()
        config_manager.load_config(profile="development")
        pipeline = FaceRecognitionPipeline(config_manager=config_manager, db_path="test_pipeline_db")
        
        image = cv2.imread(image_path)
        if image is not None:
            request = RecognitionRequest(
                image_data=image,
                search_config=SearchConfig(top_k=5, similarity_threshold=threshold)
            )
            response = pipeline.recognize_face(request)
            
            if response.success and response.search_results:
                break
        
        print(f"\n💡 No matches found with threshold {threshold}, trying lower threshold...")


if __name__ == "__main__":
    main()