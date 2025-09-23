#!/usr/bin/env python3
"""
Detailed investigation script to understand why MTI230216 was identified 
instead of the expected MTI230279 for WIN_20250923_21_58_29_Pro.jpg
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from face_recognition.pipeline import FaceRecognitionPipeline
from face_recognition.config.manager import ConfigurationManager

def analyze_face_comparison():
    """Detailed analysis of face recognition results"""
    
    print("🔍 DETAILED FACE RECOGNITION INVESTIGATION")
    print("=" * 60)
    
    # Initialize pipeline
    config_manager = ConfigurationManager()
    config = config_manager.load_config(profile="development")
    
    # Update reranking weights for better analysis
    config.reranking.similarity_weight = 0.6
    config.reranking.quality_weight = 0.2
    config.reranking.pose_weight = 0.1
    config.reranking.illumination_weight = 0.1
    
    pipeline = FaceRecognitionPipeline(
        config_manager=config_manager,
        db_path="test_pipeline_db"
    )
    
    db_info = pipeline.get_database_info()
    print(f"📊 Database loaded: {db_info['total_faces']} faces")
    print(f"🎯 Reranking enabled: {pipeline.reranker is not None}")
    print()
    
    # Test image
    test_image = "WIN_20250923_21_58_29_Pro.jpg"
    expected_match = "MTI230279"
    actual_match = "MTI230216"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"🖼️  Analyzing: {test_image}")
    print(f"🎯 Expected match: {expected_match}")
    print(f"🤖 Actual match: {actual_match}")
    print()
    
    # Perform recognition with detailed analysis
    try:
        # Load image
        import cv2
        image = cv2.imread(test_image)
        if image is None:
            print(f"❌ Could not load image: {test_image}")
            return
        
        # Create recognition request
        from face_recognition.models import RecognitionRequest, SearchConfig
        request = RecognitionRequest(
            image_data=image,
            search_config=SearchConfig(
                top_k=20,  # Get more results for analysis
                similarity_threshold=0.0,  # Get all results
                enable_reranking=True
            )
        )
        
        # Perform recognition
        response = pipeline.recognize_face(request)
        
        if not response.success:
            print(f"❌ Recognition failed: {response.error_message}")
            return
        
        results = response.search_results
        
        if not results:
            print("❌ No faces detected in the image")
            return
        
        print("🔍 FULL SEARCH RESULTS:")
        print("-" * 40)
        
        # Find positions of expected and actual matches
        expected_pos = None
        actual_pos = None
        
        for i, result in enumerate(results, 1):
            # Extract person ID from metadata or embedding_id
            person_id = result.metadata.get('person_id', result.embedding_id)
            
            is_expected = expected_match in person_id
            is_actual = actual_match in person_id
            
            if is_expected:
                expected_pos = i
            if is_actual:
                actual_pos = i
            
            status = ""
            if is_expected:
                status += " 🎯 EXPECTED"
            if is_actual:
                status += " 🤖 ACTUAL"
            
            rerank_score = result.rerank_score if result.rerank_score is not None else 0.0
            print(f"{i:2d}. {person_id} | Sim: {result.similarity_score:.6f} | Rerank: {rerank_score:.6f}{status}")
        
        print("📈 COMPARISON SUMMARY:")
        print("-" * 40)
        
        if expected_pos and actual_pos:
            expected_result = results[expected_pos - 1]
            actual_result = results[actual_pos - 1]
            
            expected_rerank = expected_result.rerank_score if expected_result.rerank_score is not None else 0.0
            actual_rerank = actual_result.rerank_score if actual_result.rerank_score is not None else 0.0
            
            print(f"Expected ({expected_match}):")
            print(f"  Position: #{expected_pos}")
            print(f"  Similarity: {expected_result.similarity_score:.6f}")
            print(f"  Rerank Score: {expected_rerank:.6f}")
            print()
            
            print(f"Actual ({actual_match}):")
            print(f"  Position: #{actual_pos}")
            print(f"  Similarity: {actual_result.similarity_score:.6f}")
            print(f"  Rerank Score: {actual_rerank:.6f}")
            print()
            
            # Calculate differences
            sim_diff = actual_result.similarity_score - expected_result.similarity_score
            rerank_diff = actual_rerank - expected_rerank
            
            print("🔄 DIFFERENCES:")
            print(f"  Similarity difference: {sim_diff:+.6f}")
            print(f"  Rerank difference: {rerank_diff:+.6f}")
            
            if sim_diff > 0:
                print(f"  ✅ {actual_match} has higher base similarity")
            else:
                print(f"  ⚠️  {expected_match} has higher base similarity")
            
            if rerank_diff > 0:
                print(f"  ✅ {actual_match} has higher rerank score")
            else:
                print(f"  ⚠️  {expected_match} has higher rerank score")
        
        else:
            if not expected_pos:
                print(f"⚠️  {expected_match} not found in top results!")
            if not actual_pos:
                print(f"⚠️  {actual_match} not found in top results!")
        
        print()
        print("🧪 TESTING WITHOUT RERANKING:")
        print("-" * 40)
        
        # Test without reranking
        print("🧪 TESTING WITHOUT RERANKING:")
        print("-" * 40)
        
        request_no_rerank = RecognitionRequest(
            image_data=image,
            search_config=SearchConfig(
                top_k=20,
                similarity_threshold=0.0,
                enable_reranking=False  # Disable reranking
            )
        )
        
        response_no_rerank = pipeline.recognize_face(request_no_rerank)
        
        if response_no_rerank.success and response_no_rerank.search_results:
            results_no_rerank = response_no_rerank.search_results
            top_without_rerank = results_no_rerank[0]
            top_person_id = top_without_rerank.metadata.get('person_id', top_without_rerank.embedding_id)
            print(f"Top match without reranking: {top_person_id}")
            print(f"Similarity: {top_without_rerank.similarity_score:.6f}")
            
            # Find expected match without reranking
            for i, result in enumerate(results_no_rerank, 1):
                result_person_id = result.metadata.get('person_id', result.embedding_id)
                if expected_match in result_person_id:
                    print(f"{expected_match} position without reranking: #{i}")
                    print(f"Similarity: {result.similarity_score:.6f}")
                    break
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def test_direct_comparison():
    """Test direct 1-to-1 comparison"""
    print("\n" + "=" * 60)
    print("🎯 DIRECT 1-TO-1 COMPARISON TEST")
    print("=" * 60)
    
    config_manager = ConfigurationManager()
    config = config_manager.load_config(profile="development")
    pipeline = FaceRecognitionPipeline(
        config_manager=config_manager,
        db_path="test_pipeline_db"
    )
    
    test_image = "WIN_20250923_21_58_29_Pro.jpg"
    candidates = ["MTI230279.jpg", "MTI230216.jpg"]
    
    print(f"🖼️  Test image: {test_image}")
    print(f"🆚 Comparing against: {candidates}")
    print()
    
    # Extract embedding from test image
    try:
        import cv2
        from PIL import Image
        
        # Load test image
        test_image_cv = cv2.imread(test_image)
        if test_image_cv is None:
            print(f"❌ Could not load test image: {test_image}")
            return
            
        test_faces = pipeline.face_detector.detect_faces(test_image_cv)
        if not test_faces:
            print("❌ No face detected in test image")
            return
        
        # Preprocess and extract embedding from test image
        processed_test_face = pipeline.face_detector.preprocess_face(test_image_cv, test_faces[0])
        test_embedding = pipeline.embedding_extractor.extract_embedding(processed_test_face)
        
        print("✅ Test image embedding extracted")
        print()
        
        # Compare with each candidate
        for candidate in candidates:
            candidate_path = f"data/{candidate}"
            if not os.path.exists(candidate_path):
                print(f"⚠️  Candidate not found: {candidate_path}")
                continue
            
            try:
                candidate_image_cv = cv2.imread(candidate_path)
                if candidate_image_cv is None:
                    print(f"❌ Could not load candidate image: {candidate_path}")
                    continue
                    
                candidate_faces = pipeline.face_detector.detect_faces(candidate_image_cv)
                if not candidate_faces:
                    print(f"❌ No face detected in {candidate}")
                    continue
                
                # Preprocess and extract embedding from candidate image
                processed_candidate_face = pipeline.face_detector.preprocess_face(candidate_image_cv, candidate_faces[0])
                candidate_embedding = pipeline.embedding_extractor.extract_embedding(processed_candidate_face)
                    
                # Calculate cosine similarity
                similarity = np.dot(test_embedding.vector, candidate_embedding.vector) / (
                    np.linalg.norm(test_embedding.vector) * np.linalg.norm(candidate_embedding.vector)
                )
                
                print(f"📊 {candidate}:")
                print(f"   Direct similarity: {similarity:.6f}")
                print(f"   Face detection: ✅")
                print()
                
            except Exception as e:
                print(f"❌ Error processing {candidate}: {e}")
    
    except Exception as e:
        print(f"❌ Error in direct comparison: {e}")

if __name__ == "__main__":
    analyze_face_comparison()
    test_direct_comparison()
    
    print("\n" + "=" * 60)
    print("🏁 INVESTIGATION COMPLETE")
    print("=" * 60)