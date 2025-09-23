"""Test face detection with your real photo."""

import cv2
import numpy as np
from face_recognition.face_detection import FaceDetector
from face_recognition.models import RecognitionRequest, SearchConfig, RecognitionResponse
from datetime import datetime

def test_with_your_photo():
    """Test face detection with MTI230279.jpg."""
    print("🎯 Testing Face Detection with Your Photo")
    print("=" * 50)
    
    # Load your photo
    image = cv2.imread("MTI230279.jpg")
    
    if image is None:
        print("❌ Could not load MTI230279.jpg")
        return
    
    print(f"📸 Photo loaded: {image.shape}")
    
    # Initialize face detector
    detector = FaceDetector(method="haar", min_face_size=(20, 20))
    print("✅ Face detector initialized")
    
    try:
        # Detect faces
        print("\n🔍 Detecting faces...")
        faces = detector.detect_faces(image)
        
        print(f"✅ Detection complete!")
        print(f"   Found {len(faces)} face(s)")
        
        if len(faces) == 0:
            print("   No faces detected in this image")
            print("   This could be because:")
            print("   - The image doesn't contain clear frontal faces")
            print("   - The faces are too small or at an angle")
            print("   - The lighting conditions are challenging")
            print("   - The Haar cascade model has limitations")
        
        # Process each detected face
        for i, face in enumerate(faces):
            print(f"\n👤 Face {i+1}:")
            print(f"   Position: ({face.x}, {face.y})")
            print(f"   Size: {face.width}x{face.height}")
            print(f"   Confidence: {face.confidence:.3f}")
            
            # Preprocess the face
            try:
                processed_face = detector.preprocess_face(image, face)
                quality_score = detector.get_face_quality_score(processed_face)
                
                print(f"   Processed size: {processed_face.shape}")
                print(f"   Quality score: {quality_score:.3f}")
                
                # Show some statistics about the processed face
                print(f"   Pixel range: {processed_face.min():.3f} - {processed_face.max():.3f}")
                print(f"   Mean brightness: {processed_face.mean():.3f}")
                
            except Exception as e:
                print(f"   ❌ Error preprocessing face: {e}")
        
        # Create a complete recognition workflow
        print(f"\n🔄 Complete Recognition Workflow:")
        
        config = SearchConfig(top_k=5, similarity_threshold=0.8)
        request = RecognitionRequest(
            image_data=image,
            search_config=config,
            extract_features=True
        )
        
        # Simulate processing time
        start_time = datetime.now()
        
        # In a real system, this would include:
        # 1. Face detection (done)
        # 2. Embedding extraction (not implemented yet)
        # 3. Similarity search (not implemented yet)
        # 4. Reranking (not implemented yet)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        response = RecognitionResponse(
            detected_faces=faces,
            search_results=[],  # Empty for now - no database yet
            processing_time_ms=processing_time,
            success=True
        )
        
        print(f"✅ Recognition workflow completed:")
        print(f"   Faces detected: {len(response.detected_faces)}")
        print(f"   Processing time: {response.processing_time_ms:.1f}ms")
        print(f"   Success: {response.success}")
        
        return faces
        
    except Exception as e:
        print(f"❌ Error during face detection: {e}")
        return []

def visualize_detection_results(faces):
    """Create a visualization of the detection results."""
    if not faces:
        print("\n📊 No faces to visualize")
        return
    
    print(f"\n📊 Detection Results Summary:")
    print(f"   Total faces: {len(faces)}")
    
    # Calculate statistics
    if faces:
        avg_confidence = sum(f.confidence for f in faces) / len(faces)
        avg_size = sum(f.width * f.height for f in faces) / len(faces)
        
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Average face area: {avg_size:.0f} pixels")
        
        # Find the best face (highest confidence)
        best_face = max(faces, key=lambda f: f.confidence)
        print(f"   Best face: {best_face.width}x{best_face.height} at ({best_face.x}, {best_face.y})")
        print(f"   Best confidence: {best_face.confidence:.3f}")

def test_different_detection_settings():
    """Test different detection settings with your photo."""
    print(f"\n🔧 Testing Different Detection Settings:")
    print("=" * 40)
    
    image = cv2.imread("MTI230279.jpg")
    if image is None:
        return
    
    # Test different minimum face sizes
    min_sizes = [(10, 10), (20, 20), (30, 30), (50, 50)]
    
    for min_size in min_sizes:
        detector = FaceDetector(method="haar", min_face_size=min_size)
        faces = detector.detect_faces(image)
        print(f"   Min size {min_size}: {len(faces)} faces detected")

def main():
    """Run all tests."""
    faces = test_with_your_photo()
    visualize_detection_results(faces)
    test_different_detection_settings()
    
    print(f"\n🎉 Face detection testing complete!")
    print(f"   Your photo has been processed with the face detection system.")
    print(f"   Next step: Implement embedding extraction (Task 3)")

if __name__ == "__main__":
    main()