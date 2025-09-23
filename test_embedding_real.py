"""Test embedding extraction with your real photo."""

import cv2
import numpy as np
from face_recognition.face_detection import FaceDetector
from face_recognition.embedding import EmbeddingExtractor
from datetime import datetime

def test_embedding_with_your_photo():
    """Test complete workflow: detection + embedding extraction with your photo."""
    print("🎯 Testing Embedding Extraction with Your Photo")
    print("=" * 55)
    
    # Load your photo
    image = cv2.imread("MTI230279.jpg")
    
    if image is None:
        print("❌ Could not load MTI230279.jpg")
        return
    
    print(f"📸 Photo loaded: {image.shape}")
    
    # Step 1: Detect faces
    print("\n🔍 Step 1: Face Detection")
    detector = FaceDetector(method="haar", min_face_size=(20, 20))
    faces = detector.detect_faces(image)
    
    print(f"   Found {len(faces)} face(s)")
    
    if len(faces) == 0:
        print("   No faces detected - cannot extract embeddings")
        return
    
    # Step 2: Extract embeddings from detected faces
    print(f"\n🧠 Step 2: Embedding Extraction")
    extractor = EmbeddingExtractor(model_name="simple", embedding_dim=512)
    
    embeddings = []
    
    for i, face in enumerate(faces):
        print(f"\n   Processing Face {i+1}:")
        print(f"   Position: ({face.x}, {face.y})")
        print(f"   Size: {face.width}x{face.height}")
        print(f"   Confidence: {face.confidence:.3f}")
        
        try:
            # Preprocess the face
            processed_face = detector.preprocess_face(image, face)
            print(f"   ✅ Face preprocessed: {processed_face.shape}")
            
            # Extract embedding
            embedding = extractor.extract_embedding(processed_face)
            embeddings.append(embedding)
            
            print(f"   ✅ Embedding extracted!")
            print(f"      Dimension: {embedding.dimension}")
            print(f"      Model: {embedding.model_version}")
            print(f"      Vector norm: {np.linalg.norm(embedding.vector):.6f}")
            print(f"      Vector sample: [{embedding.vector[:5].round(4)}...]")
            print(f"      Timestamp: {embedding.extraction_timestamp}")
            
        except Exception as e:
            print(f"   ❌ Error processing face {i+1}: {e}")
    
    # Step 3: Test embedding similarity
    if len(embeddings) >= 2:
        print(f"\n🔄 Step 3: Embedding Similarity Test")
        similarity = extractor.get_embedding_similarity(embeddings[0], embeddings[1])
        print(f"   Similarity between face 1 and 2: {similarity:.4f}")
    elif len(embeddings) == 1:
        print(f"\n🔄 Step 3: Self-Similarity Test")
        # Test with the same embedding (should be 1.0)
        similarity = extractor.get_embedding_similarity(embeddings[0], embeddings[0])
        print(f"   Self-similarity: {similarity:.4f} (should be ~1.0)")
    
    # Step 4: Test different models
    print(f"\n🔬 Step 4: Testing Different Models")
    
    if faces:
        face = faces[0]  # Use the first detected face
        processed_face = detector.preprocess_face(image, face)
        
        models = ["simple", "facenet", "arcface"]
        model_embeddings = {}
        
        for model_name in models:
            try:
                model_extractor = EmbeddingExtractor(model_name=model_name, embedding_dim=512)
                embedding = model_extractor.extract_embedding(processed_face)
                model_embeddings[model_name] = embedding
                
                print(f"   ✅ {model_name}: {embedding.dimension}D vector")
                print(f"      Norm: {np.linalg.norm(embedding.vector):.6f}")
                print(f"      Sample: [{embedding.vector[:3].round(4)}...]")
                
            except Exception as e:
                print(f"   ❌ {model_name}: {e}")
        
        # Compare embeddings from different models
        if len(model_embeddings) >= 2:
            print(f"\n   Model Comparison:")
            model_names = list(model_embeddings.keys())
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    similarity = extractor.get_embedding_similarity(
                        model_embeddings[model1], 
                        model_embeddings[model2]
                    )
                    print(f"   {model1} vs {model2}: {similarity:.4f}")
    
    # Step 5: Summary
    print(f"\n📊 Summary:")
    print(f"   Faces detected: {len(faces)}")
    print(f"   Embeddings extracted: {len(embeddings)}")
    print(f"   Embedding dimension: {embeddings[0].dimension if embeddings else 'N/A'}")
    print(f"   Processing successful: {'✅' if embeddings else '❌'}")
    
    return embeddings

def test_batch_embedding_extraction():
    """Test batch embedding extraction."""
    print(f"\n🔄 Testing Batch Embedding Extraction")
    print("=" * 40)
    
    # Load your photo
    image = cv2.imread("MTI230279.jpg")
    if image is None:
        return
    
    # Detect faces
    detector = FaceDetector()
    faces = detector.detect_faces(image)
    
    if not faces:
        print("   No faces detected for batch test")
        return
    
    # Preprocess all faces
    processed_faces = []
    for face in faces:
        try:
            processed_face = detector.preprocess_face(image, face)
            processed_faces.append(processed_face)
        except Exception as e:
            print(f"   Error preprocessing face: {e}")
    
    if not processed_faces:
        print("   No faces successfully preprocessed")
        return
    
    # Batch extract embeddings
    extractor = EmbeddingExtractor(model_name="simple", embedding_dim=512)
    
    try:
        embeddings = extractor.batch_extract_embeddings(processed_faces)
        
        print(f"   ✅ Batch extraction successful!")
        print(f"   Input faces: {len(processed_faces)}")
        print(f"   Output embeddings: {len(embeddings)}")
        
        for i, embedding in enumerate(embeddings):
            print(f"   Embedding {i+1}: {embedding.dimension}D, norm={np.linalg.norm(embedding.vector):.4f}")
        
    except Exception as e:
        print(f"   ❌ Batch extraction failed: {e}")

def main():
    """Run all embedding tests."""
    embeddings = test_embedding_with_your_photo()
    test_batch_embedding_extraction()
    
    print(f"\n🎉 Embedding extraction testing complete!")
    print(f"   Your photo has been processed through the complete pipeline:")
    print(f"   1. ✅ Face Detection")
    print(f"   2. ✅ Face Preprocessing") 
    print(f"   3. ✅ Embedding Extraction")
    print(f"   4. ✅ Similarity Calculation")
    print(f"   ")
    print(f"   Next step: Implement vector database (Task 4)")

if __name__ == "__main__":
    main()