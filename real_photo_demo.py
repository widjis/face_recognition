"""Demo with real photos - showing current capabilities."""

import cv2
import numpy as np
import os
from face_recognition.models import RecognitionRequest, SearchConfig, FaceRegion
from face_recognition.exceptions import InvalidImageError

def load_and_analyze_photo(image_path):
    """Load a real photo and analyze what we can do with it."""
    print(f"📸 Loading photo: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return None
    
    try:
        # Load the image
        image = cv2.imread(image_path)
        
        if image is None:
            raise InvalidImageError(f"Could not load image from {image_path}")
        
        # Analyze the image
        height, width, channels = image.shape
        print(f"✅ Image loaded successfully!")
        print(f"   Dimensions: {width}x{height}")
        print(f"   Channels: {channels}")
        print(f"   Data type: {image.dtype}")
        print(f"   Size in memory: {image.nbytes / 1024:.1f} KB")
        
        # Create a recognition request with the real image
        config = SearchConfig(top_k=5, similarity_threshold=0.8)
        request = RecognitionRequest(
            image_data=image,
            search_config=config,
            extract_features=True
        )
        
        print(f"✅ Created RecognitionRequest with real image data")
        print(f"   Search config: top_k={request.search_config.top_k}")
        print(f"   Threshold: {request.search_config.similarity_threshold}")
        
        # Show image statistics
        print(f"\n📊 Image Statistics:")
        print(f"   Min pixel value: {image.min()}")
        print(f"   Max pixel value: {image.max()}")
        print(f"   Mean pixel value: {image.mean():.1f}")
        
        # You could manually specify where you think faces are
        print(f"\n💡 What you could do manually:")
        print(f"   - Specify face regions if you know where they are")
        print(f"   - Create mock embeddings for testing")
        print(f"   - Test the data structures with real image data")
        
        return image
        
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

def create_mock_face_data(image):
    """Create mock face detection data for demonstration."""
    if image is None:
        return
    
    height, width = image.shape[:2]
    
    print(f"\n🎭 Creating mock face data:")
    
    # Create some plausible face regions
    # (In reality, these would come from face detection)
    mock_faces = [
        FaceRegion(
            x=width//4, 
            y=height//4, 
            width=width//3, 
            height=height//3, 
            confidence=0.95
        ),
        FaceRegion(
            x=width//2, 
            y=height//3, 
            width=width//4, 
            height=width//4, 
            confidence=0.87
        )
    ]
    
    for i, face in enumerate(mock_faces):
        print(f"   Face {i+1}: ({face.x}, {face.y}) {face.width}x{face.height} (conf: {face.confidence})")
    
    return mock_faces

def main():
    """Main demo function."""
    print("🎯 Real Photo Demo - Current Capabilities")
    print("=" * 50)
    
    # Try to find some common image files
    possible_images = [
        "MTI230279.jpg",
        "test_image.jpg",
        "photo.jpg", 
        "image.png",
        "sample.jpg"
    ]
    
    # Check if user has any images
    found_image = None
    for img_path in possible_images:
        if os.path.exists(img_path):
            found_image = img_path
            break
    
    if found_image:
        image = load_and_analyze_photo(found_image)
        if image is not None:
            create_mock_face_data(image)
    else:
        print("📁 No test images found in current directory.")
        print("   To try with a real photo:")
        print("   1. Put an image file (jpg, png) in this directory")
        print("   2. Run this script again")
        print("   3. Or specify the path in the code")
        
        # Create a synthetic image for demo
        print("\n🎨 Creating synthetic image for demo...")
        synthetic_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        config = SearchConfig()
        request = RecognitionRequest(
            image_data=synthetic_image,
            search_config=config
        )
        
        print(f"✅ Created request with synthetic image: {request.image_data.shape}")
    
    print(f"\n🚀 Next Steps:")
    print(f"   - Implement Task 2: Face Detection Module")
    print(f"   - Then you'll be able to automatically detect faces in real photos!")
    print(f"   - After that: embedding extraction, similarity search, etc.")

if __name__ == "__main__":
    main()