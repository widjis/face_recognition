"""Tests for face detection module."""

import pytest
import numpy as np
import cv2
from face_recognition.face_detection import FaceDetector
from face_recognition.models import FaceRegion
from face_recognition.exceptions import FaceDetectionError, InvalidImageError


class TestFaceDetector:
    """Test cases for FaceDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = FaceDetector(method="haar")
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = FaceDetector(method="haar", min_face_size=(50, 50))
        assert detector.method == "haar"
        assert detector.min_face_size == (50, 50)
        assert detector._haar_cascade is not None
    
    def test_invalid_method_initialization(self):
        """Test initialization with invalid method."""
        # This should still work but only use haar
        detector = FaceDetector(method="invalid")
        assert detector.method == "invalid"
    
    def test_detect_faces_with_valid_image(self):
        """Test face detection with a valid synthetic image."""
        # Create a synthetic image with some structure
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        # Add some face-like patterns (simple rectangles)
        cv2.rectangle(image, (50, 50), (150, 150), (200, 200, 200), -1)
        cv2.rectangle(image, (60, 60), (140, 140), (150, 150, 150), -1)
        
        # This might or might not detect faces, but should not crash
        faces = self.detector.detect_faces(image)
        assert isinstance(faces, list)
        
        # Each detected face should be a valid FaceRegion
        for face in faces:
            assert isinstance(face, FaceRegion)
            assert face.x >= 0
            assert face.y >= 0
            assert face.width > 0
            assert face.height > 0
            assert 0.0 <= face.confidence <= 1.0
    
    def test_detect_faces_with_empty_image(self):
        """Test face detection with empty image."""
        empty_image = np.array([])
        
        with pytest.raises(InvalidImageError, match="Image is empty or None"):
            self.detector.detect_faces(empty_image)
    
    def test_detect_faces_with_none_image(self):
        """Test face detection with None image."""
        with pytest.raises(InvalidImageError, match="Image is empty or None"):
            self.detector.detect_faces(None)
    
    def test_detect_faces_with_wrong_channels(self):
        """Test face detection with wrong number of channels."""
        # Grayscale image (2D)
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        with pytest.raises(InvalidImageError, match="Image must be a 3-channel BGR image"):
            self.detector.detect_faces(gray_image)
        
        # 4-channel image
        rgba_image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        
        with pytest.raises(InvalidImageError, match="Image must be a 3-channel BGR image"):
            self.detector.detect_faces(rgba_image)
    
    def test_preprocess_face_valid(self):
        """Test face preprocessing with valid inputs."""
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        face_region = FaceRegion(x=50, y=50, width=100, height=100, confidence=0.9)
        
        processed_face = self.detector.preprocess_face(image, face_region)
        
        # Should be resized to 224x224 and normalized
        assert processed_face.shape == (224, 224, 3)
        assert processed_face.dtype == np.float32
        assert 0.0 <= processed_face.min() <= processed_face.max() <= 1.0
    
    def test_preprocess_face_out_of_bounds(self):
        """Test face preprocessing with out-of-bounds region."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Face region extends beyond image
        face_region = FaceRegion(x=80, y=80, width=50, height=50, confidence=0.9)
        
        # Should handle gracefully by clipping to image bounds
        processed_face = self.detector.preprocess_face(image, face_region)
        assert processed_face.shape == (224, 224, 3)
    
    def test_preprocess_face_invalid_region(self):
        """Test face preprocessing with completely invalid region."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Face region completely outside image
        face_region = FaceRegion(x=200, y=200, width=50, height=50, confidence=0.9)
        
        with pytest.raises(InvalidImageError, match="Face region is outside image bounds"):
            self.detector.preprocess_face(image, face_region)
    
    def test_preprocess_face_empty_image(self):
        """Test face preprocessing with empty image."""
        empty_image = np.array([])
        face_region = FaceRegion(x=10, y=10, width=50, height=50, confidence=0.9)
        
        with pytest.raises(InvalidImageError, match="Image is empty or None"):
            self.detector.preprocess_face(empty_image, face_region)
    
    def test_get_face_quality_score(self):
        """Test face quality score calculation."""
        # Create a test face image
        face_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        quality_score = self.detector.get_face_quality_score(face_image)
        
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0
    
    def test_get_face_quality_score_grayscale(self):
        """Test face quality score with grayscale image."""
        face_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        quality_score = self.detector.get_face_quality_score(face_image)
        
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0
    
    def test_calculate_iou(self):
        """Test IoU calculation between face regions."""
        face1 = FaceRegion(x=10, y=10, width=50, height=50, confidence=0.9)
        face2 = FaceRegion(x=30, y=30, width=50, height=50, confidence=0.8)
        
        iou = self.detector._calculate_iou(face1, face2)
        
        assert isinstance(iou, float)
        assert 0.0 <= iou <= 1.0
        
        # Test with identical regions
        iou_identical = self.detector._calculate_iou(face1, face1)
        assert iou_identical == 1.0
        
        # Test with non-overlapping regions
        face3 = FaceRegion(x=100, y=100, width=50, height=50, confidence=0.7)
        iou_no_overlap = self.detector._calculate_iou(face1, face3)
        assert iou_no_overlap == 0.0
    
    def test_remove_duplicate_faces(self):
        """Test duplicate face removal."""
        faces = [
            FaceRegion(x=10, y=10, width=50, height=50, confidence=0.9),
            FaceRegion(x=15, y=15, width=50, height=50, confidence=0.8),  # Overlapping
            FaceRegion(x=100, y=100, width=50, height=50, confidence=0.7)  # Separate
        ]
        
        filtered_faces = self.detector._remove_duplicate_faces(faces)
        
        # Should remove the overlapping face with lower confidence
        assert len(filtered_faces) == 2
        assert filtered_faces[0].confidence == 0.9  # Highest confidence kept
        assert filtered_faces[1].confidence == 0.7  # Non-overlapping kept
    
    def test_remove_duplicate_faces_empty_list(self):
        """Test duplicate removal with empty list."""
        faces = []
        filtered_faces = self.detector._remove_duplicate_faces(faces)
        assert filtered_faces == []
    
    def test_remove_duplicate_faces_single_face(self):
        """Test duplicate removal with single face."""
        faces = [FaceRegion(x=10, y=10, width=50, height=50, confidence=0.9)]
        filtered_faces = self.detector._remove_duplicate_faces(faces)
        assert len(filtered_faces) == 1
        assert filtered_faces[0] == faces[0]


class TestFaceDetectorIntegration:
    """Integration tests for face detection with real-world scenarios."""
    
    def test_detect_faces_with_synthetic_face_pattern(self):
        """Test detection with a more realistic synthetic face pattern."""
        # Create a larger image with face-like patterns
        image = np.ones((400, 400, 3), dtype=np.uint8) * 128  # Gray background
        
        # Add face-like oval patterns
        center1 = (150, 150)
        center2 = (250, 250)
        
        # Draw face-like shapes
        cv2.ellipse(image, center1, (40, 50), 0, 0, 360, (200, 180, 160), -1)
        cv2.ellipse(image, center2, (35, 45), 0, 0, 360, (190, 170, 150), -1)
        
        # Add some noise to make it more realistic
        noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        detector = FaceDetector(method="haar", min_face_size=(20, 20))
        faces = detector.detect_faces(image)
        
        # Verify the detection results
        assert isinstance(faces, list)
        for face in faces:
            assert isinstance(face, FaceRegion)
            assert face.width >= 20 and face.height >= 20
            assert 0.0 <= face.confidence <= 1.0
    
    def test_full_workflow_with_preprocessing(self):
        """Test complete workflow from detection to preprocessing."""
        # Create test image
        image = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
        
        # Add a simple rectangular pattern that might be detected
        cv2.rectangle(image, (100, 100), (200, 200), (150, 150, 150), -1)
        cv2.rectangle(image, (110, 110), (190, 190), (100, 100, 100), -1)
        
        detector = FaceDetector()
        
        # Detect faces
        faces = detector.detect_faces(image)
        
        # If faces are detected, preprocess them
        for face in faces:
            processed_face = detector.preprocess_face(image, face)
            quality_score = detector.get_face_quality_score(processed_face)
            
            assert processed_face.shape == (224, 224, 3)
            assert 0.0 <= quality_score <= 1.0