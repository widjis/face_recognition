"""Face detection implementation using OpenCV."""

import cv2
import numpy as np
from typing import List, Optional
from ..models import FaceRegion
from ..exceptions import FaceDetectionError, InvalidImageError


class FaceDetector:
    """Face detector using OpenCV's Haar cascades and DNN models."""
    
    def __init__(self, method: str = "haar", min_face_size: tuple = (30, 30)):
        """
        Initialize the face detector.
        
        Args:
            method: Detection method - "haar", "dnn", or "both"
            min_face_size: Minimum face size (width, height) to detect
        """
        self.method = method
        self.min_face_size = min_face_size
        self._haar_cascade = None
        self._dnn_net = None
        
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize the detection models."""
        try:
            if self.method in ["haar", "both"]:
                # Load Haar cascade for face detection
                self._haar_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                
                if self._haar_cascade.empty():
                    raise FaceDetectionError("Failed to load Haar cascade classifier")
            
            if self.method in ["dnn", "both"]:
                # For now, we'll focus on Haar cascades
                # DNN implementation can be added later
                pass
                
        except Exception as e:
            raise FaceDetectionError(f"Failed to initialize face detector: {str(e)}")
    
    def detect_faces(self, image: np.ndarray) -> List[FaceRegion]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detected face regions
            
        Raises:
            InvalidImageError: If image is invalid
            FaceDetectionError: If detection fails
        """
        if image is None or image.size == 0:
            raise InvalidImageError("Image is empty or None")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise InvalidImageError("Image must be a 3-channel BGR image")
        
        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = []
            
            if self.method in ["haar", "both"]:
                faces.extend(self._detect_with_haar(gray))
            
            if self.method in ["dnn", "both"]:
                # DNN detection can be implemented later
                pass
            
            # Remove duplicates if using both methods
            if self.method == "both":
                faces = self._remove_duplicate_faces(faces)
            
            return faces
            
        except cv2.error as e:
            raise FaceDetectionError(f"OpenCV error during face detection: {str(e)}")
        except Exception as e:
            raise FaceDetectionError(f"Unexpected error during face detection: {str(e)}")
    
    def _detect_with_haar(self, gray_image: np.ndarray) -> List[FaceRegion]:
        """Detect faces using Haar cascades."""
        faces = self._haar_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        face_regions = []
        for (x, y, w, h) in faces:
            # Calculate confidence based on face size (larger faces = higher confidence)
            # This is a simple heuristic; real confidence would come from the detector
            image_area = gray_image.shape[0] * gray_image.shape[1]
            face_area = w * h
            confidence = min(0.95, 0.5 + (face_area / image_area) * 2)
            
            face_region = FaceRegion(
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                confidence=float(confidence)
            )
            face_regions.append(face_region)
        
        return face_regions
    
    def _remove_duplicate_faces(self, faces: List[FaceRegion]) -> List[FaceRegion]:
        """Remove duplicate face detections using IoU threshold."""
        if len(faces) <= 1:
            return faces
        
        # Sort by confidence (highest first)
        faces.sort(key=lambda f: f.confidence, reverse=True)
        
        filtered_faces = []
        for face in faces:
            is_duplicate = False
            for existing_face in filtered_faces:
                if self._calculate_iou(face, existing_face) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_faces.append(face)
        
        return filtered_faces
    
    def _calculate_iou(self, face1: FaceRegion, face2: FaceRegion) -> float:
        """Calculate Intersection over Union (IoU) between two face regions."""
        # Calculate intersection
        x1 = max(face1.x, face2.x)
        y1 = max(face1.y, face2.y)
        x2 = min(face1.x + face1.width, face2.x + face2.width)
        y2 = min(face1.y + face1.height, face2.y + face2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = face1.width * face1.height
        area2 = face2.width * face2.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def preprocess_face(self, image: np.ndarray, face_region: FaceRegion) -> np.ndarray:
        """
        Extract and preprocess a face region from an image.
        
        Args:
            image: Source image
            face_region: Face region to extract
            
        Returns:
            Preprocessed face image
            
        Raises:
            InvalidImageError: If image or face region is invalid
        """
        if image is None or image.size == 0:
            raise InvalidImageError("Image is empty or None")
        
        try:
            # Extract face region
            x, y, w, h = face_region.x, face_region.y, face_region.width, face_region.height
            
            # Ensure coordinates are within image bounds
            img_height, img_width = image.shape[:2]
            
            # Check if face region is completely outside image
            if x >= img_width or y >= img_height or x + w <= 0 or y + h <= 0:
                raise InvalidImageError("Face region is outside image bounds")
            
            # Clip coordinates to image bounds
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            
            if w <= 0 or h <= 0:
                raise InvalidImageError("Face region is outside image bounds")
            
            # Extract face
            face_image = image[y:y+h, x:x+w]
            
            # Resize to standard size (224x224 is common for face recognition models)
            face_image = cv2.resize(face_image, (224, 224))
            
            # Normalize pixel values to [0, 1]
            face_image = face_image.astype(np.float32) / 255.0
            
            return face_image
            
        except Exception as e:
            raise InvalidImageError(f"Failed to preprocess face: {str(e)}")
    
    def get_face_quality_score(self, face_image: np.ndarray) -> float:
        """
        Calculate a quality score for a face image.
        
        Args:
            face_image: Face image to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            # Convert to grayscale if needed
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 range (values above 100 are considered sharp)
            sharpness_score = min(1.0, laplacian_var / 100.0)
            
            # Calculate brightness (faces should be well-lit)
            mean_brightness = np.mean(gray)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
            
            # Combine scores
            quality_score = (sharpness_score * 0.7 + brightness_score * 0.3)
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception:
            return 0.5  # Default quality score if calculation fails