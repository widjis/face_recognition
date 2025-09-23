"""Embedding extraction implementation using pre-trained models."""

import cv2
import numpy as np
from typing import List, Optional
from datetime import datetime
from ..models import FaceEmbedding
from ..exceptions import EmbeddingExtractionError, InvalidImageError


class EmbeddingExtractor:
    """Face embedding extractor using various pre-trained models."""
    
    def __init__(self, model_name: str = "simple", embedding_dim: int = 512):
        """
        Initialize the embedding extractor.
        
        Args:
            model_name: Name of the model to use ("simple", "facenet", "arcface")
            embedding_dim: Dimension of the output embedding vector
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.model_version = f"{model_name}_v1.0"
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        if self.model_name == "simple":
            # Simple feature extractor using traditional computer vision
            # This is a placeholder - in production you'd use FaceNet, ArcFace, etc.
            pass
        elif self.model_name == "facenet":
            # Would load FaceNet model here
            # For now, we'll simulate it
            pass
        elif self.model_name == "arcface":
            # Would load ArcFace model here
            # For now, we'll simulate it
            pass
        else:
            raise EmbeddingExtractionError(f"Unsupported model: {self.model_name}")
    
    def extract_embedding(self, face_image: np.ndarray) -> FaceEmbedding:
        """
        Extract embedding from a face image.
        
        Args:
            face_image: Preprocessed face image (224x224x3, normalized to [0,1])
            
        Returns:
            FaceEmbedding object containing the embedding vector
            
        Raises:
            InvalidImageError: If face image is invalid
            EmbeddingExtractionError: If extraction fails
        """
        if face_image is None or face_image.size == 0:
            raise InvalidImageError("Face image is empty or None")
        
        # Validate input dimensions
        if len(face_image.shape) != 3 or face_image.shape[:2] != (224, 224):
            raise InvalidImageError("Face image must be 224x224x3")
        
        # Validate pixel value range
        if face_image.min() < 0 or face_image.max() > 1:
            raise InvalidImageError("Face image pixels must be in range [0, 1]")
        
        try:
            if self.model_name == "simple":
                embedding_vector = self._extract_simple_features(face_image)
            elif self.model_name == "facenet":
                embedding_vector = self._extract_facenet_features(face_image)
            elif self.model_name == "arcface":
                embedding_vector = self._extract_arcface_features(face_image)
            else:
                raise EmbeddingExtractionError(f"Unknown model: {self.model_name}")
            
            # Normalize the embedding vector
            embedding_vector = self._normalize_embedding(embedding_vector)
            
            return FaceEmbedding(
                vector=embedding_vector,
                dimension=self.embedding_dim,
                model_version=self.model_version,
                extraction_timestamp=datetime.now()
            )
            
        except Exception as e:
            if isinstance(e, (InvalidImageError, EmbeddingExtractionError)):
                raise
            raise EmbeddingExtractionError(f"Failed to extract embedding: {str(e)}")
    
    def _extract_simple_features(self, face_image: np.ndarray) -> np.ndarray:
        """Extract simple handcrafted features from face image."""
        # Convert to grayscale for feature extraction
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_image
        
        features = []
        
        # 1. Histogram features (64 bins)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 1])
        hist = hist.flatten() / hist.sum()  # Normalize
        features.extend(hist)
        
        # 2. LBP (Local Binary Pattern) features
        lbp_features = self._extract_lbp_features(gray)
        features.extend(lbp_features)
        
        # 3. HOG (Histogram of Oriented Gradients) features
        hog_features = self._extract_hog_features(gray)
        features.extend(hog_features)
        
        # 4. Statistical features
        stat_features = self._extract_statistical_features(gray)
        features.extend(stat_features)
        
        # Convert to numpy array and pad/truncate to desired dimension
        features = np.array(features, dtype=np.float32)
        
        if len(features) > self.embedding_dim:
            features = features[:self.embedding_dim]
        elif len(features) < self.embedding_dim:
            # Pad with zeros
            padding = np.zeros(self.embedding_dim - len(features), dtype=np.float32)
            features = np.concatenate([features, padding])
        
        return features
    
    def _extract_lbp_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract Local Binary Pattern features."""
        # Simple LBP implementation
        rows, cols = gray_image.shape
        lbp_image = np.zeros_like(gray_image)
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = gray_image[i, j]
                binary_string = ""
                
                # Check 8 neighbors
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += "1" if neighbor >= center else "0"
                
                lbp_image[i, j] = int(binary_string, 2)
        
        # Create histogram of LBP values
        hist, _ = np.histogram(lbp_image.flatten(), bins=32, range=(0, 256))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-7)  # Normalize
        
        return hist.tolist()
    
    def _extract_hog_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract HOG features using OpenCV."""
        # Convert to uint8 format (0-255) as required by OpenCV HOG
        if gray_image.dtype == np.float32:
            gray_uint8 = (gray_image * 255).astype(np.uint8)
        else:
            gray_uint8 = gray_image.astype(np.uint8)
        
        # Resize to standard HOG size
        resized = cv2.resize(gray_uint8, (64, 128))
        
        # HOG parameters
        win_size = (64, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        features = hog.compute(resized)
        
        if features is not None:
            return features.flatten().tolist()
        else:
            return [0.0] * 100  # Fallback
    
    def _extract_statistical_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract statistical features from the image."""
        features = []
        
        # Convert to uint8 for OpenCV operations
        if gray_image.dtype == np.float32:
            gray_uint8 = (gray_image * 255).astype(np.uint8)
        else:
            gray_uint8 = gray_image.astype(np.uint8)
        
        # Basic statistics
        features.append(float(np.mean(gray_image)))
        features.append(float(np.std(gray_image)))
        features.append(float(np.min(gray_image)))
        features.append(float(np.max(gray_image)))
        features.append(float(np.median(gray_image)))
        
        # Moments (using uint8 version)
        moments = cv2.moments(gray_uint8)
        for key in ['m00', 'm10', 'm01', 'm20', 'm11', 'm02']:
            features.append(float(moments.get(key, 0)))
        
        # Texture features (using Laplacian on uint8)
        laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        features.append(float(np.var(laplacian)))
        
        return features
    
    def _extract_facenet_features(self, face_image: np.ndarray) -> np.ndarray:
        """Simulate FaceNet feature extraction."""
        # In a real implementation, this would use a pre-trained FaceNet model
        # For now, we'll create a more sophisticated simulation
        
        # Use the simple features as a base
        simple_features = self._extract_simple_features(face_image)
        
        # Add some "learned" transformations to simulate deep features
        # This is just for demonstration - real FaceNet would be much more sophisticated
        transformed_features = np.tanh(simple_features * 2.0)  # Non-linear transformation
        
        # Add some random but consistent variations based on image content
        np.random.seed(int(np.sum(face_image * 1000) % 2**32))  # Deterministic based on image
        noise = np.random.normal(0, 0.1, len(transformed_features))
        
        return transformed_features + noise
    
    def _extract_arcface_features(self, face_image: np.ndarray) -> np.ndarray:
        """Simulate ArcFace feature extraction."""
        # Similar to FaceNet simulation but with different characteristics
        simple_features = self._extract_simple_features(face_image)
        
        # ArcFace typically produces more discriminative features
        # Simulate this with different transformations
        transformed_features = np.sin(simple_features * np.pi)  # Different non-linearity
        
        # Add deterministic variations
        np.random.seed(int(np.sum(face_image * 2000) % 2**32))
        noise = np.random.normal(0, 0.05, len(transformed_features))
        
        return transformed_features + noise
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding vector to unit length."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        else:
            return embedding
    
    def batch_extract_embeddings(self, face_images: List[np.ndarray]) -> List[FaceEmbedding]:
        """
        Extract embeddings from multiple face images.
        
        Args:
            face_images: List of preprocessed face images
            
        Returns:
            List of FaceEmbedding objects
            
        Raises:
            EmbeddingExtractionError: If batch extraction fails
        """
        if not face_images:
            return []
        
        embeddings = []
        failed_indices = []
        
        for i, face_image in enumerate(face_images):
            try:
                embedding = self.extract_embedding(face_image)
                embeddings.append(embedding)
            except Exception as e:
                failed_indices.append(i)
                # Log the error but continue with other images
                print(f"Warning: Failed to extract embedding for image {i}: {e}")
        
        if len(failed_indices) == len(face_images):
            raise EmbeddingExtractionError("Failed to extract embeddings from all images")
        
        return embeddings
    
    def get_embedding_similarity(self, embedding1: FaceEmbedding, embedding2: FaceEmbedding) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if embedding1.dimension != embedding2.dimension:
            raise ValueError("Embeddings must have the same dimension")
        
        # Calculate cosine similarity
        dot_product = np.dot(embedding1.vector, embedding2.vector)
        norm1 = np.linalg.norm(embedding1.vector)
        norm2 = np.linalg.norm(embedding2.vector)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        
        # Convert to 0-1 range (cosine similarity is in [-1, 1])
        return (cosine_sim + 1.0) / 2.0