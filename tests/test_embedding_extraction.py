"""Tests for embedding extraction module."""

import pytest
import numpy as np
from datetime import datetime
from face_recognition.embedding import EmbeddingExtractor
from face_recognition.models import FaceEmbedding
from face_recognition.exceptions import EmbeddingExtractionError, InvalidImageError


class TestEmbeddingExtractor:
    """Test cases for EmbeddingExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = EmbeddingExtractor(model_name="simple", embedding_dim=512)
    
    def test_extractor_initialization(self):
        """Test extractor initialization."""
        extractor = EmbeddingExtractor(model_name="simple", embedding_dim=256)
        assert extractor.model_name == "simple"
        assert extractor.embedding_dim == 256
        assert extractor.model_version == "simple_v1.0"
    
    def test_invalid_model_initialization(self):
        """Test initialization with invalid model."""
        with pytest.raises(EmbeddingExtractionError, match="Unsupported model"):
            EmbeddingExtractor(model_name="invalid_model")
    
    def test_extract_embedding_valid_image(self):
        """Test embedding extraction with valid face image."""
        # Create a valid preprocessed face image
        face_image = np.random.rand(224, 224, 3).astype(np.float32)
        
        embedding = self.extractor.extract_embedding(face_image)
        
        assert isinstance(embedding, FaceEmbedding)
        assert embedding.dimension == 512
        assert len(embedding.vector) == 512
        assert embedding.model_version == "simple_v1.0"
        assert isinstance(embedding.extraction_timestamp, datetime)
        
        # Check that embedding is normalized (unit length)
        norm = np.linalg.norm(embedding.vector)
        assert abs(norm - 1.0) < 1e-6
    
    def test_extract_embedding_empty_image(self):
        """Test embedding extraction with empty image."""
        empty_image = np.array([])
        
        with pytest.raises(InvalidImageError, match="Face image is empty or None"):
            self.extractor.extract_embedding(empty_image)
    
    def test_extract_embedding_none_image(self):
        """Test embedding extraction with None image."""
        with pytest.raises(InvalidImageError, match="Face image is empty or None"):
            self.extractor.extract_embedding(None)
    
    def test_extract_embedding_wrong_dimensions(self):
        """Test embedding extraction with wrong image dimensions."""
        # Wrong size image
        wrong_size_image = np.random.rand(100, 100, 3).astype(np.float32)
        
        with pytest.raises(InvalidImageError, match="Face image must be 224x224x3"):
            self.extractor.extract_embedding(wrong_size_image)
        
        # Wrong number of channels
        wrong_channels_image = np.random.rand(224, 224).astype(np.float32)
        
        with pytest.raises(InvalidImageError, match="Face image must be 224x224x3"):
            self.extractor.extract_embedding(wrong_channels_image)
    
    def test_extract_embedding_wrong_pixel_range(self):
        """Test embedding extraction with wrong pixel value range."""
        # Pixels outside [0, 1] range
        invalid_image = np.random.rand(224, 224, 3).astype(np.float32) * 255  # [0, 255] range
        
        with pytest.raises(InvalidImageError, match="Face image pixels must be in range"):
            self.extractor.extract_embedding(invalid_image)
    
    def test_extract_embedding_different_models(self):
        """Test embedding extraction with different model types."""
        face_image = np.random.rand(224, 224, 3).astype(np.float32)
        
        models = ["simple", "facenet", "arcface"]
        
        for model_name in models:
            extractor = EmbeddingExtractor(model_name=model_name, embedding_dim=512)
            embedding = extractor.extract_embedding(face_image)
            
            assert embedding.dimension == 512
            assert embedding.model_version == f"{model_name}_v1.0"
            assert len(embedding.vector) == 512
    
    def test_extract_embedding_consistency(self):
        """Test that same image produces same embedding."""
        face_image = np.random.rand(224, 224, 3).astype(np.float32)
        
        embedding1 = self.extractor.extract_embedding(face_image)
        embedding2 = self.extractor.extract_embedding(face_image)
        
        # Should produce identical embeddings for the same input
        np.testing.assert_array_almost_equal(embedding1.vector, embedding2.vector, decimal=6)
    
    def test_extract_embedding_different_images(self):
        """Test that different images produce different embeddings."""
        face_image1 = np.random.rand(224, 224, 3).astype(np.float32)
        face_image2 = np.random.rand(224, 224, 3).astype(np.float32)
        
        embedding1 = self.extractor.extract_embedding(face_image1)
        embedding2 = self.extractor.extract_embedding(face_image2)
        
        # Should produce different embeddings for different inputs
        similarity = np.dot(embedding1.vector, embedding2.vector)
        assert similarity < 0.99  # Should not be too similar
    
    def test_batch_extract_embeddings_valid(self):
        """Test batch embedding extraction with valid images."""
        face_images = [
            np.random.rand(224, 224, 3).astype(np.float32),
            np.random.rand(224, 224, 3).astype(np.float32),
            np.random.rand(224, 224, 3).astype(np.float32)
        ]
        
        embeddings = self.extractor.batch_extract_embeddings(face_images)
        
        assert len(embeddings) == 3
        for embedding in embeddings:
            assert isinstance(embedding, FaceEmbedding)
            assert embedding.dimension == 512
            assert len(embedding.vector) == 512
    
    def test_batch_extract_embeddings_empty_list(self):
        """Test batch embedding extraction with empty list."""
        embeddings = self.extractor.batch_extract_embeddings([])
        assert embeddings == []
    
    def test_batch_extract_embeddings_with_failures(self):
        """Test batch embedding extraction with some invalid images."""
        face_images = [
            np.random.rand(224, 224, 3).astype(np.float32),  # Valid
            np.random.rand(100, 100, 3).astype(np.float32),  # Invalid size
            np.random.rand(224, 224, 3).astype(np.float32)   # Valid
        ]
        
        # Should extract embeddings from valid images and skip invalid ones
        embeddings = self.extractor.batch_extract_embeddings(face_images)
        
        assert len(embeddings) == 2  # Only valid images processed
        for embedding in embeddings:
            assert isinstance(embedding, FaceEmbedding)
    
    def test_batch_extract_embeddings_all_failures(self):
        """Test batch embedding extraction when all images are invalid."""
        face_images = [
            np.random.rand(100, 100, 3).astype(np.float32),  # Invalid size
            np.random.rand(50, 50, 3).astype(np.float32),    # Invalid size
        ]
        
        with pytest.raises(EmbeddingExtractionError, match="Failed to extract embeddings from all images"):
            self.extractor.batch_extract_embeddings(face_images)
    
    def test_get_embedding_similarity(self):
        """Test similarity calculation between embeddings."""
        face_image1 = np.random.rand(224, 224, 3).astype(np.float32)
        face_image2 = np.random.rand(224, 224, 3).astype(np.float32)
        
        embedding1 = self.extractor.extract_embedding(face_image1)
        embedding2 = self.extractor.extract_embedding(face_image2)
        
        similarity = self.extractor.get_embedding_similarity(embedding1, embedding2)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_get_embedding_similarity_identical(self):
        """Test similarity calculation for identical embeddings."""
        face_image = np.random.rand(224, 224, 3).astype(np.float32)
        
        embedding1 = self.extractor.extract_embedding(face_image)
        embedding2 = self.extractor.extract_embedding(face_image)
        
        similarity = self.extractor.get_embedding_similarity(embedding1, embedding2)
        
        # Identical embeddings should have similarity close to 1.0
        assert similarity > 0.99
    
    def test_get_embedding_similarity_different_dimensions(self):
        """Test similarity calculation with different embedding dimensions."""
        extractor1 = EmbeddingExtractor(model_name="simple", embedding_dim=256)
        extractor2 = EmbeddingExtractor(model_name="simple", embedding_dim=512)
        
        face_image = np.random.rand(224, 224, 3).astype(np.float32)
        
        embedding1 = extractor1.extract_embedding(face_image)
        embedding2 = extractor2.extract_embedding(face_image)
        
        with pytest.raises(ValueError, match="Embeddings must have the same dimension"):
            extractor1.get_embedding_similarity(embedding1, embedding2)
    
    def test_normalize_embedding(self):
        """Test embedding normalization."""
        # Create a non-normalized vector
        vector = np.array([3.0, 4.0, 0.0], dtype=np.float32)  # Length = 5
        
        normalized = self.extractor._normalize_embedding(vector)
        
        # Should be unit length
        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 1e-6
        
        # Should maintain direction
        expected = np.array([0.6, 0.8, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(normalized, expected, decimal=6)
    
    def test_normalize_embedding_zero_vector(self):
        """Test normalization of zero vector."""
        zero_vector = np.zeros(512, dtype=np.float32)
        
        normalized = self.extractor._normalize_embedding(zero_vector)
        
        # Should remain zero vector
        np.testing.assert_array_equal(normalized, zero_vector)


class TestEmbeddingExtractionIntegration:
    """Integration tests for embedding extraction."""
    
    def test_different_embedding_dimensions(self):
        """Test extraction with different embedding dimensions."""
        face_image = np.random.rand(224, 224, 3).astype(np.float32)
        
        dimensions = [128, 256, 512, 1024]
        
        for dim in dimensions:
            extractor = EmbeddingExtractor(model_name="simple", embedding_dim=dim)
            embedding = extractor.extract_embedding(face_image)
            
            assert embedding.dimension == dim
            assert len(embedding.vector) == dim
    
    def test_feature_extraction_methods(self):
        """Test individual feature extraction methods."""
        extractor = EmbeddingExtractor(model_name="simple", embedding_dim=512)
        face_image = np.random.rand(224, 224, 3).astype(np.float32)
        
        # Test LBP features
        gray_image = face_image[:, :, 0]  # Use first channel as grayscale
        lbp_features = extractor._extract_lbp_features(gray_image)
        
        assert isinstance(lbp_features, list)
        assert len(lbp_features) == 32  # 32 bins in histogram
        assert all(isinstance(f, float) for f in lbp_features)
        
        # Test HOG features
        hog_features = extractor._extract_hog_features(gray_image)
        
        assert isinstance(hog_features, list)
        assert len(hog_features) > 0
        assert all(isinstance(f, float) for f in hog_features)
        
        # Test statistical features
        stat_features = extractor._extract_statistical_features(gray_image)
        
        assert isinstance(stat_features, list)
        assert len(stat_features) > 0
        assert all(isinstance(f, float) for f in stat_features)