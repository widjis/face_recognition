"""Tests for core data models."""

import pytest
import numpy as np
from datetime import datetime
from face_recognition.models import (
    FaceEmbedding, FaceRegion, SearchResult, SearchConfig, 
    RerankingFeatures, RecognitionRequest, RecognitionResponse
)


class TestFaceEmbedding:
    """Test cases for FaceEmbedding model."""
    
    def test_valid_embedding_creation(self):
        """Test creating a valid face embedding."""
        vector = np.random.rand(512)
        embedding = FaceEmbedding(
            vector=vector,
            dimension=512,
            model_version="facenet_v1",
            extraction_timestamp=datetime.now()
        )
        assert len(embedding.vector) == 512
        assert embedding.dimension == 512
        assert embedding.model_version == "facenet_v1"
    
    def test_invalid_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError."""
        vector = np.random.rand(256)
        with pytest.raises(ValueError, match="Vector length 256 doesn't match dimension 512"):
            FaceEmbedding(
                vector=vector,
                dimension=512,
                model_version="facenet_v1",
                extraction_timestamp=datetime.now()
            )
    
    def test_invalid_negative_dimension(self):
        """Test that negative dimension raises ValueError."""
        vector = np.random.rand(512)
        with pytest.raises(ValueError, match="Dimension must be positive"):
            FaceEmbedding(
                vector=vector,
                dimension=-1,
                model_version="facenet_v1",
                extraction_timestamp=datetime.now()
            )


class TestFaceRegion:
    """Test cases for FaceRegion model."""
    
    def test_valid_face_region_creation(self):
        """Test creating a valid face region."""
        region = FaceRegion(x=10, y=20, width=100, height=120, confidence=0.95)
        assert region.x == 10
        assert region.y == 20
        assert region.width == 100
        assert region.height == 120
        assert region.confidence == 0.95
    
    def test_invalid_negative_dimensions(self):
        """Test that negative dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Face region width and height must be positive"):
            FaceRegion(x=10, y=20, width=-100, height=120, confidence=0.95)
    
    def test_invalid_confidence_range(self):
        """Test that confidence outside [0,1] raises ValueError."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            FaceRegion(x=10, y=20, width=100, height=120, confidence=1.5)


class TestSearchResult:
    """Test cases for SearchResult model."""
    
    def test_valid_search_result_creation(self):
        """Test creating a valid search result."""
        result = SearchResult(
            embedding_id="emb_123",
            similarity_score=0.85,
            metadata={"name": "John Doe"},
            rerank_score=0.90
        )
        assert result.embedding_id == "emb_123"
        assert result.similarity_score == 0.85
        assert result.rerank_score == 0.90
    
    def test_invalid_similarity_score_range(self):
        """Test that similarity score outside [0,1] raises ValueError."""
        with pytest.raises(ValueError, match="Similarity score must be between 0.0 and 1.0"):
            SearchResult(
                embedding_id="emb_123",
                similarity_score=1.5,
                metadata={"name": "John Doe"}
            )


class TestSearchConfig:
    """Test cases for SearchConfig model."""
    
    def test_valid_search_config_creation(self):
        """Test creating a valid search configuration."""
        config = SearchConfig(
            top_k=5,
            similarity_threshold=0.8,
            enable_reranking=False,
            distance_metric="euclidean"
        )
        assert config.top_k == 5
        assert config.similarity_threshold == 0.8
        assert config.enable_reranking is False
        assert config.distance_metric == "euclidean"
    
    def test_invalid_top_k(self):
        """Test that non-positive top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            SearchConfig(top_k=0)
    
    def test_invalid_distance_metric(self):
        """Test that unsupported distance metric raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported distance metric"):
            SearchConfig(distance_metric="manhattan")


class TestRerankingFeatures:
    """Test cases for RerankingFeatures model."""
    
    def test_valid_reranking_features_creation(self):
        """Test creating valid reranking features."""
        features = RerankingFeatures(
            face_quality_score=0.9,
            landmark_confidence=0.85,
            pose_angle=15.0,
            illumination_score=0.7
        )
        assert features.face_quality_score == 0.9
        assert features.landmark_confidence == 0.85
        assert features.pose_angle == 15.0
        assert features.illumination_score == 0.7
    
    def test_invalid_quality_score_range(self):
        """Test that quality score outside [0,1] raises ValueError."""
        with pytest.raises(ValueError, match="face_quality_score must be between 0.0 and 1.0"):
            RerankingFeatures(
                face_quality_score=1.5,
                landmark_confidence=0.85,
                pose_angle=15.0,
                illumination_score=0.7
            )


class TestRecognitionRequest:
    """Test cases for RecognitionRequest model."""
    
    def test_valid_recognition_request_creation(self):
        """Test creating a valid recognition request."""
        image_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        config = SearchConfig()
        request = RecognitionRequest(
            image_data=image_data,
            search_config=config,
            extract_features=True
        )
        assert request.image_data.shape == (224, 224, 3)
        assert request.extract_features is True
    
    def test_invalid_empty_image_data(self):
        """Test that empty image data raises ValueError."""
        empty_image = np.array([])
        config = SearchConfig()
        with pytest.raises(ValueError, match="Image data cannot be empty"):
            RecognitionRequest(
                image_data=empty_image,
                search_config=config
            )


class TestRecognitionResponse:
    """Test cases for RecognitionResponse model."""
    
    def test_valid_recognition_response_creation(self):
        """Test creating a valid recognition response."""
        face_region = FaceRegion(x=10, y=20, width=100, height=120, confidence=0.95)
        search_result = SearchResult(
            embedding_id="emb_123",
            similarity_score=0.85,
            metadata={"name": "John Doe"}
        )
        response = RecognitionResponse(
            detected_faces=[face_region],
            search_results=[search_result],
            processing_time_ms=150.5,
            success=True
        )
        assert len(response.detected_faces) == 1
        assert len(response.search_results) == 1
        assert response.processing_time_ms == 150.5
        assert response.success is True
    
    def test_invalid_negative_processing_time(self):
        """Test that negative processing time raises ValueError."""
        with pytest.raises(ValueError, match="Processing time must be non-negative"):
            RecognitionResponse(
                detected_faces=[],
                search_results=[],
                processing_time_ms=-10.0,
                success=True
            )
    
    def test_invalid_failed_without_error_message(self):
        """Test that failed response without error message raises ValueError."""
        with pytest.raises(ValueError, match="Error message must be provided when success is False"):
            RecognitionResponse(
                detected_faces=[],
                search_results=[],
                processing_time_ms=100.0,
                success=False
            )