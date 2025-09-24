"""Core data models for the face recognition system."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from .exceptions import ConfigurationError


@dataclass
class FaceEmbedding:
    """Represents a facial embedding vector with metadata."""
    vector: np.ndarray
    dimension: int
    model_version: str
    extraction_timestamp: datetime
    
    def __post_init__(self):
        """Validate embedding data after initialization."""
        if self.dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {self.dimension}")
        if len(self.vector) != self.dimension:
            raise ValueError(f"Vector length {len(self.vector)} doesn't match dimension {self.dimension}")


@dataclass
class FaceRegion:
    """Represents a detected face region in an image."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    
    def __post_init__(self):
        """Validate face region data after initialization."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Face region width and height must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class SearchResult:
    """Represents a similarity search result."""
    embedding_id: str
    similarity_score: float
    metadata: Dict
    rerank_score: Optional[float] = None
    
    def __post_init__(self):
        """Validate and clamp search result data after initialization."""
        # Clamp similarity score to handle floating-point precision issues
        self.similarity_score = max(0.0, min(1.0, self.similarity_score))
        
        if self.rerank_score is not None:
            # Clamp rerank score to handle floating-point precision issues
            self.rerank_score = max(0.0, min(1.0, self.rerank_score))


@dataclass
class SearchConfig:
    """Configuration for similarity search operations."""
    top_k: int = 10
    similarity_threshold: float = 0.7
    enable_reranking: bool = True
    distance_metric: str = "cosine"
    
    def __post_init__(self):
        """Validate search configuration after initialization."""
        if self.top_k <= 0:
            raise ConfigurationError(f"top_k must be positive, got {self.top_k}")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ConfigurationError(f"Similarity threshold must be between 0.0 and 1.0, got {self.similarity_threshold}")
        if self.distance_metric not in ["cosine", "euclidean", "dot_product"]:
            raise ConfigurationError(f"Unsupported distance metric: {self.distance_metric}")


@dataclass
class RerankingFeatures:
    """Additional features used for reranking search results."""
    face_quality_score: float
    landmark_confidence: float
    pose_angle: float
    illumination_score: float
    
    def __post_init__(self):
        """Validate reranking features after initialization."""
        for score_name, score_value in [
            ("face_quality_score", self.face_quality_score),
            ("landmark_confidence", self.landmark_confidence),
            ("illumination_score", self.illumination_score)
        ]:
            if not 0.0 <= score_value <= 1.0:
                raise ValueError(f"{score_name} must be between 0.0 and 1.0, got {score_value}")


@dataclass
class RecognitionRequest:
    """Request data for face recognition operations."""
    image_data: np.ndarray
    search_config: SearchConfig
    extract_features: bool = True
    
    def __post_init__(self):
        """Validate recognition request after initialization."""
        if self.image_data.size == 0:
            raise ValueError("Image data cannot be empty")


@dataclass
class RecognitionResponse:
    """Response data for face recognition operations."""
    detected_faces: List[FaceRegion]
    search_results: List[SearchResult]
    processing_time_ms: float
    success: bool
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate recognition response after initialization."""
        if self.processing_time_ms < 0:
            raise ValueError(f"Processing time must be non-negative, got {self.processing_time_ms}")
        if not self.success and self.error_message is None:
            raise ValueError("Error message must be provided when success is False")