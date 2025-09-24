"""API request and response models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import base64
import numpy as np
import cv2


class ImageData(BaseModel):
    """Model for image data in API requests."""
    
    # Image can be provided as base64 string or file path
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    image_path: Optional[str] = Field(None, description="Path to image file")
    
    @validator('image_base64', 'image_path')
    def validate_image_source(cls, v, values):
        """Ensure at least one image source is provided."""
        if not v and not values.get('image_base64') and not values.get('image_path'):
            raise ValueError("Either image_base64 or image_path must be provided")
        return v
    
    def to_numpy_array(self) -> np.ndarray:
        """Convert image data to numpy array."""
        if self.image_base64:
            # Decode base64 image
            image_bytes = base64.b64decode(self.image_base64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode base64 image")
            return image
        elif self.image_path:
            # Load image from path
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError(f"Failed to load image from path: {self.image_path}")
            return image
        else:
            raise ValueError("No image data provided")


class SearchConfigAPI(BaseModel):
    """API model for search configuration."""
    
    top_k: int = Field(default=10, ge=1, le=100, description="Number of top results to return")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")
    enable_reranking: bool = Field(default=True, description="Whether to enable reranking")
    distance_metric: str = Field(default="cosine", description="Distance metric for similarity search")
    
    @validator('distance_metric')
    def validate_distance_metric(cls, v):
        """Validate distance metric."""
        valid_metrics = ["cosine", "euclidean", "dot_product"]
        if v not in valid_metrics:
            raise ValueError(f"distance_metric must be one of {valid_metrics}")
        return v


class FaceRegionAPI(BaseModel):
    """API model for detected face region."""
    
    x: int = Field(description="Top-left x coordinate")
    y: int = Field(description="Top-left y coordinate")
    width: int = Field(description="Face region width")
    height: int = Field(description="Face region height")
    confidence: float = Field(description="Detection confidence score")


class SearchResultAPI(BaseModel):
    """API model for similarity search result."""
    
    embedding_id: str = Field(description="Unique identifier for the matched embedding")
    similarity_score: float = Field(description="Similarity score (0.0 to 1.0)")
    rerank_score: Optional[float] = Field(None, description="Score after reranking")
    metadata: Dict[str, Any] = Field(description="Associated metadata")


class RecognitionRequest(BaseModel):
    """API request model for face recognition."""
    
    image: ImageData = Field(description="Image data to process")
    search_config: SearchConfigAPI = Field(default_factory=SearchConfigAPI, description="Search configuration")
    extract_features: bool = Field(default=True, description="Whether to extract facial features")


class RecognitionResponse(BaseModel):
    """API response model for face recognition."""
    
    success: bool = Field(description="Whether the operation was successful")
    detected_faces: List[FaceRegionAPI] = Field(description="List of detected face regions")
    search_results: List[SearchResultAPI] = Field(description="List of similarity search results")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    quality_info: Optional[Dict[str, Any]] = Field(None, description="Image quality information")


class RegistrationRequest(BaseModel):
    """API request model for face registration."""
    
    image: ImageData = Field(description="Image containing face to register")
    person_id: str = Field(description="Unique identifier for the person")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RegistrationResponse(BaseModel):
    """API response model for face registration."""
    
    success: bool = Field(description="Whether the registration was successful")
    embedding_id: Optional[str] = Field(None, description="Unique identifier for the stored embedding")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if registration failed")
    quality_info: Optional[Dict[str, Any]] = Field(None, description="Image quality information")


class BatchRecognitionRequest(BaseModel):
    """API request model for batch face recognition."""
    
    images: List[ImageData] = Field(description="List of images to process")
    search_config: SearchConfigAPI = Field(default_factory=SearchConfigAPI, description="Search configuration")
    max_workers: int = Field(default=4, ge=1, le=16, description="Maximum number of concurrent workers")


class BatchRecognitionResponse(BaseModel):
    """API response model for batch face recognition."""
    
    success: bool = Field(description="Whether the batch operation was successful")
    results: List[RecognitionResponse] = Field(description="List of recognition results")
    summary: Dict[str, Any] = Field(description="Batch processing summary")
    total_processing_time_ms: float = Field(description="Total processing time in milliseconds")


class BatchRegistrationRequest(BaseModel):
    """API request model for batch face registration."""
    
    registrations: List[RegistrationRequest] = Field(description="List of registration requests")
    max_workers: int = Field(default=4, ge=1, le=16, description="Maximum number of concurrent workers")


class BatchRegistrationResponse(BaseModel):
    """API response model for batch face registration."""
    
    success: bool = Field(description="Whether the batch operation was successful")
    results: List[RegistrationResponse] = Field(description="List of registration results")
    summary: Dict[str, Any] = Field(description="Batch processing summary")
    total_processing_time_ms: float = Field(description="Total processing time in milliseconds")


class DatabaseInfoResponse(BaseModel):
    """API response model for database information."""
    
    total_faces: int = Field(description="Total number of faces in database")
    database_path: str = Field(description="Path to database storage")
    index_type: str = Field(description="Type of vector index used")
    distance_metric: str = Field(description="Distance metric for similarity search")
    embedding_dimension: int = Field(description="Dimension of face embeddings")
    statistics: Dict[str, Any] = Field(description="Database usage statistics")


class PerformanceMetricsResponse(BaseModel):
    """API response model for performance metrics."""
    
    pipeline_stats: Dict[str, Any] = Field(description="Pipeline operation statistics")
    performance_summary: Dict[str, Any] = Field(description="Performance summary")
    error_summary: Dict[str, Any] = Field(description="Error summary")
    system_metrics: Dict[str, Any] = Field(description="System resource metrics")
    bottlenecks: List[Dict[str, Any]] = Field(description="Identified performance bottlenecks")


class OptimizationResponse(BaseModel):
    """API response model for performance optimization analysis."""
    
    analysis_timestamp: str = Field(description="Timestamp of analysis")
    recommendations: List[Dict[str, Any]] = Field(description="Optimization recommendations")
    metrics_summary: Dict[str, Any] = Field(description="Summary of current metrics")


class BenchmarkRequest(BaseModel):
    """API request model for performance benchmarking."""
    
    test_image: Optional[ImageData] = Field(None, description="Test image to use (optional)")
    num_iterations: int = Field(default=10, ge=1, le=100, description="Number of benchmark iterations")


class BenchmarkResponse(BaseModel):
    """API response model for performance benchmarking."""
    
    test_config: Dict[str, Any] = Field(description="Benchmark test configuration")
    results: Dict[str, Any] = Field(description="Benchmark results by operation")
    system_resources: Dict[str, Any] = Field(description="System resource usage during benchmark")


class HealthCheckResponse(BaseModel):
    """API response model for health check."""
    
    status: str = Field(description="Service status")
    version: str = Field(description="Service version")
    uptime_seconds: float = Field(description="Service uptime in seconds")
    database_status: str = Field(description="Database status")
    components_status: Dict[str, str] = Field(description="Status of individual components")


class ErrorResponse(BaseModel):
    """API response model for errors."""
    
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(description="Error timestamp")