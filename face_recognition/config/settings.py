"""Configuration settings and validation using Pydantic."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
from enum import Enum


class DetectionMethod(str, Enum):
    """Supported face detection methods."""
    HAAR = "haar"
    DNN = "dnn"
    BOTH = "both"


class EmbeddingModel(str, Enum):
    """Supported embedding models."""
    SIMPLE = "simple"
    FACENET = "facenet"
    ARCFACE = "arcface"


class IndexType(str, Enum):
    """Supported vector database index types."""
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"


class DistanceMetric(str, Enum):
    """Supported distance metrics."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


class FaceDetectionConfig(BaseModel):
    """Configuration for face detection."""
    method: DetectionMethod = DetectionMethod.HAAR
    min_face_size: tuple = Field(default=(30, 30), description="Minimum face size (width, height)")
    scale_factor: float = Field(default=1.1, ge=1.01, le=2.0, description="Detection scale factor")
    min_neighbors: int = Field(default=5, ge=1, le=20, description="Minimum neighbors for detection")
    
    @validator('min_face_size')
    def validate_min_face_size(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError("min_face_size must be a tuple of two positive integers")
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for embedding extraction."""
    model_name: EmbeddingModel = EmbeddingModel.SIMPLE
    embedding_dim: int = Field(default=512, ge=128, le=2048, description="Embedding dimension")
    batch_size: int = Field(default=32, ge=1, le=256, description="Batch processing size")
    normalize_vectors: bool = Field(default=True, description="Normalize embedding vectors")


class VectorDatabaseConfig(BaseModel):
    """Configuration for vector database."""
    dimension: int = Field(default=512, ge=128, le=2048, description="Vector dimension")
    index_type: IndexType = IndexType.FLAT
    db_path: str = Field(default="face_database", description="Database storage path")
    
    # Index-specific parameters
    ivf_nlist: int = Field(default=100, ge=10, le=1000, description="IVF clusters (for IVF index)")
    hnsw_m: int = Field(default=32, ge=4, le=128, description="HNSW connections (for HNSW index)")
    
    # Performance settings
    max_database_size: int = Field(default=1000000, ge=1000, description="Maximum database size")
    auto_save: bool = Field(default=True, description="Auto-save database changes")


class SearchConfig(BaseModel):
    """Configuration for similarity search."""
    top_k: int = Field(default=10, ge=1, le=1000, description="Number of results to return")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")
    enable_reranking: bool = Field(default=True, description="Enable result reranking")
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    
    # Advanced search settings
    enable_filtering: bool = Field(default=True, description="Enable metadata filtering")
    max_search_time_ms: int = Field(default=1000, ge=100, description="Maximum search time in milliseconds")


class RerankingConfig(BaseModel):
    """Configuration for result reranking."""
    enable_quality_scoring: bool = Field(default=True, description="Enable face quality assessment")
    enable_pose_analysis: bool = Field(default=True, description="Enable pose angle analysis")
    enable_illumination_analysis: bool = Field(default=True, description="Enable illumination quality analysis")
    
    # Reranking weights
    similarity_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="Similarity score weight")
    quality_weight: float = Field(default=0.2, ge=0.0, le=1.0, description="Quality score weight")
    pose_weight: float = Field(default=0.1, ge=0.0, le=1.0, description="Pose quality weight")
    illumination_weight: float = Field(default=0.1, ge=0.0, le=1.0, description="Illumination quality weight")
    
    @validator('illumination_weight')
    def validate_weights_sum(cls, v, values):
        """Ensure all weights sum to approximately 1.0."""
        total = (values.get('similarity_weight', 0) + 
                values.get('quality_weight', 0) + 
                values.get('pose_weight', 0) + v)
        
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Reranking weights must sum to 1.0, got {total}")
        return v


class PerformanceConfig(BaseModel):
    """Configuration for performance settings."""
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_size: int = Field(default=1000, ge=10, le=10000, description="Maximum cache size")
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    max_workers: int = Field(default=4, ge=1, le=32, description="Maximum worker threads")
    
    # Memory management
    max_memory_usage_mb: int = Field(default=2048, ge=512, description="Maximum memory usage in MB")
    enable_memory_monitoring: bool = Field(default=True, description="Enable memory usage monitoring")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="text", description="Log format (text or json)")
    log_file: Optional[str] = Field(default=None, description="Log file path (None for console only)")
    enable_performance_logging: bool = Field(default=True, description="Enable performance metrics logging")
    enable_error_logging: bool = Field(default=True, description="Enable error logging")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()


class FaceRecognitionConfig(BaseModel):
    """Main configuration class for the face recognition system."""
    
    # Component configurations
    face_detection: FaceDetectionConfig = Field(default_factory=FaceDetectionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_database: VectorDatabaseConfig = Field(default_factory=VectorDatabaseConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Global settings
    system_name: str = Field(default="Face Recognition System", description="System identifier")
    version: str = Field(default="1.0.0", description="Configuration version")
    environment: str = Field(default="development", description="Environment (development/production)")
    
    # Feature flags
    enable_face_detection: bool = Field(default=True, description="Enable face detection")
    enable_embedding_extraction: bool = Field(default=True, description="Enable embedding extraction")
    enable_similarity_search: bool = Field(default=True, description="Enable similarity search")
    enable_reranking: bool = Field(default=True, description="Enable result reranking")
    
    # Security settings
    enable_input_validation: bool = Field(default=True, description="Enable input validation")
    max_image_size_mb: int = Field(default=10, ge=1, le=100, description="Maximum image size in MB")
    allowed_image_formats: List[str] = Field(
        default=["jpg", "jpeg", "png", "bmp"], 
        description="Allowed image formats"
    )
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"  # Don't allow extra fields
    
    def get_component_config(self, component_name: str) -> BaseModel:
        """Get configuration for a specific component."""
        component_configs = {
            'face_detection': self.face_detection,
            'embedding': self.embedding,
            'vector_database': self.vector_database,
            'search': self.search,
            'reranking': self.reranking,
            'performance': self.performance,
            'logging': self.logging
        }
        
        if component_name not in component_configs:
            raise ValueError(f"Unknown component: {component_name}")
        
        return component_configs[component_name]
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        feature_flags = {
            'face_detection': self.enable_face_detection,
            'embedding_extraction': self.enable_embedding_extraction,
            'similarity_search': self.enable_similarity_search,
            'reranking': self.enable_reranking
        }
        
        return feature_flags.get(feature_name, False)
    
    def get_performance_settings(self) -> Dict:
        """Get performance-related settings."""
        return {
            'enable_caching': self.performance.enable_caching,
            'cache_size': self.performance.cache_size,
            'enable_parallel_processing': self.performance.enable_parallel_processing,
            'max_workers': self.performance.max_workers,
            'max_memory_usage_mb': self.performance.max_memory_usage_mb
        }
    
    def validate_consistency(self) -> List[str]:
        """Validate configuration consistency and return any warnings."""
        warnings = []
        
        # Check embedding dimension consistency
        if self.embedding.embedding_dim != self.vector_database.dimension:
            warnings.append(
                f"Embedding dimension ({self.embedding.embedding_dim}) doesn't match "
                f"vector database dimension ({self.vector_database.dimension})"
            )
        
        # Check search configuration consistency
        if self.search.enable_reranking and not self.enable_reranking:
            warnings.append("Search reranking is enabled but global reranking is disabled")
        
        # Check performance settings
        if self.performance.max_workers > 16 and self.environment == "development":
            warnings.append("High number of workers in development environment may cause issues")
        
        # Check memory settings
        if self.performance.max_memory_usage_mb < 1024:
            warnings.append("Low memory limit may affect performance with large databases")
        
        return warnings


# Predefined configuration profiles
class ConfigurationProfiles:
    """Predefined configuration profiles for common use cases."""
    
    @staticmethod
    def development() -> FaceRecognitionConfig:
        """Development configuration with debugging enabled."""
        config = FaceRecognitionConfig()
        config.environment = "development"
        config.logging.log_level = "DEBUG"
        config.performance.enable_caching = False  # Disable caching for development
        config.vector_database.auto_save = True
        return config
    
    @staticmethod
    def production() -> FaceRecognitionConfig:
        """Production configuration optimized for performance."""
        config = FaceRecognitionConfig()
        config.environment = "production"
        config.logging.log_level = "INFO"
        config.performance.enable_caching = True
        config.performance.cache_size = 5000
        config.performance.max_workers = 8
        config.vector_database.index_type = IndexType.HNSW  # Faster for large datasets
        return config
    
    @staticmethod
    def high_accuracy() -> FaceRecognitionConfig:
        """Configuration optimized for accuracy over speed."""
        config = FaceRecognitionConfig()
        config.face_detection.min_neighbors = 8  # More strict detection
        config.search.similarity_threshold = 0.8  # Higher threshold
        config.reranking.quality_weight = 0.3  # Higher quality weight
        config.reranking.similarity_weight = 0.5
        config.vector_database.index_type = IndexType.FLAT  # Exact search
        return config
    
    @staticmethod
    def high_speed() -> FaceRecognitionConfig:
        """Configuration optimized for speed over accuracy."""
        config = FaceRecognitionConfig()
        config.face_detection.min_neighbors = 3  # Faster detection
        config.search.similarity_threshold = 0.6  # Lower threshold
        config.enable_reranking = False  # Skip reranking for speed
        config.performance.enable_parallel_processing = True
        config.performance.max_workers = 16
        config.vector_database.index_type = IndexType.IVF  # Approximate search
        return config
    
    @staticmethod
    def memory_efficient() -> FaceRecognitionConfig:
        """Configuration for memory-constrained environments."""
        config = FaceRecognitionConfig()
        config.embedding.batch_size = 8  # Smaller batches
        config.performance.cache_size = 100  # Smaller cache
        config.performance.max_memory_usage_mb = 512
        config.vector_database.max_database_size = 10000  # Smaller database
        return config