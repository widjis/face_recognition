"""Custom exceptions for the face recognition system."""


class FaceRecognitionError(Exception):
    """Base exception for face recognition system."""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "FACE_RECOGNITION_ERROR"


class FaceDetectionError(FaceRecognitionError):
    """Raised when face detection fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "FACE_DETECTION_ERROR")


class EmbeddingExtractionError(FaceRecognitionError):
    """Raised when embedding extraction fails."""
    
    def __init__(self, message: str):
        super().__init__(message, "EMBEDDING_EXTRACTION_ERROR")


class VectorDatabaseError(FaceRecognitionError):
    """Raised when vector database operations fail."""
    
    def __init__(self, message: str):
        super().__init__(message, "VECTOR_DATABASE_ERROR")


class InvalidImageError(FaceRecognitionError):
    """Raised when input image is invalid or corrupted."""
    
    def __init__(self, message: str):
        super().__init__(message, "INVALID_IMAGE_ERROR")


class SimilaritySearchError(FaceRecognitionError):
    """Raised when similarity search operations fail."""
    
    def __init__(self, message: str):
        super().__init__(message, "SIMILARITY_SEARCH_ERROR")


class RerankingError(FaceRecognitionError):
    """Raised when reranking operations fail."""
    
    def __init__(self, message: str):
        super().__init__(message, "RERANKING_ERROR")


class ConfigurationError(FaceRecognitionError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str):
        super().__init__(message, "CONFIGURATION_ERROR")