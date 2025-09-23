"""Main face recognition pipeline that orchestrates all modules."""

import cv2
import numpy as np
import time
import tempfile
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Import core modules
from .face_detection import FaceDetector
from .embedding import EmbeddingExtractor
from .vector_db import VectorDatabase
from .reranking.reranker import Reranker
from .config.manager import ConfigurationManager
from .models import (
    RecognitionRequest, 
    RecognitionResponse, 
    FaceRegion, 
    SearchResult, 
    FaceEmbedding,
    SearchConfig,
    RerankingFeatures
)
from .exceptions import (
    FaceDetectionError, 
    EmbeddingExtractionError, 
    VectorDatabaseError,
    ConfigurationError,
    InvalidImageError
)

import faiss


class FaceRecognitionPipeline:
    """
    Main face recognition pipeline that orchestrates all modules.
    
    This class provides a high-level interface for:
    - Single image face recognition
    - Face registration for database population
    - Batch processing capabilities
    - End-to-end pipeline management
    """
    
    def __init__(self, 
                 config_manager: Optional[ConfigurationManager] = None,
                 db_path: str = "face_recognition_db"):
        """
        Initialize the face recognition pipeline.
        
        Args:
            config_manager: Configuration manager instance
            db_path: Path to the vector database storage
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.db_path = db_path
        self.config = self.config_manager.get_config()
        
        # Initialize components
        self._initialize_components()
        
        # Initialize vector database
        self._initialize_vector_database()
        
        # Statistics tracking
        self.stats = {
            'total_recognitions': 0,
            'total_registrations': 0,
            'average_processing_time': 0.0,
            'last_operation_time': None
        }
        
        print(f"🎯 Face Recognition Pipeline Initialized")
        print(f"   Database: {self.db_path}")
        print(f"   Configuration: {self.config.environment}")
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components based on configuration."""
        try:
            # Initialize face detector
            if self.config.enable_face_detection:
                self.face_detector = FaceDetector(
                    method=self.config.face_detection.method,
                    min_face_size=tuple(self.config.face_detection.min_face_size)
                )
            else:
                self.face_detector = None
            
            # Initialize embedding extractor
            if self.config.enable_embedding_extraction:
                self.embedding_extractor = EmbeddingExtractor(
                    model_name=self.config.embedding.model_name,
                    embedding_dim=self.config.embedding.embedding_dim
                )
            else:
                self.embedding_extractor = None
            
            # Initialize reranker
            if self.config.enable_reranking:
                reranker_config = self.config.reranking
                self.reranker = Reranker(
                    enable_quality_scoring=reranker_config.enable_quality_scoring,
                    enable_pose_analysis=reranker_config.enable_pose_analysis,
                    enable_illumination_analysis=reranker_config.enable_illumination_analysis
                )
                # Set custom weights
                self.reranker.set_reranking_weights(
                    similarity=reranker_config.similarity_weight,
                    quality=reranker_config.quality_weight,
                    pose=reranker_config.pose_weight,
                    illumination=reranker_config.illumination_weight
                )
            else:
                self.reranker = None
                
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize components: {str(e)}")
    
    def _initialize_vector_database(self) -> None:
        """Initialize the vector database with FAISS."""
        try:
            # Create database directory
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize FAISS index
            dimension = self.config.embedding.embedding_dim
            
            if self.config.vector_database.index_type == "flat":
                if self.config.search.distance_metric == "cosine":
                    self.index = faiss.IndexFlatIP(dimension)
                else:  # euclidean
                    self.index = faiss.IndexFlatL2(dimension)
            else:
                # Default to flat index for now
                self.index = faiss.IndexFlatIP(dimension)
            
            # Metadata storage
            self.metadata_store = {}
            self.stored_images = {}  # For reranking
            self.id_counter = 0
            
            # Load existing database
            self._load_database()
            
        except Exception as e:
            raise VectorDatabaseError(f"Failed to initialize vector database: {str(e)}")
    
    def _load_database(self) -> None:
        """Load existing database from disk."""
        try:
            index_path = os.path.join(self.db_path, "faiss_index.bin")
            metadata_path = os.path.join(self.db_path, "metadata.json")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load metadata
                import json
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata_store = json.load(f)
                
                # Update counter
                if self.metadata_store:
                    self.id_counter = max(int(k) for k in self.metadata_store.keys()) + 1
                
                print(f"   Loaded existing database: {len(self.metadata_store)} entries")
            
        except Exception as e:
            print(f"   Warning: Could not load existing database: {e}")
            # Continue with empty database
    
    def _save_database(self) -> None:
        """Save database to disk."""
        try:
            index_path = os.path.join(self.db_path, "faiss_index.bin")
            metadata_path = os.path.join(self.db_path, "metadata.json")
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_store, f, indent=2, default=str)
                
        except Exception as e:
            print(f"   Warning: Could not save database: {e}")
    
    def recognize_face(self, request: RecognitionRequest) -> RecognitionResponse:
        """
        Recognize faces in a single image.
        
        Args:
            request: Recognition request containing image and configuration
            
        Returns:
            RecognitionResponse with detected faces and search results
        """
        start_time = time.time()
        
        try:
            # Step 1: Face Detection
            if not self.face_detector:
                raise FaceDetectionError("Face detection is disabled")
            
            detected_faces = self.face_detector.detect_faces(request.image_data)
            
            if not detected_faces:
                return RecognitionResponse(
                    detected_faces=[],
                    search_results=[],
                    processing_time_ms=(time.time() - start_time) * 1000,
                    success=True
                )
            
            # Step 2: Extract embeddings and search for each face
            search_results = []
            
            for face_region in detected_faces:
                try:
                    # Extract embedding
                    if not self.embedding_extractor:
                        continue
                    
                    processed_face = self.face_detector.preprocess_face(
                        request.image_data, face_region
                    )
                    embedding = self.embedding_extractor.extract_embedding(processed_face)
                    
                    # Search in database
                    if self.index.ntotal > 0:  # Only search if database has entries
                        face_results = self._search_similar_faces(
                            embedding, request.search_config
                        )
                        
                        # Apply reranking if enabled
                        if self.reranker and face_results:
                            face_results = self._apply_reranking(
                                face_results, processed_face, face_region
                            )
                        
                        search_results.extend(face_results)
                
                except Exception as e:
                    print(f"   Warning: Failed to process face: {e}")
                    continue
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_stats('recognition', processing_time)
            
            return RecognitionResponse(
                detected_faces=detected_faces,
                search_results=search_results,
                processing_time_ms=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return RecognitionResponse(
                detected_faces=[],
                search_results=[],
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def add_face_to_database(self, 
                           image: np.ndarray, 
                           metadata: Dict,
                           person_id: Optional[str] = None) -> str:
        """
        Add a face to the database for future recognition.
        
        Args:
            image: Input image containing a face
            metadata: Metadata to associate with the face
            person_id: Optional person identifier
            
        Returns:
            Embedding ID of the stored face
        """
        start_time = time.time()
        
        try:
            # Step 1: Detect faces
            if not self.face_detector:
                raise FaceDetectionError("Face detection is disabled")
            
            faces = self.face_detector.detect_faces(image)
            
            if not faces:
                raise FaceDetectionError("No faces detected in the image")
            
            # Use the largest/most confident face
            best_face = max(faces, key=lambda f: f.confidence * f.width * f.height)
            
            # Step 2: Extract embedding
            if not self.embedding_extractor:
                raise EmbeddingExtractionError("Embedding extraction is disabled")
            
            processed_face = self.face_detector.preprocess_face(image, best_face)
            embedding = self.embedding_extractor.extract_embedding(processed_face)
            
            # Step 3: Store in database
            embedding_id = str(self.id_counter)
            
            # Add to FAISS index
            embedding_vector = embedding.vector.reshape(1, -1)
            if self.config.search.distance_metric == "cosine":
                # Normalize for cosine similarity
                faiss.normalize_L2(embedding_vector)
            
            self.index.add(embedding_vector)
            
            # Store metadata
            face_metadata = {
                **metadata,
                'embedding_id': embedding_id,
                'person_id': person_id or f"person_{embedding_id}",
                'face_region': {
                    'x': best_face.x,
                    'y': best_face.y,
                    'width': best_face.width,
                    'height': best_face.height,
                    'confidence': best_face.confidence
                },
                'embedding_info': {
                    'model_version': embedding.model_version,
                    'dimension': embedding.dimension,
                    'extraction_timestamp': embedding.extraction_timestamp.isoformat()
                },
                'registration_timestamp': datetime.now().isoformat()
            }
            
            self.metadata_store[embedding_id] = face_metadata
            
            # Store processed face for reranking
            if self.reranker:
                self.stored_images[embedding_id] = processed_face
            
            self.id_counter += 1
            
            # Save database
            self._save_database()
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_stats('registration', processing_time)
            
            print(f"   ✅ Face registered: {embedding_id} ({metadata.get('name', 'Unknown')})")
            
            return embedding_id
            
        except (FaceDetectionError, EmbeddingExtractionError, InvalidImageError, ValueError, TypeError) as e:
            # Re-raise specific exceptions without wrapping
            raise e
        except Exception as e:
            raise VectorDatabaseError(f"Failed to add face to database: {str(e)}")
    
    def _search_similar_faces(self, 
                            query_embedding: FaceEmbedding, 
                            search_config: SearchConfig) -> List[SearchResult]:
        """Search for similar faces in the database."""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Prepare query vector
            query_vector = query_embedding.vector.reshape(1, -1)
            if self.config.search.distance_metric == "cosine":
                faiss.normalize_L2(query_vector)
            
            # Search
            scores, indices = self.index.search(query_vector, search_config.top_k)
            
            # Convert to SearchResult objects
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # No more results
                    break
                
                embedding_id = str(idx)
                if embedding_id in self.metadata_store:
                    # Convert FAISS score to similarity
                    if self.config.search.distance_metric == "cosine":
                        similarity = float(score)  # Already similarity for IP
                    else:
                        similarity = 1.0 / (1.0 + float(score))  # Convert distance to similarity
                    
                    # Clamp similarity to valid range [0.0, 1.0] to handle floating-point precision issues
                    similarity = max(0.0, min(1.0, similarity))
                    
                    if similarity >= search_config.similarity_threshold:
                        results.append(SearchResult(
                            embedding_id=embedding_id,
                            similarity_score=similarity,
                            metadata=self.metadata_store[embedding_id]
                        ))
            
            return results
            
        except Exception as e:
            print(f"   Warning: Search failed: {e}")
            return []
    
    def _apply_reranking(self, 
                        results: List[SearchResult], 
                        query_face: np.ndarray,
                        face_region: FaceRegion) -> List[SearchResult]:
        """Apply reranking to search results."""
        try:
            if not self.reranker or not results:
                return results
            
            # Prepare reranking features for query
            query_features = RerankingFeatures(
                face_quality_score=min(face_region.confidence, 1.0),
                landmark_confidence=0.8,  # Default value
                pose_angle=0.0,  # Default value
                illumination_score=0.8  # Default value
            )
            
            # Get stored images for comparison
            stored_faces = []
            for result in results:
                if result.embedding_id in self.stored_images:
                    stored_faces.append(self.stored_images[result.embedding_id])
                else:
                    stored_faces.append(None)
            
            # Apply reranking
            reranked_results = self.reranker.rerank_results(
                results, query_features, stored_faces
            )
            
            return reranked_results
            
        except Exception as e:
            print(f"   Warning: Reranking failed: {e}")
            return results
    
    def _update_stats(self, operation_type: str, processing_time: float) -> None:
        """Update pipeline statistics."""
        if operation_type == 'recognition':
            self.stats['total_recognitions'] += 1
        elif operation_type == 'registration':
            self.stats['total_registrations'] += 1
        
        # Update average processing time
        total_ops = self.stats['total_recognitions'] + self.stats['total_registrations']
        if total_ops > 0:
            current_avg = self.stats['average_processing_time']
            self.stats['average_processing_time'] = (
                (current_avg * (total_ops - 1) + processing_time) / total_ops
            )
        
        self.stats['last_operation_time'] = datetime.now().isoformat()
    
    def get_database_info(self) -> Dict:
        """Get information about the current database."""
        return {
            'total_faces': len(self.metadata_store),
            'database_path': self.db_path,
            'index_type': self.config.vector_database.index_type,
            'distance_metric': self.config.search.distance_metric,
            'embedding_dimension': self.config.embedding.embedding_dim,
            'statistics': self.stats
        }
    
    def batch_process_images(self, 
                           images: List[np.ndarray],
                           search_config: Optional[SearchConfig] = None) -> List[RecognitionResponse]:
        """
        Process multiple images in batch.
        
        Args:
            images: List of images to process
            search_config: Search configuration to use
            
        Returns:
            List of recognition responses
        """
        if search_config is None:
            search_config = SearchConfig()
        
        results = []
        
        for i, image in enumerate(images):
            try:
                request = RecognitionRequest(
                    image_data=image,
                    search_config=search_config
                )
                
                response = self.recognize_face(request)
                results.append(response)
                
                print(f"   Processed image {i+1}/{len(images)}: "
                      f"{'✅' if response.success else '❌'}")
                
            except Exception as e:
                # Create error response
                error_response = RecognitionResponse(
                    detected_faces=[],
                    search_results=[],
                    processing_time_ms=0.0,
                    success=False,
                    error_message=str(e)
                )
                results.append(error_response)
                print(f"   Failed to process image {i+1}/{len(images)}: {e}")
        
        return results
    
    def clear_database(self) -> None:
        """Clear all data from the database."""
        try:
            # Reinitialize empty index
            dimension = self.config.embedding.embedding_dim
            if self.config.search.distance_metric == "cosine":
                self.index = faiss.IndexFlatIP(dimension)
            else:
                self.index = faiss.IndexFlatL2(dimension)
            
            # Clear metadata
            self.metadata_store.clear()
            self.stored_images.clear()
            self.id_counter = 0
            
            # Save empty database
            self._save_database()
            
            print("   ✅ Database cleared successfully")
            
        except Exception as e:
            raise VectorDatabaseError(f"Failed to clear database: {str(e)}")