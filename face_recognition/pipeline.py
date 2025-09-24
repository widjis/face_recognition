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
from .preprocessing import ImageProcessor
from .logging import FaceRecognitionLogger, ErrorHandler, ErrorRecoveryManager, PerformanceMonitor, setup_logging
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
        
        # Initialize logging and error handling
        self.logger = setup_logging(self.config.logging)
        self.error_handler = ErrorHandler()
        self.recovery_manager = ErrorRecoveryManager(self.error_handler)
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize components
        self._initialize_components()
        
        # Initialize image processor
        self.image_processor = ImageProcessor(
            min_resolution=(100, 100),
            target_resolution=(224, 224),
            auto_enhance=True,
            quality_threshold=0.3
        )
        
        # Initialize vector database
        self._initialize_vector_database()
        
        # Statistics tracking
        self.stats = {
            'total_recognitions': 0,
            'total_registrations': 0,
            'average_processing_time': 0.0,
            'last_operation_time': None
        }
        
        self.logger.info("Face Recognition Pipeline Initialized",
                         database_path=self.db_path,
                         environment=self.config.environment,
                         components_enabled={
                             'face_detection': self.config.enable_face_detection,
                             'embedding_extraction': self.config.enable_embedding_extraction,
                             'reranking': self.config.enable_reranking
                         })
    
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
    
    def process_image_with_validation(self, 
                                    image_source,
                                    perform_quality_check: bool = True) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Process and validate an image using the preprocessing pipeline.
        
        Args:
            image_source: Image file path, bytes, or numpy array
            perform_quality_check: Whether to perform quality assessment
            
        Returns:
            Tuple of (processed_image, quality_info)
            
        Raises:
            InvalidImageError: If image processing fails
        """
        try:
            processed_image, quality_metrics = self.image_processor.load_and_validate_image(
                image_source, perform_quality_check
            )
            
            # Preprocess for face detection
            processed_image = self.image_processor.preprocess_for_face_detection(processed_image)
            
            quality_info = None
            if quality_metrics:
                quality_info = {
                    'overall_score': quality_metrics.overall_score,
                    'sharpness_score': quality_metrics.sharpness_score,
                    'brightness_score': quality_metrics.brightness_score,
                    'contrast_score': quality_metrics.contrast_score,
                    'noise_level': quality_metrics.noise_level,
                    'resolution_score': quality_metrics.resolution_score,
                    'warnings': quality_metrics.warnings,
                    'acceptable': self.image_processor.quality_assessor.is_acceptable_quality(quality_metrics),
                    'recommendations': self.image_processor.quality_assessor.get_quality_recommendations(quality_metrics)
                }
            
            return processed_image, quality_info
            
        except Exception as e:
            raise InvalidImageError(f"Image processing failed: {str(e)}")

    def recognize_face(self, request: RecognitionRequest) -> RecognitionResponse:
        """
        Recognize faces in a single image.
        
        Args:
            request: Recognition request containing image and configuration
            
        Returns:
            RecognitionResponse with detected faces and search results
        """
        with self.performance_monitor.measure_operation(
            "face_recognition",
            image_shape=request.image_data.shape,
            search_config=request.search_config.__dict__
        ) as metric:
            try:
                return self._recognize_face_internal(request, metric)
            except Exception as e:
                self.error_handler.handle_error(e, "face_recognition", {
                    'image_shape': request.image_data.shape,
                    'search_config': request.search_config.__dict__
                })
                
                # Return error response
                return RecognitionResponse(
                    detected_faces=[],
                    search_results=[],
                    processing_time_ms=metric.duration_ms,
                    success=False,
                    error_message=str(e)
                )
    
    def _recognize_face_internal(self, request: RecognitionRequest, metric) -> RecognitionResponse:
        """Internal face recognition implementation with detailed logging."""
        try:
            # Step 1: Face Detection
            if not self.face_detector:
                raise FaceDetectionError("Face detection is disabled")
            
            face_detection_start = time.time()
            detected_faces = self.face_detector.detect_faces(request.image_data)
            face_detection_time = (time.time() - face_detection_start) * 1000
            
            # Log face detection results
            self.logger.log_face_detection(
                image_info={
                    'width': request.image_data.shape[1],
                    'height': request.image_data.shape[0],
                    'channels': request.image_data.shape[2] if len(request.image_data.shape) > 2 else 1
                },
                detected_faces=len(detected_faces),
                processing_time=face_detection_time,
                success=True
            )
            
            if not detected_faces:
                self.logger.info("No faces detected in image")
                return RecognitionResponse(
                    detected_faces=[],
                    search_results=[],
                    processing_time_ms=metric.duration_ms,
                    success=True
                )
            
            # Step 2: Extract embeddings and search for each face
            search_results = []
            
            for face_region in detected_faces:
                try:
                    # Extract embedding
                    if not self.embedding_extractor:
                        self.logger.warning("Embedding extraction is disabled, skipping face")
                        continue
                    
                    embedding_start = time.time()
                    processed_face = self.face_detector.preprocess_face(
                        request.image_data, face_region
                    )
                    embedding = self.embedding_extractor.extract_embedding(processed_face)
                    embedding_time = (time.time() - embedding_start) * 1000
                    
                    # Log embedding extraction
                    self.logger.log_embedding_extraction(
                        face_count=1,
                        embedding_dim=embedding.dimension,
                        processing_time=embedding_time,
                        model_version=embedding.model_version,
                        success=True
                    )
                    
                    # Search in database
                    if self.index.ntotal > 0:  # Only search if database has entries
                        search_start = time.time()
                        face_results = self._search_similar_faces(
                            embedding, request.search_config
                        )
                        search_time = (time.time() - search_start) * 1000
                        
                        # Log similarity search
                        self.logger.log_similarity_search(
                            query_embedding_dim=embedding.dimension,
                            database_size=self.index.ntotal,
                            top_k=request.search_config.top_k,
                            results_found=len(face_results),
                            processing_time=search_time,
                            success=True
                        )
                        
                        # Apply reranking if enabled
                        if self.reranker and face_results:
                            rerank_start = time.time()
                            face_results = self._apply_reranking(
                                face_results, processed_face, face_region
                            )
                            rerank_time = (time.time() - rerank_start) * 1000
                            
                            # Log reranking
                            self.logger.log_reranking(
                                initial_results=len(face_results),
                                final_results=len(face_results),
                                processing_time=rerank_time,
                                reranking_features={
                                    'face_quality': face_region.confidence,
                                    'face_size': face_region.width * face_region.height
                                },
                                success=True
                            )
                        
                        search_results.extend(face_results)
                    else:
                        self.logger.info("Database is empty, no similarity search performed")
                
                except Exception as e:
                    self.error_handler.handle_error(e, "face_processing", {
                        'face_region': {
                            'x': face_region.x, 'y': face_region.y,
                            'width': face_region.width, 'height': face_region.height,
                            'confidence': face_region.confidence
                        }
                    })
                    self.logger.warning("Failed to process face", 
                                      exception=e,
                                      face_region=face_region.__dict__)
                    continue
            
            # Update statistics
            self._update_stats('recognition', metric.duration_ms)
            
            self.logger.info("Face recognition completed successfully",
                           detected_faces=len(detected_faces),
                           search_results=len(search_results),
                           processing_time_ms=metric.duration_ms)
            
            return RecognitionResponse(
                detected_faces=detected_faces,
                search_results=search_results,
                processing_time_ms=metric.duration_ms,
                success=True
            )
            
        except Exception as e:
            self.logger.error("Face recognition failed", exception=e)
            raise  # Re-raise to be handled by outer try-catch
    
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
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            'pipeline_stats': self.stats,
            'performance_summary': self.performance_monitor.get_performance_summary(),
            'error_summary': self.error_handler.get_error_summary(),
            'system_metrics': self.performance_monitor.get_system_stats(time_window_hours=1.0),
            'bottlenecks': self.performance_monitor.identify_bottlenecks(),
            'database_info': self.get_database_info()
        }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """
        Analyze performance and provide optimization recommendations.
        
        Returns:
            Dictionary with optimization recommendations
        """
        recommendations = []
        metrics = self.get_performance_metrics()
        
        # Check for slow operations
        bottlenecks = metrics['bottlenecks']
        if bottlenecks:
            for bottleneck in bottlenecks:
                recommendations.append({
                    'type': 'performance',
                    'severity': 'high' if bottleneck['type'] == 'consistently_slow' else 'medium',
                    'operation': bottleneck['operation'],
                    'issue': bottleneck['type'],
                    'recommendation': bottleneck['recommendation']
                })
        
        # Check database size vs performance
        db_size = len(self.metadata_store)
        if db_size > 10000:
            recommendations.append({
                'type': 'database',
                'severity': 'medium',
                'issue': 'large_database',
                'recommendation': f'Database has {db_size} entries. Consider using IVF index for better search performance.'
            })
        
        # Check memory usage
        system_stats = metrics['system_metrics']
        if system_stats.get('process_memory', {}).get('current_mb', 0) > 1000:
            recommendations.append({
                'type': 'memory',
                'severity': 'medium',
                'issue': 'high_memory_usage',
                'recommendation': 'High memory usage detected. Consider clearing old metrics or reducing batch sizes.'
            })
        
        # Check error rates
        error_summary = metrics['error_summary']
        total_ops = self.stats['total_recognitions'] + self.stats['total_registrations']
        if total_ops > 0:
            error_rate = error_summary['total_errors'] / total_ops
            if error_rate > 0.1:  # More than 10% error rate
                recommendations.append({
                    'type': 'reliability',
                    'severity': 'high',
                    'issue': 'high_error_rate',
                    'recommendation': f'Error rate is {error_rate:.1%}. Check logs for common failure patterns.'
                })
        
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'recommendations': recommendations,
            'metrics_summary': {
                'total_operations': total_ops,
                'average_processing_time': self.stats['average_processing_time'],
                'database_size': db_size,
                'error_count': error_summary['total_errors']
            }
        }
    
    def batch_process_images(self, 
                           images: List[np.ndarray],
                           search_config: Optional[SearchConfig] = None,
                           max_workers: int = 4,
                           progress_callback: Optional[callable] = None) -> List[RecognitionResponse]:
        """
        Process multiple images in batch with concurrent processing.
        
        Args:
            images: List of images to process
            search_config: Search configuration to use
            max_workers: Maximum number of concurrent workers
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of recognition responses
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        if search_config is None:
            search_config = SearchConfig()
        
        total_images = len(images)
        results = [None] * total_images  # Pre-allocate results list
        completed_count = 0
        lock = threading.Lock()
        
        def process_single_image(index: int, image: np.ndarray) -> tuple:
            """Process a single image and return (index, result)."""
            nonlocal completed_count
            
            try:
                request = RecognitionRequest(
                    image_data=image,
                    search_config=search_config
                )
                
                response = self.recognize_face(request)
                
                with lock:
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, total_images, True, None)
                    else:
                        print(f"   Processed image {completed_count}/{total_images}: ✅")
                
                return (index, response)
                
            except Exception as e:
                # Create error response
                error_response = RecognitionResponse(
                    detected_faces=[],
                    search_results=[],
                    processing_time_ms=0.0,
                    success=False,
                    error_message=str(e)
                )
                
                with lock:
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, total_images, False, str(e))
                    else:
                        print(f"   Failed to process image {completed_count}/{total_images}: ❌ {e}")
                
                return (index, error_response)
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_image, i, image): i 
                for i, image in enumerate(images)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index, result = future.result()
                results[index] = result
        
        return results
    
    def batch_register_faces(self,
                           images: List[np.ndarray],
                           metadata_list: List[Dict],
                           max_workers: int = 4,
                           progress_callback: Optional[callable] = None) -> List[Optional[str]]:
        """
        Register multiple faces in batch with concurrent processing.
        
        Args:
            images: List of images containing faces to register
            metadata_list: List of metadata dictionaries for each image
            max_workers: Maximum number of concurrent workers
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of embedding IDs (None for failed registrations)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        if len(images) != len(metadata_list):
            raise ValueError("Number of images must match number of metadata entries")
        
        total_images = len(images)
        results = [None] * total_images
        completed_count = 0
        lock = threading.Lock()
        
        def register_single_face(index: int, image: np.ndarray, metadata: Dict) -> tuple:
            """Register a single face and return (index, embedding_id)."""
            nonlocal completed_count
            
            try:
                embedding_id = self.add_face_to_database(
                    image=image,
                    metadata=metadata,
                    person_id=metadata.get('person_id')
                )
                
                with lock:
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, total_images, True, None)
                    else:
                        print(f"   Registered face {completed_count}/{total_images}: ✅")
                
                return (index, embedding_id)
                
            except Exception as e:
                with lock:
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, total_images, False, str(e))
                    else:
                        print(f"   Failed to register face {completed_count}/{total_images}: ❌ {e}")
                
                return (index, None)
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(register_single_face, i, image, metadata): i 
                for i, (image, metadata) in enumerate(zip(images, metadata_list))
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index, result = future.result()
                results[index] = result
        
        return results
    
    def get_batch_processing_summary(self, results: List[RecognitionResponse]) -> Dict:
        """
        Generate a summary of batch processing results.
        
        Args:
            results: List of recognition responses from batch processing
            
        Returns:
            Dictionary containing processing summary statistics
        """
        total_processed = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total_processed - successful
        
        total_faces_detected = sum(len(r.detected_faces) for r in results if r.success)
        total_matches_found = sum(len(r.search_results) for r in results if r.success)
        
        processing_times = [r.processing_time_ms for r in results if r.success]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            'total_processed': total_processed,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_processed if total_processed > 0 else 0,
            'total_faces_detected': total_faces_detected,
            'total_matches_found': total_matches_found,
            'average_processing_time_ms': avg_processing_time,
            'total_processing_time_ms': sum(processing_times)
        }
    
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
            
            self.logger.info("Database cleared successfully")
            
        except Exception as e:
            raise VectorDatabaseError(f"Failed to clear database: {str(e)}")
    
    def benchmark_performance(self, 
                            test_image: Optional[np.ndarray] = None,
                            num_iterations: int = 10) -> Dict[str, Any]:
        """
        Run performance benchmarks on the system.
        
        Args:
            test_image: Test image to use (creates synthetic if None)
            num_iterations: Number of iterations to run
            
        Returns:
            Benchmark results
        """
        if test_image is None:
            # Create synthetic test image
            test_image = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.rectangle(test_image, (50, 50), (174, 174), (255, 255, 255), -1)
            cv2.circle(test_image, (112, 112), 30, (128, 128, 128), -1)
        
        benchmark_results = {
            'test_config': {
                'num_iterations': num_iterations,
                'image_shape': test_image.shape,
                'database_size': len(self.metadata_store)
            },
            'results': {}
        }
        
        # Benchmark face detection
        face_detection_times = []
        for i in range(num_iterations):
            with self.performance_monitor.measure_operation("benchmark_face_detection") as metric:
                try:
                    faces = self.face_detector.detect_faces(test_image)
                    face_detection_times.append(metric.duration_ms)
                except Exception as e:
                    self.logger.warning("Face detection benchmark failed", exception=e)
        
        if face_detection_times:
            benchmark_results['results']['face_detection'] = {
                'avg_time_ms': sum(face_detection_times) / len(face_detection_times),
                'min_time_ms': min(face_detection_times),
                'max_time_ms': max(face_detection_times),
                'successful_runs': len(face_detection_times)
            }
        
        # Benchmark embedding extraction (if faces detected)
        if self.face_detector and self.embedding_extractor:
            try:
                faces = self.face_detector.detect_faces(test_image)
                if faces:
                    processed_face = self.face_detector.preprocess_face(test_image, faces[0])
                    
                    embedding_times = []
                    for i in range(num_iterations):
                        with self.performance_monitor.measure_operation("benchmark_embedding") as metric:
                            try:
                                embedding = self.embedding_extractor.extract_embedding(processed_face)
                                embedding_times.append(metric.duration_ms)
                            except Exception as e:
                                self.logger.warning("Embedding extraction benchmark failed", exception=e)
                    
                    if embedding_times:
                        benchmark_results['results']['embedding_extraction'] = {
                            'avg_time_ms': sum(embedding_times) / len(embedding_times),
                            'min_time_ms': min(embedding_times),
                            'max_time_ms': max(embedding_times),
                            'successful_runs': len(embedding_times)
                        }
            except Exception as e:
                self.logger.warning("Could not benchmark embedding extraction", exception=e)
        
        # Benchmark similarity search (if database has entries)
        if self.index.ntotal > 0 and self.embedding_extractor:
            try:
                faces = self.face_detector.detect_faces(test_image)
                if faces:
                    processed_face = self.face_detector.preprocess_face(test_image, faces[0])
                    query_embedding = self.embedding_extractor.extract_embedding(processed_face)
                    
                    search_times = []
                    for i in range(num_iterations):
                        with self.performance_monitor.measure_operation("benchmark_search") as metric:
                            try:
                                results = self._search_similar_faces(query_embedding, SearchConfig(top_k=10))
                                search_times.append(metric.duration_ms)
                            except Exception as e:
                                self.logger.warning("Similarity search benchmark failed", exception=e)
                    
                    if search_times:
                        benchmark_results['results']['similarity_search'] = {
                            'avg_time_ms': sum(search_times) / len(search_times),
                            'min_time_ms': min(search_times),
                            'max_time_ms': max(search_times),
                            'successful_runs': len(search_times)
                        }
            except Exception as e:
                self.logger.warning("Could not benchmark similarity search", exception=e)
        
        # Benchmark end-to-end recognition
        e2e_times = []
        for i in range(num_iterations):
            with self.performance_monitor.measure_operation("benchmark_e2e") as metric:
                try:
                    request = RecognitionRequest(
                        image_data=test_image,
                        search_config=SearchConfig(top_k=5)
                    )
                    response = self.recognize_face(request)
                    if response.success:
                        e2e_times.append(metric.duration_ms)
                except Exception as e:
                    self.logger.warning("End-to-end benchmark failed", exception=e)
        
        if e2e_times:
            benchmark_results['results']['end_to_end'] = {
                'avg_time_ms': sum(e2e_times) / len(e2e_times),
                'min_time_ms': min(e2e_times),
                'max_time_ms': max(e2e_times),
                'successful_runs': len(e2e_times)
            }
        
        # Add system resource usage during benchmark
        system_stats = self.performance_monitor.get_system_stats(time_window_hours=0.1)
        benchmark_results['system_resources'] = system_stats
        
        self.logger.info("Performance benchmark completed", 
                        benchmark_results=benchmark_results)
        
        return benchmark_results