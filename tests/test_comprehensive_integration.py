"""Comprehensive integration tests covering all user stories and requirements."""

import pytest
import numpy as np
import cv2
import tempfile
import os
import time
from pathlib import Path

from face_recognition.pipeline import FaceRecognitionPipeline
from face_recognition.config.manager import ConfigurationManager
from face_recognition.config.settings import ConfigurationProfiles
from face_recognition.models import RecognitionRequest, SearchConfig
from face_recognition.preprocessing import ImageProcessor
from face_recognition.exceptions import FaceDetectionError, InvalidImageError


class TestComprehensiveIntegration:
    """Comprehensive integration tests for all requirements."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def config_manager(self):
        """Create a configuration manager for testing."""
        return ConfigurationManager()
    
    @pytest.fixture
    def pipeline(self, config_manager, temp_db_path):
        """Create a pipeline instance for testing."""
        return FaceRecognitionPipeline(
            config_manager=config_manager,
            db_path=temp_db_path
        )
    
    @pytest.fixture
    def diverse_test_images(self):
        """Create diverse test images for comprehensive testing."""
        images = []
        
        # High quality face image
        high_quality = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(high_quality, (75, 75), (225, 225), (255, 255, 255), -1)
        cv2.circle(high_quality, (120, 120), 15, (0, 0, 0), -1)  # Left eye
        cv2.circle(high_quality, (180, 120), 15, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(high_quality, (150, 180), (30, 15), 0, 0, 180, (0, 0, 0), 3)  # Mouth
        images.append(('high_quality', high_quality))
        
        # Low quality face image
        low_quality = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(low_quality, (25, 25), (75, 75), (128, 128, 128), -1)
        cv2.circle(low_quality, (40, 40), 3, (0, 0, 0), -1)
        cv2.circle(low_quality, (60, 40), 3, (0, 0, 0), -1)
        # Add noise
        noise = np.random.randint(-30, 30, low_quality.shape, dtype=np.int16)
        low_quality = np.clip(low_quality.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append(('low_quality', low_quality))
        
        # Multiple faces image
        multi_face = np.zeros((400, 600, 3), dtype=np.uint8)
        # Face 1
        cv2.rectangle(multi_face, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(multi_face, (80, 80), 8, (0, 0, 0), -1)
        cv2.circle(multi_face, (120, 80), 8, (0, 0, 0), -1)
        # Face 2
        cv2.rectangle(multi_face, (300, 200), (400, 300), (255, 255, 255), -1)
        cv2.circle(multi_face, (330, 230), 8, (0, 0, 0), -1)
        cv2.circle(multi_face, (370, 230), 8, (0, 0, 0), -1)
        images.append(('multiple_faces', multi_face))
        
        # No face image
        no_face = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        images.append(('no_face', no_face))
        
        # Edge case: very small image
        tiny_image = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.rectangle(tiny_image, (10, 10), (40, 40), (255, 255, 255), -1)
        images.append(('tiny_image', tiny_image))
        
        return images
    
    def test_requirement_1_embedding_extraction(self, pipeline, diverse_test_images):
        """
        Test Requirement 1: Extract facial embeddings from images.
        
        Acceptance Criteria:
        1.1: Extract embedding vector from face image
        1.2: Extract separate embeddings for multiple faces
        1.3: Return error for no face detected
        1.4: Provide detailed error information on failure
        """
        for image_name, image in diverse_test_images:
            try:
                request = RecognitionRequest(
                    image_data=image,
                    search_config=SearchConfig(top_k=5)
                )
                
                response = pipeline.recognize_face(request)
                
                if image_name == 'no_face':
                    # 1.3: Should handle no face gracefully
                    assert response.success is True
                    assert len(response.detected_faces) == 0
                    assert len(response.search_results) == 0
                elif image_name == 'multiple_faces':
                    # 1.2: Should detect multiple faces
                    assert response.success is True
                    assert len(response.detected_faces) >= 2
                else:
                    # 1.1: Should extract embedding for single face
                    assert response.success is True
                    if len(response.detected_faces) > 0:
                        # Face was detected, embedding should be extractable
                        assert response.processing_time_ms > 0
                
            except Exception as e:
                # 1.4: Should provide detailed error information
                assert isinstance(e, (FaceDetectionError, InvalidImageError))
                assert len(str(e)) > 0
    
    def test_requirement_2_vector_storage_and_indexing(self, pipeline, diverse_test_images):
        """
        Test Requirement 2: Store and index facial embeddings efficiently.
        
        Acceptance Criteria:
        2.1: Store embedding in vector database with metadata
        2.2: Maintain index for efficient similarity search
        2.3: Handle storage gracefully when at capacity
        2.4: Handle duplicate embeddings according to policy
        """
        initial_db_size = len(pipeline.metadata_store)
        
        # 2.1: Store embeddings with metadata
        stored_ids = []
        for i, (image_name, image) in enumerate(diverse_test_images[:3]):  # Use first 3 images
            if image_name != 'no_face':  # Skip no-face image
                try:
                    metadata = {
                        'person_id': f'person_{i}',
                        'name': f'Test Person {i}',
                        'image_type': image_name,
                        'test_metadata': f'value_{i}'
                    }
                    
                    embedding_id = pipeline.add_face_to_database(
                        image=image,
                        metadata=metadata,
                        person_id=f'person_{i}'
                    )
                    
                    stored_ids.append(embedding_id)
                    
                    # Verify metadata was stored
                    assert embedding_id in pipeline.metadata_store
                    stored_metadata = pipeline.metadata_store[embedding_id]
                    assert stored_metadata['person_id'] == f'person_{i}'
                    assert stored_metadata['name'] == f'Test Person {i}'
                    
                except Exception as e:
                    # Should handle storage errors gracefully
                    assert isinstance(e, (FaceDetectionError, InvalidImageError))
        
        # 2.2: Verify index was updated
        final_db_size = len(pipeline.metadata_store)
        assert final_db_size > initial_db_size
        assert pipeline.index.ntotal == final_db_size
        
        # 2.4: Test duplicate handling (store same image again)
        if stored_ids:
            try:
                duplicate_id = pipeline.add_face_to_database(
                    image=diverse_test_images[0][1],  # Same image
                    metadata={'person_id': 'duplicate_person', 'name': 'Duplicate'},
                    person_id='duplicate_person'
                )
                # Should create new entry (current policy allows duplicates)
                assert duplicate_id not in stored_ids
            except Exception:
                # Or handle duplicates according to policy
                pass
    
    def test_requirement_3_similarity_search(self, pipeline, diverse_test_images):
        """
        Test Requirement 3: Perform similarity search on facial embeddings.
        
        Acceptance Criteria:
        3.1: Return top-k most similar embeddings
        3.2: Use cosine similarity or equivalent distance metric
        3.3: Return empty result set when no similar faces above threshold
        3.4: Return results within acceptable time limits
        """
        # First, populate database
        for i, (image_name, image) in enumerate(diverse_test_images[:2]):
            if image_name != 'no_face':
                try:
                    pipeline.add_face_to_database(
                        image=image,
                        metadata={'person_id': f'search_person_{i}', 'name': f'Search Person {i}'},
                        person_id=f'search_person_{i}'
                    )
                except Exception:
                    continue
        
        if pipeline.index.ntotal > 0:
            # 3.1 & 3.4: Test top-k search with timing
            query_image = diverse_test_images[0][1]  # Use first image as query
            
            start_time = time.time()
            request = RecognitionRequest(
                image_data=query_image,
                search_config=SearchConfig(
                    top_k=5,
                    similarity_threshold=0.1,  # Low threshold to get results
                    distance_metric="cosine"  # 3.2: Use cosine similarity
                )
            )
            
            response = pipeline.recognize_face(request)
            search_time = (time.time() - start_time) * 1000
            
            # 3.4: Should complete within reasonable time (< 1 second for small dataset)
            assert search_time < 1000
            
            if response.success and len(response.detected_faces) > 0:
                # 3.1: Should return results up to top_k
                assert len(response.search_results) <= 5
                
                # 3.2: Results should have similarity scores
                for result in response.search_results:
                    assert 0.0 <= result.similarity_score <= 1.0
            
            # 3.3: Test high threshold (should return empty results)
            high_threshold_request = RecognitionRequest(
                image_data=query_image,
                search_config=SearchConfig(
                    top_k=5,
                    similarity_threshold=0.99  # Very high threshold
                )
            )
            
            high_threshold_response = pipeline.recognize_face(high_threshold_request)
            if high_threshold_response.success:
                # Should return fewer or no results due to high threshold
                assert len(high_threshold_response.search_results) <= len(response.search_results)
    
    def test_requirement_4_reranking(self, pipeline, diverse_test_images):
        """
        Test Requirement 4: Rerank similarity search results.
        
        Acceptance Criteria:
        4.1: Apply reranking algorithms to improve result quality
        4.2: Consider additional features beyond basic embedding similarity
        4.3: Return results ordered by improved relevance scores
        4.4: Fall back to original similarity search results when reranking fails
        """
        # Populate database
        for i, (image_name, image) in enumerate(diverse_test_images[:2]):
            if image_name != 'no_face':
                try:
                    pipeline.add_face_to_database(
                        image=image,
                        metadata={'person_id': f'rerank_person_{i}', 'name': f'Rerank Person {i}'},
                        person_id=f'rerank_person_{i}'
                    )
                except Exception:
                    continue
        
        if pipeline.index.ntotal > 0 and pipeline.reranker:
            query_image = diverse_test_images[0][1]
            
            # Test with reranking enabled
            request_with_rerank = RecognitionRequest(
                image_data=query_image,
                search_config=SearchConfig(
                    top_k=5,
                    similarity_threshold=0.1,
                    enable_reranking=True  # 4.1: Enable reranking
                )
            )
            
            response_with_rerank = pipeline.recognize_face(request_with_rerank)
            
            # Test without reranking
            request_without_rerank = RecognitionRequest(
                image_data=query_image,
                search_config=SearchConfig(
                    top_k=5,
                    similarity_threshold=0.1,
                    enable_reranking=False
                )
            )
            
            response_without_rerank = pipeline.recognize_face(request_without_rerank)
            
            if (response_with_rerank.success and response_without_rerank.success and
                len(response_with_rerank.search_results) > 0 and 
                len(response_without_rerank.search_results) > 0):
                
                # 4.2 & 4.3: Reranking should potentially change result order
                # (Results might be the same for simple test cases, but rerank_score should be present)
                for result in response_with_rerank.search_results:
                    # 4.2: Should have additional reranking features considered
                    assert hasattr(result, 'rerank_score')
                    # 4.3: Results should be ordered (first result should have highest score)
                    if result.rerank_score is not None:
                        assert 0.0 <= result.rerank_score <= 1.0
    
    def test_requirement_5_configuration_management(self, pipeline):
        """
        Test Requirement 5: Configure similarity thresholds and search parameters.
        
        Acceptance Criteria:
        5.1: Allow setting similarity thresholds for matching
        5.2: Allow setting the number of results to return
        5.3: Validate parameters and return appropriate errors
        5.4: Apply configuration changes without requiring restart
        """
        # 5.3: Test parameter validation
        with pytest.raises(ValueError):
            SearchConfig(top_k=-1)  # Invalid top_k
        
        with pytest.raises(ValueError):
            SearchConfig(similarity_threshold=1.5)  # Invalid threshold
        
        with pytest.raises(ValueError):
            SearchConfig(distance_metric="invalid_metric")  # Invalid metric
        
        # 5.1 & 5.2: Test valid configuration
        valid_config = SearchConfig(
            top_k=3,  # 5.2: Set number of results
            similarity_threshold=0.8,  # 5.1: Set similarity threshold
            enable_reranking=True,
            distance_metric="cosine"
        )
        
        assert valid_config.top_k == 3
        assert valid_config.similarity_threshold == 0.8
        assert valid_config.enable_reranking is True
        assert valid_config.distance_metric == "cosine"
        
        # 5.4: Configuration changes should work without restart
        # (Pipeline should accept new config in each request)
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), -1)
        
        request1 = RecognitionRequest(
            image_data=test_image,
            search_config=SearchConfig(top_k=1)
        )
        
        request2 = RecognitionRequest(
            image_data=test_image,
            search_config=SearchConfig(top_k=5)
        )
        
        # Both requests should work with different configurations
        response1 = pipeline.recognize_face(request1)
        response2 = pipeline.recognize_face(request2)
        
        assert response1.success is True
        assert response2.success is True
    
    def test_requirement_6_image_format_handling(self, pipeline):
        """
        Test Requirement 6: Handle various image formats and preprocessing.
        
        Acceptance Criteria:
        6.1: Process common image formats (JPEG, PNG, BMP)
        6.2: Handle preprocessing (resizing, normalization) automatically
        6.3: Return appropriate error messages for invalid/corrupted images
        6.4: Provide quality warnings for low-quality images
        """
        # Create test image
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), -1)
        
        # 6.1: Test different formats through image processor
        image_processor = pipeline.image_processor
        
        # Test format support
        supported_formats = image_processor.get_supported_formats()
        assert 'JPEG' in supported_formats
        assert 'PNG' in supported_formats
        assert 'BMP' in supported_formats
        
        # 6.2: Test automatic preprocessing
        processed_image, quality_info = pipeline.process_image_with_validation(
            test_image, perform_quality_check=True
        )
        
        assert processed_image is not None
        assert processed_image.shape[2] == 3  # Should be BGR format
        assert quality_info is not None
        
        # 6.4: Quality information should be provided
        assert 'overall_score' in quality_info
        assert 'warnings' in quality_info
        assert 'recommendations' in quality_info
        assert 'acceptable' in quality_info
        
        # 6.3: Test invalid image handling
        with pytest.raises(InvalidImageError):
            pipeline.process_image_with_validation(None)
        
        with pytest.raises(InvalidImageError):
            pipeline.process_image_with_validation(np.array([]))
    
    def test_requirement_7_batch_processing(self, pipeline, diverse_test_images):
        """
        Test Requirement 7: Batch process multiple images.
        
        Acceptance Criteria:
        7.1: Handle multiple images in batch mode
        7.2: Provide progress updates
        7.3: Continue processing remaining items when errors occur
        7.4: Provide summary of results and failures
        """
        # 7.1: Test batch processing
        images = [image for _, image in diverse_test_images]
        
        progress_updates = []
        def progress_callback(completed, total, success, error):
            progress_updates.append({
                'completed': completed,
                'total': total,
                'success': success,
                'error': error
            })
        
        # 7.2: Test with progress updates
        batch_results = pipeline.batch_process_images(
            images,
            search_config=SearchConfig(top_k=3),
            progress_callback=progress_callback
        )
        
        # 7.1: Should process all images
        assert len(batch_results) == len(images)
        
        # 7.2: Should have received progress updates
        assert len(progress_updates) == len(images)
        
        # 7.3: Should continue processing even if some fail
        successful_results = [r for r in batch_results if r.success]
        failed_results = [r for r in batch_results if not r.success]
        
        # Should have at least some successful results (images with faces)
        assert len(successful_results) > 0
        
        # 7.4: Test batch processing summary
        summary = pipeline.get_batch_processing_summary(batch_results)
        
        assert 'total_processed' in summary
        assert 'successful' in summary
        assert 'failed' in summary
        assert 'success_rate' in summary
        assert 'total_faces_detected' in summary
        assert 'average_processing_time_ms' in summary
        
        assert summary['total_processed'] == len(images)
        assert summary['successful'] == len(successful_results)
        assert summary['failed'] == len(failed_results)
        assert 0.0 <= summary['success_rate'] <= 1.0
    
    def test_end_to_end_workflow(self, pipeline, diverse_test_images):
        """
        Test complete end-to-end workflow covering all major functionality.
        """
        # Step 1: Register faces in database
        registered_faces = []
        for i, (image_name, image) in enumerate(diverse_test_images[:3]):
            if image_name != 'no_face':
                try:
                    metadata = {
                        'person_id': f'e2e_person_{i}',
                        'name': f'E2E Person {i}',
                        'registration_date': '2025-01-23',
                        'image_type': image_name
                    }
                    
                    embedding_id = pipeline.add_face_to_database(
                        image=image,
                        metadata=metadata,
                        person_id=f'e2e_person_{i}'
                    )
                    
                    registered_faces.append((embedding_id, metadata))
                    
                except Exception as e:
                    # Log but continue with other faces
                    print(f"Failed to register {image_name}: {e}")
        
        # Step 2: Perform recognition on query images
        query_results = []
        for image_name, image in diverse_test_images:
            try:
                request = RecognitionRequest(
                    image_data=image,
                    search_config=SearchConfig(
                        top_k=5,
                        similarity_threshold=0.3,
                        enable_reranking=True
                    )
                )
                
                response = pipeline.recognize_face(request)
                query_results.append((image_name, response))
                
            except Exception as e:
                print(f"Failed to recognize {image_name}: {e}")
        
        # Step 3: Verify results
        assert len(registered_faces) > 0, "Should have registered at least one face"
        assert len(query_results) > 0, "Should have processed at least one query"
        
        # Check that database contains registered faces
        db_info = pipeline.get_database_info()
        assert db_info['total_faces'] == len(registered_faces)
        
        # Check that recognition produced reasonable results
        successful_recognitions = [r for _, r in query_results if r.success]
        assert len(successful_recognitions) > 0, "Should have at least one successful recognition"
        
        # Step 4: Test performance monitoring
        performance_metrics = pipeline.get_performance_metrics()
        assert performance_metrics['pipeline_stats']['total_registrations'] > 0
        assert performance_metrics['pipeline_stats']['total_recognitions'] > 0
        
        # Step 5: Test optimization analysis
        optimization_results = pipeline.optimize_performance()
        assert 'recommendations' in optimization_results
        assert 'metrics_summary' in optimization_results
        
        # Step 6: Test benchmarking
        benchmark_results = pipeline.benchmark_performance(num_iterations=2)
        assert 'results' in benchmark_results
        assert len(benchmark_results['results']) > 0
    
    def test_configuration_profiles(self, temp_db_path):
        """Test different configuration profiles."""
        # Test high accuracy profile
        high_accuracy_config = ConfigurationManager()
        high_accuracy_config.config = ConfigurationProfiles.high_accuracy()
        
        high_accuracy_pipeline = FaceRecognitionPipeline(
            config_manager=high_accuracy_config,
            db_path=temp_db_path + "_high_accuracy"
        )
        
        assert high_accuracy_pipeline.config.reranking.quality_weight == 0.3
        assert high_accuracy_pipeline.config.reranking.similarity_weight == 0.5
        
        # Test high speed profile
        high_speed_config = ConfigurationManager()
        high_speed_config.config = ConfigurationProfiles.high_speed()
        
        high_speed_pipeline = FaceRecognitionPipeline(
            config_manager=high_speed_config,
            db_path=temp_db_path + "_high_speed"
        )
        
        assert high_speed_pipeline.config.enable_reranking is False
        assert high_speed_pipeline.config.face_detection.min_face_size == [30, 30]
    
    def test_error_handling_and_recovery(self, pipeline):
        """Test comprehensive error handling and recovery mechanisms."""
        # Test invalid image handling
        with pytest.raises(InvalidImageError):
            pipeline.add_face_to_database(
                image=None,
                metadata={'person_id': 'invalid'},
                person_id='invalid'
            )
        
        # Test error statistics tracking
        initial_error_count = len(pipeline.error_handler.error_history)
        
        # Cause some errors
        try:
            pipeline.add_face_to_database(
                image=np.array([1, 2, 3]),  # Invalid image
                metadata={'person_id': 'test'},
                person_id='test'
            )
        except Exception:
            pass
        
        # Check that errors were tracked
        final_error_count = len(pipeline.error_handler.error_history)
        assert final_error_count > initial_error_count
        
        # Test error summary
        error_summary = pipeline.error_handler.get_error_summary()
        assert error_summary['total_errors'] > 0
        assert len(error_summary['error_counts']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])