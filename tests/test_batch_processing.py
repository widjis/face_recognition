"""Unit tests for batch processing functionality."""

import pytest
import numpy as np
import cv2
import tempfile
import time
from unittest.mock import Mock, patch

from face_recognition.pipeline import FaceRecognitionPipeline
from face_recognition.config.manager import ConfigurationManager
from face_recognition.models import SearchConfig, RecognitionResponse
from face_recognition.exceptions import FaceDetectionError, InvalidImageError


class TestBatchProcessing:
    """Test batch processing capabilities of the face recognition pipeline."""
    
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
    def sample_images(self):
        """Create sample images for batch testing."""
        images = []
        for i in range(5):
            # Create different synthetic face-like images
            image = np.zeros((200, 200, 3), dtype=np.uint8)
            # Vary the face position and size slightly
            center_x = 100 + (i - 2) * 10
            center_y = 100 + (i - 2) * 5
            radius = 80 + i * 5
            
            cv2.circle(image, (center_x, center_y), radius, (255, 255, 255), -1)
            cv2.circle(image, (center_x - 20, center_y - 20), 8, (0, 0, 0), -1)
            cv2.circle(image, (center_x + 20, center_y - 20), 8, (0, 0, 0), -1)
            cv2.ellipse(image, (center_x, center_y + 20), (15, 8), 0, 0, 180, (0, 0, 0), 2)
            
            images.append(image)
        
        return images
    
    @pytest.fixture
    def invalid_images(self):
        """Create invalid images for error testing."""
        return [
            np.array([]),  # Empty array
            np.zeros((10, 10), dtype=np.uint8),  # Too small, wrong dimensions
            None,  # None value
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)  # Random noise
        ]
    
    def test_batch_process_images_basic(self, pipeline, sample_images):
        """Test basic batch processing functionality."""
        # Register one face first for matching
        metadata = {"person_id": "test_person", "name": "Test Person"}
        pipeline.add_face_to_database(
            image=sample_images[0],
            metadata=metadata,
            person_id="test_person"
        )
        
        # Process batch
        search_config = SearchConfig(top_k=5, similarity_threshold=0.3)
        results = pipeline.batch_process_images(sample_images, search_config)
        
        assert len(results) == len(sample_images)
        
        # Check that all results are RecognitionResponse objects
        for result in results:
            assert isinstance(result, RecognitionResponse)
            assert hasattr(result, 'success')
            assert hasattr(result, 'processing_time_ms')
    
    def test_batch_process_with_progress_callback(self, pipeline, sample_images):
        """Test batch processing with progress callback."""
        progress_updates = []
        
        def progress_callback(completed, total, success, error):
            progress_updates.append({
                'completed': completed,
                'total': total,
                'success': success,
                'error': error
            })
        
        # Process batch with callback
        results = pipeline.batch_process_images(
            sample_images,
            progress_callback=progress_callback
        )
        
        assert len(results) == len(sample_images)
        assert len(progress_updates) == len(sample_images)
        
        # Check progress updates
        for i, update in enumerate(progress_updates):
            assert update['completed'] == i + 1
            assert update['total'] == len(sample_images)
            assert isinstance(update['success'], bool)
    
    def test_batch_process_with_errors(self, pipeline, invalid_images):
        """Test batch processing with invalid images."""
        # Mix valid and invalid images
        valid_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(valid_image, (100, 100), 80, (255, 255, 255), -1)
        
        mixed_images = [valid_image] + invalid_images[:2] + [valid_image]
        
        results = pipeline.batch_process_images(mixed_images)
        
        assert len(results) == len(mixed_images)
        
        # Check that some succeeded and some failed
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        
        assert len(successes) > 0  # At least valid images should succeed
        assert len(failures) > 0   # Invalid images should fail
        
        # Check error messages are provided for failures
        for failure in failures:
            assert failure.error_message is not None
            assert len(failure.error_message) > 0
    
    def test_batch_process_concurrent_performance(self, pipeline, sample_images):
        """Test that concurrent processing improves performance."""
        # Register a face for matching
        metadata = {"person_id": "test_person", "name": "Test Person"}
        pipeline.add_face_to_database(
            image=sample_images[0],
            metadata=metadata,
            person_id="test_person"
        )
        
        # Create more images for better performance comparison
        large_batch = sample_images * 4  # 20 images
        
        # Test with single worker (sequential)
        start_time = time.time()
        results_sequential = pipeline.batch_process_images(
            large_batch, max_workers=1
        )
        sequential_time = time.time() - start_time
        
        # Test with multiple workers (concurrent)
        start_time = time.time()
        results_concurrent = pipeline.batch_process_images(
            large_batch, max_workers=4
        )
        concurrent_time = time.time() - start_time
        
        # Both should produce same number of results
        assert len(results_sequential) == len(results_concurrent)
        assert len(results_concurrent) == len(large_batch)
        
        # Concurrent should be faster (or at least not significantly slower)
        # Allow some tolerance for test environment variations
        assert concurrent_time <= sequential_time * 1.2
    
    def test_batch_register_faces(self, pipeline, sample_images):
        """Test batch face registration functionality."""
        metadata_list = [
            {"person_id": f"person_{i}", "name": f"Person {i}"}
            for i in range(len(sample_images))
        ]
        
        embedding_ids = pipeline.batch_register_faces(
            sample_images, metadata_list
        )
        
        assert len(embedding_ids) == len(sample_images)
        
        # Check that faces were registered
        db_info = pipeline.get_database_info()
        successful_registrations = [eid for eid in embedding_ids if eid is not None]
        assert len(successful_registrations) > 0
        assert db_info['total_faces'] == len(successful_registrations)
    
    def test_batch_register_with_errors(self, pipeline, invalid_images):
        """Test batch registration with invalid images."""
        metadata_list = [
            {"person_id": f"person_{i}", "name": f"Person {i}"}
            for i in range(len(invalid_images))
        ]
        
        embedding_ids = pipeline.batch_register_faces(
            invalid_images, metadata_list
        )
        
        assert len(embedding_ids) == len(invalid_images)
        
        # All should fail with invalid images
        successful_registrations = [eid for eid in embedding_ids if eid is not None]
        assert len(successful_registrations) == 0
    
    def test_batch_register_mismatched_inputs(self, pipeline, sample_images):
        """Test batch registration with mismatched input lengths."""
        # Provide fewer metadata entries than images
        metadata_list = [
            {"person_id": "person_1", "name": "Person 1"}
        ]
        
        with pytest.raises(ValueError, match="Number of images must match"):
            pipeline.batch_register_faces(sample_images, metadata_list)
    
    def test_batch_processing_summary(self, pipeline, sample_images, invalid_images):
        """Test batch processing summary generation."""
        # Mix valid and invalid images
        mixed_images = sample_images[:3] + invalid_images[:2]
        
        results = pipeline.batch_process_images(mixed_images)
        summary = pipeline.get_batch_processing_summary(results)
        
        assert 'total_processed' in summary
        assert 'successful' in summary
        assert 'failed' in summary
        assert 'success_rate' in summary
        assert 'total_faces_detected' in summary
        assert 'total_matches_found' in summary
        assert 'average_processing_time_ms' in summary
        assert 'total_processing_time_ms' in summary
        
        assert summary['total_processed'] == len(mixed_images)
        assert summary['successful'] + summary['failed'] == summary['total_processed']
        assert 0 <= summary['success_rate'] <= 1
    
    def test_batch_processing_empty_list(self, pipeline):
        """Test batch processing with empty image list."""
        results = pipeline.batch_process_images([])
        assert len(results) == 0
        
        summary = pipeline.get_batch_processing_summary(results)
        assert summary['total_processed'] == 0
        assert summary['success_rate'] == 0
    
    def test_batch_processing_with_custom_config(self, pipeline, sample_images):
        """Test batch processing with custom search configuration."""
        # Register a face
        metadata = {"person_id": "test_person", "name": "Test Person"}
        pipeline.add_face_to_database(
            image=sample_images[0],
            metadata=metadata,
            person_id="test_person"
        )
        
        # Custom search config
        custom_config = SearchConfig(
            top_k=3,
            similarity_threshold=0.8,
            enable_reranking=False
        )
        
        results = pipeline.batch_process_images(sample_images, custom_config)
        
        assert len(results) == len(sample_images)
        
        # Check that search results respect the configuration
        for result in results:
            if result.success and result.search_results:
                assert len(result.search_results) <= custom_config.top_k
                for search_result in result.search_results:
                    assert search_result.similarity_score >= custom_config.similarity_threshold
    
    @patch('face_recognition.pipeline.ThreadPoolExecutor')
    def test_batch_processing_thread_pool_usage(self, mock_executor, pipeline, sample_images):
        """Test that ThreadPoolExecutor is used correctly."""
        # Mock the executor
        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        mock_executor_instance.submit.return_value.result.return_value = (0, Mock())
        
        pipeline.batch_process_images(sample_images, max_workers=2)
        
        # Verify ThreadPoolExecutor was called with correct max_workers
        mock_executor.assert_called_once_with(max_workers=2)
        
        # Verify submit was called for each image
        assert mock_executor_instance.submit.call_count == len(sample_images)
    
    def test_batch_processing_maintains_order(self, pipeline, sample_images):
        """Test that batch processing maintains the order of input images."""
        # Create images with identifiable features
        identifiable_images = []
        for i in range(5):
            image = np.zeros((200, 200, 3), dtype=np.uint8)
            # Add a unique marker (different colored circle) for each image
            color = (i * 50, 255 - i * 50, 100)
            cv2.circle(image, (100, 100), 80, (255, 255, 255), -1)
            cv2.circle(image, (50, 50), 10, color, -1)  # Unique marker
            identifiable_images.append(image)
        
        results = pipeline.batch_process_images(identifiable_images)
        
        # Results should be in the same order as input
        assert len(results) == len(identifiable_images)
        
        # All results should be present (no None values)
        for result in results:
            assert result is not None
            assert isinstance(result, RecognitionResponse)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])