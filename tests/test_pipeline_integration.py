"""Integration tests for the FaceRecognitionPipeline."""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path

from face_recognition.pipeline import FaceRecognitionPipeline
from face_recognition.config.manager import ConfigurationManager
from face_recognition.models import RecognitionRequest, SearchConfig
from face_recognition.exceptions import FaceDetectionError, VectorDatabaseError, InvalidImageError


class TestFaceRecognitionPipelineIntegration:
    """Integration tests for the complete face recognition pipeline."""
    
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
    def sample_face_image(self):
        """Create a sample face image for testing."""
        # Create a simple synthetic face-like image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add some basic face-like features
        cv2.circle(image, (100, 100), 80, (255, 255, 255), -1)  # Face
        cv2.circle(image, (80, 80), 10, (0, 0, 0), -1)   # Left eye
        cv2.circle(image, (120, 80), 10, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(image, (100, 120), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        return image
    
    @pytest.fixture
    def real_face_images(self):
        """Load real face images if available."""
        images = []
        test_images = [
            "WIN_20250222_15_21_37_Pro.jpg",
            "MTI230279.jpg"
        ]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                image = cv2.imread(img_path)
                if image is not None:
                    images.append(image)
        
        return images
    
    def test_pipeline_initialization(self, pipeline):
        """Test that the pipeline initializes correctly."""
        assert pipeline is not None
        assert pipeline.face_detector is not None
        assert pipeline.embedding_extractor is not None
        assert pipeline.reranker is not None
        assert pipeline.index is not None
        assert pipeline.config is not None
    
    def test_face_registration_workflow(self, pipeline, sample_face_image):
        """Test the complete face registration workflow."""
        # Register a face
        metadata = {
            "person_id": "test_person_1",
            "name": "Test Person",
            "timestamp": "2025-01-23T10:00:00"
        }
        
        face_id = pipeline.add_face_to_database(
            image=sample_face_image,
            metadata=metadata,
            person_id="test_person_1"
        )
        
        assert face_id is not None
        assert isinstance(face_id, str)
        
        # Verify the face was added to the database
        db_info = pipeline.get_database_info()
        assert db_info['total_faces'] == 1
    
    def test_face_recognition_workflow(self, pipeline, sample_face_image):
        """Test the complete face recognition workflow."""
        # First register a face
        metadata = {
            "person_id": "test_person_1",
            "name": "Test Person"
        }
        
        face_id = pipeline.add_face_to_database(
            image=sample_face_image,
            metadata=metadata,
            person_id="test_person_1"
        )
        
        # Create recognition request
        request = RecognitionRequest(
            image_data=sample_face_image,
            search_config=SearchConfig(
                top_k=5,
                similarity_threshold=0.5
            )
        )
        
        # Perform recognition
        response = pipeline.recognize_face(request)
        
        assert response is not None
        assert response.success is True
        assert len(response.detected_faces) > 0
        assert response.processing_time_ms > 0
    
    def test_batch_processing_workflow(self, pipeline, sample_face_image):
        """Test batch processing of multiple images."""
        # Register a face first
        metadata = {
            "person_id": "test_person_1",
            "name": "Test Person"
        }
        
        pipeline.add_face_to_database(
            image=sample_face_image,
            metadata=metadata,
            person_id="test_person_1"
        )
        
        # Create multiple test images
        images = [sample_face_image, sample_face_image.copy()]
        
        # Process batch
        results = pipeline.batch_process_images(images)
        
        assert len(results) == 2
        for result in results:
            assert result.success is True
            assert result.processing_time_ms > 0
    
    def test_database_persistence(self, config_manager, temp_db_path, sample_face_image):
        """Test that database changes persist across pipeline instances."""
        # Create first pipeline instance and add a face
        pipeline1 = FaceRecognitionPipeline(
            config_manager=config_manager,
            db_path=temp_db_path
        )
        
        metadata = {
            "person_id": "test_person_1",
            "name": "Test Person"
        }
        
        face_id = pipeline1.add_face_to_database(
            image=sample_face_image,
            metadata=metadata,
            person_id="test_person_1"
        )
        
        # Create second pipeline instance
        pipeline2 = FaceRecognitionPipeline(
            config_manager=config_manager,
            db_path=temp_db_path
        )
        
        # Verify the face is still in the database
        db_info = pipeline2.get_database_info()
        assert db_info['total_faces'] == 1
    
    def test_error_handling_invalid_image(self, pipeline):
        """Test error handling with invalid image data."""
        # Test with None image
        with pytest.raises(InvalidImageError):
            pipeline.add_face_to_database(
                image=None,
                metadata={"person_id": "test"},
                person_id="test"
            )
        
        # Test with invalid image array
        invalid_image = np.array([1, 2, 3])
        with pytest.raises(InvalidImageError):
            pipeline.add_face_to_database(
                image=invalid_image,
                metadata={"person_id": "test"},
                person_id="test"
            )
    
    def test_configuration_profiles(self, temp_db_path):
        """Test pipeline with different configuration profiles."""
        from face_recognition.config.settings import ConfigurationProfiles
        
        # Test with high accuracy profile
        config_manager = ConfigurationManager()
        config_manager.config = ConfigurationProfiles.high_accuracy()
        
        pipeline = FaceRecognitionPipeline(
            config_manager=config_manager,
            db_path=temp_db_path
        )
        
        assert pipeline.config.reranking.quality_weight == 0.3
        assert pipeline.config.reranking.similarity_weight == 0.5
    
    def test_statistics_tracking(self, pipeline, sample_face_image):
        """Test that pipeline statistics are tracked correctly."""
        initial_stats = pipeline.stats.copy()
        
        # Perform some operations
        metadata = {"person_id": "test_person_1", "name": "Test Person"}
        pipeline.add_face_to_database(
            image=sample_face_image,
            metadata=metadata,
            person_id="test_person_1"
        )
        
        request = RecognitionRequest(
            image_data=sample_face_image,
            search_config=SearchConfig(top_k=5, similarity_threshold=0.5)
        )
        pipeline.recognize_face(request)
        
        # Check that statistics were updated
        final_stats = pipeline.stats
        assert final_stats['total_registrations'] > initial_stats['total_registrations']
        assert final_stats['total_recognitions'] > initial_stats['total_recognitions']
        assert final_stats['average_processing_time'] > 0
    
    def test_database_operations(self, pipeline, sample_face_image):
        """Test database operations like clear and info."""
        # Add some faces
        for i in range(3):
            metadata = {
                "person_id": f"test_person_{i}",
                "name": f"Test Person {i}"
            }
            pipeline.add_face_to_database(
                image=sample_face_image,
                metadata=metadata,
                person_id=f"test_person_{i}"
            )
        
        # Check database info
        db_info = pipeline.get_database_info()
        assert db_info['total_faces'] == 3
        
        # Clear database
        pipeline.clear_database()
        
        # Verify database is empty
        db_info = pipeline.get_database_info()
        assert db_info['total_faces'] == 0
    
    @pytest.mark.skipif(not os.path.exists("WIN_20250222_15_21_37_Pro.jpg"), 
                       reason="Real test images not available")
    def test_real_image_processing(self, pipeline, real_face_images):
        """Test pipeline with real face images."""
        if not real_face_images:
            pytest.skip("No real face images available")
        
        # Register first image
        metadata = {
            "person_id": "real_person_1",
            "name": "Real Person"
        }
        
        face_id = pipeline.add_face_to_database(
            image=real_face_images[0],
            metadata=metadata,
            person_id="real_person_1"
        )
        
        assert face_id is not None
        
        # Try to recognize with second image (if available)
        if len(real_face_images) > 1:
            request = RecognitionRequest(
                image_data=real_face_images[1],
                search_config=SearchConfig(
                    top_k=5,
                    similarity_threshold=0.3
                )
            )
            
            response = pipeline.recognize_face(request)
            assert response.success is True
            assert len(response.detected_faces) > 0


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])