"""Integration tests for the Face Recognition API."""

import pytest
import numpy as np
import cv2
import base64
import tempfile
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Import the FastAPI app
from face_recognition.api.app import create_app


class TestFaceRecognitionAPI:
    """Test the Face Recognition API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def sample_image_base64(self):
        """Create a sample image encoded as base64."""
        # Create test image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(image, (100, 100), 30, (128, 128, 128), -1)
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64
    
    @pytest.fixture
    def temp_image_file(self):
        """Create a temporary image file."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            cv2.imwrite(f.name, image)
            yield f.name
        
        import os
        os.unlink(f.name)
    
    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Face Recognition API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "healthy"
    
    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data
        assert "database_status" in data
        assert "components_status" in data
    
    def test_database_info(self, client):
        """Test the database info endpoint."""
        response = client.get("/api/v1/database/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_faces" in data
        assert "database_path" in data
        assert "index_type" in data
        assert "distance_metric" in data
        assert "embedding_dimension" in data
        assert "statistics" in data
    
    def test_recognize_face_with_base64(self, client, sample_image_base64):
        """Test face recognition with base64 image."""
        request_data = {
            "image": {
                "image_base64": sample_image_base64
            },
            "search_config": {
                "top_k": 5,
                "similarity_threshold": 0.7,
                "enable_reranking": True,
                "distance_metric": "cosine"
            }
        }
        
        response = client.post("/api/v1/recognize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "detected_faces" in data
        assert "search_results" in data
        assert "processing_time_ms" in data
        assert "quality_info" in data
        
        # Should be successful (even if no faces detected)
        assert data["success"] is True
        assert isinstance(data["detected_faces"], list)
        assert isinstance(data["search_results"], list)
        assert data["processing_time_ms"] > 0
    
    def test_recognize_face_with_file_path(self, client, temp_image_file):
        """Test face recognition with file path."""
        request_data = {
            "image": {
                "image_path": temp_image_file
            },
            "search_config": {
                "top_k": 3,
                "similarity_threshold": 0.5
            }
        }
        
        response = client.post("/api/v1/recognize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
    
    def test_register_face(self, client, sample_image_base64):
        """Test face registration."""
        request_data = {
            "image": {
                "image_base64": sample_image_base64
            },
            "person_id": "test_person_1",
            "metadata": {
                "name": "Test Person",
                "department": "Engineering",
                "registration_date": "2025-01-23"
            }
        }
        
        response = client.post("/api/v1/register", json=request_data)
        
        # Should succeed or fail gracefully
        assert response.status_code in [200, 422]  # 422 if no face detected
        
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "embedding_id" in data
            assert "processing_time_ms" in data
            assert "quality_info" in data
            
            if data["success"]:
                assert data["embedding_id"] is not None
                assert data["processing_time_ms"] > 0
    
    def test_batch_recognize(self, client, sample_image_base64):
        """Test batch face recognition."""
        # Create multiple images for batch processing
        images = []
        for i in range(3):
            images.append({
                "image_base64": sample_image_base64
            })
        
        request_data = {
            "images": images,
            "search_config": {
                "top_k": 5,
                "similarity_threshold": 0.6
            },
            "max_workers": 2
        }
        
        response = client.post("/api/v1/batch/recognize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "results" in data
        assert "summary" in data
        assert "total_processing_time_ms" in data
        
        # Should have results for all images
        assert len(data["results"]) == 3
        
        # Check summary structure
        summary = data["summary"]
        assert "total_processed" in summary
        assert "successful" in summary
        assert "failed" in summary
        assert "success_rate" in summary
    
    def test_batch_register(self, client, sample_image_base64):
        """Test batch face registration."""
        registrations = []
        for i in range(2):
            registrations.append({
                "image": {
                    "image_base64": sample_image_base64
                },
                "person_id": f"batch_person_{i}",
                "metadata": {
                    "name": f"Batch Person {i}",
                    "batch_id": "test_batch_1"
                }
            })
        
        request_data = {
            "registrations": registrations,
            "max_workers": 2
        }
        
        response = client.post("/api/v1/batch/register", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "results" in data
        assert "summary" in data
        assert "total_processing_time_ms" in data
        
        # Should have results for all registrations
        assert len(data["results"]) == 2
    
    def test_performance_metrics(self, client):
        """Test performance metrics endpoint."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "pipeline_stats" in data
        assert "performance_summary" in data
        assert "error_summary" in data
        assert "system_metrics" in data
        assert "bottlenecks" in data
    
    def test_optimization_recommendations(self, client):
        """Test optimization recommendations endpoint."""
        response = client.get("/api/v1/optimize")
        assert response.status_code == 200
        
        data = response.json()
        assert "analysis_timestamp" in data
        assert "recommendations" in data
        assert "metrics_summary" in data
        
        # Recommendations should be a list
        assert isinstance(data["recommendations"], list)
    
    def test_benchmark(self, client, sample_image_base64):
        """Test performance benchmark endpoint."""
        request_data = {
            "test_image": {
                "image_base64": sample_image_base64
            },
            "num_iterations": 2  # Small number for testing
        }
        
        response = client.post("/api/v1/benchmark", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "test_config" in data
        assert "results" in data
        assert "system_resources" in data
        
        # Test config should match request
        assert data["test_config"]["num_iterations"] == 2
    
    def test_clear_database(self, client):
        """Test database clear endpoint."""
        response = client.delete("/api/v1/database/clear")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "successfully" in data["message"].lower()
    
    def test_invalid_image_data(self, client):
        """Test handling of invalid image data."""
        request_data = {
            "image": {
                "image_base64": "invalid_base64_data"
            }
        }
        
        response = client.post("/api/v1/recognize", json=request_data)
        assert response.status_code == 400  # Bad request for invalid image
    
    def test_missing_image_data(self, client):
        """Test handling of missing image data."""
        request_data = {
            "image": {},  # No image_base64 or image_path
            "search_config": {
                "top_k": 5
            }
        }
        
        response = client.post("/api/v1/recognize", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_search_config(self, client, sample_image_base64):
        """Test handling of invalid search configuration."""
        request_data = {
            "image": {
                "image_base64": sample_image_base64
            },
            "search_config": {
                "top_k": -1,  # Invalid value
                "similarity_threshold": 1.5  # Invalid value
            }
        }
        
        response = client.post("/api/v1/recognize", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_api_documentation(self, client):
        """Test that API documentation is accessible."""
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Check that our endpoints are documented
        paths = schema["paths"]
        assert "/api/v1/recognize" in paths
        assert "/api/v1/register" in paths
        assert "/api/v1/health" in paths
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/v1/health")
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
    
    def test_process_time_header(self, client):
        """Test that process time header is added."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        # Should have process time header
        assert "x-process-time" in response.headers
        
        # Process time should be a valid float
        process_time = float(response.headers["x-process-time"])
        assert process_time > 0
    
    def test_gzip_compression(self, client):
        """Test that gzip compression is working."""
        # Make request with gzip acceptance
        headers = {"Accept-Encoding": "gzip"}
        response = client.get("/api/v1/metrics", headers=headers)
        
        assert response.status_code == 200
        # FastAPI test client automatically decompresses, so we just check it works
        assert len(response.content) > 0
    
    def test_end_to_end_workflow(self, client, sample_image_base64):
        """Test complete end-to-end API workflow."""
        # Step 1: Check initial database state
        db_response = client.get("/api/v1/database/info")
        initial_faces = db_response.json()["total_faces"]
        
        # Step 2: Register a face
        register_request = {
            "image": {"image_base64": sample_image_base64},
            "person_id": "e2e_test_person",
            "metadata": {"name": "E2E Test Person", "test": "end_to_end"}
        }
        
        register_response = client.post("/api/v1/register", json=register_request)
        
        # Step 3: Check database was updated (if registration succeeded)
        if register_response.status_code == 200 and register_response.json().get("success"):
            db_response = client.get("/api/v1/database/info")
            final_faces = db_response.json()["total_faces"]
            assert final_faces > initial_faces
        
        # Step 4: Try recognition
        recognize_request = {
            "image": {"image_base64": sample_image_base64},
            "search_config": {"top_k": 5, "similarity_threshold": 0.3}
        }
        
        recognize_response = client.post("/api/v1/recognize", json=recognize_request)
        assert recognize_response.status_code == 200
        
        # Step 5: Check metrics
        metrics_response = client.get("/api/v1/metrics")
        assert metrics_response.status_code == 200
        
        metrics = metrics_response.json()
        pipeline_stats = metrics["pipeline_stats"]
        
        # Should have recorded some operations
        assert (pipeline_stats["total_registrations"] > 0 or 
                pipeline_stats["total_recognitions"] > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])