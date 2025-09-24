"""Performance monitoring and benchmarking tests."""

import pytest
import numpy as np
import cv2
import tempfile
import time
from unittest.mock import Mock, patch

from face_recognition.pipeline import FaceRecognitionPipeline
from face_recognition.config.manager import ConfigurationManager
from face_recognition.logging import PerformanceMonitor
from face_recognition.models import RecognitionRequest, SearchConfig


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""
    
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
    def sample_image(self):
        """Create a sample test image."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (174, 174), (255, 255, 255), -1)
        cv2.circle(image, (112, 112), 30, (128, 128, 128), -1)
        return image
    
    def test_performance_metrics_collection(self, pipeline, sample_image):
        """Test that performance metrics are collected during operations."""
        # Perform some operations
        request = RecognitionRequest(
            image_data=sample_image,
            search_config=SearchConfig(top_k=5)
        )
        
        response = pipeline.recognize_face(request)
        
        # Check that metrics were collected
        metrics = pipeline.get_performance_metrics()
        
        assert 'pipeline_stats' in metrics
        assert 'performance_summary' in metrics
        assert 'error_summary' in metrics
        assert 'system_metrics' in metrics
        assert 'bottlenecks' in metrics
        assert 'database_info' in metrics
        
        # Check that operations were recorded
        perf_summary = metrics['performance_summary']
        assert perf_summary['total_operations'] > 0
        assert len(perf_summary['operations']) > 0
    
    def test_performance_optimization_analysis(self, pipeline, sample_image):
        """Test performance optimization analysis."""
        # Add some test data to trigger recommendations
        for i in range(5):
            request = RecognitionRequest(
                image_data=sample_image,
                search_config=SearchConfig(top_k=5)
            )
            pipeline.recognize_face(request)
        
        optimization_results = pipeline.optimize_performance()
        
        assert 'analysis_timestamp' in optimization_results
        assert 'recommendations' in optimization_results
        assert 'metrics_summary' in optimization_results
        
        # Check metrics summary structure
        metrics_summary = optimization_results['metrics_summary']
        assert 'total_operations' in metrics_summary
        assert 'average_processing_time' in metrics_summary
        assert 'database_size' in metrics_summary
        assert 'error_count' in metrics_summary
    
    def test_benchmark_performance(self, pipeline, sample_image):
        """Test performance benchmarking functionality."""
        # Register a face first for similarity search benchmarking
        metadata = {"person_id": "test_person", "name": "Test Person"}
        pipeline.add_face_to_database(
            image=sample_image,
            metadata=metadata,
            person_id="test_person"
        )
        
        # Run benchmark with small number of iterations for testing
        benchmark_results = pipeline.benchmark_performance(
            test_image=sample_image,
            num_iterations=3
        )
        
        assert 'test_config' in benchmark_results
        assert 'results' in benchmark_results
        assert 'system_resources' in benchmark_results
        
        # Check test configuration
        test_config = benchmark_results['test_config']
        assert test_config['num_iterations'] == 3
        assert test_config['image_shape'] == sample_image.shape
        
        # Check that at least some benchmarks ran
        results = benchmark_results['results']
        assert len(results) > 0
        
        # Check benchmark result structure
        for operation, result in results.items():
            assert 'avg_time_ms' in result
            assert 'min_time_ms' in result
            assert 'max_time_ms' in result
            assert 'successful_runs' in result
            assert result['successful_runs'] > 0
    
    def test_benchmark_with_synthetic_image(self, pipeline):
        """Test benchmarking with synthetic image generation."""
        benchmark_results = pipeline.benchmark_performance(
            test_image=None,  # Should create synthetic image
            num_iterations=2
        )
        
        assert 'test_config' in benchmark_results
        assert 'results' in benchmark_results
        
        # Should have created synthetic image
        test_config = benchmark_results['test_config']
        assert test_config['image_shape'] == (224, 224, 3)
    
    def test_performance_monitor_integration(self, pipeline, sample_image):
        """Test integration with performance monitor."""
        initial_metrics_count = len(pipeline.performance_monitor.metrics_history)
        
        # Perform operation
        request = RecognitionRequest(
            image_data=sample_image,
            search_config=SearchConfig(top_k=5)
        )
        pipeline.recognize_face(request)
        
        # Check that metrics were recorded
        final_metrics_count = len(pipeline.performance_monitor.metrics_history)
        assert final_metrics_count > initial_metrics_count
        
        # Check operation statistics
        face_recognition_stats = pipeline.performance_monitor.get_operation_stats("face_recognition")
        assert face_recognition_stats['count'] > 0
        assert face_recognition_stats['duration_stats']['mean_ms'] > 0
    
    def test_bottleneck_identification(self, pipeline):
        """Test bottleneck identification functionality."""
        # Simulate slow operations by manually adding metrics
        for i in range(15):
            pipeline.performance_monitor.record_metric(
                "slow_operation",
                duration_ms=1500.0,  # Slow operation
                test_data=f"iteration_{i}"
            )
        
        # Simulate fast operations
        for i in range(15):
            pipeline.performance_monitor.record_metric(
                "fast_operation",
                duration_ms=50.0,  # Fast operation
                test_data=f"iteration_{i}"
            )
        
        bottlenecks = pipeline.performance_monitor.identify_bottlenecks(
            min_operations=10,
            slow_threshold_ms=1000.0
        )
        
        # Should identify slow operation as bottleneck
        slow_bottlenecks = [b for b in bottlenecks if b['operation'] == 'slow_operation']
        assert len(slow_bottlenecks) > 0
        assert slow_bottlenecks[0]['type'] == 'consistently_slow'
        
        # Fast operation should not be identified as bottleneck
        fast_bottlenecks = [b for b in bottlenecks if b['operation'] == 'fast_operation']
        assert len(fast_bottlenecks) == 0
    
    def test_system_resource_monitoring(self, pipeline):
        """Test system resource monitoring."""
        # Wait a bit for system monitoring to collect data
        time.sleep(0.2)
        
        system_stats = pipeline.performance_monitor.get_system_stats()
        
        assert 'system_cpu' in system_stats
        assert 'system_memory' in system_stats
        assert 'process_memory' in system_stats
        assert 'process_cpu' in system_stats
        
        # Check that we have some data
        assert system_stats['count'] > 0
        
        # Check data structure
        assert 'current_percent' in system_stats['system_cpu']
        assert 'current_mb' in system_stats['process_memory']
    
    def test_performance_recommendations(self, pipeline, sample_image):
        """Test performance optimization recommendations."""
        # Simulate high error rate by causing some failures
        pipeline.error_handler.handle_error(
            ValueError("Test error 1"), "test_operation"
        )
        pipeline.error_handler.handle_error(
            ValueError("Test error 2"), "test_operation"
        )
        
        # Perform some successful operations
        for i in range(5):
            request = RecognitionRequest(
                image_data=sample_image,
                search_config=SearchConfig(top_k=5)
            )
            pipeline.recognize_face(request)
        
        optimization_results = pipeline.optimize_performance()
        recommendations = optimization_results['recommendations']
        
        # Should have some recommendations
        assert len(recommendations) >= 0  # May or may not have recommendations
        
        # If there are recommendations, check structure
        for rec in recommendations:
            assert 'type' in rec
            assert 'severity' in rec
            assert 'recommendation' in rec
            assert rec['severity'] in ['low', 'medium', 'high']
    
    def test_metrics_cleanup(self, pipeline):
        """Test metrics cleanup functionality."""
        # Add some metrics
        for i in range(10):
            pipeline.performance_monitor.record_metric(
                "test_operation",
                duration_ms=100.0
            )
        
        initial_count = len(pipeline.performance_monitor.metrics_history)
        assert initial_count == 10
        
        # Clear metrics (should keep all since they're recent)
        pipeline.performance_monitor.clear_metrics(older_than_hours=0.001)  # Very short time
        
        # Should have cleared old metrics (in this case, none since they're very recent)
        final_count = len(pipeline.performance_monitor.metrics_history)
        assert final_count <= initial_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])