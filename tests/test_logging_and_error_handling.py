"""Unit tests for logging and error handling functionality."""

import pytest
import time
import tempfile
import os
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from face_recognition.logging import (
    FaceRecognitionLogger, 
    ErrorHandler, 
    ErrorRecoveryManager,
    PerformanceMonitor
)
from face_recognition.logging.error_handler import RetryConfig, CircuitBreaker
from face_recognition.config.settings import LoggingConfig
from face_recognition.exceptions import FaceDetectionError, InvalidImageError


class TestFaceRecognitionLogger:
    """Test face recognition logger functionality."""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def logger_config(self, temp_log_file):
        """Create logger configuration."""
        return LoggingConfig(
            log_level="DEBUG",
            log_format="json",
            log_file=temp_log_file
        )
    
    @pytest.fixture
    def logger(self, logger_config):
        """Create logger instance."""
        return FaceRecognitionLogger("test_logger", logger_config)
    
    def test_logger_initialization(self, logger):
        """Test logger initialization."""
        assert logger.name == "test_logger"
        assert logger.logger is not None
        assert len(logger.logger.handlers) > 0
    
    def test_basic_logging_methods(self, logger):
        """Test basic logging methods."""
        # Test all log levels
        logger.debug("Debug message", extra_field="debug_value")
        logger.info("Info message", extra_field="info_value")
        logger.warning("Warning message", extra_field="warning_value")
        logger.error("Error message", extra_field="error_value")
        logger.critical("Critical message", extra_field="critical_value")
        
        # Should not raise any exceptions
        assert True
    
    def test_error_logging_with_exception(self, logger):
        """Test error logging with exception details."""
        test_exception = ValueError("Test error")
        
        logger.error("Test error occurred", exception=test_exception, context="test")
        
        # Check error count tracking
        assert "ValueError" in logger.error_counts
        assert logger.error_counts["ValueError"] == 1
    
    def test_performance_context(self, logger):
        """Test performance context manager."""
        with logger.performance_context("test_operation", param1="value1") as operation_id:
            assert operation_id is not None
            time.sleep(0.1)  # Simulate work
        
        # Check performance metrics
        assert "test_operation" in logger.performance_metrics
        assert len(logger.performance_metrics["test_operation"]) == 1
        assert logger.performance_metrics["test_operation"][0] >= 0.1
    
    def test_performance_context_with_exception(self, logger):
        """Test performance context with exception."""
        with pytest.raises(ValueError):
            with logger.performance_context("failing_operation") as operation_id:
                raise ValueError("Test failure")
        
        # Should still track the operation
        assert "failing_operation" in logger.performance_metrics
    
    def test_specialized_logging_methods(self, logger):
        """Test specialized logging methods."""
        # Face detection logging
        logger.log_face_detection(
            image_info={'width': 640, 'height': 480},
            detected_faces=2,
            processing_time=150.5,
            success=True
        )
        
        # Embedding extraction logging
        logger.log_embedding_extraction(
            face_count=2,
            embedding_dim=512,
            processing_time=75.2,
            model_version="v1.0",
            success=True
        )
        
        # Similarity search logging
        logger.log_similarity_search(
            query_embedding_dim=512,
            database_size=1000,
            top_k=10,
            results_found=5,
            processing_time=25.8,
            success=True
        )
        
        # Should not raise exceptions
        assert True
    
    def test_get_performance_summary(self, logger):
        """Test performance summary generation."""
        # Add some performance data
        with logger.performance_context("operation1"):
            time.sleep(0.05)
        
        with logger.performance_context("operation2"):
            time.sleep(0.03)
        
        summary = logger.get_performance_summary()
        
        assert 'operation1' in summary
        assert 'operation2' in summary
        assert summary['operation1']['count'] == 1
        assert summary['operation2']['count'] == 1
        assert 'error_counts' in summary
    
    def test_reset_metrics(self, logger):
        """Test metrics reset functionality."""
        # Add some data
        logger.error("Test error", exception=ValueError("test"))
        with logger.performance_context("test_op"):
            pass
        
        # Verify data exists
        assert len(logger.error_counts) > 0
        assert len(logger.performance_metrics) > 0
        
        # Reset and verify cleared
        logger.reset_metrics()
        assert len(logger.error_counts) == 0
        assert len(logger.performance_metrics) == 0


class TestErrorHandler:
    """Test error handler functionality."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler instance."""
        return ErrorHandler()
    
    def test_handle_error(self, error_handler):
        """Test error handling."""
        test_error = FaceDetectionError("Test detection error")
        
        error_context = error_handler.handle_error(
            test_error, 
            "face_detection",
            {"image_size": "640x480"}
        )
        
        assert error_context.operation == "face_detection"
        assert error_context.error_type == "FaceDetectionError"
        assert error_context.error_message == "Test detection error"
        assert error_context.context_data["image_size"] == "640x480"
        
        # Check error tracking
        assert len(error_handler.error_history) == 1
        assert "FaceDetectionError" in error_handler.error_counts
        assert error_handler.error_counts["FaceDetectionError"] == 1
    
    def test_should_retry_logic(self, error_handler):
        """Test retry decision logic."""
        retry_config = RetryConfig(max_retries=3)
        
        # Should retry transient errors
        transient_error = ConnectionError("Network error")
        assert error_handler.should_retry(transient_error, "test_op", 0, retry_config) is True
        assert error_handler.should_retry(transient_error, "test_op", 2, retry_config) is True
        assert error_handler.should_retry(transient_error, "test_op", 3, retry_config) is False
        
        # Should not retry certain error types
        config_error = InvalidImageError("Invalid image")
        assert error_handler.should_retry(config_error, "test_op", 0, retry_config) is False
    
    def test_calculate_delay(self, error_handler):
        """Test retry delay calculation."""
        retry_config = RetryConfig(
            base_delay=1.0,
            max_delay=10.0,
            exponential_backoff=True,
            jitter=False
        )
        
        # Test exponential backoff
        delay1 = error_handler.calculate_delay(0, retry_config)
        delay2 = error_handler.calculate_delay(1, retry_config)
        delay3 = error_handler.calculate_delay(2, retry_config)
        
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0
        
        # Test max delay cap
        delay_high = error_handler.calculate_delay(10, retry_config)
        assert delay_high == retry_config.max_delay
    
    def test_get_error_summary(self, error_handler):
        """Test error summary generation."""
        # Add some errors
        error_handler.handle_error(ValueError("Error 1"), "op1")
        error_handler.handle_error(ValueError("Error 2"), "op1")
        error_handler.handle_error(TypeError("Error 3"), "op2")
        
        summary = error_handler.get_error_summary()
        
        assert summary['total_errors'] == 3
        assert summary['error_counts']['ValueError'] == 2
        assert summary['error_counts']['TypeError'] == 1
        assert 'recent_errors_1h' in summary
        assert 'circuit_breaker_status' in summary


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker instance."""
        return CircuitBreaker("test_breaker", failure_threshold=3, recovery_timeout=1)
    
    def test_circuit_breaker_closed_state(self, circuit_breaker):
        """Test circuit breaker in closed state."""
        def successful_function():
            return "success"
        
        result = circuit_breaker.call(successful_function)
        assert result == "success"
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0
    
    def test_circuit_breaker_opening(self, circuit_breaker):
        """Test circuit breaker opening after failures."""
        def failing_function():
            raise ValueError("Test failure")
        
        # Cause failures to open circuit
        for i in range(3):
            with pytest.raises(ValueError):
                circuit_breaker.call(failing_function)
        
        assert circuit_breaker.state == "OPEN"
        assert circuit_breaker.failure_count == 3
        
        # Should raise exception when circuit is open
        with pytest.raises(Exception, match="Circuit breaker.*is OPEN"):
            circuit_breaker.call(failing_function)
    
    def test_circuit_breaker_recovery(self, circuit_breaker):
        """Test circuit breaker recovery."""
        def failing_function():
            raise ValueError("Test failure")
        
        def successful_function():
            return "success"
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(ValueError):
                circuit_breaker.call(failing_function)
        
        assert circuit_breaker.state == "OPEN"
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Should attempt recovery and succeed
        result = circuit_breaker.call(successful_function)
        assert result == "success"
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0
    
    def test_get_status(self, circuit_breaker):
        """Test circuit breaker status reporting."""
        status = circuit_breaker.get_status()
        
        assert 'state' in status
        assert 'failure_count' in status
        assert 'failure_threshold' in status
        assert 'last_failure_time' in status
        
        assert status['state'] == 'CLOSED'
        assert status['failure_count'] == 0


class TestErrorRecoveryManager:
    """Test error recovery manager functionality."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler instance."""
        return ErrorHandler()
    
    @pytest.fixture
    def recovery_manager(self, error_handler):
        """Create error recovery manager instance."""
        return ErrorRecoveryManager(error_handler)
    
    def test_register_fallback(self, recovery_manager):
        """Test fallback strategy registration."""
        def fallback_func():
            return "fallback_result"
        
        recovery_manager.register_fallback("test_operation", fallback_func)
        
        assert "test_operation" in recovery_manager.fallback_strategies
        assert recovery_manager.fallback_strategies["test_operation"] == fallback_func
    
    def test_with_retry_and_fallback_success(self, recovery_manager):
        """Test retry decorator with successful operation."""
        call_count = 0
        
        @recovery_manager.with_retry_and_fallback("test_op")
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = test_function()
        
        assert result == "success"
        assert call_count == 1
    
    def test_with_retry_and_fallback_with_retries(self, recovery_manager):
        """Test retry decorator with retries."""
        call_count = 0
        
        @recovery_manager.with_retry_and_fallback("test_op", RetryConfig(max_retries=2, base_delay=0.01))
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = test_function()
        
        assert result == "success"
        assert call_count == 3
    
    def test_with_retry_and_fallback_with_fallback(self, recovery_manager):
        """Test retry decorator with fallback."""
        def fallback_func():
            return "fallback_result"
        
        recovery_manager.register_fallback("test_op", fallback_func)
        
        @recovery_manager.with_retry_and_fallback("test_op", RetryConfig(max_retries=1, base_delay=0.01))
        def test_function():
            raise ConnectionError("Persistent failure")
        
        result = test_function()
        
        assert result == "fallback_result"
    
    def test_graceful_degradation(self, recovery_manager):
        """Test graceful degradation functionality."""
        def primary_func():
            raise ValueError("Primary failed")
        
        def fallback_func():
            return "fallback_success"
        
        result = recovery_manager.graceful_degradation(
            primary_func, fallback_func, "test_operation"
        )
        
        assert result == "fallback_success"


class TestPerformanceMonitor:
    """Test performance monitor functionality."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor instance."""
        # Use shorter interval for testing
        monitor = PerformanceMonitor(
            max_metrics_history=100,
            system_monitoring_interval=0.1
        )
        yield monitor
        monitor.stop_system_monitoring()
    
    def test_measure_operation_context(self, performance_monitor):
        """Test operation measurement context manager."""
        with performance_monitor.measure_operation("test_operation", param="value") as metric:
            time.sleep(0.05)
            assert metric.operation == "test_operation"
            assert metric.context_data["param"] == "value"
        
        # Check that metric was recorded
        assert len(performance_monitor.metrics_history) == 1
        recorded_metric = performance_monitor.metrics_history[0]
        assert recorded_metric.operation == "test_operation"
        assert recorded_metric.duration_ms >= 50  # At least 50ms
    
    def test_record_metric_manually(self, performance_monitor):
        """Test manual metric recording."""
        performance_monitor.record_metric(
            "manual_operation",
            duration_ms=123.45,
            custom_field="test_value"
        )
        
        assert len(performance_monitor.metrics_history) == 1
        metric = performance_monitor.metrics_history[0]
        assert metric.operation == "manual_operation"
        assert metric.duration_ms == 123.45
        assert metric.context_data["custom_field"] == "test_value"
    
    def test_get_operation_stats(self, performance_monitor):
        """Test operation statistics generation."""
        # Record multiple operations
        for i in range(5):
            performance_monitor.record_metric(
                "test_operation",
                duration_ms=100 + i * 10  # 100, 110, 120, 130, 140
            )
        
        stats = performance_monitor.get_operation_stats("test_operation")
        
        assert stats['operation'] == "test_operation"
        assert stats['count'] == 5
        assert stats['duration_stats']['mean_ms'] == 120.0
        assert stats['duration_stats']['min_ms'] == 100.0
        assert stats['duration_stats']['max_ms'] == 140.0
        assert 'memory_stats' in stats
        assert 'cpu_stats' in stats
    
    def test_get_operation_stats_with_time_window(self, performance_monitor):
        """Test operation statistics with time window."""
        # Record old metric
        old_metric = performance_monitor.metrics_history.append(
            type('Metric', (), {
                'operation': 'test_operation',
                'timestamp': datetime.now() - timedelta(hours=2),
                'duration_ms': 50.0,
                'memory_usage_mb': 100.0,
                'cpu_percent': 10.0,
                'context_data': {}
            })()
        )
        
        # Record recent metric
        performance_monitor.record_metric("test_operation", duration_ms=150.0)
        
        # Get stats for last hour (should only include recent metric)
        stats = performance_monitor.get_operation_stats("test_operation", time_window_hours=1.0)
        
        assert stats['count'] == 1
        assert stats['duration_stats']['mean_ms'] == 150.0
    
    def test_get_performance_summary(self, performance_monitor):
        """Test performance summary generation."""
        # Add some metrics
        performance_monitor.record_metric("operation1", duration_ms=100.0)
        performance_monitor.record_metric("operation1", duration_ms=200.0)
        performance_monitor.record_metric("operation2", duration_ms=50.0)
        
        summary = performance_monitor.get_performance_summary()
        
        assert summary['total_operations'] == 3
        assert summary['unique_operations'] == 2
        assert 'operation1' in summary['operations']
        assert 'operation2' in summary['operations']
        assert summary['operations']['operation1']['count'] == 2
        assert summary['operations']['operation2']['count'] == 1
        assert 'system_stats' in summary
    
    def test_identify_bottlenecks(self, performance_monitor):
        """Test bottleneck identification."""
        # Add slow operations
        for i in range(15):
            performance_monitor.record_metric("slow_operation", duration_ms=1500.0)
        
        # Add fast operations
        for i in range(15):
            performance_monitor.record_metric("fast_operation", duration_ms=50.0)
        
        # Add inconsistent operations
        for i in range(15):
            duration = 100.0 if i % 2 == 0 else 500.0  # High variance
            performance_monitor.record_metric("inconsistent_operation", duration_ms=duration)
        
        bottlenecks = performance_monitor.identify_bottlenecks(
            min_operations=10,
            slow_threshold_ms=1000.0
        )
        
        # Should identify slow operation as bottleneck
        slow_bottlenecks = [b for b in bottlenecks if b['operation'] == 'slow_operation']
        assert len(slow_bottlenecks) > 0
        assert slow_bottlenecks[0]['type'] == 'consistently_slow'
        
        # Should identify inconsistent operation
        inconsistent_bottlenecks = [b for b in bottlenecks if b['operation'] == 'inconsistent_operation']
        assert len(inconsistent_bottlenecks) > 0
        assert inconsistent_bottlenecks[0]['type'] == 'high_variance'
    
    def test_clear_metrics(self, performance_monitor):
        """Test metrics clearing functionality."""
        # Add some metrics
        performance_monitor.record_metric("test_operation", duration_ms=100.0)
        
        # Add old metric manually
        old_metric = type('Metric', (), {
            'operation': 'old_operation',
            'timestamp': datetime.now() - timedelta(hours=25),
            'duration_ms': 50.0,
            'memory_usage_mb': 100.0,
            'cpu_percent': 10.0,
            'context_data': {}
        })()
        performance_monitor.metrics_history.append(old_metric)
        
        assert len(performance_monitor.metrics_history) == 2
        
        # Clear old metrics
        performance_monitor.clear_metrics(older_than_hours=24)
        
        # Should only have recent metric
        assert len(performance_monitor.metrics_history) == 1
        assert performance_monitor.metrics_history[0].operation == "test_operation"
    
    def test_system_monitoring(self, performance_monitor):
        """Test system monitoring functionality."""
        # Wait a bit for system monitoring to collect data
        time.sleep(0.2)
        
        # Should have collected some system metrics
        assert len(performance_monitor.system_metrics_history) > 0
        
        # Get system stats
        system_stats = performance_monitor.get_system_stats()
        
        assert 'system_cpu' in system_stats
        assert 'system_memory' in system_stats
        assert 'process_memory' in system_stats
        assert 'process_cpu' in system_stats
        assert system_stats['count'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])