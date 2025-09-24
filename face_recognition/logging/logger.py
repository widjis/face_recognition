"""Comprehensive logging system for face recognition."""

import logging
import logging.handlers
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager

from ..config.settings import LoggingConfig


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class FaceRecognitionLogger:
    """
    Centralized logger for the face recognition system.
    
    Provides structured logging with performance metrics, error tracking,
    and contextual information.
    """
    
    def __init__(self, name: str, config: Optional[LoggingConfig] = None):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            config: Logging configuration
        """
        self.name = name
        self.config = config or LoggingConfig()
        self.logger = logging.getLogger(name)
        self._setup_logger()
        
        # Performance tracking
        self.performance_metrics = {}
        self.error_counts = {}
    
    def _setup_logger(self):
        """Setup logger with handlers and formatters."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if self.config.log_format == "json":
            console_formatter = JSONFormatter()
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.log_file:
            log_path = Path(self.config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with extra context."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with extra context."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with extra context."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with exception details."""
        if exception:
            kwargs['error_type'] = type(exception).__name__
            kwargs['error_message'] = str(exception)
            self.logger.error(message, exc_info=exception, extra=kwargs)
            
            # Track error counts
            error_type = type(exception).__name__
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        else:
            self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message with exception details."""
        if exception:
            kwargs['error_type'] = type(exception).__name__
            kwargs['error_message'] = str(exception)
            self.logger.critical(message, exc_info=exception, extra=kwargs)
        else:
            self.logger.critical(message, extra=kwargs)
    
    @contextmanager
    def performance_context(self, operation: str, **context):
        """Context manager for performance logging."""
        start_time = datetime.now()
        operation_id = f"{operation}_{start_time.timestamp()}"
        
        self.info(f"Starting operation: {operation}", 
                 operation=operation, 
                 operation_id=operation_id,
                 **context)
        
        try:
            yield operation_id
            
            # Success
            duration = (datetime.now() - start_time).total_seconds()
            self.info(f"Completed operation: {operation}", 
                     operation=operation,
                     operation_id=operation_id,
                     duration_seconds=duration,
                     status="success",
                     **context)
            
            # Track performance metrics
            if operation not in self.performance_metrics:
                self.performance_metrics[operation] = []
            self.performance_metrics[operation].append(duration)
            
        except Exception as e:
            # Failure
            duration = (datetime.now() - start_time).total_seconds()
            self.error(f"Failed operation: {operation}",
                      exception=e,
                      operation=operation,
                      operation_id=operation_id,
                      duration_seconds=duration,
                      status="failed",
                      **context)
            raise
    
    def log_face_detection(self, 
                          image_info: Dict,
                          detected_faces: int,
                          processing_time: float,
                          success: bool = True,
                          error: Optional[str] = None):
        """Log face detection operation."""
        log_data = {
            'operation': 'face_detection',
            'image_width': image_info.get('width'),
            'image_height': image_info.get('height'),
            'detected_faces': detected_faces,
            'processing_time_ms': processing_time,
            'success': success
        }
        
        if error:
            log_data['error'] = error
            self.error("Face detection failed", **log_data)
        else:
            self.info("Face detection completed", **log_data)
    
    def log_embedding_extraction(self,
                                face_count: int,
                                embedding_dim: int,
                                processing_time: float,
                                model_version: str,
                                success: bool = True,
                                error: Optional[str] = None):
        """Log embedding extraction operation."""
        log_data = {
            'operation': 'embedding_extraction',
            'face_count': face_count,
            'embedding_dimension': embedding_dim,
            'model_version': model_version,
            'processing_time_ms': processing_time,
            'success': success
        }
        
        if error:
            log_data['error'] = error
            self.error("Embedding extraction failed", **log_data)
        else:
            self.info("Embedding extraction completed", **log_data)
    
    def log_similarity_search(self,
                            query_embedding_dim: int,
                            database_size: int,
                            top_k: int,
                            results_found: int,
                            processing_time: float,
                            success: bool = True,
                            error: Optional[str] = None):
        """Log similarity search operation."""
        log_data = {
            'operation': 'similarity_search',
            'query_embedding_dimension': query_embedding_dim,
            'database_size': database_size,
            'top_k': top_k,
            'results_found': results_found,
            'processing_time_ms': processing_time,
            'success': success
        }
        
        if error:
            log_data['error'] = error
            self.error("Similarity search failed", **log_data)
        else:
            self.info("Similarity search completed", **log_data)
    
    def log_reranking(self,
                     initial_results: int,
                     final_results: int,
                     processing_time: float,
                     reranking_features: Dict,
                     success: bool = True,
                     error: Optional[str] = None):
        """Log reranking operation."""
        log_data = {
            'operation': 'reranking',
            'initial_results': initial_results,
            'final_results': final_results,
            'processing_time_ms': processing_time,
            'reranking_features': reranking_features,
            'success': success
        }
        
        if error:
            log_data['error'] = error
            self.error("Reranking failed", **log_data)
        else:
            self.info("Reranking completed", **log_data)
    
    def log_batch_operation(self,
                          operation_type: str,
                          total_items: int,
                          successful_items: int,
                          failed_items: int,
                          total_processing_time: float,
                          average_processing_time: float):
        """Log batch operation summary."""
        self.info(f"Batch {operation_type} completed",
                 operation=f"batch_{operation_type}",
                 total_items=total_items,
                 successful_items=successful_items,
                 failed_items=failed_items,
                 success_rate=successful_items / total_items if total_items > 0 else 0,
                 total_processing_time_ms=total_processing_time,
                 average_processing_time_ms=average_processing_time)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        summary = {}
        
        for operation, times in self.performance_metrics.items():
            if times:
                summary[operation] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        
        summary['error_counts'] = self.error_counts.copy()
        
        return summary
    
    def reset_metrics(self):
        """Reset performance metrics and error counts."""
        self.performance_metrics.clear()
        self.error_counts.clear()


def setup_logging(config: LoggingConfig) -> FaceRecognitionLogger:
    """
    Setup centralized logging for the face recognition system.
    
    Args:
        config: Logging configuration
        
    Returns:
        Configured logger instance
    """
    # Create main logger
    main_logger = FaceRecognitionLogger("face_recognition", config)
    
    # Set up module-specific loggers
    module_loggers = [
        "face_recognition.pipeline",
        "face_recognition.face_detection",
        "face_recognition.embedding",
        "face_recognition.vector_db",
        "face_recognition.reranking",
        "face_recognition.preprocessing"
    ]
    
    for module_name in module_loggers:
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
        
        # Prevent duplicate logs by not propagating to root logger
        module_logger.propagate = False
        
        # Add the same handlers as main logger
        for handler in main_logger.logger.handlers:
            module_logger.addHandler(handler)
    
    return main_logger