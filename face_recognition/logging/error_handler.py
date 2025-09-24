"""Error handling and recovery mechanisms."""

import time
import functools
from typing import Dict, Any, Optional, Callable, Type, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading

from ..exceptions import (
    FaceRecognitionError,
    FaceDetectionError,
    EmbeddingExtractionError,
    VectorDatabaseError,
    InvalidImageError,
    ConfigurationError
)


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    timestamp: datetime
    error_type: str
    error_message: str
    retry_count: int = 0
    context_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_backoff: bool = True
    jitter: bool = True


class ErrorHandler:
    """
    Centralized error handling with retry logic and graceful degradation.
    
    Provides consistent error handling across the face recognition system
    with configurable retry policies and fallback mechanisms.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.error_history: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        self._lock = threading.Lock()
    
    def handle_error(self, 
                    error: Exception,
                    operation: str,
                    context_data: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """
        Handle an error with logging and tracking.
        
        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            context_data: Additional context information
            
        Returns:
            ErrorContext with error details
        """
        error_context = ErrorContext(
            operation=operation,
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            context_data=context_data or {}
        )
        
        with self._lock:
            self.error_history.append(error_context)
            self.error_counts[error_context.error_type] = (
                self.error_counts.get(error_context.error_type, 0) + 1
            )
        
        return error_context
    
    def should_retry(self, 
                    error: Exception,
                    operation: str,
                    retry_count: int,
                    retry_config: RetryConfig) -> bool:
        """
        Determine if an operation should be retried.
        
        Args:
            error: The exception that occurred
            operation: Name of the operation
            retry_count: Current retry count
            retry_config: Retry configuration
            
        Returns:
            True if should retry, False otherwise
        """
        # Don't retry if max retries exceeded
        if retry_count >= retry_config.max_retries:
            return False
        
        # Don't retry certain error types
        non_retryable_errors = (
            InvalidImageError,
            ConfigurationError,
            ValueError,
            TypeError
        )
        
        if isinstance(error, non_retryable_errors):
            return False
        
        # Check circuit breaker
        circuit_breaker = self.circuit_breakers.get(operation)
        if circuit_breaker and circuit_breaker.is_open():
            return False
        
        return True
    
    def calculate_delay(self, 
                       retry_count: int,
                       retry_config: RetryConfig) -> float:
        """
        Calculate delay before retry.
        
        Args:
            retry_count: Current retry count
            retry_config: Retry configuration
            
        Returns:
            Delay in seconds
        """
        if retry_config.exponential_backoff:
            delay = retry_config.base_delay * (2 ** retry_count)
        else:
            delay = retry_config.base_delay
        
        # Cap at max delay
        delay = min(delay, retry_config.max_delay)
        
        # Add jitter to prevent thundering herd
        if retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def get_circuit_breaker(self, operation: str) -> 'CircuitBreaker':
        """Get or create circuit breaker for operation."""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreaker(operation)
        return self.circuit_breakers[operation]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error statistics."""
        with self._lock:
            recent_errors = [
                error for error in self.error_history
                if error.timestamp > datetime.now() - timedelta(hours=1)
            ]
            
            return {
                'total_errors': len(self.error_history),
                'recent_errors_1h': len(recent_errors),
                'error_counts': self.error_counts.copy(),
                'circuit_breaker_status': {
                    name: cb.get_status() 
                    for name, cb in self.circuit_breakers.items()
                }
            }
    
    def clear_error_history(self, older_than_hours: int = 24):
        """Clear old error history."""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        with self._lock:
            self.error_history = [
                error for error in self.error_history
                if error.timestamp > cutoff_time
            ]


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for preventing cascading failures.
    """
    
    def __init__(self, 
                 name: str,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Type[Exception] = Exception):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            failure_threshold: Number of failures before opening
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to monitor
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        with self._lock:
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self.failure_count = 0
            self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == 'OPEN'
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time
        }


class ErrorRecoveryManager:
    """
    Manages error recovery strategies and graceful degradation.
    """
    
    def __init__(self, error_handler: ErrorHandler):
        """Initialize recovery manager."""
        self.error_handler = error_handler
        self.fallback_strategies: Dict[str, Callable] = {}
    
    def register_fallback(self, operation: str, fallback_func: Callable):
        """
        Register a fallback strategy for an operation.
        
        Args:
            operation: Operation name
            fallback_func: Fallback function to call
        """
        self.fallback_strategies[operation] = fallback_func
    
    def with_retry_and_fallback(self, 
                               operation: str,
                               retry_config: Optional[RetryConfig] = None):
        """
        Decorator for adding retry logic and fallback to functions.
        
        Args:
            operation: Operation name
            retry_config: Retry configuration
            
        Returns:
            Decorated function
        """
        if retry_config is None:
            retry_config = RetryConfig()
        
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                retry_count = 0
                last_error = None
                
                while retry_count <= retry_config.max_retries:
                    try:
                        # Try the main function
                        return func(*args, **kwargs)
                        
                    except Exception as e:
                        last_error = e
                        
                        # Handle the error
                        error_context = self.error_handler.handle_error(
                            e, operation, {'retry_count': retry_count}
                        )
                        
                        # Check if should retry
                        if self.error_handler.should_retry(
                            e, operation, retry_count, retry_config
                        ):
                            retry_count += 1
                            delay = self.error_handler.calculate_delay(
                                retry_count, retry_config
                            )
                            time.sleep(delay)
                            continue
                        else:
                            break
                
                # All retries exhausted, try fallback
                if operation in self.fallback_strategies:
                    try:
                        return self.fallback_strategies[operation](*args, **kwargs)
                    except Exception as fallback_error:
                        # Log fallback failure but raise original error
                        self.error_handler.handle_error(
                            fallback_error, f"{operation}_fallback"
                        )
                
                # No fallback or fallback failed, raise original error
                raise last_error
            
            return wrapper
        return decorator
    
    def graceful_degradation(self, 
                           primary_func: Callable,
                           fallback_func: Callable,
                           operation: str,
                           *args, **kwargs):
        """
        Execute function with graceful degradation.
        
        Args:
            primary_func: Primary function to try
            fallback_func: Fallback function if primary fails
            operation: Operation name
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Result from primary or fallback function
        """
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            self.error_handler.handle_error(e, operation)
            
            try:
                return fallback_func(*args, **kwargs)
            except Exception as fallback_error:
                self.error_handler.handle_error(
                    fallback_error, f"{operation}_fallback"
                )
                raise e  # Raise original error