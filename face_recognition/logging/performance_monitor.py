"""Performance monitoring and metrics collection."""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
import statistics


@dataclass
class PerformanceMetric:
    """Container for performance metrics."""
    operation: str
    timestamp: datetime
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    context_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    process_memory_mb: float
    process_cpu_percent: float


class PerformanceMonitor:
    """
    Monitors and tracks performance metrics for the face recognition system.
    
    Collects timing, memory usage, and system resource metrics to help
    identify performance bottlenecks and optimize system performance.
    """
    
    def __init__(self, 
                 max_metrics_history: int = 10000,
                 system_monitoring_interval: float = 30.0):
        """
        Initialize performance monitor.
        
        Args:
            max_metrics_history: Maximum number of metrics to keep in memory
            system_monitoring_interval: Interval for system metrics collection (seconds)
        """
        self.max_metrics_history = max_metrics_history
        self.system_monitoring_interval = system_monitoring_interval
        
        self.metrics_history: List[PerformanceMetric] = []
        self.system_metrics_history: List[SystemMetrics] = []
        
        self._lock = threading.Lock()
        self._system_monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        # Current process for monitoring
        self.process = psutil.Process()
        
        # Start system monitoring
        self.start_system_monitoring()
    
    def start_system_monitoring(self):
        """Start background system monitoring."""
        if self._system_monitor_thread is None or not self._system_monitor_thread.is_alive():
            self._stop_monitoring.clear()
            self._system_monitor_thread = threading.Thread(
                target=self._system_monitoring_loop,
                daemon=True
            )
            self._system_monitor_thread.start()
    
    def stop_system_monitoring(self):
        """Stop background system monitoring."""
        self._stop_monitoring.set()
        if self._system_monitor_thread:
            self._system_monitor_thread.join(timeout=5.0)
    
    def _system_monitoring_loop(self):
        """Background loop for collecting system metrics."""
        while not self._stop_monitoring.wait(self.system_monitoring_interval):
            try:
                self._collect_system_metrics()
            except Exception:
                # Silently continue if system metrics collection fails
                pass
    
    def _collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process metrics
            process_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            process_cpu = self.process.cpu_percent()
            
            system_metric = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_usage_percent=disk.percent,
                process_memory_mb=process_memory,
                process_cpu_percent=process_cpu
            )
            
            with self._lock:
                self.system_metrics_history.append(system_metric)
                
                # Keep only recent system metrics
                if len(self.system_metrics_history) > 1000:
                    self.system_metrics_history = self.system_metrics_history[-1000:]
                    
        except Exception:
            # Ignore errors in system metrics collection
            pass
    
    @contextmanager
    def measure_operation(self, operation: str, **context_data):
        """
        Context manager for measuring operation performance.
        
        Args:
            operation: Name of the operation being measured
            **context_data: Additional context information
            
        Yields:
            Performance metric object (updated after operation completes)
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = self.process.cpu_percent()
        
        # Create metric object
        metric = PerformanceMetric(
            operation=operation,
            timestamp=datetime.now(),
            duration_ms=0.0,
            memory_usage_mb=start_memory,
            cpu_percent=start_cpu,
            context_data=context_data
        )
        
        try:
            yield metric
        finally:
            # Update metric with final measurements
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = self.process.cpu_percent()
            
            metric.duration_ms = (end_time - start_time) * 1000
            metric.memory_usage_mb = max(end_memory, start_memory)  # Peak memory
            metric.cpu_percent = max(end_cpu, start_cpu)  # Peak CPU
            
            # Store metric
            self._store_metric(metric)
    
    def record_metric(self, 
                     operation: str,
                     duration_ms: float,
                     **context_data):
        """
        Record a performance metric manually.
        
        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            **context_data: Additional context information
        """
        metric = PerformanceMetric(
            operation=operation,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            memory_usage_mb=self._get_memory_usage(),
            cpu_percent=self.process.cpu_percent(),
            context_data=context_data
        )
        
        self._store_metric(metric)
    
    def _store_metric(self, metric: PerformanceMetric):
        """Store a performance metric."""
        with self._lock:
            self.metrics_history.append(metric)
            
            # Keep only recent metrics
            if len(self.metrics_history) > self.max_metrics_history:
                self.metrics_history = self.metrics_history[-self.max_metrics_history:]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def get_operation_stats(self, 
                          operation: str,
                          time_window_hours: Optional[float] = None) -> Dict[str, Any]:
        """
        Get statistics for a specific operation.
        
        Args:
            operation: Operation name
            time_window_hours: Time window for analysis (None for all time)
            
        Returns:
            Dictionary with operation statistics
        """
        with self._lock:
            metrics = self.metrics_history.copy()
        
        # Filter by operation
        operation_metrics = [m for m in metrics if m.operation == operation]
        
        # Filter by time window
        if time_window_hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            operation_metrics = [
                m for m in operation_metrics 
                if m.timestamp > cutoff_time
            ]
        
        if not operation_metrics:
            return {
                'operation': operation,
                'count': 0,
                'time_window_hours': time_window_hours
            }
        
        # Calculate statistics
        durations = [m.duration_ms for m in operation_metrics]
        memory_usage = [m.memory_usage_mb for m in operation_metrics]
        cpu_usage = [m.cpu_percent for m in operation_metrics]
        
        return {
            'operation': operation,
            'count': len(operation_metrics),
            'time_window_hours': time_window_hours,
            'duration_stats': {
                'mean_ms': statistics.mean(durations),
                'median_ms': statistics.median(durations),
                'min_ms': min(durations),
                'max_ms': max(durations),
                'std_dev_ms': statistics.stdev(durations) if len(durations) > 1 else 0.0
            },
            'memory_stats': {
                'mean_mb': statistics.mean(memory_usage),
                'median_mb': statistics.median(memory_usage),
                'min_mb': min(memory_usage),
                'max_mb': max(memory_usage)
            },
            'cpu_stats': {
                'mean_percent': statistics.mean(cpu_usage),
                'median_percent': statistics.median(cpu_usage),
                'min_percent': min(cpu_usage),
                'max_percent': max(cpu_usage)
            }
        }
    
    def get_system_stats(self, 
                        time_window_hours: Optional[float] = None) -> Dict[str, Any]:
        """
        Get system resource statistics.
        
        Args:
            time_window_hours: Time window for analysis (None for all time)
            
        Returns:
            Dictionary with system statistics
        """
        with self._lock:
            system_metrics = self.system_metrics_history.copy()
        
        # Filter by time window
        if time_window_hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            system_metrics = [
                m for m in system_metrics 
                if m.timestamp > cutoff_time
            ]
        
        if not system_metrics:
            return {
                'count': 0,
                'time_window_hours': time_window_hours
            }
        
        # Calculate statistics
        cpu_usage = [m.cpu_percent for m in system_metrics]
        memory_usage = [m.memory_percent for m in system_metrics]
        process_memory = [m.process_memory_mb for m in system_metrics]
        process_cpu = [m.process_cpu_percent for m in system_metrics]
        
        return {
            'count': len(system_metrics),
            'time_window_hours': time_window_hours,
            'system_cpu': {
                'mean_percent': statistics.mean(cpu_usage),
                'max_percent': max(cpu_usage),
                'current_percent': system_metrics[-1].cpu_percent
            },
            'system_memory': {
                'mean_percent': statistics.mean(memory_usage),
                'max_percent': max(memory_usage),
                'current_percent': system_metrics[-1].memory_percent,
                'available_mb': system_metrics[-1].memory_available_mb
            },
            'process_memory': {
                'mean_mb': statistics.mean(process_memory),
                'max_mb': max(process_memory),
                'current_mb': system_metrics[-1].process_memory_mb
            },
            'process_cpu': {
                'mean_percent': statistics.mean(process_cpu),
                'max_percent': max(process_cpu),
                'current_percent': system_metrics[-1].process_cpu_percent
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary with performance summary
        """
        with self._lock:
            metrics = self.metrics_history.copy()
        
        # Group by operation
        operations = {}
        for metric in metrics:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric)
        
        # Calculate summary for each operation
        operation_summaries = {}
        for operation, op_metrics in operations.items():
            durations = [m.duration_ms for m in op_metrics]
            operation_summaries[operation] = {
                'count': len(op_metrics),
                'total_time_ms': sum(durations),
                'avg_time_ms': statistics.mean(durations),
                'max_time_ms': max(durations),
                'min_time_ms': min(durations)
            }
        
        return {
            'total_operations': len(metrics),
            'unique_operations': len(operations),
            'monitoring_duration_hours': (
                (datetime.now() - metrics[0].timestamp).total_seconds() / 3600
                if metrics else 0
            ),
            'operations': operation_summaries,
            'system_stats': self.get_system_stats(time_window_hours=1.0)
        }
    
    def identify_bottlenecks(self, 
                           min_operations: int = 10,
                           slow_threshold_ms: float = 1000.0) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks.
        
        Args:
            min_operations: Minimum number of operations to consider
            slow_threshold_ms: Threshold for considering operations slow
            
        Returns:
            List of potential bottlenecks
        """
        bottlenecks = []
        
        # Get operation stats
        with self._lock:
            metrics = self.metrics_history.copy()
        
        operations = {}
        for metric in metrics:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric)
        
        for operation, op_metrics in operations.items():
            if len(op_metrics) < min_operations:
                continue
            
            durations = [m.duration_ms for m in op_metrics]
            avg_duration = statistics.mean(durations)
            max_duration = max(durations)
            
            # Check if operation is consistently slow
            if avg_duration > slow_threshold_ms:
                bottlenecks.append({
                    'operation': operation,
                    'type': 'consistently_slow',
                    'avg_duration_ms': avg_duration,
                    'max_duration_ms': max_duration,
                    'operation_count': len(op_metrics),
                    'recommendation': f"Operation {operation} averages {avg_duration:.1f}ms, consider optimization"
                })
            
            # Check for high variance (inconsistent performance)
            if len(durations) > 1:
                std_dev = statistics.stdev(durations)
                if std_dev > avg_duration * 0.5:  # High variance
                    bottlenecks.append({
                        'operation': operation,
                        'type': 'high_variance',
                        'avg_duration_ms': avg_duration,
                        'std_dev_ms': std_dev,
                        'operation_count': len(op_metrics),
                        'recommendation': f"Operation {operation} has inconsistent performance (std dev: {std_dev:.1f}ms)"
                    })
        
        return bottlenecks
    
    def clear_metrics(self, older_than_hours: int = 24):
        """
        Clear old metrics to free memory.
        
        Args:
            older_than_hours: Clear metrics older than this many hours
        """
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        with self._lock:
            self.metrics_history = [
                m for m in self.metrics_history 
                if m.timestamp > cutoff_time
            ]
            
            self.system_metrics_history = [
                m for m in self.system_metrics_history 
                if m.timestamp > cutoff_time
            ]
    
    def __del__(self):
        """Cleanup when monitor is destroyed."""
        self.stop_system_monitoring()