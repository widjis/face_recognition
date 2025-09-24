"""API endpoints for face recognition system."""

import time
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from .models import (
    RecognitionRequest, RecognitionResponse,
    RegistrationRequest, RegistrationResponse,
    BatchRecognitionRequest, BatchRecognitionResponse,
    BatchRegistrationRequest, BatchRegistrationResponse,
    DatabaseInfoResponse, PerformanceMetricsResponse,
    OptimizationResponse, BenchmarkRequest, BenchmarkResponse,
    HealthCheckResponse, ErrorResponse,
    SearchConfigAPI, FaceRegionAPI, SearchResultAPI
)
from ..pipeline import FaceRecognitionPipeline
from ..config.manager import ConfigurationManager
from ..models import RecognitionRequest as PipelineRecognitionRequest, SearchConfig
from ..exceptions import FaceRecognitionError, FaceDetectionError, InvalidImageError


# Global pipeline instance (will be initialized by the app)
pipeline: FaceRecognitionPipeline = None


def get_pipeline() -> FaceRecognitionPipeline:
    """Dependency to get the pipeline instance."""
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    return pipeline


# Create router
router = APIRouter()


@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_face(request: RecognitionRequest, pipeline: FaceRecognitionPipeline = Depends(get_pipeline)):
    """
    Recognize faces in an image.
    
    This endpoint detects faces in the provided image and searches for similar faces
    in the database using embedding similarity and optional reranking.
    """
    try:
        start_time = time.time()
        
        # Convert API request to pipeline request
        image_array = request.image.to_numpy_array()
        
        # Process image with quality validation
        processed_image, quality_info = pipeline.process_image_with_validation(
            image_array, perform_quality_check=True
        )
        
        # Create pipeline request
        pipeline_request = PipelineRecognitionRequest(
            image_data=processed_image,
            search_config=SearchConfig(
                top_k=request.search_config.top_k,
                similarity_threshold=request.search_config.similarity_threshold,
                enable_reranking=request.search_config.enable_reranking,
                distance_metric=request.search_config.distance_metric
            )
        )
        
        # Perform recognition
        pipeline_response = pipeline.recognize_face(pipeline_request)
        
        # Convert pipeline response to API response
        detected_faces = [
            FaceRegionAPI(
                x=face.x,
                y=face.y,
                width=face.width,
                height=face.height,
                confidence=face.confidence
            )
            for face in pipeline_response.detected_faces
        ]
        
        search_results = [
            SearchResultAPI(
                embedding_id=result.embedding_id,
                similarity_score=result.similarity_score,
                rerank_score=result.rerank_score,
                metadata=result.metadata
            )
            for result in pipeline_response.search_results
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return RecognitionResponse(
            success=pipeline_response.success,
            detected_faces=detected_faces,
            search_results=search_results,
            processing_time_ms=processing_time,
            error_message=pipeline_response.error_message,
            quality_info=quality_info
        )
        
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    except FaceDetectionError as e:
        raise HTTPException(status_code=422, detail=f"Face detection failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/register", response_model=RegistrationResponse)
async def register_face(request: RegistrationRequest, pipeline: FaceRecognitionPipeline = Depends(get_pipeline)):
    """
    Register a face in the database.
    
    This endpoint extracts facial features from the provided image and stores them
    in the database with the associated metadata for future recognition.
    """
    try:
        start_time = time.time()
        
        # Convert API request to pipeline format
        image_array = request.image.to_numpy_array()
        
        # Process image with quality validation
        processed_image, quality_info = pipeline.process_image_with_validation(
            image_array, perform_quality_check=True
        )
        
        # Register face
        embedding_id = pipeline.add_face_to_database(
            image=processed_image,
            metadata=request.metadata,
            person_id=request.person_id
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return RegistrationResponse(
            success=True,
            embedding_id=embedding_id,
            processing_time_ms=processing_time,
            quality_info=quality_info
        )
        
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    except FaceDetectionError as e:
        raise HTTPException(status_code=422, detail=f"Face detection failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/batch/recognize", response_model=BatchRecognitionResponse)
async def batch_recognize_faces(request: BatchRecognitionRequest, 
                               background_tasks: BackgroundTasks,
                               pipeline: FaceRecognitionPipeline = Depends(get_pipeline)):
    """
    Recognize faces in multiple images (batch processing).
    
    This endpoint processes multiple images concurrently and returns recognition
    results for each image along with a processing summary.
    """
    try:
        start_time = time.time()
        
        # Convert images to numpy arrays
        images = []
        for image_data in request.images:
            try:
                image_array = image_data.to_numpy_array()
                images.append(image_array)
            except Exception as e:
                # Add placeholder for failed image conversion
                images.append(None)
        
        # Create search config
        search_config = SearchConfig(
            top_k=request.search_config.top_k,
            similarity_threshold=request.search_config.similarity_threshold,
            enable_reranking=request.search_config.enable_reranking,
            distance_metric=request.search_config.distance_metric
        )
        
        # Process batch
        pipeline_results = pipeline.batch_process_images(
            images=[img for img in images if img is not None],
            search_config=search_config,
            max_workers=request.max_workers
        )
        
        # Convert results to API format
        api_results = []
        result_idx = 0
        
        for i, image in enumerate(images):
            if image is None:
                # Failed image conversion
                api_results.append(RecognitionResponse(
                    success=False,
                    detected_faces=[],
                    search_results=[],
                    processing_time_ms=0.0,
                    error_message="Failed to convert image data"
                ))
            else:
                # Use pipeline result
                pipeline_result = pipeline_results[result_idx]
                result_idx += 1
                
                detected_faces = [
                    FaceRegionAPI(
                        x=face.x, y=face.y, width=face.width, 
                        height=face.height, confidence=face.confidence
                    )
                    for face in pipeline_result.detected_faces
                ]
                
                search_results = [
                    SearchResultAPI(
                        embedding_id=result.embedding_id,
                        similarity_score=result.similarity_score,
                        rerank_score=result.rerank_score,
                        metadata=result.metadata
                    )
                    for result in pipeline_result.search_results
                ]
                
                api_results.append(RecognitionResponse(
                    success=pipeline_result.success,
                    detected_faces=detected_faces,
                    search_results=search_results,
                    processing_time_ms=pipeline_result.processing_time_ms,
                    error_message=pipeline_result.error_message
                ))
        
        # Generate summary
        summary = pipeline.get_batch_processing_summary(pipeline_results)
        total_processing_time = (time.time() - start_time) * 1000
        
        return BatchRecognitionResponse(
            success=True,
            results=api_results,
            summary=summary,
            total_processing_time_ms=total_processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@router.post("/batch/register", response_model=BatchRegistrationResponse)
async def batch_register_faces(request: BatchRegistrationRequest,
                              background_tasks: BackgroundTasks,
                              pipeline: FaceRecognitionPipeline = Depends(get_pipeline)):
    """
    Register multiple faces in the database (batch processing).
    
    This endpoint processes multiple face registration requests concurrently
    and returns registration results for each request.
    """
    try:
        start_time = time.time()
        
        # Convert to pipeline format
        images = []
        metadata_list = []
        
        for reg_request in request.registrations:
            try:
                image_array = reg_request.image.to_numpy_array()
                images.append(image_array)
                
                # Prepare metadata with person_id
                metadata = reg_request.metadata.copy()
                metadata['person_id'] = reg_request.person_id
                metadata_list.append(metadata)
                
            except Exception as e:
                images.append(None)
                metadata_list.append({'error': str(e)})
        
        # Process batch registration
        embedding_ids = pipeline.batch_register_faces(
            images=[img for img in images if img is not None],
            metadata_list=[meta for img, meta in zip(images, metadata_list) if img is not None],
            max_workers=request.max_workers
        )
        
        # Convert results to API format
        api_results = []
        result_idx = 0
        
        for i, (image, metadata) in enumerate(zip(images, metadata_list)):
            if image is None:
                api_results.append(RegistrationResponse(
                    success=False,
                    embedding_id=None,
                    processing_time_ms=0.0,
                    error_message=metadata.get('error', 'Failed to process image')
                ))
            else:
                embedding_id = embedding_ids[result_idx]
                result_idx += 1
                
                api_results.append(RegistrationResponse(
                    success=embedding_id is not None,
                    embedding_id=embedding_id,
                    processing_time_ms=0.0,  # Individual timing not available in batch
                    error_message=None if embedding_id else "Registration failed"
                ))
        
        # Generate summary
        successful = sum(1 for result in api_results if result.success)
        failed = len(api_results) - successful
        
        summary = {
            'total_processed': len(api_results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(api_results) if api_results else 0
        }
        
        total_processing_time = (time.time() - start_time) * 1000
        
        return BatchRegistrationResponse(
            success=True,
            results=api_results,
            summary=summary,
            total_processing_time_ms=total_processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch registration failed: {str(e)}")


@router.get("/database/info", response_model=DatabaseInfoResponse)
async def get_database_info(pipeline: FaceRecognitionPipeline = Depends(get_pipeline)):
    """
    Get information about the face database.
    
    Returns statistics about the current database including number of faces,
    storage configuration, and usage statistics.
    """
    try:
        db_info = pipeline.get_database_info()
        
        return DatabaseInfoResponse(
            total_faces=db_info['total_faces'],
            database_path=db_info['database_path'],
            index_type=db_info['index_type'],
            distance_metric=db_info['distance_metric'],
            embedding_dimension=db_info['embedding_dimension'],
            statistics=db_info['statistics']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database info: {str(e)}")


@router.delete("/database/clear")
async def clear_database(pipeline: FaceRecognitionPipeline = Depends(get_pipeline)):
    """
    Clear all data from the face database.
    
    WARNING: This operation is irreversible and will delete all stored faces
    and their associated metadata.
    """
    try:
        pipeline.clear_database()
        return {"message": "Database cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")


@router.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(pipeline: FaceRecognitionPipeline = Depends(get_pipeline)):
    """
    Get comprehensive performance metrics.
    
    Returns detailed performance statistics including operation timings,
    error rates, system resource usage, and identified bottlenecks.
    """
    try:
        metrics = pipeline.get_performance_metrics()
        
        return PerformanceMetricsResponse(
            pipeline_stats=metrics['pipeline_stats'],
            performance_summary=metrics['performance_summary'],
            error_summary=metrics['error_summary'],
            system_metrics=metrics['system_metrics'],
            bottlenecks=metrics['bottlenecks']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/optimize", response_model=OptimizationResponse)
async def get_optimization_recommendations(pipeline: FaceRecognitionPipeline = Depends(get_pipeline)):
    """
    Get performance optimization recommendations.
    
    Analyzes current system performance and provides actionable recommendations
    for improving speed, accuracy, and resource usage.
    """
    try:
        optimization_results = pipeline.optimize_performance()
        
        return OptimizationResponse(
            analysis_timestamp=optimization_results['analysis_timestamp'],
            recommendations=optimization_results['recommendations'],
            metrics_summary=optimization_results['metrics_summary']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimization recommendations: {str(e)}")


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_performance_benchmark(request: BenchmarkRequest,
                                   pipeline: FaceRecognitionPipeline = Depends(get_pipeline)):
    """
    Run performance benchmarks on the system.
    
    Executes standardized performance tests to measure system capabilities
    and identify performance characteristics under controlled conditions.
    """
    try:
        test_image = None
        if request.test_image:
            test_image = request.test_image.to_numpy_array()
        
        benchmark_results = pipeline.benchmark_performance(
            test_image=test_image,
            num_iterations=request.num_iterations
        )
        
        return BenchmarkResponse(
            test_config=benchmark_results['test_config'],
            results=benchmark_results['results'],
            system_resources=benchmark_results['system_resources']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(pipeline: FaceRecognitionPipeline = Depends(get_pipeline)):
    """
    Check the health status of the face recognition service.
    
    Returns the current status of the service and its components,
    useful for monitoring and load balancer health checks.
    """
    try:
        # Calculate uptime (simplified - would need actual start time in production)
        uptime_seconds = time.time() - getattr(pipeline, '_start_time', time.time())
        
        # Check component status
        components_status = {
            'face_detection': 'healthy' if pipeline.face_detector else 'disabled',
            'embedding_extraction': 'healthy' if pipeline.embedding_extractor else 'disabled',
            'vector_database': 'healthy' if pipeline.index else 'error',
            'reranking': 'healthy' if pipeline.reranker else 'disabled',
            'performance_monitoring': 'healthy' if pipeline.performance_monitor else 'error'
        }
        
        # Check database status
        try:
            db_info = pipeline.get_database_info()
            database_status = 'healthy'
        except Exception:
            database_status = 'error'
        
        return HealthCheckResponse(
            status='healthy',
            version='1.0.0',  # Would come from package metadata in production
            uptime_seconds=uptime_seconds,
            database_status=database_status,
            components_status=components_status
        )
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


# Error handlers
@router.exception_handler(FaceRecognitionError)
async def face_recognition_error_handler(request, exc: FaceRecognitionError):
    """Handle face recognition specific errors."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error=type(exc).__name__,
            message=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@router.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="ValidationError",
            message=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )