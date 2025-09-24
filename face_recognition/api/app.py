"""FastAPI application for face recognition system."""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from .endpoints import router, pipeline
from ..pipeline import FaceRecognitionPipeline
from ..config.manager import ConfigurationManager


# Global variables for application state
app_start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global app_start_time, pipeline
    
    # Startup
    app_start_time = time.time()
    
    # Initialize pipeline
    config_manager = ConfigurationManager()
    pipeline_instance = FaceRecognitionPipeline(
        config_manager=config_manager,
        db_path="face_recognition_api_db"
    )
    
    # Store pipeline instance globally
    import face_recognition.api.endpoints as endpoints_module
    endpoints_module.pipeline = pipeline_instance
    
    # Store start time for health checks
    pipeline_instance._start_time = app_start_time
    
    print(f"🚀 Face Recognition API started successfully")
    print(f"   Database: {pipeline_instance.db_path}")
    print(f"   Configuration: {pipeline_instance.config.environment}")
    
    yield
    
    # Shutdown
    if hasattr(pipeline_instance, 'performance_monitor'):
        pipeline_instance.performance_monitor.stop_system_monitoring()
    
    print("🛑 Face Recognition API shutting down")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="Face Recognition API",
        description="""
        A comprehensive face recognition system API that provides:
        
        - **Face Detection**: Detect faces in images with confidence scores
        - **Face Recognition**: Match faces against a database using similarity search
        - **Face Registration**: Add new faces to the database with metadata
        - **Batch Processing**: Process multiple images concurrently
        - **Performance Monitoring**: Track system performance and identify bottlenecks
        - **Quality Assessment**: Evaluate image quality and provide recommendations
        
        The system uses state-of-the-art deep learning models for face embedding extraction
        and efficient vector databases for similarity search with optional reranking.
        """,
        version="1.0.0",
        contact={
            "name": "Face Recognition System",
            "email": "support@facerecognition.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Include routers
    app.include_router(
        router,
        prefix="/api/v1",
        tags=["Face Recognition"]
    )
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Face Recognition API",
            "version": "1.0.0",
            "status": "healthy",
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "api_prefix": "/api/v1"
        }
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title="Face Recognition API",
            version="1.0.0",
            description=app.description,
            routes=app.routes,
        )
        
        # Add custom schema information
        openapi_schema["info"]["x-logo"] = {
            "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
        }
        
        # Add example servers
        openapi_schema["servers"] = [
            {"url": "http://localhost:8000", "description": "Development server"},
            {"url": "https://api.facerecognition.com", "description": "Production server"}
        ]
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "details": str(exc) if app.debug else None,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        )
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "face_recognition.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )