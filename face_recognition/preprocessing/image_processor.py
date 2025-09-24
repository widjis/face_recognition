"""Main image processor that combines format handling and quality assessment."""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

from .format_handler import ImageFormatHandler
from .quality_assessor import ImageQualityAssessor, QualityMetrics
from ..exceptions import InvalidImageError


class ImageProcessor:
    """
    Main image processor for the face recognition system.
    
    Handles image loading, format conversion, quality assessment,
    preprocessing, and validation.
    """
    
    def __init__(self,
                 min_resolution: Tuple[int, int] = (100, 100),
                 target_resolution: Tuple[int, int] = (224, 224),
                 auto_enhance: bool = True,
                 quality_threshold: float = 0.3):
        """
        Initialize the image processor.
        
        Args:
            min_resolution: Minimum acceptable resolution
            target_resolution: Target resolution for processing
            auto_enhance: Whether to automatically enhance low-quality images
            quality_threshold: Minimum quality score to accept images
        """
        self.format_handler = ImageFormatHandler()
        self.quality_assessor = ImageQualityAssessor(min_resolution, target_resolution)
        self.auto_enhance = auto_enhance
        self.quality_threshold = quality_threshold
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def load_and_validate_image(self, 
                              image_source: Union[str, bytes, np.ndarray],
                              perform_quality_check: bool = True) -> Tuple[np.ndarray, QualityMetrics]:
        """
        Load and validate an image from various sources.
        
        Args:
            image_source: Image file path, bytes, or numpy array
            perform_quality_check: Whether to perform quality assessment
            
        Returns:
            Tuple of (processed_image, quality_metrics)
            
        Raises:
            InvalidImageError: If image cannot be loaded or is invalid
        """
        try:
            # Load image based on source type
            if isinstance(image_source, str):
                image = self.format_handler.load_image_from_path(image_source)
                self.logger.info(f"Loaded image from path: {image_source}")
            elif isinstance(image_source, bytes):
                image = self.format_handler.load_image_from_bytes(image_source)
                self.logger.info("Loaded image from bytes")
            elif isinstance(image_source, np.ndarray):
                image = image_source.copy()
                self.logger.info("Using provided numpy array")
            else:
                raise InvalidImageError(f"Unsupported image source type: {type(image_source)}")
            
            # Basic validation
            if image is None or image.size == 0:
                raise InvalidImageError("Loaded image is empty")
            
            # Check dimensions
            if len(image.shape) not in [2, 3]:
                raise InvalidImageError(f"Invalid image dimensions: {image.shape}")
            
            if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
                raise InvalidImageError(f"Invalid number of channels: {image.shape[2]}")
            
            # Ensure 3-channel BGR format
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            # Perform quality assessment
            quality_metrics = None
            if perform_quality_check:
                quality_metrics = self.quality_assessor.assess_quality(image)
                
                # Log quality warnings
                for warning in quality_metrics.warnings:
                    self.logger.warning(f"Image quality issue: {warning}")
                
                # Check if quality is acceptable
                if quality_metrics.overall_score < self.quality_threshold:
                    if self.auto_enhance:
                        self.logger.info("Attempting to enhance low-quality image")
                        image = self.quality_assessor.enhance_image_quality(image)
                        # Re-assess quality after enhancement
                        quality_metrics = self.quality_assessor.assess_quality(image)
                    else:
                        self.logger.warning(
                            f"Image quality below threshold: {quality_metrics.overall_score:.2f} < {self.quality_threshold}"
                        )
            
            return image, quality_metrics
            
        except InvalidImageError:
            raise
        except Exception as e:
            raise InvalidImageError(f"Failed to load and validate image: {str(e)}")
    
    def preprocess_for_face_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for optimal face detection.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        try:
            processed = image.copy()
            
            # Resize if too large (for performance)
            height, width = processed.shape[:2]
            max_dimension = 1024
            
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_AREA)
                self.logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Failed to preprocess image for face detection: {e}")
            return image
    
    def batch_process_images(self, 
                           image_sources: List[Union[str, bytes, np.ndarray]],
                           perform_quality_check: bool = True) -> List[Tuple[Optional[np.ndarray], Optional[QualityMetrics], Optional[str]]]:
        """
        Process multiple images in batch.
        
        Args:
            image_sources: List of image sources (paths, bytes, or arrays)
            perform_quality_check: Whether to perform quality assessment
            
        Returns:
            List of tuples (image, quality_metrics, error_message)
            None values indicate processing failure
        """
        results = []
        
        for i, source in enumerate(image_sources):
            try:
                image, quality_metrics = self.load_and_validate_image(
                    source, perform_quality_check
                )
                results.append((image, quality_metrics, None))
                self.logger.info(f"Successfully processed image {i+1}/{len(image_sources)}")
                
            except Exception as e:
                error_msg = str(e)
                results.append((None, None, error_msg))
                self.logger.error(f"Failed to process image {i+1}/{len(image_sources)}: {error_msg}")
        
        return results
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        Get list of supported image formats.
        
        Returns:
            Dictionary mapping format names to file extensions
        """
        return self.format_handler.SUPPORTED_FORMATS.copy()
    
    def convert_image_format(self, 
                           image: np.ndarray,
                           target_format: str,
                           quality: int = 95) -> bytes:
        """
        Convert image to specified format.
        
        Args:
            image: Input image
            target_format: Target format ('JPEG', 'PNG', etc.)
            quality: Quality for lossy formats (0-100)
            
        Returns:
            Image bytes in target format
        """
        return self.format_handler.convert_format(image, target_format, quality)
    
    def resize_image(self, 
                    image: np.ndarray,
                    target_size: Tuple[int, int],
                    maintain_aspect_ratio: bool = True,
                    interpolation: int = cv2.INTER_AREA) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            interpolation: OpenCV interpolation method
            
        Returns:
            Resized image
        """
        try:
            if not maintain_aspect_ratio:
                return cv2.resize(image, target_size, interpolation=interpolation)
            
            # Calculate scaling to maintain aspect ratio
            height, width = image.shape[:2]
            target_width, target_height = target_size
            
            # Calculate scale to fit within target size
            scale_w = target_width / width
            scale_h = target_height / height
            scale = min(scale_w, scale_h)
            
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
            
            # Create canvas with target size and center the resized image
            canvas = np.zeros((target_height, target_width, 3), dtype=image.dtype)
            
            # Calculate position to center the image
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
            return canvas
            
        except Exception as e:
            raise InvalidImageError(f"Failed to resize image: {str(e)}")
    
    def normalize_image(self, 
                       image: np.ndarray,
                       target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """
        Normalize image pixel values to target range.
        
        Args:
            image: Input image
            target_range: Target value range (min, max)
            
        Returns:
            Normalized image
        """
        try:
            # Convert to float
            normalized = image.astype(np.float32)
            
            # Normalize to 0-1 range first
            normalized = normalized / 255.0
            
            # Scale to target range
            min_val, max_val = target_range
            normalized = normalized * (max_val - min_val) + min_val
            
            return normalized
            
        except Exception as e:
            raise InvalidImageError(f"Failed to normalize image: {str(e)}")
    
    def get_processing_summary(self, 
                             batch_results: List[Tuple[Optional[np.ndarray], Optional[QualityMetrics], Optional[str]]]) -> Dict:
        """
        Generate summary statistics for batch processing results.
        
        Args:
            batch_results: Results from batch_process_images()
            
        Returns:
            Dictionary with processing statistics
        """
        total_images = len(batch_results)
        successful = sum(1 for result in batch_results if result[0] is not None)
        failed = total_images - successful
        
        # Quality statistics for successful images
        quality_scores = [
            result[1].overall_score 
            for result in batch_results 
            if result[1] is not None
        ]
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Count warnings
        total_warnings = sum(
            len(result[1].warnings)
            for result in batch_results
            if result[1] is not None
        )
        
        return {
            'total_processed': total_images,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_images if total_images > 0 else 0,
            'average_quality_score': avg_quality,
            'total_quality_warnings': total_warnings,
            'error_messages': [
                result[2] for result in batch_results if result[2] is not None
            ]
        }