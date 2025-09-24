"""Image quality assessment utilities for face recognition."""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from ..exceptions import InvalidImageError


@dataclass
class QualityMetrics:
    """Container for image quality metrics."""
    overall_score: float  # 0.0 to 1.0
    sharpness_score: float
    brightness_score: float
    contrast_score: float
    noise_level: float
    resolution_score: float
    warnings: list


class ImageQualityAssessor:
    """
    Assesses image quality for face recognition purposes.
    
    Evaluates various quality metrics including sharpness, brightness,
    contrast, noise level, and resolution adequacy.
    """
    
    def __init__(self, 
                 min_resolution: Tuple[int, int] = (100, 100),
                 target_resolution: Tuple[int, int] = (224, 224)):
        """
        Initialize the quality assessor.
        
        Args:
            min_resolution: Minimum acceptable resolution (width, height)
            target_resolution: Target resolution for optimal quality
        """
        self.min_resolution = min_resolution
        self.target_resolution = target_resolution
    
    def assess_quality(self, image: np.ndarray) -> QualityMetrics:
        """
        Perform comprehensive quality assessment of an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            QualityMetrics object with detailed quality information
            
        Raises:
            InvalidImageError: If image is invalid
        """
        if image is None or image.size == 0:
            raise InvalidImageError("Image is empty or None")
        
        try:
            warnings = []
            
            # Calculate individual quality metrics
            sharpness_score = self._calculate_sharpness(image)
            brightness_score = self._calculate_brightness_quality(image)
            contrast_score = self._calculate_contrast_quality(image)
            noise_level = self._calculate_noise_level(image)
            resolution_score = self._calculate_resolution_quality(image)
            
            # Generate warnings based on thresholds
            if sharpness_score < 0.3:
                warnings.append("Image appears blurry or out of focus")
            
            if brightness_score < 0.3:
                warnings.append("Image is too dark or too bright")
            
            if contrast_score < 0.3:
                warnings.append("Image has poor contrast")
            
            if noise_level > 0.7:
                warnings.append("Image has high noise levels")
            
            if resolution_score < 0.5:
                warnings.append("Image resolution is below recommended minimum")
            
            # Calculate overall quality score (weighted average)
            weights = {
                'sharpness': 0.3,
                'brightness': 0.2,
                'contrast': 0.2,
                'noise': 0.15,  # Lower noise is better, so we'll invert this
                'resolution': 0.15
            }
            
            overall_score = (
                weights['sharpness'] * sharpness_score +
                weights['brightness'] * brightness_score +
                weights['contrast'] * contrast_score +
                weights['noise'] * (1.0 - noise_level) +  # Invert noise level
                weights['resolution'] * resolution_score
            )
            
            return QualityMetrics(
                overall_score=max(0.0, min(1.0, overall_score)),
                sharpness_score=sharpness_score,
                brightness_score=brightness_score,
                contrast_score=contrast_score,
                noise_level=noise_level,
                resolution_score=resolution_score,
                warnings=warnings
            )
            
        except Exception as e:
            raise InvalidImageError(f"Failed to assess image quality: {str(e)}")
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.
        
        Args:
            image: Input image
            
        Returns:
            Sharpness score (0.0 to 1.0, higher is sharper)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 range (empirically determined thresholds)
            # Values above 1000 are considered very sharp
            normalized_score = min(1.0, laplacian_var / 1000.0)
            
            return float(normalized_score)
            
        except Exception:
            return 0.5  # Default moderate sharpness
    
    def _calculate_brightness_quality(self, image: np.ndarray) -> float:
        """
        Calculate brightness quality (how close to optimal brightness).
        
        Args:
            image: Input image
            
        Returns:
            Brightness quality score (0.0 to 1.0)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate mean brightness
            mean_brightness = np.mean(gray) / 255.0
            
            # Optimal brightness is around 0.4-0.6 (102-153 in 0-255 range)
            optimal_range = (0.4, 0.6)
            
            if optimal_range[0] <= mean_brightness <= optimal_range[1]:
                # Within optimal range
                return 1.0
            elif mean_brightness < optimal_range[0]:
                # Too dark
                return mean_brightness / optimal_range[0]
            else:
                # Too bright
                return (1.0 - mean_brightness) / (1.0 - optimal_range[1])
            
        except Exception:
            return 0.5  # Default moderate brightness
    
    def _calculate_contrast_quality(self, image: np.ndarray) -> float:
        """
        Calculate contrast quality using standard deviation.
        
        Args:
            image: Input image
            
        Returns:
            Contrast quality score (0.0 to 1.0)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate standard deviation as contrast measure
            contrast = np.std(gray) / 255.0
            
            # Normalize to 0-1 range
            # Good contrast typically has std dev > 0.2 (51 in 0-255 range)
            normalized_score = min(1.0, contrast / 0.2)
            
            return float(normalized_score)
            
        except Exception:
            return 0.5  # Default moderate contrast
    
    def _calculate_noise_level(self, image: np.ndarray) -> float:
        """
        Estimate noise level in the image.
        
        Args:
            image: Input image
            
        Returns:
            Noise level (0.0 to 1.0, higher means more noise)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Use median filter to estimate noise
            # High frequency components after median filtering indicate noise
            median_filtered = cv2.medianBlur(gray, 5)
            noise_estimate = np.mean(np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32)))
            
            # Normalize to 0-1 range (empirically determined threshold)
            normalized_noise = min(1.0, noise_estimate / 20.0)
            
            return float(normalized_noise)
            
        except Exception:
            return 0.3  # Default low-moderate noise
    
    def _calculate_resolution_quality(self, image: np.ndarray) -> float:
        """
        Calculate resolution quality based on image dimensions.
        
        Args:
            image: Input image
            
        Returns:
            Resolution quality score (0.0 to 1.0)
        """
        try:
            height, width = image.shape[:2]
            
            # Check if below minimum resolution
            if width < self.min_resolution[0] or height < self.min_resolution[1]:
                # Calculate how far below minimum
                width_ratio = width / self.min_resolution[0]
                height_ratio = height / self.min_resolution[1]
                return min(width_ratio, height_ratio) * 0.5  # Cap at 0.5 for below minimum
            
            # Calculate score based on how close to target resolution
            target_width, target_height = self.target_resolution
            
            width_score = min(1.0, width / target_width)
            height_score = min(1.0, height / target_height)
            
            # Use the minimum to ensure both dimensions are adequate
            return min(width_score, height_score)
            
        except Exception:
            return 0.5  # Default moderate resolution
    
    def is_acceptable_quality(self, 
                            quality_metrics: QualityMetrics,
                            min_overall_score: float = 0.4) -> bool:
        """
        Determine if image quality is acceptable for face recognition.
        
        Args:
            quality_metrics: Quality metrics from assess_quality()
            min_overall_score: Minimum acceptable overall score
            
        Returns:
            True if quality is acceptable, False otherwise
        """
        return quality_metrics.overall_score >= min_overall_score
    
    def get_quality_recommendations(self, quality_metrics: QualityMetrics) -> list:
        """
        Get recommendations for improving image quality.
        
        Args:
            quality_metrics: Quality metrics from assess_quality()
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if quality_metrics.sharpness_score < 0.5:
            recommendations.append("Ensure the image is in focus and not blurry")
        
        if quality_metrics.brightness_score < 0.5:
            recommendations.append("Improve lighting conditions - avoid too dark or too bright images")
        
        if quality_metrics.contrast_score < 0.5:
            recommendations.append("Increase image contrast for better feature definition")
        
        if quality_metrics.noise_level > 0.6:
            recommendations.append("Reduce image noise - use better lighting or camera settings")
        
        if quality_metrics.resolution_score < 0.7:
            recommendations.append(f"Use higher resolution images (recommended: {self.target_resolution[0]}x{self.target_resolution[1]} or higher)")
        
        if not recommendations:
            recommendations.append("Image quality is good for face recognition")
        
        return recommendations
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Apply basic quality enhancement to an image.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            enhanced = image.copy()
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if len(enhanced.shape) == 3:
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(enhanced)
            
            # Apply slight Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
            
            return enhanced
            
        except Exception:
            # Return original image if enhancement fails
            return image