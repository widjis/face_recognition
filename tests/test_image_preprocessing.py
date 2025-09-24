"""Unit tests for image preprocessing functionality."""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from face_recognition.preprocessing import ImageProcessor, ImageFormatHandler, ImageQualityAssessor
from face_recognition.preprocessing.quality_assessor import QualityMetrics
from face_recognition.exceptions import InvalidImageError


class TestImageFormatHandler:
    """Test image format handling functionality."""
    
    @pytest.fixture
    def format_handler(self):
        """Create format handler instance."""
        return ImageFormatHandler()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 80), (255, 255, 255), -1)
        return image
    
    def test_detect_format_from_path(self, format_handler):
        """Test format detection from file path."""
        test_cases = [
            ("image.jpg", "JPEG"),
            ("image.jpeg", "JPEG"),
            ("image.png", "PNG"),
            ("image.bmp", "BMP"),
            ("image.tiff", "TIFF"),
            ("image.webp", "WEBP"),
            ("image.txt", None),
            ("image", None)
        ]
        
        for file_path, expected_format in test_cases:
            result = format_handler.detect_format_from_path(file_path)
            assert result == expected_format
    
    def test_detect_format_from_bytes(self, format_handler):
        """Test format detection from byte content."""
        # JPEG magic number
        jpeg_bytes = b'\xff\xd8\xff\xe0' + b'\x00' * 100
        assert format_handler.detect_format_from_bytes(jpeg_bytes) == "JPEG"
        
        # PNG magic number
        png_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        assert format_handler.detect_format_from_bytes(png_bytes) == "PNG"
        
        # BMP magic number
        bmp_bytes = b'BM' + b'\x00' * 100
        assert format_handler.detect_format_from_bytes(bmp_bytes) == "BMP"
        
        # Unknown format
        unknown_bytes = b'\x00\x00\x00\x00' + b'\x00' * 100
        assert format_handler.detect_format_from_bytes(unknown_bytes) is None
        
        # Empty bytes
        assert format_handler.detect_format_from_bytes(b'') is None
    
    def test_is_supported_format(self, format_handler):
        """Test format support checking."""
        assert format_handler.is_supported_format("image.jpg") is True
        assert format_handler.is_supported_format("image.png") is True
        assert format_handler.is_supported_format("image.txt") is False
        assert format_handler.is_supported_format("image") is False
    
    def test_convert_format(self, format_handler, sample_image):
        """Test image format conversion."""
        # Test JPEG conversion
        jpeg_bytes = format_handler.convert_format(sample_image, "JPEG", quality=90)
        assert isinstance(jpeg_bytes, bytes)
        assert len(jpeg_bytes) > 0
        
        # Test PNG conversion
        png_bytes = format_handler.convert_format(sample_image, "PNG")
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        
        # Test unsupported format
        with pytest.raises(InvalidImageError):
            format_handler.convert_format(sample_image, "UNSUPPORTED")
    
    def test_get_image_info(self, format_handler, sample_image):
        """Test image information extraction."""
        info = format_handler.get_image_info(sample_image)
        
        assert info['width'] == 100
        assert info['height'] == 100
        assert info['channels'] == 3
        assert info['aspect_ratio'] == 1.0
        assert info['total_pixels'] == 10000
        assert 'dtype' in info
        assert 'size_bytes' in info


class TestImageQualityAssessor:
    """Test image quality assessment functionality."""
    
    @pytest.fixture
    def quality_assessor(self):
        """Create quality assessor instance."""
        return ImageQualityAssessor()
    
    @pytest.fixture
    def high_quality_image(self):
        """Create a high-quality test image."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        # Add high-contrast features
        cv2.rectangle(image, (50, 50), (174, 174), (255, 255, 255), -1)
        cv2.rectangle(image, (75, 75), (149, 149), (0, 0, 0), -1)
        cv2.circle(image, (112, 112), 30, (128, 128, 128), -1)
        return image
    
    @pytest.fixture
    def low_quality_image(self):
        """Create a low-quality test image."""
        # Very small, low contrast image
        image = np.full((50, 50, 3), 128, dtype=np.uint8)
        # Add some noise
        noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return image
    
    def test_assess_quality_high_quality(self, quality_assessor, high_quality_image):
        """Test quality assessment on high-quality image."""
        metrics = quality_assessor.assess_quality(high_quality_image)
        
        assert isinstance(metrics, QualityMetrics)
        assert 0.0 <= metrics.overall_score <= 1.0
        assert 0.0 <= metrics.sharpness_score <= 1.0
        assert 0.0 <= metrics.brightness_score <= 1.0
        assert 0.0 <= metrics.contrast_score <= 1.0
        assert 0.0 <= metrics.noise_level <= 1.0
        assert 0.0 <= metrics.resolution_score <= 1.0
        assert isinstance(metrics.warnings, list)
        
        # High-quality image should have good scores
        assert metrics.overall_score > 0.5
        assert metrics.resolution_score > 0.8  # 224x224 is good resolution
    
    def test_assess_quality_low_quality(self, quality_assessor, low_quality_image):
        """Test quality assessment on low-quality image."""
        metrics = quality_assessor.assess_quality(low_quality_image)
        
        # Low-quality image should have lower scores
        assert metrics.overall_score < 0.7
        assert metrics.resolution_score < 0.5  # 50x50 is below minimum
        assert len(metrics.warnings) > 0
    
    def test_assess_quality_invalid_image(self, quality_assessor):
        """Test quality assessment with invalid images."""
        # Empty image
        with pytest.raises(InvalidImageError):
            quality_assessor.assess_quality(np.array([]))
        
        # None image
        with pytest.raises(InvalidImageError):
            quality_assessor.assess_quality(None)
    
    def test_is_acceptable_quality(self, quality_assessor):
        """Test quality acceptance checking."""
        # Create mock quality metrics
        good_metrics = QualityMetrics(
            overall_score=0.8,
            sharpness_score=0.8,
            brightness_score=0.7,
            contrast_score=0.9,
            noise_level=0.2,
            resolution_score=1.0,
            warnings=[]
        )
        
        poor_metrics = QualityMetrics(
            overall_score=0.2,
            sharpness_score=0.3,
            brightness_score=0.2,
            contrast_score=0.1,
            noise_level=0.8,
            resolution_score=0.3,
            warnings=["Multiple issues"]
        )
        
        assert quality_assessor.is_acceptable_quality(good_metrics) is True
        assert quality_assessor.is_acceptable_quality(poor_metrics) is False
        assert quality_assessor.is_acceptable_quality(poor_metrics, min_overall_score=0.1) is True
    
    def test_get_quality_recommendations(self, quality_assessor):
        """Test quality recommendation generation."""
        poor_metrics = QualityMetrics(
            overall_score=0.2,
            sharpness_score=0.3,
            brightness_score=0.2,
            contrast_score=0.1,
            noise_level=0.8,
            resolution_score=0.3,
            warnings=[]
        )
        
        recommendations = quality_assessor.get_quality_recommendations(poor_metrics)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should have recommendations for all poor metrics
        rec_text = " ".join(recommendations).lower()
        assert "focus" in rec_text or "blur" in rec_text  # Sharpness
        assert "lighting" in rec_text  # Brightness
        assert "contrast" in rec_text  # Contrast
        assert "noise" in rec_text  # Noise
        assert "resolution" in rec_text  # Resolution
    
    def test_enhance_image_quality(self, quality_assessor, low_quality_image):
        """Test image quality enhancement."""
        enhanced = quality_assessor.enhance_image_quality(low_quality_image)
        
        assert enhanced.shape == low_quality_image.shape
        assert enhanced.dtype == low_quality_image.dtype
        
        # Enhanced image should be different from original
        assert not np.array_equal(enhanced, low_quality_image)


class TestImageProcessor:
    """Test main image processor functionality."""
    
    @pytest.fixture
    def image_processor(self):
        """Create image processor instance."""
        return ImageProcessor(auto_enhance=True, quality_threshold=0.3)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(image, (100, 100), 30, (128, 128, 128), -1)
        return image
    
    @pytest.fixture
    def temp_image_file(self, sample_image):
        """Create a temporary image file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            cv2.imwrite(f.name, sample_image)
            yield f.name
        os.unlink(f.name)
    
    def test_load_and_validate_image_from_array(self, image_processor, sample_image):
        """Test loading image from numpy array."""
        image, quality_metrics = image_processor.load_and_validate_image(sample_image)
        
        assert isinstance(image, np.ndarray)
        assert image.shape == sample_image.shape
        assert isinstance(quality_metrics, QualityMetrics)
    
    def test_load_and_validate_image_from_path(self, image_processor, temp_image_file):
        """Test loading image from file path."""
        image, quality_metrics = image_processor.load_and_validate_image(temp_image_file)
        
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3
        assert image.shape[2] == 3  # BGR format
        assert isinstance(quality_metrics, QualityMetrics)
    
    def test_load_and_validate_image_invalid_source(self, image_processor):
        """Test loading image with invalid source."""
        with pytest.raises(InvalidImageError):
            image_processor.load_and_validate_image(123)  # Invalid type
        
        with pytest.raises(InvalidImageError):
            image_processor.load_and_validate_image("nonexistent_file.jpg")
    
    def test_preprocess_for_face_detection(self, image_processor, sample_image):
        """Test preprocessing for face detection."""
        # Test with normal size image
        processed = image_processor.preprocess_for_face_detection(sample_image)
        assert processed.shape == sample_image.shape
        
        # Test with large image
        large_image = np.zeros((2000, 2000, 3), dtype=np.uint8)
        processed_large = image_processor.preprocess_for_face_detection(large_image)
        assert max(processed_large.shape[:2]) <= 1024
    
    def test_batch_process_images(self, image_processor, sample_image, temp_image_file):
        """Test batch processing of images."""
        # Mix of different source types
        sources = [
            sample_image,
            temp_image_file,
            sample_image.copy()
        ]
        
        results = image_processor.batch_process_images(sources)
        
        assert len(results) == len(sources)
        
        for image, quality_metrics, error_msg in results:
            if error_msg is None:
                assert isinstance(image, np.ndarray)
                assert isinstance(quality_metrics, QualityMetrics)
            else:
                assert image is None
                assert quality_metrics is None
                assert isinstance(error_msg, str)
    
    def test_batch_process_with_errors(self, image_processor):
        """Test batch processing with invalid images."""
        invalid_sources = [
            "nonexistent.jpg",
            np.array([]),  # Empty array
            None
        ]
        
        results = image_processor.batch_process_images(invalid_sources)
        
        assert len(results) == len(invalid_sources)
        
        # All should fail
        for image, quality_metrics, error_msg in results:
            assert image is None
            assert quality_metrics is None
            assert error_msg is not None
    
    def test_resize_image(self, image_processor, sample_image):
        """Test image resizing functionality."""
        # Test without maintaining aspect ratio
        resized = image_processor.resize_image(sample_image, (100, 150), maintain_aspect_ratio=False)
        assert resized.shape[:2] == (150, 100)  # Height, Width
        
        # Test with maintaining aspect ratio
        resized_ar = image_processor.resize_image(sample_image, (100, 150), maintain_aspect_ratio=True)
        assert resized_ar.shape[:2] == (150, 100)  # Should be padded to target size
    
    def test_normalize_image(self, image_processor, sample_image):
        """Test image normalization."""
        # Test default normalization (0-1)
        normalized = image_processor.normalize_image(sample_image)
        assert normalized.dtype == np.float32
        assert 0.0 <= normalized.min() <= normalized.max() <= 1.0
        
        # Test custom range
        normalized_custom = image_processor.normalize_image(sample_image, target_range=(-1.0, 1.0))
        assert -1.0 <= normalized_custom.min() <= normalized_custom.max() <= 1.0
    
    def test_get_supported_formats(self, image_processor):
        """Test getting supported formats."""
        formats = image_processor.get_supported_formats()
        
        assert isinstance(formats, dict)
        assert 'JPEG' in formats
        assert 'PNG' in formats
        assert '.jpg' in formats['JPEG']
        assert '.png' in formats['PNG']
    
    def test_convert_image_format(self, image_processor, sample_image):
        """Test image format conversion."""
        jpeg_bytes = image_processor.convert_image_format(sample_image, "JPEG", quality=90)
        assert isinstance(jpeg_bytes, bytes)
        assert len(jpeg_bytes) > 0
        
        png_bytes = image_processor.convert_image_format(sample_image, "PNG")
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
    
    def test_get_processing_summary(self, image_processor, sample_image):
        """Test processing summary generation."""
        # Create mock batch results
        good_metrics = QualityMetrics(0.8, 0.8, 0.7, 0.9, 0.2, 1.0, [])
        poor_metrics = QualityMetrics(0.3, 0.3, 0.2, 0.1, 0.8, 0.3, ["Low quality"])
        
        batch_results = [
            (sample_image, good_metrics, None),
            (sample_image, poor_metrics, None),
            (None, None, "Failed to load"),
            (sample_image, good_metrics, None)
        ]
        
        summary = image_processor.get_processing_summary(batch_results)
        
        assert summary['total_processed'] == 4
        assert summary['successful'] == 3
        assert summary['failed'] == 1
        assert summary['success_rate'] == 0.75
        assert 0.0 <= summary['average_quality_score'] <= 1.0
        assert summary['total_quality_warnings'] == 1
        assert len(summary['error_messages']) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])