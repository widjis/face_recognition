"""Image format handling and validation utilities."""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path
import io
from PIL import Image, ImageFile
import magic

from ..exceptions import InvalidImageError


class ImageFormatHandler:
    """
    Handles various image formats and provides format detection and conversion.
    
    Supports common formats: JPEG, PNG, BMP, TIFF, WebP
    """
    
    SUPPORTED_FORMATS = {
        'JPEG': ['.jpg', '.jpeg', '.jpe'],
        'PNG': ['.png'],
        'BMP': ['.bmp'],
        'TIFF': ['.tiff', '.tif'],
        'WEBP': ['.webp']
    }
    
    # MIME type mappings
    MIME_TYPES = {
        'image/jpeg': 'JPEG',
        'image/png': 'PNG',
        'image/bmp': 'BMP',
        'image/tiff': 'TIFF',
        'image/webp': 'WEBP'
    }
    
    def __init__(self):
        """Initialize the format handler."""
        # Enable loading of truncated images
        ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    def detect_format_from_path(self, file_path: str) -> Optional[str]:
        """
        Detect image format from file extension.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Format name (e.g., 'JPEG', 'PNG') or None if unsupported
        """
        try:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            for format_name, extensions in self.SUPPORTED_FORMATS.items():
                if extension in extensions:
                    return format_name
            
            return None
            
        except Exception:
            return None
    
    def detect_format_from_bytes(self, image_bytes: bytes) -> Optional[str]:
        """
        Detect image format from byte content using magic numbers.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Format name or None if unsupported
        """
        try:
            # Try using python-magic if available
            try:
                import magic
                mime_type = magic.from_buffer(image_bytes, mime=True)
                return self.MIME_TYPES.get(mime_type)
            except ImportError:
                pass
            
            # Fallback to manual magic number detection
            if len(image_bytes) < 12:
                return None
            
            # JPEG magic numbers
            if image_bytes[:2] == b'\xff\xd8':
                return 'JPEG'
            
            # PNG magic number
            if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                return 'PNG'
            
            # BMP magic number
            if image_bytes[:2] == b'BM':
                return 'BMP'
            
            # TIFF magic numbers (little and big endian)
            if image_bytes[:4] in [b'II*\x00', b'MM\x00*']:
                return 'TIFF'
            
            # WebP magic number
            if image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
                return 'WEBP'
            
            return None
            
        except Exception:
            return None
    
    def is_supported_format(self, file_path: str) -> bool:
        """
        Check if the image format is supported.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            True if format is supported, False otherwise
        """
        return self.detect_format_from_path(file_path) is not None
    
    def load_image_from_path(self, file_path: str) -> np.ndarray:
        """
        Load an image from file path with format validation.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Image as numpy array in BGR format
            
        Raises:
            InvalidImageError: If image cannot be loaded or format is unsupported
        """
        try:
            if not Path(file_path).exists():
                raise InvalidImageError(f"Image file does not exist: {file_path}")
            
            # Check format support
            if not self.is_supported_format(file_path):
                detected_format = self.detect_format_from_path(file_path)
                raise InvalidImageError(
                    f"Unsupported image format: {detected_format or 'unknown'}"
                )
            
            # Load image using OpenCV
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            
            if image is None:
                # Fallback to PIL for better format support
                try:
                    pil_image = Image.open(file_path)
                    # Convert to RGB if necessary
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # Convert PIL to OpenCV format (RGB to BGR)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    
                except Exception as e:
                    raise InvalidImageError(f"Failed to load image: {str(e)}")
            
            return image
            
        except InvalidImageError:
            raise
        except Exception as e:
            raise InvalidImageError(f"Failed to load image from {file_path}: {str(e)}")
    
    def load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Load an image from byte data with format validation.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Image as numpy array in BGR format
            
        Raises:
            InvalidImageError: If image cannot be loaded or format is unsupported
        """
        try:
            if not image_bytes:
                raise InvalidImageError("Image bytes are empty")
            
            # Check format support
            detected_format = self.detect_format_from_bytes(image_bytes)
            if detected_format is None:
                raise InvalidImageError("Unsupported or unrecognized image format")
            
            # Try OpenCV first
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                # Fallback to PIL
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    # Convert to RGB if necessary
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # Convert PIL to OpenCV format (RGB to BGR)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    
                except Exception as e:
                    raise InvalidImageError(f"Failed to decode image: {str(e)}")
            
            return image
            
        except InvalidImageError:
            raise
        except Exception as e:
            raise InvalidImageError(f"Failed to load image from bytes: {str(e)}")
    
    def convert_format(self, 
                      image: np.ndarray, 
                      target_format: str,
                      quality: int = 95) -> bytes:
        """
        Convert image to specified format.
        
        Args:
            image: Input image as numpy array
            target_format: Target format ('JPEG', 'PNG', etc.)
            quality: Quality for lossy formats (0-100)
            
        Returns:
            Image bytes in target format
            
        Raises:
            InvalidImageError: If conversion fails
        """
        try:
            if target_format not in self.SUPPORTED_FORMATS:
                raise InvalidImageError(f"Unsupported target format: {target_format}")
            
            if target_format == 'JPEG':
                # JPEG compression
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                success, encoded_img = cv2.imencode('.jpg', image, encode_params)
            elif target_format == 'PNG':
                # PNG compression (lossless)
                compression_level = max(0, min(9, (100 - quality) // 10))
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
                success, encoded_img = cv2.imencode('.png', image, encode_params)
            elif target_format == 'BMP':
                success, encoded_img = cv2.imencode('.bmp', image)
            elif target_format == 'WEBP':
                encode_params = [cv2.IMWRITE_WEBP_QUALITY, quality]
                success, encoded_img = cv2.imencode('.webp', image, encode_params)
            else:
                # Use PIL for other formats
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                buffer = io.BytesIO()
                
                if target_format == 'TIFF':
                    pil_image.save(buffer, format='TIFF')
                else:
                    pil_image.save(buffer, format=target_format)
                
                return buffer.getvalue()
            
            if not success:
                raise InvalidImageError(f"Failed to encode image to {target_format}")
            
            return encoded_img.tobytes()
            
        except InvalidImageError:
            raise
        except Exception as e:
            raise InvalidImageError(f"Failed to convert image to {target_format}: {str(e)}")
    
    def get_image_info(self, image: np.ndarray) -> dict:
        """
        Get information about an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with image information
        """
        try:
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) > 2 else 1
            
            return {
                'width': width,
                'height': height,
                'channels': channels,
                'dtype': str(image.dtype),
                'size_bytes': image.nbytes,
                'aspect_ratio': width / height if height > 0 else 0,
                'total_pixels': width * height
            }
            
        except Exception as e:
            return {
                'error': f"Failed to get image info: {str(e)}"
            }