"""
Real-time face detection using OpenCV and the current face recognition pipeline configuration.
This script captures video from your webcam and detects faces in real-time using the same
face detection module used in the pipeline.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
import argparse
import sys

from face_recognition.face_detection import FaceDetector
from face_recognition.config.manager import ConfigurationManager
from face_recognition.models import FaceRegion
from face_recognition.exceptions import FaceDetectionError, InvalidImageError


class RealTimeFaceDetection:
    """Real-time face detection using webcam feed."""
    
    def __init__(self, config_profile: str = "development"):
        """
        Initialize real-time face detection.
        
        Args:
            config_profile: Configuration profile to use (development, production, high_accuracy, high_speed)
        """
        self.config_manager = ConfigurationManager()
        self.config_manager.load_config(profile=config_profile)
        
        # Initialize face detector with current config
        self.face_detector = FaceDetector(self.config_manager.config.face_detection)
        
        # Video capture
        self.cap = None
        self.is_running = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        print(f"🎯 Real-time Face Detection Initialized")
        print(f"   Configuration Profile: {config_profile}")
        print(f"   Detection Method: {self.config_manager.config.face_detection.method}")
        print(f"   Scale Factor: {self.config_manager.config.face_detection.scale_factor}")
        print(f"   Min Face Size: {self.config_manager.config.face_detection.min_face_size}")
    
    def start_camera(self, camera_index: int = 0) -> bool:
        """
        Start camera capture.
        
        Args:
            camera_index: Camera index (0 for default camera)
            
        Returns:
            True if camera started successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                print(f"❌ Error: Could not open camera {camera_index}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"✅ Camera {camera_index} started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture and cleanup."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        self.is_running = False
        print("📹 Camera stopped")
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_face_regions(self, frame: np.ndarray, face_regions: List[FaceRegion]) -> np.ndarray:
        """
        Draw face detection results on the frame.
        
        Args:
            frame: Input frame
            face_regions: Detected face regions
            
        Returns:
            Frame with face regions drawn
        """
        result_frame = frame.copy()
        
        for i, face in enumerate(face_regions):
            # Draw bounding box
            color = (0, 255, 0)  # Green for detected faces
            thickness = 2
            
            cv2.rectangle(
                result_frame,
                (face.x, face.y),
                (face.x + face.width, face.y + face.height),
                color,
                thickness
            )
            
            # Draw confidence score
            confidence_text = f"Face {i+1}: {face.confidence:.2f}"
            text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            
            # Background for text
            cv2.rectangle(
                result_frame,
                (face.x, face.y - text_size[1] - 10),
                (face.x + text_size[0] + 10, face.y),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                result_frame,
                confidence_text,
                (face.x + 5, face.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
        
        return result_frame
    
    def draw_info_overlay(self, frame: np.ndarray, face_count: int) -> np.ndarray:
        """
        Draw information overlay on the frame.
        
        Args:
            frame: Input frame
            face_count: Number of detected faces
            
        Returns:
            Frame with info overlay
        """
        height, width = frame.shape[:2]
        
        # Info background
        overlay_height = 80
        cv2.rectangle(frame, (0, 0), (width, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (width, overlay_height), (255, 255, 255), 1)
        
        # Info text
        info_lines = [
            f"FPS: {self.current_fps:.1f}",
            f"Faces Detected: {face_count}",
            f"Model: {self.config_manager.config.face_detection.model_name}",
            f"Press 'q' to quit, 's' to save frame"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 20 + i * 15
            cv2.putText(
                frame,
                line,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return frame
    
    def save_frame(self, frame: np.ndarray, face_regions: List[FaceRegion]):
        """Save current frame with detected faces."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detected_faces_{timestamp}.jpg"
        
        # Draw face regions on the frame before saving
        frame_with_faces = self.draw_face_regions(frame, face_regions)
        
        cv2.imwrite(filename, frame_with_faces)
        print(f"💾 Frame saved as {filename} ({len(face_regions)} faces detected)")
    
    def run(self, camera_index: int = 0, save_frames: bool = False):
        """
        Run real-time face detection.
        
        Args:
            camera_index: Camera index to use
            save_frames: Whether to enable frame saving with 's' key
        """
        if not self.start_camera(camera_index):
            return
        
        self.is_running = True
        print("\n🚀 Starting real-time face detection...")
        print("   Press 'q' to quit")
        if save_frames:
            print("   Press 's' to save current frame")
        print("   Press 'c' to show/hide confidence scores")
        
        show_confidence = True
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                try:
                    # Detect faces using the pipeline's face detector
                    face_regions = self.face_detector.detect_faces(frame)
                    
                    # Draw face regions if any detected
                    if face_regions and show_confidence:
                        frame = self.draw_face_regions(frame, face_regions)
                    elif face_regions:
                        # Just draw boxes without confidence scores
                        for face in face_regions:
                            cv2.rectangle(
                                frame,
                                (face.x, face.y),
                                (face.x + face.width, face.y + face.height),
                                (0, 255, 0),
                                2
                            )
                    
                    # Draw info overlay
                    frame = self.draw_info_overlay(frame, len(face_regions))
                    
                except (FaceDetectionError, InvalidImageError) as e:
                    # Draw error message on frame
                    cv2.putText(
                        frame,
                        f"Detection Error: {str(e)}",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )
                    face_regions = []
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow('Real-time Face Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and save_frames and face_regions:
                    self.save_frame(frame, face_regions)
                elif key == ord('c'):
                    show_confidence = not show_confidence
                    print(f"Confidence display: {'ON' if show_confidence else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        
        finally:
            self.stop_camera()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Real-time face detection using OpenCV")
    parser.add_argument(
        "--camera", 
        type=int, 
        default=0, 
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--profile", 
        type=str, 
        default="development",
        choices=["development", "production", "high_accuracy", "high_speed"],
        help="Configuration profile to use (default: development)"
    )
    parser.add_argument(
        "--save-frames", 
        action="store_true",
        help="Enable frame saving with 's' key"
    )
    
    args = parser.parse_args()
    
    try:
        # Create and run face detection
        detector = RealTimeFaceDetection(config_profile=args.profile)
        detector.run(camera_index=args.camera, save_frames=args.save_frames)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()