"""
Real-time face recognition using OpenCV and the face recognition pipeline.
This script captures video from your webcam, detects faces, and identifies known faces
from your database in real-time.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
import argparse
import sys
import os

from face_recognition.pipeline import FaceRecognitionPipeline
from face_recognition.config.manager import ConfigurationManager
from face_recognition.models import FaceRegion, SearchResult, RecognitionRequest, SearchConfig
from face_recognition.exceptions import FaceDetectionError, InvalidImageError, VectorDatabaseError


class RealTimeFaceRecognition:
    """Real-time face recognition using webcam feed."""
    
    def __init__(self, db_path: str, config_profile: str = "development"):
        """
        Initialize real-time face recognition.
        
        Args:
            db_path: Path to the face database
            config_profile: Configuration profile to use
        """
        self.db_path = db_path
        self.config_manager = ConfigurationManager()
        self.config_manager.load_config(profile=config_profile)
        
        # Initialize pipeline
        try:
            self.pipeline = FaceRecognitionPipeline(
                config_manager=self.config_manager,
                db_path=db_path
            )
            
            # Get database info
            db_info = self.pipeline.get_database_info()
            self.has_faces = db_info['total_faces'] > 0
            
        except Exception as e:
            print(f"❌ Error initializing pipeline: {e}")
            raise
        
        # Video capture
        self.cap = None
        self.is_running = False
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Recognition settings
        self.recognition_enabled = self.has_faces
        self.similarity_threshold = 0.7  # Minimum similarity for recognition
        
        print(f"🎯 Real-time Face Recognition Initialized")
        print(f"   Database: {db_path}")
        print(f"   Configuration Profile: {config_profile}")
        print(f"   Known Faces: {db_info['total_faces']}")
        print(f"   Recognition: {'Enabled' if self.recognition_enabled else 'Disabled (no faces in DB)'}")
        print(f"   Detection Method: {self.config_manager.config.face_detection.method}")
    
    def start_camera(self, camera_index: int = 0) -> bool:
        """Start camera capture."""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                print(f"❌ Error: Could not open camera {camera_index}")
                return False
            
            # Set camera properties
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
    
    def recognize_faces_in_frame(self, frame: np.ndarray) -> List[Tuple[FaceRegion, Optional[SearchResult]]]:
        """
        Recognize faces in the current frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of tuples containing face regions and recognition results
        """
        if not self.recognition_enabled:
            return []
        
        try:
            # Create recognition request
            request = RecognitionRequest(
                image_data=frame,
                search_config=SearchConfig(
                    top_k=1,
                    similarity_threshold=self.similarity_threshold,
                    enable_reranking=True
                )
            )
            
            # Perform recognition
            response = self.pipeline.recognize_face(request)
            
            # Pair face regions with recognition results
            results = []
            for i, face_region in enumerate(response.detected_faces):
                # Find corresponding recognition result
                recognition_result = None
                if i < len(response.search_results):
                    # Get the search result for this face
                    search_result = response.search_results[i]
                    if search_result and search_result.similarity_score >= self.similarity_threshold:
                        recognition_result = search_result
                
                results.append((face_region, recognition_result))
            
            return results
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return []
    
    def draw_recognition_results(self, frame: np.ndarray, results: List[Tuple[FaceRegion, Optional[SearchResult]]]) -> np.ndarray:
        """
        Draw face recognition results on the frame.
        
        Args:
            frame: Input frame
            results: Recognition results
            
        Returns:
            Frame with recognition results drawn
        """
        result_frame = frame.copy()
        
        for i, (face_region, recognition_result) in enumerate(results):
            # Determine color based on recognition status
            if recognition_result:
                color = (0, 255, 0)  # Green for recognized faces
                label = f"{recognition_result.metadata.get('person_id', 'Unknown')}"
                similarity = recognition_result.similarity_score
            else:
                color = (0, 165, 255)  # Orange for unknown faces
                label = "Unknown"
                similarity = 0.0
            
            # Draw bounding box
            thickness = 2
            cv2.rectangle(
                result_frame,
                (face_region.x, face_region.y),
                (face_region.x + face_region.width, face_region.y + face_region.height),
                color,
                thickness
            )
            
            # Prepare text
            if recognition_result:
                text_lines = [
                    f"{label}",
                    f"Similarity: {similarity:.2f}",
                    f"Confidence: {face_region.confidence:.2f}"
                ]
            else:
                text_lines = [
                    f"{label}",
                    f"Confidence: {face_region.confidence:.2f}"
                ]
            
            # Draw text background and text
            y_offset = face_region.y - 10
            for line in reversed(text_lines):
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                
                # Background
                cv2.rectangle(
                    result_frame,
                    (face_region.x, y_offset - text_size[1] - 5),
                    (face_region.x + text_size[0] + 10, y_offset + 5),
                    color,
                    -1
                )
                
                # Text
                cv2.putText(
                    result_frame,
                    line,
                    (face_region.x + 5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
                
                y_offset -= text_size[1] + 8
        
        return result_frame
    
    def draw_info_overlay(self, frame: np.ndarray, face_count: int, recognized_count: int) -> np.ndarray:
        """Draw information overlay on the frame."""
        height, width = frame.shape[:2]
        
        # Info background
        overlay_height = 100
        cv2.rectangle(frame, (0, 0), (width, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (width, overlay_height), (255, 255, 255), 1)
        
        # Info text
        info_lines = [
            f"FPS: {self.current_fps:.1f}",
            f"Faces Detected: {face_count}",
            f"Faces Recognized: {recognized_count}",
            f"Recognition: {'ON' if self.recognition_enabled else 'OFF'}",
            f"Similarity Threshold: {self.similarity_threshold:.2f}",
            f"Press 'q' to quit, 'r' to toggle recognition"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 15 + i * 15
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
    
    def save_frame(self, frame: np.ndarray, results: List[Tuple[FaceRegion, Optional[SearchResult]]]):
        """Save current frame with recognition results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"recognition_result_{timestamp}.jpg"
        
        # Draw recognition results on the frame before saving
        frame_with_results = self.draw_recognition_results(frame, results)
        
        cv2.imwrite(filename, frame_with_results)
        
        recognized_count = sum(1 for _, result in results if result is not None)
        print(f"💾 Frame saved as {filename} ({len(results)} faces, {recognized_count} recognized)")
    
    def run(self, camera_index: int = 0, save_frames: bool = False):
        """Run real-time face recognition."""
        if not self.start_camera(camera_index):
            return
        
        self.is_running = True
        print("\n🚀 Starting real-time face recognition...")
        print("   Press 'q' to quit")
        print("   Press 'r' to toggle recognition on/off")
        print("   Press '+' to increase similarity threshold")
        print("   Press '-' to decrease similarity threshold")
        if save_frames:
            print("   Press 's' to save current frame")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                try:
                    if self.recognition_enabled and self.has_faces:
                        # Perform face recognition
                        results = self.recognize_faces_in_frame(frame)
                        
                        # Draw recognition results
                        if results:
                            frame = self.draw_recognition_results(frame, results)
                        
                        # Count recognized faces
                        recognized_count = sum(1 for _, result in results if result is not None)
                        face_count = len(results)
                    else:
                        # Just detect faces without recognition
                        face_regions = self.pipeline.face_detector.detect_faces(frame)
                        
                        # Draw simple bounding boxes
                        for face in face_regions:
                            cv2.rectangle(
                                frame,
                                (face.x, face.y),
                                (face.x + face.width, face.y + face.height),
                                (255, 255, 0),  # Yellow for detection only
                                2
                            )
                        
                        face_count = len(face_regions)
                        recognized_count = 0
                        results = [(face, None) for face in face_regions]
                    
                    # Draw info overlay
                    frame = self.draw_info_overlay(frame, face_count, recognized_count)
                    
                except Exception as e:
                    # Draw error message on frame
                    cv2.putText(
                        frame,
                        f"Error: {str(e)}",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )
                    results = []
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow('Real-time Face Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if self.has_faces:
                        self.recognition_enabled = not self.recognition_enabled
                        print(f"Recognition: {'ON' if self.recognition_enabled else 'OFF'}")
                    else:
                        print("No faces in database - recognition unavailable")
                elif key == ord('+') or key == ord('='):
                    self.similarity_threshold = min(1.0, self.similarity_threshold + 0.05)
                    print(f"Similarity threshold: {self.similarity_threshold:.2f}")
                elif key == ord('-'):
                    self.similarity_threshold = max(0.0, self.similarity_threshold - 0.05)
                    print(f"Similarity threshold: {self.similarity_threshold:.2f}")
                elif key == ord('s') and save_frames:
                    self.save_frame(frame, results)
        
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user")
        
        finally:
            self.stop_camera()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Real-time face recognition using OpenCV")
    parser.add_argument(
        "--database", 
        type=str, 
        default="demo_face_db",
        help="Path to face database (default: demo_face_db)"
    )
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
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.7,
        help="Similarity threshold for recognition (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    # Check if database exists
    if not os.path.exists(args.database):
        print(f"❌ Error: Database path '{args.database}' does not exist")
        print("Available databases:")
        for item in os.listdir("."):
            if os.path.isdir(item) and any(f.endswith(('.index', '.faiss', '.json')) for f in os.listdir(item) if os.path.isfile(os.path.join(item, f))):
                print(f"  - {item}")
        sys.exit(1)
    
    try:
        # Create and run face recognition
        recognizer = RealTimeFaceRecognition(
            db_path=args.database,
            config_profile=args.profile
        )
        recognizer.similarity_threshold = args.threshold
        recognizer.run(camera_index=args.camera, save_frames=args.save_frames)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()