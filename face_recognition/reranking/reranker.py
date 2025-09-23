"""Reranking module for improving face recognition search result accuracy."""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from ..models import SearchResult, RerankingFeatures
from ..exceptions import RerankingError


class Reranker:
    """Advanced reranking system for improving search result quality."""
    
    def __init__(self, enable_quality_scoring: bool = True, 
                 enable_pose_analysis: bool = True,
                 enable_illumination_analysis: bool = True):
        """
        Initialize the reranker.
        
        Args:
            enable_quality_scoring: Enable face quality assessment
            enable_pose_analysis: Enable pose angle analysis
            enable_illumination_analysis: Enable illumination quality analysis
        """
        self.enable_quality_scoring = enable_quality_scoring
        self.enable_pose_analysis = enable_pose_analysis
        self.enable_illumination_analysis = enable_illumination_analysis
        
        # Reranking weights
        self.weights = {
            'similarity': 0.6,      # Original similarity score weight
            'quality': 0.2,         # Face quality weight
            'pose': 0.1,           # Pose quality weight
            'illumination': 0.1    # Illumination quality weight
        }
        
        # Statistics tracking
        self.reranking_stats = {
            'total_rerankings': 0,
            'average_improvement': 0.0,
            'quality_improvements': 0,
            'pose_improvements': 0,
            'illumination_improvements': 0
        }
    
    def rerank_results(self, results: List[SearchResult], 
                      query_features: Optional[RerankingFeatures] = None,
                      result_images: Optional[List[np.ndarray]] = None) -> List[SearchResult]:
        """
        Rerank search results using additional features.
        
        Args:
            results: Original search results from similarity search
            query_features: Features extracted from query image
            result_images: Images corresponding to search results (for feature extraction)
            
        Returns:
            Reranked list of SearchResult objects
            
        Raises:
            RerankingError: If reranking fails
        """
        if not results:
            return results
        
        try:
            # Track original order for comparison
            original_scores = [r.similarity_score for r in results]
            
            # Extract features for result images if provided
            result_features = []
            if result_images:
                for img in result_images:
                    if img is not None:
                        features = self.extract_reranking_features(img)
                        result_features.append(features)
                    else:
                        # Create default features if image not available
                        features = RerankingFeatures(
                            face_quality_score=0.5,
                            landmark_confidence=0.5,
                            pose_angle=0.0,
                            illumination_score=0.5
                        )
                        result_features.append(features)
            else:
                # Create default features for all results
                result_features = [
                    RerankingFeatures(
                        face_quality_score=0.5,
                        landmark_confidence=0.5,
                        pose_angle=0.0,
                        illumination_score=0.5
                    ) for _ in results
                ]
            
            # Calculate reranking scores
            reranked_results = []
            for result, features in zip(results, result_features):
                rerank_score = self._calculate_rerank_score(
                    result.similarity_score, 
                    features, 
                    query_features
                )
                
                # Create new result with rerank score
                reranked_result = SearchResult(
                    embedding_id=result.embedding_id,
                    similarity_score=result.similarity_score,
                    metadata=result.metadata,
                    rerank_score=rerank_score
                )
                reranked_results.append(reranked_result)
            
            # Sort by rerank score (highest first)
            reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)
            
            # Update statistics
            self._update_reranking_stats(original_scores, reranked_results)
            
            return reranked_results
            
        except Exception as e:
            # Fallback to original results if reranking fails
            print(f"⚠️ Reranking failed, using original results: {e}")
            return results
    
    def extract_reranking_features(self, face_image: np.ndarray) -> RerankingFeatures:
        """
        Extract additional features for reranking from a face image.
        
        Args:
            face_image: Face image (224x224x3, normalized to [0,1] or uint8)
            
        Returns:
            RerankingFeatures object with extracted features
            
        Raises:
            RerankingError: If feature extraction fails
        """
        try:
            # Convert to uint8 if needed
            if face_image.dtype == np.float32:
                img_uint8 = (face_image * 255).astype(np.uint8)
            else:
                img_uint8 = face_image.astype(np.uint8)
            
            # Convert to grayscale for some analyses
            if len(img_uint8.shape) == 3:
                gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_uint8
            
            # Extract individual features
            quality_score = self._extract_face_quality_score(gray) if self.enable_quality_scoring else 0.5
            landmark_confidence = self._extract_landmark_confidence(gray)
            pose_angle = self._extract_pose_angle(gray) if self.enable_pose_analysis else 0.0
            illumination_score = self._extract_illumination_score(gray) if self.enable_illumination_analysis else 0.5
            
            return RerankingFeatures(
                face_quality_score=quality_score,
                landmark_confidence=landmark_confidence,
                pose_angle=pose_angle,
                illumination_score=illumination_score
            )
            
        except Exception as e:
            raise RerankingError(f"Failed to extract reranking features: {str(e)}")
    
    def _calculate_rerank_score(self, similarity_score: float, 
                               result_features: RerankingFeatures,
                               query_features: Optional[RerankingFeatures] = None) -> float:
        """Calculate the final reranking score."""
        # Start with original similarity score
        rerank_score = similarity_score * self.weights['similarity']
        
        # Add quality component
        if self.enable_quality_scoring:
            quality_component = result_features.face_quality_score * self.weights['quality']
            rerank_score += quality_component
        
        # Add pose component (prefer frontal faces)
        if self.enable_pose_analysis:
            # Convert pose angle to quality score (0° = best, 90° = worst)
            pose_quality = max(0.0, 1.0 - abs(result_features.pose_angle) / 90.0)
            pose_component = pose_quality * self.weights['pose']
            rerank_score += pose_component
        
        # Add illumination component
        if self.enable_illumination_analysis:
            illumination_component = result_features.illumination_score * self.weights['illumination']
            rerank_score += illumination_component
        
        # Add landmark confidence bonus
        landmark_bonus = result_features.landmark_confidence * 0.05  # Small bonus
        rerank_score += landmark_bonus
        
        # Compare with query features if available
        if query_features:
            # Bonus for similar quality levels
            quality_similarity = 1.0 - abs(result_features.face_quality_score - query_features.face_quality_score)
            quality_bonus = quality_similarity * 0.05
            rerank_score += quality_bonus
            
            # Bonus for similar pose angles
            pose_similarity = 1.0 - abs(result_features.pose_angle - query_features.pose_angle) / 180.0
            pose_bonus = pose_similarity * 0.03
            rerank_score += pose_bonus
        
        # Ensure score is in valid range
        return float(np.clip(rerank_score, 0.0, 1.0))
    
    def _extract_face_quality_score(self, gray_image: np.ndarray) -> float:
        """Extract face quality score based on sharpness and contrast."""
        try:
            # Calculate sharpness using Laplacian variance
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Normalize sharpness (values above 100 are considered sharp)
            sharpness_score = min(1.0, sharpness / 100.0)
            
            # Calculate contrast using standard deviation
            contrast = np.std(gray_image.astype(np.float32))
            contrast_score = min(1.0, contrast / 64.0)  # Normalize by typical std
            
            # Calculate brightness distribution quality
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            hist_normalized = hist / hist.sum()
            
            # Prefer images with good brightness distribution (not too dark/bright)
            brightness_mean = np.mean(gray_image)
            brightness_quality = 1.0 - abs(brightness_mean - 128) / 128.0
            
            # Combine quality metrics
            quality_score = (sharpness_score * 0.4 + 
                           contrast_score * 0.3 + 
                           brightness_quality * 0.3)
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception:
            return 0.5  # Default quality if calculation fails
    
    def _extract_landmark_confidence(self, gray_image: np.ndarray) -> float:
        """Extract landmark confidence based on edge detection."""
        try:
            # Use Canny edge detection to find facial features
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Calculate edge density in key facial regions
            h, w = gray_image.shape
            
            # Define regions of interest (eyes, nose, mouth areas)
            eye_region = edges[int(h*0.3):int(h*0.5), int(w*0.2):int(w*0.8)]
            nose_region = edges[int(h*0.4):int(h*0.7), int(w*0.4):int(w*0.6)]
            mouth_region = edges[int(h*0.6):int(h*0.8), int(w*0.3):int(w*0.7)]
            
            # Calculate edge density for each region
            eye_density = np.sum(eye_region > 0) / eye_region.size
            nose_density = np.sum(nose_region > 0) / nose_region.size
            mouth_density = np.sum(mouth_region > 0) / mouth_region.size
            
            # Combine densities (higher density = more landmarks detected)
            landmark_confidence = (eye_density * 0.4 + 
                                 nose_density * 0.3 + 
                                 mouth_density * 0.3)
            
            # Normalize to [0, 1] range
            landmark_confidence = min(1.0, landmark_confidence * 10.0)
            
            return float(landmark_confidence)
            
        except Exception:
            return 0.5  # Default confidence if calculation fails
    
    def _extract_pose_angle(self, gray_image: np.ndarray) -> float:
        """Estimate pose angle based on facial symmetry."""
        try:
            h, w = gray_image.shape
            
            # Split face into left and right halves
            left_half = gray_image[:, :w//2]
            right_half = gray_image[:, w//2:]
            
            # Flip right half to compare with left
            right_half_flipped = cv2.flip(right_half, 1)
            
            # Resize to match if needed
            if left_half.shape != right_half_flipped.shape:
                min_width = min(left_half.shape[1], right_half_flipped.shape[1])
                left_half = left_half[:, :min_width]
                right_half_flipped = right_half_flipped[:, :min_width]
            
            # Calculate difference between halves
            diff = np.abs(left_half.astype(np.float32) - right_half_flipped.astype(np.float32))
            asymmetry = np.mean(diff)
            
            # Convert asymmetry to pose angle estimate (0-45 degrees)
            # Higher asymmetry suggests more pose deviation
            pose_angle = min(45.0, asymmetry / 2.0)
            
            return float(pose_angle)
            
        except Exception:
            return 0.0  # Default to frontal pose if calculation fails
    
    def _extract_illumination_score(self, gray_image: np.ndarray) -> float:
        """Extract illumination quality score."""
        try:
            # Calculate overall brightness
            mean_brightness = np.mean(gray_image)
            
            # Calculate brightness uniformity
            brightness_std = np.std(gray_image.astype(np.float32))
            
            # Prefer moderate brightness (around 128) and good uniformity
            brightness_quality = 1.0 - abs(mean_brightness - 128) / 128.0
            uniformity_quality = min(1.0, 64.0 / (brightness_std + 1.0))
            
            # Check for over/under exposure
            overexposed_pixels = np.sum(gray_image > 240) / gray_image.size
            underexposed_pixels = np.sum(gray_image < 15) / gray_image.size
            
            exposure_penalty = (overexposed_pixels + underexposed_pixels) * 2.0
            
            # Combine illumination metrics
            illumination_score = (brightness_quality * 0.4 + 
                                uniformity_quality * 0.4 + 
                                (1.0 - exposure_penalty) * 0.2)
            
            return float(np.clip(illumination_score, 0.0, 1.0))
            
        except Exception:
            return 0.5  # Default illumination score if calculation fails
    
    def _update_reranking_stats(self, original_scores: List[float], 
                               reranked_results: List[SearchResult]):
        """Update reranking performance statistics."""
        try:
            self.reranking_stats['total_rerankings'] += 1
            
            # Calculate improvement metrics
            reranked_scores = [r.rerank_score for r in reranked_results]
            
            # Simple improvement metric: compare top result scores
            if original_scores and reranked_scores:
                original_top = max(original_scores)
                reranked_top = max(reranked_scores)
                improvement = reranked_top - original_top
                
                # Update average improvement
                total_rerankings = self.reranking_stats['total_rerankings']
                current_avg = self.reranking_stats['average_improvement']
                new_avg = ((current_avg * (total_rerankings - 1)) + improvement) / total_rerankings
                self.reranking_stats['average_improvement'] = new_avg
                
                # Count improvements by type
                if improvement > 0.01:  # Significant improvement threshold
                    self.reranking_stats['quality_improvements'] += 1
            
        except Exception:
            pass  # Don't fail reranking due to stats update issues
    
    def get_reranking_statistics(self) -> Dict:
        """Get reranking performance statistics."""
        return {
            'total_rerankings': self.reranking_stats['total_rerankings'],
            'average_improvement': self.reranking_stats['average_improvement'],
            'quality_improvements': self.reranking_stats['quality_improvements'],
            'improvement_rate': (self.reranking_stats['quality_improvements'] / 
                               max(1, self.reranking_stats['total_rerankings'])),
            'weights': self.weights.copy(),
            'enabled_features': {
                'quality_scoring': self.enable_quality_scoring,
                'pose_analysis': self.enable_pose_analysis,
                'illumination_analysis': self.enable_illumination_analysis
            }
        }
    
    def set_reranking_weights(self, similarity: float = 0.6, quality: float = 0.2,
                             pose: float = 0.1, illumination: float = 0.1):
        """
        Set custom weights for reranking components.
        
        Args:
            similarity: Weight for original similarity score
            quality: Weight for face quality score
            pose: Weight for pose quality
            illumination: Weight for illumination quality
        """
        # Normalize weights to sum to 1.0
        total = similarity + quality + pose + illumination
        if total > 0:
            self.weights = {
                'similarity': similarity / total,
                'quality': quality / total,
                'pose': pose / total,
                'illumination': illumination / total
            }
        
        print(f"✅ Updated reranking weights: {self.weights}")
    
    def enable_feature(self, feature_name: str, enabled: bool = True):
        """Enable or disable specific reranking features."""
        if feature_name == "quality_scoring":
            self.enable_quality_scoring = enabled
        elif feature_name == "pose_analysis":
            self.enable_pose_analysis = enabled
        elif feature_name == "illumination_analysis":
            self.enable_illumination_analysis = enabled
        else:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        print(f"✅ {feature_name} {'enabled' if enabled else 'disabled'}")


class AdvancedReranker(Reranker):
    """Advanced reranker with machine learning-based improvements."""
    
    def __init__(self, **kwargs):
        """Initialize advanced reranker."""
        super().__init__(**kwargs)
        
        # Additional advanced features
        self.enable_context_analysis = True
        self.enable_demographic_consistency = True
        
        # Learning components (simplified for demo)
        self.feature_importance = {
            'quality': 1.0,
            'pose': 0.8,
            'illumination': 0.9,
            'context': 0.6
        }
    
    def rerank_with_context(self, results: List[SearchResult], 
                           context_info: Dict = None) -> List[SearchResult]:
        """Rerank results with additional context information."""
        # This would implement more sophisticated reranking
        # using context like time of day, location, etc.
        
        # For now, use the base reranking
        return self.rerank_results(results)
    
    def learn_from_feedback(self, query_id: str, selected_result_id: str, 
                           all_results: List[SearchResult]):
        """Learn from user feedback to improve reranking."""
        # This would implement learning from user selections
        # to improve future reranking performance
        
        print(f"📚 Learning from feedback: query {query_id}, selected {selected_result_id}")
        
        # Simple learning: adjust weights based on selected result characteristics
        # In a real system, this would use more sophisticated ML techniques