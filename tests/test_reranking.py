"""Tests for reranking module."""

import pytest
import numpy as np
from datetime import datetime
from face_recognition.reranking import Reranker
from face_recognition.models import SearchResult, RerankingFeatures
from face_recognition.exceptions import RerankingError


class TestReranker:
    """Test cases for Reranker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reranker = Reranker()
        
        # Create test search results
        self.test_results = [
            SearchResult("emb_001", 0.95, {"name": "Alice", "quality": "high"}),
            SearchResult("emb_002", 0.87, {"name": "Bob", "quality": "medium"}),
            SearchResult("emb_003", 0.82, {"name": "Charlie", "quality": "low"}),
        ]
        
        # Create test reranking features
        self.test_features = [
            RerankingFeatures(0.9, 0.8, 5.0, 0.85),   # High quality
            RerankingFeatures(0.7, 0.6, 15.0, 0.65),  # Medium quality
            RerankingFeatures(0.4, 0.3, 30.0, 0.45),  # Low quality
        ]
    
    def test_reranker_initialization(self):
        """Test reranker initialization."""
        reranker = Reranker(
            enable_quality_scoring=True,
            enable_pose_analysis=True,
            enable_illumination_analysis=True
        )
        
        assert reranker.enable_quality_scoring is True
        assert reranker.enable_pose_analysis is True
        assert reranker.enable_illumination_analysis is True
        assert 'similarity' in reranker.weights
        assert 'quality' in reranker.weights
    
    def test_rerank_results_basic(self):
        """Test basic reranking functionality."""
        # Rerank without additional features (should use defaults)
        reranked = self.reranker.rerank_results(self.test_results)
        
        assert len(reranked) == len(self.test_results)
        assert all(isinstance(r, SearchResult) for r in reranked)
        assert all(hasattr(r, 'rerank_score') for r in reranked)
        assert all(r.rerank_score is not None for r in reranked)
    
    def test_rerank_results_with_features(self):
        """Test reranking with provided features."""
        # Create test images (synthetic)
        test_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        ]
        
        reranked = self.reranker.rerank_results(
            self.test_results, 
            result_images=test_images
        )
        
        assert len(reranked) == len(self.test_results)
        assert all(r.rerank_score is not None for r in reranked)
        
        # Results should be sorted by rerank_score (highest first)
        rerank_scores = [r.rerank_score for r in reranked]
        assert rerank_scores == sorted(rerank_scores, reverse=True)
    
    def test_rerank_results_empty_list(self):
        """Test reranking with empty results list."""
        reranked = self.reranker.rerank_results([])
        assert reranked == []
    
    def test_extract_reranking_features_uint8(self):
        """Test feature extraction from uint8 image."""
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        features = self.reranker.extract_reranking_features(test_image)
        
        assert isinstance(features, RerankingFeatures)
        assert 0.0 <= features.face_quality_score <= 1.0
        assert 0.0 <= features.landmark_confidence <= 1.0
        assert features.pose_angle >= 0.0
        assert 0.0 <= features.illumination_score <= 1.0
    
    def test_extract_reranking_features_float32(self):
        """Test feature extraction from float32 image."""
        # Create test image (normalized)
        test_image = np.random.rand(224, 224, 3).astype(np.float32)
        
        features = self.reranker.extract_reranking_features(test_image)
        
        assert isinstance(features, RerankingFeatures)
        assert 0.0 <= features.face_quality_score <= 1.0
        assert 0.0 <= features.landmark_confidence <= 1.0
        assert features.pose_angle >= 0.0
        assert 0.0 <= features.illumination_score <= 1.0
    
    def test_extract_reranking_features_grayscale(self):
        """Test feature extraction from grayscale image."""
        # Create grayscale test image
        test_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        features = self.reranker.extract_reranking_features(test_image)
        
        assert isinstance(features, RerankingFeatures)
        assert 0.0 <= features.face_quality_score <= 1.0
    
    def test_calculate_rerank_score(self):
        """Test rerank score calculation."""
        result_features = RerankingFeatures(0.8, 0.7, 10.0, 0.75)
        query_features = RerankingFeatures(0.9, 0.8, 5.0, 0.85)
        
        rerank_score = self.reranker._calculate_rerank_score(
            0.85, result_features, query_features
        )
        
        assert isinstance(rerank_score, float)
        assert 0.0 <= rerank_score <= 1.0
    
    def test_calculate_rerank_score_without_query(self):
        """Test rerank score calculation without query features."""
        result_features = RerankingFeatures(0.8, 0.7, 10.0, 0.75)
        
        rerank_score = self.reranker._calculate_rerank_score(
            0.85, result_features, None
        )
        
        assert isinstance(rerank_score, float)
        assert 0.0 <= rerank_score <= 1.0
    
    def test_face_quality_score_extraction(self):
        """Test face quality score extraction."""
        # Create test image with known characteristics
        test_image = np.ones((224, 224), dtype=np.uint8) * 128  # Medium gray
        
        quality_score = self.reranker._extract_face_quality_score(test_image)
        
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0
    
    def test_landmark_confidence_extraction(self):
        """Test landmark confidence extraction."""
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        confidence = self.reranker._extract_landmark_confidence(test_image)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_pose_angle_extraction(self):
        """Test pose angle extraction."""
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        pose_angle = self.reranker._extract_pose_angle(test_image)
        
        assert isinstance(pose_angle, float)
        assert pose_angle >= 0.0
    
    def test_illumination_score_extraction(self):
        """Test illumination score extraction."""
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        illumination_score = self.reranker._extract_illumination_score(test_image)
        
        assert isinstance(illumination_score, float)
        assert 0.0 <= illumination_score <= 1.0
    
    def test_set_reranking_weights(self):
        """Test setting custom reranking weights."""
        original_weights = self.reranker.weights.copy()
        
        self.reranker.set_reranking_weights(
            similarity=0.5, quality=0.3, pose=0.1, illumination=0.1
        )
        
        # Weights should be normalized
        total_weight = sum(self.reranker.weights.values())
        assert abs(total_weight - 1.0) < 1e-6
        
        # Check individual weights
        assert self.reranker.weights['similarity'] == 0.5
        assert self.reranker.weights['quality'] == 0.3
    
    def test_enable_disable_features(self):
        """Test enabling and disabling reranking features."""
        # Test enabling/disabling quality scoring
        self.reranker.enable_feature("quality_scoring", False)
        assert self.reranker.enable_quality_scoring is False
        
        self.reranker.enable_feature("quality_scoring", True)
        assert self.reranker.enable_quality_scoring is True
        
        # Test enabling/disabling pose analysis
        self.reranker.enable_feature("pose_analysis", False)
        assert self.reranker.enable_pose_analysis is False
        
        # Test enabling/disabling illumination analysis
        self.reranker.enable_feature("illumination_analysis", False)
        assert self.reranker.enable_illumination_analysis is False
    
    def test_enable_invalid_feature(self):
        """Test enabling invalid feature raises error."""
        with pytest.raises(ValueError, match="Unknown feature"):
            self.reranker.enable_feature("invalid_feature", True)
    
    def test_get_reranking_statistics(self):
        """Test getting reranking statistics."""
        # Perform some reranking to generate stats
        self.reranker.rerank_results(self.test_results)
        
        stats = self.reranker.get_reranking_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_rerankings' in stats
        assert 'average_improvement' in stats
        assert 'improvement_rate' in stats
        assert 'weights' in stats
        assert 'enabled_features' in stats
        
        assert stats['total_rerankings'] > 0
    
    def test_reranking_improves_order(self):
        """Test that reranking can improve result ordering."""
        # Create results where similarity doesn't match quality
        poor_quality_high_sim = SearchResult("emb_001", 0.95, {"name": "Poor Quality"})
        good_quality_low_sim = SearchResult("emb_002", 0.75, {"name": "Good Quality"})
        
        results = [poor_quality_high_sim, good_quality_low_sim]
        
        # Create corresponding images with different qualities
        poor_image = np.ones((224, 224, 3), dtype=np.uint8) * 50   # Dark, poor quality
        good_image = np.ones((224, 224, 3), dtype=np.uint8) * 128  # Better quality
        
        images = [poor_image, good_image]
        
        reranked = self.reranker.rerank_results(results, result_images=images)
        
        # Check that reranking was applied
        assert all(r.rerank_score is not None for r in reranked)
        
        # Results should still be ordered by rerank_score
        rerank_scores = [r.rerank_score for r in reranked]
        assert rerank_scores == sorted(rerank_scores, reverse=True)
    
    def test_reranking_with_disabled_features(self):
        """Test reranking with some features disabled."""
        # Disable some features
        reranker = Reranker(
            enable_quality_scoring=False,
            enable_pose_analysis=False,
            enable_illumination_analysis=True
        )
        
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        reranked = reranker.rerank_results(
            self.test_results, 
            result_images=[test_image] * len(self.test_results)
        )
        
        assert len(reranked) == len(self.test_results)
        assert all(r.rerank_score is not None for r in reranked)


class TestAdvancedReranker:
    """Test cases for AdvancedReranker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from face_recognition.reranking.reranker import AdvancedReranker
        self.advanced_reranker = AdvancedReranker()
        
        self.test_results = [
            SearchResult("emb_001", 0.95, {"name": "Alice"}),
            SearchResult("emb_002", 0.87, {"name": "Bob"}),
        ]
    
    def test_advanced_reranker_initialization(self):
        """Test advanced reranker initialization."""
        assert hasattr(self.advanced_reranker, 'enable_context_analysis')
        assert hasattr(self.advanced_reranker, 'enable_demographic_consistency')
        assert hasattr(self.advanced_reranker, 'feature_importance')
    
    def test_rerank_with_context(self):
        """Test context-aware reranking."""
        context = {"time_of_day": "morning", "location": "office"}
        
        reranked = self.advanced_reranker.rerank_with_context(
            self.test_results, 
            context_info=context
        )
        
        assert len(reranked) == len(self.test_results)
        assert all(isinstance(r, SearchResult) for r in reranked)
    
    def test_learn_from_feedback(self):
        """Test learning from user feedback."""
        # This should not raise an error
        self.advanced_reranker.learn_from_feedback(
            "query_123", 
            "emb_001", 
            self.test_results
        )
        
        # In a real implementation, this would update internal models