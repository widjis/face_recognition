"""Tests for similarity search functionality."""

import pytest
import numpy as np
import tempfile
import shutil
from datetime import datetime
from face_recognition.search import SimilaritySearcher
from face_recognition.models import FaceEmbedding, SearchResult, SearchConfig
from face_recognition.exceptions import SimilaritySearchError, ConfigurationError


# Mock VectorDatabase for testing
class MockVectorDatabase:
    """Mock vector database for testing."""
    
    def __init__(self):
        self.embeddings = {}
        self.metadata = {}
        self.id_counter = 0
    
    def store_embedding(self, embedding, metadata):
        embedding_id = f"emb_{self.id_counter:06d}"
        self.id_counter += 1
        self.embeddings[embedding_id] = embedding
        self.metadata[embedding_id] = metadata
        return embedding_id
    
    def search_similar(self, query_embedding, top_k=10, threshold=0.0):
        results = []
        
        for emb_id, stored_embedding in self.embeddings.items():
            # Calculate cosine similarity
            similarity = np.dot(query_embedding.vector, stored_embedding.vector)
            
            if similarity >= threshold:
                result = SearchResult(
                    embedding_id=emb_id,
                    similarity_score=float(similarity),
                    metadata=self.metadata[emb_id]
                )
                results.append(result)
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    def get_embedding_info(self, embedding_id):
        if embedding_id in self.metadata:
            return {'metadata': self.metadata[embedding_id]}
        return None


class TestSimilaritySearcher:
    """Test cases for SimilaritySearcher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_db = MockVectorDatabase()
        self.searcher = SimilaritySearcher(self.mock_db)
        
        # Create test embeddings
        self.test_embeddings = []
        for i in range(5):
            vector = np.random.rand(512).astype(np.float32)
            vector = vector / np.linalg.norm(vector)  # Normalize
            
            embedding = FaceEmbedding(
                vector=vector,
                dimension=512,
                model_version="test_v1.0",
                extraction_timestamp=datetime.now()
            )
            self.test_embeddings.append(embedding)
        
        # Store embeddings in mock database
        self.stored_ids = []
        for i, embedding in enumerate(self.test_embeddings):
            metadata = {"name": f"Person_{i}", "id": f"00{i}"}
            emb_id = self.mock_db.store_embedding(embedding, metadata)
            self.stored_ids.append(emb_id)
    
    def test_searcher_initialization(self):
        """Test searcher initialization."""
        assert self.searcher.vector_db == self.mock_db
        assert isinstance(self.searcher.default_config, SearchConfig)
        assert self.searcher._performance_stats['total_searches'] == 0
    
    def test_basic_search(self):
        """Test basic similarity search."""
        query_embedding = self.test_embeddings[0]
        
        results = self.searcher.search(query_embedding)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        
        # First result should be the same embedding (highest similarity)
        assert results[0].similarity_score > 0.99
    
    def test_search_with_custom_config(self):
        """Test search with custom configuration."""
        query_embedding = self.test_embeddings[0]
        config = SearchConfig(top_k=3, similarity_threshold=0.8)
        
        results = self.searcher.search(query_embedding, config)
        
        assert len(results) <= 3
        assert all(r.similarity_score >= 0.8 for r in results)
    
    def test_search_with_invalid_config(self):
        """Test search with invalid configuration."""
        query_embedding = self.test_embeddings[0]
        
        # Invalid top_k
        invalid_config = SearchConfig(top_k=0)
        with pytest.raises(ConfigurationError, match="top_k must be positive"):
            self.searcher.search(query_embedding, invalid_config)
        
        # Invalid threshold
        invalid_config = SearchConfig(similarity_threshold=1.5)
        with pytest.raises(ConfigurationError, match="Similarity threshold must be between"):
            self.searcher.search(query_embedding, invalid_config)
    
    def test_batch_search(self):
        """Test batch search functionality."""
        query_embeddings = self.test_embeddings[:3]
        
        batch_results = self.searcher.batch_search(query_embeddings)
        
        assert len(batch_results) == 3
        assert all(isinstance(results, list) for results in batch_results)
        assert all(len(results) > 0 for results in batch_results)
    
    def test_batch_search_empty_list(self):
        """Test batch search with empty list."""
        batch_results = self.searcher.batch_search([])
        assert batch_results == []
    
    def test_search_with_filters(self):
        """Test search with metadata filters."""
        query_embedding = self.test_embeddings[0]
        
        # Filter by specific name
        filters = {"name": "Person_1"}
        results = self.searcher.search_with_filters(query_embedding, filters)
        
        # Should only return results matching the filter
        matching_results = [r for r in results if r.metadata["name"] == "Person_1"]
        assert len(matching_results) == len(results)
    
    def test_find_duplicates(self):
        """Test duplicate detection."""
        query_embedding = self.test_embeddings[0]
        
        duplicates = self.searcher.find_duplicates(query_embedding, duplicate_threshold=0.95)
        
        assert isinstance(duplicates, list)
        # Should find at least the same embedding as a duplicate
        assert len(duplicates) >= 1
        assert duplicates[0].similarity_score >= 0.95
    
    def test_set_threshold(self):
        """Test setting similarity threshold."""
        original_threshold = self.searcher.default_config.similarity_threshold
        
        self.searcher.set_threshold(0.9)
        assert self.searcher.default_config.similarity_threshold == 0.9
        
        # Test invalid threshold
        with pytest.raises(ConfigurationError, match="Threshold must be between"):
            self.searcher.set_threshold(1.5)
    
    def test_search_statistics(self):
        """Test search statistics tracking."""
        query_embedding = self.test_embeddings[0]
        
        # Perform some searches
        for _ in range(3):
            self.searcher.search(query_embedding)
        
        stats = self.searcher.get_search_statistics()
        
        assert stats['total_searches'] == 3
        assert stats['average_search_time'] > 0
        assert 'cache_hit_rate' in stats
        assert 'cache_size' in stats
    
    def test_search_caching(self):
        """Test search result caching."""
        query_embedding = self.test_embeddings[0]
        
        # First search
        results1 = self.searcher.search(query_embedding)
        
        # Second search with same parameters (should hit cache)
        results2 = self.searcher.search(query_embedding)
        
        # Results should be identical
        assert len(results1) == len(results2)
        
        stats = self.searcher.get_search_statistics()
        assert stats['cache_hits'] > 0
    
    def test_clear_cache(self):
        """Test cache clearing."""
        query_embedding = self.test_embeddings[0]
        
        # Perform search to populate cache
        self.searcher.search(query_embedding)
        
        # Clear cache
        self.searcher.clear_cache()
        
        stats = self.searcher.get_search_statistics()
        assert stats['cache_size'] == 0
    
    def test_search_history(self):
        """Test search history tracking."""
        query_embedding = self.test_embeddings[0]
        
        # Perform searches
        self.searcher.search(query_embedding)
        self.searcher.search(query_embedding, SearchConfig(top_k=5))
        
        history = self.searcher.get_search_history()
        
        assert len(history) == 2
        assert all('timestamp' in entry for entry in history)
        assert all('config' in entry for entry in history)
        assert all('result_count' in entry for entry in history)


class TestAdvancedSearchFilters:
    """Test cases for AdvancedSearchFilters helper class."""
    
    def test_by_name_filter(self):
        """Test name filter creation."""
        from face_recognition.search.searcher import AdvancedSearchFilters
        
        filter_dict = AdvancedSearchFilters.by_name("John Doe")
        assert filter_dict == {"name": "John Doe"}
    
    def test_by_names_filter(self):
        """Test multiple names filter creation."""
        from face_recognition.search.searcher import AdvancedSearchFilters
        
        names = ["John Doe", "Jane Smith"]
        filter_dict = AdvancedSearchFilters.by_names(names)
        assert filter_dict == {"name": names}
    
    def test_by_age_range_filter(self):
        """Test age range filter creation."""
        from face_recognition.search.searcher import AdvancedSearchFilters
        
        filter_dict = AdvancedSearchFilters.by_age_range(18, 65)
        assert filter_dict == {"age": {"min": 18, "max": 65}}
    
    def test_combine_filters(self):
        """Test filter combination."""
        from face_recognition.search.searcher import AdvancedSearchFilters
        
        name_filter = AdvancedSearchFilters.by_name("John Doe")
        age_filter = AdvancedSearchFilters.by_age_range(25, 35)
        
        combined = AdvancedSearchFilters.combine_filters(name_filter, age_filter)
        
        assert "name" in combined
        assert "age" in combined
        assert combined["name"] == "John Doe"
        assert combined["age"] == {"min": 25, "max": 35}


class TestSearchResultAnalyzer:
    """Test cases for SearchResultAnalyzer utility class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test search results
        self.test_results = [
            SearchResult("emb_001", 0.95, {"name": "Alice"}),
            SearchResult("emb_002", 0.87, {"name": "Bob"}),
            SearchResult("emb_003", 0.82, {"name": "Charlie"}),
            SearchResult("emb_004", 0.76, {"name": "David"}),
            SearchResult("emb_005", 0.65, {"name": "Eve"}),
        ]
    
    def test_similarity_distribution(self):
        """Test similarity distribution analysis."""
        from face_recognition.search.searcher import SearchResultAnalyzer
        
        distribution = SearchResultAnalyzer.get_similarity_distribution(self.test_results)
        
        assert distribution['count'] == 5
        assert distribution['min_similarity'] == 0.65
        assert distribution['max_similarity'] == 0.95
        assert 0.65 <= distribution['mean_similarity'] <= 0.95
    
    def test_similarity_distribution_empty(self):
        """Test similarity distribution with empty results."""
        from face_recognition.search.searcher import SearchResultAnalyzer
        
        distribution = SearchResultAnalyzer.get_similarity_distribution([])
        assert distribution == {}
    
    def test_find_outliers(self):
        """Test outlier detection."""
        from face_recognition.search.searcher import SearchResultAnalyzer
        
        # Add an outlier
        outlier_results = self.test_results + [
            SearchResult("emb_006", 0.1, {"name": "Outlier"})  # Very low similarity
        ]
        
        outliers = SearchResultAnalyzer.find_outliers(outlier_results, threshold=1.5)
        
        assert len(outliers) > 0
        assert any(r.metadata["name"] == "Outlier" for r in outliers)
    
    def test_group_by_similarity(self):
        """Test grouping by similarity ranges."""
        from face_recognition.search.searcher import SearchResultAnalyzer
        
        groups = SearchResultAnalyzer.group_by_similarity(self.test_results)
        
        assert isinstance(groups, dict)
        assert len(groups) > 0
        
        # Check that all results are assigned to groups
        total_grouped = sum(len(group) for group in groups.values())
        assert total_grouped == len(self.test_results)
    
    def test_group_by_similarity_custom_bins(self):
        """Test grouping with custom similarity bins."""
        from face_recognition.search.searcher import SearchResultAnalyzer
        
        custom_bins = [0.0, 0.8, 1.0]
        groups = SearchResultAnalyzer.group_by_similarity(self.test_results, custom_bins)
        
        assert len(groups) == 2  # Two ranges: 0.0-0.8 and 0.8-1.0
        assert "0.00-0.80" in groups
        assert "0.80-1.00" in groups