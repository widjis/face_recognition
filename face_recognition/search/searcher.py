"""Similarity search functionality for face recognition."""

import time
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from datetime import datetime

from ..models import FaceEmbedding, SearchResult, SearchConfig
from ..exceptions import SimilaritySearchError, ConfigurationError
from ..vector_db import VectorDatabase


class SimilaritySearcher:
    """Handles similarity search operations for face embeddings."""
    
    def __init__(self, vector_db: VectorDatabase, default_config: Optional[SearchConfig] = None):
        """Initialize the similarity searcher.
        
        Args:
            vector_db: Vector database instance for storing and searching embeddings
            default_config: Default search configuration
        """
        self.vector_db = vector_db
        self.default_config = default_config or SearchConfig()
        self.search_cache = {}
        self.search_history = []
        self._performance_stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'average_results': 0.0,
            'average_search_time': 0.0,
            'cache_hit_rate': 0.0
        }
        self.search_stats = self._performance_stats
        self._search_times = []
        self._cache = {}
    
    def search(self, query_embedding: FaceEmbedding, config: Optional[SearchConfig] = None) -> List[SearchResult]:
        """Search for similar embeddings.
        
        Args:
            query_embedding: The query embedding to search for
            config: Search configuration (uses default if not provided)
            
        Returns:
            List of search results sorted by similarity
            
        Raises:
            SimilaritySearchError: If search fails
        """
        import time
        
        try:
            search_config = config or self.default_config
            
            # Validate configuration
            if search_config.top_k <= 0:
                raise ConfigurationError(f"top_k must be positive, got {search_config.top_k}")
            if not (0.0 <= search_config.similarity_threshold <= 1.0):
                raise ConfigurationError(f"similarity_threshold must be between 0.0 and 1.0, got {search_config.similarity_threshold}")
            
            # Update total searches first
            self.search_stats['total_searches'] += 1
            
            # Check cache
            cache_key = self._get_cache_key(query_embedding, search_config)
            if cache_key in self.search_cache:
                self.search_stats['cache_hits'] += 1
                self.search_stats['cache_hit_rate'] = self.search_stats['cache_hits'] / self.search_stats['total_searches']
                return self.search_cache[cache_key]
            
            # Track search time
            start_time = time.time()
            
            # Perform search
            raw_results = self.vector_db.search_similar(
                query_embedding,
                top_k=search_config.top_k,
                threshold=search_config.similarity_threshold
            )
            
            # Convert to SearchResult objects if needed
            results = []
            for result in raw_results:
                if hasattr(result, 'similarity_score'):
                    search_result = SearchResult(
                        embedding_id=result.embedding_id,
                        similarity_score=result.similarity_score,
                        metadata=getattr(result, 'metadata', {})
                    )
                    results.append(search_result)
                else:
                    results.append(result)
            
            search_time = time.time() - start_time
            self._search_times.append(search_time)
            
            # Update statistics
            self.search_stats['average_results'] = (
                self.search_stats['average_results'] * (self.search_stats['total_searches'] - 1) + len(results)
            ) / self.search_stats['total_searches']
            self.search_stats['average_search_time'] = sum(self._search_times) / len(self._search_times)
            self.search_stats['cache_hit_rate'] = self.search_stats['cache_hits'] / self.search_stats['total_searches']
            
            # Cache results
            self.search_cache[cache_key] = results
            
            # Add to search history
            self.search_history.append({
                'timestamp': datetime.now(),
                'query_embedding_id': getattr(query_embedding, 'id', 'unknown'),
                'config': search_config,
                'result_count': len(results),
                'search_time': search_time
            })
            
            return results
            
        except Exception as e:
            if isinstance(e, (SimilaritySearchError, ConfigurationError)):
                raise
            raise SimilaritySearchError(f"Search failed: {str(e)}")
    
    def batch_search(self, query_embeddings: List[FaceEmbedding], config: Optional[SearchConfig] = None) -> List[List[SearchResult]]:
        """Perform batch search for multiple embeddings.
        
        Args:
            query_embeddings: List of query embeddings
            config: Search configuration
            
        Returns:
            List of search results for each query embedding
        """
        if not query_embeddings:
            return []
        
        results = []
        for embedding in query_embeddings:
            results.append(self.search(embedding, config))
        
        return results
    
    def search_with_filters(self, query_embedding: FaceEmbedding, filters: Dict[str, Any], config: Optional[SearchConfig] = None) -> List[SearchResult]:
        """Search with metadata filters.
        
        Args:
            query_embedding: The query embedding
            filters: Metadata filters to apply
            config: Search configuration
            
        Returns:
            Filtered search results
        """
        # Get all results first
        results = self.search(query_embedding, config)
        
        # Apply filters
        filtered_results = []
        for result in results:
            if self._matches_filters(result.metadata, filters):
                filtered_results.append(result)
        
        return filtered_results
    
    def find_duplicates(self, query_embedding: FaceEmbedding, duplicate_threshold: float = 0.95) -> List[SearchResult]:
        """Find duplicate embeddings based on similarity threshold.
        
        Args:
            query_embedding: Embedding to find duplicates for
            duplicate_threshold: Similarity threshold for duplicates
            
        Returns:
            List of duplicate search results
        """
        results = self.search(query_embedding)
        return [result for result in results if result.similarity_score >= duplicate_threshold]
    
    def set_threshold(self, threshold: float) -> None:
        """Set the default similarity threshold.
        
        Args:
            threshold: New similarity threshold
        """
        if not 0.0 <= threshold <= 1.0:
            raise ConfigurationError("Threshold must be between 0.0 and 1.0")
        
        self.default_config.similarity_threshold = threshold
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics.
        
        Returns:
            Dictionary containing search statistics
        """
        stats = self.search_stats.copy()
        stats['cache_size'] = len(self.search_cache)
        stats['cache_hits'] = self.search_stats['cache_hits']
        return stats
    
    def clear_cache(self) -> None:
        """Clear the search cache."""
        self.search_cache.clear()
        self.search_stats['cache_size'] = 0
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get the search history.
        
        Returns:
            List of search history entries
        """
        return self.search_history.copy()
    
    def _get_cache_key(self, embedding: FaceEmbedding, config: SearchConfig) -> str:
        """Generate cache key for search results."""
        embedding_hash = hash(embedding.vector.tobytes())
        config_hash = hash((config.top_k, config.similarity_threshold, config.enable_reranking))
        return f"{embedding_hash}_{config_hash}"
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the given filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif isinstance(value, dict) and 'min' in value and 'max' in value:
                # Range filter
                if not (value['min'] <= metadata[key] <= value['max']):
                    return False
            elif metadata[key] != value:
                return False
        
        return True


class AdvancedSearchFilters:
    """Helper class for creating advanced search filters."""
    
    @staticmethod
    def by_name(name: str) -> Dict[str, str]:
        """Create a filter by name."""
        return {"name": name}
    
    @staticmethod
    def by_names(names: List[str]) -> Dict[str, List[str]]:
        """Create a filter by multiple names."""
        return {"name": names}
    
    @staticmethod
    def by_age_range(min_age: int, max_age: int) -> Dict[str, Dict[str, int]]:
        """Create a filter by age range."""
        return {"age": {"min": min_age, "max": max_age}}
    
    @staticmethod
    def combine_filters(*filters: Dict) -> Dict:
        """Combine multiple filters into one."""
        combined = {}
        for filter_dict in filters:
            combined.update(filter_dict)
        return combined


class SearchResultAnalyzer:
    """Utility class for analyzing search results."""
    
    @staticmethod
    def get_similarity_distribution(results: List[SearchResult]) -> Dict[str, float]:
        """Analyze the distribution of similarity scores."""
        if not results:
            return {}
        
        scores = [result.similarity_score for result in results]
        return {
            'count': len(scores),
            'min_similarity': min(scores),
            'max_similarity': max(scores),
            'mean_similarity': sum(scores) / len(scores),
            'std_similarity': np.std(scores) if len(scores) > 1 else 0.0
        }
    
    @staticmethod
    def find_outliers(results: List[SearchResult], threshold: float = 2.0) -> List[SearchResult]:
        """Find outliers in similarity scores using standard deviation."""
        if len(results) < 3:
            return []
        
        scores = [result.similarity_score for result in results]
        mean_score = sum(scores) / len(scores)
        std_score = np.std(scores)
        
        outliers = []
        for result in results:
            z_score = abs(result.similarity_score - mean_score) / std_score if std_score > 0 else 0
            if z_score > threshold:
                outliers.append(result)
        
        return outliers
    
    @staticmethod
    def group_by_similarity(results: List[SearchResult], bins=3) -> Dict[str, List[SearchResult]]:
        """Group results by similarity score ranges."""
        if not results:
            return {}
        
        groups = {}
        
        # Handle custom bins (list of thresholds)
        if isinstance(bins, list):
            for i in range(len(bins) - 1):
                bin_start = bins[i]
                bin_end = bins[i + 1]
                bin_key = f"{bin_start:.2f}-{bin_end:.2f}"
                groups[bin_key] = []
                
                for result in results:
                    if bin_start <= result.similarity_score < bin_end or (bin_end == 1.0 and result.similarity_score == 1.0):
                        groups[bin_key].append(result)
        else:
            # Handle integer bins (number of equal-sized bins)
            scores = [result.similarity_score for result in results]
            min_score = min(scores)
            max_score = max(scores)
            
            if min_score == max_score:
                return {f"{min_score:.2f}": results}
            
            bin_size = (max_score - min_score) / bins
            
            for result in results:
                bin_index = min(int((result.similarity_score - min_score) / bin_size), bins - 1)
                bin_start = min_score + bin_index * bin_size
                bin_end = min_score + (bin_index + 1) * bin_size
                bin_key = f"{bin_start:.2f}-{bin_end:.2f}"
                
                if bin_key not in groups:
                    groups[bin_key] = []
                groups[bin_key].append(result)
        
        return groups