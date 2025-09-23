"""
Advanced Similarity Search with Confidence Levels
Using WIN_20250222_15_21_37_Pro.jpg as query image

This script demonstrates the complete face recognition pipeline with detailed confidence reporting:
1. Face detection with confidence scores
2. Embedding extraction with multiple models
3. Similarity search with confidence levels
4. Reranking with quality assessment
5. Comprehensive confidence reporting
"""

import cv2
import numpy as np
import os
import tempfile
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Import face recognition modules
from face_recognition.face_detection import FaceDetector
from face_recognition.embedding import EmbeddingExtractor
from face_recognition.models import SearchResult, RerankingFeatures, FaceRegion
from face_recognition.reranking.reranker import Reranker

# Import FAISS for vector database
import faiss


class ConfidenceAwareSimilaritySearch:
    """Advanced similarity search system with comprehensive confidence reporting."""
    
    def __init__(self, db_path: str = "confidence_search_db"):
        """Initialize the confidence-aware similarity search system."""
        self.db_path = db_path
        self.dimension = 512
        
        # Initialize components
        self.detector = FaceDetector(method="haar", min_face_size=(30, 30))
        self.extractor = EmbeddingExtractor(model_name="simple", embedding_dim=512)
        self.reranker = Reranker(
            enable_quality_scoring=True,
            enable_pose_analysis=True,
            enable_illumination_analysis=True
        )
        
        # Initialize vector database
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = {}
        self.stored_images = {}
        self.confidence_store = {}  # Store confidence data
        self.id_counter = 0
        
        # Create database directory
        os.makedirs(db_path, exist_ok=True)
        
        print(f"🎯 Confidence-Aware Similarity Search System Initialized")
        print(f"   Database: {self.db_path}")
        print(f"   Embedding dimension: {self.dimension}")
    
    def detect_faces_with_confidence(self, image: np.ndarray, image_name: str = "image") -> List[Dict]:
        """
        Detect faces with detailed confidence reporting.
        
        Returns:
            List of face detection results with confidence data
        """
        print(f"\n🔍 Face Detection Analysis for {image_name}")
        print("-" * 50)
        
        faces = self.detector.detect_faces(image)
        face_results = []
        
        if not faces:
            print("❌ No faces detected")
            return face_results
        
        print(f"✅ Detected {len(faces)} face(s)")
        
        for i, face in enumerate(faces):
            # Calculate additional confidence metrics
            face_area = face.width * face.height
            image_area = image.shape[0] * image.shape[1]
            face_coverage = face_area / image_area
            
            # Assess face quality
            face_roi = image[face.y:face.y+face.height, face.x:face.x+face.width]
            quality_score = self._assess_face_quality(face_roi)
            
            face_data = {
                'face_region': face,
                'detection_confidence': face.confidence,
                'face_coverage': face_coverage,
                'quality_score': quality_score,
                'face_area': face_area,
                'aspect_ratio': face.width / face.height,
                'position': (face.x + face.width//2, face.y + face.height//2)
            }
            
            face_results.append(face_data)
            
            print(f"   Face {i+1}:")
            print(f"     Detection confidence: {face.confidence:.3f}")
            print(f"     Quality score: {quality_score:.3f}")
            print(f"     Face coverage: {face_coverage:.1%}")
            print(f"     Size: {face.width}x{face.height} ({face_area:,} pixels)")
            print(f"     Aspect ratio: {face.width/face.height:.2f}")
            print(f"     Position: ({face.x + face.width//2}, {face.y + face.height//2})")
        
        return face_results
    
    def _assess_face_quality(self, face_roi: np.ndarray) -> float:
        """Assess face quality using multiple metrics."""
        if face_roi.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
        
        # Calculate sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 500.0)  # Normalize
        
        # Calculate contrast (standard deviation)
        contrast = gray.std() / 128.0  # Normalize to 0-1
        contrast = min(1.0, contrast)
        
        # Calculate brightness (mean intensity)
        brightness = gray.mean() / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Prefer mid-range brightness
        
        # Combined quality score
        quality = (sharpness * 0.4 + contrast * 0.3 + brightness_score * 0.3)
        return min(1.0, quality)
    
    def extract_embeddings_with_confidence(self, face_data: Dict, image: np.ndarray) -> Dict:
        """Extract embeddings with confidence metrics."""
        print(f"\n🧠 Embedding Extraction with Confidence Analysis")
        print("-" * 50)
        
        face_region = face_data['face_region']
        processed_face = self.detector.preprocess_face(image, face_region)
        
        # Test multiple models for confidence comparison
        models = ["simple", "facenet", "arcface"]
        embedding_results = {}
        
        for model_name in models:
            try:
                model_extractor = EmbeddingExtractor(model_name=model_name, embedding_dim=512)
                embedding = model_extractor.extract_embedding(processed_face)
                
                # Calculate embedding confidence metrics
                vector_norm = np.linalg.norm(embedding.vector)
                vector_std = np.std(embedding.vector)
                vector_mean = np.mean(embedding.vector)
                
                # Assess embedding quality
                embedding_confidence = self._assess_embedding_confidence(embedding.vector)
                
                embedding_results[model_name] = {
                    'embedding': embedding,
                    'vector_norm': vector_norm,
                    'vector_std': vector_std,
                    'vector_mean': vector_mean,
                    'embedding_confidence': embedding_confidence,
                    'model_version': embedding.model_version
                }
                
                print(f"   {model_name.upper()} Model:")
                print(f"     Embedding confidence: {embedding_confidence:.3f}")
                print(f"     Vector norm: {vector_norm:.6f}")
                print(f"     Vector std: {vector_std:.6f}")
                print(f"     Vector mean: {vector_mean:.6f}")
                print(f"     Model version: {embedding.model_version}")
                
            except Exception as e:
                print(f"   ❌ {model_name}: {e}")
                embedding_results[model_name] = None
        
        return embedding_results
    
    def _assess_embedding_confidence(self, vector: np.ndarray) -> float:
        """Assess embedding confidence based on vector properties."""
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm == 0:
            return 0.0
        
        normalized_vector = vector / norm
        
        # Calculate confidence metrics
        # 1. Vector magnitude (should be reasonable)
        magnitude_score = min(1.0, norm / 10.0)  # Adjust threshold as needed
        
        # 2. Vector distribution (should not be too concentrated)
        std_score = min(1.0, np.std(normalized_vector) * 5.0)
        
        # 3. Non-zero elements ratio
        non_zero_ratio = np.count_nonzero(vector) / len(vector)
        
        # Combined confidence
        confidence = (magnitude_score * 0.4 + std_score * 0.4 + non_zero_ratio * 0.2)
        return min(1.0, confidence)
    
    def build_reference_database(self, reference_image: np.ndarray, image_name: str = "reference") -> bool:
        """Build a reference database with the provided image."""
        print(f"\n📚 Building Reference Database with {image_name}")
        print("=" * 60)
        
        # Detect faces
        face_results = self.detect_faces_with_confidence(reference_image, image_name)
        
        if not face_results:
            print("❌ No faces found to build database")
            return False
        
        # Use the best quality face
        best_face = max(face_results, key=lambda x: x['quality_score'] * x['detection_confidence'])
        
        # Extract embeddings
        embedding_results = self.extract_embeddings_with_confidence(best_face, reference_image)
        
        # Store embeddings in database
        stored_count = 0
        for model_name, result in embedding_results.items():
            if result is not None:
                embedding = result['embedding']
                embedding_id = f"ref_{model_name}_{self.id_counter:06d}"
                
                # Store in FAISS index
                self.index.add(embedding.vector.reshape(1, -1))
                
                # Store metadata
                self.metadata_store[embedding_id] = {
                    'source_image': image_name,
                    'model_name': model_name,
                    'face_region': best_face['face_region'],
                    'detection_confidence': best_face['detection_confidence'],
                    'quality_score': best_face['quality_score'],
                    'embedding_confidence': result['embedding_confidence'],
                    'vector_norm': result['vector_norm'],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store processed face image for reranking
                processed_face = self.detector.preprocess_face(reference_image, best_face['face_region'])
                self.stored_images[embedding_id] = processed_face
                
                self.id_counter += 1
                stored_count += 1
                
                print(f"   ✅ Stored {model_name} embedding (ID: {embedding_id})")
        
        print(f"\n📊 Database Summary:")
        print(f"   Total embeddings stored: {stored_count}")
        print(f"   Database size: {len(self.metadata_store)} entries")
        
        return stored_count > 0
    
    def search_with_confidence(self, query_image: np.ndarray, query_name: str = "query", top_k: int = 5) -> Dict:
        """Perform similarity search with comprehensive confidence reporting."""
        print(f"\n🔍 Similarity Search with Confidence Analysis")
        print("=" * 60)
        
        # Detect faces in query image
        face_results = self.detect_faces_with_confidence(query_image, query_name)
        
        if not face_results:
            return {'success': False, 'error': 'No faces detected in query image'}
        
        # Use the best quality face
        best_face = max(face_results, key=lambda x: x['quality_score'] * x['detection_confidence'])
        
        print(f"\n🎯 Selected Query Face:")
        print(f"   Detection confidence: {best_face['detection_confidence']:.3f}")
        print(f"   Quality score: {best_face['quality_score']:.3f}")
        
        # Extract query embeddings
        query_embeddings = self.extract_embeddings_with_confidence(best_face, query_image)
        
        # Perform search with each model
        search_results = {}
        
        for model_name, embedding_result in query_embeddings.items():
            if embedding_result is None:
                continue
            
            print(f"\n🔍 Searching with {model_name.upper()} model:")
            
            query_embedding = embedding_result['embedding']
            
            # Search in FAISS index
            similarities, indices = self.index.search(query_embedding.vector.reshape(1, -1), top_k)
            
            # Process results
            model_results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # No more results
                    break
                
                # Find metadata for this result
                embedding_id = None
                for eid, metadata in self.metadata_store.items():
                    if metadata.get('index_position') == idx:
                        embedding_id = eid
                        break
                
                if embedding_id is None:
                    # Fallback: use index to find embedding_id
                    embedding_ids = list(self.metadata_store.keys())
                    if idx < len(embedding_ids):
                        embedding_id = embedding_ids[idx]
                
                if embedding_id and embedding_id in self.metadata_store:
                    metadata = self.metadata_store[embedding_id]
                    
                    # Calculate confidence score
                    overall_confidence = self._calculate_overall_confidence(
                        similarity,
                        best_face['detection_confidence'],
                        best_face['quality_score'],
                        embedding_result['embedding_confidence'],
                        metadata['detection_confidence'],
                        metadata['quality_score'],
                        metadata['embedding_confidence']
                    )
                    
                    result = {
                        'rank': i + 1,
                        'embedding_id': embedding_id,
                        'similarity_score': float(similarity),
                        'overall_confidence': overall_confidence,
                        'query_face_confidence': best_face['detection_confidence'],
                        'query_quality': best_face['quality_score'],
                        'query_embedding_confidence': embedding_result['embedding_confidence'],
                        'result_face_confidence': metadata['detection_confidence'],
                        'result_quality': metadata['quality_score'],
                        'result_embedding_confidence': metadata['embedding_confidence'],
                        'model_name': metadata['model_name'],
                        'source_image': metadata['source_image']
                    }
                    
                    model_results.append(result)
                    
                    print(f"   Rank {i+1}: {metadata['source_image']} ({metadata['model_name']})")
                    print(f"     Similarity: {similarity:.4f}")
                    print(f"     Overall confidence: {overall_confidence:.3f}")
                    print(f"     Result quality: {metadata['quality_score']:.3f}")
            
            search_results[model_name] = model_results
        
        # Apply reranking if we have stored images
        if search_results and len(self.stored_images) > 0:
            print(f"\n🔄 Applying Advanced Reranking...")
            search_results = self._apply_reranking_with_confidence(
                search_results, best_face, query_image
            )
        
        return {
            'success': True,
            'query_face': best_face,
            'search_results': search_results,
            'total_database_size': len(self.metadata_store)
        }
    
    def _calculate_overall_confidence(self, similarity: float, query_face_conf: float, 
                                    query_quality: float, query_emb_conf: float,
                                    result_face_conf: float, result_quality: float, 
                                    result_emb_conf: float) -> float:
        """Calculate overall confidence score combining multiple factors."""
        # Normalize similarity (assuming cosine similarity range -1 to 1)
        normalized_similarity = (similarity + 1.0) / 2.0
        
        # Weight different confidence factors
        weights = {
            'similarity': 0.4,
            'query_face': 0.15,
            'query_quality': 0.1,
            'query_embedding': 0.1,
            'result_face': 0.15,
            'result_quality': 0.05,
            'result_embedding': 0.05
        }
        
        overall_confidence = (
            normalized_similarity * weights['similarity'] +
            query_face_conf * weights['query_face'] +
            query_quality * weights['query_quality'] +
            query_emb_conf * weights['query_embedding'] +
            result_face_conf * weights['result_face'] +
            result_quality * weights['result_quality'] +
            result_emb_conf * weights['result_embedding']
        )
        
        return min(1.0, overall_confidence)
    
    def _apply_reranking_with_confidence(self, search_results: Dict, query_face: Dict, 
                                       query_image: np.ndarray) -> Dict:
        """Apply reranking with confidence adjustments."""
        # Extract reranking features for query
        query_processed = self.detector.preprocess_face(query_image, query_face['face_region'])
        query_features = self.reranker.extract_reranking_features(query_processed)
        
        print(f"   Query reranking features:")
        print(f"     Quality: {query_features.face_quality_score:.3f}")
        print(f"     Pose angle: {query_features.pose_angle:.1f}°")
        print(f"     Illumination: {query_features.illumination_score:.3f}")
        print(f"     Landmark confidence: {query_features.landmark_confidence:.3f}")
        
        # Apply reranking to each model's results
        reranked_results = {}
        
        for model_name, results in search_results.items():
            if not results:
                reranked_results[model_name] = results
                continue
            
            # Create SearchResult objects for reranking
            search_result_objects = []
            result_images = []
            
            for result in results:
                embedding_id = result['embedding_id']
                search_result = SearchResult(
                    embedding_id=embedding_id,
                    similarity_score=result['similarity_score'],
                    metadata=self.metadata_store.get(embedding_id, {}),
                    rerank_score=result['similarity_score']  # Initial rerank score
                )
                search_result_objects.append(search_result)
                
                # Get stored image for reranking
                if embedding_id in self.stored_images:
                    result_images.append(self.stored_images[embedding_id])
                else:
                    result_images.append(None)
            
            # Apply reranking
            reranked = self.reranker.rerank_results(
                search_result_objects, 
                query_features, 
                result_images
            )
            
            # Update results with reranking scores and confidence adjustments
            reranked_list = []
            for i, reranked_result in enumerate(reranked):
                original_result = next(r for r in results if r['embedding_id'] == reranked_result.embedding_id)
                
                # Adjust confidence based on reranking improvement
                rerank_improvement = reranked_result.rerank_score - reranked_result.similarity_score
                confidence_boost = min(0.1, max(-0.1, rerank_improvement * 0.5))
                adjusted_confidence = min(1.0, original_result['overall_confidence'] + confidence_boost)
                
                updated_result = original_result.copy()
                updated_result.update({
                    'rank': i + 1,
                    'rerank_score': reranked_result.rerank_score,
                    'rerank_improvement': rerank_improvement,
                    'adjusted_confidence': adjusted_confidence
                })
                
                reranked_list.append(updated_result)
            
            reranked_results[model_name] = reranked_list
            
            print(f"   {model_name.upper()} reranked results:")
            for result in reranked_list[:3]:  # Show top 3
                print(f"     Rank {result['rank']}: Similarity {result['similarity_score']:.4f} → "
                      f"Rerank {result['rerank_score']:.4f} (Δ{result['rerank_improvement']:+.4f})")
        
        return reranked_results
    
    def print_detailed_results(self, results: Dict):
        """Print detailed search results with confidence analysis."""
        if not results['success']:
            print(f"❌ Search failed: {results.get('error', 'Unknown error')}")
            return
        
        print(f"\n📊 DETAILED SIMILARITY SEARCH RESULTS")
        print("=" * 70)
        
        query_face = results['query_face']
        print(f"🎯 Query Face Analysis:")
        print(f"   Detection confidence: {query_face['detection_confidence']:.3f}")
        print(f"   Quality score: {query_face['quality_score']:.3f}")
        print(f"   Face coverage: {query_face['face_coverage']:.1%}")
        print(f"   Face size: {query_face['face_area']:,} pixels")
        
        search_results = results['search_results']
        
        for model_name, model_results in search_results.items():
            if not model_results:
                continue
            
            print(f"\n🔍 {model_name.upper()} MODEL RESULTS:")
            print("-" * 50)
            
            for result in model_results:
                print(f"   Rank {result['rank']}: {result['source_image']}")
                print(f"     Similarity Score: {result['similarity_score']:.4f}")
                
                if 'rerank_score' in result:
                    print(f"     Reranked Score: {result['rerank_score']:.4f} "
                          f"(Δ{result['rerank_improvement']:+.4f})")
                    print(f"     Final Confidence: {result['adjusted_confidence']:.3f}")
                else:
                    print(f"     Overall Confidence: {result['overall_confidence']:.3f}")
                
                print(f"     Component Confidences:")
                print(f"       Query face: {result['query_face_confidence']:.3f}")
                print(f"       Query quality: {result['query_quality']:.3f}")
                print(f"       Query embedding: {result['query_embedding_confidence']:.3f}")
                print(f"       Result face: {result['result_face_confidence']:.3f}")
                print(f"       Result quality: {result['result_quality']:.3f}")
                print(f"       Result embedding: {result['result_embedding_confidence']:.3f}")
                print()
        
        print(f"📈 Search Statistics:")
        print(f"   Database size: {results['total_database_size']} embeddings")
        print(f"   Models tested: {len(search_results)}")
        print(f"   Total results: {sum(len(results) for results in search_results.values())}")


def main():
    """Main function to demonstrate similarity search with confidence levels."""
    print("🎯 ADVANCED SIMILARITY SEARCH WITH CONFIDENCE LEVELS")
    print("=" * 70)
    print("Using WIN_20250222_15_21_37_Pro.jpg as query image")
    print()
    
    # Initialize the system
    search_system = ConfidenceAwareSimilaritySearch("confidence_demo_db")
    
    # Load the query image
    query_image_path = "WIN_20250222_15_21_37_Pro.jpg"
    query_image = cv2.imread(query_image_path)
    
    if query_image is None:
        print(f"❌ Could not load {query_image_path}")
        return False
    
    print(f"📸 Query image loaded: {query_image.shape}")
    
    # Build reference database using the same image (for demo purposes)
    # In a real scenario, you would use different reference images
    success = search_system.build_reference_database(query_image, "WIN_20250222_15_21_37_Pro.jpg")
    
    if not success:
        print("❌ Failed to build reference database")
        return False
    
    # Load additional reference image if available
    reference_image_path = "MTI230279.jpg"
    if os.path.exists(reference_image_path):
        reference_image = cv2.imread(reference_image_path)
        if reference_image is not None:
            print(f"\n📸 Adding additional reference: {reference_image_path}")
            search_system.build_reference_database(reference_image, "MTI230279.jpg")
    
    # Perform similarity search
    results = search_system.search_with_confidence(query_image, "WIN_20250222_15_21_37_Pro.jpg", top_k=5)
    
    # Print detailed results
    search_system.print_detailed_results(results)
    
    print(f"\n✅ SIMILARITY SEARCH COMPLETED!")
    print(f"   🎯 Comprehensive confidence analysis provided")
    print(f"   📊 Multiple model comparison available")
    print(f"   🔄 Advanced reranking applied")
    print(f"   📈 Detailed confidence breakdown included")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)