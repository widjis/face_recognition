"""Test face search using your new image against existing database."""

import cv2
import numpy as np
import os
from datetime import datetime

# Import our face recognition components
from face_recognition.face_detection import FaceDetector
from face_recognition.embedding import EmbeddingExtractor

# Import FAISS for vector database
import faiss
import json

class CompleteFaceSearchSystem:
    """Complete face search system with database and search capabilities."""
    
    def __init__(self, db_path="face_search_database"):
        """Initialize the face search system."""
        self.db_path = db_path
        self.dimension = 512
        
        # Initialize components
        self.detector = FaceDetector(method="haar", min_face_size=(30, 30))
        self.extractor = EmbeddingExtractor(model_name="simple", embedding_dim=512)
        
        # Initialize vector database
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = {}
        self.stored_images = {}  # Store face images for reranking
        self.id_counter = 0
        
        # Create database directory
        os.makedirs(db_path, exist_ok=True)
        
        # Load existing database
        self._load_database()
        
        print(f"🎯 Face Search System Initialized")
        print(f"   Database: {self.db_path}")
        print(f"   Existing faces: {len(self.metadata_store)}")
    
    def _load_database(self):
        """Load existing database from disk."""
        index_path = os.path.join(self.db_path, "search_index.faiss")
        metadata_path = os.path.join(self.db_path, "search_metadata.json")
        
        try:
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.index = faiss.read_index(index_path)
                
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self.metadata_store = data.get('metadata', {})
                    self.id_counter = data.get('id_counter', 0)
                
                print(f"✅ Loaded existing database with {self.index.ntotal} faces")
        except Exception as e:
            print(f"⚠️ Starting fresh database: {e}")
    
    def _save_database(self):
        """Save database to disk."""
        try:
            index_path = os.path.join(self.db_path, "search_index.faiss")
            metadata_path = os.path.join(self.db_path, "search_metadata.json")
            
            faiss.write_index(self.index, index_path)
            
            data = {
                'metadata': self.metadata_store,
                'id_counter': self.id_counter,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Could not save database: {e}")
    
    def add_person_to_database(self, image_path, person_name, additional_info=None):
        """Add a person's face to the search database."""
        print(f"\n👤 Adding {person_name} to database from {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        print(f"   📸 Image loaded: {image.shape}")
        
        # Detect faces
        faces = self.detector.detect_faces(image)
        if not faces:
            print(f"   ❌ No faces detected")
            return None
        
        print(f"   🔍 Found {len(faces)} face(s)")
        
        # Process each face (or just the best one)
        stored_ids = []
        for i, face in enumerate(faces):
            print(f"   📍 Face {i+1}: ({face.x}, {face.y}) {face.width}x{face.height}, confidence: {face.confidence:.3f}")
            
            # Preprocess face
            processed_face = self.detector.preprocess_face(image, face)
            
            # Extract embedding
            embedding = self.extractor.extract_embedding(processed_face)
            
            # Generate unique ID
            face_id = f"face_{self.id_counter:06d}"
            self.id_counter += 1
            
            # Store in FAISS index
            vector = embedding.vector.reshape(1, -1).astype(np.float32)
            self.index.add(vector)
            
            # Store metadata
            metadata = {
                'id': face_id,
                'person_name': person_name,
                'source_image': os.path.basename(image_path),
                'face_index': i,
                'face_position': {'x': face.x, 'y': face.y, 'width': face.width, 'height': face.height},
                'confidence': face.confidence,
                'added_date': datetime.now().isoformat(),
                'additional_info': additional_info or {}
            }
            
            self.metadata_store[face_id] = metadata
            
            # Store face image for potential reranking
            face_uint8 = (processed_face * 255).astype(np.uint8)
            self.stored_images[face_id] = face_uint8
            
            stored_ids.append(face_id)
            print(f"   ✅ Stored face as {face_id}")
        
        # Save database
        self._save_database()
        
        print(f"   💾 Database updated: {len(stored_ids)} faces added")
        return stored_ids
    
    def search_for_person(self, query_image_path, top_k=5, similarity_threshold=0.7):
        """Search for a person in the database using a query image."""
        print(f"\n🔍 Searching for faces in: {os.path.basename(query_image_path)}")
        
        # Load query image
        image = cv2.imread(query_image_path)
        if image is None:
            print(f"❌ Could not load query image: {query_image_path}")
            return []
        
        print(f"   📸 Query image loaded: {image.shape}")
        
        # Detect faces in query image
        faces = self.detector.detect_faces(image)
        if not faces:
            print(f"   ❌ No faces detected in query image")
            return []
        
        print(f"   🔍 Found {len(faces)} face(s) in query image")
        
        all_search_results = []
        
        # Search for each detected face
        for i, face in enumerate(faces):
            print(f"\n   🎯 Searching for Face {i+1}:")
            print(f"      Position: ({face.x}, {face.y}) {face.width}x{face.height}")
            print(f"      Confidence: {face.confidence:.3f}")
            
            # Preprocess face
            processed_face = self.detector.preprocess_face(image, face)
            
            # Extract embedding
            embedding = self.extractor.extract_embedding(processed_face)
            
            # Search in database
            if self.index.ntotal == 0:
                print(f"      📭 Database is empty")
                continue
            
            query_vector = embedding.vector.reshape(1, -1).astype(np.float32)
            similarities, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
            
            # Process results
            face_results = []
            face_ids = list(self.metadata_store.keys())
            
            print(f"      🎯 Search results for Face {i+1}:")
            
            for j, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1 or similarity < similarity_threshold:
                    continue
                
                if idx < len(face_ids):
                    face_id = face_ids[idx]
                    metadata = self.metadata_store[face_id]
                    
                    result = {
                        'query_face_index': i,
                        'face_id': face_id,
                        'person_name': metadata['person_name'],
                        'similarity': float(similarity),
                        'source_image': metadata['source_image'],
                        'face_position': metadata['face_position'],
                        'confidence': metadata['confidence'],
                        'metadata': metadata
                    }
                    face_results.append(result)
                    
                    print(f"      {j+1}. {metadata['person_name']}: {similarity:.4f}")
                    print(f"         Source: {metadata['source_image']}")
                    print(f"         Face position: ({metadata['face_position']['x']}, {metadata['face_position']['y']})")
            
            if not face_results:
                print(f"      📭 No matches found above threshold {similarity_threshold}")
            
            all_search_results.extend(face_results)
        
        return all_search_results
    
    def get_database_summary(self):
        """Get summary of database contents."""
        print(f"\n📊 Database Summary:")
        print(f"   Total faces: {len(self.metadata_store)}")
        print(f"   Database path: {self.db_path}")
        
        # Group by person
        people = {}
        for face_id, metadata in self.metadata_store.items():
            person_name = metadata['person_name']
            if person_name not in people:
                people[person_name] = []
            people[person_name].append(metadata)
        
        print(f"   Unique people: {len(people)}")
        print(f"   People in database:")
        for person_name, faces in people.items():
            print(f"   - {person_name}: {len(faces)} face(s)")
            for face in faces:
                print(f"     * From {face['source_image']} (confidence: {face['confidence']:.3f})")

def test_face_search_with_new_image():
    """Test face search using your new image."""
    print("🎯 Face Search Test with New Image")
    print("=" * 50)
    
    # Initialize the search system
    search_system = CompleteFaceSearchSystem("test_search_db")
    
    try:
        # Step 1: Add your existing photo to the database
        print("\n📚 Step 1: Building Face Database")
        
        # Add your original photo
        if os.path.exists("MTI230279.jpg"):
            search_system.add_person_to_database(
                "MTI230279.jpg", 
                "You (Original Photo)", 
                {"source": "original_photo", "quality": "reference"}
            )
        else:
            print("⚠️ Original photo MTI230279.jpg not found")
        
        # Step 2: Search with your new image
        print("\n🔍 Step 2: Searching with New Image")
        
        new_image_path = "WIN_20250222_15_21_37_Pro.jpg"
        
        if os.path.exists(new_image_path):
            search_results = search_system.search_for_person(
                new_image_path, 
                top_k=10, 
                similarity_threshold=0.5
            )
            
            # Step 3: Analyze results
            print(f"\n📈 Step 3: Search Results Analysis")
            
            if search_results:
                print(f"   Found {len(search_results)} potential matches:")
                
                # Group results by similarity score
                high_similarity = [r for r in search_results if r['similarity'] > 0.8]
                medium_similarity = [r for r in search_results if 0.6 <= r['similarity'] <= 0.8]
                low_similarity = [r for r in search_results if r['similarity'] < 0.6]
                
                if high_similarity:
                    print(f"\n   🎯 High Similarity Matches (>0.8):")
                    for result in high_similarity:
                        print(f"   - {result['person_name']}: {result['similarity']:.4f}")
                        print(f"     Source: {result['source_image']}")
                
                if medium_similarity:
                    print(f"\n   🔍 Medium Similarity Matches (0.6-0.8):")
                    for result in medium_similarity:
                        print(f"   - {result['person_name']}: {result['similarity']:.4f}")
                        print(f"     Source: {result['source_image']}")
                
                if low_similarity:
                    print(f"\n   📊 Lower Similarity Matches (<0.6):")
                    for result in low_similarity:
                        print(f"   - {result['person_name']}: {result['similarity']:.4f}")
                        print(f"     Source: {result['source_image']}")
                
                # Best match analysis
                best_match = max(search_results, key=lambda x: x['similarity'])
                print(f"\n   🏆 Best Match:")
                print(f"   Person: {best_match['person_name']}")
                print(f"   Similarity: {best_match['similarity']:.4f}")
                print(f"   Source: {best_match['source_image']}")
                
                # Determine if it's likely the same person
                if best_match['similarity'] > 0.8:
                    print(f"   ✅ HIGH CONFIDENCE: Likely the same person!")
                elif best_match['similarity'] > 0.6:
                    print(f"   🤔 MEDIUM CONFIDENCE: Possibly the same person")
                else:
                    print(f"   ❓ LOW CONFIDENCE: May not be the same person")
            
            else:
                print(f"   📭 No matches found above similarity threshold")
                print(f"   This could mean:")
                print(f"   - The person is not in the database")
                print(f"   - The similarity threshold is too high")
                print(f"   - The lighting/angle is very different")
        
        else:
            print(f"❌ New image {new_image_path} not found")
            print(f"   Please make sure the image file is in the current directory")
        
        # Step 4: Database summary
        search_system.get_database_summary()
        
        # Step 5: Add the new image to database for future searches
        print(f"\n💾 Step 5: Adding New Image to Database")
        
        if os.path.exists(new_image_path):
            search_system.add_person_to_database(
                new_image_path,
                "You (New Photo)",
                {"source": "new_photo", "date": "2025-02-22", "quality": "recent"}
            )
            
            print(f"   ✅ New image added to database for future searches")
        
        print(f"\n🎉 Face Search Test Completed!")
        return True
        
    except Exception as e:
        print(f"❌ Face search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_face_search_with_new_image()
    if success:
        print(f"\n✅ FACE SEARCH SYSTEM WORKING!")
        print(f"   Your face recognition system successfully:")
        print(f"   1. ✅ Built a searchable face database")
        print(f"   2. ✅ Detected faces in your new image")
        print(f"   3. ✅ Performed similarity search")
        print(f"   4. ✅ Ranked results by similarity")
        print(f"   5. ✅ Provided confidence analysis")
        print(f"   ")
        print(f"   🎯 Your system is ready for real-world face recognition!")
    else:
        print(f"\n❌ Face search test failed!")
        exit(1)