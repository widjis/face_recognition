#!/usr/bin/env python3
"""
Script to rebuild face recognition database using images from data folder.
"""

import os
import sys
import shutil
import cv2
from pathlib import Path
from face_recognition.pipeline import FaceRecognitionPipeline
from face_recognition.config.manager import ConfigurationManager

def clear_database(database_name: str):
    """Clear existing database files."""
    db_path = Path(database_name)
    if db_path.exists():
        print(f"🗑️ Clearing existing database: {database_name}")
        shutil.rmtree(db_path)
    
    # Create fresh database directory
    db_path.mkdir(exist_ok=True)
    print(f"✅ Created fresh database directory: {database_name}")

def rebuild_database(data_folder: str, database_name: str, profile: str = "development"):
    """Rebuild database using images from data folder."""
    
    # Clear existing database
    clear_database(database_name)
    
    # Initialize pipeline
    config_manager = ConfigurationManager()
    config_manager.load_config(profile=profile)
    
    pipeline = FaceRecognitionPipeline(
        config_manager=config_manager,
        db_path=database_name
    )
    
    print(f"🎯 Face Recognition Pipeline Initialized")
    print(f"   Database: {database_name}")
    print(f"   Configuration: {profile}")
    
    # Get all image files from data folder
    data_path = Path(data_folder)
    if not data_path.exists():
        print(f"❌ Data folder not found: {data_folder}")
        return False
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for file_path in data_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    if not image_files:
        print(f"❌ No image files found in: {data_folder}")
        return False
    
    print(f"📁 Found {len(image_files)} images to process")
    
    # Process each image
    successful_registrations = 0
    failed_registrations = 0
    
    for i, image_path in enumerate(image_files, 1):
        try:
            # Extract person ID from filename (remove extension)
            person_id = image_path.stem
            
            print(f"📸 Processing {i}/{len(image_files)}: {person_id}")
            
            # Load and register face
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"   ❌ Could not load image: {image_path.name}")
                failed_registrations += 1
                continue
                
            embedding_id = pipeline.add_face_to_database(
                image=image,
                metadata={
                    "source": "data_folder",
                    "filename": image_path.name,
                    "person_id": person_id
                },
                person_id=person_id
            )
            
            print(f"   ✅ Successfully registered: {person_id} (ID: {embedding_id})")
            successful_registrations += 1
                
        except Exception as e:
            print(f"   ❌ Error processing {image_path.name}: {str(e)}")
            failed_registrations += 1
    
    # Summary
    print(f"\n📊 Registration Summary:")
    print(f"   ✅ Successful: {successful_registrations}")
    print(f"   ❌ Failed: {failed_registrations}")
    print(f"   📁 Total processed: {len(image_files)}")
    
    if successful_registrations > 0:
        print(f"\n🎉 Database rebuilt successfully!")
        print(f"   Database: {database_name}")
        print(f"   Registered faces: {successful_registrations}")
        return True
    else:
        print(f"\n❌ No faces were successfully registered!")
        return False

def main():
    """Main function."""
    # Configuration
    data_folder = "data"
    database_name = "test_pipeline_db"  # You can change this
    profile = "development"
    
    print("🚀 Starting database rebuild process...")
    print(f"   Data folder: {data_folder}")
    print(f"   Database: {database_name}")
    print(f"   Profile: {profile}")
    print()
    
    success = rebuild_database(data_folder, database_name, profile)
    
    if success:
        print(f"\n✅ Database rebuild completed successfully!")
        print(f"   You can now run real-time face recognition with:")
        print(f"   python realtime_face_recognition.py --database {database_name} --profile {profile}")
    else:
        print(f"\n❌ Database rebuild failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()