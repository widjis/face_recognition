"""Demonstration of the configuration management system."""

import os
import tempfile
import shutil
from datetime import datetime

# Import configuration management
from face_recognition.config import ConfigurationManager, FaceRecognitionConfig
from face_recognition.config.settings import ConfigurationProfiles

def demo_configuration_management():
    """Demonstrate comprehensive configuration management features."""
    print("🎯 Configuration Management System Demo")
    print("=" * 50)
    
    # Create temporary directory for demo
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Step 1: Basic Configuration Loading
        print("\n📝 Step 1: Basic Configuration Management")
        
        config_manager = ConfigurationManager()
        
        # Load default configuration
        config = config_manager.load_config()
        print(f"   ✅ Default config loaded: {config.system_name}")
        print(f"   Environment: {config.environment}")
        print(f"   Embedding dimension: {config.embedding.embedding_dim}")
        print(f"   Detection method: {config.face_detection.method}")
        
        # Step 2: Configuration Profiles
        print("\n🎭 Step 2: Predefined Configuration Profiles")
        
        profiles = ["development", "production", "high_accuracy", "high_speed", "memory_efficient"]
        
        for profile_name in profiles:
            profile_config = config_manager.load_config(profile=profile_name)
            print(f"   ✅ {profile_name.title()} Profile:")
            print(f"      Environment: {profile_config.environment}")
            print(f"      Log level: {profile_config.logging.log_level}")
            print(f"      Cache enabled: {profile_config.performance.enable_caching}")
            print(f"      Index type: {profile_config.vector_database.index_type}")
        
        # Step 3: Configuration Updates
        print("\n⚙️ Step 3: Runtime Configuration Updates")
        
        # Load production profile for updates
        config_manager.load_config(profile="production")
        
        # Show original settings
        print(f"   Original settings:")
        print(f"   - System name: {config_manager.config.system_name}")
        print(f"   - Similarity threshold: {config_manager.config.search.similarity_threshold}")
        print(f"   - Batch size: {config_manager.config.embedding.batch_size}")
        
        # Apply updates
        updates = {
            'system_name': 'My Custom Face Recognition System',
            'search': {
                'similarity_threshold': 0.85,
                'top_k': 15
            },
            'embedding': {
                'batch_size': 64
            },
            'reranking': {
                'quality_weight': 0.25,
                'similarity_weight': 0.55
            }
        }
        
        updated_config = config_manager.update_config(updates)
        
        print(f"\n   Updated settings:")
        print(f"   - System name: {updated_config.system_name}")
        print(f"   - Similarity threshold: {updated_config.search.similarity_threshold}")
        print(f"   - Batch size: {updated_config.embedding.batch_size}")
        print(f"   - Quality weight: {updated_config.reranking.quality_weight}")
        
        # Step 4: Configuration Validation
        print("\n✅ Step 4: Configuration Validation")
        
        validation_result = config_manager.validate_current_config()
        print(f"   Configuration valid: {validation_result['valid']}")
        print(f"   Warnings: {validation_result['warning_count']}")
        
        if validation_result.get('warnings'):
            for warning in validation_result['warnings']:
                print(f"   ⚠️ {warning}")
        
        # Step 5: Configuration Persistence
        print("\n💾 Step 5: Configuration Persistence")
        
        # Save configuration in different formats
        yaml_path = os.path.join(temp_dir, "config.yaml")
        json_path = os.path.join(temp_dir, "config.json")
        
        config_manager.save_config(yaml_path, format="yaml")
        config_manager.save_config(json_path, format="json")
        
        print(f"   ✅ Saved YAML config: {os.path.basename(yaml_path)}")
        print(f"   ✅ Saved JSON config: {os.path.basename(json_path)}")
        
        # Load from saved file
        new_manager = ConfigurationManager()
        loaded_config = new_manager.load_config(yaml_path)
        
        print(f"   ✅ Loaded from file: {loaded_config.system_name}")
        
        # Step 6: Configuration History and Rollback
        print("\n🔄 Step 6: Configuration History and Rollback")
        
        # Make several changes to build history
        config_manager.update_config({'system_name': 'Version 1'})
        config_manager.update_config({'system_name': 'Version 2'})
        config_manager.update_config({'system_name': 'Version 3'})
        
        print(f"   Current system name: {config_manager.config.system_name}")
        
        # Show history
        history = config_manager.get_config_history()
        print(f"   Configuration history entries: {len(history)}")
        
        # Rollback
        rolled_back = config_manager.rollback_config(steps=2)
        print(f"   After rollback: {rolled_back.system_name}")
        
        # Step 7: Configuration Summary and Export
        print("\n📊 Step 7: Configuration Summary and Export")
        
        summary = config_manager.get_config_summary()
        print(f"   Configuration Summary:")
        print(f"   - System: {summary['system_name']}")
        print(f"   - Version: {summary['version']}")
        print(f"   - Environment: {summary['environment']}")
        print(f"   - Database path: {summary['database_path']}")
        print(f"   - Embedding model: {summary['embedding_model']}")
        
        print(f"\n   Enabled Features:")
        for feature, enabled in summary['enabled_features'].items():
            status = "✅" if enabled else "❌"
            print(f"   {status} {feature.replace('_', ' ').title()}")
        
        print(f"\n   Performance Settings:")
        perf = summary['performance_settings']
        print(f"   - Caching: {'✅' if perf['enable_caching'] else '❌'}")
        print(f"   - Cache size: {perf['cache_size']}")
        print(f"   - Max workers: {perf['max_workers']}")
        print(f"   - Memory limit: {perf['max_memory_usage_mb']} MB")
        
        # Export configuration
        export_path = os.path.join(temp_dir, "exported_config.yaml")
        config_manager.export_config(export_path)
        print(f"   ✅ Configuration exported for sharing")
        
        # Step 8: Advanced Configuration Features
        print("\n🔬 Step 8: Advanced Configuration Features")
        
        # Test different use case configurations
        use_cases = {
            "Security System": {
                'face_detection': {'min_neighbors': 8},
                'search': {'similarity_threshold': 0.9},
                'reranking': {'quality_weight': 0.4, 'similarity_weight': 0.4}
            },
            "Social Media App": {
                'face_detection': {'min_neighbors': 3},
                'search': {'similarity_threshold': 0.6, 'top_k': 20},
                'performance': {'max_workers': 12}
            },
            "Mobile App": {
                'embedding': {'batch_size': 8},
                'performance': {'cache_size': 50, 'max_memory_usage_mb': 256},
                'vector_database': {'max_database_size': 5000}
            }
        }
        
        print(f"   Testing configurations for different use cases:")
        
        for use_case, settings in use_cases.items():
            # Reset to defaults
            config_manager.reset_to_defaults()
            
            # Apply use case specific settings
            config_manager.update_config(settings)
            
            print(f"   ✅ {use_case}:")
            print(f"      Similarity threshold: {config_manager.config.search.similarity_threshold}")
            print(f"      Batch size: {config_manager.config.embedding.batch_size}")
            print(f"      Cache size: {config_manager.config.performance.cache_size}")
        
        # Step 9: Configuration Schema and Documentation
        print("\n📚 Step 9: Configuration Schema")
        
        # Show available configuration options
        print("   Available Configuration Profiles:")
        profiles = ["development", "production", "high_accuracy", "high_speed", "memory_efficient"]
        for profile in profiles:
            print(f"   - {profile}")
        
        print("\n   Key Configuration Sections:")
        sections = [
            "face_detection", "embedding", "vector_database", 
            "search", "reranking", "performance", "logging"
        ]
        for section in sections:
            print(f"   - {section}")
        
        # Step 10: Configuration Best Practices Demo
        print("\n💡 Step 10: Configuration Best Practices")
        
        print("   Best Practices Demonstrated:")
        print("   ✅ Use predefined profiles for common scenarios")
        print("   ✅ Validate configuration after changes")
        print("   ✅ Keep configuration history for rollbacks")
        print("   ✅ Export configurations for sharing/backup")
        print("   ✅ Use environment-specific settings")
        print("   ✅ Monitor configuration consistency")
        
        # Final validation
        final_validation = config_manager.validate_current_config()
        print(f"\n   Final configuration status: {'✅ Valid' if final_validation['valid'] else '❌ Invalid'}")
        
        print(f"\n🎉 Configuration Management Demo Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    success = demo_configuration_management()
    if success:
        print(f"\n✅ TASK 7: CONFIGURATION MANAGEMENT - COMPLETED!")
        print(f"   Advanced Configuration Features:")
        print(f"   1. ✅ Pydantic-based configuration validation")
        print(f"   2. ✅ Predefined profiles (dev, prod, high-accuracy, etc.)")
        print(f"   3. ✅ Runtime configuration updates")
        print(f"   4. ✅ YAML and JSON configuration persistence")
        print(f"   5. ✅ Configuration history and rollback")
        print(f"   6. ✅ Configuration export and import")
        print(f"   7. ✅ Consistency validation and warnings")
        print(f"   8. ✅ Use-case specific configurations")
        print(f"   ")
        print(f"   🎯 Your face recognition system is now FULLY CONFIGURABLE!")
        print(f"   Easy to adapt for different environments and use cases!")
    else:
        print(f"\n❌ Task 7 failed!")
        exit(1)