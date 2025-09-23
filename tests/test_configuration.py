"""Tests for configuration management system."""

import pytest
import tempfile
import os
import json
import yaml
from face_recognition.config import ConfigurationManager, FaceRecognitionConfig
from face_recognition.config.settings import ConfigurationProfiles
from face_recognition.exceptions import ConfigurationError


class TestFaceRecognitionConfig:
    """Test cases for FaceRecognitionConfig."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = FaceRecognitionConfig()
        
        assert config.system_name == "Face Recognition System"
        assert config.version == "1.0.0"
        assert config.environment == "development"
        assert config.enable_face_detection is True
        assert config.face_detection.method == "haar"
        assert config.embedding.embedding_dim == 512
        assert config.vector_database.dimension == 512
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = FaceRecognitionConfig()
        warnings = config.validate_consistency()
        
        # Default config should have no warnings
        assert isinstance(warnings, list)
    
    def test_config_validation_with_inconsistency(self):
        """Test configuration validation with inconsistent settings."""
        config = FaceRecognitionConfig()
        config.embedding.embedding_dim = 256
        config.vector_database.dimension = 512
        
        warnings = config.validate_consistency()
        assert len(warnings) > 0
        assert any("dimension" in warning.lower() for warning in warnings)
    
    def test_get_component_config(self):
        """Test getting component configuration."""
        config = FaceRecognitionConfig()
        
        face_detection_config = config.get_component_config('face_detection')
        assert face_detection_config.method == "haar"
        
        embedding_config = config.get_component_config('embedding')
        assert embedding_config.embedding_dim == 512
    
    def test_get_invalid_component_config(self):
        """Test getting invalid component configuration."""
        config = FaceRecognitionConfig()
        
        with pytest.raises(ValueError, match="Unknown component"):
            config.get_component_config('invalid_component')
    
    def test_is_feature_enabled(self):
        """Test feature enablement checking."""
        config = FaceRecognitionConfig()
        
        assert config.is_feature_enabled('face_detection') is True
        assert config.is_feature_enabled('embedding_extraction') is True
        assert config.is_feature_enabled('invalid_feature') is False
    
    def test_get_performance_settings(self):
        """Test getting performance settings."""
        config = FaceRecognitionConfig()
        
        perf_settings = config.get_performance_settings()
        
        assert 'enable_caching' in perf_settings
        assert 'cache_size' in perf_settings
        assert 'max_workers' in perf_settings
        assert isinstance(perf_settings['cache_size'], int)


class TestConfigurationProfiles:
    """Test cases for predefined configuration profiles."""
    
    def test_development_profile(self):
        """Test development configuration profile."""
        config = ConfigurationProfiles.development()
        
        assert config.environment == "development"
        assert config.logging.log_level == "DEBUG"
        assert config.performance.enable_caching is False
        assert config.vector_database.auto_save is True
    
    def test_production_profile(self):
        """Test production configuration profile."""
        config = ConfigurationProfiles.production()
        
        assert config.environment == "production"
        assert config.logging.log_level == "INFO"
        assert config.performance.enable_caching is True
        assert config.performance.cache_size == 5000
        assert config.vector_database.index_type == "hnsw"
    
    def test_high_accuracy_profile(self):
        """Test high accuracy configuration profile."""
        config = ConfigurationProfiles.high_accuracy()
        
        assert config.face_detection.min_neighbors == 8
        assert config.search.similarity_threshold == 0.8
        assert config.reranking.quality_weight == 0.3
        assert config.vector_database.index_type == "flat"
    
    def test_high_speed_profile(self):
        """Test high speed configuration profile."""
        config = ConfigurationProfiles.high_speed()
        
        assert config.face_detection.min_neighbors == 3
        assert config.search.similarity_threshold == 0.6
        assert config.enable_reranking is False
        assert config.performance.max_workers == 16
        assert config.vector_database.index_type == "ivf"
    
    def test_memory_efficient_profile(self):
        """Test memory efficient configuration profile."""
        config = ConfigurationProfiles.memory_efficient()
        
        assert config.embedding.batch_size == 8
        assert config.performance.cache_size == 100
        assert config.performance.max_memory_usage_mb == 512
        assert config.vector_database.max_database_size == 10000


class TestConfigurationManager:
    """Test cases for ConfigurationManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigurationManager()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = self.config_manager.load_config()
        
        assert isinstance(config, FaceRecognitionConfig)
        assert config.system_name == "Face Recognition System"
        assert self.config_manager.config is not None
    
    def test_load_profile(self):
        """Test loading predefined profile."""
        config = self.config_manager.load_config(profile="production")
        
        assert config.environment == "production"
        assert config.logging.log_level == "INFO"
    
    def test_load_invalid_profile(self):
        """Test loading invalid profile."""
        with pytest.raises(ConfigurationError, match="Unknown profile"):
            self.config_manager.load_config(profile="invalid_profile")
    
    def test_save_and_load_yaml_config(self):
        """Test saving and loading YAML configuration."""
        # Load default config
        original_config = self.config_manager.load_config()
        original_config.system_name = "Test System"
        
        # Save to temporary file
        config_path = os.path.join(self.temp_dir, "test_config.yaml")
        saved_path = self.config_manager.save_config(config_path, format="yaml")
        
        assert os.path.exists(saved_path)
        
        # Load from file
        new_manager = ConfigurationManager()
        loaded_config = new_manager.load_config(config_path)
        
        assert loaded_config.system_name == "Test System"
    
    def test_save_and_load_json_config(self):
        """Test saving and loading JSON configuration."""
        # Load default config
        original_config = self.config_manager.load_config()
        original_config.system_name = "Test System JSON"
        
        # Save to temporary file
        config_path = os.path.join(self.temp_dir, "test_config.json")
        saved_path = self.config_manager.save_config(config_path, format="json")
        
        assert os.path.exists(saved_path)
        
        # Load from file
        new_manager = ConfigurationManager()
        loaded_config = new_manager.load_config(config_path)
        
        assert loaded_config.system_name == "Test System JSON"
    
    def test_update_config(self):
        """Test updating configuration."""
        # Load default config
        self.config_manager.load_config()
        
        # Update configuration
        updates = {
            'system_name': 'Updated System',
            'face_detection': {
                'min_neighbors': 7
            },
            'embedding': {
                'batch_size': 16
            }
        }
        
        updated_config = self.config_manager.update_config(updates)
        
        assert updated_config.system_name == 'Updated System'
        assert updated_config.face_detection.min_neighbors == 7
        assert updated_config.embedding.batch_size == 16
    
    def test_get_config_summary(self):
        """Test getting configuration summary."""
        self.config_manager.load_config()
        
        summary = self.config_manager.get_config_summary()
        
        assert 'system_name' in summary
        assert 'version' in summary
        assert 'enabled_features' in summary
        assert 'performance_settings' in summary
        assert isinstance(summary['enabled_features'], dict)
    
    def test_validate_current_config(self):
        """Test validating current configuration."""
        self.config_manager.load_config()
        
        validation_result = self.config_manager.validate_current_config()
        
        assert 'valid' in validation_result
        assert validation_result['valid'] is True
        assert 'warnings' in validation_result
    
    def test_config_history(self):
        """Test configuration history tracking."""
        # Load initial config
        self.config_manager.load_config()
        
        # Make some updates
        self.config_manager.update_config({'system_name': 'Version 1'})
        self.config_manager.update_config({'system_name': 'Version 2'})
        
        history = self.config_manager.get_config_history()
        
        assert len(history) >= 2
        assert all('timestamp' in entry for entry in history)
        assert all('config' in entry for entry in history)
    
    def test_rollback_config(self):
        """Test configuration rollback."""
        # Load initial config
        self.config_manager.load_config()
        original_name = self.config_manager.config.system_name
        
        # Make updates
        self.config_manager.update_config({'system_name': 'Updated Name'})
        assert self.config_manager.config.system_name == 'Updated Name'
        
        # Rollback
        rolled_back_config = self.config_manager.rollback_config(steps=1)
        assert rolled_back_config.system_name == original_name
    
    def test_rollback_config_insufficient_history(self):
        """Test rollback with insufficient history."""
        self.config_manager.load_config()
        
        with pytest.raises(ConfigurationError, match="Cannot rollback"):
            self.config_manager.rollback_config(steps=10)
    
    def test_export_config(self):
        """Test configuration export."""
        self.config_manager.load_config()
        
        export_path = os.path.join(self.temp_dir, "exported_config.yaml")
        exported_path = self.config_manager.export_config(export_path)
        
        assert os.path.exists(exported_path)
        
        # Verify exported content
        with open(exported_path, 'r') as f:
            exported_data = yaml.safe_load(f)
        
        assert '_export_metadata' in exported_data
        assert 'system_name' in exported_data
    
    def test_reset_to_defaults(self):
        """Test resetting configuration to defaults."""
        # Load and modify config
        self.config_manager.load_config()
        self.config_manager.update_config({'system_name': 'Modified System'})
        
        # Reset to defaults
        default_config = self.config_manager.reset_to_defaults()
        
        assert default_config.system_name == "Face Recognition System"
        assert default_config.environment == "development"
    
    def test_auto_save_disabled(self):
        """Test configuration with auto-save disabled."""
        self.config_manager.auto_save = False
        self.config_manager.load_config()
        
        # Update should not auto-save
        self.config_manager.update_config({'system_name': 'No Auto Save'})
        
        # Config should be updated but not saved to file
        assert self.config_manager.config.system_name == 'No Auto Save'