"""Configuration manager for loading, saving, and managing configurations."""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

from .settings import FaceRecognitionConfig, ConfigurationProfiles
from ..exceptions import ConfigurationError


class ConfigurationManager:
    """Manages configuration loading, saving, and runtime updates."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.config: Optional[FaceRecognitionConfig] = None
        self.config_history: list = []
        self.auto_save = True
        
        # Default configuration locations
        self.default_config_paths = [
            "face_recognition_config.yaml",
            "face_recognition_config.json",
            ".face_recognition/config.yaml",
            os.path.expanduser("~/.face_recognition/config.yaml")
        ]
    
    def load_config(self, config_path: Optional[str] = None, 
                   profile: Optional[str] = None) -> FaceRecognitionConfig:
        """
        Load configuration from file or create default.
        
        Args:
            config_path: Path to configuration file
            profile: Predefined profile name (development, production, etc.)
            
        Returns:
            Loaded configuration
            
        Raises:
            ConfigurationError: If configuration loading fails
        """
        try:
            # Use profile if specified
            if profile:
                self.config = self._load_profile(profile)
                print(f"✅ Loaded '{profile}' configuration profile")
                return self.config
            
            # Determine config file path
            config_file = config_path or self.config_path
            
            if not config_file:
                # Try to find config file in default locations
                config_file = self._find_config_file()
            
            if config_file and os.path.exists(config_file):
                # Load from file
                self.config = self._load_from_file(config_file)
                self.config_path = config_file
                print(f"✅ Loaded configuration from {config_file}")
            else:
                # Create default configuration
                self.config = FaceRecognitionConfig()
                print("✅ Created default configuration")
            
            # Validate configuration
            warnings = self.config.validate_consistency()
            if warnings:
                print("⚠️ Configuration warnings:")
                for warning in warnings:
                    print(f"   - {warning}")
            
            # Save config history
            self._save_to_history()
            
            return self.config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def save_config(self, config_path: Optional[str] = None, 
                   format: str = "yaml") -> str:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration
            format: File format ('yaml' or 'json')
            
        Returns:
            Path where configuration was saved
            
        Raises:
            ConfigurationError: If saving fails
        """
        if not self.config:
            raise ConfigurationError("No configuration loaded to save")
        
        try:
            # Determine save path
            save_path = config_path or self.config_path
            if not save_path:
                save_path = f"face_recognition_config.{format}"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # Convert config to dict
            config_dict = self.config.dict()
            
            # Add metadata
            config_dict['_metadata'] = {
                'saved_at': datetime.now().isoformat(),
                'version': self.config.version,
                'format_version': '1.0'
            }
            
            # Save based on format
            if format.lower() == 'yaml':
                with open(save_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                with open(save_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ConfigurationError(f"Unsupported format: {format}")
            
            self.config_path = save_path
            print(f"✅ Configuration saved to {save_path}")
            
            return save_path
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")
    
    def update_config(self, updates: Dict[str, Any], 
                     validate: bool = True) -> FaceRecognitionConfig:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            validate: Whether to validate the updated configuration
            
        Returns:
            Updated configuration
            
        Raises:
            ConfigurationError: If update fails
        """
        if not self.config:
            raise ConfigurationError("No configuration loaded to update")
        
        try:
            # Save current config to history
            self._save_to_history()
            
            # Apply updates
            config_dict = self.config.dict()
            config_dict = self._deep_update(config_dict, updates)
            
            # Create new config instance
            new_config = FaceRecognitionConfig(**config_dict)
            
            if validate:
                warnings = new_config.validate_consistency()
                if warnings:
                    print("⚠️ Configuration update warnings:")
                    for warning in warnings:
                        print(f"   - {warning}")
            
            self.config = new_config
            
            # Auto-save if enabled
            if self.auto_save and self.config_path:
                self.save_config()
            
            print("✅ Configuration updated successfully")
            return self.config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to update configuration: {str(e)}")
    
    def get_config(self) -> FaceRecognitionConfig:
        """Get current configuration."""
        if not self.config:
            return self.load_config()
        return self.config
    
    def reset_to_defaults(self) -> FaceRecognitionConfig:
        """Reset configuration to defaults."""
        self._save_to_history()
        self.config = FaceRecognitionConfig()
        print("✅ Configuration reset to defaults")
        return self.config
    
    def load_profile(self, profile_name: str) -> FaceRecognitionConfig:
        """Load a predefined configuration profile."""
        self.config = self._load_profile(profile_name)
        print(f"✅ Loaded '{profile_name}' profile")
        return self.config
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        if not self.config:
            return {}
        
        return {
            'system_name': self.config.system_name,
            'version': self.config.version,
            'environment': self.config.environment,
            'enabled_features': {
                'face_detection': self.config.enable_face_detection,
                'embedding_extraction': self.config.enable_embedding_extraction,
                'similarity_search': self.config.enable_similarity_search,
                'reranking': self.config.enable_reranking
            },
            'performance_settings': self.config.get_performance_settings(),
            'database_path': self.config.vector_database.db_path,
            'embedding_model': self.config.embedding.model_name,
            'detection_method': self.config.face_detection.method
        }
    
    def validate_current_config(self) -> Dict[str, Any]:
        """Validate current configuration and return results."""
        if not self.config:
            return {'valid': False, 'error': 'No configuration loaded'}
        
        try:
            warnings = self.config.validate_consistency()
            return {
                'valid': True,
                'warnings': warnings,
                'warning_count': len(warnings)
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def get_config_history(self) -> list:
        """Get configuration change history."""
        return self.config_history.copy()
    
    def rollback_config(self, steps: int = 1) -> FaceRecognitionConfig:
        """Rollback configuration to previous state."""
        if len(self.config_history) < steps:
            raise ConfigurationError(f"Cannot rollback {steps} steps, only {len(self.config_history)} available")
        
        # Get previous config
        previous_config_dict = self.config_history[-(steps)]['config']
        self.config = FaceRecognitionConfig(**previous_config_dict)
        
        print(f"✅ Configuration rolled back {steps} step(s)")
        return self.config
    
    def export_config(self, export_path: str, include_metadata: bool = True) -> str:
        """Export configuration for sharing or backup."""
        if not self.config:
            raise ConfigurationError("No configuration to export")
        
        try:
            config_dict = self.config.dict()
            
            if include_metadata:
                config_dict['_export_metadata'] = {
                    'exported_at': datetime.now().isoformat(),
                    'exported_by': 'ConfigurationManager',
                    'original_path': self.config_path,
                    'export_version': '1.0'
                }
            
            # Determine format from extension
            if export_path.endswith('.yaml') or export_path.endswith('.yml'):
                with open(export_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                with open(export_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            
            print(f"✅ Configuration exported to {export_path}")
            return export_path
            
        except Exception as e:
            raise ConfigurationError(f"Failed to export configuration: {str(e)}")
    
    def _load_profile(self, profile_name: str) -> FaceRecognitionConfig:
        """Load a predefined profile."""
        profiles = {
            'development': ConfigurationProfiles.development,
            'production': ConfigurationProfiles.production,
            'high_accuracy': ConfigurationProfiles.high_accuracy,
            'high_speed': ConfigurationProfiles.high_speed,
            'memory_efficient': ConfigurationProfiles.memory_efficient
        }
        
        if profile_name not in profiles:
            available = ', '.join(profiles.keys())
            raise ConfigurationError(f"Unknown profile '{profile_name}'. Available: {available}")
        
        return profiles[profile_name]()
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in default locations."""
        for path in self.default_config_paths:
            if os.path.exists(path):
                return path
        return None
    
    def _load_from_file(self, config_path: str) -> FaceRecognitionConfig:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            
            # Remove metadata if present
            config_dict.pop('_metadata', None)
            config_dict.pop('_export_metadata', None)
            
            return FaceRecognitionConfig(**config_dict)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {config_path}: {str(e)}")
    
    def _save_to_history(self):
        """Save current configuration to history."""
        if self.config:
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config.dict()
            }
            self.config_history.append(history_entry)
            
            # Keep only last 10 entries
            if len(self.config_history) > 10:
                self.config_history = self.config_history[-10:]
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """Deep update dictionary with nested values."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                base_dict[key] = self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict


class ConfigurationValidator:
    """Utility class for validating configurations."""
    
    @staticmethod
    def validate_file_format(config_path: str) -> bool:
        """Validate configuration file format."""
        try:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                with open(config_path, 'r') as f:
                    yaml.safe_load(f)
            elif config_path.endswith('.json'):
                with open(config_path, 'r') as f:
                    json.load(f)
            else:
                return False
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_config_dict(config_dict: Dict) -> tuple:
        """Validate configuration dictionary."""
        try:
            config = FaceRecognitionConfig(**config_dict)
            warnings = config.validate_consistency()
            return True, warnings
        except Exception as e:
            return False, [str(e)]
    
    @staticmethod
    def get_config_schema() -> Dict:
        """Get configuration schema for documentation."""
        return FaceRecognitionConfig.schema()