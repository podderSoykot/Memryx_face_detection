"""
============
Information:
============
Project: Multi-Camera Face Recognition System
File Name: config.py

============
Description:
============
Configuration file for camera setup, database connections, and system parameters.
"""

import os
from typing import Dict, List, Tuple, Optional

###################################################################################################
# Camera Configuration
###################################################################################################

CAMERA_CONFIG = {
    # Camera sources - can be device paths, URLs, or video files
    'sources': [
        '/dev/video0',    # Primary camera
        '/dev/video1',    # Secondary camera
        # '/dev/video2',    # Additional camera
        # 'rtsp://192.168.1.100:554/stream1',  # IP camera example
        # 'sample_video.mp4',  # Video file example
    ],
    
    # Camera-specific settings
    'camera_settings': {
        0: {
            'name': 'Main Entrance',
            'location': 'entrance_main',
            'resolution': (1280, 720),
            'fps': 30,
            'enabled': True,
        },
        1: {
            'name': 'Side Entrance',
            'location': 'entrance_side',
            'resolution': (1280, 720),
            'fps': 30,
            'enabled': True,
        },
        2: {
            'name': 'Lobby',
            'location': 'lobby',
            'resolution': (1920, 1080),
            'fps': 25,
            'enabled': False,
        },
    },
    
    # Display settings
    'display': {
        'show_video': True,
        'show_fps': True,
        'show_detection_boxes': True,
        'show_face_info': True,
        'window_scale': 0.8,
    }
}

###################################################################################################
# MemryX Model Configuration
###################################################################################################

MEMRYX_CONFIG = {
    # Model paths for MemryX DFP files
    'models': {
        'retinaface': {
            'dfp_path': 'dfp_models/retinaface_memryx.dfp',
            'input_size': (320, 320),
            'confidence_threshold': 0.7,
            'nms_threshold': 0.4,
        },
        'facenet512': {
            'dfp_path': 'dfp_models/facenet512_memryx_raw_embeddings.dfp',
            'input_size': (112, 112),
            'embedding_size': 512,
        },
        'age_gender': {
            'dfp_path': 'dfp_models/genderage.dfp',
            'input_size': (224, 224),
            'age_output_size': 101,  
            'gender_output_size': 2, 
        }
    },
    
    # Performance settings
    'performance': {
        'max_batch_size': 4,
        'use_async': True,
        'thread_count': 4,
    }
}

###################################################################################################
# Face Recognition Configuration
###################################################################################################

FACE_RECOGNITION_CONFIG = {
    # Face detection settings
    'detection': {
        'min_face_size': 50,
        'max_faces_per_frame': 10,
        'detection_threshold': 0.7,
        'face_padding': 0.2,  
    },
    
    # Face embedding settings
    'embedding': {
        'similarity_threshold': 0.6,  
        'embedding_dimension': 512,
        'normalization': True,
    },
    
    # FAISS database settings
    'faiss': {
        'index_type': 'IndexFlatIP', 
        'index_file': 'data/face_embeddings.index',
        'metadata_file': 'data/face_metadata.json',
        'save_interval': 100,  
    },
    
    # Age/Gender inference settings
    'demographics': {
        'age_range': (0, 100),
        'gender_labels': ['Female', 'Male'],
        'confidence_threshold': 0.5,
        'cache_duration': 30,  # seconds
    }
}

###################################################################################################
# Database Configuration
###################################################################################################

DATABASE_CONFIG = {
    'sqlite': {
        'database_path': os.getenv('DB_PATH', 'data/face_recognition.db'),
        'timeout': 30.0,  # Database timeout in seconds
        'check_same_thread': False,  # Allow multi-threading
        'isolation_level': None,  # Autocommit mode
    },
    
    'tables': {
        'face_detections': 'face_detections',
        'face_demographics': 'face_demographics',
        'face_embeddings': 'face_embeddings',
    }
}

###################################################################################################
# Logging Configuration
###################################################################################################

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/face_recognition.log',
    'max_bytes': 10485760,  # 10MB
    'backup_count': 5,
    'console_output': True,
}

###################################################################################################
# System Configuration
###################################################################################################

SYSTEM_CONFIG = {
    # Processing settings
    'processing': {
        'frame_skip_interval': 1,  # Process every N frames
        'max_queue_size': 10,
        'worker_threads': 4,
        'batch_processing': True,
    },
    
    # Storage settings
    'storage': {
        'data_directory': 'data',
        'log_directory': 'logs',
        'model_directory': 'models',
        'backup_directory': 'backups',
    },
    
    # Security settings
    'security': {
        'enable_encryption': False,
        'api_key_required': False,
        'max_failed_attempts': 5,
        'lockout_duration': 300,  # seconds
    }
}

###################################################################################################
# Validation Functions
###################################################################################################

def validate_camera_config() -> bool:
    """Validate camera configuration settings."""
    try:
        if not CAMERA_CONFIG['sources']:
            raise ValueError("No camera sources configured")
        
        for idx, source in enumerate(CAMERA_CONFIG['sources']):
            if not isinstance(source, str):
                raise ValueError(f"Camera source {idx} must be a string")
        
        return True
    except Exception as e:
        print(f"Camera configuration validation failed: {e}")
        return False

def validate_memryx_config() -> bool:
    """Validate MemryX model configuration."""
    try:
        required_models = ['retinaface', 'facenet512', 'age_gender']
        
        for model_name in required_models:
            if model_name not in MEMRYX_CONFIG['models']:
                raise ValueError(f"Missing configuration for {model_name}")
            
            model_config = MEMRYX_CONFIG['models'][model_name]
            if 'dfp_path' not in model_config:
                raise ValueError(f"Missing DFP path for {model_name}")
        
        return True
    except Exception as e:
        print(f"MemryX configuration validation failed: {e}")
        return False

def validate_database_config() -> bool:
    """Validate database configuration."""
    try:
        db_config = DATABASE_CONFIG['sqlite']
        
        # Check if database path is configured
        if 'database_path' not in db_config or not db_config['database_path']:
            raise ValueError("Missing or empty database_path")
        
        # Ensure the directory for the database exists
        db_dir = os.path.dirname(db_config['database_path'])
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        return True
    except Exception as e:
        print(f"Database configuration validation failed: {e}")
        return False

def get_camera_info(camera_id: int) -> Optional[Dict]:
    """Get camera information by ID."""
    return CAMERA_CONFIG['camera_settings'].get(camera_id)

def get_model_path(model_name: str) -> Optional[str]:
    """Get DFP model path by name."""
    model_config = MEMRYX_CONFIG['models'].get(model_name)
    return model_config.get('dfp_path') if model_config else None

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        SYSTEM_CONFIG['storage']['data_directory'],
        SYSTEM_CONFIG['storage']['log_directory'],
        SYSTEM_CONFIG['storage']['backup_directory'],
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

###################################################################################################
# Environment Setup
###################################################################################################

def setup_environment():
    """Setup the environment for the face recognition system."""
    # Create necessary directories
    create_directories()
    
    # Validate configurations
    validations = [
        validate_camera_config(),
        validate_memryx_config(),
        validate_database_config(),
    ]
    
    if not all(validations):
        raise RuntimeError("Configuration validation failed")
    
    print("Environment setup completed successfully")

###################################################################################################

if __name__ == "__main__":
    # Test configuration
    print("Testing configuration...")
    setup_environment()
    
    # Print configuration summary
    print(f"Cameras configured: {len(CAMERA_CONFIG['sources'])}")
    print(f"Models configured: {len(MEMRYX_CONFIG['models'])}")
    print(f"Database: {DATABASE_CONFIG['sqlite']['database_path']}")
    print("Configuration test completed!")

# eof 