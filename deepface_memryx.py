"""
============
Information:
============
Project: Multi-Camera Face Recognition System
File Name: deepface.py

============
Description:
============
Comprehensive DeepFace integration with MemryX acceleration for face detection, 
recognition, and demographic analysis. Includes FAISS database and SQLite logging.
"""

import os
import json
import uuid
import time
import logging
import threading
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

import cv2
import numpy as np
import faiss
import memryx as mx

from config import (
    MEMRYX_CONFIG, 
    FACE_RECOGNITION_CONFIG, 
    DATABASE_CONFIG, 
    LOGGING_CONFIG
)

###################################################################################################
# Setup logging
###################################################################################################

logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['file']),
        logging.StreamHandler() if LOGGING_CONFIG['console_output'] else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

###################################################################################################
# Database Manager
###################################################################################################

class DatabaseManager:
    """Handles SQLite database operations for face recognition system."""
    
    def __init__(self):
        self.db_path = DATABASE_CONFIG['sqlite']['database_path']
        self.lock = threading.Lock()
        self._ensure_db_directory()
        self.create_tables()
    
    def _ensure_db_directory(self):
        """Ensure the directory for the database file exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    def get_connection(self):
        """Get a SQLite database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable row access by column name
        return conn
    
    def create_tables(self):
        """Create necessary database tables."""
        tables = {
            'face_detections': """
                CREATE TABLE IF NOT EXISTS face_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_uuid TEXT NOT NULL,
                    camera_id INTEGER NOT NULL,
                    camera_name TEXT,
                    location TEXT,
                    detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    bbox_x INTEGER,
                    bbox_y INTEGER,
                    bbox_width INTEGER,
                    bbox_height INTEGER,
                    confidence REAL
                );
            """,
            'face_demographics': """
                CREATE TABLE IF NOT EXISTS face_demographics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_uuid TEXT NOT NULL,
                    age INTEGER,
                    gender TEXT,
                    gender_confidence REAL,
                    analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (face_uuid) REFERENCES face_detections(face_uuid)
                );
            """,
            'face_embeddings': """
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_uuid TEXT UNIQUE NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_model TEXT,
                    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
        }
        
        with self.lock:
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                for table_name, query in tables.items():
                    cursor.execute(query)
                    logger.info(f"Table {table_name} created/verified")
                
                conn.commit()
                conn.close()
                logger.info("SQLite database tables initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to create tables: {e}")
                raise
    
    def log_face_detection(self, face_uuid: str, camera_id: int, camera_name: str, 
                          location: str, bbox: Tuple[int, int, int, int], confidence: float):
        """Log face detection to database."""
        with self.lock:
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO face_detections 
                    (face_uuid, camera_id, camera_name, location, bbox_x, bbox_y, bbox_width, bbox_height, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (face_uuid, camera_id, camera_name, location, bbox[0], bbox[1], bbox[2], bbox[3], confidence))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logger.error(f"Failed to log face detection: {e}")
    
    def log_face_demographics(self, face_uuid: str, age: int, gender: str, gender_confidence: float):
        """Log face demographics to database."""
        with self.lock:
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO face_demographics (face_uuid, age, gender, gender_confidence)
                    VALUES (?, ?, ?, ?)
                """, (face_uuid, age, gender, gender_confidence))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logger.error(f"Failed to log face demographics: {e}")
    
    def store_face_embedding(self, face_uuid: str, embedding: np.ndarray, model_name: str):
        """Store face embedding in database."""
        with self.lock:
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                embedding_bytes = embedding.tobytes()
                
                # SQLite UPSERT syntax
                cursor.execute("""
                    INSERT OR REPLACE INTO face_embeddings 
                    (face_uuid, embedding, embedding_model, created_time, updated_time)
                    VALUES (
                        ?, ?, ?, 
                        COALESCE((SELECT created_time FROM face_embeddings WHERE face_uuid = ?), CURRENT_TIMESTAMP),
                        CURRENT_TIMESTAMP
                    )
                """, (face_uuid, embedding_bytes, model_name, face_uuid))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logger.error(f"Failed to store face embedding: {e}")
    
    def get_face_statistics(self) -> Dict[str, int]:
        """Get database statistics."""
        with self.lock:
            try:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                stats = {}
                
                # Count face detections
                cursor.execute("SELECT COUNT(*) as count FROM face_detections")
                stats['total_detections'] = cursor.fetchone()['count']
                
                # Count unique faces
                cursor.execute("SELECT COUNT(DISTINCT face_uuid) as count FROM face_detections")
                stats['unique_faces'] = cursor.fetchone()['count']
                
                # Count embeddings
                cursor.execute("SELECT COUNT(*) as count FROM face_embeddings")
                stats['stored_embeddings'] = cursor.fetchone()['count']
                
                conn.close()
                return stats
                
            except Exception as e:
                logger.error(f"Failed to get face statistics: {e}")
                return {}
    
    def cleanup(self):
        """Cleanup database resources."""
        # SQLite doesn't need special cleanup like connection pools
        logger.info("SQLite database cleanup completed")

###################################################################################################
# FAISS Database Manager
###################################################################################################

class FAISSManager:
    """Handles FAISS database operations for face embeddings."""
    
    def __init__(self):
        self.index = None
        self.face_metadata = {}
        self.lock = threading.Lock()
        self.embedding_dim = FACE_RECOGNITION_CONFIG['embedding']['embedding_dimension']
        self.similarity_threshold = FACE_RECOGNITION_CONFIG['embedding']['similarity_threshold']
        self.index_file = FACE_RECOGNITION_CONFIG['faiss']['index_file']
        self.metadata_file = FACE_RECOGNITION_CONFIG['faiss']['metadata_file']
        self.save_counter = 0
        self.init_index()
    
    def init_index(self):
        """Initialize FAISS index."""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                # Load existing index
                self.index = faiss.read_index(self.index_file)
                with open(self.metadata_file, 'r') as f:
                    self.face_metadata = json.load(f)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} faces")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for normalized embeddings
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
    
    def search_face(self, embedding: np.ndarray, k: int = 1) -> Tuple[Optional[str], float]:
        """Search for similar face in FAISS index."""
        with self.lock:
            if self.index.ntotal == 0:
                return None, 0.0
            
            try:
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                embedding = embedding.reshape(1, -1).astype(np.float32)
                
                # Search
                similarities, indices = self.index.search(embedding, k)
                
                if similarities[0][0] > self.similarity_threshold:
                    face_id = str(indices[0][0])
                    if face_id in self.face_metadata:
                        return self.face_metadata[face_id]['uuid'], similarities[0][0]
                
                return None, similarities[0][0] if len(similarities[0]) > 0 else 0.0
                
            except Exception as e:
                logger.error(f"Face search failed: {e}")
                return None, 0.0
    
    def add_face(self, face_uuid: str, embedding: np.ndarray) -> bool:
        """Add new face to FAISS index."""
        with self.lock:
            try:
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                embedding = embedding.reshape(1, -1).astype(np.float32)
                
                # Add to index
                face_id = str(self.index.ntotal)
                self.index.add(embedding)
                
                # Store metadata
                self.face_metadata[face_id] = {
                    'uuid': face_uuid,
                    'added_time': datetime.now().isoformat(),
                    'embedding_norm': float(np.linalg.norm(embedding))
                }
                
                self.save_counter += 1
                
                # Periodic save
                if self.save_counter % FACE_RECOGNITION_CONFIG['faiss']['save_interval'] == 0:
                    self.save_index()
                
                logger.info(f"Added face {face_uuid} to FAISS index")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add face to FAISS index: {e}")
                return False
    
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            faiss.write_index(self.index, self.index_file)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self.face_metadata, f, indent=2)
            
            logger.info(f"Saved FAISS index with {self.index.ntotal} faces")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

###################################################################################################
# MemryX Model Wrappers
###################################################################################################

class RetinaFaceMXA:
    """RetinaFace model wrapper for MemryX acceleration."""
    
    def __init__(self):
        self.model_config = MEMRYX_CONFIG['models']['retinaface']
        self.accl = mx.AsyncAccl(self.model_config['dfp_path'])
        self.input_size = self.model_config['input_size']
        self.confidence_threshold = self.model_config['confidence_threshold']
        self.nms_threshold = self.model_config['nms_threshold']
        logger.info("RetinaFace MXA model initialized")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for RetinaFace."""
        # Resize image
        img_resized = cv2.resize(image, self.input_size)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension and rearrange to NCHW if needed
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """Detect faces in image using RetinaFace."""
        try:
            # Preprocess
            input_tensor = self.preprocess(image)
            
            # Run inference
            outputs = self.accl.run(input_tensor)
            
            # Postprocess
            faces = self.postprocess(outputs, image.shape[:2])
            
            return faces
            
        except Exception as e:
            logger.error(f"RetinaFace detection failed: {e}")
            return []
    
    def postprocess(self, outputs: List[np.ndarray], original_size: Tuple[int, int]) -> List[Dict]:
        """Postprocess RetinaFace outputs."""
        faces = []
        
        try:
            # Extract bounding boxes, landmarks, and scores
            # Note: This is a simplified postprocessing - actual implementation depends on model output format
            boxes = outputs[0]  # Assuming first output is bounding boxes
            scores = outputs[1]  # Assuming second output is confidence scores
            
            # Filter by confidence threshold
            valid_indices = np.where(scores > self.confidence_threshold)[0]
            
            for idx in valid_indices:
                box = boxes[idx]
                score = scores[idx]
                
                # Scale bounding box to original image size
                x1, y1, x2, y2 = box
                x1 = int(x1 * original_size[1] / self.input_size[0])
                y1 = int(y1 * original_size[0] / self.input_size[1])
                x2 = int(x2 * original_size[1] / self.input_size[0])
                y2 = int(y2 * original_size[0] / self.input_size[1])
                
                face = {
                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # x, y, w, h
                    'confidence': float(score),
                    'landmarks': None  # Can be added if model provides landmarks
                }
                
                faces.append(face)
            
            # Apply NMS
            faces = self.apply_nms(faces)
            
        except Exception as e:
            logger.error(f"RetinaFace postprocessing failed: {e}")
        
        return faces
    
    def apply_nms(self, faces: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression."""
        if not faces:
            return faces
        
        # Convert to format expected by cv2.dnn.NMSBoxes
        boxes = [[f['bbox'][0], f['bbox'][1], f['bbox'][2], f['bbox'][3]] for f in faces]
        scores = [f['confidence'] for f in faces]
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.nms_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [faces[i] for i in indices]
        
        return []

class FaceNet512MXA:
    """FaceNet512 model wrapper for MemryX acceleration."""
    
    def __init__(self):
        self.model_config = MEMRYX_CONFIG['models']['facenet512']
        self.accl = mx.AsyncAccl(self.model_config['dfp_path'])
        self.input_size = self.model_config['input_size']
        self.embedding_size = self.model_config['embedding_size']
        logger.info("FaceNet512 MXA model initialized")
    
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for FaceNet512."""
        # Resize to model input size
        face_resized = cv2.resize(face_image, self.input_size)
        
        # Normalize to [-1, 1]
        face_normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
        
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding using FaceNet512."""
        try:
            # Preprocess
            input_tensor = self.preprocess(face_image)
            
            # Run inference
            outputs = self.accl.run(input_tensor)
            
            # Extract embedding
            embedding = outputs[0].flatten()
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"FaceNet512 embedding failed: {e}")
            return None

class AgeGenderMXA:
    """Age and Gender model wrapper for MemryX acceleration."""
    
    def __init__(self):
        self.model_config = MEMRYX_CONFIG['models']['age_gender']
        self.accl = mx.AsyncAccl(self.model_config['dfp_path'])
        self.input_size = self.model_config['input_size']
        self.gender_labels = FACE_RECOGNITION_CONFIG['demographics']['gender_labels']
        logger.info("Age/Gender MXA model initialized")
    
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for age/gender prediction."""
        # Resize to model input size
        face_resized = cv2.resize(face_image, self.input_size)
        
        # Normalize to [0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def predict_age_gender(self, face_image: np.ndarray) -> Tuple[int, str, float]:
        """Predict age and gender from face image."""
        try:
            # Preprocess
            input_tensor = self.preprocess(face_image)
            
            # Run inference
            outputs = self.accl.run(input_tensor)
            
            # Extract predictions
            age_output = outputs[0]  # Assuming first output is age
            gender_output = outputs[1]  # Assuming second output is gender
            
            # Process age (assuming regression output)
            age = int(np.argmax(age_output))
            
            # Process gender
            gender_probs = np.softmax(gender_output)
            gender_idx = np.argmax(gender_probs)
            gender = self.gender_labels[gender_idx]
            gender_confidence = float(gender_probs[gender_idx])
            
            return age, gender, gender_confidence
            
        except Exception as e:
            logger.error(f"Age/Gender prediction failed: {e}")
            return 0, "Unknown", 0.0

###################################################################################################
# Main DeepFace System
###################################################################################################

class DeepFaceSystem:
    """Main DeepFace system integrating all components."""
    
    def __init__(self):
        # Initialize components
        self.db_manager = DatabaseManager()
        self.faiss_manager = FAISSManager()
        
        # Initialize MXA models
        self.retinaface = RetinaFaceMXA()
        self.facenet512 = FaceNet512MXA()
        self.age_gender = AgeGenderMXA()
        
        # Demographics cache
        self.demographics_cache = {}
        self.cache_lock = threading.Lock()
        
        logger.info("DeepFace system initialized successfully")
    
    def process_frame(self, frame: np.ndarray, camera_id: int, camera_info: Dict) -> List[Dict]:
        """Process a single frame for face recognition."""
        results = []
        
        try:
            # Detect faces using RetinaFace
            faces = self.retinaface.detect_faces(frame)
            
            for face in faces:
                face_result = self.process_single_face(frame, face, camera_id, camera_info)
                if face_result:
                    results.append(face_result)
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
        
        return results
    
    def process_single_face(self, frame: np.ndarray, face: Dict, camera_id: int, camera_info: Dict) -> Optional[Dict]:
        """Process a single detected face."""
        try:
            # Extract face region
            x, y, w, h = face['bbox']
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return None
            
            # Get face embedding
            embedding = self.facenet512.get_embedding(face_roi)
            if embedding is None:
                return None
            
            # Search for existing face in FAISS
            existing_uuid, similarity = self.faiss_manager.search_face(embedding)
            
            if existing_uuid:
                # Face already exists
                face_uuid = existing_uuid
                is_new_face = False
            else:
                # New face
                face_uuid = str(uuid.uuid4())
                self.faiss_manager.add_face(face_uuid, embedding)
                self.db_manager.store_face_embedding(face_uuid, embedding, 'facenet512')
                is_new_face = True
            
            # Log face detection
            self.db_manager.log_face_detection(
                face_uuid, 
                camera_id, 
                camera_info.get('name', f'Camera_{camera_id}'), 
                camera_info.get('location', 'unknown'),
                face['bbox'],
                face['confidence']
            )
            
            # Get demographics if new face or cache expired
            age, gender, gender_confidence = self.get_demographics(face_uuid, face_roi, is_new_face)
            
            # Prepare result
            result = {
                'uuid': face_uuid,
                'bbox': face['bbox'],
                'confidence': face['confidence'],
                'age': age,
                'gender': gender,
                'gender_confidence': gender_confidence,
                'similarity': similarity,
                'is_new_face': is_new_face
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Single face processing failed: {e}")
            return None
    
    def get_demographics(self, face_uuid: str, face_roi: np.ndarray, is_new_face: bool) -> Tuple[int, str, float]:
        """Get age and gender for a face."""
        with self.cache_lock:
            # Check cache first
            if not is_new_face and face_uuid in self.demographics_cache:
                cache_entry = self.demographics_cache[face_uuid]
                if time.time() - cache_entry['timestamp'] < FACE_RECOGNITION_CONFIG['demographics']['cache_duration']:
                    return cache_entry['age'], cache_entry['gender'], cache_entry['gender_confidence']
            
            # Run inference for new faces or cache miss
            if is_new_face:
                age, gender, gender_confidence = self.age_gender.predict_age_gender(face_roi)
                
                # Cache the result
                self.demographics_cache[face_uuid] = {
                    'age': age,
                    'gender': gender,
                    'gender_confidence': gender_confidence,
                    'timestamp': time.time()
                }
                
                # Log to database
                self.db_manager.log_face_demographics(face_uuid, age, gender, gender_confidence)
                
                return age, gender, gender_confidence
            else:
                # Return cached or default values for existing faces
                if face_uuid in self.demographics_cache:
                    cache_entry = self.demographics_cache[face_uuid]
                    return cache_entry['age'], cache_entry['gender'], cache_entry['gender_confidence']
                else:
                    return 0, "Unknown", 0.0
    
    def draw_results(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """Draw face recognition results on frame."""
        for result in results:
            x, y, w, h = result['bbox']
            
            # Draw bounding box
            color = (0, 255, 0) if not result['is_new_face'] else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare text
            age = result['age']
            gender = result['gender']
            confidence = result['gender_confidence']
            
            if result['is_new_face']:
                text = f"NEW: {gender}, {age}"
            else:
                text = f"{gender}, {age} ({result['similarity']:.2f})"
            
            # Draw text background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(frame, (x, y - 25), (x + text_size[0] + 5, y), color, -1)
            
            # Draw text
            cv2.putText(frame, text, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw UUID (small text)
            uuid_text = result['uuid'][:8]
            cv2.putText(frame, uuid_text, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def cleanup(self):
        """Cleanup resources."""
        self.faiss_manager.save_index()
        logger.info("DeepFace system cleanup completed")

###################################################################################################

if __name__ == "__main__":
    # Test the system
    print("Testing DeepFace system...")
    
    try:
        system = DeepFaceSystem()
        print("DeepFace system initialized successfully")
        
        # Test with a dummy frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        camera_info = {'name': 'Test Camera', 'location': 'test_location'}
        
        results = system.process_frame(test_frame, 0, camera_info)
        print(f"Processed frame with {len(results)} faces detected")
        
        system.cleanup()
        print("Test completed successfully")
        
    except Exception as e:
        print(f"Test failed: {e}")

# eof 