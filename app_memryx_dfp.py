"""
============
Information:
============
Project: Multi-Camera Face Recognition System with MemryX DFP Models
File Name: app_memryx_dfp.py

============
Description:
============
A script to perform real-time face recognition using three separate MemryX DFP models:
- retinaface_memryx.dfp (face detection)
- facenet512_memryx_raw_embeddings.dfp (face recognition) 
- genderage.dfp (age/gender classification)
"""

import os
import time
import argparse
import numpy as np
import cv2
import threading
import sqlite3
import json
import uuid
from typing import Dict, List, Optional, Tuple
import memryx as mx
import faiss
from config import CAMERA_CONFIG, MEMRYX_CONFIG, FACE_RECOGNITION_CONFIG, DATABASE_CONFIG, get_camera_info, setup_environment

###################################################################################################

class FaceRecognitionMemryXDFP:
    """
    Multi-camera face recognition system using separate MemryX DFP models.
    """

    def __init__(self, video_paths=None, show=True):
        """
        Initialize the multi-camera face recognition system.
        
        Args:
            video_paths: List of camera sources or video paths
            show: Whether to display video output
        """
        # Setup environment
        setup_environment()
        
        # Use configured camera sources if no video paths provided
        if video_paths is None:
            video_paths = CAMERA_CONFIG['sources']
        
        # System control
        self.done = False
        self.show = show and CAMERA_CONFIG['display']['show_video']
        
        # Camera setup
        self.video_paths = video_paths
        self.num_streams = len(video_paths)
        self.streams = []
        self.camera_info = {}
        
        # Initialize video captures
        self._initialize_cameras()
        
        # Initialize MemryX accelerators for all models
        self._initialize_memryx_models()
        
        # Initialize database
        self._initialize_database()
        
        # Initialize FAISS index
        self._initialize_faiss()
        
        # Results storage (thread-safe)
        self.face_results = {i: [] for i in range(self.num_streams)}
        self.results_lock = threading.Lock()
        
        # FPS tracking
        self.fps_tracker = {i: {'last_time': time.time(), 'frame_count': 0, 'fps': 0} 
                           for i in range(self.num_streams)}
        
        # Display thread
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True

    def _initialize_cameras(self):
        """Initialize camera captures and settings."""
        for i, video_path in enumerate(self.video_paths):
            try:
                # Create video capture
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Failed to open camera: {video_path}")
                
                # Set camera properties if specified
                if i in CAMERA_CONFIG['camera_settings']:
                    settings = CAMERA_CONFIG['camera_settings'][i]
                    if 'resolution' in settings:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['resolution'][0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['resolution'][1])
                    if 'fps' in settings:
                        cap.set(cv2.CAP_PROP_FPS, settings['fps'])
                
                self.streams.append(cap)
                
                # Get camera info
                self.camera_info[i] = get_camera_info(i) or {
                    'name': f'Camera_{i}',
                    'location': f'location_{i}',
                    'resolution': (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                    'fps': int(cap.get(cv2.CAP_PROP_FPS))
                }
                
                print(f"✅ Camera {i} initialized: {self.camera_info[i]['name']}")
                
            except Exception as e:
                print(f"❌ Error initializing camera {i}: {e}")
                raise

    def _initialize_memryx_models(self):
        """Initialize all three MemryX DFP models."""
        try:
            # Initialize MemryX accelerator
            self.accl = mx.Accelerator()
            
            # Load all three DFP models
            self.dfp_models = {}
            
            # 1. RetinaFace (Face Detection)
            retinaface_path = MEMRYX_CONFIG['models']['retinaface']['dfp_path']
            if not os.path.exists(retinaface_path):
                raise FileNotFoundError(f"RetinaFace DFP not found: {retinaface_path}")
            
            self.dfp_models['retinaface'] = mx.DFP(retinaface_path)
            self.accl.connect_dfp(self.dfp_models['retinaface'])
            
            # 2. FaceNet512 (Face Recognition)
            facenet_path = MEMRYX_CONFIG['models']['facenet512']['dfp_path']
            if not os.path.exists(facenet_path):
                raise FileNotFoundError(f"FaceNet512 DFP not found: {facenet_path}")
            
            self.dfp_models['facenet512'] = mx.DFP(facenet_path)
            self.accl.connect_dfp(self.dfp_models['facenet512'])
            
            # 3. Age/Gender (Demographics)
            age_gender_path = MEMRYX_CONFIG['models']['age_gender']['dfp_path']
            if not os.path.exists(age_gender_path):
                raise FileNotFoundError(f"Age/Gender DFP not found: {age_gender_path}")
            
            self.dfp_models['age_gender'] = mx.DFP(age_gender_path)
            self.accl.connect_dfp(self.dfp_models['age_gender'])
            
            # Setup input/output streams for each model
            self.setup_model_streams()
            
            print(f"✅ All MemryX DFP models loaded successfully:")
            print(f"   📋 RetinaFace: {retinaface_path}")
            print(f"   📋 FaceNet512: {facenet_path}")
            print(f"   📋 Age/Gender: {age_gender_path}")
            
        except Exception as e:
            print(f"❌ Error initializing MemryX models: {e}")
            raise

    def setup_model_streams(self):
        """Setup input/output streams for each model."""
        # RetinaFace streams (assuming model index 0)
        self.retinaface_input = self.accl.get_input_stream(0)
        self.retinaface_output = self.accl.get_output_stream(0)
        
        # FaceNet512 streams (assuming model index 1)
        self.facenet_input = self.accl.get_input_stream(1)
        self.facenet_output = self.accl.get_output_stream(1)
        
        # Age/Gender streams (assuming model index 2)
        self.age_gender_input = self.accl.get_input_stream(2)
        self.age_gender_output = self.accl.get_output_stream(2)

    def _initialize_database(self):
        """Initialize SQLite database connection."""
        try:
            db_path = DATABASE_CONFIG['sqlite']['database_path']
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            self.db_connection = sqlite3.connect(
                db_path, 
                timeout=DATABASE_CONFIG['sqlite']['timeout'],
                check_same_thread=DATABASE_CONFIG['sqlite']['check_same_thread']
            )
            
            # Create demographics table if it doesn't exist
            self._create_demographics_table()
            
            print("✅ Database connection established")
            
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            raise

    def _create_demographics_table(self):
        """Create demographics table if it doesn't exist."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_demographics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_uuid TEXT NOT NULL,
                    age INTEGER,
                    gender TEXT,
                    gender_confidence REAL,
                    analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.db_connection.commit()
        except Exception as e:
            print(f"❌ Error creating demographics table: {e}")

    def _initialize_faiss(self):
        """Initialize FAISS index for face embeddings."""
        try:
            faiss_config = FACE_RECOGNITION_CONFIG['faiss']
            embedding_dim = FACE_RECOGNITION_CONFIG['embedding']['embedding_dimension']
            
            # Create or load FAISS index
            if os.path.exists(faiss_config['index_file']):
                self.faiss_index = faiss.read_index(faiss_config['index_file'])
                print(f"✅ Loaded FAISS index with {self.faiss_index.ntotal} faces")
            else:
                # Create new index
                self.faiss_index = faiss.IndexFlatIP(embedding_dim)
                print("✅ Created new FAISS index")
            
            # Load metadata
            if os.path.exists(faiss_config['metadata_file']):
                with open(faiss_config['metadata_file'], 'r') as f:
                    self.face_metadata = json.load(f)
                print(f"✅ Loaded metadata for {len(self.face_metadata)} faces")
            else:
                self.face_metadata = {}
                print("✅ Initialized empty face metadata")
                
        except Exception as e:
            print(f"❌ FAISS initialization error: {e}")
            raise

    def preprocess_for_detection(self, image):
        """Preprocess image for RetinaFace detection."""
        target_size = MEMRYX_CONFIG['models']['retinaface']['input_size']
        
        # Resize image
        resized = cv2.resize(image, target_size)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Add batch dimension
        batch_image = np.expand_dims(normalized, axis=0)
        
        return batch_image

    def preprocess_for_recognition(self, face_image):
        """Preprocess face image for FaceNet512 recognition."""
        target_size = MEMRYX_CONFIG['models']['facenet512']['input_size']
        
        # Resize face image
        resized = cv2.resize(face_image, target_size)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Add batch dimension
        batch_image = np.expand_dims(normalized, axis=0)
        
        return batch_image

    def preprocess_for_age_gender(self, face_image):
        """Preprocess face image for Age/Gender classification."""
        target_size = MEMRYX_CONFIG['models']['age_gender']['input_size']
        
        # Resize face image
        resized = cv2.resize(face_image, target_size)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Add batch dimension
        batch_image = np.expand_dims(normalized, axis=0)
        
        return batch_image

    def detect_faces(self, image):
        """Detect faces using RetinaFace model."""
        try:
            # Preprocess image
            input_data = self.preprocess_for_detection(image)
            
            # Run inference
            self.retinaface_input.send_data(input_data)
            outputs = self.retinaface_output.get_data()
            
            # Process outputs to extract face bounding boxes
            faces = self.process_detection_outputs(outputs, image.shape)
            
            return faces
            
        except Exception as e:
            print(f"❌ Face detection error: {e}")
            return []

    def get_face_embedding(self, face_image):
        """Get face embedding using FaceNet512 model."""
        try:
            # Preprocess face image
            input_data = self.preprocess_for_recognition(face_image)
            
            # Run inference
            self.facenet_input.send_data(input_data)
            embedding = self.facenet_output.get_data()
            
            # Normalize embedding
            if FACE_RECOGNITION_CONFIG['embedding']['normalization']:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"❌ Face embedding error: {e}")
            return None

    def analyze_demographics(self, face_image):
        """Analyze age and gender using Age/Gender model."""
        try:
            # Preprocess face image
            input_data = self.preprocess_for_age_gender(face_image)
            
            # Run inference
            self.age_gender_input.send_data(input_data)
            outputs = self.age_gender_output.get_data()
            
            # Process outputs
            age, gender, gender_confidence = self.process_age_gender_outputs(outputs)
            
            return {
                'age': age,
                'gender': gender,
                'gender_confidence': gender_confidence
            }
            
        except Exception as e:
            print(f"❌ Demographics analysis error: {e}")
            return {'age': 0, 'gender': 'Unknown', 'gender_confidence': 0.0}

    def process_detection_outputs(self, outputs, image_shape):
        """Process RetinaFace outputs to extract face bounding boxes."""
        # This is a placeholder - implement based on your RetinaFace model outputs
        # Typically RetinaFace outputs include boxes, scores, and landmarks
        
        faces = []
        try:
            # Extract bounding boxes and scores from outputs
            # This will depend on your specific RetinaFace model output format
            
            # Example processing (adjust based on actual output format):
            # boxes = outputs[0]  # Bounding boxes
            # scores = outputs[1]  # Confidence scores
            # landmarks = outputs[2]  # Facial landmarks (optional)
            
            # Filter by confidence threshold
            conf_threshold = MEMRYX_CONFIG['models']['retinaface']['confidence_threshold']
            
            # Process each detection
            # for i, score in enumerate(scores):
            #     if score >= conf_threshold:
            #         box = boxes[i]
            #         # Convert from normalized coordinates to image coordinates
            #         x1, y1, x2, y2 = box
            #         x1 = int(x1 * image_shape[1])
            #         y1 = int(y1 * image_shape[0])
            #         x2 = int(x2 * image_shape[1])
            #         y2 = int(y2 * image_shape[0])
            #         
            #         faces.append((x1, y1, x2, y2, score))
            
            # For now, return empty list - implement based on your model outputs
            
        except Exception as e:
            print(f"❌ Error processing detection outputs: {e}")
        
        return faces

    def process_age_gender_outputs(self, outputs):
        """Process Age/Gender model outputs."""
        try:
            # This is a placeholder - implement based on your Age/Gender model outputs
            # Typically outputs include age predictions and gender probabilities
            
            # Example processing (adjust based on actual output format):
            # age_output = outputs[0]  # Age predictions
            # gender_output = outputs[1]  # Gender probabilities
            
            # Process age (assuming it's a classification with 101 classes for ages 0-100)
            # age = np.argmax(age_output)
            
            # Process gender (assuming 2 classes: Female=0, Male=1)
            # gender_probs = gender_output[0]  # Remove batch dimension
            # gender_idx = np.argmax(gender_probs)
            # gender_confidence = gender_probs[gender_idx]
            # gender_labels = FACE_RECOGNITION_CONFIG['demographics']['gender_labels']
            # gender = gender_labels[gender_idx]
            
            # For now, return placeholder values
            age = 25
            gender = "Unknown"
            gender_confidence = 0.5
            
            return age, gender, gender_confidence
            
        except Exception as e:
            print(f"❌ Error processing age/gender outputs: {e}")
            return 0, "Unknown", 0.0

    def search_face(self, embedding: np.ndarray) -> Optional[Dict]:
        """Search for similar face in FAISS index."""
        if self.faiss_index.ntotal == 0:
            return None
        
        try:
            # Search in FAISS index
            similarities, indices = self.faiss_index.search(
                embedding.reshape(1, -1), k=1
            )
            
            similarity = similarities[0][0]
            threshold = FACE_RECOGNITION_CONFIG['embedding']['similarity_threshold']
            
            if similarity >= threshold:
                index_id = indices[0][0]
                # Get the actual UUID from metadata using index ID
                face_uuid = None
                for uuid, metadata in self.face_metadata.items():
                    if metadata.get('index_id') == index_id:
                        face_uuid = uuid
                        break
                
                if face_uuid:
                    return {
                        'face_uuid': face_uuid,
                        'similarity': float(similarity),
                        'metadata': self.face_metadata.get(face_uuid, {})
                    }
            
            return None
            
        except Exception as e:
            print(f"❌ Face search error: {e}")
            return None

    def add_face_to_index(self, face_uuid: str, embedding: np.ndarray, demographics: Dict = None):
        """Add a new face to the FAISS index."""
        try:
            # Add embedding to FAISS index
            self.faiss_index.add(embedding.reshape(1, -1))
            
            # Get the index ID (last added item)
            index_id = self.faiss_index.ntotal - 1
            
            # Store metadata with UUID mapping
            self.face_metadata[face_uuid] = {
                'index_id': index_id,
                'first_seen': time.time(),
                'last_seen': time.time(),
                'demographics': demographics or {},
                'detection_count': 1
            }
            
            # Save the updated index and metadata
            self.save_face_data()
            
            print(f"✅ Added new face: {face_uuid[:8]}... (Index: {index_id})")
            
        except Exception as e:
            print(f"❌ Error adding face to index: {e}")

    def save_face_data(self):
        """Save FAISS index and metadata to disk."""
        try:
            faiss_config = FACE_RECOGNITION_CONFIG['faiss']
            
            # Save FAISS index
            faiss.write_index(self.faiss_index, faiss_config['index_file'])
            
            # Save metadata
            with open(faiss_config['metadata_file'], 'w') as f:
                json.dump(self.face_metadata, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error saving face data: {e}")

    def generate_face_uuid(self, stream_idx: int) -> str:
        """Generate a unique UUID for a new face."""
        return f"face_{stream_idx}_{uuid.uuid4().hex[:8]}_{int(time.time())}"

    def process_frame(self, frame, stream_idx):
        """Process a single frame through all models."""
        try:
            # Step 1: Detect faces
            faces = self.detect_faces(frame)
            
            results = []
            for face_bbox in faces:
                x1, y1, x2, y2 = face_bbox[:4]
                confidence = face_bbox[4] if len(face_bbox) > 4 else 0.0
                
                # Extract face region
                face_image = frame[y1:y2, x1:x2]
                
                if face_image.size == 0:
                    continue
                
                # Step 2: Get face embedding
                embedding = self.get_face_embedding(face_image)
                
                if embedding is None:
                    continue
                
                # Step 3: Search for existing face
                match = self.search_face(embedding)
                
                # Step 4: Analyze demographics
                demographics = self.analyze_demographics(face_image)
                
                if match:
                    # Update existing face
                    face_uuid = match['face_uuid']
                    self.update_face_metadata(face_uuid)
                    
                    # Log demographics
                    self.log_demographics(face_uuid, demographics)
                    
                    results.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'face_uuid': face_uuid,
                        'similarity': match['similarity'],
                        'demographics': demographics,
                        'status': 'recognized'
                    })
                else:
                    # Add new face
                    face_uuid = self.generate_face_uuid(stream_idx)
                    self.add_face_to_index(face_uuid, embedding, demographics)
                    
                    # Log demographics
                    self.log_demographics(face_uuid, demographics)
                    
                    results.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'face_uuid': face_uuid,
                        'similarity': 0.0,
                        'demographics': demographics,
                        'status': 'new'
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return []

    def update_face_metadata(self, face_uuid: str):
        """Update metadata for an existing face."""
        if face_uuid in self.face_metadata:
            self.face_metadata[face_uuid]['last_seen'] = time.time()
            self.face_metadata[face_uuid]['detection_count'] += 1

    def log_demographics(self, face_uuid: str, demographics: Dict):
        """Log demographics information to database."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO face_demographics (face_uuid, age, gender, gender_confidence)
                VALUES (?, ?, ?, ?)
            """, (
                face_uuid,
                demographics.get('age', 0),
                demographics.get('gender', 'Unknown'),
                demographics.get('gender_confidence', 0.0)
            ))
            self.db_connection.commit()
        except Exception as e:
            print(f"❌ Error logging demographics: {e}")

    def draw_results(self, frame, results):
        """Draw detection results on frame."""
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            confidence = result['confidence']
            demographics = result['demographics']
            status = result['status']
            
            # Choose color based on status
            color = (0, 255, 0) if status == 'recognized' else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw information
            age = demographics.get('age', 0)
            gender = demographics.get('gender', 'Unknown')
            
            label = f"{gender}, {age}"
            if status == 'recognized':
                label += f" (Sim: {result['similarity']:.2f})"
            
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

    def run(self):
        """Run the face recognition system."""
        try:
            print("🚀 Starting MemryX Face Recognition System...")
            
            # Start display thread if showing video
            if self.show:
                self.display_thread.start()
            
            # Main processing loop
            while not self.done:
                for stream_idx in range(self.num_streams):
                    # Capture frame
                    ret, frame = self.streams[stream_idx].read()
                    
                    if not ret:
                        print(f"❌ Camera {stream_idx} disconnected")
                        continue
                    
                    # Process frame
                    results = self.process_frame(frame, stream_idx)
                    
                    # Store results for display
                    with self.results_lock:
                        self.face_results[stream_idx] = results
                        setattr(self, f'_display_frame_{stream_idx}', frame.copy())
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping system...")
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
        finally:
            self.cleanup()

    def _display_loop(self):
        """Display loop for showing video output."""
        while not self.done:
            try:
                for stream_idx in range(self.num_streams):
                    # Get frame and results
                    frame = getattr(self, f'_display_frame_{stream_idx}', None)
                    
                    if frame is not None:
                        with self.results_lock:
                            results = self.face_results[stream_idx].copy()
                        
                        # Draw results
                        display_frame = self.draw_results(frame.copy(), results)
                        
                        # Add FPS info
                        fps = self.fps_tracker[stream_idx]['fps']
                        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Show frame
                        cv2.imshow(f'Camera {stream_idx}', display_frame)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.done = True
                    break
                    
            except Exception as e:
                print(f"❌ Display error: {e}")
                break
        
        cv2.destroyAllWindows()

    def cleanup(self):
        """Clean up resources."""
        try:
            self.done = True
            
            # Release camera resources
            for stream in self.streams:
                stream.release()
            
            # Close database connection
            if hasattr(self, 'db_connection'):
                self.db_connection.close()
            
            # Save final face data
            self.save_face_data()
            
            cv2.destroyAllWindows()
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"❌ Cleanup error: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='MemryX Face Recognition System')
    parser.add_argument('--video_paths', nargs='+', default=[0], 
                       help='Camera sources or video paths')
    parser.add_argument('--no_display', action='store_true',
                       help='Disable video display')
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = FaceRecognitionMemryXDFP(
            video_paths=args.video_paths,
            show=not args.no_display
        )
        
        # Run system
        system.run()
        
    except Exception as e:
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main() 