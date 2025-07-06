#!/usr/bin/env python3
"""
============
Information:
============
Project: Multi-Camera Face Recognition System (CPU Mode)
File Name: app_cpu.py

============
Description:
============
CPU-only version of the face recognition system that doesn't require MemryX hardware.
Uses standard OpenCV and DeepFace models for face detection and recognition.
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
from deepface import DeepFace
import faiss

# Import configurations
from config import CAMERA_CONFIG, FACE_RECOGNITION_CONFIG, DATABASE_CONFIG, setup_environment

###################################################################################################

class FaceRecognitionCPU:
    """
    Multi-camera face recognition system using CPU processing only.
    No MemryX hardware required - runs on standard CPU.
    """

    def __init__(self, video_paths=None, show=True):
        """
        Initialize the CPU-based face recognition system.
        
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
        
        # Initialize face processing components
        self._initialize_face_models()
        
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
        
        # Processing threads
        self.processing_threads = []
        self.display_thread = None

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
                    if settings.get('enabled', True):
                        if 'resolution' in settings:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['resolution'][0])
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['resolution'][1])
                        if 'fps' in settings:
                            cap.set(cv2.CAP_PROP_FPS, settings['fps'])
                
                self.streams.append(cap)
                
                # Get camera info
                self.camera_info[i] = CAMERA_CONFIG['camera_settings'].get(i, {
                    'name': f'Camera_{i}',
                    'location': f'location_{i}',
                    'resolution': (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                    'fps': int(cap.get(cv2.CAP_PROP_FPS))
                })
                
                print(f"✅ Camera {i} initialized: {self.camera_info[i]['name']}")
                
            except Exception as e:
                print(f"❌ Error initializing camera {i}: {e}")
                raise

    def _initialize_face_models(self):
        """Initialize face detection and recognition models."""
        print("🔄 Initializing face recognition models...")
        
        # Face detection using RetinaFace (CPU)
        self.face_detector = 'retinaface'
        
        # Face recognition using FaceNet512 (CPU)
        self.face_model = 'Facenet512'
        
        # Test models by running a dummy detection
        try:
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            _ = DeepFace.extract_faces(dummy_img, detector_backend=self.face_detector, enforce_detection=False)
            print("✅ Face detection model loaded successfully")
        except Exception as e:
            print(f"⚠️  Face detection model warning: {e}")
            self.face_detector = 'opencv'  # Fallback to OpenCV
            print("🔄 Using OpenCV Haar Cascade as fallback")

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

    def _initialize_faiss(self):
        """Initialize FAISS index for face embeddings."""
        try:
            faiss_config = FACE_RECOGNITION_CONFIG['faiss']
            index_file = faiss_config['index_file']
            metadata_file = faiss_config['metadata_file']
            
            # Create directories
            os.makedirs(os.path.dirname(index_file), exist_ok=True)
            
            # Load or create FAISS index
            if os.path.exists(index_file):
                self.faiss_index = faiss.read_index(index_file)
                print(f"✅ Loaded FAISS index: {self.faiss_index.ntotal} faces")
            else:
                # Create new index
                embedding_dim = FACE_RECOGNITION_CONFIG['embedding']['embedding_dimension']
                self.faiss_index = faiss.IndexFlatIP(embedding_dim)
                print("✅ Created new FAISS index")
            
            # Load metadata
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.face_metadata = json.load(f)
                print(f"✅ Loaded face metadata: {len(self.face_metadata)} entries")
            else:
                self.face_metadata = {}
                print("✅ Created new face metadata")
                
        except Exception as e:
            print(f"❌ FAISS initialization error: {e}")
            # Create fallback
            embedding_dim = FACE_RECOGNITION_CONFIG['embedding']['embedding_dimension']
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)
            self.face_metadata = {}

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in a frame using fast CPU processing.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected faces with bounding boxes
        """
        try:
            # Use OpenCV Haar Cascade for fast face detection
            if not hasattr(self, 'face_cascade'):
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale for faster processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with optimized parameters for speed
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            detected_faces = []
            min_face_size = FACE_RECOGNITION_CONFIG['detection']['min_face_size']
            
            for (x, y, w, h) in faces:
                # Check minimum face size
                if w >= min_face_size and h >= min_face_size:
                    detected_faces.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 0.85,
                        'region': {'x': x, 'y': y, 'w': w, 'h': h}
                    })
            
            return detected_faces
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return []

    def get_face_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding using CPU processing.
        
        Args:
            face_img: Cropped face image
            
        Returns:
            Face embedding vector or None if failed
        """
        try:
            # Use DeepFace to extract embedding
            embedding = DeepFace.represent(
                face_img,
                model_name=self.face_model,
                enforce_detection=False
            )
            
            if embedding and len(embedding) > 0:
                # Convert to numpy array and normalize
                embedding_vector = np.array(embedding[0]['embedding'], dtype=np.float32)
                
                # Normalize embedding
                if FACE_RECOGNITION_CONFIG['embedding']['normalization']:
                    embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
                
                return embedding_vector
            
            return None
            
        except Exception as e:
            print(f"Face embedding error: {e}")
            return None

    def analyze_demographics(self, face_img: np.ndarray) -> Dict:
        """
        Analyze age and gender from face image using DeepFace.
        
        Args:
            face_img: Cropped face image
            
        Returns:
            Dictionary with age and gender information
        """
        try:
            # Use DeepFace to analyze demographics
            analysis = DeepFace.analyze(
                face_img,
                actions=['age', 'gender'],
                enforce_detection=False
            )
            
            if analysis and len(analysis) > 0:
                result = analysis[0]
                
                # Extract age
                age = int(result.get('age', 0))
                
                # Extract gender with confidence
                gender_data = result.get('gender', {})
                if isinstance(gender_data, dict):
                    # Find the gender with highest confidence
                    gender = max(gender_data.items(), key=lambda x: x[1])
                    gender_label = gender[0]
                    gender_confidence = gender[1] / 100.0  # Convert to 0-1 scale
                else:
                    gender_label = "Unknown"
                    gender_confidence = 0.0
                
                return {
                    'age': age,
                    'gender': gender_label,
                    'gender_confidence': gender_confidence
                }
            
            return {'age': 0, 'gender': 'Unknown', 'gender_confidence': 0.0}
            
        except Exception as e:
            print(f"Demographics analysis error: {e}")
            return {'age': 0, 'gender': 'Unknown', 'gender_confidence': 0.0}

    def search_face(self, embedding: np.ndarray) -> Optional[Dict]:
        """
        Search for similar face in FAISS index.
        
        Args:
            embedding: Face embedding vector
            
        Returns:
            Match information or None if no match
        """
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
            print(f"Face search error: {e}")
            return None

    def add_face_to_index(self, face_uuid: str, embedding: np.ndarray, demographics: Dict = None):
        """
        Add a new face to the FAISS index with proper UUID management.
        
        Args:
            face_uuid: Unique identifier for the face
            embedding: Face embedding vector
            demographics: Optional demographics data
        """
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

    def update_face_metadata(self, face_uuid: str):
        """Update metadata for an existing face."""
        if face_uuid in self.face_metadata:
            self.face_metadata[face_uuid]['last_seen'] = time.time()
            self.face_metadata[face_uuid]['detection_count'] += 1

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

    def process_stream(self, stream_idx: int):
        """
        Process a single camera stream with optimized performance.
        
        Args:
            stream_idx: Camera stream index
        """
        frame_count = 0
        recognition_interval = 5  # Only do face recognition every 5th frame
        
        while not self.done:
            try:
                # Capture frame
                ret, frame = self.streams[stream_idx].read()
                if not ret:
                    print(f"❌ Camera {stream_idx} disconnected")
                    break
                
                frame_count += 1
                
                # Always detect faces for bounding boxes (fast)
                detected_faces = self.detect_faces(frame)
                
                # Debug output every 30 frames
                if frame_count % 30 == 0 and detected_faces:
                    print(f"📸 Camera {stream_idx}: Detected {len(detected_faces)} faces")
                
                # Process each detected face
                face_results = []
                for face in detected_faces:
                    x1, y1, x2, y2 = face['bbox']
                    
                    # Always add face with bounding box
                    face_result = {
                        'bbox': face['bbox'],
                        'confidence': face['confidence'],
                        'embedding': None,
                        'match': None,
                        'timestamp': time.time()
                    }
                    
                    # Only do face recognition every few frames to save CPU
                    if frame_count % recognition_interval == 0:
                        # Extract face region
                        face_img = frame[y1:y2, x1:x2]
                        
                        if face_img.size > 0:
                            # Get face embedding
                            embedding = self.get_face_embedding(face_img)
                            
                            if embedding is not None:
                                # Search for match
                                match = self.search_face(embedding)
                                
                                face_result['embedding'] = embedding
                                face_result['match'] = match
                                
                                # Analyze demographics (age/gender)
                                demographics = self.analyze_demographics(face_img)
                                face_result['demographics'] = demographics
                                
                                if match:
                                    # Existing face - update metadata
                                    face_uuid = match['face_uuid']
                                    self.update_face_metadata(face_uuid)
                                    face_result['face_uuid'] = face_uuid
                                    print(f"👤 Recognized: {face_uuid[:8]}... (Similarity: {match['similarity']:.3f})")
                                else:
                                    # New face - add to index
                                    face_uuid = self.generate_face_uuid(stream_idx)
                                    self.add_face_to_index(face_uuid, embedding, demographics)
                                    self.log_demographics(face_uuid, demographics)
                                    face_result['face_uuid'] = face_uuid
                                    print(f"🆕 New face: {face_uuid[:8]}...")
                    
                    face_results.append(face_result)
                
                # Store results (thread-safe)
                with self.results_lock:
                    self.face_results[stream_idx] = {
                        'frame': frame.copy(),
                        'faces': face_results,
                        'timestamp': time.time()
                    }
                
                # Update FPS
                self._update_fps(stream_idx)
                
                # Minimal delay for smooth processing
                time.sleep(0.005)
                
            except Exception as e:
                print(f"❌ Error processing stream {stream_idx}: {e}")
                time.sleep(1)

    def _update_fps(self, stream_idx: int):
        """Update FPS tracking for a stream."""
        current_time = time.time()
        fps_info = self.fps_tracker[stream_idx]
        
        fps_info['frame_count'] += 1
        
        if current_time - fps_info['last_time'] >= 1.0:
            fps_info['fps'] = fps_info['frame_count']
            fps_info['frame_count'] = 0
            fps_info['last_time'] = current_time

    def _display_loop(self):
        """Display results from all camera streams."""
        while not self.done:
            try:
                frames_to_display = []
                
                with self.results_lock:
                    for stream_idx in range(self.num_streams):
                        if stream_idx in self.face_results:
                            result = self.face_results[stream_idx]
                            
                            if 'frame' in result:
                                frame = result['frame'].copy()
                                faces = result.get('faces', [])
                                
                                # Draw face detections
                                for face in faces:
                                    x1, y1, x2, y2 = face['bbox']
                                    confidence = face['confidence']
                                    match = face.get('match')
                                    demographics = face.get('demographics', {})
                                    
                                    # Draw bounding box
                                    color = (0, 255, 0) if match else (0, 255, 255)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                                    
                                    # Build label with demographics
                                    label_parts = []
                                    
                                    # Add age and gender if available
                                    if demographics:
                                        age = demographics.get('age', 0)
                                        gender = demographics.get('gender', 'Unknown')
                                        gender_conf = demographics.get('gender_confidence', 0.0)
                                        
                                        if age > 0:
                                            label_parts.append(f"{gender} {age}")
                                        else:
                                            label_parts.append(gender)
                                        
                                        if gender_conf > 0:
                                            label_parts.append(f"({gender_conf:.2f})")
                                    
                                    # Add recognition info
                                    face_uuid = face.get('face_uuid')
                                    if match and face_uuid:
                                        # Show UUID and similarity for recognized faces
                                        short_uuid = face_uuid[:8] if face_uuid else "Unknown"
                                        label_parts.append(f"ID: {short_uuid} ({match['similarity']:.2f})")
                                    elif face_uuid:
                                        # Show UUID for new faces
                                        short_uuid = face_uuid[:8]
                                        label_parts.append(f"New: {short_uuid}")
                                    else:
                                        label_parts.append("Detecting...")
                                    
                                    # Create final label
                                    label = " | ".join(label_parts) if label_parts else "Face"
                                    
                                    # Draw label background
                                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                    cv2.rectangle(frame, (x1, y1-25), (x1 + text_size[0] + 10, y1), color, -1)
                                    
                                    cv2.putText(frame, label, (x1 + 5, y1-8), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                
                                # Draw FPS
                                fps = self.fps_tracker[stream_idx]['fps']
                                cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                
                                # Draw camera info
                                camera_name = self.camera_info[stream_idx]['name']
                                cv2.putText(frame, camera_name, (10, 60), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                
                                frames_to_display.append((stream_idx, frame))
                
                # Display frames
                for stream_idx, frame in frames_to_display:
                    window_name = f"Camera {stream_idx}"
                    cv2.imshow(window_name, frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.done = True
                    break
                
                time.sleep(0.01)  # ~60 FPS display for smoother output
                
            except Exception as e:
                print(f"❌ Display error: {e}")
                time.sleep(1)

    def run(self):
        """Run the face recognition system."""
        print("🚀 Starting Multi-Camera Face Recognition System (CPU Mode)")
        print(f"📹 Processing {self.num_streams} camera streams")
        
        try:
            # Start processing threads for each camera
            for i in range(self.num_streams):
                thread = threading.Thread(target=self.process_stream, args=(i,))
                thread.daemon = True
                thread.start()
                self.processing_threads.append(thread)
                print(f"✅ Started processing thread for camera {i}")
            
            # Start display thread if enabled
            if self.show:
                self.display_thread = threading.Thread(target=self._display_loop)
                self.display_thread.daemon = True
                self.display_thread.start()
                print("✅ Display thread started")
                print("💡 Press 'q' in any window to quit")
            
            # Wait for completion
            try:
                while not self.done:
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n🛑 Received interrupt signal")
                self.done = True
        
        except Exception as e:
            print(f"❌ Runtime error: {e}")
            self.done = True
        
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up resources...")
        
        # Stop processing
        self.done = True
        
        # Save face data
        try:
            self.save_face_data()
            print("✅ Face data saved")
        except Exception as e:
            print(f"❌ Error saving face data: {e}")
        
        # Close camera streams
        for i, cap in enumerate(self.streams):
            try:
                cap.release()
                print(f"✅ Camera {i} released")
            except Exception as e:
                print(f"❌ Error releasing camera {i}: {e}")
        
        # Close database connection
        if hasattr(self, 'db_connection'):
            try:
                self.db_connection.close()
                print("✅ Database connection closed")
            except Exception as e:
                print(f"❌ Error closing database: {e}")
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        print("✅ Cleanup completed")

###################################################################################################

def test_face_detection(camera_index=0):
    """Test face detection only for debugging."""
    print(f"🧪 Testing face detection on camera {camera_index}")
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_index}")
        return
    
    # Initialize face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("✅ Camera and face detection initialized")
    print("💡 Press 'q' to quit")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Could not read frame")
            break
        
        frame_count += 1
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw bounding boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
            cv2.putText(frame, f"Face {w}x{h}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show frame count and face count
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow("Face Detection Test", frame)
        
        # Print face count every 30 frames
        if frame_count % 30 == 0:
            print(f"📸 Frame {frame_count}: Detected {len(faces)} faces")
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Face detection test completed")

def test_gender_detection(camera_index=0):
    """Test gender detection specifically for debugging."""
    print(f"🧪 Testing gender detection on camera {camera_index}")
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_index}")
        return
    
    # Initialize face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("✅ Camera and gender detection initialized")
    print("💡 Press 'q' to quit, 'a' to analyze current frame")
    
    frame_count = 0
    
    # Initialize face recognition system for demographics
    system = FaceRecognitionCPU(video_paths=[camera_index], show=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Could not read frame")
            break
        
        frame_count += 1
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Draw bounding boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
            cv2.putText(frame, f"Face {w}x{h}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show frame count and face count
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow("Gender Detection Test", frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('a') and len(faces) > 0:
            print("🔍 Analyzing current frame...")
            # Analyze first detected face
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size > 0:
                demographics = system.analyze_demographics(face_img)
                print(f"📊 Analysis Result:")
                print(f"   Age: {demographics['age']}")
                print(f"   Gender: {demographics['gender']}")
                print(f"   Gender Confidence: {demographics['gender_confidence']:.2f}")
                print("-" * 40)
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Gender detection test completed")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="CPU-Based Multi-Camera Face Recognition System"
    )
    
    parser.add_argument(
        '--video_paths',
        nargs='+',
        type=str,
        help='Camera indices or video file paths (e.g., 0 1 or video1.mp4 video2.mp4)'
    )
    
    parser.add_argument(
        '--no_display',
        action='store_true',
        help='Disable video display (useful for headless mode)'
    )
    
    parser.add_argument(
        '--test_detection',
        action='store_true',
        help='Test face detection only (debug mode)'
    )
    
    parser.add_argument(
        '--test_gender',
        action='store_true',
        help='Test gender detection specifically (debug mode)'
    )
    
    args = parser.parse_args()
    
    # Convert string camera indices to integers where applicable
    video_paths = []
    if args.video_paths:
        for path in args.video_paths:
            try:
                # Try to convert to integer (camera index)
                video_paths.append(int(path))
            except ValueError:
                # Keep as string (file path or URL)
                video_paths.append(path)
    else:
        video_paths = None
    
    # Test modes for debugging
    if args.test_detection:
        test_face_detection(video_paths[0] if video_paths else 0)
    elif args.test_gender:
        test_gender_detection(video_paths[0] if video_paths else 0)
    else:
        # Create and run face recognition system
        system = FaceRecognitionCPU(
            video_paths=video_paths,
            show=not args.no_display
        )
        
        system.run()

if __name__ == "__main__":
    main() 