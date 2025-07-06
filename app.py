"""
============
Information:
============
Project: Multi-Camera Face Recognition System with MemryX
File Name: app.py

============
Description:
============
A script to perform real-time face recognition on multiple camera streams using 
MemryX MultiStreamAsyncAccl API with DeepFace integration.
"""

###################################################################################################

# Imports
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
from memryx import MultiStreamAsyncAccl
from deepface import DeepFaceSystem, DeepFace
import faiss
from config import CAMERA_CONFIG, MEMRYX_CONFIG, FACE_RECOGNITION_CONFIG, DATABASE_CONFIG, get_camera_info, setup_environment

###################################################################################################

class FaceRecognitionMXA:
    """
    Multi-camera face recognition system using MemryX MultiStreamAsyncAccl
    for optimized hardware acceleration and DeepFace integration.
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
        
        # Initialize DeepFace system
        self.deepface_system = DeepFaceSystem()
        
        # Initialize database
        self._initialize_database()
        
        # Initialize FAISS index
        self._initialize_faiss()
        
        # MemryX MultiStreamAsyncAccl setup
        self.accl = None
        self._initialize_memryx_accelerator()
        
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
                
                print(f"Camera {i} initialized: {self.camera_info[i]['name']}")
                
            except Exception as e:
                print(f"Error initializing camera {i}: {e}")
                raise

    def _initialize_memryx_accelerator(self):
        """Initialize MemryX MultiStreamAsyncAccl for face detection."""
        try:
            # Get RetinaFace model path
            retinaface_dfp = MEMRYX_CONFIG['models']['retinaface']['dfp_path']
            
            if not retinaface_dfp or not os.path.exists(retinaface_dfp):
                raise FileNotFoundError(f"RetinaFace DFP not found: {retinaface_dfp}")
            
            # Initialize MultiStreamAsyncAccl
            self.accl = MultiStreamAsyncAccl(
                dfp=retinaface_dfp,
                group_id=0,
                stream_workers=self.num_streams
            )
            
            print(f"MemryX MultiStreamAsyncAccl initialized for {self.num_streams} streams")
            
        except Exception as e:
            print(f"Error initializing MemryX accelerator: {e}")
            raise

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

    def input_callback(self, stream_idx: int) -> Optional[np.ndarray]:
        """
        Input callback for MemryX MultiStreamAsyncAccl.
        
        Args:
            stream_idx: Camera stream index
            
        Returns:
            Preprocessed frame for face detection or None if done
        """
        if self.done:
            return None
            
        try:
            # Capture frame from camera
            ret, frame = self.streams[stream_idx].read()
            
            if not ret:
                print(f"Camera {stream_idx} disconnected")
                return None
            
            # Store original frame for display (in a thread-safe way)
            with self.results_lock:
                # Store frame for display thread
                setattr(self, f'_display_frame_{stream_idx}', frame.copy())
            
            # Preprocess frame for RetinaFace
            preprocessed = self.deepface_system.retinaface.preprocess(frame)
            
            # Update FPS tracking
            self._update_fps(stream_idx)
            
            return preprocessed
            
        except Exception as e:
            print(f"Error in input callback for stream {stream_idx}: {e}")
            return None

    def output_callback(self, stream_idx: int, *outputs):
        """
        Output callback for MemryX MultiStreamAsyncAccl.
        
        Args:
            stream_idx: Camera stream index
            outputs: Face detection outputs from RetinaFace
        """
        try:
            # Get original frame size
            original_frame = getattr(self, f'_display_frame_{stream_idx}', None)
            if original_frame is None:
                return
                
            original_size = (original_frame.shape[1], original_frame.shape[0])
            
            # Process face detection outputs
            detected_faces = self.deepface_system.retinaface.postprocess(
                list(outputs), original_size
            )
            
            # Process each detected face with enhanced UUID management
            face_results = []
            for face in detected_faces:
                try:
                    # Get face embedding using DeepFace system
                    face_result = self.deepface_system.process_single_face(
                        original_frame, face, stream_idx, self.camera_info[stream_idx]
                    )
                    
                    if face_result and 'embedding' in face_result:
                        embedding = face_result['embedding']
                        
                        # Search for similar faces in our FAISS index
                        match_result = self.search_face(embedding)
                        
                        face_uuid = None
                        demographics = None
                        similarity = 0.0
                        
                        if match_result:
                            # Existing face found
                            face_uuid = match_result['face_uuid']
                            similarity = match_result['similarity']
                            demographics = match_result['metadata'].get('demographics', {})
                            
                            # Update metadata
                            self.update_face_metadata(face_uuid)
                            
                            print(f"✅ Recognized face: {face_uuid[:12]}... (Similarity: {similarity:.3f})")
                            
                        else:
                            # New face detected
                            face_uuid = self.generate_face_uuid(stream_idx)
                            
                            # Extract face region for demographics analysis
                            if 'bbox' in face_result:
                                x1, y1, x2, y2 = face_result['bbox']
                                face_region = original_frame[y1:y2, x1:x2]
                                
                                # Analyze demographics for new face
                                demographics = self.analyze_demographics(face_region)
                                
                                # Add face to index
                                self.add_face_to_index(face_uuid, embedding, demographics)
                                
                                # Log demographics to database
                                self.log_demographics(face_uuid, demographics)
                                
                                print(f"🆕 New face detected: {face_uuid[:12]}... | {demographics['gender']} {demographics['age']}")
                        
                        # Update face result with new information
                        face_result['face_uuid'] = face_uuid
                        face_result['similarity'] = similarity
                        face_result['demographics'] = demographics or {}
                        face_result['is_new'] = match_result is None
                        
                        face_results.append(face_result)
                    
                    elif face_result:
                        # Face detected but no embedding (fallback)
                        face_results.append(face_result)
                
                except Exception as e:
                    print(f"Error processing individual face: {e}")
                    continue
            
            # Store results in thread-safe manner
            with self.results_lock:
                self.face_results[stream_idx] = face_results
                
        except Exception as e:
            print(f"Error in output callback for stream {stream_idx}: {e}")

    def _update_fps(self, stream_idx: int):
        """Update FPS tracking for a specific stream."""
        tracker = self.fps_tracker[stream_idx]
        tracker['frame_count'] += 1
        
        current_time = time.time()
        if current_time - tracker['last_time'] >= 1.0:  # Update every second
            tracker['fps'] = tracker['frame_count'] / (current_time - tracker['last_time'])
            tracker['frame_count'] = 0
            tracker['last_time'] = current_time

    def draw_enhanced_results(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        Draw enhanced face detection results with demographics information.
        
        Args:
            frame: Image frame to draw on
            results: List of face detection results
            
        Returns:
            Frame with drawn results
        """
        try:
            for result in results:
                # Get bounding box
                if 'bbox' in result:
                    x1, y1, x2, y2 = result['bbox']
                    
                    # Get face information
                    face_uuid = result.get('face_uuid', 'Unknown')
                    similarity = result.get('similarity', 0.0)
                    demographics = result.get('demographics', {})
                    is_new = result.get('is_new', False)
                    
                    # Choose color based on recognition status
                    if is_new:
                        color = (0, 255, 255)  # Yellow for new faces
                        status = "New"
                    else:
                        color = (0, 255, 0)  # Green for recognized faces
                        status = "ID"
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Prepare label text
                    age = demographics.get('age', 0)
                    gender = demographics.get('gender', 'Unknown')
                    
                    # Format labels
                    if age > 0 and gender != 'Unknown':
                        demo_text = f"{gender} {age}"
                    else:
                        demo_text = "Analyzing..."
                    
                    # UUID display (shortened)
                    uuid_short = face_uuid[:12] if len(face_uuid) > 12 else face_uuid
                    
                    if is_new:
                        main_text = f"New: {uuid_short}"
                    else:
                        main_text = f"{status}: {uuid_short} ({similarity:.3f})"
                    
                    # Draw demographics text
                    if demo_text != "Analyzing...":
                        demo_label_size = cv2.getTextSize(demo_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - 50), (x1 + demo_label_size[0] + 10, y1 - 25), color, -1)
                        cv2.putText(frame, demo_text, (x1 + 5, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw main identification text
                    main_label_size = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + main_label_size[0] + 10, y1), color, -1)
                    cv2.putText(frame, main_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Draw confidence indicator if available
                    if 'confidence' in result and result['confidence'] > 0:
                        conf_text = f"Conf: {result['confidence']:.2f}"
                        cv2.putText(frame, conf_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            return frame
            
        except Exception as e:
            print(f"Error drawing enhanced results: {e}")
            return frame

    def _display_loop(self):
        """Display thread for showing processed frames."""
        print("Display thread started")
        
        while not self.done:
            try:
                for stream_idx in range(self.num_streams):
                    # Get frame and results
                    frame = getattr(self, f'_display_frame_{stream_idx}', None)
                    if frame is None:
                        continue
                    
                    # Get face results
                    with self.results_lock:
                        results = self.face_results[stream_idx].copy()
                    
                    # Draw face detection results with enhanced display
                    if results:
                        frame = self.draw_enhanced_results(frame, results)
                    
                    # Add FPS and camera info
                    fps = self.fps_tracker[stream_idx]['fps']
                    camera_name = self.camera_info[stream_idx]['name']
                    
                    fps_text = f"{camera_name} - {fps:.1f} FPS"
                    cv2.putText(frame, fps_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display frame
                    if self.show:
                        window_name = f"{camera_name} - Face Recognition"
                        cv2.imshow(window_name, frame)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exit key pressed")
                    self.done = True
                    break
                    
            except Exception as e:
                print(f"Error in display loop: {e}")
                time.sleep(0.1)
        
        print("Display thread stopped")

    def run(self):
        """
        Start the multi-camera face recognition system.
        """
        print("Starting Multi-Camera Face Recognition System with MemryX")
        print(f"Processing {self.num_streams} camera streams")
        
        try:
            # Start display thread
            if self.show:
                self.display_thread.start()
            
            # Connect input and output callbacks to MemryX accelerator
            self.accl.connect_streams(
                input_callback=self.input_callback,
                output_callback=self.output_callback,
                stream_count=self.num_streams,
                model_idx=0
            )
            
            print("MemryX MultiStreamAsyncAccl started successfully")
            print("Press 'q' to quit")
            
            # Wait for processing to complete
            self.accl.wait()
            
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            self.cleanup()

    def test_face_detection(self):
        """Test face detection functionality with single camera."""
        print("Testing face detection functionality...")
        
        if not self.streams:
            print("No camera available for testing")
            return
        
        test_stream = self.streams[0]
        
        while True:
            ret, frame = test_stream.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Display original frame
            cv2.putText(frame, "Face Detection Test - Press 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Face Detection Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()

    def test_demographics_analysis(self):
        """Test demographics analysis with manual trigger."""
        print("Testing demographics analysis...")
        print("Press 'a' to analyze demographics, 'q' to quit")
        
        if not self.streams:
            print("No camera available for testing")
            return
        
        test_stream = self.streams[0]
        
        while True:
            ret, frame = test_stream.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Display instructions
            cv2.putText(frame, "Demographics Test - Press 'a' to analyze, 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Status: Waiting for input...", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Demographics Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                # Analyze demographics
                print("Analyzing demographics...")
                demographics = self.analyze_demographics(frame)
                print(f"Demographics: {demographics}")
                
                # Display results on frame
                result_text = f"Age: {demographics['age']}, Gender: {demographics['gender']}"
                cv2.putText(frame, result_text, 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Demographics Test", frame)
                cv2.waitKey(2000)  # Show results for 2 seconds
                
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()

    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up resources...")
        
        # Signal shutdown
        self.done = True
        
        # Stop MemryX accelerator
        if self.accl:
            try:
                self.accl.stop()
                print("MemryX accelerator stopped")
            except Exception as e:
                print(f"Error stopping accelerator: {e}")
        
        # Wait for display thread
        if self.display_thread.is_alive():
            self.display_thread.join(timeout=2)
        
        # Release cameras
        for i, stream in enumerate(self.streams):
            try:
                stream.release()
                print(f"Camera {i} released")
            except Exception as e:
                print(f"Error releasing camera {i}: {e}")
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        # Cleanup DeepFace system
        if hasattr(self, 'deepface_system'):
            self.deepface_system.cleanup()
        
        # Close database connection
        if hasattr(self, 'db_connection'):
            try:
                self.db_connection.close()
                print("Database connection closed")
            except Exception as e:
                print(f"Error closing database: {e}")
        
        print("Cleanup completed")

###################################################################################################

def main(args):
    """
    Main function to start the face recognition system.
    """
    try:
        # Initialize the system
        face_recognition_system = FaceRecognitionMXA(
            video_paths=args.video_paths, 
            show=args.show
        )
        
        # Run different modes based on arguments
        if args.test_detection:
            face_recognition_system.test_face_detection()
        elif args.test_demographics:
            face_recognition_system.test_demographics_analysis()
        else:
            face_recognition_system.run()
        
    except Exception as e:
        print(f"System error: {e}")
        return 1
    
    return 0

###################################################################################################

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="\033[34mMemryX Multi-Camera Face Recognition System\033[0m"
    )
    
    # Video input paths
    parser.add_argument(
        '--video_paths', 
        nargs='+', 
        dest="video_paths", 
        default=None,
        help="Paths to video sources. Use '/dev/video0' for webcam. "
             "If not specified, uses config.py settings."
    )
    
    # Display option
    parser.add_argument(
        '--no_display', 
        dest="show", 
        action="store_false", 
        default=True,
        help="Turn off video display"
    )
    
    # Configuration file option
    parser.add_argument(
        '-c', '--config', 
        type=str, 
        default=None,
        help="Path to custom configuration file (optional)"
    )
    
    # Test modes
    parser.add_argument(
        '--test_detection',
        action='store_true',
        help='Test face detection functionality only'
    )
    
    parser.add_argument(
        '--test_demographics',
        action='store_true',
        help='Test demographics analysis functionality'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function
    exit_code = main(args)
    exit(exit_code)

# EOF 