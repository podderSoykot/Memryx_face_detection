#!/usr/bin/env python3
"""
Face Recognition App with Single DFP
====================================

Uses the single multi-model DFP file containing all face recognition models.
"""

import cv2
import numpy as np
import memryx as mx
from pathlib import Path
import argparse
import time

class SingleDFPFaceRecognition:
    def __init__(self, dfp_path="dfp_models/face_recognition_complete.dfp"):
        """Initialize with single DFP file."""
        self.dfp_path = Path(dfp_path)
        
        # Check if DFP file exists
        if not self.dfp_path.exists():
            raise FileNotFoundError(f"DFP file not found: {self.dfp_path}")
        
        # Initialize MemryX accelerator
        self.accl = mx.Accelerator()
        
        # Load the single DFP containing all models
        print(f"📦 Loading single DFP: {self.dfp_path}")
        self.dfp = mx.DFP(str(self.dfp_path))
        
        # Connect all models to accelerator
        self.accl.connect_dfp(self.dfp)
        
        # Model indices in the DFP
        self.RETINAFACE_MODEL = 0  # Face detection
        self.ARCFACE_MODEL = 1     # Face recognition
        self.AGE_GENDER_MODEL = 2  # Age and gender
        
        # Input/output stream maps for each model
        self.setup_model_streams()
        
        print("✅ Single DFP loaded successfully!")
        print(f"   📋 Model 0: RetinaFace (Face Detection)")
        print(f"   📋 Model 1: ArcFace (Face Recognition)")  
        print(f"   📋 Model 2: Age/Gender (Demographics)")
    
    def setup_model_streams(self):
        """Setup input/output streams for each model."""
        
        # RetinaFace (Model 0) - Face Detection
        self.retinaface_input = self.accl.get_input_stream(self.RETINAFACE_MODEL)
        self.retinaface_output = self.accl.get_output_stream(self.RETINAFACE_MODEL)
        
        # ArcFace (Model 1) - Face Recognition
        self.arcface_input = self.accl.get_input_stream(self.ARCFACE_MODEL)
        self.arcface_output = self.accl.get_output_stream(self.ARCFACE_MODEL)
        
        # Age/Gender (Model 2) - Demographics
        self.age_gender_input = self.accl.get_input_stream(self.AGE_GENDER_MODEL)
        self.age_gender_output = self.accl.get_output_stream(self.AGE_GENDER_MODEL)
    
    def preprocess_for_detection(self, image):
        """Preprocess image for RetinaFace detection (320x320)."""
        # Resize to 320x320
        resized = cv2.resize(image, (320, 320))
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        # Add batch dimension
        batch_image = np.expand_dims(normalized, axis=0)
        return batch_image
    
    def preprocess_for_recognition(self, face_image):
        """Preprocess face image for ArcFace recognition (112x112)."""
        # Resize to 112x112
        resized = cv2.resize(face_image, (112, 112))
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        # Add batch dimension
        batch_image = np.expand_dims(normalized, axis=0)
        return batch_image
    
    def preprocess_for_age_gender(self, face_image):
        """Preprocess face image for Age/Gender classification (224x224)."""
        # Resize to 224x224
        resized = cv2.resize(face_image, (224, 224))
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        # Add batch dimension
        batch_image = np.expand_dims(normalized, axis=0)
        return batch_image
    
    def detect_faces(self, image):
        """Detect faces using RetinaFace (Model 0)."""
        # Preprocess
        input_data = self.preprocess_for_detection(image)
        
        # Run inference on Model 0 (RetinaFace)
        self.retinaface_input.send_data(input_data)
        outputs = self.retinaface_output.get_data()
        
        # Process outputs to get bounding boxes
        # (Implementation depends on RetinaFace output format)
        faces = self.process_detection_outputs(outputs, image.shape)
        
        return faces
    
    def get_face_embedding(self, face_image):
        """Get face embedding using ArcFace (Model 1)."""
        # Preprocess
        input_data = self.preprocess_for_recognition(face_image)
        
        # Run inference on Model 1 (ArcFace)
        self.arcface_input.send_data(input_data)
        embedding = self.arcface_output.get_data()
        
        return embedding
    
    def predict_age_gender(self, face_image):
        """Predict age and gender using Age/Gender model (Model 2)."""
        # Preprocess
        input_data = self.preprocess_for_age_gender(face_image)
        
        # Run inference on Model 2 (Age/Gender)
        self.age_gender_input.send_data(input_data)
        outputs = self.age_gender_output.get_data()
        
        # Process outputs
        age, gender = self.process_age_gender_outputs(outputs)
        
        return age, gender
    
    def process_detection_outputs(self, outputs, image_shape):
        """Process RetinaFace outputs to extract face bounding boxes."""
        # Placeholder - implement based on RetinaFace output format
        # Typically returns: [(x1, y1, x2, y2, confidence), ...]
        faces = []
        
        # Example processing (adjust based on actual output format)
        # This is a simplified version - real implementation depends on model outputs
        
        return faces
    
    def process_age_gender_outputs(self, outputs):
        """Process Age/Gender model outputs."""
        # Placeholder - implement based on model output format
        # Typically: age = argmax(age_logits), gender = argmax(gender_logits)
        
        age = 25  # Placeholder
        gender = "Male"  # Placeholder
        
        return age, gender
    
    def process_frame(self, frame):
        """Process a single frame through all models."""
        # Step 1: Detect faces
        faces = self.detect_faces(frame)
        
        results = []
        for face_bbox in faces:
            x1, y1, x2, y2 = face_bbox[:4]
            
            # Extract face region
            face_image = frame[y1:y2, x1:x2]
            
            # Step 2: Get face embedding
            embedding = self.get_face_embedding(face_image)
            
            # Step 3: Predict age and gender
            age, gender = self.predict_age_gender(face_image)
            
            results.append({
                'bbox': (x1, y1, x2, y2),
                'embedding': embedding,
                'age': age,
                'gender': gender
            })
        
        return results
    
    def run_video(self, video_source=0):
        """Run face recognition on video stream."""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video source {video_source}")
            return
        
        print(f"🎥 Starting video processing...")
        print(f"📊 Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Process frame through all models
            results = self.process_frame(frame)
            
            # Draw results
            for result in results:
                x1, y1, x2, y2 = result['bbox']
                age = result['age']
                gender = result['gender']
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw age and gender
                label = f"{gender}, {age}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Single DFP Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Face Recognition with Single DFP')
    parser.add_argument('--dfp', default='dfp_models/face_recognition_complete.dfp',
                       help='Path to single DFP file')
    parser.add_argument('--video', default=0, 
                       help='Video source (0 for camera, or path to video file)')
    
    args = parser.parse_args()
    
    try:
        # Initialize face recognition system
        face_system = SingleDFPFaceRecognition(args.dfp)
        
        # Run video processing
        face_system.run_video(args.video)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 