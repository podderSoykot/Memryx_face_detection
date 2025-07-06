#!/usr/bin/env python3
"""
Test MemryX DFP Models
======================

Test script to verify all three DFP models are working correctly:
- retinaface_memryx.dfp (face detection)
- facenet512_memryx_raw_embeddings.dfp (face recognition)
- genderage.dfp (age/gender classification)
"""

import os
import cv2
import numpy as np
import memryx as mx
from pathlib import Path
import time

def test_dfp_models():
    """Test all three DFP models."""
    
    # Model paths
    models = {
        'retinaface': 'dfp_models/retinaface_memryx.dfp',
        'facenet512': 'dfp_models/facenet512_memryx_raw_embeddings.dfp',
        'age_gender': 'dfp_models/genderage.dfp'
    }
    
    # Check if all models exist
    print("🔍 Checking DFP model files...")
    for model_name, path in models.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✅ {model_name}: {path} ({size/1024/1024:.1f} MB)")
        else:
            print(f"❌ {model_name}: {path} - NOT FOUND")
            return False
    
    print("\n🚀 Initializing MemryX Accelerator...")
    
    try:
        # Initialize accelerator
        accl = mx.Accelerator()
        
        # Load DFP models
        dfp_models = {}
        for model_name, path in models.items():
            print(f"📦 Loading {model_name}...")
            dfp_models[model_name] = mx.DFP(path)
            accl.connect_dfp(dfp_models[model_name])
        
        print("✅ All DFP models loaded successfully!")
        
        # Get model streams (assuming models are loaded in order)
        print("\n🔗 Setting up model streams...")
        
        # RetinaFace streams (model 0)
        retinaface_input = accl.get_input_stream(0)
        retinaface_output = accl.get_output_stream(0)
        
        # FaceNet512 streams (model 1)
        facenet_input = accl.get_input_stream(1)
        facenet_output = accl.get_output_stream(1)
        
        # Age/Gender streams (model 2)
        age_gender_input = accl.get_input_stream(2)
        age_gender_output = accl.get_output_stream(2)
        
        print("✅ Model streams setup complete!")
        
        # Test with dummy data
        print("\n🧪 Testing models with dummy data...")
        
        # Test RetinaFace (320x320x3)
        print("🔍 Testing RetinaFace model...")
        dummy_detection_input = np.random.rand(1, 320, 320, 3).astype(np.float32)
        
        start_time = time.time()
        retinaface_input.send_data(dummy_detection_input)
        detection_output = retinaface_output.get_data()
        detection_time = time.time() - start_time
        
        print(f"✅ RetinaFace inference: {detection_time*1000:.1f}ms")
        print(f"   Input shape: {dummy_detection_input.shape}")
        print(f"   Output shape: {detection_output.shape if hasattr(detection_output, 'shape') else 'N/A'}")
        
        # Test FaceNet512 (112x112x3)
        print("\n🔍 Testing FaceNet512 model...")
        dummy_face_input = np.random.rand(1, 112, 112, 3).astype(np.float32)
        
        start_time = time.time()
        facenet_input.send_data(dummy_face_input)
        embedding_output = facenet_output.get_data()
        embedding_time = time.time() - start_time
        
        print(f"✅ FaceNet512 inference: {embedding_time*1000:.1f}ms")
        print(f"   Input shape: {dummy_face_input.shape}")
        print(f"   Output shape: {embedding_output.shape if hasattr(embedding_output, 'shape') else 'N/A'}")
        
        # Test Age/Gender (224x224x3)
        print("\n🔍 Testing Age/Gender model...")
        dummy_age_gender_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        start_time = time.time()
        age_gender_input.send_data(dummy_age_gender_input)
        age_gender_output_data = age_gender_output.get_data()
        age_gender_time = time.time() - start_time
        
        print(f"✅ Age/Gender inference: {age_gender_time*1000:.1f}ms")
        print(f"   Input shape: {dummy_age_gender_input.shape}")
        print(f"   Output shape: {age_gender_output_data.shape if hasattr(age_gender_output_data, 'shape') else 'N/A'}")
        
        # Summary
        print(f"\n🎉 All models tested successfully!")
        print(f"📊 Performance Summary:")
        print(f"   RetinaFace: {detection_time*1000:.1f}ms")
        print(f"   FaceNet512: {embedding_time*1000:.1f}ms")
        print(f"   Age/Gender: {age_gender_time*1000:.1f}ms")
        print(f"   Total: {(detection_time + embedding_time + age_gender_time)*1000:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def test_with_camera():
    """Test models with actual camera input."""
    print("\n📹 Testing with camera input...")
    
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return False
        
        # Initialize models
        accl = mx.Accelerator()
        
        # Load models
        retinaface_dfp = mx.DFP('dfp_models/retinaface_memryx.dfp')
        facenet_dfp = mx.DFP('dfp_models/facenet512_memryx_raw_embeddings.dfp')
        age_gender_dfp = mx.DFP('dfp_models/genderage.dfp')
        
        accl.connect_dfp(retinaface_dfp)
        accl.connect_dfp(facenet_dfp)
        accl.connect_dfp(age_gender_dfp)
        
        # Get streams
        retinaface_input = accl.get_input_stream(0)
        retinaface_output = accl.get_output_stream(0)
        
        print("✅ Camera test setup complete!")
        print("📹 Press 'q' to quit camera test")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess for RetinaFace
            resized = cv2.resize(frame, (320, 320))
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb_frame.astype(np.float32) / 255.0
            input_data = np.expand_dims(normalized, axis=0)
            
            # Run RetinaFace inference
            start_time = time.time()
            retinaface_input.send_data(input_data)
            output = retinaface_output.get_data()
            inference_time = time.time() - start_time
            
            # Display results
            cv2.putText(frame, f"Inference: {inference_time*1000:.1f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('MemryX DFP Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("✅ Camera test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Camera test error: {e}")
        return False

def main():
    """Main function."""
    print("🔍 MemryX DFP Models Test")
    print("=" * 50)
    
    # Test 1: Basic model loading and inference
    if test_dfp_models():
        print("\n" + "=" * 50)
        
        # Test 2: Camera test (optional)
        response = input("Would you like to test with camera? (y/n): ").lower()
        if response == 'y':
            test_with_camera()
    
    print("\n🎯 Test completed!")

if __name__ == "__main__":
    main() 