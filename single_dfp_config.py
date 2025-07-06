#!/usr/bin/env python3
"""
Single DFP Configuration
========================

Configuration for using the single multi-model DFP file.
"""

from pathlib import Path

# Single DFP File Path
SINGLE_DFP_PATH = "dfp_models/face_recognition_complete.dfp"

# Model Configuration
MODELS = {
    # Model 0: RetinaFace (Face Detection)
    'retinaface': {
        'model_id': 0,
        'input_size': (320, 320, 3),
        'purpose': 'Face Detection',
        'outputs': 'Bounding boxes, confidence scores'
    },
    
    # Model 1: ArcFace (Face Recognition)
    'arcface': {
        'model_id': 1,
        'input_size': (112, 112, 3),
        'purpose': 'Face Recognition',
        'outputs': '512-dimensional face embeddings'
    },
    
    # Model 2: Age/Gender (Demographics)
    'age_gender': {
        'model_id': 2,
        'input_size': (224, 224, 3),
        'purpose': 'Age and Gender Prediction',
        'outputs': 'Age (0-100), Gender (Male/Female)'
    }
}

# Usage Instructions
USAGE_INSTRUCTIONS = """
How to Use Single DFP:
=====================

1. Load the DFP file:
   ```python
   import memryx as mx
   
   # Initialize accelerator
   accl = mx.Accelerator()
   
   # Load single DFP
   dfp = mx.DFP("dfp_models/face_recognition_complete.dfp")
   accl.connect_dfp(dfp)
   ```

2. Access individual models:
   ```python
   # Model 0: RetinaFace (Face Detection)
   retinaface_input = accl.get_input_stream(0)
   retinaface_output = accl.get_output_stream(0)
   
   # Model 1: ArcFace (Face Recognition)
   arcface_input = accl.get_input_stream(1)
   arcface_output = accl.get_output_stream(1)
   
   # Model 2: Age/Gender (Demographics)
   age_gender_input = accl.get_input_stream(2)
   age_gender_output = accl.get_output_stream(2)
   ```

3. Run inference:
   ```python
       # Example: Face detection
    preprocessed_image = preprocess_image(image, (320, 320))
    retinaface_input.send_data(preprocessed_image)
    detection_results = retinaface_output.get_data()
   
   # Example: Face recognition
   face_crop = extract_face(image, bbox)
   preprocessed_face = preprocess_image(face_crop, (112, 112))
   arcface_input.send_data(preprocessed_face)
   face_embedding = arcface_output.get_data()
   
   # Example: Age/Gender prediction
   preprocessed_face = preprocess_image(face_crop, (224, 224))
   age_gender_input.send_data(preprocessed_face)
   age_gender_results = age_gender_output.get_data()
   ```

4. Complete pipeline:
   ```python
   # Step 1: Detect faces
   faces = detect_faces(image)  # Using Model 0
   
   for face_bbox in faces:
       face_image = extract_face(image, face_bbox)
       
       # Step 2: Get face embedding
       embedding = get_face_embedding(face_image)  # Using Model 1
       
       # Step 3: Predict age and gender
       age, gender = predict_age_gender(face_image)  # Using Model 2
   ```

Benefits of Single DFP:
======================
✅ Faster loading (one file vs three)
✅ Better memory efficiency
✅ Simplified deployment
✅ Optimized execution across models
✅ Easier version management

Example Usage:
=============
python3 app_single_dfp.py --dfp dfp_models/face_recognition_complete.dfp --video 0
"""

def print_usage():
    """Print usage instructions."""
    print(USAGE_INSTRUCTIONS)

def validate_single_dfp():
    """Validate that the single DFP file exists."""
    dfp_path = Path(SINGLE_DFP_PATH)
    
    if dfp_path.exists():
        file_size = dfp_path.stat().st_size
        print(f"✅ Single DFP found: {dfp_path}")
        print(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"📋 Contains {len(MODELS)} models:")
        
        for model_name, config in MODELS.items():
            print(f"   Model {config['model_id']}: {model_name} - {config['purpose']}")
        
        return True
    else:
        print(f"❌ Single DFP not found: {dfp_path}")
        print(f"💡 Run: python3 compile_single_dfp.py")
        return False

if __name__ == "__main__":
    validate_single_dfp()
    print("\n" + "="*50)
    print_usage() 