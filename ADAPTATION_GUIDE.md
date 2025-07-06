# MemryX Tutorial Adaptation Guide

## Downloaded Resources

1. **models.dfp** - Working multi-model DFP file
   - Model 0: Face detection (face_detection_short_range.tflite)
   - Model 1: Emotion classification (mobilenet_7.h5)

2. **Python Code** - Complete AsyncAccl implementation
3. **C++ Code** - Complete MxAccl implementation

## Adapting for Face Recognition System

### Your Target Models:
- RetinaFace (face detection) 
- FaceNet512 (face embeddings)
- Age/Gender (demographics)

### Compilation Steps:

1. **Get Source Models** (ONNX/TensorFlow format):
```bash
# Download or convert your models to ONNX
wget https://github.com/onnx/models/raw/main/vision/body_analysis/age_gender/models/age_gender_inception.onnx
# ... other models
```

2. **Compile Multi-Model DFP**:
```bash
# Compile all models into single DFP
mx_nc -v -m retinaface.onnx facenet512.onnx age_gender.onnx --autocrop -o face_recognition.dfp

# Or compile individually
mx_nc -v -m retinaface.onnx -o retinaface.dfp
mx_nc -v -m facenet512.onnx -o facenet512.dfp  
mx_nc -v -m age_gender.onnx -o age_gender.dfp
```

### Code Adaptation:

1. **Update Model Paths** in your config.py:
```python
MEMRYX_CONFIG = {
    'models': {
        'retinaface': {
            'dfp_path': 'tutorial_models/face_recognition.dfp',  # Multi-model DFP
            'model_id': 0,  # First model in DFP
        },
        'facenet512': {
            'dfp_path': 'tutorial_models/face_recognition.dfp',  # Same DFP
            'model_id': 1,  # Second model in DFP
        },
        'age_gender': {
            'dfp_path': 'tutorial_models/face_recognition.dfp',  # Same DFP
            'model_id': 2,  # Third model in DFP
        }
    }
}
```

2. **Update Model Loading** in deepface.py:
```python
class RetinaFaceMXA:
    def __init__(self):
        self.accl = mx.AsyncAccl('tutorial_models/face_recognition.dfp')
        self.model_id = 0  # Face detection model
        
class FaceNet512MXA:
    def __init__(self):
        self.accl = mx.AsyncAccl('tutorial_models/face_recognition.dfp') 
        self.model_id = 1  # Face embedding model
```

3. **Use Tutorial Code Structure**:
   - Copy input/output callback patterns
   - Adapt preprocessing/postprocessing functions
   - Use multi-stream architecture from tutorial

### Testing Steps:

1. **Test with Tutorial Models**:
```bash
cd tutorial_code
python main.py  # Test tutorial face detection works
```

2. **Gradually Replace Models**:
   - Replace face detection model first
   - Then add face recognition
   - Finally add age/gender detection

3. **Integration**:
   - Copy working patterns to your app.py
   - Update deepface.py with tutorial approaches
   - Test with your camera setup

## Next Steps:

1. Test tutorial code with your cameras
2. Replace models step by step  
3. Adapt preprocessing/postprocessing
4. Integrate with your database system
