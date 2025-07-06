#!/usr/bin/env python3
"""
MemryX Setup Workflow
====================

Complete step-by-step guide for setting up DFP files and adapting your face recognition system.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

class MemryXSetupWorkflow:
    def __init__(self):
        self.base_dir = Path(".")
        self.tutorial_dir = self.base_dir / "tutorial_models"
        self.models_dir = self.base_dir / "models"
        
    def step_1_manual_download(self):
        """Step 1: Manual Download Instructions"""
        print("🚀 STEP 1: Manual Download from MemryX Tutorial")
        print("=" * 60)
        print("📋 Go to: https://developer.memryx.com/tutorials/realtime_inf/realtime_multimodel.html")
        print("📥 Download these 3 files:")
        print("   1. models.dfp (Pre-compiled multi-model DFP)")
        print("   2. multimodel_python.tar.xz (Python code)")
        print("   3. MultiModel_C++.zip (C++ code)")
        print()
        print("📁 Place downloaded files in:")
        print(f"   - models.dfp → {self.tutorial_dir}/models.dfp")
        print(f"   - multimodel_python.tar.xz → tutorial_code/")
        print(f"   - MultiModel_C++.zip → tutorial_code/")
        print()
        
        # Check if files exist
        models_dfp = self.tutorial_dir / "models.dfp"
        if models_dfp.exists():
            print("✅ models.dfp found!")
        else:
            print("❌ models.dfp NOT found - please download manually")
            
        return models_dfp.exists()
    
    def step_2_verify_tutorial_dfp(self):
        """Step 2: Verify Tutorial DFP File"""
        print("\n🔍 STEP 2: Verify Tutorial DFP File")
        print("=" * 60)
        
        models_dfp = self.tutorial_dir / "models.dfp"
        if not models_dfp.exists():
            print("❌ models.dfp not found. Please complete Step 1 first.")
            return False
            
        file_size = models_dfp.stat().st_size
        print(f"✅ Found: {models_dfp}")
        print(f"📊 Size: {file_size:,} bytes")
        
        # This DFP contains:
        print("\n📋 Tutorial DFP contains:")
        print("   - Model 0: Face Detection (face_detection_short_range.tflite)")
        print("   - Model 1: Emotion Classification (mobilenet_7.h5)")
        print("   - Both models optimized for MemryX MXA hardware")
        
        return True
    
    def step_3_test_tutorial_code(self):
        """Step 3: Test Tutorial Code (Optional)"""
        print("\n🧪 STEP 3: Test Tutorial Code")
        print("=" * 60)
        print("📂 Extract and test tutorial Python code:")
        print("   1. Extract multimodel_python.tar.xz to tutorial_code/")
        print("   2. cd tutorial_code/")
        print("   3. python main.py")
        print()
        print("🎯 This will test face detection + emotion recognition")
        print("   Use this to verify MemryX hardware is working")
        
    def step_4_adapt_config(self):
        """Step 4: Adapt Your Configuration"""
        print("\n⚙️ STEP 4: Adapt Your Configuration")
        print("=" * 60)
        
        # Update config.py to use tutorial DFP temporarily
        config_content = '''
# Updated config.py for MemryX Tutorial DFP
MEMRYX_CONFIG = {
    'enabled': True,
    'models': {
        'retinaface': {
            'dfp_path': 'tutorial_models/models.dfp',
            'model_id': 0,  # Face detection model from tutorial
            'input_size': (128, 128),  # Tutorial face detection input size
            'confidence_threshold': 0.5
        },
        'facenet512': {
            'dfp_path': 'tutorial_models/models.dfp',
            'model_id': 1,  # Use emotion model temporarily
            'input_size': (224, 224),  # Tutorial emotion model input size
            'embedding_size': 7  # Emotion classes, not face embedding
        },
        'age_gender': {
            'dfp_path': 'tutorial_models/models.dfp',
            'model_id': 1,  # Same as above for testing
            'input_size': (224, 224),
            'age_classes': 7,  # Emotion classes
            'gender_classes': 2
        }
    }
}
'''
        
        print("📝 Create config_tutorial.py with tutorial DFP settings:")
        with open('config_tutorial.py', 'w') as f:
            f.write(config_content)
        
        print("✅ Created: config_tutorial.py")
        print("   Use this to test with tutorial models first")
        
    def step_5_test_integration(self):
        """Step 5: Test Integration"""
        print("\n🔗 STEP 5: Test Integration")
        print("=" * 60)
        print("🧪 Test your app with tutorial models:")
        print("   1. Backup your original config.py")
        print("   2. Copy config_tutorial.py to config.py")
        print("   3. python app.py --video_paths 0  # Test with webcam")
        print("   4. Check if face detection works")
        print()
        print("⚠️  Note: Face recognition won't work properly since tutorial")
        print("   uses emotion detection, not face embeddings")
        
    def step_6_compile_your_models(self):
        """Step 6: Compile Your Face Recognition Models"""
        print("\n🏗️ STEP 6: Compile Your Face Recognition Models")
        print("=" * 60)
        print("📥 First, get your models in ONNX format:")
        print("   - RetinaFace.onnx")
        print("   - FaceNet512.onnx")
        print("   - Age_Gender.onnx")
        print()
        print("🔧 Compile your DFP file:")
        print("   mx_nc -v -m RetinaFace.onnx FaceNet512.onnx Age_Gender.onnx \\")
        print("         --autocrop -o face_recognition.dfp")
        print()
        print("📁 Final structure:")
        print("   models/")
        print("   ├── face_recognition.dfp  # Your compiled models")
        print("   └── tutorial_models/")
        print("       └── models.dfp        # Tutorial models")
        
    def step_7_final_config(self):
        """Step 7: Final Configuration"""
        print("\n🎯 STEP 7: Final Configuration")
        print("=" * 60)
        print("⚙️ Update config.py with your compiled DFP:")
        
        final_config = '''
# Final config.py for your face recognition system
MEMRYX_CONFIG = {
    'enabled': True,
    'models': {
        'retinaface': {
            'dfp_path': 'models/face_recognition.dfp',
            'model_id': 0,  # First model in your DFP
            'input_size': (640, 640),
            'confidence_threshold': 0.5
        },
        'facenet512': {
            'dfp_path': 'models/face_recognition.dfp',
            'model_id': 1,  # Second model in your DFP
            'input_size': (160, 160),
            'embedding_size': 512
        },
        'age_gender': {
            'dfp_path': 'models/face_recognition.dfp',
            'model_id': 2,  # Third model in your DFP
            'input_size': (224, 224),
            'age_classes': 101,
            'gender_classes': 2
        }
    }
}
'''
        
        print("📝 Update config.py with final settings")
        print("🚀 Launch: python app.py --video_paths /dev/video0 /dev/video1")
        
    def run_workflow(self):
        """Run the complete workflow"""
        print("🎯 MemryX Face Recognition Setup Workflow")
        print("=" * 80)
        print()
        
        # Step 1: Manual Download
        has_tutorial_dfp = self.step_1_manual_download()
        
        if has_tutorial_dfp:
            # Step 2: Verify DFP
            self.step_2_verify_tutorial_dfp()
            
            # Step 3: Test tutorial code
            self.step_3_test_tutorial_code()
            
            # Step 4: Adapt config
            self.step_4_adapt_config()
            
            # Step 5: Test integration
            self.step_5_test_integration()
        
        # Step 6: Compile your models
        self.step_6_compile_your_models()
        
        # Step 7: Final config
        self.step_7_final_config()
        
        print("\n🎉 WORKFLOW SUMMARY")
        print("=" * 80)
        print("✅ Tutorial DFP provides working face detection")
        print("✅ Use tutorial code as template for your system")
        print("✅ Compile your own models for full face recognition")
        print("✅ Replace tutorial DFP with your compiled DFP")
        print()
        print("🚀 Ready to run your multi-camera face recognition system!")

def main():
    """Main workflow function"""
    workflow = MemryXSetupWorkflow()
    workflow.run_workflow()

if __name__ == "__main__":
    main() 