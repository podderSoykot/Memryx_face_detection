#!/usr/bin/env python3
"""
Single DFP Compiler
==================

Compiles all face recognition models into a single multi-model DFP file.
"""

import subprocess
import os
from pathlib import Path

def compile_single_multi_dfp():
    """Compile all models into a single DFP file."""
    onnx_dir = Path("onnx_models")
    dfp_dir = Path("dfp_models")
    dfp_dir.mkdir(exist_ok=True)
    
    # Check if mx_nc is available
    try:
        result = subprocess.run(['mx_nc', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ ERROR: mx_nc not found. Please install MemryX SDK first.")
            return False
    except FileNotFoundError:
        print("❌ ERROR: mx_nc not found. Please install MemryX SDK first.")
        return False
    
    print("✅ SUCCESS: MemryX Neural Compiler found")
    
    # Model files and their input shapes
    models = [
        {'file': 'det_2.5g.onnx', 'shape': '320,320,3', 'name': 'RetinaFace'},
        {'file': 'w600k_r50.onnx', 'shape': '112,112,3', 'name': 'ArcFace'},
        {'file': 'genderage.onnx', 'shape': '224,224,3', 'name': 'Age/Gender'}
    ]
    
    # Check if all models exist
    model_paths = []
    for model in models:
        model_path = onnx_dir / model['file']
        if not model_path.exists():
            print(f"❌ ERROR: Model not found: {model_path}")
            return False
        model_paths.append(str(model_path))
        print(f"✅ Found: {model['name']} - {model['file']}")
    
    # Single multi-model DFP compilation
    print(f"\n🔧 Compiling Single Multi-Model DFP")
    print("=" * 50)
    
    output_dfp = dfp_dir / 'face_recognition_complete.dfp'
    
    # Build command
    cmd = ['mx_nc', '-v', '-m'] + model_paths
    
    # Add input shapes for each model
    for model in models:
        cmd.extend(['-is', model['shape']])
    
    # Add final parameters
    cmd.extend(['--autocrop', '--dfp_fname', str(output_dfp)])
    
    print(f"🚀 Command: {' '.join(cmd)}")
    print(f"📦 Output: {output_dfp}")
    print(f"⏳ This may take 10-20 minutes for all models...")
    print(f"🔄 Compiling...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\n🎉 SUCCESS: Single DFP created!")
            print(f"📁 File: {output_dfp}")
            
            # Show file size
            if output_dfp.exists():
                file_size = output_dfp.stat().st_size
                print(f"📊 Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            
            print(f"\n📋 Model Access:")
            for i, model in enumerate(models):
                print(f"   Model {i}: {model['name']} ({model['shape']})")
            
            print(f"\n🎯 Usage in your app:")
            print(f"   dfp_path = 'dfp_models/face_recognition_complete.dfp'")
            print(f"   # Model 0: RetinaFace (face detection)")
            print(f"   # Model 1: ArcFace (face recognition)")
            print(f"   # Model 2: Age/Gender (demographics)")
            
            return True
        else:
            print(f"\n❌ ERROR: Compilation failed")
            print(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Single DFP Compiler")
    print("=" * 50)
    compile_single_multi_dfp() 