#!/usr/bin/env python3
"""
MemryX DFP Compilation Script
============================

Run this script AFTER installing MemryX SDK to compile ONNX models into DFP files.
"""

import subprocess
import os
from pathlib import Path

def compile_models():
    """Compile ONNX models to DFP files."""
    onnx_dir = Path("onnx_models")
    dfp_dir = Path("dfp_models")
    dfp_dir.mkdir(exist_ok=True)
    
    # Check if mx_nc is available
    try:
        result = subprocess.run(['mx_nc', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("ERROR: mx_nc not found. Please install MemryX SDK first.")
            return False
    except FileNotFoundError:
        print("ERROR: mx_nc not found. Please install MemryX SDK first.")
        return False
    
    print("SUCCESS: MemryX Neural Compiler found")
    
    # Models to compile
    models = [
        {
            'name': 'retinaface',
            'onnx': 'det_10g.onnx',
            'dfp': 'retinaface.dfp'
        },
        {
            'name': 'arcface', 
            'onnx': 'w600k_r50.onnx',
            'dfp': 'arcface.dfp'
        },
        {
            'name': 'age_gender',
            'onnx': 'genderage.onnx', 
            'dfp': 'age_gender.dfp'
        }
    ]
    
    compiled_models = []
    
    # Compile individual models
    for model in models:
        onnx_path = onnx_dir / model['onnx']
        dfp_path = dfp_dir / model['dfp']
        
        if not onnx_path.exists():
            print(f"ERROR: ONNX model not found: {onnx_path}")
            continue
        
        print(f"\nCompiling {model['name']}")
        cmd = ['mx_nc', '-v', '-m', str(onnx_path), '--dfp_fname', str(dfp_path)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"SUCCESS: Compiled {dfp_path}")
                compiled_models.append(model)
            else:
                print(f"ERROR: Compilation failed: {result.stderr}")
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Compile multi-model DFP
    if len(compiled_models) >= 2:
        print(f"\nCreating Multi-Model DFP")
        onnx_paths = [str(onnx_dir / model['onnx']) for model in compiled_models]
        multi_dfp = dfp_dir / 'face_recognition_multi.dfp'
        
        cmd = ['mx_nc', '-v', '-m'] + onnx_paths + ['--autocrop', '--dfp_fname', str(multi_dfp)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"SUCCESS: Multi-model DFP created: {multi_dfp}")
                print(f"Models accessible as:")
                for i, model in enumerate(compiled_models):
                    print(f"  - Model {i}: {model['name']}")
            else:
                print(f"ERROR: Multi-model compilation failed: {result.stderr}")
        except Exception as e:
            print(f"ERROR: {e}")
    
    print(f"\nCompilation Complete!")
    print(f"SUCCESS: Compiled {len(compiled_models)} models")
    return True

if __name__ == "__main__":
    compile_models()
