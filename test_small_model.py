#!/usr/bin/env python3
"""
Test Small RetinaFace Model
===========================

Test script to verify det_2.5g.onnx works correctly.
"""

import subprocess
import os
from pathlib import Path

def test_small_retinaface():
    """Test compilation of the smaller RetinaFace model."""
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
    
    # Check if small model exists
    small_model_path = onnx_dir / 'det_2.5g.onnx'
    if not small_model_path.exists():
        print(f"❌ ERROR: Small model not found: {small_model_path}")
        return False
    
    print(f"✅ Found small RetinaFace model: {small_model_path}")
    file_size = small_model_path.stat().st_size
    print(f"📊 Model size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    
    # Test compilation
    print(f"\n🔧 Testing compilation of small RetinaFace model...")
    output_dfp = dfp_dir / 'retinaface_small_test.dfp'
    
    cmd = [
        'mx_nc', '-v', 
        '-m', str(small_model_path),
        '-is', '320,320,3',
        '--autocrop',
        '--dfp_fname', str(output_dfp)
    ]
    
    print(f"🚀 Command: {' '.join(cmd)}")
    print(f"⏳ Compiling...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\n🎉 SUCCESS: Small RetinaFace compiled!")
            print(f"📁 Output: {output_dfp}")
            
            if output_dfp.exists():
                dfp_size = output_dfp.stat().st_size
                print(f"📊 DFP size: {dfp_size:,} bytes ({dfp_size/1024/1024:.1f} MB)")
            
            print(f"\n✅ The small model (det_2.5g.onnx) works correctly!")
            print(f"💡 Use this model instead of det_10g.onnx")
            
            return True
        else:
            print(f"\n❌ ERROR: Compilation failed")
            print(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def compare_models():
    """Compare the two RetinaFace models."""
    onnx_dir = Path("onnx_models")
    
    models = [
        {'file': 'det_2.5g.onnx', 'name': 'Small RetinaFace', 'input': '320x320'},
        {'file': 'det_10g.onnx', 'name': 'Large RetinaFace', 'input': '640x640'}
    ]
    
    print(f"\n📊 Model Comparison:")
    print("=" * 50)
    
    for model in models:
        model_path = onnx_dir / model['file']
        if model_path.exists():
            size = model_path.stat().st_size
            print(f"✅ {model['name']}")
            print(f"   File: {model['file']}")
            print(f"   Size: {size:,} bytes ({size/1024/1024:.1f} MB)")
            print(f"   Input: {model['input']}")
            print()
        else:
            print(f"❌ {model['name']}: {model['file']} - NOT FOUND")
    
    print("💡 Recommendation: Use det_2.5g.onnx for better compatibility")

if __name__ == "__main__":
    print("🔍 Testing Small RetinaFace Model")
    print("=" * 50)
    
    compare_models()
    
    print("\n" + "=" * 50)
    test_small_retinaface() 