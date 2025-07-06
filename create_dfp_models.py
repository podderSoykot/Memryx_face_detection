#!/usr/bin/env python3
"""
MemryX DFP Model Creation Script
===============================

This script downloads ONNX models and compiles them into DFP (DataFlow Program) files
for MemryX hardware acceleration.

Based on the MemryX tutorial: https://developer.memryx.com/tutorials/realtime_inf/realtime_multimodel.html

Face Recognition Models:
- RetinaFace: Face detection
- ArcFace: Face recognition embeddings  
- Age/Gender: Demographics classification
"""

import os
import sys
import requests
import subprocess
import zipfile
import tarfile
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Optional

# Model configurations with download URLs
FACE_MODELS = {
    'retinaface': {
        'description': 'RetinaFace face detection model',
        'source_url': 'https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/det_10g.onnx',
        'filename': 'retinaface.onnx',
        'input_size': (640, 640),
        'format': 'onnx'
    },
    'arcface': {
        'description': 'ArcFace face recognition model',
        'source_url': 'https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/w600k_r50.onnx',
        'filename': 'arcface.onnx',
        'input_size': (112, 112),
        'format': 'onnx'
    },
    'age_gender': {
        'description': 'Age and Gender classification model',
        'source_url': 'https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/genderage.onnx',
        'filename': 'age_gender.onnx',
        'input_size': (224, 224),
        'format': 'onnx'
    }
}

# Alternative models (backup sources)
ALTERNATIVE_MODELS = {
    'retinaface_alt': {
        'description': 'Alternative RetinaFace (smaller)',
        'source_url': 'https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/det_2.5g.onnx',
        'filename': 'retinaface_small.onnx',
        'input_size': (320, 320),
        'format': 'onnx'
    }
}

class DFPModelCreator:
    def __init__(self):
        self.models_dir = Path("models")
        self.source_dir = Path("source_models")
        self.dfp_dir = Path("dfp_models")
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.source_dir.mkdir(exist_ok=True)
        self.dfp_dir.mkdir(exist_ok=True)
    
    def download_model(self, model_config: Dict, target_path: Path) -> bool:
        """Download a model file from URL."""
        try:
            print(f"🔄 Downloading {model_config['description']}")
            print(f"   URL: {model_config['source_url']}")
            
            response = requests.get(model_config['source_url'], stream=True, timeout=60)
            response.raise_for_status()
            
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = target_path.stat().st_size
            print(f"✅ Downloaded: {target_path} ({file_size:,} bytes)")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def verify_mx_nc_installed(self) -> bool:
        """Verify MemryX Neural Compiler is installed."""
        try:
            result = subprocess.run(['mx_nc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ MemryX Neural Compiler found: {result.stdout.strip()}")
                return True
            else:
                print("❌ mx_nc not found or not working")
                return False
        except FileNotFoundError:
            print("❌ mx_nc (MemryX Neural Compiler) not found in PATH")
            return False
    
    def compile_single_dfp(self, model_path: Path, output_path: Path) -> bool:
        """Compile a single ONNX model to DFP."""
        try:
            print(f"🔧 Compiling {model_path.name} to DFP...")
            
            cmd = [
                'mx_nc',
                '-v',
                '-m', str(model_path),
                '--dfp_fname', str(output_path)
            ]
            
            print(f"   Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Compiled: {output_path}")
                print(f"   Output: {result.stdout}")
                return True
            else:
                print(f"❌ Compilation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Compilation error: {e}")
            return False
    
    def compile_multi_model_dfp(self, model_paths: List[Path], output_path: Path) -> bool:
        """Compile multiple ONNX models into a single DFP."""
        try:
            print(f"🔧 Compiling multi-model DFP...")
            print(f"   Models: {[p.name for p in model_paths]}")
            
            cmd = [
                'mx_nc',
                '-v',
                '-m'
            ] + [str(p) for p in model_paths] + [
                '--autocrop',
                '--dfp_fname', str(output_path)
            ]
            
            print(f"   Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Multi-model DFP compiled: {output_path}")
                print(f"   Models accessible as:")
                for i, model in enumerate(model_paths):
                    print(f"   - Model {i}: {model.name}")
                return True
            else:
                print(f"❌ Multi-model compilation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Multi-model compilation error: {e}")
            return False
    
    def create_config_file(self, dfp_path: Path, model_info: Dict):
        """Create configuration file for the compiled DFP."""
        config_content = f'''
"""
MemryX DFP Configuration
========================

Generated for: {dfp_path.name}
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DFP_DIR = BASE_DIR / "dfp_models"

# MemryX Configuration
MEMRYX_CONFIG = {{
    'enabled': True,
    'device_id': 0,
    'models': {{
        'retinaface': {{
            'dfp_path': str(DFP_DIR / '{dfp_path.name}'),
            'model_id': 0,  # First model in multi-model DFP
            'input_size': {model_info.get('retinaface', {}).get('input_size', (640, 640))},
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4
        }},
        'arcface': {{
            'dfp_path': str(DFP_DIR / '{dfp_path.name}'),
            'model_id': 1,  # Second model in multi-model DFP
            'input_size': {model_info.get('arcface', {}).get('input_size', (112, 112))},
            'embedding_size': 512
        }},
        'age_gender': {{
            'dfp_path': str(DFP_DIR / '{dfp_path.name}'),
            'model_id': 2,  # Third model in multi-model DFP
            'input_size': {model_info.get('age_gender', {}).get('input_size', (224, 224))},
            'age_classes': 101,
            'gender_classes': 2
        }}
    }}
}}

# Validation
def validate_dfp_files():
    """Validate DFP files exist."""
    missing_files = []
    
    for model_name, config in MEMRYX_CONFIG['models'].items():
        dfp_path = Path(config['dfp_path'])
        if not dfp_path.exists():
            missing_files.append(str(dfp_path))
    
    if missing_files:
        print("❌ Missing DFP files:")
        for file in missing_files:
            print(f"   - {{file}}")
        return False
    
    print("✅ All DFP files found")
    return True

if __name__ == "__main__":
    validate_dfp_files()
'''
        
        config_path = Path("config_dfp.py")
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Created configuration: {config_path}")
    
    def run_full_pipeline(self):
        """Run the complete model download and compilation pipeline."""
        print("🚀 MemryX DFP Model Creation Pipeline")
        print("=" * 80)
        
        # Step 1: Check MemryX compiler
        print("\n🔍 Step 1: Check MemryX Neural Compiler")
        print("-" * 50)
        if not self.verify_mx_nc_installed():
            print("\n❌ MemryX Neural Compiler not found!")
            print("Please install MemryX SDK first:")
            print("   1. Download from: https://developer.memryx.com/")
            print("   2. Follow installation guide")
            print("   3. Ensure mx_nc is in your PATH")
            return False
        
        # Step 2: Download models
        print("\n📥 Step 2: Download ONNX Models")
        print("-" * 50)
        downloaded_models = {}
        
        for model_name, config in FACE_MODELS.items():
            target_path = self.source_dir / config['filename']
            
            if target_path.exists():
                print(f"✅ Model already exists: {target_path}")
                downloaded_models[model_name] = target_path
            else:
                if self.download_model(config, target_path):
                    downloaded_models[model_name] = target_path
                else:
                    print(f"❌ Failed to download {model_name}")
        
        if len(downloaded_models) == 0:
            print("❌ No models downloaded successfully")
            return False
        
        # Step 3: Compile individual DFPs
        print(f"\n🔧 Step 3: Compile Individual DFP Files")
        print("-" * 50)
        compiled_dfps = []
        
        for model_name, model_path in downloaded_models.items():
            dfp_path = self.dfp_dir / f"{model_name}.dfp"
            if self.compile_single_dfp(model_path, dfp_path):
                compiled_dfps.append((model_name, dfp_path))
        
        # Step 4: Compile multi-model DFP
        print(f"\n🏗️ Step 4: Compile Multi-Model DFP")
        print("-" * 50)
        
        if len(downloaded_models) >= 2:
            model_paths = list(downloaded_models.values())
            multi_dfp_path = self.dfp_dir / "face_recognition_multi.dfp"
            
            if self.compile_multi_model_dfp(model_paths, multi_dfp_path):
                print(f"✅ Multi-model DFP created: {multi_dfp_path}")
                
                # Step 5: Create configuration
                print(f"\n⚙️ Step 5: Create Configuration")
                print("-" * 50)
                self.create_config_file(multi_dfp_path, FACE_MODELS)
            else:
                print(f"❌ Multi-model DFP compilation failed")
        
        # Step 6: Summary
        print(f"\n🎉 Pipeline Complete!")
        print("=" * 80)
        print(f"✅ Downloaded models: {len(downloaded_models)}")
        print(f"✅ Compiled DFPs: {len(compiled_dfps)}")
        
        print(f"\n📁 Generated Files:")
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.dfp'):
                    full_path = os.path.join(root, file)
                    size = os.path.getsize(full_path)
                    print(f"   {full_path} ({size:,} bytes)")
        
        print(f"\n🚀 Next Steps:")
        print(f"1. Update your app.py to use: config_dfp.py")
        print(f"2. Test with: python app.py --video_paths 0")
        print(f"3. Verify face detection and recognition work")
        
        return True

def main():
    """Main function."""
    creator = DFPModelCreator()
    creator.run_full_pipeline()

if __name__ == "__main__":
    main() 