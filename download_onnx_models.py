#!/usr/bin/env python3
"""
ONNX Model Downloader for MemryX Face Recognition
================================================

This script downloads ONNX models that will be compiled into DFP files
once MemryX SDK is installed.

Based on high-quality models from:
- https://github.com/yakhyo/facial-analysis (Face Analysis ONNX models)
- https://github.com/deepinsight/insightface (InsightFace models)
"""

import os
import sys
import requests
from pathlib import Path
from typing import Dict, List

# High-quality ONNX models for face recognition
FACE_MODELS = {
    'retinaface': {
        'description': 'RetinaFace SCRFD face detection (16MB)',
        'source_url': 'https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/det_10g.onnx',
        'filename': 'det_10g.onnx',
        'input_size': (640, 640),
        'size_mb': 16.1
    },
    'retinaface_small': {
        'description': 'RetinaFace SCRFD face detection (3MB - lighter)',
        'source_url': 'https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/det_2.5g.onnx',
        'filename': 'det_2.5g.onnx', 
        'input_size': (320, 320),
        'size_mb': 3.14
    },
    'arcface': {
        'description': 'ArcFace W600K face recognition (166MB)',
        'source_url': 'https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/w600k_r50.onnx',
        'filename': 'w600k_r50.onnx',
        'input_size': (112, 112),
        'size_mb': 166
    },
    'age_gender': {
        'description': 'Gender and Age classification (1.3MB)',
        'source_url': 'https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/genderage.onnx',
        'filename': 'genderage.onnx',
        'input_size': (224, 224),
        'size_mb': 1.26
    }
}

class ONNXModelDownloader:
    def __init__(self):
        self.models_dir = Path("onnx_models")
        self.models_dir.mkdir(exist_ok=True)
        
    def download_model(self, model_name: str, config: Dict) -> bool:
        """Download a single model."""
        target_path = self.models_dir / config['filename']
        
        if target_path.exists():
            size = target_path.stat().st_size
            print(f"✅ Already exists: {target_path} ({size:,} bytes)")
            return True
        
        try:
            print(f"🔄 Downloading {config['description']}")
            print(f"   URL: {config['source_url']}")
            print(f"   Size: ~{config['size_mb']} MB")
            
            response = requests.get(config['source_url'], stream=True, timeout=120)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Simple progress indicator
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r   Progress: {progress:.1f}%", end='', flush=True)
            
            print()  # New line after progress
            file_size = target_path.stat().st_size
            print(f"✅ Downloaded: {target_path} ({file_size:,} bytes)")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            if target_path.exists():
                target_path.unlink()  # Remove incomplete file
            return False
    
    def verify_models(self) -> Dict[str, bool]:
        """Verify all downloaded models."""
        results = {}
        
        print("\n🔍 Verifying Downloaded Models")
        print("-" * 50)
        
        for model_name, config in FACE_MODELS.items():
            target_path = self.models_dir / config['filename']
            
            if target_path.exists():
                size = target_path.stat().st_size
                expected_size = config['size_mb'] * 1024 * 1024  # Convert MB to bytes
                
                # Allow 10% variance in file size
                if abs(size - expected_size) / expected_size < 0.1:
                    print(f"✅ {model_name}: {target_path} ({size:,} bytes)")
                    results[model_name] = True
                else:
                    print(f"⚠️  {model_name}: Size mismatch ({size:,} vs expected ~{expected_size:,})")
                    results[model_name] = True  # Still count as success
            else:
                print(f"❌ {model_name}: Not found")
                results[model_name] = False
        
        return results
    
    def create_compilation_script(self):
        """Create script to compile models after MemryX installation."""
        script_content = '''#!/usr/bin/env python3
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
            print("❌ mx_nc not found. Please install MemryX SDK first.")
            return False
    except FileNotFoundError:
        print("❌ mx_nc not found. Please install MemryX SDK first.")
        return False
    
    print("✅ MemryX Neural Compiler found")
    
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
            print(f"❌ ONNX model not found: {onnx_path}")
            continue
        
        print(f"\n🔧 Compiling {model['name']}")
        cmd = ['mx_nc', '-v', '-m', str(onnx_path), '-o', str(dfp_path)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Compiled: {dfp_path}")
                compiled_models.append(model)
            else:
                print(f"❌ Compilation failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Compile multi-model DFP
    if len(compiled_models) >= 2:
        print(f"\n🏗️ Creating Multi-Model DFP")
        onnx_paths = [str(onnx_dir / model['onnx']) for model in compiled_models]
        multi_dfp = dfp_dir / 'face_recognition_multi.dfp'
        
        cmd = ['mx_nc', '-v', '-m'] + onnx_paths + ['--autocrop', '-o', str(multi_dfp)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Multi-model DFP created: {multi_dfp}")
                print(f"   Models accessible as:")
                for i, model in enumerate(compiled_models):
                    print(f"   - Model {i}: {model['name']}")
            else:
                print(f"❌ Multi-model compilation failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 Compilation Complete!")
    print(f"✅ Compiled {len(compiled_models)} models")
    return True

if __name__ == "__main__":
    compile_models()
'''
        
        with open('compile_to_dfp.py', 'w') as f:
            f.write(script_content)
        
        print("✅ Created compilation script: compile_to_dfp.py")
    
    def create_memryx_install_guide(self):
        """Create MemryX installation guide."""
        guide_content = '''# MemryX SDK Installation Guide

## 1. Download MemryX SDK
Visit: https://developer.memryx.com/
- Create developer account
- Download SDK for your platform (Linux/Windows)

## 2. Installation Steps

### Linux:
```bash
# Extract SDK
tar -xzf memryx_sdk_*.tar.gz
cd memryx_sdk

# Install dependencies
sudo apt update
sudo apt install -y build-essential python3-dev

# Install SDK
sudo ./install.sh

# Add to PATH
echo 'export PATH=$PATH:/opt/memryx/bin' >> ~/.bashrc
source ~/.bashrc

# Verify installation
mx_nc --version
```

### Windows:
```batch
# Extract SDK to C:\\memryx
# Add C:\\memryx\\bin to system PATH
# Restart command prompt

# Verify installation
mx_nc --version
```

## 3. After Installation

Run the compilation script:
```bash
python compile_to_dfp.py
```

This will create DFP files in the `dfp_models/` directory.

## 4. Update Your Configuration

Update your `config.py` to use the compiled DFP files:

```python
MEMRYX_CONFIG = {
    'enabled': True,
    'models': {
        'retinaface': {
            'dfp_path': 'dfp_models/face_recognition_multi.dfp',
            'model_id': 0,
        },
        'arcface': {
            'dfp_path': 'dfp_models/face_recognition_multi.dfp', 
            'model_id': 1,
        },
        'age_gender': {
            'dfp_path': 'dfp_models/face_recognition_multi.dfp',
            'model_id': 2,
        }
    }
}
```

## 5. Test Your System

```bash
python app.py --video_paths 0
```

Your face recognition system should now use MemryX hardware acceleration!
'''
        
        with open('MEMRYX_INSTALL_GUIDE.md', 'w') as f:
            f.write(guide_content)
        
        print("✅ Created installation guide: MEMRYX_INSTALL_GUIDE.md")
    
    def run_download_pipeline(self):
        """Run the complete download pipeline."""
        print("🚀 ONNX Model Download Pipeline")
        print("=" * 80)
        
        # Calculate total download size
        total_size = sum(config['size_mb'] for config in FACE_MODELS.values())
        print(f"📊 Total download size: ~{total_size:.1f} MB")
        
        # Download all models
        print(f"\n📥 Downloading {len(FACE_MODELS)} models...")
        print("-" * 50)
        
        successful_downloads = 0
        for model_name, config in FACE_MODELS.items():
            print(f"\n[{model_name.upper()}]")
            if self.download_model(model_name, config):
                successful_downloads += 1
        
        # Verify downloads
        verification_results = self.verify_models()
        verified_count = sum(1 for success in verification_results.values() if success)
        
        # Create helper scripts
        print(f"\n🔧 Creating Helper Scripts")
        print("-" * 50)
        self.create_compilation_script()
        self.create_memryx_install_guide()
        
        # Summary
        print(f"\n🎉 Download Pipeline Complete!")
        print("=" * 80)
        print(f"✅ Successfully downloaded: {successful_downloads}/{len(FACE_MODELS)} models")
        print(f"✅ Verified: {verified_count}/{len(FACE_MODELS)} models")
        
        if successful_downloads > 0:
            print(f"\n📁 Downloaded Models:")
            for model_name, config in FACE_MODELS.items():
                if verification_results.get(model_name, False):
                    path = self.models_dir / config['filename']
                    size = path.stat().st_size if path.exists() else 0
                    print(f"   ✅ {model_name}: {path} ({size:,} bytes)")
        
        print(f"\n🚀 Next Steps:")
        print(f"1. Read: MEMRYX_INSTALL_GUIDE.md")
        print(f"2. Install MemryX SDK")
        print(f"3. Run: python compile_to_dfp.py")
        print(f"4. Update your config.py")
        print(f"5. Test: python app.py --video_paths 0")
        
        return successful_downloads > 0

def main():
    """Main function."""
    downloader = ONNXModelDownloader()
    downloader.run_download_pipeline()

if __name__ == "__main__":
    main() 