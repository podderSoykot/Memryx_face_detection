#!/bin/bash

# MemryX Multi-Camera Face Recognition Setup Script
# This script automates the setup process for MemryX hardware

set -e  # Exit on any error

echo "=============================================="
echo "MemryX Multi-Camera Face Recognition Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root. Please run as a regular user."
   exit 1
fi

# Step 1: Check hardware
print_step "1. Checking MemryX hardware..."
if lspci | grep -i memryx > /dev/null; then
    print_status "MemryX hardware detected"
else
    print_warning "MemryX hardware not detected. Please check hardware installation."
    read -p "Do you want to continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 2: Check cameras
print_step "2. Checking camera devices..."
if ls /dev/video* > /dev/null 2>&1; then
    print_status "Camera devices found: $(ls /dev/video*)"
else
    print_warning "No camera devices found"
fi

# Step 3: Install system dependencies
print_step "3. Installing system dependencies..."
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    libopencv-dev \
    python3-opencv \
    sqlite3 \
    v4l-utils \
    ffmpeg

print_status "System dependencies installed"

# Step 4: Install Python dependencies
print_step "4. Installing Python dependencies..."
pip3 install --upgrade pip

# Install requirements if file exists
if [[ -f "requirements.txt" ]]; then
    pip3 install -r requirements.txt
else
    print_warning "requirements.txt not found. Installing basic dependencies..."
    pip3 install \
        numpy \
        opencv-python \
        faiss-cpu \
        deepface \
        tensorflow
fi

print_status "Python dependencies installed"

# Step 5: Setup SQLite database
print_step "5. Setting up SQLite database..."
# SQLite is file-based, no server setup needed
# Database will be created automatically when the application runs

print_status "SQLite database ready (file-based, no server required)"

# Step 6: Create environment file
print_step "6. Creating environment configuration..."
cat > .env << EOF
# Database Configuration (SQLite)
DB_PATH=data/face_recognition.db

# MemryX Configuration
MEMRYX_DEVICE_ID=0
MEMRYX_GROUP_ID=0
EOF

print_status "Environment file created (.env)"

# Step 7: Create model directories
print_step "7. Creating model directories..."
mkdir -p models/retinaface
mkdir -p models/facenet
mkdir -p models/age_gender
mkdir -p logs
mkdir -p data

print_status "Model directories created"

# Step 8: Test camera access
print_step "8. Testing camera access..."
# Add user to video group
sudo usermod -a -G video $USER

# Test camera access
if [[ -e "/dev/video0" ]]; then
    v4l2-ctl --device=/dev/video0 --info > /dev/null 2>&1 && \
    print_status "Camera /dev/video0 accessible" || \
    print_warning "Camera /dev/video0 not accessible"
fi

if [[ -e "/dev/video1" ]]; then
    v4l2-ctl --device=/dev/video1 --info > /dev/null 2>&1 && \
    print_status "Camera /dev/video1 accessible" || \
    print_warning "Camera /dev/video1 not accessible"
fi

# Step 9: Initialize database
print_step "9. Initializing database..."
if [[ -f "setup_database.py" ]]; then
    python3 setup_database.py
    print_status "Database tables created"
else
    print_warning "setup_database.py not found. Please run it manually later."
fi

# Step 10: Test configuration
print_step "10. Testing configuration..."
python3 << EOF
try:
    from config import CAMERA_CONFIG, MEMRYX_CONFIG
    print("Configuration loaded successfully")
    print(f"Camera sources: {CAMERA_CONFIG['sources']}")
    print(f"Model paths configured: {len(MEMRYX_CONFIG['models'])} models")
except Exception as e:
    print(f"Configuration error: {e}")
    exit(1)
EOF

print_status "Configuration test passed"

# Final instructions
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
print_status "Your MemryX multi-camera face recognition system is ready!"
echo ""
echo "Next steps:"
echo "1. Obtain and place your .dfp model files in the models/ directory:"
echo "   - models/retinaface/retinaface_mnet025_v1.dfp"
echo "   - models/facenet/facenet512.dfp"
echo "   - models/age_gender/age_gender_model.dfp"
echo ""
echo "2. Test the system:"
echo "   python3 app.py --video_paths /dev/video0 /dev/video1"
echo ""
echo "3. For headless operation:"
echo "   python3 app.py --video_paths /dev/video0 /dev/video1 --no_display"
echo ""
echo "4. Check the setup guide:"
echo "   cat MEMRYX_SETUP_GUIDE.md"
echo ""
print_warning "Important: You need to reboot or run 'newgrp video' to apply video group permissions"
echo ""
echo "Database configuration saved in .env file:"
echo "SQLite database path: data/face_recognition.db"
echo ""
print_status "Setup completed successfully!" 