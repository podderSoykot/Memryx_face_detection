# MemryX Multi-Camera Face Recognition Setup Guide

This guide provides step-by-step instructions to run your multi-camera face recognition system on MemryX hardware.

## 🛠️ Prerequisites

### Hardware Requirements

- MemryX MXA M.2 AI Accelerator card (installed in PCIe M.2 slot)
- USB cameras or IP cameras (minimum 2 cameras)
- System with minimum 8GB RAM
- Ubuntu 20.04+ or compatible Linux distribution

### Software Requirements

- Python 3.8+
- Git
- SQLite (built into Python)
- MemryX SDK

---

## 📋 Step-by-Step Installation

### Step 1: Verify MemryX Hardware Installation

```bash
# Check if MemryX card is detected
lspci | grep -i memryx

# Expected output: Something like "MemryX MXA" device
# If nothing appears, check hardware installation
```

### Step 2: Install MemryX SDK

```bash
# Download MemryX SDK (follow official MemryX documentation)
# Visit: https://developer.memryx.com/

# Install MemryX SDK
wget https://developer.memryx.com/downloads/memryx-sdk-latest.tar.gz
tar -xzf memryx-sdk-latest.tar.gz
cd memryx-sdk

# Install SDK
sudo ./install.sh

# Verify installation
python3 -c "import memryx; print('MemryX SDK installed successfully')"
```

### Step 3: System Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3-pip python3-dev build-essential cmake
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y sqlite3  # SQLite database
sudo apt install -y v4l-utils  # For camera utilities

# Install additional dependencies
pip3 install --upgrade pip
```

### Step 4: Clone and Setup Project

```bash
# Navigate to your project directory
cd /path/to/your/project

# Install Python dependencies
pip3 install -r requirements.txt

# If requirements.txt is missing, install manually:
pip3 install numpy opencv-python faiss-cpu
pip3 install deepface tensorflow
```

### Step 5: SQLite Database Setup

```bash
# SQLite is file-based and requires no server setup
# The database file will be created automatically when the application runs

# Create data directory for the database
mkdir -p data

# Test SQLite installation
sqlite3 --version

# Database file will be created at: data/face_recognition.db
```

### Step 6: Environment Configuration

```bash
# Create environment file
cat > .env << EOF
# Database Configuration (SQLite)
DB_PATH=data/face_recognition.db

# MemryX Configuration
MEMRYX_DEVICE_ID=0
MEMRYX_GROUP_ID=0
EOF

# Source environment variables
source .env
export $(cat .env | xargs)
```

### Step 7: Model Files Setup

```bash
# Create model directories
mkdir -p models/retinaface models/facenet models/age_gender

# Download or copy your .dfp model files
# You need to obtain these from MemryX Model Explorer or compile them yourself

# Example structure:
# models/
# ├── retinaface/
# │   └── retinaface_mnet025_v1.dfp
# ├── facenet/
# │   └── facenet512.dfp
# └── age_gender/
#     └── age_gender_model.dfp

# Verify model files exist
ls -la models/*/
```

### Step 8: Camera Setup and Testing

```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera access
v4l2-ctl --device=/dev/video0 --all
v4l2-ctl --device=/dev/video1 --all

# Test camera capture
ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 test_cam0.jpg
ffmpeg -f v4l2 -i /dev/video1 -frames:v 1 test_cam1.jpg

# Verify images were captured
ls -la test_cam*.jpg
```

### Step 9: Database Initialization

```bash
# Initialize database tables
python3 setup_database.py

# Verify database tables were created
python3 setup_database.py --info
```

### Step 10: Configuration Verification

```bash
# Test configuration
python3 -c "
from config import CAMERA_CONFIG, MEMRYX_CONFIG
print('Camera sources:', CAMERA_CONFIG['sources'])
print('RetinaFace model:', MEMRYX_CONFIG['models']['retinaface']['dfp_path'])
print('Configuration loaded successfully')
"
```

---

## 🚀 Running the System

### Basic Run Commands

```bash
# Run with default configuration (2 cameras)
python3 app.py

# Run with specific cameras
python3 app.py --video_paths /dev/video0 /dev/video1

# Run with custom camera configuration
python3 app.py --video_paths /dev/video0 /dev/video1 /dev/video2

# Run without display (headless mode)
python3 app.py --no_display

# Run with IP cameras
python3 app.py --video_paths /dev/video0 rtsp://192.168.1.100:554/stream
```

### Advanced Run Commands

```bash
# Run with custom configuration file
python3 app.py -c custom_config.py

# Run with specific MemryX device
MEMRYX_DEVICE_ID=1 python3 app.py

# Run with debug logging
python3 app.py --video_paths /dev/video0 /dev/video1 --debug

# Run with performance monitoring
python3 app.py --video_paths /dev/video0 /dev/video1 --monitor
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. MemryX Card Not Detected

```bash
# Check hardware connection
lspci | grep -i memryx

# If not detected, power cycle the system
sudo reboot

# Check dmesg for hardware errors
dmesg | grep -i memryx
```

#### 2. Camera Access Issues

```bash
# Check camera permissions
ls -la /dev/video*

# Add user to video group
sudo usermod -a -G video $USER
newgrp video

# Test camera access
v4l2-ctl --device=/dev/video0 --info
```

#### 3. Database Connection Issues

```bash
# Check SQLite database file
ls -la data/face_recognition.db

# Test database connection
sqlite3 data/face_recognition.db "SELECT sqlite_version();"

# Check database tables
python3 setup_database.py --info
```

#### 4. Model File Issues

```bash
# Check model file existence
find models/ -name "*.dfp" -type f

# Check file permissions
ls -la models/*/*.dfp

# Verify model file integrity
file models/retinaface/retinaface_mnet025_v1.dfp
```

#### 5. Memory Issues

```bash
# Check system memory
free -h

# Monitor memory usage during runtime
top -p $(pgrep -f "python3 app.py")

# If memory issues persist, reduce camera resolution in config.py
```

#### 6. Performance Issues

```bash
# Check MemryX utilization
# (Use MemryX monitoring tools if available)

# Monitor CPU usage
htop

# Check camera frame rates
v4l2-ctl --device=/dev/video0 --get-parm
```

---

## 📊 Performance Monitoring

### Real-time Monitoring Commands

```bash
# Monitor system resources
watch -n 1 "free -h && echo '---' && df -h"

# Monitor camera processes
watch -n 1 "ps aux | grep -E 'python3|app.py'"

# Monitor database file size
watch -n 1 "ls -lh data/face_recognition.db"

# Monitor network (for IP cameras)
watch -n 1 "netstat -i"
```

### Log Monitoring

```bash
# Follow application logs
tail -f logs/face_recognition.log

# Monitor system logs
sudo journalctl -f -u memryx

# Check error logs
grep -i error logs/face_recognition.log
```

---

## 🎯 Testing Commands

### System Testing

```bash
# Test basic functionality
python3 -c "
from app import FaceRecognitionMXA
system = FaceRecognitionMXA(video_paths=['/dev/video0'], show=False)
print('System initialized successfully')
system.cleanup()
"

# Test MemryX integration
python3 -c "
from memryx import MultiStreamAsyncAccl
print('MemryX MultiStreamAsyncAccl import successful')
"

# Test DeepFace system
python3 -c "
from deepface import DeepFaceSystem
system = DeepFaceSystem()
print('DeepFace system initialized successfully')
system.cleanup()
"
```

### Performance Testing

```bash
# Run performance benchmark
python3 benchmark_multistream.py --streams 2 --duration 60

# Test with different camera configurations
python3 app.py --video_paths /dev/video0 --no_display &
sleep 30
pkill -f "python3 app.py"
```

---

## 🔒 Security Considerations

### Database Security

```bash
# Secure database file permissions
chmod 600 data/face_recognition.db

# Backup database regularly
cp data/face_recognition.db data/face_recognition_backup_$(date +%Y%m%d_%H%M%S).db
```

### System Security

```bash
# Limit camera access
sudo chown root:video /dev/video*
sudo chmod 660 /dev/video*

# Secure log files
chmod 600 logs/face_recognition.log
```

---

## 📈 Optimization Tips

### Camera Optimization

```bash
# Set optimal camera parameters
v4l2-ctl --device=/dev/video0 --set-ctrl=brightness=128
v4l2-ctl --device=/dev/video0 --set-ctrl=contrast=128
v4l2-ctl --device=/dev/video0 --set-ctrl=saturation=128

# Set camera resolution and frame rate
v4l2-ctl --device=/dev/video0 --set-fmt-video=width=1280,height=720,pixelformat=MJPG
v4l2-ctl --device=/dev/video0 --set-parm=30
```

### System Optimization

```bash
# Increase system limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize SQLite database
sqlite3 data/face_recognition.db << EOF
PRAGMA cache_size = 10000;
PRAGMA temp_store = MEMORY;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
EOF
```

---

## 🆘 Emergency Stop

### Stop All Processes

```bash
# Stop application gracefully
pkill -SIGTERM -f "python3 app.py"

# Force stop if needed
pkill -SIGKILL -f "python3 app.py"

# Release camera resources
sudo fuser -k /dev/video0 /dev/video1

# Stop database connections
sudo systemctl restart postgresql
```

---

## 📞 Support Resources

### MemryX Support

- Documentation: https://developer.memryx.com/
- Support: contact MemryX support team
- Forums: MemryX community forums

### System Logs Location

- Application logs: `logs/face_recognition.log`
- System logs: `/var/log/syslog`
- SQLite database: `data/face_recognition.db`

### Common File Locations

- Configuration: `config.py`
- Environment: `.env`
- Models: `models/`
- Database setup: `setup_database.py`

---

**Note**: This guide assumes you have the necessary MemryX model files (.dfp) and proper hardware setup.
