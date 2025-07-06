# Multi-Camera Face Recognition System with MemryX Acceleration

A high-performance, real-time face recognition system designed for multiple camera streams using MemryX hardware acceleration with DeepFace integration.

## 🚀 Key Features

- **MemryX MultiStreamAsyncAccl**: Hardware-optimized multi-camera processing
- **Real-time Face Detection**: RetinaFace with MemryX acceleration
- **Face Recognition**: FaceNet512 embeddings with FAISS similarity search
- **Demographic Analysis**: Age and gender detection with MemryX acceleration
- **Multi-Camera Support**: Simultaneous processing of multiple camera streams
- **Database Integration**: SQLite logging and FAISS embedding storage
- **UUID-based Tracking**: Unique identification for face matching

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MemryX MultiStreamAsyncAccl                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Camera 1 → Input Callback → RetinaFace (MXA) → Output Callback → Results  │
│  Camera 2 → Input Callback → RetinaFace (MXA) → Output Callback → Results  │
│  Camera N → Input Callback → RetinaFace (MXA) → Output Callback → Results  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DeepFace Processing Pipeline                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Face Detection Results → FaceNet512 (MXA) → FAISS Search → UUID Matching │
│                      ↓                                                     │
│               Age/Gender Analysis (MXA) → SQLite Logging                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Real-time Display                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Stream-specific Windows → Face Boxes → Age/Gender → UUID → FPS Counter   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

- **Hardware**: MemryX MXA (M.2 AI Accelerator)
- **Framework**: DeepFace with MemryX SDK
- **Face Detection**: RetinaFace (MemryX accelerated)
- **Face Recognition**: FaceNet512 (MemryX accelerated)
- **Demographics**: Age/Gender models (MemryX accelerated)
- **Database**: SQLite + FAISS vector database
- **Programming**: Python 3.8+, OpenCV, NumPy

## 📋 Prerequisites

### Hardware Requirements

- MemryX MXA M.2 AI Accelerator card
- USB cameras or IP cameras
- Sufficient system RAM (8GB+ recommended)
- PCIe M.2 slot for MemryX card

### Software Requirements

- Python 3.8 or higher
- MemryX SDK installed
- SQLite database (built into Python)
- CUDA-compatible GPU (optional, for CPU fallback)

## 🔧 Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd multi-camera-face-recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. MemryX SDK Installation

Follow the official MemryX SDK installation guide:

```bash
# Install MemryX SDK (example)
pip install memryx-sdk
```

### 4. Database Setup

```bash
# Setup SQLite database
python setup_database.py
```

### 5. Model Files

Ensure MemryX .dfp model files are present:

```
models/
├── retinaface/
│   └── retinaface_mnet025_v1.dfp
├── facenet/
│   └── facenet512.dfp
└── age_gender/
    └── age_gender_model.dfp
```

## ⚙️ Configuration

### Camera Configuration (`config.py`)

```python
CAMERA_CONFIG = {
    'sources': [
        '/dev/video0',    # USB Camera 1
        '/dev/video1',    # USB Camera 2
        # 'rtsp://ip:port/stream',  # IP Camera
    ],
    'camera_settings': {
        0: {
            'name': 'Main Entrance',
            'location': 'entrance_main',
            'resolution': (1280, 720),
            'fps': 30,
        },
        # Additional cameras...
    }
}
```

### MemryX Model Configuration

```python
MEMRYX_CONFIG = {
    'models': {
        'retinaface': {
            'dfp_path': 'models/retinaface/retinaface_mnet025_v1.dfp',
            'input_size': (640, 640),
            'confidence_threshold': 0.7,
        },
        'facenet512': {
            'dfp_path': 'models/facenet/facenet512.dfp',
            'input_size': (160, 160),
            'embedding_size': 512,
        },
        'age_gender': {
            'dfp_path': 'models/age_gender/age_gender_model.dfp',
            'input_size': (224, 224),
        }
    }
}
```

## 🚦 Usage

### Basic Usage

```bash
# Use default camera configuration
python app.py

# Specify custom camera sources
python app.py --video_paths /dev/video0 /dev/video1

# Run without display (headless mode)
python app.py --no_display

# Use custom configuration
python app.py -c custom_config.py
```

### Advanced Usage

```bash
# Multi-camera with IP cameras
python app.py --video_paths /dev/video0 rtsp://192.168.1.100:554/stream

# Video file processing
python app.py --video_paths sample_video1.mp4 sample_video2.mp4
```

## 🎯 System Features

### MemryX MultiStreamAsyncAccl Benefits

- **Hardware Optimization**: Direct MemryX hardware acceleration
- **Stream Management**: Built-in multi-camera stream handling
- **Performance**: Optimized for high-throughput inference
- **Scalability**: Efficient resource utilization across streams

### Face Recognition Pipeline

1. **Input Callback**: Captures frames from each camera stream
2. **RetinaFace Detection**: MemryX-accelerated face detection
3. **Output Callback**: Processes detection results
4. **Face Embedding**: FaceNet512 feature extraction
5. **FAISS Search**: Fast similarity matching
6. **Demographics**: Age/gender analysis for new faces
7. **Database Logging**: SQLite storage with UUID tracking

### Real-time Display

- **Stream-specific Windows**: Individual display for each camera
- **Face Annotations**: Bounding boxes, age, gender, confidence
- **FPS Counter**: Performance monitoring per stream
- **UUID Display**: Face matching identification

## 🧪 Testing

### Unit Tests

```bash
# Test configuration validation
python test_deepface_system.py

# Test MemryX MultiStreamAsyncAccl implementation
python test_memryx_multistream.py
```

### Performance Testing

```bash
# Benchmark multi-camera performance
python benchmark_multistream.py --streams 4 --duration 60
```

## 📊 Performance Metrics

### Expected Performance (with MemryX MXA)

- **Face Detection**: 60+ FPS per stream (RetinaFace)
- **Face Recognition**: 30+ FPS per stream (FaceNet512)
- **Demographics**: 25+ FPS per stream (Age/Gender)
- **Multi-Camera**: 4+ simultaneous streams at full performance

### Performance Monitoring

- Real-time FPS display per camera stream
- Processing latency measurement
- Memory usage tracking
- Database query performance

## 🔍 Database Schema

### Face Detections

```sql
CREATE TABLE face_detections (
    id SERIAL PRIMARY KEY,
    face_uuid UUID NOT NULL,
    camera_id INTEGER NOT NULL,
    camera_name VARCHAR(255),
    location VARCHAR(255),
    detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,
    confidence FLOAT
);
```

### Face Demographics

```sql
CREATE TABLE face_demographics (
    id SERIAL PRIMARY KEY,
    face_uuid UUID NOT NULL,
    age INTEGER,
    gender VARCHAR(10),
    gender_confidence FLOAT,
    analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Face Embeddings

```sql
CREATE TABLE face_embeddings (
    id SERIAL PRIMARY KEY,
    face_uuid UUID UNIQUE NOT NULL,
    embedding BYTEA NOT NULL,
    embedding_model VARCHAR(100),
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🛡️ Security Features

- **Database Encryption**: Optional encryption for sensitive data
- **Access Control**: Database user authentication
- **Data Privacy**: Configurable data retention policies
- **Secure Logging**: Structured logging with sensitive data filtering

## 📝 Configuration Files

### `config.py`

- Camera settings and sources
- MemryX model configurations
- Database connection parameters
- System performance settings

### `requirements.txt`

- Python dependencies
- MemryX SDK requirements
- Database drivers
- Image processing libraries

## 🔧 Troubleshooting

### Common Issues

1. **MemryX Card Not Detected**

   ```bash
   # Check MemryX card status
   lspci | grep -i memryx

   # Verify MemryX SDK installation
   python -c "import memryx; print('MemryX SDK OK')"
   ```

2. **Camera Access Issues**

   ```bash
   # Check camera permissions
   ls -la /dev/video*

   # Test camera access
   v4l2-ctl --list-devices
   ```

3. **Database Connection Issues**

   ```bash
   # Test SQLite database
   python setup_database.py --info

   # Check database setup
   python setup_database.py --verify
   ```

4. **Model File Issues**

   ```bash
   # Verify model files exist
   find models/ -name "*.dfp" -type f

   # Check model file permissions
   ls -la models/*/
   ```

### Performance Optimization

1. **Multi-Camera Performance**

   - Adjust `stream_workers` parameter
   - Optimize camera resolution settings
   - Balance processing threads

2. **Memory Management**

   - Monitor FAISS index size
   - Implement embedding cache cleanup
   - Optimize database connection pooling

3. **MemryX Optimization**
   - Use appropriate batch sizes
   - Optimize model input preprocessing
   - Monitor accelerator utilization

## 📈 Monitoring and Logging

### System Logs

```bash
# View system logs
tail -f logs/face_recognition.log

# View error logs
grep ERROR logs/face_recognition.log
```

### Performance Metrics

```bash
# Monitor system performance
python monitor_performance.py

# Generate performance report
python generate_report.py --duration 24h
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MemryX team for hardware acceleration support
- DeepFace community for face recognition frameworks
- OpenCV contributors for computer vision tools
- SQLite team for database infrastructure

## 📞 Support

For technical support or questions:

- Create an issue in the repository
- Check troubleshooting guide above
- Review MemryX SDK documentation

---

**Note**: This system requires MemryX MXA hardware for optimal performance. CPU fallback is available but may result in reduced performance for multi-camera scenarios.
