# Traffic Signal Optimization System - Edge Device

Production-ready edge device software for AI-powered traffic signal optimization.

## 🎯 Features

- **Real-time Vehicle Detection**: YOLOv8-based detection optimized for Indian traffic
- **Multi-Object Tracking**: DeepSORT-style tracking with unique vehicle IDs
- **Indian Traffic Support**: Direction-based analysis (N, S, E, W) without lane markings
- **Microservices Architecture**: Independent, scalable components
- **Robust Communication**: MQTT for regional coordination
- **Time-series Storage**: InfluxDB for metrics, Redis for real-time state
- **Fail-safe Operation**: Automatic fallback mechanisms
- **Production-ready**: Comprehensive logging, monitoring, and error handling

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Edge Device                          │
├─────────────────────────────────────────────────────────┤
│  Video Processing → AI Detection → Tracking →           │
│  Metrics Aggregation → Decision Making →                │
│  Signal Control + Communication + Storage               │
└─────────────────────────────────────────────────────────┘
```

### Microservices Components

1. **Video Processing Service** (`video_processing/`)
   - Multi-camera stream handling
   - Automatic reconnection
   - Frame buffering

2. **AI Detection Service** (`ai/`)
   - YOLOv8 vehicle detection
   - DeepSORT tracking
   - Indian vehicle types support

3. **Analytics Service** (`analytics/`)
   - Traffic metrics calculation
   - State vector generation
   - PCU-based density

4. **Signal Control Service** (`signal_control/`)
   - NTCIP/MODBUS interface
   - Safety checks
   - Fail-safe mechanisms

5. **Communication Service** (`communication/`)
   - MQTT pub/sub
   - Message queuing
   - Auto-reconnect

6. **Storage Services** (`storage/`)
   - InfluxDB (time-series metrics)
   - Redis (real-time state)

## 📋 Prerequisites

### Hardware Requirements

**Minimum:**
- CPU: Intel i5 / AMD Ryzen 5 or better
- RAM: 8GB
- Storage: 100GB SSD
- Network: Gigabit Ethernet

**Recommended:**
- CPU: Intel i7 / AMD Ryzen 7
- GPU: NVIDIA RTX 3060 or better (for TensorRT)
- RAM: 16GB
- Storage: 256GB NVMe SSD
- Network: Gigabit Ethernet + 4G/5G backup

### Software Requirements

- **OS**: Ubuntu 22.04 LTS (recommended) or compatible Linux
- **Python**: 3.10 or higher
- **CUDA**: 12.x (if using NVIDIA GPU)
- **Docker**: 24.x (for dependencies)
- **Docker Compose**: 2.x

### External Services

- **MQTT Broker**: Mosquitto 2.0+
- **InfluxDB**: 2.7+
- **Redis**: 7.0+
- **Signal Controller**: NTCIP-compatible or Modbus

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository_url>
cd traffic-signal-optimization/edge

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install system dependencies (Ubuntu)
sudo apt-get update
sudo apt-get install -y \
    python3-opencv \
    libhdf5-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev
```

### 2. Start Infrastructure Services

```bash
# Start supporting services (MQTT, InfluxDB, Redis)
cd ..  # Go to project root
docker-compose up -d mqtt influxdb redis

# Verify services are running
docker-compose ps
```

### 3. Configure Your Intersection

```bash
# Copy example configuration
cp config/intersection_config.example.yaml config/intersection_config.yaml

# Edit configuration
nano config/intersection_config.yaml

# Key settings to update:
# - intersection_id, intersection_name, city
# - camera URLs and positions
# - controller IP and protocol
# - MQTT, InfluxDB, Redis connection details
```

### 4. Download AI Models

```bash
# Download pre-trained YOLOv8 model
mkdir -p models
cd models

# Download Indian traffic model (if available)
wget <model_url>/yolov8_indian_traffic.pt

# Or use standard YOLOv8
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
mv yolov8m.pt yolov8_indian_traffic.pt

cd ..
```

### 5. Test Configuration

```bash
# Test in simulation mode (no cameras required)
python src/main.py --config config/intersection_config.yaml --simulation

# You should see:
# - All services initializing
# - Health checks passing
# - Processing loop running
```

### 6. Run with Real Cameras

```bash
# Update config with real camera URLs
nano config/intersection_config.yaml

# Run the system
python src/main.py --config config/intersection_config.yaml

# Monitor logs
tail -f logs/edge_device.log
```

## 📊 Monitoring

### Grafana Dashboard

```bash
# Access Grafana
http://localhost:3000
# Default credentials: admin/admin

# Import dashboard from monitoring/grafana/dashboards/edge_metrics.json
```

### Prometheus Metrics

```bash
# Access Prometheus
http://localhost:9090

# View edge device metrics
http://localhost:9091/metrics
```

### Live Logs

```bash
# All logs
tail -f logs/edge_device.log

# Only errors
tail -f logs/edge_device.log | grep ERROR

# Follow with colors
tail -f logs/edge_device.log | ccze -A
```

## 🔧 Configuration Guide

### Camera Configuration

```yaml
cameras:
  - id: "cam_north"
    url: "rtsp://admin:password@192.168.1.101/stream1"  # RTSP URL
    # OR
    url: "0"  # USB camera device ID
    # OR
    url: "/path/to/video.mp4"  # Video file for testing
    
    direction: "north"  # north, south, east, west
    fov: 110  # Field of view in degrees
    height: 6.0  # Mounting height in meters
    angle: 50  # Mounting angle in degrees
    fps: 30  # Target frame rate
    resolution: [1920, 1080]  # Width x Height
    enabled: true
```

### Detection Zones

Detection zones define where to monitor traffic:

```yaml
detection_zones:
  north:
    - name: "stop_line_zone"
      zone_type: "polygon"
      points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]  # Polygon corners
      area_m2: 75  # Physical area in square meters
      purpose: "queue_at_stop_line"
```

**How to define zones:**
1. Capture a frame from your camera
2. Use an image editor to identify pixel coordinates
3. Measure physical area of the zone
4. Add to configuration

### Signal Phase Configuration

For Indian intersections (direction-based):

```yaml
signal_phases:
  - id: 1
    name: "NS_STRAIGHT_RIGHT"
    directions: ["north", "south"]  # Active directions
    movements: ["straight", "right"]  # Allowed movements
    min_green: 20  # Minimum green time (seconds)
    max_green: 90  # Maximum green time
    yellow_time: 5  # Yellow clearance
    all_red_time: 3  # All-red clearance
    priority: 1  # Phase priority (1 = highest)
```

### Environment Variables

Create `.env` file:

```bash
# Intersection
TRAFFIC_INTERSECTION_ID=INT_MUM_001
TRAFFIC_CITY=Mumbai
TRAFFIC_REGION=Maharashtra

# MQTT
TRAFFIC_MQTT_BROKER=localhost
TRAFFIC_MQTT_PORT=1883
TRAFFIC_MQTT_USERNAME=traffic_user
TRAFFIC_MQTT_PASSWORD=secure_password

# InfluxDB
TRAFFIC_INFLUXDB_URL=http://localhost:8086
TRAFFIC_INFLUXDB_TOKEN=your_influxdb_token
TRAFFIC_INFLUXDB_ORG=traffic-org
TRAFFIC_INFLUXDB_BUCKET=traffic-metrics

# Redis
TRAFFIC_REDIS_HOST=localhost
TRAFFIC_REDIS_PORT=6379
TRAFFIC_REDIS_PASSWORD=redis_password

# AI Models
TRAFFIC_DETECTOR_MODEL_PATH=models/yolov8_indian_traffic.pt
TRAFFIC_DEVICE=cuda  # cuda, cpu, or mps

# Logging
TRAFFIC_LOG_LEVEL=INFO
TRAFFIC_LOG_FILE=logs/edge_device.log
```

## 🐛 Troubleshooting

### Camera Connection Issues

```bash
# Test RTSP stream
ffplay rtsp://admin:password@192.168.1.101/stream1

# Check camera accessibility
ping 192.168.1.101

# Test with OpenCV
python -c "import cv2; cap = cv2.VideoCapture('rtsp://...'); print(cap.isOpened())"
```

### CUDA/GPU Issues

```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If GPU not working, switch to CPU in config
device: "cpu"
```

### Service Connection Issues

```bash
# Check if services are running
docker-compose ps

# Check MQTT
mosquitto_pub -h localhost -t test -m "hello"

# Check InfluxDB
curl http://localhost:8086/health

# Check Redis
redis-cli ping
```

### Performance Issues

```bash
# Reduce frame rate
target_fps: 15  # Instead of 30

# Enable frame skipping
frame_skip: 2  # Process every 2nd frame

# Reduce detection confidence
detection_confidence_threshold: 0.6  # Higher = fewer detections

# Use lighter model
detector_model_path: models/yolov8n.pt  # Nano model (faster)
```

## 📈 Performance Tuning

### For High Throughput

```yaml
# Increase batch sizes
influxdb_batch_size: 500
influxdb_flush_interval: 5

# Optimize camera settings
cameras:
  - fps: 20  # Lower FPS = less processing
    resolution: [1280, 720]  # Lower resolution
```

### For Low Latency

```yaml
# Real-time processing
target_fps: 30
frame_skip: 1

# Immediate flushing
influxdb_batch_size: 10
influxdb_flush_interval: 1

# TensorRT optimization (NVIDIA only)
use_tensorrt: true
```

### For Resource-Constrained Devices

```yaml
# Lightweight configuration
detector_model_path: models/yolov8n.pt  # Nano model
target_fps: 10
frame_skip: 3
batch_size: 1
```

## 🔐 Security Best Practices

1. **Change default passwords**
   ```bash
   # MQTT, InfluxDB, Redis passwords
   # Camera admin passwords
   ```

2. **Use HTTPS/TLS**
   ```yaml
   mqtt_port: 8883  # TLS port
   influxdb_url: https://localhost:8086
   ```

3. **Restrict network access**
   ```bash
   # Firewall rules
   sudo ufw allow from 192.168.1.0/24 to any port 1883
   ```

4. **Regular updates**
   ```bash
   pip install --upgrade -r requirements.txt
   docker-compose pull
   ```

## 📝 Development

### Running Tests

```bash
# All tests
pytest tests/

# Specific module
pytest tests/unit/test_detector.py

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/
isort src/

# Lint
flake8 src/
pylint src/

# Type checking
mypy src/
```

### Adding New Features

1. Create feature branch
2. Implement in appropriate microservice
3. Add tests
4. Update documentation
5. Submit PR

## 📚 API Documentation

### MQTT Topics

**Published by Edge Device:**
- `traffic/{city}/{region}/{intersection}/metrics` - Traffic metrics (QoS 1)
- `traffic/{city}/{region}/{intersection}/events` - Traffic events (QoS 1)
- `traffic/{city}/{region}/{intersection}/status` - System status (QoS 1, retained)
- `traffic/{city}/{region}/{intersection}/health` - Health check (QoS 1)

**Subscribed by Edge Device:**
- `traffic/{city}/{region}/{intersection}/commands` - Control commands (QoS 1)

### Command Format

```json
{
  "type": "change_phase",
  "phase_id": 3,
  "duration": 60,
  "timestamp": 1699999999.99
}
```

### Metrics Format

```json
{
  "intersection_id": "INT_MUM_001",
  "timestamp": 1699999999.99,
  "directions": {
    "north": {
      "density": 0.45,
      "queue_depth": 25.5,
      "avg_wait_time": 45.2,
      "vehicle_counts": {
        "two_wheeler": 15,
        "car": 8,
        "bus": 2
      }
    }
  }
}
```

## 🆘 Support

### Getting Help

1. Check [Troubleshooting](#troubleshooting) section
2. Review logs: `logs/edge_device.log`
3. Check system health: `http://localhost:9091/metrics`
4. Open GitHub issue with:
   - Log output
   - Configuration file (remove passwords)
   - System information

### Contributing

See `CONTRIBUTING.md` for guidelines.

## 📄 License

Proprietary - See LICENSE file

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- DeepSORT tracking algorithm
- Indian Driving Dataset (IDD)
- OpenCV community