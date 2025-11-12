#!/bin/bash
# Setup script to create all Docker-related files and folders

set -e

echo "=========================================="
echo "Creating Docker Configuration Structure"
echo "=========================================="

# Get the root directory (where this script is)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "Root directory: $ROOT_DIR"
echo ""

# 1. Create docker folders
echo "[1/10] Creating docker folder structure..."
mkdir -p docker/mosquitto/config
mkdir -p docker/mosquitto/data
mkdir -p docker/mosquitto/log
mkdir -p docker/prometheus
mkdir -p docker/grafana/provisioning/datasources
mkdir -p docker/grafana/provisioning/dashboards
mkdir -p docker/grafana/dashboards
echo "✓ Docker folders created"

# 2. Create MQTT configuration
echo "[2/10] Creating MQTT configuration..."
cat > docker/mosquitto/config/mosquitto.conf << 'EOF'
# Mosquitto Configuration for Traffic System

listener 1883
protocol mqtt

listener 9001
protocol websockets

# Authentication
allow_anonymous false
password_file /mosquitto/config/passwd

# Persistence
persistence true
persistence_location /mosquitto/data/

# Logging
log_dest file /mosquitto/log/mosquitto.log
log_dest stdout
log_type all
log_timestamp true

# Security
max_connections -1
max_inflight_messages 20
max_queued_messages 1000
EOF
echo "✓ MQTT config created"

# 3. Create MQTT password file (will be generated later)
echo "[3/10] Creating MQTT password file placeholder..."
touch docker/mosquitto/config/passwd
echo "✓ MQTT password file created"

# 4. Create Prometheus configuration
echo "[4/10] Creating Prometheus configuration..."
cat > docker/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 10s
  evaluation_interval: 10s

scrape_configs:
  - job_name: 'traffic-edge-devices'
    static_configs:
      - targets: ['host.docker.internal:9091']
        labels:
          device: 'edge_001'
          intersection: 'mg_road'
          
  - job_name: 'infrastructure'
    static_configs:
      - targets:
        - 'mqtt:1883'
        - 'influxdb:8086'
        - 'redis:6379'
EOF
echo "✓ Prometheus config created"

# 5. Create Grafana datasource configuration
echo "[5/10] Creating Grafana datasource configuration..."
cat > docker/grafana/provisioning/datasources/influxdb.yml << 'EOF'
apiVersion: 1

datasources:
  - name: InfluxDB
    type: influxdb
    access: proxy
    url: http://influxdb:8086
    jsonData:
      version: Flux
      organization: traffic-org
      defaultBucket: traffic-metrics
      tlsSkipVerify: true
    secureJsonData:
      token: my-super-secret-auth-token
    isDefault: true
    editable: true
EOF
echo "✓ Grafana datasource created"

# 6. Create Grafana dashboard provisioning
echo "[6/10] Creating Grafana dashboard provisioning..."
cat > docker/grafana/provisioning/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'Traffic Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF
echo "✓ Grafana dashboard provisioning created"

# 7. Create sample Grafana dashboard
echo "[7/10] Creating sample Grafana dashboard..."
cat > docker/grafana/dashboards/traffic-overview.json << 'EOF'
{
  "dashboard": {
    "title": "Traffic Signal Overview",
    "panels": [
      {
        "id": 1,
        "title": "Vehicle Count by Direction",
        "type": "timeseries",
        "targets": [
          {
            "query": "from(bucket: \"traffic-metrics\")\n  |> range(start: -1h)\n  |> filter(fn: (r) => r._measurement == \"traffic_metrics\")\n  |> filter(fn: (r) => r._field == \"vehicle_count\")"
          }
        ]
      }
    ],
    "refresh": "5s",
    "time": {
      "from": "now-1h",
      "to": "now"
    }
  }
}
EOF
echo "✓ Grafana dashboard created"

# 8. Create deployment folder
echo "[8/10] Creating deployment folder..."
mkdir -p deployment
echo "✓ Deployment folder created"

# 9. Create .env.example
echo "[9/10] Creating .env.example..."
cat > .env.example << 'EOF'
# Traffic Signal Optimization - Environment Variables Template
# Copy this file to .env and update with your values

# Environment
TRAFFIC_ENV=production
DEVICE_ID=edge_001
LOG_LEVEL=INFO

# MQTT Configuration
MQTT_BROKER=localhost
MQTT_PORT=1883
MQTT_USERNAME=traffic_edge
MQTT_PASSWORD=change_me_please

# InfluxDB Configuration
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=my-super-secret-auth-token
INFLUXDB_ORG=traffic-org
INFLUXDB_BUCKET=traffic-metrics

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=traffic-redis-pass

# Signal Controller
CONTROLLER_IP=192.168.1.100
CONTROLLER_PROTOCOL=NTCIP
CONTROLLER_PORT=161

# Camera Configuration
CAMERA_NORTH_URL=rtsp://192.168.1.101:554/stream1
CAMERA_SOUTH_URL=rtsp://192.168.1.102:554/stream1
CAMERA_EAST_URL=rtsp://192.168.1.103:554/stream1
CAMERA_WEST_URL=rtsp://192.168.1.104:554/stream1
EOF
echo "✓ .env.example created"

# 10. Create .gitignore
echo "[10/10] Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.env.local

# Logs
*.log
logs/
*.log.*

# Data
data/
models/*.pt
models/*.pth
models/*.engine
*.h5
*.hdf5

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Docker
docker/mosquitto/data/*
docker/mosquitto/log/*
docker/mosquitto/config/passwd
!docker/mosquitto/config/mosquitto.conf

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Deployment
*.pid
*.sock
EOF
echo "✓ .gitignore created"

echo ""
echo "=========================================="
echo "✅ Docker Structure Created Successfully!"
echo "=========================================="
echo ""
echo "Created folders:"
echo "  - docker/mosquitto/"
echo "  - docker/prometheus/"
echo "  - docker/grafana/"
echo "  - deployment/"
echo ""
echo "Created files:"
echo "  - docker/mosquitto/config/mosquitto.conf"
echo "  - docker/prometheus/prometheus.yml"
echo "  - docker/grafana/provisioning/datasources/influxdb.yml"
echo "  - docker/grafana/provisioning/dashboards/dashboard.yml"
echo "  - docker/grafana/dashboards/traffic-overview.json"
echo "  - .env.example"
echo "  - .gitignore"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and update values"
echo "2. Generate MQTT passwords (see below)"
echo "3. Run: docker-compose up -d"
echo ""
echo "Generate MQTT password:"
echo "  docker run -it --rm -v \$(pwd)/docker/mosquitto/config:/mosquitto/config eclipse-mosquitto mosquitto_passwd -c /mosquitto/config/passwd traffic_edge"
echo ""
