#!/bin/bash
# Installation script for Traffic Signal Optimization - Edge Device

set -e

echo "=========================================="
echo "Traffic Signal Optimization - Installation"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Configuration
INSTALL_DIR="/opt/traffic-signal-optimization/edge"
LOG_DIR="/var/log/traffic"
DATA_DIR="/var/traffic"
SERVICE_USER="traffic"

# Create user
echo "[1/10] Creating service user..."
if ! id "$SERVICE_USER" &>/dev/null; then
    useradd -r -s /bin/bash -d "$INSTALL_DIR" "$SERVICE_USER"
    echo "✓ User created: $SERVICE_USER"
else
    echo "✓ User already exists: $SERVICE_USER"
fi

# Create directories
echo "[2/10] Creating directories..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/video_buffer"
mkdir -p "$DATA_DIR/models"

# Install system dependencies
echo "[3/10] Installing system dependencies..."
apt-get update
apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3-opencv \
    libhdf5-dev \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    gfortran \
    git \
    curl \
    snmp \
    snmp-mibs-downloader

# Install Docker (if not present)
echo "[4/10] Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker "$SERVICE_USER"
    echo "✓ Docker installed"
else
    echo "✓ Docker already installed"
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
        -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo "✓ Docker Compose installed"
fi

# Copy application files
echo "[5/10] Copying application files..."
cp -r ../edge/* "$INSTALL_DIR/"
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

# Create Python virtual environment
echo "[6/10] Creating Python virtual environment..."
cd "$INSTALL_DIR"
sudo -u "$SERVICE_USER" python3.10 -m venv venv
sudo -u "$SERVICE_USER" venv/bin/pip install --upgrade pip wheel setuptools

# Install Python dependencies
echo "[7/10] Installing Python dependencies..."
sudo -u "$SERVICE_USER" venv/bin/pip install -r requirements.txt

# Create .env file if not exists
echo "[8/10] Creating environment file..."
if [ ! -f "$INSTALL_DIR/.env" ]; then
    cat > "$INSTALL_DIR/.env" << EOF
# Traffic Signal Optimization - Environment Variables
TRAFFIC_ENV=production
DEVICE_ID=edge_001
LOG_LEVEL=INFO

# MQTT Configuration
MQTT_BROKER=localhost
MQTT_PORT=1883
MQTT_USERNAME=traffic_edge
MQTT_PASSWORD=$(openssl rand -hex 16)

# InfluxDB Configuration
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=my-super-secret-auth-token
INFLUXDB_ORG=traffic-org
INFLUXDB_BUCKET=traffic-metrics

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=traffic-redis-pass
EOF
    chown "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR/.env"
    echo "✓ Environment file created"
else
    echo "✓ Environment file already exists"
fi

# Install systemd service
echo "[9/10] Installing systemd service..."
cp deployment/traffic-edge.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable traffic-edge.service
echo "✓ Systemd service installed"

# Set permissions
echo "[10/10] Setting permissions..."
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
chown -R "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"
chown -R "$SERVICE_USER:$SERVICE_USER" "$DATA_DIR"
chmod -R 755 "$INSTALL_DIR"
chmod -R 755 "$LOG_DIR"
chmod -R 755 "$DATA_DIR"

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit configuration: $INSTALL_DIR/config/intersection_config.yaml"
echo "2. Start infrastructure: cd $INSTALL_DIR && docker-compose up -d"
echo "3. Start edge service: systemctl start traffic-edge"
echo "4. Check status: systemctl status traffic-edge"
echo "5. View logs: journalctl -u traffic-edge -f"
echo ""
echo "Dashboards:"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - InfluxDB: http://localhost:8086"
echo "  - Prometheus: http://localhost:9090"
echo ""
