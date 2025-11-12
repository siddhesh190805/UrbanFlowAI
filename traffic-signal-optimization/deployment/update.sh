#!/bin/bash
# Update script for Traffic Signal Optimization

set -e

INSTALL_DIR="/opt/traffic-signal-optimization/edge"
SERVICE_USER="traffic"
BACKUP_DIR="/opt/traffic-backups/$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Traffic Signal Optimization - Update"
echo "=========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Create backup
echo "[1/5] Creating backup..."
mkdir -p "$BACKUP_DIR"
cp -r "$INSTALL_DIR" "$BACKUP_DIR/"
echo "✓ Backup created: $BACKUP_DIR"

# Stop service
echo "[2/5] Stopping service..."
systemctl stop traffic-edge
echo "✓ Service stopped"

# Update code
echo "[3/5] Updating application..."
cd "$INSTALL_DIR"
git pull origin main
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
echo "✓ Code updated"

# Update dependencies
echo "[4/5] Updating dependencies..."
sudo -u "$SERVICE_USER" venv/bin/pip install --upgrade -r requirements.txt
echo "✓ Dependencies updated"

# Restart service
echo "[5/5] Starting service..."
systemctl start traffic-edge
sleep 3
systemctl status traffic-edge --no-pager
echo "✓ Service restarted"

echo ""
echo "=========================================="
echo "✅ Update Complete!"
echo "=========================================="
echo "Backup location: $BACKUP_DIR"
echo "Check logs: journalctl -u traffic-edge -f"
