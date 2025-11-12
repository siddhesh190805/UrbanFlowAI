#!/bin/bash
# Uninstall script

set -e

echo "=========================================="
echo "Traffic Signal Optimization - Uninstall"
echo "=========================================="

if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root"
    exit 1
fi

read -p "This will remove all traffic system files. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Stop and disable service
echo "[1/4] Stopping service..."
systemctl stop traffic-edge || true
systemctl disable traffic-edge || true
rm -f /etc/systemd/system/traffic-edge.service
systemctl daemon-reload

# Stop Docker containers
echo "[2/4] Stopping Docker containers..."
cd /opt/traffic-signal-optimization/edge
docker-compose down -v || true

# Remove files
echo "[3/4] Removing files..."
rm -rf /opt/traffic-signal-optimization
rm -rf /var/log/traffic
rm -rf /var/traffic

# Remove user
echo "[4/4] Removing service user..."
userdel -r traffic || true

echo ""
echo "✅ Uninstall complete"
