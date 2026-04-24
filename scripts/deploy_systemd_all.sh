#!/usr/bin/env bash
# Install / update systemd units for all camera services with tuned defaults.
# Run on the server as a sudoer: sudo bash scripts/deploy_systemd_all.sh
# Restarts every service after writing the unit file.

set -euo pipefail

PROJECT_DIR="/home/aesthetics-lab/50"
USER_NAME="aesthetics-lab"
VENV_GUNICORN="$PROJECT_DIR/.venv/bin/gunicorn"
UNIT_DIR="/etc/systemd/system"

# name -> "port:rtsp_url"
declare -A CAMERAS=(
    ["dha-cam1"]="8182:rtsp://admin:GenIT%%407530@154.57.194.109:554/cam/realmonitor?channel=1&subtype=1"
    ["dha-cam2"]="8186:rtsp://admin:GenIT%%407530@203.99.178.172:554/cam/realmonitor?channel=1&subtype=1"
    ["mm-cam1"]="8187:rtsp://admin:GenIT%%407530@182.184.29.208:554/cam/realmonitor?channel=1&subtype=1"
    ["mm-cam2"]="8188:rtsp://admin:GenIT%%407530@119.63.139.242:554/cam/realmonitor?channel=1&subtype=1"
    ["fsd1"]="8183:rtsp://admin:FSD%%40cam123@115.186.118.99:554/cam/realmonitor?channel=1&subtype=1"
    ["fsd2"]="8184:rtsp://admin:Admin%%40123@115.186.118.100:554/cam/realmonitor?channel=1&subtype=1"
    ["fsd3"]="8185:rtsp://admin:Admin%%40123@115.186.118.101:554/cam/realmonitor?channel=1&subtype=1"
)

# Shared tuned config for all cameras
COMMON_ENV=(
    'Environment="PATH=/home/aesthetics-lab/50/.venv/bin:/usr/bin"'
    'Environment="REID=1"'
    'Environment="YOLO_MODEL=yolov8s.pt"'
    'Environment="YOLO_IMGSZ=640"'
    'Environment="CONF_THRESHOLD=0.25"'
    'Environment="REID_SIM=0.68"'
    'Environment="REID_REVERIFY_MARGIN=0.12"'
    'Environment="MIN_DETECTION_AREA=1500"'
    'Environment="MAX_MISSING_FRAMES=240"'
    'Environment="ABSENCE_TIMEOUT_S=1200"'
)

if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo."
    exit 1
fi

for name in "${!CAMERAS[@]}"; do
    config="${CAMERAS[$name]}"
    port="${config%%:*}"
    rtsp_url="${config#*:}"
    pretty="${name^^}"
    pretty="${pretty//-/_}"

    unit_path="$UNIT_DIR/${name}.service"
    echo "[$name] writing $unit_path (port $port)"

    {
        echo "[Unit]"
        echo "Description=AI Track App - ${pretty} (${port})"
        echo "After=network.target"
        echo ""
        echo "[Service]"
        echo "User=${USER_NAME}"
        echo "WorkingDirectory=${PROJECT_DIR}"
        echo "Environment=\"RTSP_URL=${rtsp_url}\""
        for line in "${COMMON_ENV[@]}"; do
            echo "$line"
        done
        echo "ExecStart=${VENV_GUNICORN} server:app --workers 1 --threads 2 --timeout 600 --bind 0.0.0.0:${port}"
        echo "Restart=always"
        echo "RestartSec=3"
        echo ""
        echo "[Install]"
        echo "WantedBy=multi-user.target"
    } > "$unit_path"

    # Open firewall port (ufw no-op if inactive)
    if command -v ufw >/dev/null 2>&1; then
        ufw allow "${port}/tcp" >/dev/null 2>&1 || true
    fi
done

echo ""
echo "Reloading systemd..."
systemctl daemon-reload

echo ""
echo "Enabling + restarting all services..."
for name in "${!CAMERAS[@]}"; do
    systemctl enable "${name}.service" >/dev/null
    systemctl restart "${name}.service"
    echo "  restarted ${name}"
done

echo ""
echo "Waiting 5s for services to warm up..."
sleep 5

echo ""
echo "Status summary:"
for name in "${!CAMERAS[@]}"; do
    state=$(systemctl is-active "${name}.service" || true)
    echo "  ${name}: ${state}"
done

echo ""
echo "Done. Dashboard: http://$(hostname -I | awk '{print $1}'):8010/login"
