#!/usr/bin/env bash
# Install all camera services using a shared EnvironmentFile pattern.
#
# After this runs, you tune ALL cameras at once by editing one file:
#   sudo nano /etc/ai-track/common.env
#   sudo systemctl restart 'dha-cam*' 'mm-cam*' 'fsd*'
#
# Layout it creates:
#   /etc/ai-track/common.env       — shared tuning (single source of truth)
#   /etc/ai-track/<cam>.env        — per-camera RTSP_URL + PORT only
#   /etc/systemd/system/<cam>.service — references both env files
#
# Run as root: sudo bash scripts/install_with_envfiles.sh

set -euo pipefail

PROJECT_DIR="/home/aesthetics-lab/50"
USER_NAME="aesthetics-lab"
VENV_GUNICORN="$PROJECT_DIR/.venv/bin/gunicorn"
VENV_PIP="$PROJECT_DIR/.venv/bin/pip"
ETC_DIR="/etc/ai-track"
UNIT_DIR="/etc/systemd/system"
DATA_DIR="/var/lib/ai-track"
DB_PATH="$DATA_DIR/waittime.db"
LOGIN_GATEWAY_PORT=8010

# name -> "port:rtsp_url"
declare -A CAMERAS=(
    ["dha-cam1"]="8182:rtsp://admin:GenIT%407530@154.57.194.109:554/cam/realmonitor?channel=1&subtype=1"
    ["dha-cam2"]="8186:rtsp://admin:GenIT%407530@203.99.178.172:554/cam/realmonitor?channel=1&subtype=1"
    ["mm-cam1"]="8187:rtsp://admin:GenIT%407530@182.184.29.208:554/cam/realmonitor?channel=1&subtype=1"
    ["mm-cam2"]="8188:rtsp://admin:GenIT%407530@119.63.139.242:554/cam/realmonitor?channel=1&subtype=1"
    ["fsd1"]="8183:rtsp://admin:FSD%40cam123@115.186.118.99:554/cam/realmonitor?channel=1&subtype=1"
    ["fsd2"]="8184:rtsp://admin:Admin%40123@115.186.118.100:554/cam/realmonitor?channel=1&subtype=1"
    ["fsd3"]="8185:rtsp://admin:Admin%40123@115.186.118.101:554/cam/realmonitor?channel=1&subtype=1"
)

if [[ $EUID -ne 0 ]]; then
    echo "Please run with sudo."
    exit 1
fi

# 0) Make sure openpyxl is installed (added for analytics/Excel exports)
if [[ -x "$VENV_PIP" ]]; then
    sudo -u "$USER_NAME" "$VENV_PIP" install --quiet 'openpyxl==3.1.5' || true
fi

# 0b) Shared analytics directory writable by camera processes + login gateway
mkdir -p "$DATA_DIR"
chown -R "$USER_NAME:$USER_NAME" "$DATA_DIR"
chmod 775 "$DATA_DIR"

# 1) /etc/ai-track + common.env
mkdir -p "$ETC_DIR"
chmod 755 "$ETC_DIR"

echo "Writing $ETC_DIR/common.env (shared tuning)"
cat > "$ETC_DIR/common.env" <<EOF
# AI Track App — shared tuning for ALL cameras.
# Edit values here and run:
#   sudo systemctl restart 'dha-cam*' 'mm-cam*' 'fsd*'
# Format: KEY=VALUE, no quotes, no spaces around \`=\`.

# --- YOLO detection ---
YOLO_MODEL=yolov8s.pt
YOLO_IMGSZ=640
CONF_THRESHOLD=0.25
MIN_DETECTION_AREA=1500

# --- Multi-object tracker ---
USE_YOLO_TRACK=1
YOLO_TRACKER=bytetrack.yaml

# --- ReID + pause/resume timer ---
REID=1
REID_SIM=0.70
REID_REVERIFY_MARGIN=0.12
ABSENCE_TIMEOUT_S=1200

# --- Wait-time analytics ---
WAIT_LOGGING=1
DB_PATH=$DB_PATH
MIN_LOGGED_WAIT_S=5

# --- Process env ---
PATH=$PROJECT_DIR/.venv/bin:/usr/bin
EOF
chmod 644 "$ETC_DIR/common.env"

# 2) Per-camera env files (only the bits that differ)
for name in "${!CAMERAS[@]}"; do
    config="${CAMERAS[$name]}"
    port="${config%%:*}"
    rtsp_url="${config#*:}"

    echo "Writing $ETC_DIR/${name}.env"
    cat > "$ETC_DIR/${name}.env" <<EOF
# Per-camera config for ${name^^}.
# Shared tuning (model, ReID, etc.) lives in common.env.
RTSP_URL=${rtsp_url}
PORT=${port}
CAMERA_NAME=${name}
EOF
    chmod 644 "$ETC_DIR/${name}.env"
done

# 3) Service unit files (all identical structure, both env files referenced)
for name in "${!CAMERAS[@]}"; do
    pretty="${name^^}"
    pretty="${pretty//-/_}"

    echo "Writing $UNIT_DIR/${name}.service"
    cat > "$UNIT_DIR/${name}.service" <<EOF
[Unit]
Description=AI Track App - ${pretty}
After=network.target

[Service]
User=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}
EnvironmentFile=${ETC_DIR}/common.env
EnvironmentFile=${ETC_DIR}/${name}.env
ExecStart=${VENV_GUNICORN} server:app --workers 1 --threads 2 --timeout 600 --bind 0.0.0.0:\${PORT}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

    # Open firewall (no-op if ufw is inactive or not installed)
    if command -v ufw >/dev/null 2>&1; then
        port="${CAMERAS[$name]%%:*}"
        ufw allow "${port}/tcp" >/dev/null 2>&1 || true
    fi
done

# 3b) Login gateway service (port 8010) — serves dashboard + reports + Excel exports
echo "Writing $UNIT_DIR/login-gateway.service"
cat > "$UNIT_DIR/login-gateway.service" <<EOF
[Unit]
Description=AI Track App - Login Gateway + Reports Dashboard
After=network.target

[Service]
User=${USER_NAME}
WorkingDirectory=${PROJECT_DIR}
EnvironmentFile=${ETC_DIR}/common.env
Environment="FLASK_SECRET_KEY="
ExecStart=${VENV_GUNICORN} login_gateway:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${LOGIN_GATEWAY_PORT}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

if command -v ufw >/dev/null 2>&1; then
    ufw allow "${LOGIN_GATEWAY_PORT}/tcp" >/dev/null 2>&1 || true
fi

# 4) Reload + enable + restart
echo ""
echo "Reloading systemd…"
systemctl daemon-reload

echo ""
echo "Enabling + restarting services…"
for name in "${!CAMERAS[@]}"; do
    systemctl enable "${name}.service" >/dev/null 2>&1
    systemctl restart "${name}.service"
    echo "  restarted ${name}"
done

systemctl enable login-gateway.service >/dev/null 2>&1
systemctl restart login-gateway.service
echo "  restarted login-gateway"

echo ""
echo "Waiting 5s for startup…"
sleep 5

echo ""
echo "Status:"
for name in "${!CAMERAS[@]}"; do
    state=$(systemctl is-active "${name}.service" 2>/dev/null || echo unknown)
    printf "  %-15s %s\n" "${name}:" "${state}"
done
state=$(systemctl is-active login-gateway.service 2>/dev/null || echo unknown)
printf "  %-15s %s\n" "login-gateway:" "${state}"

echo ""
echo "Done."
echo ""
LAN_IP=$(hostname -I | awk '{print $1}')
echo "Dashboard:   http://${LAN_IP}:${LOGIN_GATEWAY_PORT}/login"
echo "Reports:     http://${LAN_IP}:${LOGIN_GATEWAY_PORT}/reports"
echo "Analytics DB: $DB_PATH"
echo ""
echo "To re-tune ALL cameras at once:"
echo "  sudo nano $ETC_DIR/common.env"
echo "  sudo systemctl restart 'dha-cam*' 'mm-cam*' 'fsd*'"
echo ""
echo "To change one camera's RTSP URL:"
echo "  sudo nano $ETC_DIR/<name>.env"
echo "  sudo systemctl restart <name>"
