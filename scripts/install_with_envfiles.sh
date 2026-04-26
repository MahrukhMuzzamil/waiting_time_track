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
ETC_DIR="/etc/ai-track"
UNIT_DIR="/etc/systemd/system"

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

# 1) /etc/ai-track + common.env
mkdir -p "$ETC_DIR"
chmod 755 "$ETC_DIR"

echo "Writing $ETC_DIR/common.env (shared tuning)"
cat > "$ETC_DIR/common.env" <<'EOF'
# AI Track App — shared tuning for ALL cameras.
# Edit values here and run:
#   sudo systemctl restart 'dha-cam*' 'mm-cam*' 'fsd*'
# Format: KEY=VALUE, no quotes, no spaces around `=`.

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

# --- Process env ---
PATH=/home/aesthetics-lab/50/.venv/bin:/usr/bin
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

echo ""
echo "Waiting 5s for startup…"
sleep 5

echo ""
echo "Status:"
for name in "${!CAMERAS[@]}"; do
    state=$(systemctl is-active "${name}.service" 2>/dev/null || echo unknown)
    printf "  %-10s %s\n" "${name}:" "${state}"
done

echo ""
echo "Done."
echo ""
echo "To re-tune ALL cameras at once:"
echo "  sudo nano $ETC_DIR/common.env"
echo "  sudo systemctl restart 'dha-cam*' 'mm-cam*' 'fsd*'"
echo ""
echo "To change one camera's RTSP URL:"
echo "  sudo nano $ETC_DIR/<name>.env"
echo "  sudo systemctl restart <name>"
