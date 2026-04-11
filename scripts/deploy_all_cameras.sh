#!/usr/bin/env bash
# Deployment script for all camera services
# This script stops existing services, updates code, and restarts all cameras

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Camera configurations: name, port, RTSP URL
# Format: "port:rtsp_url"
# Note: @ in password must be URL-encoded as %40
declare -A CAMERAS=(
    ["DHA_CAM1"]="8182:rtsp://admin:GenIT%407530@154.57.194.109:554/cam/realmonitor?channel=1&subtype=1"
    ["DHA_CAM2"]="8186:rtsp://admin:GenIT%407530@203.99.178.172:554/cam/realmonitor?channel=1&subtype=1"
    ["MM_CAM1"]="8187:rtsp://admin:GenIT%407530@182.184.29.208:554/cam/realmonitor?channel=1&subtype=1"
    ["MM_CAM2"]="8188:rtsp://admin:GenIT%407530@119.63.139.242:554/cam/realmonitor?channel=1&subtype=1"
    ["FSD1"]="8183:rtsp://admin:FSD%40cam123@115.186.118.99:554/cam/realmonitor?channel=1&subtype=1"
    ["FSD2"]="8184:rtsp://admin:Admin%40123@115.186.118.100:554/cam/realmonitor?channel=1&subtype=1"
    ["FSD3"]="8185:rtsp://admin:Admin%40123@115.186.118.101:554/cam/realmonitor?channel=1&subtype=1"
)

# Login gateway port
LOGIN_GATEWAY_PORT=8010

# Environment variables
CONF_THRESHOLD="${CONF_THRESHOLD:-0.4}"
REID="${REID:-1}"
REID_SIM="${REID_SIM:-0.62}"
REID_TTL="${REID_TTL:-3600}"
MAX_MISSING_FRAMES="${MAX_MISSING_FRAMES:-90}"

# Gunicorn settings
WORKERS="${WORKERS:-1}"
THREADS="${THREADS:-2}"
TIMEOUT="${TIMEOUT:-600}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Function to find and kill processes on a port
kill_port() {
    local port=$1
    local pids=$(lsof -ti:"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log_info "Stopping processes on port $port (PIDs: $pids)"
        kill $pids 2>/dev/null || true
        sleep 2
        # Force kill if still running
        local still_running=$(lsof -ti:"$port" 2>/dev/null || true)
        if [ -n "$still_running" ]; then
            log_warn "Force killing processes on port $port"
            kill -9 $still_running 2>/dev/null || true
            sleep 1
        fi
    else
        log_info "No processes found on port $port"
    fi
}

# Function to start a camera service
start_camera() {
    local name=$1
    local config=$2
    IFS=':' read -r port rtsp_url <<< "$config"
    
    log_info "Starting $name camera on port $port"
    log_info "RTSP URL: $rtsp_url"
    
    # Activate virtual environment
    if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
        source "$PROJECT_DIR/.venv/bin/activate"
    elif [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
        source "$PROJECT_DIR/venv/bin/activate"
    else
        log_error "Virtual environment not found!"
        return 1
    fi
    
    # Set environment variables
    export RTSP_URL="$rtsp_url"
    export CONF_THRESHOLD="$CONF_THRESHOLD"
    export REID="$REID"
    export REID_SIM="$REID_SIM"
    export REID_TTL="$REID_TTL"
    export MAX_MISSING_FRAMES="$MAX_MISSING_FRAMES"
    export PORT="$port"
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    
    # Start gunicorn in background with nohup
    local log_file="$PROJECT_DIR/${name,,}.log"
    nohup gunicorn server:app \
        --workers "$WORKERS" \
        --threads "$THREADS" \
        --timeout "$TIMEOUT" \
        --bind "0.0.0.0:$port" \
        --worker-class gthread \
        --access-logfile '-' \
        --error-logfile '-' \
        > "$log_file" 2>&1 &
    
    local pid=$!
    log_info "$name camera started with PID $pid (log: $log_file)"
    sleep 3
    
    # Check if process is still running
    if kill -0 $pid 2>/dev/null; then
        log_info "$name camera is running (PID: $pid)"
    else
        log_error "$name camera failed to start. Check log: $log_file"
        tail -20 "$log_file"
        return 1
    fi
}

# Function to start the login gateway
start_login_gateway() {
    log_info "Starting login gateway on port $LOGIN_GATEWAY_PORT"
    
    # Activate virtual environment
    if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
        source "$PROJECT_DIR/.venv/bin/activate"
    elif [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
        source "$PROJECT_DIR/venv/bin/activate"
    fi
    
    local log_file="$PROJECT_DIR/login_gateway.log"
    nohup gunicorn login_gateway:app \
        --workers 1 \
        --threads 2 \
        --timeout 120 \
        --bind "0.0.0.0:$LOGIN_GATEWAY_PORT" \
        > "$log_file" 2>&1 &
    
    local pid=$!
    log_info "Login gateway started with PID $pid (log: $log_file)"
    sleep 2
    
    if kill -0 $pid 2>/dev/null; then
        log_info "Login gateway is running"
    else
        log_error "Login gateway failed to start. Check log: $log_file"
        tail -10 "$log_file"
        return 1
    fi
}

# Main deployment process
main() {
    log_info "Starting deployment of all camera services"
    log_info "Project directory: $PROJECT_DIR"
    
    # Stop all existing services
    log_info "Stopping existing services..."
    kill_port "$LOGIN_GATEWAY_PORT"
    for name in "${!CAMERAS[@]}"; do
        IFS=':' read -r port rtsp_url <<< "${CAMERAS[$name]}"
        kill_port "$port"
    done
    
    # Wait a bit for ports to be released
    sleep 2
    
    # Start login gateway
    start_login_gateway
    
    # Start all cameras
    log_info "Starting all camera services..."
    local failed=0
    for name in "${!CAMERAS[@]}"; do
        if ! start_camera "$name" "${CAMERAS[$name]}"; then
            log_error "Failed to start $name camera"
            failed=$((failed + 1))
        fi
    done
    
    # Summary
    echo ""
    log_info "Deployment complete!"
    if [ $failed -eq 0 ]; then
        log_info "All cameras started successfully"
        echo ""
        log_info "Camera status:"
        for name in "${!CAMERAS[@]}"; do
            IFS=':' read -r port rtsp_url <<< "${CAMERAS[$name]}"
            local pids=$(lsof -ti:"$port" 2>/dev/null || true)
            if [ -n "$pids" ]; then
                log_info "  $name (port $port): RUNNING (PIDs: $pids)"
            else
                log_error "  $name (port $port): NOT RUNNING"
            fi
        done
    else
        log_error "$failed camera(s) failed to start"
        return 1
    fi
    
    echo ""
    local server_ip=$(hostname -I | awk '{print $1}')
    log_info "Dashboard: http://${server_ip}:${LOGIN_GATEWAY_PORT}/login"
    echo ""
    log_info "Camera URLs:"
    for name in "${!CAMERAS[@]}"; do
        IFS=':' read -r port rtsp_url <<< "${CAMERAS[$name]}"
        echo "  $name: http://${server_ip}:$port/video_ai"
    done
}

# Run main function
main "$@"
