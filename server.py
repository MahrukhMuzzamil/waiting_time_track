import gc
import logging
import os
import time
from functools import wraps
from typing import Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, stream_with_context, make_response, request
from ultralytics import YOLO
import torch

from main import (
    IoUTracker,
    ReIDMemory,
    format_hms,
    draw_label_with_background,
    open_rtsp_with_fallbacks,
)


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Constrain CPU thread usage for low-memory environments (Render Free)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
try:
    cv2.setNumThreads(1)
except Exception:
    pass
try:
    torch.set_num_threads(1)
except Exception:
    pass


# Load environment
RTSP_URL = os.environ.get("RTSP_URL", "").strip()
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.4"))
REID_ENABLED = os.environ.get("REID", "1").strip() not in ("0", "false", "False", "")
REID_SIM = float(os.environ.get("REID_SIM", "0.62"))
# ReID TTL: how long to remember a person (default 1 hour = 3600 seconds)
REID_TTL = float(os.environ.get("REID_TTL", "3600"))
# IoU tracker: how many frames before dropping a track (default 90 frames @ ~8fps = ~11 seconds)
MAX_MISSING_FRAMES = int(os.environ.get("MAX_MISSING_FRAMES", "90"))


# Global state
_yolo_model: Optional[YOLO] = None  # lazy-loaded when needed
# Use more lenient tracking parameters to reduce ID churn
tracker = IoUTracker(
    max_missing_frames=MAX_MISSING_FRAMES,
    iou_match_threshold=0.10,  # Very lenient IoU matching
    centroid_fallback_dist=200.0,  # Larger fallback distance for centroid matching
)
# ReID is expensive - only use it for new tracks, not every frame
effective_reid_sim = min(REID_SIM, 0.50)  # Lower threshold for better matching
reid_memory: Optional[ReIDMemory] = ReIDMemory(
    similarity_threshold=effective_reid_sim,
    ttl_seconds=REID_TTL,
    max_embeddings_per_person=5,
    embedding_update_interval=10.0,  # Less frequent updates to reduce CPU
) if REID_ENABLED else None
track_to_label: Dict[int, int] = {}
# Store label -> reid_ids mapping for recovery
label_to_reid_ids: Dict[int, set] = {}
next_label_id: int = 1
# person_id -> earliest start time (ReID-aware or tracker-based)
person_start_times: Dict[int, float] = {}
person_last_seen: Dict[int, float] = {}
# Track which tids we've already done ReID on (to avoid repeated expensive calls)
reid_done_for_tid: Dict[int, float] = {}  # tid -> last_reid_time

logger.info("ReID enabled: %s", bool(reid_memory))

# Shared capture for lightweight snapshot endpoint
_snap_cap: Optional[cv2.VideoCapture] = None


def _check_basic_auth(auth) -> bool:
    """Validate HTTP Basic credentials against env vars.

    This is intentionally very simple: a single shared username/password
    per service instance. Configure via:
      BASIC_AUTH_USER, BASIC_AUTH_PASS
    """
    if auth is None or not auth.username or not auth.password:
        return False
    expected_user = os.environ.get("BASIC_AUTH_USER", "").strip()
    expected_pass = os.environ.get("BASIC_AUTH_PASS", "").strip()
    if not expected_user or not expected_pass:
        # If not configured, do not enforce auth
        return True
    return auth.username == expected_user and auth.password == expected_pass


def _auth_required(func):
    """Decorator to require HTTP Basic auth on a route when configured."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        auth = request.authorization
        if not _check_basic_auth(auth):
            return Response(
                "Authentication required",
                401,
                {"WWW-Authenticate": 'Basic realm="AI Camera"'},
            )
        return func(*args, **kwargs)

    return wrapper


def open_capture() -> cv2.VideoCapture:
    # #region agent log
    import json as _json; _log_path = '/home/aesthetics-lab/50/.cursor/debug.log'
    def _debug_log(loc, msg, data, hyp=''):
        with open(_log_path, 'a') as _f: _f.write(_json.dumps({"location": loc, "message": msg, "data": data, "timestamp": int(time.time()*1000), "hypothesisId": hyp}) + '\n')
    # #endregion
    if not RTSP_URL or not RTSP_URL.startswith("rtsp://"):
        raise RuntimeError("RTSP_URL env var is required and must start with rtsp://")
    # #region agent log
    _debug_log("server.py:open_capture", "RTSP_URL from env", {"rtsp_url": RTSP_URL, "port": os.environ.get("PORT", "unknown")}, "A,E")
    # #endregion
    logger.info("Opening RTSP capture for %s", RTSP_URL)
    cap = open_rtsp_with_fallbacks(RTSP_URL)
    if cap is None:
        # #region agent log
        _debug_log("server.py:open_capture", "RTSP connection FAILED", {"rtsp_url": RTSP_URL}, "A,B,C,D")
        # #endregion
        logger.error("Failed to open RTSP source after trying all fallback URLs. RTSP_URL=%s", RTSP_URL)
        raise RuntimeError(f"Unable to open RTSP source from RTSP_URL: {RTSP_URL}")
    # #region agent log
    _debug_log("server.py:open_capture", "RTSP connection SUCCESS", {"rtsp_url": RTSP_URL}, "A,B,C,D")
    # #endregion
    logger.info("Successfully opened RTSP capture")
    return cap


def _get_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def frame_generator_raw():
    cap = None
    last = time.time()
    target_interval = 1.0 / 8.0
    last_gc_time = time.time()
    consecutive_errors = 0
    frame_count = 0
    GC_INTERVAL = 120.0  # Raw stream is lighter, GC less often

    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _placeholder_frame("Connecting…") + b"\r\n")

    while True:
        try:
            if cap is None:
                logger.info("frame_generator_raw acquiring capture")
                cap = open_capture()
                consecutive_errors = 0
            ok, frame = cap.read()
            if not ok or frame is None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _placeholder_frame("Reconnecting…") + b"\r\n")
                time.sleep(0.8)
                continue

            h, w = frame.shape[:2]
            if w > 640:
                nh = int(h * 640 / w)
                frame = cv2.resize(frame, (640, nh), interpolation=cv2.INTER_AREA)

            jpg = _encode_jpeg(frame) or _placeholder_frame("Encoding…")
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

            now = time.time()
            dt = now - last
            if dt < target_interval:
                time.sleep(target_interval - dt)
            last = time.time()
            
            frame_count += 1
            # Periodic garbage collection
            if now - last_gc_time > GC_INTERVAL:
                gc.collect()
                last_gc_time = now
                
        except Exception:
            logger.exception("frame_generator_raw loop error")
            consecutive_errors += 1
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
            gc.collect()  # Clean up on error
            backoff = min(0.8 * (2 ** min(consecutive_errors, 4)), 10.0)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _placeholder_frame("Error…") + b"\r\n")
            time.sleep(backoff)


def _encode_jpeg(img: np.ndarray) -> Optional[bytes]:
    ok, buff = cv2.imencode('.jpg', img)
    return buff.tobytes() if ok else None


def _placeholder_frame(text: str = "Starting…") -> bytes:
    canvas = np.full((360, 640, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
    data = _encode_jpeg(canvas)
    return data or b""


def frame_generator():
    global next_label_id
    cap = None
    fps_smoother = None
    last_time = time.time()
    frame_idx = 0
    last_gc_time = time.time()
    consecutive_errors = 0
    GC_INTERVAL = 60.0  # Run garbage collection every 60 seconds
    MAX_CONSECUTIVE_ERRORS = 10  # Give up temporarily after this many errors
    RECONNECT_BACKOFF_BASE = 1.0  # Base backoff time for reconnection

    # Send a quick placeholder so client receives 200 immediately
    first = _placeholder_frame("Connecting to camera…")
    if first:
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + first + b"\r\n")

    while True:
        try:
            if cap is None:
                logger.info("frame_generator acquiring capture")
                try:
                    cap = open_capture()
                    consecutive_errors = 0  # Reset on successful connection
                    logger.info("Successfully acquired RTSP capture")
                except Exception as e:
                    logger.error("Failed to open capture: %s", e, exc_info=True)
                    consecutive_errors += 1
                    backoff_time = min(RECONNECT_BACKOFF_BASE * (2 ** min(consecutive_errors, 5)), 30.0)
                    keepalive = _placeholder_frame(f"Connection error, retrying in {int(backoff_time)}s…")
                    if keepalive:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + keepalive + b"\r\n")
                    time.sleep(backoff_time)
                    continue

            ret, frame = cap.read()
            if not ret or frame is None:
                # reconnect, but keep client alive with a placeholder frame
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                keepalive = _placeholder_frame("Reconnecting…")
                if keepalive:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + keepalive + b"\r\n")
                time.sleep(1.0)
                continue

            now_s = time.time()
            dt = now_s - last_time
            last_time = now_s
            if fps_smoother is None:
                fps_smoother = 1.0 / max(dt, 1e-6)
            else:
                fps_smoother = 0.9 * fps_smoother + 0.1 * (1.0 / max(dt, 1e-6))

            # YOLO inference for persons (reduced memory)
            # Downscale frame slightly to reduce memory/CPU
            infer_start = time.time()
            ih, iw = frame.shape[:2]
            scale = 640 / max(iw, ih)
            if scale < 1.0:
                new_w = max(320, int(iw * scale))
                new_h = max(240, int(ih * scale))
                frame_infer = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                ratio_x = iw / new_w
                ratio_y = ih / new_h
            else:
                frame_infer = frame
                ratio_x = 1.0
                ratio_y = 1.0

            with torch.no_grad():
                results = _get_model().predict(
                    source=frame_infer,
                    imgsz=480,
                    conf=CONF_THRESHOLD,
                    classes=[0],
                    verbose=False,
                    device="cpu",
                )

            boxes_xyxy: List[np.ndarray] = []
            if results and len(results) > 0:
                r0 = results[0]
                if r0.boxes is not None and len(r0.boxes) > 0:
                    b = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
                    boxes_xyxy = [bb for bb in b]

            detections = np.array(boxes_xyxy, dtype=np.float32) if boxes_xyxy else np.zeros((0, 4), dtype=np.float32)
            if detections.size > 0 and (ratio_x != 1.0 or ratio_y != 1.0):
                detections[:, [0, 2]] *= ratio_x
                detections[:, [1, 3]] *= ratio_y

            frame_idx += 1
            tracked: Dict[int, np.ndarray] = tracker.step(frame_idx, detections, now_s)
            active_labels = {label for tid_active, label in track_to_label.items() if tid_active in tracker._tracks}
            frame_labels: set[int] = set()

            # Clear ReID frame cache at start of each frame to prevent double-matching
            if reid_memory is not None:
                reid_memory.clear_frame_cache(now_s)

            for tid, bbox in tracked.items():
                x1, y1, x2, y2 = bbox.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

                # Check if this track already has a label
                label_id = track_to_label.get(tid)

                if label_id is None:
                    # NEW TRACK - this is the only time we run ReID (expensive)
                    reid_suggested_label: Optional[int] = None
                    current_reid_id: Optional[int] = None
                    
                    if reid_memory is not None:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
                            current_reid_id = reid_memory.assign_person_id(crop, now_s)
                            reid_done_for_tid[tid] = now_s
                            # Check if this reid_id was previously associated with a label
                            if current_reid_id is not None:
                                for lbl, reid_set in label_to_reid_ids.items():
                                    if current_reid_id in reid_set and lbl not in frame_labels:
                                        reid_suggested_label = lbl
                                        break
                    
                    if reid_suggested_label is not None:
                        # ReID matched an existing person - reuse their label
                        label_id = reid_suggested_label
                    else:
                        # Truly new person - assign new label
                        label_id = next_label_id
                        next_label_id += 1
                    
                    track_to_label[tid] = label_id
                    
                    # Store ReID association for future recovery
                    if current_reid_id is not None:
                        if label_id not in label_to_reid_ids:
                            label_to_reid_ids[label_id] = set()
                        label_to_reid_ids[label_id].add(current_reid_id)
                
                # Once a track has a label, KEEP IT (don't let ReID override)
                # This prevents ID flickering
                
                frame_labels.add(label_id)
                active_labels.add(label_id)

                if label_id not in person_start_times:
                    start_time_s = tracker.get_track_start_time(tid) or now_s
                    person_start_times[label_id] = start_time_s
                start_time_s = person_start_times[label_id]
                wait_s = now_s - start_time_s
                time_text = f"ID {label_id} · {format_hms(wait_s)}"
                person_last_seen[label_id] = now_s
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Track tid=%d assigned person_id=%d bbox=(%d,%d,%d,%d) wait=%.1fs",
                        tid,
                        label_id,
                        x1,
                        y1,
                        x2,
                        y2,
                        wait_s,
                    )
                label_x = int((x1 + x2) / 2)
                label_y = max(0, y1 - 6)
                draw_label_with_background(frame, time_text, (label_x, label_y), font_scale=0.6, bg_color=(50, 50, 50))

            # Overlay FPS
            fps_text = f"FPS: {fps_smoother:.1f}"
            cv2.putText(frame, fps_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode JPEG
            jpg = _encode_jpeg(frame)
            if jpg is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            else:
                keepalive = _placeholder_frame("Encoding error…")
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + keepalive + b"\r\n")

            logger.debug(
                "frame_generator iteration completed in %.2fs (fps_smoother=%.2f)",
                time.time() - infer_start,
                fps_smoother or 0.0,
            )

            # Cleanup stale person entries (match ReID TTL, default 1 hour)
            stale_cutoff = now_s - REID_TTL
            stale_ids = [pid for pid, ts in person_last_seen.items() if ts < stale_cutoff]
            for pid in stale_ids:
                person_last_seen.pop(pid, None)
                person_start_times.pop(pid, None)
                label_to_reid_ids.pop(pid, None)

            current_tids = set(tracker._tracks.keys())
            for tid_old in list(track_to_label.keys()):
                if tid_old not in current_tids:
                    track_to_label.pop(tid_old, None)
                    reid_done_for_tid.pop(tid_old, None)

            # Periodic garbage collection to prevent memory buildup
            if now_s - last_gc_time > GC_INTERVAL:
                gc.collect()
                last_gc_time = now_s
                logger.debug("Periodic garbage collection completed")

        except BaseException:
            logger.exception("frame_generator loop error")
            consecutive_errors += 1
            # Backoff on errors
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
            
            # Exponential backoff with cap
            backoff_time = min(RECONNECT_BACKOFF_BASE * (2 ** min(consecutive_errors, 5)), 30.0)
            
            # Force garbage collection on error to free memory
            gc.collect()
            
            # keep the stream alive with a placeholder frame
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                keepalive = _placeholder_frame(f"Too many errors, waiting {int(backoff_time)}s…")
            else:
                keepalive = _placeholder_frame("Reconnecting…")
            if keepalive:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + keepalive + b"\r\n")
            time.sleep(backoff_time)


INDEX_HTML = """
<!doctype html>
<title>AI Track App</title>
<style>
  body{font-family:system-ui;margin:0;background:#111;color:#eee}
  header{padding:12px 16px;background:#1b1b1b;border-bottom:1px solid #222}
  main{padding:12px}
  img{max-width:100%;height:auto;border:1px solid #222}
</style>
<header>
  <h3>AI Track App</h3>
  <div>Live Camera Stream</div>
  <small>AI-powered person tracking &amp; re-identification</small>
  <div><a href="/video_ai">/video_ai</a> · <a href="/snapshot">/snapshot</a></div>
  </header>
<main>
  <div>
    <img id="snap" src="/snapshot" alt="snapshot">
  </div>
  <script>
    const img = document.getElementById('snap');
    setInterval(() => {
      const ts = Date.now();
      img.src = '/snapshot?ts=' + ts;
    }, 1000);
  </script>
</main>
"""


@app.get("/")
@_auth_required
def index():
    return render_template_string(INDEX_HTML)


@app.get("/video")
@_auth_required
def video():
    resp = Response(
        stream_with_context(frame_generator_raw()),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )
    # Proxy/CDN friendly headers for long‑lived MJPEG streams
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["Connection"] = "keep-alive"
    # Some proxies honor this to disable buffering; harmless elsewhere
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.get("/video_ai")
@_auth_required
def video_ai():
    resp = Response(
        stream_with_context(frame_generator()),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["Connection"] = "keep-alive"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.get("/snapshot")
@_auth_required
def snapshot():
    global _snap_cap
    try:
        if _snap_cap is None or not _snap_cap.isOpened():
            _snap_cap = open_capture()

        ok, frame = _snap_cap.read()
        if not ok or frame is None:
            try:
                _snap_cap.release()
            except Exception:
                pass
            _snap_cap = None
            data = _placeholder_frame("Reconnecting…")
        else:
            data = _encode_jpeg(frame) or _placeholder_frame("Encoding…")
    except Exception:
        data = _placeholder_frame("Error…")

    resp = make_response(data)
    resp.headers['Content-Type'] = 'image/jpeg'
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)

