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
    PersonRegistry,
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
REID_SIM = float(os.environ.get("REID_SIM", "0.72"))
# Absence window: how long a person's timer is kept alive after they leave the frame.
# If they return within this window, the timer resumes. Default 20 minutes.
ABSENCE_TIMEOUT_S = float(os.environ.get("ABSENCE_TIMEOUT_S", "1200"))
# IoU tracker: how many frames before dropping a track (default 60 frames @ ~8fps ≈ 7.5s).
# Long-term re-identification is now handled by PersonRegistry+ReID, so this can be short.
MAX_MISSING_FRAMES = int(os.environ.get("MAX_MISSING_FRAMES", "60"))
# Reject detections smaller than this area (pixels) — filters noise / distant false positives.
MIN_DETECTION_AREA = int(os.environ.get("MIN_DETECTION_AREA", "2500"))
# How often to re-run ReID on an established track to catch tracker swaps.
REID_REVERIFY_INTERVAL_S = float(os.environ.get("REID_REVERIFY_INTERVAL_S", "4.0"))


# Global state
_yolo_model: Optional[YOLO] = None  # lazy-loaded when needed
tracker = IoUTracker(
    max_missing_frames=MAX_MISSING_FRAMES,
    iou_match_threshold=0.30,
    centroid_fallback_dist=120.0,
)
reid_memory: Optional[ReIDMemory] = ReIDMemory(
    similarity_threshold=REID_SIM,
    ttl_seconds=ABSENCE_TIMEOUT_S,
    max_embeddings_per_person=5,
    embedding_update_interval=10.0,
) if REID_ENABLED else None
registry = PersonRegistry(
    reid=reid_memory,
    absence_timeout_s=ABSENCE_TIMEOUT_S,
    reid_reverify_interval_s=REID_REVERIFY_INTERVAL_S,
)

logger.info(
    "ReID enabled: %s, sim=%.2f, absence=%.0fs, max_missing=%d, min_area=%d",
    bool(reid_memory), REID_SIM, ABSENCE_TIMEOUT_S, MAX_MISSING_FRAMES, MIN_DETECTION_AREA,
)

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

            # Filter tiny/noise detections — unreliable for both tracking and ReID
            if detections.size > 0 and MIN_DETECTION_AREA > 0:
                areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
                detections = detections[areas >= MIN_DETECTION_AREA]

            frame_idx += 1
            tracked: Dict[int, np.ndarray] = tracker.step(frame_idx, detections, now_s)

            # PersonRegistry owns label assignment, ReID, and pause/resume timing
            visible = registry.update(frame, tracked, now_s, dt)

            # Draw boxes + labels using the registry's accumulated wait times
            tid_to_bbox = tracked
            for label_id, rec in visible.items():
                tid = rec.current_tid
                if tid is None or tid not in tid_to_bbox:
                    continue
                bbox = tid_to_bbox[tid]
                x1, y1, x2, y2 = bbox.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                time_text = f"ID {label_id} · {format_hms(rec.accumulated_wait_s)}"
                label_x = int((x1 + x2) / 2)
                label_y = max(0, y1 - 6)
                draw_label_with_background(frame, time_text, (label_x, label_y), font_scale=0.6, bg_color=(50, 50, 50))

            # Overlay FPS + active/absent counts for quick eyeballing
            all_records = registry.snapshot()
            absent_count = sum(1 for r in all_records.values() if not r.is_visible)
            hud_text = f"FPS: {fps_smoother:.1f}  live: {len(visible)}  waiting (absent): {absent_count}"
            cv2.putText(frame, hud_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

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

            # PersonRegistry handles its own cleanup (absence timeout + tid unbind)

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

