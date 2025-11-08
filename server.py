import logging
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, stream_with_context, make_response
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


# Global state
_yolo_model: Optional[YOLO] = None  # lazy-loaded when needed
tracker = IoUTracker(max_missing_frames=30, iou_match_threshold=0.3)
reid_memory: Optional[ReIDMemory] = ReIDMemory(similarity_threshold=REID_SIM) if REID_ENABLED else None
# person_id -> earliest start time (ReID-aware)
person_start_times: Dict[int, float] = {}
person_last_seen: Dict[int, float] = {}

logger.info("ReID enabled: %s", bool(reid_memory))

# Shared capture for lightweight snapshot endpoint
_snap_cap: Optional[cv2.VideoCapture] = None


def open_capture() -> cv2.VideoCapture:
    if not RTSP_URL or not RTSP_URL.startswith("rtsp://"):
        raise RuntimeError("RTSP_URL env var is required and must start with rtsp://")
    logger.info("Opening RTSP capture for %s", RTSP_URL)
    cap = open_rtsp_with_fallbacks(RTSP_URL)
    if cap is None:
        raise RuntimeError("Unable to open RTSP source from RTSP_URL")
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

    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _placeholder_frame("Connecting…") + b"\r\n")

    while True:
        try:
            if cap is None:
                logger.info("frame_generator_raw acquiring capture")
                cap = open_capture()
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
        except Exception:
            logger.exception("frame_generator_raw loop error")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _placeholder_frame("Error…") + b"\r\n")
            time.sleep(0.8)


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

    # Send a quick placeholder so client receives 200 immediately
    first = _placeholder_frame("Connecting to camera…")
    if first:
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + first + b"\r\n")

    while True:
        try:
            if cap is None:
                logger.info("frame_generator acquiring capture")
                cap = open_capture()

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

            for tid, bbox in tracked.items():
                x1, y1, x2, y2 = bbox.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

                if reid_memory is not None:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                        person_id = reid_memory.assign_person_id(crop, now_s)
                    else:
                        person_id = tid
                else:
                    person_id = tid

                # Keep earliest sighting per person ID
                if person_id not in person_start_times:
                    start_time_s = tracker.get_track_start_time(tid) or now_s
                    person_start_times[person_id] = start_time_s
                start_time_s = person_start_times[person_id]
                wait_s = now_s - start_time_s
                time_text = f"ID {person_id} · {format_hms(wait_s)}"
                person_last_seen[person_id] = now_s
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Track tid=%d assigned person_id=%d bbox=(%d,%d,%d,%d) wait=%.1fs",
                        tid,
                        person_id,
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

            # Cleanup stale person entries (no update for 5 minutes)
            stale_cutoff = now_s - 300.0
            stale_ids = [pid for pid, ts in person_last_seen.items() if ts < stale_cutoff]
            for pid in stale_ids:
                person_last_seen.pop(pid, None)
                person_start_times.pop(pid, None)

        except BaseException:
            logger.exception("frame_generator loop error")
            # Backoff on errors
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
            # keep the stream alive with a placeholder frame
            keepalive = _placeholder_frame("Reconnecting…")
            if keepalive:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + keepalive + b"\r\n")
            time.sleep(1.0)


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
  <h3>AI Track App (Render Free)</h3>
  <div>Streaming from RTSP_URL</div>
  <small>CPU only. Expect low FPS on free tier.</small>
  <div><a href="/video">/video</a> · <a href="/snapshot">/snapshot</a></div>
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
def index():
    return render_template_string(INDEX_HTML)


@app.get("/video")
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
