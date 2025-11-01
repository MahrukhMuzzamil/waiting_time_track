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
model = YOLO("yolov8n.pt")  # CPU on Render free tier
tracker = IoUTracker(max_missing_frames=30, iou_match_threshold=0.3)
reid_memory: Optional[ReIDMemory] = ReIDMemory(similarity_threshold=REID_SIM) if REID_ENABLED else None

# Shared capture for lightweight snapshot endpoint
_snap_cap: Optional[cv2.VideoCapture] = None


def open_capture() -> cv2.VideoCapture:
    if not RTSP_URL or not RTSP_URL.startswith("rtsp://"):
        raise RuntimeError("RTSP_URL env var is required and must start with rtsp://")
    cap = open_rtsp_with_fallbacks(RTSP_URL)
    if cap is None:
        raise RuntimeError("Unable to open RTSP source from RTSP_URL")
    return cap


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

    # Send a quick placeholder so client receives 200 immediately
    first = _placeholder_frame("Connecting to camera…")
    if first:
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + first + b"\r\n")

    while True:
        try:
            if cap is None:
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
            ih, iw = frame.shape[:2]
            scale = 640 / max(iw, ih)
            if scale < 1.0:
                new_w = max(320, int(iw * scale))
                new_h = max(240, int(ih * scale))
                frame_infer = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                frame_infer = frame

            with torch.no_grad():
                results = model.predict(
                    source=frame_infer,
                    imgsz=480,
                    conf=CONF_THRESHOLD,
                    classes=[0],
                    verbose=False,
                    device="cpu",
                    fuse=False,  # avoid Conv+BN fusion to reduce memory spikes
                )

            boxes_xyxy: List[np.ndarray] = []
            if results and len(results) > 0:
                r0 = results[0]
                if r0.boxes is not None and len(r0.boxes) > 0:
                    b = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
                    boxes_xyxy = [bb for bb in b]

            detections = np.array(boxes_xyxy, dtype=np.float32) if boxes_xyxy else np.zeros((0, 4), dtype=np.float32)
            tracked: Dict[int, np.ndarray] = tracker.step(int(now_s * 1000) % 1_000_000, detections, now_s)

            for tid, bbox in tracked.items():
                x1, y1, x2, y2 = bbox.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

                label_id = tid
                if reid_memory is not None:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                        label_id = reid_memory.assign_person_id(crop, now_s)

                start_time_s = tracker.get_track_start_time(tid) or now_s
                wait_s = now_s - start_time_s
                time_text = f"ID {label_id} · {format_hms(wait_s)}"
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

        except BaseException:
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
        stream_with_context(frame_generator()),
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
