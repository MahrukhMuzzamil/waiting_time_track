import argparse
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class TrackState:
    track_id: int
    start_time_s: float
    last_seen_frame: int
    last_bbox_xyxy: np.ndarray  # shape (4,)


class IoUTracker:
    def __init__(self, max_missing_frames: int = 30, iou_match_threshold: float = 0.3) -> None:
        self.max_missing_frames = max_missing_frames
        self.iou_match_threshold = iou_match_threshold
        self._next_id: int = 1
        self._tracks: Dict[int, TrackState] = {}

    @staticmethod
    def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union

    def _assign_tracks(self, detections: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        # detections: (N, 4) xyxy
        assigned: List[Tuple[int, np.ndarray]] = []

        # Prepare cost matrix as 1 - IoU
        track_ids = list(self._tracks.keys())
        existing_boxes = np.array([self._tracks[tid].last_bbox_xyxy for tid in track_ids]) if track_ids else np.zeros((0, 4))
        if existing_boxes.size > 0 and detections.size > 0:
            iou_matrix = np.zeros((existing_boxes.shape[0], detections.shape[0]), dtype=np.float32)
            for i, tbox in enumerate(existing_boxes):
                for j, dbox in enumerate(detections):
                    iou_matrix[i, j] = self._compute_iou(tbox, dbox)
            # Greedy matching by IoU
            used_tracks = set()
            used_dets = set()
            pairs: List[Tuple[int, int]] = []
            while True:
                if len(used_tracks) == iou_matrix.shape[0] or len(used_dets) == iou_matrix.shape[1]:
                    break
                max_iou = -1.0
                max_pair = (-1, -1)
                for i in range(iou_matrix.shape[0]):
                    if i in used_tracks:
                        continue
                    for j in range(iou_matrix.shape[1]):
                        if j in used_dets:
                            continue
                        if iou_matrix[i, j] > max_iou:
                            max_iou = iou_matrix[i, j]
                            max_pair = (i, j)
                if max_iou < self.iou_match_threshold:
                    break
                ti, dj = max_pair
                used_tracks.add(ti)
                used_dets.add(dj)
                pairs.append((ti, dj))

            for ti, dj in pairs:
                tid = track_ids[ti]
                assigned.append((tid, detections[dj]))

            # Remaining detections become new tracks
            for dj in range(detections.shape[0]):
                if dj not in used_dets:
                    tid = self._next_id
                    self._next_id += 1
                    assigned.append((tid, detections[dj]))
        else:
            # No existing tracks or no detections: create new tracks for all detections
            for dbox in detections:
                tid = self._next_id
                self._next_id += 1
                assigned.append((tid, dbox))

        return assigned

    def step(self, frame_idx: int, detections_xyxy: np.ndarray, now_s: float) -> Dict[int, np.ndarray]:
        assignments = self._assign_tracks(detections_xyxy)

        # Update track states
        updated_track_ids = set()
        for tid, bbox in assignments:
            if tid in self._tracks:
                state = self._tracks[tid]
                state.last_bbox_xyxy = bbox
                state.last_seen_frame = frame_idx
            else:
                self._tracks[tid] = TrackState(
                    track_id=tid,
                    start_time_s=now_s,
                    last_seen_frame=frame_idx,
                    last_bbox_xyxy=bbox,
                )
            updated_track_ids.add(tid)

        # Drop stale tracks
        to_delete = [
            tid for tid, st in self._tracks.items()
            if frame_idx - st.last_seen_frame > self.max_missing_frames
        ]
        for tid in to_delete:
            del self._tracks[tid]

        # Return mapping: track_id -> bbox
        return {tid: self._tracks[tid].last_bbox_xyxy for tid in updated_track_ids}

    def get_track_start_time(self, track_id: int) -> Optional[float]:
        state = self._tracks.get(track_id)
        return state.start_time_s if state else None


def format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def draw_label_with_background(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.6,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
    padding: int = 4,
) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = org
    # Background rectangle above the head (y is top of bbox); shift up by baseline
    cv2.rectangle(frame, (x, y - th - 2 * padding), (x + tw + 2 * padding, y), bg_color, cv2.FILLED)
    cv2.putText(frame, text, (x + padding, y - padding), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime waiting-time overlay prototype (webcam)")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video path. '0' for default webcam")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold for person detections")
    parser.add_argument("--max-missing", type=int, default=30, help="Frames to keep track alive without detection")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold to match detections to tracks")
    parser.add_argument("--show-fps", action="store_true", help="Overlay FPS counter")
    args = parser.parse_args()

    # Resolve source
    source: Optional[int | str]
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    model = YOLO("yolov8n.pt")  # auto-downloads

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    fps_smoother = None
    last_time = time.time()

    tracker = IoUTracker(max_missing_frames=args.max_missing, iou_match_threshold=args.iou)

    frame_idx = 0
    window_name = "Clinic Wait-Time Prototype"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now_s = time.time()
            dt = now_s - last_time
            last_time = now_s
            if fps_smoother is None:
                fps_smoother = 1.0 / max(dt, 1e-6)
            else:
                # Exponential moving average
                fps_smoother = 0.9 * fps_smoother + 0.1 * (1.0 / max(dt, 1e-6))

            # Run YOLO inference
            results = model.predict(source=frame, imgsz=640, conf=args.conf, classes=[0], verbose=False)

            boxes_xyxy: List[np.ndarray] = []
            if results and len(results) > 0:
                r0 = results[0]
                if r0.boxes is not None and len(r0.boxes) > 0:
                    # xyxy tensor
                    b = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
                    boxes_xyxy = [bb for bb in b]

            detections = np.array(boxes_xyxy, dtype=np.float32) if boxes_xyxy else np.zeros((0, 4), dtype=np.float32)

            # Update tracker
            tracked: Dict[int, np.ndarray] = tracker.step(frame_idx, detections, now_s)

            # Draw
            for tid, bbox in tracked.items():
                x1, y1, x2, y2 = bbox.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1] - 1, x2)
                y2 = min(frame.shape[0] - 1, y2)

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

                # Waiting time
                start_time_s = tracker.get_track_start_time(tid) or now_s
                wait_s = now_s - start_time_s
                time_text = f"ID {tid} · {format_hms(wait_s)}"

                # Place label above head (top-center of bbox)
                label_x = int((x1 + x2) / 2)
                label_y = max(0, y1 - 6)
                draw_label_with_background(frame, time_text, (label_x, label_y), font_scale=0.6, bg_color=(50, 50, 50))

            if args.show_fps:
                fps_text = f"FPS: {fps_smoother:.1f}"
                cv2.putText(frame, fps_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
