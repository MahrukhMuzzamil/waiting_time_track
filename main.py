import argparse
import os
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl
from ultralytics import YOLO
import torch


@dataclass
class TrackState:
    track_id: int
    start_time_s: float
    last_seen_frame: int
    last_bbox_xyxy: np.ndarray  # shape (4,)


class IoUTracker:
    def __init__(
        self,
        max_missing_frames: int = 60,
        iou_match_threshold: float = 0.30,
        centroid_fallback_dist: float = 120.0,
    ) -> None:
        """
        IoU-based short-term tracker with centroid fallback.

        Defaults are intentionally stricter than before — identity persistence across
        longer absences is now handled by PersonRegistry + ReID, so this tracker only
        needs to stay stable frame-to-frame without aggressive matching that causes swaps.

        Args:
            max_missing_frames: Frames before dropping a track (60 @ ~8 fps ≈ 7.5 s)
            iou_match_threshold: Minimum IoU to bind a detection to an existing track
            centroid_fallback_dist: Max pixel distance for centroid fallback matching
        """
        self.max_missing_frames = max_missing_frames
        self.iou_match_threshold = iou_match_threshold
        self.centroid_fallback_dist = centroid_fallback_dist
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

    @staticmethod
    def _box_centroid(box: np.ndarray) -> Tuple[float, float]:
        """Get centroid (cx, cy) of a bounding box [x1, y1, x2, y2]."""
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @staticmethod
    def _centroid_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Euclidean distance between centroids of two boxes."""
        ca = IoUTracker._box_centroid(box_a)
        cb = IoUTracker._box_centroid(box_b)
        return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5

    def _assign_tracks(self, detections: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        # detections: (N, 4) xyxy
        assigned: List[Tuple[int, np.ndarray]] = []

        # Prepare cost matrix as 1 - IoU
        track_ids = list(self._tracks.keys())
        existing_boxes = (
            np.array([self._tracks[tid].last_bbox_xyxy for tid in track_ids]) if track_ids else np.zeros((0, 4))
        )
        if existing_boxes.size > 0 and detections.size > 0:
            iou_matrix = np.zeros((existing_boxes.shape[0], detections.shape[0]), dtype=np.float32)
            dist_matrix = np.zeros((existing_boxes.shape[0], detections.shape[0]), dtype=np.float32)
            for i, tbox in enumerate(existing_boxes):
                for j, dbox in enumerate(detections):
                    iou_matrix[i, j] = self._compute_iou(tbox, dbox)
                    dist_matrix[i, j] = self._centroid_distance(tbox, dbox)

            # Greedy matching by IoU first
            used_tracks = set()
            used_dets = set()
            pairs: List[Tuple[int, int]] = []

            # Phase 1: IoU-based matching
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

            # Phase 2: Centroid-based fallback for unmatched detections
            # This helps when people move quickly between frames
            while True:
                if len(used_tracks) == dist_matrix.shape[0] or len(used_dets) == dist_matrix.shape[1]:
                    break
                min_dist = float('inf')
                min_pair = (-1, -1)
                for i in range(dist_matrix.shape[0]):
                    if i in used_tracks:
                        continue
                    for j in range(dist_matrix.shape[1]):
                        if j in used_dets:
                            continue
                        if dist_matrix[i, j] < min_dist:
                            min_dist = dist_matrix[i, j]
                            min_pair = (i, j)
                if min_dist > self.centroid_fallback_dist:
                    break
                ti, dj = min_pair
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
            tid for tid, st in self._tracks.items() if frame_idx - st.last_seen_frame > self.max_missing_frames
        ]
        for tid in to_delete:
            del self._tracks[tid]

        # Return mapping: track_id -> bbox
        return {tid: self._tracks[tid].last_bbox_xyxy for tid in updated_track_ids}

    def get_track_start_time(self, track_id: int) -> Optional[float]:
        state = self._tracks.get(track_id)
        return state.start_time_s if state else None


class ReIDMemory:
    """
    Lightweight re-identification memory based on ResNet18 embeddings.
    It stores multiple embeddings per persistent person_id and matches new crops
    by cosine similarity to reconnect identities after occlusions or exits.
    
    Improved for long-term persistence (up to 1 hour) with multi-embedding gallery.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.72,
        ttl_seconds: float = 1200.0,  # 20 minute pause/resume window by default
        max_embeddings_per_person: int = 5,  # Store multiple views
        embedding_update_interval: float = 5.0,  # Only update embedding every 5 seconds
        min_crop_side: int = 40,  # Reject crops smaller than this in any dimension
        min_crop_area: int = 3000,  # Reject crops with total area below this
    ) -> None:
        # Lazy-import torchvision to avoid making it a hard runtime dependency
        try:
            import torchvision.transforms as T  # type: ignore
            from torchvision.models import resnet18, ResNet18_Weights  # type: ignore
        except Exception as exc:  # pragma: no cover - informative error for runtime
            raise RuntimeError(
                "Torchvision is required for ReID. Install torchvision or run without --reid/REID=0."
            ) from exc
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        # Remove classification head -> use global pooled features
        self.feature_extractor = torch.nn.Sequential(*(list(backbone.children())[:-1])).to(self.device)
        self.feature_extractor.eval()
        # Torchvision versions differ on where mean/std live; fall back to ImageNet defaults
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        meta = getattr(weights, 'meta', {}) or {}
        mean = meta.get('mean', imagenet_mean)
        std = meta.get('std', imagenet_std)

        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((128, 256)),  # Smaller than 224x224 for speed, person-shaped
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

        # Gallery: person_id -> list of embeddings (multiple views for robustness)
        self.person_id_to_embeddings: Dict[int, List[torch.Tensor]] = {}
        self.person_id_to_last_seen: Dict[int, float] = {}
        self.person_id_to_last_embedding_update: Dict[int, float] = {}  # Throttle embedding updates
        self.next_person_id: int = 1
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_embeddings_per_person = max_embeddings_per_person
        self.embedding_update_interval = embedding_update_interval
        self.min_crop_side = min_crop_side
        self.min_crop_area = min_crop_area

        # Track recently matched to prevent same-frame double matching
        self._frame_matched_pids: set = set()
        self._last_frame_time: float = 0.0

    def is_crop_usable(self, crop_bgr: np.ndarray) -> bool:
        """Gate low-quality crops out of ReID to avoid unreliable embeddings."""
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        h, w = crop_bgr.shape[:2]
        if h < self.min_crop_side or w < self.min_crop_side:
            return False
        if h * w < self.min_crop_area:
            return False
        # Person boxes should be roughly vertical (taller than wide); reject absurd ratios
        ratio = h / max(w, 1)
        if ratio < 0.8 or ratio > 5.0:
            return False
        return True

    @staticmethod
    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.flatten()
        b = b.flatten()
        return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    def _match_against_gallery(self, emb: torch.Tensor, gallery: List[torch.Tensor]) -> float:
        """Match against all embeddings in gallery, return max similarity."""
        if not gallery:
            return -1.0
        sims = [self._cosine_similarity(emb, ref) for ref in gallery]
        return max(sims)

    def _extract_embedding(self, crop_bgr: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            img = self.transform(crop_bgr[:, :, ::-1]).unsqueeze(0).to(self.device)  # BGR -> RGB
            feat = self.feature_extractor(img)  # [1, 512, 1, 1]
            feat = torch.nn.functional.normalize(feat.view(1, -1), dim=1)  # [1, 512]
        return feat.squeeze(0).cpu()

    def clear_frame_cache(self, now_s: float) -> None:
        """Call at start of each frame to reset per-frame matching cache."""
        if now_s - self._last_frame_time > 0.5:  # New frame
            self._frame_matched_pids.clear()
            self._last_frame_time = now_s

    def _purge_expired(self, now_s: float) -> None:
        expired = [pid for pid, ts in self.person_id_to_last_seen.items() if now_s - ts > self.ttl_seconds]
        for pid in expired:
            self.person_id_to_last_seen.pop(pid, None)
            self.person_id_to_embeddings.pop(pid, None)
            self.person_id_to_last_embedding_update.pop(pid, None)

    def match_person_id(
        self,
        crop_bgr: np.ndarray,
        now_s: float,
        exclude_pids: Optional[set] = None,
    ) -> Tuple[Optional[int], float]:
        """Attempt to match a crop to an existing person_id. Returns (pid or None, similarity)."""
        if not self.is_crop_usable(crop_bgr):
            return None, -1.0
        self._purge_expired(now_s)
        emb = self._extract_embedding(crop_bgr)
        best_pid: Optional[int] = None
        best_sim = -1.0
        for pid, gallery in self.person_id_to_embeddings.items():
            if pid in self._frame_matched_pids:
                continue
            if exclude_pids and pid in exclude_pids:
                continue
            sim = self._match_against_gallery(emb, gallery)
            if sim > best_sim:
                best_sim = sim
                best_pid = pid
        if best_pid is not None and best_sim >= self.similarity_threshold:
            return best_pid, best_sim
        return None, best_sim

    def score_against_pid(self, crop_bgr: np.ndarray, pid: int) -> float:
        """Return max cosine similarity between this crop and a specific pid's gallery."""
        if not self.is_crop_usable(crop_bgr):
            return -1.0
        gallery = self.person_id_to_embeddings.get(pid)
        if not gallery:
            return -1.0
        emb = self._extract_embedding(crop_bgr)
        return self._match_against_gallery(emb, gallery)

    def register_new_person(self, crop_bgr: np.ndarray, now_s: float) -> Optional[int]:
        """Create a new person identity from a usable crop. Returns the new pid or None if crop is bad."""
        if not self.is_crop_usable(crop_bgr):
            return None
        emb = self._extract_embedding(crop_bgr)
        pid = self.next_person_id
        self.next_person_id += 1
        self.person_id_to_embeddings[pid] = [emb]
        self.person_id_to_last_seen[pid] = now_s
        self.person_id_to_last_embedding_update[pid] = now_s
        self._frame_matched_pids.add(pid)
        return pid

    def touch_person(self, pid: int, crop_bgr: np.ndarray, now_s: float) -> None:
        """Mark person seen this frame and optionally refresh their embedding gallery."""
        self.person_id_to_last_seen[pid] = now_s
        self._frame_matched_pids.add(pid)
        if not self.is_crop_usable(crop_bgr):
            return
        last_update = self.person_id_to_last_embedding_update.get(pid, 0.0)
        if now_s - last_update <= self.embedding_update_interval:
            return
        emb = self._extract_embedding(crop_bgr)
        gallery = self.person_id_to_embeddings.setdefault(pid, [])
        if len(gallery) < self.max_embeddings_per_person:
            gallery.append(emb)
        else:
            gallery.pop(0)
            gallery.append(emb)
        self.person_id_to_last_embedding_update[pid] = now_s

    def assign_person_id(self, crop_bgr: np.ndarray, now_s: float) -> int:
        """Legacy API: match or create. Prefer match_person_id + register_new_person for new code."""
        if not self.is_crop_usable(crop_bgr):
            # Still allocate an ID so callers don't crash, but skip gallery operations
            pid = self.next_person_id
            self.next_person_id += 1
            return pid
        pid, _ = self.match_person_id(crop_bgr, now_s)
        if pid is not None:
            self.touch_person(pid, crop_bgr, now_s)
            return pid
        new_pid = self.register_new_person(crop_bgr, now_s)
        return new_pid if new_pid is not None else -1


@dataclass
class PersonRecord:
    label_id: int
    reid_pid: Optional[int]
    first_seen_s: float
    last_seen_s: float
    accumulated_wait_s: float  # Only ticks while the person is visible
    is_visible: bool
    current_tid: Optional[int]
    last_reid_check_s: float = 0.0
    last_bbox: Optional[np.ndarray] = None


class PersonRegistry:
    """
    Owns persistent person identities and wait-time accumulation.

    Key guarantees:
    - `accumulated_wait_s` only ticks while a person is actually visible.
    - When a person disappears, they are kept as "absent" for `absence_timeout_s`.
      If they reappear within that window (matched via ReID), their timer resumes.
    - After the absence window, the identity is forgotten; reappearing creates a new label.
    """

    def __init__(
        self,
        reid: Optional["ReIDMemory"],
        absence_timeout_s: float = 1200.0,  # 20 minutes
        reid_reverify_interval_s: float = 4.0,
        reid_reverify_margin: float = 0.10,
    ) -> None:
        self.reid = reid
        self.absence_timeout_s = absence_timeout_s
        self.reid_reverify_interval_s = reid_reverify_interval_s
        self.reid_reverify_margin = reid_reverify_margin
        self._records: Dict[int, PersonRecord] = {}
        self._tid_to_label: Dict[int, int] = {}
        self._pid_to_label: Dict[int, int] = {}  # ReID pid -> label_id
        self._next_label_id: int = 1

    def _crop(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)
        return frame[y1:y2, x1:x2]

    def _new_label(self) -> int:
        lbl = self._next_label_id
        self._next_label_id += 1
        return lbl

    def _resolve_label_for_new_track(
        self,
        crop: np.ndarray,
        now_s: float,
    ) -> Tuple[int, Optional[int]]:
        """Return (label_id, reid_pid) for a freshly-seen track."""
        if self.reid is None:
            return self._new_label(), None

        # Don't match against labels currently visible — they already have live tracks
        busy_pids = {
            rec.reid_pid for rec in self._records.values()
            if rec.is_visible and rec.reid_pid is not None
        }
        pid, _sim = self.reid.match_person_id(crop, now_s, exclude_pids=busy_pids)
        if pid is not None and pid in self._pid_to_label:
            existing_label = self._pid_to_label[pid]
            self.reid.touch_person(pid, crop, now_s)
            return existing_label, pid

        # No match: register a fresh ReID identity AND a new label
        new_pid = self.reid.register_new_person(crop, now_s)
        new_label = self._new_label()
        if new_pid is not None:
            self._pid_to_label[new_pid] = new_label
        return new_label, new_pid

    def _reverify_track(
        self,
        record: PersonRecord,
        crop: np.ndarray,
        now_s: float,
    ) -> None:
        """
        Periodically re-run ReID on an active track to catch tracker swaps.

        Guarded by a margin: we only swap this track's label if some other identity
        scores **significantly** higher than the current one. This prevents ReID noise
        from overwriting confirmed identities when two people have similar embeddings.
        """
        if self.reid is None:
            return
        if now_s - record.last_reid_check_s < self.reid_reverify_interval_s:
            return
        record.last_reid_check_s = now_s
        if not self.reid.is_crop_usable(crop):
            return

        # How well does this crop match its currently-claimed identity?
        current_sim = -1.0
        if record.reid_pid is not None:
            current_sim = self.reid.score_against_pid(crop, record.reid_pid)

        busy_pids = {
            rec.reid_pid for rec in self._records.values()
            if rec.is_visible and rec.reid_pid is not None and rec.label_id != record.label_id
        }
        # Also exclude our own pid — we want the best OTHER match
        if record.reid_pid is not None:
            busy_pids.add(record.reid_pid)

        other_pid, other_sim = self.reid.match_person_id(crop, now_s, exclude_pids=busy_pids)

        # Refresh current identity's gallery regardless — good embedding quality matters
        if record.reid_pid is not None and self.reid.is_crop_usable(crop):
            self.reid.touch_person(record.reid_pid, crop, now_s)

        if other_pid is None:
            return  # No competing match — nothing to do

        # Only swap if the other match is clearly better than staying put
        if other_sim < current_sim + self.reid_reverify_margin:
            return
        # And require the winning similarity to be comfortably above threshold
        if other_sim < self.reid.similarity_threshold + (self.reid_reverify_margin / 2.0):
            return

        expected_label = self._pid_to_label.get(other_pid)
        if expected_label is None:
            # ReID has this pid but no label — unusual. Just link it to our current label.
            self._pid_to_label[other_pid] = record.label_id
            record.reid_pid = other_pid
            self.reid.touch_person(other_pid, crop, now_s)
            return
        if expected_label == record.label_id:
            return  # Already pointing at us — no swap needed

        # Genuine identity mismatch: rebind this track to the correct existing label.
        if record.current_tid is not None:
            self._tid_to_label[record.current_tid] = expected_label
            target = self._records.get(expected_label)
            if target is not None:
                target.current_tid = record.current_tid
                target.is_visible = True
                target.last_seen_s = now_s
                target.last_bbox = record.last_bbox
        record.current_tid = None
        record.is_visible = False

    def update(
        self,
        frame: np.ndarray,
        tracked: Dict[int, np.ndarray],
        now_s: float,
        dt: float,
    ) -> Dict[int, PersonRecord]:
        """
        Advance the registry one frame.

        Args:
            frame: the current BGR frame (used for ReID crops).
            tracked: {tid -> bbox} from IoUTracker.step()
            now_s: wall-clock time in seconds.
            dt: seconds since the previous frame (clamped sensibly).

        Returns: {label_id -> PersonRecord} for visible persons this frame.
        """
        # Clamp dt: if a frame hangs for many seconds, don't accumulate that whole gap
        dt_clamped = max(0.0, min(dt, 2.0))

        # Reset ReID per-frame matched cache
        if self.reid is not None:
            self.reid.clear_frame_cache(now_s)

        # 1) Mark all records not visible; we'll re-mark visible ones below
        for rec in self._records.values():
            rec.is_visible = False
            rec.current_tid = None

        visible_labels: Dict[int, PersonRecord] = {}

        for tid, bbox in tracked.items():
            crop = self._crop(frame, bbox)
            label_id = self._tid_to_label.get(tid)

            if label_id is None:
                # New track: try ReID match against absent persons first
                label_id, pid = self._resolve_label_for_new_track(crop, now_s)
                self._tid_to_label[tid] = label_id
                if label_id not in self._records:
                    self._records[label_id] = PersonRecord(
                        label_id=label_id,
                        reid_pid=pid,
                        first_seen_s=now_s,
                        last_seen_s=now_s,
                        accumulated_wait_s=0.0,
                        is_visible=True,
                        current_tid=tid,
                        last_reid_check_s=now_s,
                        last_bbox=bbox,
                    )
                else:
                    rec = self._records[label_id]
                    if rec.reid_pid is None and pid is not None:
                        rec.reid_pid = pid
                    rec.is_visible = True
                    rec.current_tid = tid
                    rec.last_seen_s = now_s
                    rec.last_reid_check_s = now_s
                    rec.last_bbox = bbox
                    # Don't add dt on the first frame we re-bind — will tick next frame
            else:
                rec = self._records.get(label_id)
                if rec is None:
                    # Stale binding — rebuild
                    self._tid_to_label.pop(tid, None)
                    continue
                if rec.is_visible:
                    # Already claimed by another tid this frame; drop this duplicate assignment
                    self._tid_to_label.pop(tid, None)
                    continue
                rec.is_visible = True
                rec.current_tid = tid
                rec.accumulated_wait_s += dt_clamped
                rec.last_seen_s = now_s
                rec.last_bbox = bbox
                # Periodic re-verification
                self._reverify_track(rec, crop, now_s)

            visible_labels[self._tid_to_label[tid]] = self._records[self._tid_to_label[tid]]

        # 2) Clean up stale tid bindings (tids not in current tracker output)
        for tid in list(self._tid_to_label.keys()):
            if tid not in tracked:
                self._tid_to_label.pop(tid, None)

        # 3) Purge records past the absence window
        to_drop: List[int] = []
        for lbl, rec in self._records.items():
            if rec.is_visible:
                continue
            if now_s - rec.last_seen_s > self.absence_timeout_s:
                to_drop.append(lbl)
        for lbl in to_drop:
            rec = self._records.pop(lbl, None)
            if rec is not None and rec.reid_pid is not None:
                self._pid_to_label.pop(rec.reid_pid, None)

        return visible_labels

    def wait_seconds(self, label_id: int) -> float:
        rec = self._records.get(label_id)
        return rec.accumulated_wait_s if rec else 0.0

    def snapshot(self) -> Dict[int, PersonRecord]:
        return dict(self._records)


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
    cv2.putText(
        frame,
        text,
        (x + padding, y - padding),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def generate_rtsp_candidates(rtsp_url: str) -> List[str]:
    parsed = urlparse(rtsp_url)
    if parsed.scheme.lower() != "rtsp":
        return [rtsp_url]

    candidates: List[str] = []

    def with_path_and_query(path: str, extra_q: Dict[str, str] | None = None) -> str:
        qs = dict(parse_qsl(parsed.query, keep_blank_values=True))
        if extra_q:
            qs.update(extra_q)
        new_q = urlencode(qs)
        new_parts = (
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.params,
            new_q,
            parsed.fragment,
        )
        return urlunparse(new_parts)

    # Always include the original and the original with transportmode=unicast
    candidates.append(rtsp_url)
    if "transportmode=" not in parsed.query:
        candidates.append(with_path_and_query(parsed.path, {"transportmode": "unicast"}))

    # Add trailing-slash variants
    if not parsed.path.endswith("/"):
        candidates.append(with_path_and_query(parsed.path + "/"))
        candidates.append(with_path_and_query(parsed.path + "/", {"transportmode": "unicast"}))

    path = parsed.path or "/"

    def add_unique(url: str) -> None:
        if url not in candidates:
            candidates.append(url)

    # Normalize Channels vs channels
    if "/Streaming/Channels/" in path and "/Streaming/channels/" not in path:
        add_unique(with_path_and_query(path.replace("/Streaming/Channels/", "/Streaming/channels/")))

    # If Hikvision Channels pattern, try common main/sub and NVR mappings
    if "/Streaming/Channels/" in path or "/Streaming/channels/" in path:
        # Try a small set of common channel mappings (1..4), main (01) and sub (02)
        for cam_idx in (1, 2, 3, 4):
            for stream_suffix in (1, 2):
                chan = f"{cam_idx}0{stream_suffix}"
                for base in ("/Streaming/Channels/", "/Streaming/channels/"):
                    add_unique(with_path_and_query(f"{base}{chan}"))
                    add_unique(with_path_and_query(f"{base}{chan}", {"transportmode": "unicast"}))

    # ISAPI variant
    for cam_idx in (1, 2, 3, 4):
        for stream_suffix in (1, 2):
            chan = f"{cam_idx}0{stream_suffix}"
            add_unique(with_path_and_query(f"/ISAPI/Streaming/channels/{chan}"))
            add_unique(with_path_and_query(f"/ISAPI/Streaming/channels/{chan}", {"transportmode": "unicast"}))

    # Legacy paths used by some Hikvision firmwares
    add_unique(with_path_and_query("/h264/ch1/main/av_stream"))
    add_unique(with_path_and_query("/h264/ch1/sub/av_stream"))

    # Dahua-style paths (some OEMs too)
    for cam_idx in (1, 2, 3, 4):
        for subtype in (0, 1):  # 0 main, 1 sub
            add_unique(
                with_path_and_query(
                    "/cam/realmonitor",
                    {"channel": str(cam_idx), "subtype": str(subtype)},
                )
            )

    # Uniview-like variants
    for cam_idx in (1, 2, 3, 4):
        for subtype in (0, 1):
            add_unique(with_path_and_query(f"/live/ch{cam_idx}0{subtype}"))
            add_unique(
                with_path_and_query(
                    "/live",
                    {"channel": f"{cam_idx}", "subtype": f"{subtype}"},
                )
            )

    return candidates


def open_rtsp_with_fallbacks(rtsp_url: str, on_success: Optional[callable] = None) -> Optional[cv2.VideoCapture]:
    # #region agent log
    import json as _json
    _log_path = '/home/aesthetics-lab/50/.cursor/debug.log'
    def _debug_log(loc, msg, data, hyp=''):
        with open(_log_path, 'a') as _f: _f.write(_json.dumps({"location": loc, "message": msg, "data": data, "timestamp": int(time.time()*1000), "hypothesisId": hyp}) + '\n')
    # #endregion
    # Respect external overrides if present; otherwise use conservative defaults
    default_opts = "rtsp_transport;tcp|stimeout;15000000|max_delay;7000000|buffer_size;204800"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", default_opts)
    candidates = generate_rtsp_candidates(rtsp_url)
    # #region agent log
    _debug_log("main.py:open_rtsp_with_fallbacks", "Generated candidates", {"original_url": rtsp_url, "num_candidates": len(candidates), "first_3": candidates[:3]}, "B")
    # #endregion
    print(f"[RTSP] Attempting to connect with {len(candidates)} candidate URLs")
    for idx, candidate in enumerate(candidates, 1):
        print(f"[RTSP] [{idx}/{len(candidates)}] Trying: {candidate}")
        # #region agent log
        if idx <= 5: _debug_log("main.py:open_rtsp_with_fallbacks", f"Trying candidate {idx}", {"candidate": candidate}, "B,C")
        # #endregion
        cap_try = cv2.VideoCapture(candidate, cv2.CAP_FFMPEG)
        try:
            cap_try.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            print(f"[RTSP] Warning: Could not set buffer size: {e}")
        if cap_try.isOpened():
            ok, _ = cap_try.read()
            if ok:
                # #region agent log
                _debug_log("main.py:open_rtsp_with_fallbacks", "SUCCESS - connected", {"working_url": candidate, "tried_count": idx}, "A,B,C,D")
                # #endregion
                print(f"[RTSP] Success! Using: {candidate}")
                if on_success is not None:
                    try:
                        on_success(candidate)
                    except Exception as e:
                        print(f"[RTSP] Warning: on_success callback failed: {e}")
                return cap_try
            else:
                # #region agent log
                _debug_log("main.py:open_rtsp_with_fallbacks", "Opened but read failed", {"candidate": candidate, "idx": idx}, "D")
                # #endregion
                print(f"[RTSP] Failed to read frame from: {candidate}")
            cap_try.release()
        else:
            # #region agent log
            if idx <= 3: _debug_log("main.py:open_rtsp_with_fallbacks", "Failed to open", {"candidate": candidate, "idx": idx}, "D")
            # #endregion
            print(f"[RTSP] Failed to open: {candidate}")
    # #region agent log
    _debug_log("main.py:open_rtsp_with_fallbacks", "ALL CANDIDATES FAILED", {"total_tried": len(candidates), "original_url": rtsp_url}, "A,B,C,D")
    # #endregion
    print(f"[RTSP] ERROR: All {len(candidates)} candidate URLs failed")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime waiting-time overlay prototype (webcam)")
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        help=("Camera index, video path, RTSP url, or 'auto' to load saved RTSP"),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for person detections",
    )
    parser.add_argument(
        "--max-missing",
        type=int,
        default=30,
        help="Frames to keep track alive without detection",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.3,
        help="IoU threshold to match detections to tracks",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Overlay FPS counter",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run without display window (for autostart)",
    )
    parser.add_argument(
        "--reid",
        action="store_true",
        help="Enable ReID to persist identity across occlusions",
    )
    parser.add_argument(
        "--reid-sim",
        type=float,
        default=0.62,
        help="Cosine similarity threshold for ReID match",
    )
    args = parser.parse_args()

    # Config helpers for persisting RTSP
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

    def load_saved_rtsp() -> Optional[str]:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            url = data.get("rtsp_url")
            if isinstance(url, str) and url.startswith("rtsp://"):
                return url
        except Exception:
            pass
        return None

    def save_working_rtsp(url: str) -> None:
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump({"rtsp_url": url}, f, indent=2)
        except Exception:
            pass

    # Resolve source
    source: Optional[int | str]
    src_arg = (args.source or "").strip().lower()
    if src_arg == "auto" or src_arg == "":
        saved = load_saved_rtsp()
        if saved:
            source = saved
            print("[CONFIG] Loaded saved RTSP from config.json")
        else:
            raise RuntimeError("No saved RTSP found in config.json. Provide --source <rtsp-url> once to save it.")
    elif args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    model = YOLO("yolov8n.pt")  # auto-downloads

    if isinstance(source, str) and source.startswith("rtsp://"):
        cap = open_rtsp_with_fallbacks(source, on_success=save_working_rtsp)
        if cap is None:
            raise RuntimeError(f"Unable to open RTSP source after fallbacks. Last tried: {source}")
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    fps_smoother = None
    last_time = time.time()

    tracker = IoUTracker(max_missing_frames=args.max_missing, iou_match_threshold=args.iou)
    reid: Optional[ReIDMemory] = None
    if args.reid:
        reid = ReIDMemory(similarity_threshold=args.reid_sim)

    frame_idx = 0
    window_name = "Clinic Wait-Time Prototype"
    if not args.no_window:
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
            results = model.predict(
                source=frame,
                imgsz=640,
                conf=args.conf,
                classes=[0],
                verbose=False,
            )

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

                # Assign persistent person ID via ReID (optional)
                label_id = tid
                if reid is not None:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                        label_id = reid.assign_person_id(crop, now_s)

                # Waiting time based on first-seen time per tracker track; for persistent person id,
                # we can keep a dictionary that tracks earliest start per person id.
                # Here we store start per person id when ReID is enabled.
                if not hasattr(main, "person_start_times"):
                    main.person_start_times = {}

                if reid is not None:
                    if label_id not in main.person_start_times:
                        start_time_s = tracker.get_track_start_time(tid) or now_s
                        main.person_start_times[label_id] = start_time_s
                    start_time_s = main.person_start_times[label_id]
                else:
                    start_time_s = tracker.get_track_start_time(tid) or now_s

                wait_s = now_s - start_time_s
                time_text = f"ID {label_id} · {format_hms(wait_s)}"

                # Place label above head (top-center of bbox)
                label_x = int((x1 + x2) / 2)
                label_y = max(0, y1 - 6)
                draw_label_with_background(
                    frame,
                    time_text,
                    (label_x, label_y),
                    font_scale=0.6,
                    bg_color=(50, 50, 50),
                )

            if args.show_fps:
                fps_text = f"FPS: {fps_smoother:.1f}"
                cv2.putText(frame, fps_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if not args.no_window:
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
