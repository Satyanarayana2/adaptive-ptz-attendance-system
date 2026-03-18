# core/quality_selector.py

import cv2
import numpy as np
import math
import threading
from collections import deque

class QualitySelector:
    """
    Maintains a buffer of face crops per track_id.
    Selects the highest-quality frame using scoring:
        - Sharpness (variance of Laplacian)
        - Frontal angle (based on left/right eye coordinates)
        - Brightness score
    """

    def __init__(self, buffer_size=None, max_buffer=10, min_frames=5):
        """
        buffer_size: alias for max_buffer (used by main.py)
        """
        self.buffers = {}

        if buffer_size is not None:
            self.max_buffer = buffer_size
        else:
            self.max_buffer = max_buffer
        self.min_frames = min_frames
        self.lock = threading.Lock()


    # SCORE FUNCTIONS


    def score_sharpness(self, gray):
        """Higher is better. Expects a pre-converted grayscale image."""
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def score_brightness(self, gray):
        """Brightness scored 0–1 scale. Expects a pre-converted grayscale image."""
        return np.mean(gray) / 255.0

    def score_frontal(self, kps):
        """
        Eye alignment:
        Closer eye_y difference → more frontal.
        Score = 1.0 - normalized deviation
        """
        if kps is None or "left_eye" not in kps or "right_eye" not in kps:
            return 0.0  # No keypoints available, assume not frontal

        left_eye = kps["left_eye"]
        right_eye = kps["right_eye"]

        dy = abs(left_eye[1] - right_eye[1])
        dx = abs(left_eye[0] - right_eye[0])

        angle = math.degrees(math.atan2(dy, dx))
        angle = min(angle, 30)  # cap
        return 1.0 - (angle / 30.0)


    # ADD FRAME


    def add_frame(self, track_id, crop, kps):
        """
        Stores crop + score for this track_id if it passes a few data-driven gates.
        Grayscale conversion is done once and reused across all scoring functions.
        """
        h, w = crop.shape[:2]
        # size of the crop based on the minimum recognized face size is 31x36
        if w < 30 or h < 35:
            return

        # --- Single grayscale conversion reused by all scorers ---
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Skin detection via YCrCb.
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
        skin_mask = (cr > 133) & (cr < 173) & (cb > 77) & (cb < 127)
        skin_ratio = float(np.sum(skin_mask) / (h * w))
        if skin_ratio < 0.30:
            return

        # Pass pre-computed gray to avoid repeated conversion
        sharpness = self.score_sharpness(gray)
        brightness = self.score_brightness(gray)
        frontal = self.score_frontal(kps)

        # Weighted score
        score = (
            sharpness * 0.6 +
            brightness * 0.2 +
            frontal * 0.2
        )

        # Determine if this crop meets the Direct Pass criteria
        # Based on TestAnalysis: Sharpness >= 400, Luminance 80-200, Angle < 5 (Frontal > 0.833)
        mean_lum = brightness * 255.0
        is_direct_pass = (
            sharpness >= 400 and
            (80 < mean_lum < 200) and
            frontal > 0.833
        )

        entry = {"crop": crop, "kps": kps, "score": score, "direct_pass": is_direct_pass}

        with self.lock:
            if track_id not in self.buffers:
                # Use deque with maxlen for O(1) append + automatic eviction
                self.buffers[track_id] = deque(maxlen=self.max_buffer)

            self.buffers[track_id].append(entry)


    # GET BEST FRAME


    def get_best(self, track_id, min_frames_override=None):
        """
        Returns the best crop once enough frames are collected.
        Buffer is fully wiped after returning — retained frames are useless
        if the track_id changes, and could corrupt a new person's buffer.
        """
        required_frames = min_frames_override if min_frames_override is not None else self.min_frames
        
        with self.lock:
            if track_id not in self.buffers:
                return None

            frames = self.buffers[track_id]
            
            # Check if any frame in the buffer meets direct pass criteria
            has_direct_pass = any(f.get("direct_pass", False) for f in frames)

            if len(frames) < required_frames and not has_direct_pass:
                return None  # wait for more frames

            # O(n) max — no need to sort, we only want the single best
            best = max(frames, key=lambda f: f["score"])

            # Full wipe — stale crops from this track must not bleed into the next
            self.buffers[track_id] = deque(maxlen=self.max_buffer)

            return best["crop"], best["kps"]

    def clear(self, track_id):
        """Explicitly remove a track's buffer (e.g. when track is deleted)."""
        with self.lock:
            self.buffers.pop(track_id, None)
