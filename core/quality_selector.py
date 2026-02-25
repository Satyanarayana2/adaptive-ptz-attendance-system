# core/quality_selector.py

import cv2
import numpy as np
import math
import time
import threading

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


    def score_sharpness(self, img):
        """Higher is better."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def score_brightness(self, img):
        """Brightness scored 0–1 scale."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        return mean_val / 255.0

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
        Stores crop + score for this track_id if it passes few data-driven gates.
        """
        h, w = crop.shape[:2]
        # size of the crop based on the minimum recognized face size is 31x36
        if w < 30 or h < 35:
            return
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)      
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
        skin_mask = (cr > 133) & (cr < 173) & (cb > 77) & (cb < 127)
        skin_ratio = float(np.sum(skin_mask) / (h * w))
        if skin_ratio < 0.30:
            return

        sharpness = self.score_sharpness(crop)
        sharpness_norm = min(sharpness/2000.0,1.0)
        brightness = self.score_brightness(crop)
        frontal = self.score_frontal(kps)

        # Weighted score
        score = (
            sharpness * 0.6 +
            brightness * 0.2 +
            frontal * 0.2
        )

        entry = {"crop": crop, "kps": kps, "score": score}

        with self.lock:
            if track_id not in self.buffers:
                self.buffers[track_id] = []

            self.buffers[track_id].append(entry)

            # Keep buffer clean
            if len(self.buffers[track_id]) > self.max_buffer:
                self.buffers[track_id].pop(0)


    # GET BEST FRAME


    def get_best(self, track_id):
        """
        Returns the best crop for this ID once enough frames collected.
        After returning, the buffer for this track_id is cleared.
        """
        with self.lock:
            if track_id not in self.buffers:
                return None

            frames = self.buffers[track_id]

            if len(frames) < self.min_frames:
                return None  # wait for more frames

            # Pick best by score
            best = max(frames, key=lambda f: f["score"])

            # Clear buffer after use
            self.buffers[track_id] = []

            return best["crop"], best["kps"]
