# core/tracker.py

import numpy as np
from scipy.optimize import linear_sum_assignment


def is_valid_bbox(bbox):
    """Check if bounding box is valid: x1 < x2, y1 < y2, and all coordinates >= 0"""
    x1, y1, x2, y2 = bbox
    return x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


class KalmanTrack:
    def __init__(self, track_id, bbox, kps=None):
        if not is_valid_bbox(bbox):
            raise ValueError(f"Invalid bounding box: {bbox}")
        self.id = track_id
        self.age = 0
        self.missed = 0
        self.kps = kps  # Store keypoints

        # State: [cx, cy, w, h, vx, vy, vw, vh]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)

        self.P = np.eye(8) * 10
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = 1

        self.Q = np.eye(8) * 0.1
        self.H = np.eye(4, 8)
        self.R = np.eye(4) * 0.1

    def predict(self):
        # velocity friction/decay - zero ALL velocities on missed frames
        if self.missed>0:
            self.x[4] = 0.0  # Stop moving X
            self.x[5] = 0.0  # Stop moving Y
            self.x[6] = 0.0  # Stop expanding Width
            self.x[7] = 0.0  # Stop expanding Height
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, bbox, kps=None):
        if not is_valid_bbox(bbox):
            raise ValueError(f"Invalid bounding box: {bbox}")
        self.kps = kps  # Update keypoints
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w, h])

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.missed = 0
        self.age += 1

    def get_bbox(self):
        cx, cy, w, h = self.x[:4]
        return [
            int(cx - w/2), int(cy - h/2),
            int(cx + w/2), int(cy + h/2)
        ]


class KalmanTracker:
    def __init__(self, iou_threshold=0.3, max_missed=10, confirm_hits=1):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.confirm_hits = confirm_hits  # frames a new detection must persist before getting a real track ID
        self.tracks = {}
        self.tentative = {}  # track_id -> KalmanTrack (not yet confirmed)
        self.next_id = 1

    def update(self, detections):
        # Filter out invalid detections
        valid_detections = [d for d in detections if is_valid_bbox(d["bbox"])]

        # Predict
        for track in self.tracks.values():
            track.predict()

        track_ids = list(self.tracks.keys())
        det_boxes = [d["bbox"] for d in valid_detections]
        det_kps = [d.get("kps") for d in valid_detections]  # Get kps if available

        if len(track_ids) == 0 and len(self.tentative) == 0:
            for det in valid_detections:
                self.tentative[self.next_id] = KalmanTrack(self.next_id, det["bbox"], det.get("kps"))
                self.next_id += 1
            return self._format_results()

        # IoU cost matrix
        cost = np.zeros((len(track_ids), len(det_boxes)))
        for i, tid in enumerate(track_ids):
            tb = self.tracks[tid].get_bbox()
            for j, db in enumerate(det_boxes):
                cost[i, j] = 1 - iou(tb, db)

        row, col = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row, col):
            if cost[r, c] < 1 - self.iou_threshold:
                tid = track_ids[r]
                self.tracks[tid].update(det_boxes[c], det_kps[c])
                assigned_tracks.add(tid)
                assigned_dets.add(c)

        # Unmatched detections → try to match tentative tracks first, then create new tentatives
        unmatched_dets = [valid_detections[i] for i in range(len(valid_detections)) if i not in assigned_dets]
        tentative_ids = list(self.tentative.keys())
        assigned_tentatives = set()  # track which tentatives survived this frame

        if tentative_ids and unmatched_dets:
            t_boxes = [self.tentative[tid].get_bbox() for tid in tentative_ids]
            for det in unmatched_dets:
                matched_tentative = False
                for ti, tid in enumerate(tentative_ids):
                    if iou(t_boxes[ti], det["bbox"]) >= self.iou_threshold:
                        self.tentative[tid].update(det["bbox"], det.get("kps"))
                        assigned_tentatives.add(tid)  # mark as alive
                        matched_tentative = True
                        break
                if not matched_tentative:
                    self.tentative[self.next_id] = KalmanTrack(self.next_id, det["bbox"], det.get("kps"))
                    assigned_tentatives.add(self.next_id)
                    self.next_id += 1
        else:
            for det in unmatched_dets:
                self.tentative[self.next_id] = KalmanTrack(self.next_id, det["bbox"], det.get("kps"))
                assigned_tentatives.add(self.next_id)
                self.next_id += 1

        # Promote confirmed tentative tracks → real tracks
        to_promote = []
        for tid, t in self.tentative.items():
            if t.age >= self.confirm_hits:
                to_promote.append(tid)
        for tid in to_promote:
            self.tracks[tid] = self.tentative.pop(tid)
            assigned_tentatives.discard(tid)  # no longer tentative

        # FIXED CLEANUP: Evict any tentative not matched this frame → prevents RAM leak
        self.tentative = {tid: t for tid, t in self.tentative.items() if tid in assigned_tentatives}

        # Cleanup
        to_delete = []
        for tid, track in self.tracks.items():
            if tid not in assigned_tracks:
                track.missed += 1
            if track.missed > self.max_missed:
                to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        return self._format_results()

    def predict_only(self):
        """Advances the Kalman filter physics by one frame without any measurement correction."""
        for track in self.tracks.values():
            track.predict()
        for track in self.tentative.values():
            track.predict()
        return self._format_results()

    def _format_results(self):
        results = []
        for track in self.tracks.values():
            results.append({
                "track_id": track.id,
                "bbox": track.get_bbox(),
                "kps": track.kps
            })
        return results
