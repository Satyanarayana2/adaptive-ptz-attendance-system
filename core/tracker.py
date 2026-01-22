# core/tracker.py

import numpy as np
from scipy.optimize import linear_sum_assignment


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
    def __init__(self, track_id, bbox):
        self.id = track_id
        self.age = 0
        self.missed = 0

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

        self.Q = np.eye(8) * 0.01
        self.H = np.eye(4, 8)
        self.R = np.eye(4) * 1.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, bbox):
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
    def __init__(self, iou_threshold=0.3, max_missed=10):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        # Predict
        for track in self.tracks.values():
            track.predict()

        track_ids = list(self.tracks.keys())
        det_boxes = [d["bbox"] for d in detections]

        if len(track_ids) == 0:
            for det in detections:
                self.tracks[self.next_id] = KalmanTrack(self.next_id, det["bbox"])
                self.next_id += 1
            return self._format_results(detections)

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
                self.tracks[tid].update(det_boxes[c])
                assigned_tracks.add(tid)
                assigned_dets.add(c)

        # New tracks
        for i, det in enumerate(detections):
            if i not in assigned_dets:
                self.tracks[self.next_id] = KalmanTrack(self.next_id, det["bbox"])
                self.next_id += 1

        # Cleanup
        to_delete = []
        for tid, track in self.tracks.items():
            if tid not in assigned_tracks:
                track.missed += 1
            if track.missed > self.max_missed:
                to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        return self._format_results(detections)

    def _format_results(self, detections):
        results = []
        for track in self.tracks.values():
            results.append({
                "track_id": track.id,
                "bbox": track.get_bbox(),
                "kps": None
            })
        return results
