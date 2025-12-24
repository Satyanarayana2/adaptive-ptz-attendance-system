# core/tracker.py

import numpy as np


def iou(boxA, boxB):
    """
    Compute Intersection over Union between two boxes.
    box = [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


class Track:
    def __init__(self, track_id, bbox, kps, frame_id):
        self.id = track_id
        self.bbox = bbox
        self.kps = kps
        self.last_seen = frame_id
        self.missed = 0
        self.updated = True
        self.age = 0  # Track how long this track has been active


class SimpleTracker:
    """
    Stable IoU-based tracker for attendance systems.

    Responsibilities:
      - Maintain consistent track_id across frames
      - Tolerate bbox jitter and short detection loss
      - Do NOT know about recognition or attendance
    """

    def __init__(self, iou_threshold=0.25, max_missed=5):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

        self.tracks = {}
        self.next_track_id = 1
        self.frame_id = 0

    # ----------------------------------------------------

    def update(self, detections):
        """
        detections: list of dicts
          {
            "bbox": [x1, y1, x2, y2],
            "kps": keypoints
          }

        Returns:
          list of dicts with track_id added
        """

        self.frame_id += 1

        # Mark all tracks as not updated
        for track in self.tracks.values():
            track.updated = False

        results = []

        for det in detections:
            bbox = det["bbox"]
            kps = det["kps"]

            best_iou = 0
            best_track = None

            for track in self.tracks.values():
                score = iou(bbox, track.bbox)
                if score > best_iou:
                    best_iou = score
                    best_track = track

            if best_track and best_iou >= self.iou_threshold:
                # Assign existing track
                best_track.bbox = bbox
                best_track.kps = kps
                best_track.last_seen = self.frame_id
                best_track.missed = 0
                best_track.updated = True
                best_track.age += 1  # Increment age on successful update

                track_id = best_track.id
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1

                self.tracks[track_id] = Track(
                    track_id=track_id,
                    bbox=bbox,
                    kps=kps,
                    frame_id=self.frame_id
                )

            results.append({
                "track_id": track_id,
                "bbox": bbox,
                "kps": kps
            })

        # Cleanup dead tracks
        to_delete = []
        for track_id, track in self.tracks.items():
            if not track.updated:
                track.missed += 1
            if track.missed > self.max_missed:
                to_delete.append(track_id)

        for track_id in to_delete:
            del self.tracks[track_id]

        return results
