# utils/detectors/insight_detector.py

from .base_detector import BaseDetector
from insightface.app import FaceAnalysis
import numpy as np
import warnings

class InsightDetector(BaseDetector):
    """
    Face detector using InsightFace FaceAnalysis (SCRFD/landmark).
    This is a thin wrapper that normalizes InsightFace output into the
    project's detection dict format.

    Parameters
    ----------
    det_size : tuple(int,int)
        Detection input size (width, height). Larger -> more accurate slower.
    ctx_id : int
        Device id: -1 for CPU, 0 for first GPU (if available).
    allowed_modules : list[str] or None
        InsightFace modules to load. Defaults to ['detection','landmark'].
    """

    def __init__(self, det_size=(640, 640), ctx_id=-1, allowed_modules=None):
        self.det_size = det_size
        self.ctx_id = ctx_id
        self.allowed_modules = allowed_modules or ['detection', 'landmark']
        self.model = None
        self._prepared = False

    def prepare(self):
        """
        Initialize InsightFace FaceAnalysis model. Safe to call multiple times.
        """
        if self._prepared:
            return

        try:
            self.model = FaceAnalysis(allowed_modules=self.allowed_modules)
            # prepare: ctx_id=-1 -> CPU, ctx_id=0 -> GPU
            self.model.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
            self._prepared = True
        except Exception as e:

            warnings.warn(
                f"InsightDetector.prepare() failed: {e}. "
                "Make sure insightface is installed and models are reachable."
            )
            raise

    def detect(self, frame):
        """
        Run detection on a BGR frame and return list of normalized detections.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image (H, W, 3) as returned by cv2.

        Returns
        -------
        List[dict]
            List of detection dicts (see BaseDetector.detect docstring).
        """
        if not self._prepared:
            raise RuntimeError("InsightDetector not prepared. Call prepare() first.")

        # FaceAnalysis.get returns a list of Face objects
        faces = self.model.get(frame)
        detections = []

        for f in faces:
            # bbox comes as (x1, y1, x2, y2) in float; convert to ints and clamp
            try:
                bbox = np.array(f.bbox).astype(int).tolist()
            except Exception:
                # fallback: skip malformed detections
                continue

            # build keypoints dict if available (InsightFace provides 5-point landmarks)
            kps = {}
            kps_raw = getattr(f, "kps", None)
            if kps_raw is not None and len(kps_raw) >= 5:
                # ensure floats and tuples
                try:
                    kps = {
                        'left_eye': tuple(map(float, kps_raw[0])),
                        'right_eye': tuple(map(float, kps_raw[1])),
                        'nose': tuple(map(float, kps_raw[2])),
                        'left_mouth': tuple(map(float, kps_raw[3])),
                        'right_mouth': tuple(map(float, kps_raw[4])),
                    }
                except Exception:
                    kps = {}

            det = {
                'bbox': tuple(bbox),
                'score': float(getattr(f, "det_score", 0.0)),
                'kps': kps
            }
            detections.append(det)

        return detections
