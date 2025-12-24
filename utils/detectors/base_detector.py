# utils/detectors/base_detector.py

class BaseDetector:
    """
    Abstract interface for all face detection backends.
    Every detector (InsightFace, YOLO, RetinaFace...) will follow this API.
    """

    def prepare(self):
        """
        Load model weights, allocate resources, set device.
        Called once before using `detect`.
        """
        raise NotImplementedError("prepare() must be implemented by subclasses")

    def detect(self, frame):
        """
        Detect faces in a BGR frame.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image (H, W, 3) as read by cv2.

        Returns
        -------
        List[dict]
            A list of detection dictionaries with this shape:
            [
                {
                    'bbox': (x1, y1, x2, y2),
                    'score': float,
                    'kps': {
                        'left_eye': (x, y),
                        'right_eye': (x, y),
                        'nose': (x, y),
                        'left_mouth': (x, y),
                        'right_mouth': (x, y)
                    }
                },
                ...
            ]

        Notes
        -----
        - Coordinates are in image pixel space (int).
        - kps may be empty or partial if detector doesn't provide all points.
        """
        raise NotImplementedError("detect() must be implemented by subclasses")
