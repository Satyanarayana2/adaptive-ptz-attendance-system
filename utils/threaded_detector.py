import threading
import time

class ThreadedDetector:
    """
    Runs Face Detection asynchronously on a background thread.
    This bypasses the GIL via ONNXRuntime's C++ execution, providing true
    parallelism and decoupling the AI framerate from the visual display framerate.
    """
    def __init__(self, detector):
        self.detector = detector
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        
        self.latest_detections = []
        self.latest_detected_frame_id = -1
        
        self.stopped = False
        self.new_frame_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update_frame(self, frame, frame_id):
        """Called by the main loop to provide the freshest frame."""
        if frame is None:
            return
        with self.lock:
            self.frame = frame.copy()
            self.frame_id = frame_id
        self.new_frame_event.set()

    def update(self):
        """Background thread loop: constantly detects faces on the freshest frame."""
        while not self.stopped:
            self.new_frame_event.wait(timeout=0.05)
            self.new_frame_event.clear()
            
            with self.lock:
                frame_to_process = self.frame
                current_frame_id = self.frame_id
                
            if frame_to_process is not None:
                # Heavy math (bypasses GIL)
                faces = self.detector.detect(frame_to_process)
                faces = [f for f in faces if f.get('score', 0.0) > 0.65]
                
                with self.lock:
                    self.latest_detections = faces
                    self.latest_detected_frame_id = current_frame_id

    def get_latest_detections(self):
        """Called by the main loop to grab the asynchronously generated boxes."""
        with self.lock:
            return self.latest_detections, self.latest_detected_frame_id

    def stop(self):
        self.stopped = True
        self.new_frame_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
