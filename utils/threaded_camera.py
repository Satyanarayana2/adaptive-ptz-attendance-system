import threading
import time

class ThreadedCamera:
    """
    Background frame acquisition to prevent I/O blocking.
    Maintains only the most recent frame and drops backlogged frames.
    """
    def __init__(self, camera_source):
        self.camera = camera_source
        self.ret = False
        self.frame = None
        self.stopped = False
        self.thread = None
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()

        # Try to read the first frame synchronously to verify connection
        if hasattr(self.camera, 'read_frame'):
            self.ret, self.frame = self.camera.read_frame()
        else:
            self.ret, self.frame = self.camera.read()

    def start(self):
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            if hasattr(self.camera, 'read_frame'):
                ret, frame = self.camera.read_frame()
            else:
                ret, frame = self.camera.read()

            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
                self.new_frame_event.set()
            else:
                time.sleep(0.01)

    def read(self):
        # Wait up to 1s for a *new* frame.
        self.new_frame_event.wait(timeout=1.0)
        self.new_frame_event.clear()
        
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            else:
                return self.ret, None

    def release(self):
        self.stopped = True
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        if hasattr(self.camera, 'release'):
            self.camera.release()
