# utils/ptz/axis_camera.py

import requests
from requests.auth import HTTPDigestAuth
import cv2
import time


class AxisCamera:
    """
    Axis PTZ Camera Controller + Video Stream Handler
    Uses HTTP Digest Auth and MJPEG stream.
    """

    def __init__(self, ip, username, password, timeout=3):
        self.ip = ip
        self.username = username
        self.password = password
        self.timeout = timeout

        self.ptz_url = f"http://{ip}/axis-cgi/com/ptz.cgi"
        self.mjpeg_url = f"http://{username}:{password}@{ip}/axis-cgi/mjpg/video.cgi"

        self.auth = HTTPDigestAuth(username, password)
        self.cap = None


    # Connection

    def connect(self):
        """Test PTZ connectivity."""
        try:
            resp = requests.get(
                self.ptz_url,
                params={"query": "position"},
                auth=self.auth,
                timeout=self.timeout
            )
            resp.raise_for_status()
            print("[PTZ] Connected to Axis camera")
            return True
        except Exception as e:
            print(f"[PTZ][ERROR] Unable to connect: {e}")
            return False


    # Video Stream

    def open_stream(self):
        """Open MJPEG stream using OpenCV."""
        self.cap = cv2.VideoCapture(self.mjpeg_url)
        if not self.cap.isOpened():
            print("[PTZ][ERROR] Failed to open MJPEG stream")
            return False
        print("[PTZ] MJPEG stream opened")
        return True


    def read(self):
        """Read frame from camera."""
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.open_stream()

        ret, frame = self.cap.read()

        if not ret or frame is None:
            self.fail_count += 1
            if self.fail_count > 10:
                self.cap.release()
                self.cap = None
                time.sleep(1)
            return False, None

        self.fail_count = 0
        return True, frame

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    # PTZ Controls

    def goto_preset(self, preset_name):
        """Move camera to a named server preset."""
        print(f"[PTZ] Moving to preset: {preset_name}")
        try:
            resp = requests.get(
                self.ptz_url,
                params={"gotoserverpresetname": preset_name},
                auth=self.auth,
                timeout=self.timeout
            )
            resp.raise_for_status()
            time.sleep(2)  # allow camera to settle
        except Exception as e:
            print(f"[PTZ][ERROR] Preset move failed: {e}")

    def move(self, pan=0, tilt=0, zoom=0):
        """
        Continuous move:
        pan, tilt, zoom ∈ [-100, 100]
        """
        try:
            params = {"pan": pan, "tilt": tilt, "zoom": zoom}
            requests.get(self.ptz_url, params=params, auth=self.auth, timeout=1)
        except Exception:
            pass

    def stop(self):
        """Stop all movement."""
        self.move(0, 0, 0)
