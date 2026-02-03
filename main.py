import cv2
import time
import json
import os
from datetime import datetime

from utils.db import Database
from utils.detectors.insight_detector import InsightDetector
from utils.embeddings.insight_embedder import InsightEmbedder
from utils.face_alignment import FaceAligner

from core.recognizer import Recognizer
from core.attendance_logger import AttendanceLogger
from core.tracker import KalmanTracker
from core.quality_selector import QualitySelector
from core.timetable_loader import load_timetable_from_json

from utils.ptz.axis_camera import AxisCamera
from utils.ptz.presets import ENTRANCE_VIEW
last_unknown_save = {}



def main():
    print("=" * 60)
    print("PTZ CAMERA BASED ATTENDANCE SYSTEM STARTED")
    print("=" * 60)

    # Database
    db = Database()
    person_count = db.get_person_count()

    print(f"[INFO] Found {person_count} known persons in database.")

    # Load timetable into DB
    load_timetable_from_json(db, json_path="config/timetable.json")
    print("[INFO] Timetable data loaded into database.")

    # Load app config
    with open("config/app_config.json", "r") as f:
        app_config = json.load(f)

    # Initializing ML components

    detector = InsightDetector()
    detector.prepare()

    embedder = InsightEmbedder()
    embedder.prepare()

    aligner = FaceAligner()

    recognizer = Recognizer(
        embedder=embedder,
        db=db,
        threshold=app_config["recognition_threshold"]  # tune
    )

    tracker = KalmanTracker(iou_threshold=app_config["iou_threshold"], max_missed=app_config["max_missed"])
    quality_selector = QualitySelector(buffer_size=app_config["buffer_size"], min_frames=app_config["min_frames"])

    attendance_logger = AttendanceLogger(db, cooldown_seconds=app_config["cooldown_seconds"]) 
    # cooldown period is a duration in seconds during which
    # repeated attendance logs for the same person are ignored

    # Camera connection

    camera_type = app_config.get("camera_type", "webcam")
    if camera_type == "ptz":
        ptz_config = app_config["ptz"]
        camera = AxisCamera(
            ip = ptz_config["ip"],
            username = ptz_config["username"],
            password = ptz_config["password"]
        )

        if camera.connect():
            camera.goto_preset(ENTRANCE_VIEW)
            time.sleep(5)  # wait for PTZ to move
            camera.open_stream()
            print("[INFO] PTZ camera stream ready")
        else:
            print("[ERROR] Unable to connect to PTZ camera. Exiting.")
            print("[INFO] Using local webcam")
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("[ERROR] Unable to open webcam. Exiting.")
                return
    
    # if not camera.open_stream():
    #     print("[ERROR] Unable to open camera")
    #     return

    # Main loop
    while True:
        ret, frame = camera.read() # for PTZ camera, use read_frame() method
        if not ret:
            continue

        faces = detector.detect(frame)
        print(f"[DEBUG] Detected {len(faces)} faces")
        tracked_faces = tracker.update(faces)
        if len(tracked_faces) > 0:
                print(f"[DEBUG] Tracked {len(tracked_faces)} faces")

        # Clean up old tracks from recognition cache
        current_track_ids = [track["track_id"] for track in tracked_faces]
        attendance_logger.cleanup_old_tracks(current_track_ids)

        for track in tracked_faces:
            track_id = track["track_id"]
            bbox = track["bbox"]
            kps = track["kps"]

            # Optional: Print track age if available
            track_obj = tracker.tracks.get(track_id)
            if track_obj:
                print(f"[DEBUG] Track {track_id} age: {track_obj.age}")

            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                print(f"[DEBUG] Empty crop for track {track_id}")
                continue

            quality_selector.add_frame(track_id, crop, kps)

            best = quality_selector.get_best(track_id)
            if best is None:
                continue

            crop, kps = best

            print(f"[DEBUG] Got best frame for track {track_id}")

            # Layer 1: Check if we already recognized this person in this track
            cached_person_id = attendance_logger.get_recognized_person(track_id)
            
            if cached_person_id is not None:
                # Use cached result, skip expensive embedding
                result = {
                    "matched": True,
                    "person_id": cached_person_id,
                    "score": 0.0,
                    "name": "cached",
                    "cached": True
                }
                print(f"[DEBUG] Using CACHED recognition for track {track_id}")
            else:
                # First time: do expensive alignment + embedding + recognition
                try:
                    aligned = aligner.align(frame, kps)
                    print(f"[DEBUG] Aligned face for track {track_id}")
                except Exception as e:
                    print(f"[DEBUG] Alignment failed for track {track_id}: {e}")
                    continue

                result = recognizer.recognize(aligned)
                print(f"[DEBUG] Recognition result for track {track_id}: matched={result['matched']}, score={result.get('score', 'N/A')}")
                
                # Cache the recognition if successful
                if result["matched"]:
                    attendance_logger.cache_recognition(track_id, result["person_id"])

            if result["matched"]:
                attendance_logger.mark_attendance(
                    person_id=result["person_id"],
                    confidence=result["score"],
                    track_id=track_id,
                    source=app_config.get("camera_type")
                )

                color = (0, 255, 0)
                label = f"{result['name']} ({result['score']:.2f})"
            else:
                color = (0, 0, 255)
                label = "Unknown"
                save_unknown_face(track_id, crop)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Print final cache statistics
    stats = attendance_logger.get_cache_stats()
    print("\n" + "=" * 60)
    print("CACHE PERFORMANCE STATISTICS")
    print("=" * 60)
    print(f"Cache Hits:           {stats['cache_hits']}")
    print(f"Cache Misses:         {stats['cache_misses']}")
    print(f"Total Lookups:        {stats['total_lookups']}")
    print(f"Hit Rate:             {stats['hit_rate_percent']:.2f}%")
    print(f"Active Cached Tracks: {stats['active_cached_tracks']}")
    print("=" * 60 + "\n")

    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    db.close()

    print("=" * 60)
    print("SYSTEM SHUTDOWN CLEANLY")
    print("=" * 60)


def save_unknown_face(track_id, crop, cooldown=10):
    os.makedirs("unknown_faces", exist_ok=True)
    now = time.time()

    # Check cooldown
    if track_id in last_unknown_save:
        last_time = last_unknown_save[track_id]
        if now - last_time < cooldown:
            return  

    # Current timestamp string
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"unknown_faces/track{track_id}_{timestamp_str}.jpg"

    cv2.imwrite(filename, crop)
    last_unknown_save[track_id] = now
    print(f"[INFO] Unknown face saved: {filename}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting...")
