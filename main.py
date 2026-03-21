import threading
import uvicorn
import cv2
import time
import json
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from utils.db import Database
from utils.detectors.insight_detector import InsightDetector
from utils.embeddings.insight_embedder import InsightEmbedder
from utils.face_alignment import FaceAligner

from core.recognizer import Recognizer
from core.attendance_logger import AttendanceLogger
from core.tracker import KalmanTracker
from core.quality_selector import QualitySelector
from core.folder_watcher import FolderWatcher
from core.adaptive_manager import AdaptiveManager
from core.session_controller import SessionController
from core.performance_results import PerformanceProfiler

from utils.ptz.axis_camera import AxisCamera
from utils.ptz.presets import ENTRANCE_VIEW
from utils.logs import Logger

# Import FastAPI app and shared state
from app import app, lock
import app as web_app

sys.stdout = Logger()  # Redirect print statements to both console and log file
last_unknown_save = {}


def process_single_face(track, frame, quality_selector, attendance_logger, aligner, recognizer, adaptive_manager, current_phase, profiler=None, unknown_save_threshold=0.2):
    """
    Worker function to process a single face in a separate thread.
    """
    track_id = track["track_id"]
    bbox = track["bbox"]
    kps = track["kps"]

    # 1. Safe Cropping
    t0 = time.time()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    elapsed = time.time() - t0
    if profiler: profiler.record_module_time(track_id, "crop", elapsed)

    if crop.size == 0:
        return None

    # 2. Cache Check FIRST — skip quality buffering entirely if already recognized.
    t0 = time.time()
    cached_data = attendance_logger.check_cache(track_id) if current_phase != "ENTRY" else None
    elapsed = time.time() - t0
    if profiler: profiler.record_module_time(track_id, "cache_check", elapsed)

    if cached_data is not None:
        person_id, score = cached_data

        if person_id is not None:
            # Already recognized — return green immediately, no buffering needed
            return {
                "bbox": (x1, y1, x2, y2),
                "label": f"ID:{person_id} ({score:.2f})",
                "color": (0, 255, 0)
            }

        # person_id is None → cached as Unknown.
        # Keep adding frames silently and retry recognition when buffer is ready.
        # Show "Unknown" or "Retrying..." continuously instead of flipping back to "Analyzing..."
        
        # Check if previous attempt scored between 0.30 and 0.41 to enter Retry Phase
        _sc = score if isinstance(score, (int, float)) else 0.0
        is_retry_phase = (0.30 <= _sc <= 0.41)
        required_frames = 3 if is_retry_phase else None
        
        t0 = time.time()
        quality_selector.add_frame(track_id, crop, kps, profiler=profiler)
        elapsed = time.time() - t0
        if profiler: profiler.record_module_time(track_id, "quality_buffer", elapsed)

        t0 = time.time()
        best = quality_selector.get_best(track_id, min_frames_override=required_frames, profiler=profiler)
        elapsed = time.time() - t0
        if profiler: profiler.record_module_time(track_id, "get_best", elapsed)

        if best is None:
            # Buffer still filling — hold the label, do not show Analyzing
            label = "Retrying..." if is_retry_phase else "Unknown"
            color = (0, 165, 255) if is_retry_phase else (0, 0, 255) # Orange for retry
            return {
                "bbox": (x1, y1, x2, y2),
                "label": label,
                "color": color
            }
        # Buffer ready — fall through to recognition retry below
        best_crop, best_kps = best

        t0 = time.time()
        try:
            aligned = aligner.align(frame, best_kps)
            result = recognizer.recognize(aligned)
        except Exception:
            print(f"[ERROR] Recognition retry failed for unknown track {track_id}")
            result = {"matched": False}
        elapsed = time.time() - t0
        if profiler: profiler.record_module_time(track_id, "aligned & recognize", elapsed)
        if profiler: profiler.record_recognition_result(track_id, result["matched"], result.get("score", 0.0), result.get("person_id"))


        if result["matched"]:
            # Upgraded from Unknown → now recognized!
            attendance_logger.cache_recognition(track_id, result["person_id"], result["score"])
            if attendance_logger.should_log(track_id, result["person_id"]):
                rec_path = save_recognized_face(result["person_id"], best_crop)
                attendance_logger.mark_attendance(
                    person_id=result["person_id"],
                    confidence=result["score"],
                    track_id=track_id,
                    face_crop_path=rec_path
                )
            if "embedding" in result and result["embedding"] is not None:
                adaptive_manager.process(
                    person_id=result["person_id"],
                    crop=best_crop,
                    kps=best_kps,
                    embedding=result["embedding"],
                    sim_score=result["score"],
                    template_type=result.get("matched_template_type", "ANCHOR")
                )
            return {
                "bbox": (x1, y1, x2, y2),
                "label": f"ID:{result['person_id']} ({result['score']:.2f})",
                "color": (0, 255, 0)
            }
        else:
            _sc = result.get("score", 0.0)
            last_score = _sc if isinstance(_sc, (int, float)) else 0.0
            # Still unknown after retry — reset cache sentinel with actual score
            attendance_logger.cache_recognition(track_id, None, last_score)
            # Only save crop if score is truly low (not an enrolled person at bad angle)
            if last_score < unknown_save_threshold:
                save_unknown_face(track_id, best_crop)
            
            label = "Retrying..." if 0.30 <= last_score <= 0.41 else "Unknown"
            color = (0, 165, 255) if label == "Retrying..." else (0, 0, 255)
            return {
                "bbox": (x1, y1, x2, y2),
                "label": label,
                "color": color
            }

    # 3. Quality Selection (cache miss or ENTRY phase — first-time recognition path)
    quality_selector.add_frame(track_id, crop, kps, profiler=profiler)
    best = quality_selector.get_best(track_id, profiler=profiler)

    if best is None:
        return {
            "bbox": (x1, y1, x2, y2),
            "label": "Analyzing...",
            "color": (255, 255, 0)
        }

    best_crop, best_kps = best

    # 4. ENTRY phase cache check
    if current_phase == "ENTRY":
        cached_data = None
    else:
        cached_data = attendance_logger.check_cache(track_id)

    if cached_data is not None:
        person_id, score = cached_data
        if person_id is not None:
            result = {"matched": True, "person_id": person_id, "score": score, "name": "Cached"}
        else:
            result = {"matched": False}
    else:
        # 5. Alignment & Recognition
        try:
            aligned = aligner.align(frame, best_kps)
            result = recognizer.recognize(aligned)
        except Exception:
            print(f"[ERROR] Alignment/Recognition failed for track {track_id}")
            result = {"matched": False}

        if profiler: 
            profiler.record_recognition_result(track_id, result["matched"], result.get("score", 0.0), result.get("person_id"))

        if result["matched"]:
            attendance_logger.cache_recognition(track_id, result["person_id"], result["score"])
        else:
            # Cache the Unknown state so next frames show "Unknown" not "Analyzing..."
            attendance_logger.cache_recognition(track_id, None, result.get("score", 0.0))

    # 6. Logging & Saving
    _sc = result.get("score", 0.0)
    last_score = _sc if isinstance(_sc, (int, float)) else 0.0
    label = "Retrying..." if (not result.get("matched") and 0.30 <= last_score <= 0.41) else "Unknown"
    color = (0, 165, 255) if label == "Retrying..." else (0, 0, 255)

    if result["matched"]:
        color = (0, 255, 0)
        label = f"ID:{result['person_id']} ({result['score']:.2f})"

        if attendance_logger.should_log(track_id, result["person_id"]):
            rec_path = save_recognized_face(result["person_id"], best_crop)
            attendance_logger.mark_attendance(
                person_id=result["person_id"],
                confidence=result["score"],
                track_id=track_id,
                face_crop_path=rec_path
            )
        if "embedding" in result and result["embedding"] is not None:
            adaptive_manager.process(
                person_id=result["person_id"],
                crop=best_crop,
                kps=best_kps,
                embedding=result["embedding"],
                sim_score=result["score"],
                template_type=result.get("matched_template_type", "ANCHOR")
            )
    else:
        # Only save crop if score is truly low — not just a bad angle of an enrolled face
        if result.get("score", 0.0) < unknown_save_threshold:
            save_unknown_face(track_id, best_crop)

    return {
        "bbox": (x1, y1, x2, y2),
        "label": label,
        "color": color
    }



def main():
    print("=" * 60)
    print("PTZ CAMERA BASED ATTENDANCE SYSTEM STARTED")
    print("=" * 60)

    # initializing performance profiler
    profiler = PerformanceProfiler()

    # Database
    db = Database()

    # Load timetable into DB
    db.sync_timetable(json_file_path="config/timetable.json")
    print("[INFO] Timetable data loaded into database.")

    print("[INFO] Checking for new faces to enroll...")
    watcher = FolderWatcher(image_dir="Face_images")
    watcher.run()
    person_count = db.get_person_count()

    print(f"[INFO] Found {person_count} known persons in database.")

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
    # intializing Adaptive Manager module
    adaptive_manager = AdaptiveManager(
        config=app_config["adaptive_gallery"], db=db, 
    )

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
            camera.open_stream()
            print("[INFO] PTZ camera stream ready")
        else:
            print("[ERROR] Unable to connect to PTZ camera. Exiting.")
            print("[INFO] Using local webcam")
            camera_type = "webcam"
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("[ERROR] Unable to open webcam. Exiting.")
                return
    
    # this module is for making the ptz movements accordingly to the time_table session
    session_controller = SessionController(db=db, ptz=camera if camera_type == "ptz" else None, adaptive_manager=adaptive_manager, tracker=tracker, profiler=profiler)

    # Start Flask server in a background thread (non-daemon so it survives after AI loop stops)
    api_thread = threading.Thread(
        target = lambda: uvicorn.run(app, host='0.0.0.0', port = 5000, log_level="warning"),
        daemon=False
    )
    api_thread.start()
    print("[INFO] FastAPI server started on http://127.0.0.1:5000")
    time.sleep(2)
    
    # checking if this is running in docker or not
    IS_DOCKER = os.path.exists("/.dockerenv")
    # Main loop
    # Use 4 workers 
    executor = ThreadPoolExecutor(max_workers=8)
    while True:
        if web_app.stop_signal:
            print("[INFO] Stop signal received. Ending main loop.")
            break
        session_controller.update()
        ret, frame = camera.read_frame() if camera_type == "ptz" else camera.read()
        if not ret:
            time.sleep(0.05)
            continue

        profiler.record_camera_frame()
        profiler.start_frame_processing()

        faces = detector.detect(frame)
        faces = [f for f in faces if f.get('score', 0.0) > 0.50]
        # if len(faces) > 0:
        #     print(f"[DEBUG] Detected {len(faces)} faces")
        tracked_faces = tracker.update(faces)
        profiler.record_detection(len(faces))
        profiler.record_tracking(len(tracked_faces))

        # Clean up old tracks from recognition cache
        current_track_ids = [track["track_id"] for track in tracked_faces]
        attendance_logger.cleanup_old_tracks(current_track_ids)
        # concurrently running per-track processing using ThreadPoolExecutor
        futures = []
        for track in tracked_faces:
            future = executor.submit(
                process_single_face,
                track, frame.copy(), quality_selector, attendance_logger, aligner, recognizer, adaptive_manager, session_controller.state, profiler=profiler
            )
            futures.append(future)
        # Collect results and draw on frame
        for future in futures:
            try:
                data = future.result() # wait for thread to finish
                if data:
                    x1, y1, x2, y2 = data["bbox"]
                    label = data["label"]
                    color = data["color"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                print(f"[ERROR] Error processing tracked face: {e}")

        profiler.end_frame_processing()        
        # Display the resulting frame in to the Flask app
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (320, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"SYSTEM PHASE: {session_controller.state}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"TRACKING: {len(tracked_faces)} Faces", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        active_cache = attendance_logger.get_cache_stats()["active_cached_tracks"]
        cv2.putText(frame, f"CACHE ACTIVE: {active_cache}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        with lock:
            web_app.output_frame = frame.copy()
        if not IS_DOCKER:
            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                web_app.stop_signal = True
            
    profiler.end_system()
    profiler.print_session_report() if profiler.sessions_data else None
    print("\n[PROFILER] Performance metrics saved to logs/performance/")

    # Print final cache statistics
    stats = attendance_logger.get_cache_stats()

    web_app.final_results = {
        "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "cache_hits": stats["cache_hits"],
        "cache_misses": stats["cache_misses"],
        "total_lookups": stats["total_lookups"],
        "hit_rate_percent": stats["hit_rate_percent"],
        "active_cached_tracks": stats["active_cached_tracks"],
        "status": "stopped_by_user"
    }

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

    print("\n" + "=" * 60)
    print("AI LOOP SHUTDOWN COMPLETE")
    print("Flask server is still running - you can view results!")
    print("Visit http://127.0.0.1:5000/results to see the results")
    print("=" * 60)

def save_recognized_face(person_id, crop):
    os.makedirs("recognized_faces", exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"recognized_faces/person{person_id}_{timestamp_str}.jpg"
    cv2.imwrite(filename, crop)
    print(f"[INFO] Recognized face saved: {filename}")
    return filename

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
        os._exit(0)